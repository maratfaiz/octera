"""Trains the OCTera diagnosis classifier.

Without --data-dir, trains on procedurally generated synthetic images as a
smoke test of the pipeline only -- the resulting checkpoint is NOT trained on
real patients. Point --data-dir at a local checkout of the "Dataset of Eye
Fundus and OCT Images for the study of Diabetic Macular Edema and Diabetic
Retinopathy" (Hughes Cano, Olivares Pinto & Thebault; CONACYT/UNAM/IMO/APEC/
INDEREB) to train on real, ophthalmologist-labeled OCT scans:

    git clone https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET.git
    python -m app.ml.train --data-dir /path/to/OCT-AND-EYE-FUNDUS-DATASET

That repository's root must contain OCT.csv and an OCT/ directory (OCT1..OCTn
subfolders of .jpg files) -- the layout is used as-is, no restructuring
needed. Labels come from OCT.csv's DME column (1 -> DME, 0 -> NORMAL); this
dataset does not label CNV/DRUSEN, which is why the classifier only covers
NORMAL/DME.

Picks the best of a few candidate scikit-learn classifiers via cross-
validation (balanced accuracy, since DME is a minority class in this
dataset), then reports held-out test metrics and writes a model-card JSON
(metrics.json) alongside the checkpoint.

Usage:
    python -m app.ml.train --data-dir /path/to/OCT-AND-EYE-FUNDUS-DATASET
    python -m app.ml.train  # synthetic smoke test
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from app.ml.augmentation import augment as augment_image
from app.ml.augmentation import random_brightness_contrast
from app.ml.augmentation import random_perspective
from app.ml.augmentation import random_shift
from app.ml.augmentation import rotate as rotate_image
from app.ml.features import extract_features
from app.ml.model import CLASSES, DEFAULT_CHECKPOINT_PATH, OCTClassifier
from app.ml.synthetic_dataset import generate_dataset


def _candidate_estimators(random_state: int = 42, probability: bool = True) -> dict[str, object]:
    # Each estimator is wrapped in a StandardScaler so the handful of domain
    # features (band thickness, dark-zone fraction, ...) get comparable weight
    # to the 2304 raw-pixel features instead of being drowned out -- and the
    # saved checkpoint carries the scaler, so inference stays consistent.
    #
    # `probability` defaults to True (needed once a candidate is actually the
    # winner -- see select_best_model) but select_best_model's own CV-scoring
    # pass builds these with probability=False: SVC's Platt-scaling probability
    # calibration runs an internal 5-fold CV inside every .fit() call, which is
    # pure waste during selection, since cross_val_score(..., scoring=
    # "balanced_accuracy") only ever calls .predict() -- and predict() is based
    # on the decision function, not the probability calibration, so it's
    # identical either way (sklearn's own docs note predict_proba can disagree
    # with predict for exactly this reason: they're unrelated code paths).
    return {
        "mlp_128_64": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=600, random_state=random_state
            ),
        ),
        "mlp_64": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,), activation="relu", alpha=1e-4, max_iter=600, random_state=random_state
            ),
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=probability, class_weight="balanced", random_state=random_state),
        ),
        # Round 11: the 1578-dim HOG+domain vector against ~700-900 patient-grouped
        # training images is a high-dim/low-N regime prone to overfitting, especially
        # for SVM. PCA(30) ahead of svm_rbf measured cv balanced accuracy 0.879 vs
        # 0.816-0.831 for every un-reduced or MLP+PCA combination tried -- see README.
        "svm_rbf_pca30": make_pipeline(
            StandardScaler(),
            PCA(n_components=30, random_state=random_state),
            SVC(kernel="rbf", probability=probability, class_weight="balanced", random_state=random_state),
        ),
        # Round 27: round 23's rotation augmentation grew the actual training set
        # ~5x past round 11's original ~700-900 images, so 30 components (tuned on
        # the smaller pre-augmentation set) is no longer necessarily optimal --
        # confirmed on the real dataset, PCA(75) beats PCA(30) on the augmented
        # data in 4/5 split seeds. Added as a candidate rather than replacing
        # PCA(30) outright, so cross-validation keeps picking whichever actually
        # wins for a given training run instead of hardcoding the choice.
        "svm_rbf_pca75": make_pipeline(
            StandardScaler(),
            PCA(n_components=75, random_state=random_state),
            SVC(kernel="rbf", probability=probability, class_weight="balanced", random_state=random_state),
        ),
    }


def _merge_duplicate_groups(patient_ids: list[str], content_hashes: list[str]) -> list[str]:
    """Merges patient-ID groups that share a byte-identical image under the hood.

    Round 14 found 12 pairs of exact-duplicate image files in the dataset filed
    under two *different* nominal patient IDs (adjacent/nearby IDs, e.g. 1330 and
    1348 -- likely a duplicate-entry artifact in the source dataset, not a labeling
    error: both members of every pair agree on DME). A patient-ID-only group-aware
    split (round 10) doesn't catch this: 2 of those 12 pairs still ended up split
    across train and test, since they nominally belong to different "patients" --
    letting the model see the literal same scan at train time and trivially get it
    right at test time. Connected components over a (patient_id, content_hash)
    co-occurrence graph merges any patient IDs that ever share a duplicate image
    into one group -- the same union-find problem scipy already solves.
    """
    unique_ids = sorted(set(patient_ids))
    index = {pid: i for i, pid in enumerate(unique_ids)}

    by_hash: dict[str, list[str]] = {}
    for pid, content_hash in zip(patient_ids, content_hashes):
        by_hash.setdefault(content_hash, []).append(pid)

    rows: list[int] = []
    cols: list[int] = []
    for pids in by_hash.values():
        first = index[pids[0]]
        for other in pids[1:]:
            rows.append(first)
            cols.append(index[other])

    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(unique_ids), len(unique_ids)))
    _, component_labels = connected_components(csgraph=graph, directed=False)
    return [str(component_labels[index[pid]]) for pid in patient_ids]


def _load_real_dataset(data_dir: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Loads the OCT-AND-EYE-FUNDUS-DATASET layout: <data_dir>/OCT.csv +
    <data_dir>/OCT/OCT*/<Name>.jpg, labeled by the CSV's DME column.

    Also returns a group ID per image for a patient-grouped train/test split (see
    README round 10): the leading numeric patient ID in filenames like
    "1222_OD_o_2" (eye + visit index follow the underscore), since ~25% of images
    share a patient with at least one other image in the dataset (both eyes
    and/or repeat visits) -- with same-content duplicates across different nominal
    patient IDs merged into one group (round 14, see _merge_duplicate_groups).
    """
    root = Path(data_dir)
    csv_path = root / "OCT.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"Expected {csv_path} -- pass --data-dir at a checkout of "
            "github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET"
        )

    image_index = {p.stem: p for p in root.glob("OCT/OCT*/*.jpg")}

    images: list[np.ndarray] = []
    labels: list[str] = []
    patient_ids: list[str] = []
    content_hashes: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = image_index.get(row["Name"])
            if path is None:
                continue
            raw_bytes = path.read_bytes()
            images.append(np.array(Image.open(path).convert("L")))
            labels.append("DME" if row["DME"].strip() == "1" else "NORMAL")
            patient_ids.append(row["Name"].split("_")[0])
            content_hashes.append(hashlib.md5(raw_bytes).hexdigest())
    groups = _merge_duplicate_groups(patient_ids, content_hashes)
    return images, labels, groups


def _group_aware_split(
    labels: list[str], groups: list[str], n_splits: int = 5, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split (~1/n_splits as test) that also keeps every
    group (patient) entirely on one side -- unlike a plain stratified split,
    which only balances DME/NORMAL and can (does, on this dataset: 79 of 831
    patients) split one patient's images across train and test.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy_X = np.zeros(len(labels))
    train_idx, test_idx = next(sgkf.split(dummy_X, labels, groups))
    return train_idx, test_idx


def _oversample_minority(
    images: list[np.ndarray],
    labels: list[str],
    rng: np.random.Generator,
    groups: list[str] | None = None,
    max_factor: int = 4,
) -> tuple[list[np.ndarray], list[str], list[str] | None]:
    """Tops up under-represented classes with augmented copies of their own images.

    MLPClassifier (the model round 3 selected) has no `class_weight` knob, so this
    was tried as a way to address DME's ~15% share of the dataset from the data
    side. Empirically (round 4) it does not raise DME recall -- the same held-out
    cases are missed at every factor tried -- and lowers precision, so it ships
    disabled by default (see the --minority-oversample flag and the README).
    Capped at `max_factor`x the original count per class so it stays a top-up,
    not a full rebalance to majority size.

    If `groups` is given, each augmented copy is assigned its source image's group
    (patient) rather than being left ungrouped -- an earlier version of this flag
    dropped grouping entirely for the whole training split when combined with
    oversampling, letting near-duplicate augmented copies of the same source image
    land in different folds of the internal (group-aware) CV, reintroducing exactly
    the same-image leakage patient-grouping was built to eliminate.
    """
    counts = Counter(labels)
    majority_count = max(counts.values())
    indices_by_class: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        indices_by_class.setdefault(lbl, []).append(i)

    out_images, out_labels = list(images), list(labels)
    out_groups = list(groups) if groups is not None else None
    for cls, count in counts.items():
        target = min(majority_count, count * max_factor)
        cls_indices = indices_by_class[cls]
        for i in range(target - count):
            src_idx = cls_indices[i % len(cls_indices)]
            out_images.append(augment_image(images[src_idx], rng))
            out_labels.append(cls)
            if out_groups is not None:
                out_groups.append(groups[src_idx])
    return out_images, out_labels, out_groups


def _rotation_augment(
    images: list[np.ndarray],
    labels: list[str],
    groups: list[str] | None = None,
    angles: tuple[int, ...] = (-10, -5, 5, 10),
) -> tuple[list[np.ndarray], list[str], list[str] | None]:
    """Adds copies of every image rotated by each angle in `angles`.

    Round 19 found the shipped classifier's accuracy collapses under even mild
    rotation -- e.g. a phone photo of a screen/printout rather than a direct
    digital export, which is how a real patient actually sent one: test
    accuracy dropped from 0.876 (no rotation) to 0.831 at +5 degrees and 0.453
    at +15 degrees. Rounds 20-22 tried to detect/correct rotation at inference
    time and found no signal reliable enough to ship (see README). Round 23
    instead bakes robustness into training itself. Each augmented copy keeps
    its source image's group so patient-grouped CV isn't broken by treating
    rotated copies as unrelated to their source patient -- the same fix
    applied to _oversample_minority above, and for the same reason.
    """
    out_images, out_labels = list(images), list(labels)
    out_groups = list(groups) if groups is not None else None
    for idx, (img, label) in enumerate(zip(images, labels)):
        for angle in angles:
            out_images.append(rotate_image(img, angle))
            out_labels.append(label)
            if out_groups is not None:
                out_groups.append(groups[idx])
    return out_images, out_labels, out_groups


def _flip_augment(
    images: list[np.ndarray],
    labels: list[str],
    groups: list[str] | None = None,
) -> tuple[list[np.ndarray], list[str], list[str] | None]:
    """Adds a horizontally-mirrored copy of every image.

    Round 26 first tried this (mirroring doesn't touch the depth axis the
    layer analysis relies on, so it's low-risk for OCT B-scans) against a
    rotation-augmented baseline that predates round 27/28's fix -- model
    selection now happens on whichever data is actually trained on, and
    svm_rbf_pca75 is a real candidate rather than a hardcoded pca30. Round 26
    measured only a modest, mixed effect and shipped nothing, flagging that
    the question needed a proper multi-seed check against the *current*
    recipe before deciding. Round 29 re-ran exactly that check (5 split
    seeds, flip applied on top of rotation augmentation, selection on the
    doubled data): DME recall improved in 5/5 seeds (mean 0.787->0.856),
    accuracy improved in 4/5 (flat in the 5th), precision improved in 4/5
    (one seed dipped 2.1 points) -- a strictly better result than round 23's
    own rotation-augmentation trade-off, so this ships enabled by default
    despite roughly doubling training set size (and time) again. Applied
    after rotation augmentation (doubling that already-augmented set,
    matching what round 26/29 measured), each mirrored copy keeps its
    source's group for the same patient-leakage reason as
    _oversample_minority/_rotation_augment above.
    """
    out_images, out_labels = list(images), list(labels)
    out_groups = list(groups) if groups is not None else None
    for idx, (img, label) in enumerate(zip(images, labels)):
        out_images.append(np.fliplr(img))
        out_labels.append(label)
        if out_groups is not None:
            out_groups.append(groups[idx])
    return out_images, out_labels, out_groups


def _brightness_contrast_augment(
    images: list[np.ndarray],
    labels: list[str],
    groups: list[str] | None = None,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[str], list[str] | None]:
    """Adds one randomized brightness/contrast-perturbed copy of every image
    (reusing app.ml.augmentation.random_brightness_contrast, already used for
    synthetic-data augmentation and --minority-oversample, but never before
    tried in the real-data training path).

    Real uploads plausibly vary in exposure/glare as much as they vary in
    rotation -- the same "phone photo of a screen/printout" scenario rounds
    19/23 built rotation robustness for. Round 30 tested this on top of the
    current recipe (rotation + flip, round 29) across 5 split seeds: accuracy
    improved 0.951->0.958 mean (4/5 seeds), DME precision improved
    0.803->0.853 mean (4/5 seeds, the largest gain of any single-round
    augmentation change tried so far), and DME recall stayed flat within
    noise (0.856->0.852 mean; two seeds dipped ~2.5-2.6 points, one gained 2.9,
    two were unchanged) -- a cleaner result than round 23's own accepted
    rotation-augmentation trade-off (which cost ~4 points of recall on
    average for its robustness gain). Ships enabled by default.

    A fixed RNG seed (independent of the split seed) makes the shipped
    checkpoint's exact augmented pixels reproducible run to run, matching
    _oversample_minority's use of a fixed seed for the same reason. Each
    perturbed copy keeps its source's group for the same patient-leakage
    reason as every other augmentation stage above.
    """
    rng = np.random.default_rng(seed)
    out_images, out_labels = list(images), list(labels)
    out_groups = list(groups) if groups is not None else None
    for idx, (img, label) in enumerate(zip(images, labels)):
        out_images.append(random_brightness_contrast(img, rng))
        out_labels.append(label)
        if out_groups is not None:
            out_groups.append(groups[idx])
    return out_images, out_labels, out_groups


def _apply_augmentation_stage(
    images: list[np.ndarray],
    labels: list[str],
    groups: list[str] | np.ndarray | None,
    X_existing: np.ndarray,
    augment_fn,
    stage_name: str,
) -> tuple[list[np.ndarray], list[str], np.ndarray | None, np.ndarray, np.ndarray]:
    """Applies one augmentation stage (`_rotation_augment` or `_flip_augment`) and
    extends an already-extracted feature matrix to match, extracting features only
    for the newly-appended images rather than the whole augmented set.

    Every augment_fn here keeps its input unchanged as a prefix of its output
    (see their docstrings), so `X_existing` already covers `images[:n_before]` --
    this used to be four hand-copied instances of the same "slice the new tail,
    extract, concatenate" sequence (rotation and flip, each in both the
    training-split and round-16 full-refit call sites); round 30 factored it out
    before adding a third augmentation stage would have made it five and six.
    """
    n_before = len(images)
    groups_list = list(groups) if groups is not None else None
    images, labels, groups_list = augment_fn(images, labels, groups_list)
    groups_out = np.array(groups_list) if groups_list is not None else None
    print(f"{stage_name}: {n_before} -> {len(images)} images")
    X_new = np.stack([extract_features(Image.fromarray(img)) for img in images[n_before:]])
    X = np.concatenate([X_existing, X_new])
    y = np.array(labels)
    return images, labels, groups_out, X, y


def _extract_sibling_augmentation_features(
    images: list[np.ndarray], single_image_augment_fn, seed: int, stage_name: str
) -> np.ndarray:
    """Extracts one new "sibling" augmentation group's features from `images`
    -- the training split's raw-pixel list AFTER rotation/flip/brightness
    (which chain onto each other) but never itself doubled by this function.

    `_apply_augmentation_stage` (rotation/flip/brightness) returns the
    doubled raw-pixel list so the next stage can build on top of it. But
    once a stage's job is to add a "sibling" group -- a parallel variant of
    the SAME base images, not a further transform of the previous stage's
    output -- doubling that list stops being free: applied as a 4th stage
    (round 40, shift), it pushed the select phase's ~17560-image training
    split to ~35000 raw grayscale images at native resolution -- ~27GB of
    pixel data alone, which does not fit this project's 15GB training
    machine regardless of what happens afterward (confirmed via a real OOM
    kill, dmesg: anon-rss at the machine's ceiling, mid-augmentation). Same
    failure mode round 38 hit and fixed in its own throwaway experiment
    script -- holding two full raw-image lists at once -- now hit by the
    real pipeline once a 4th stage was added.

    Instead, this extracts each sibling copy's features one image at a time
    straight into a feature vector, discarding the augmented pixel array
    immediately, and never appends to `images` at all -- peak memory stays
    at roughly the *single* base raw-image list the caller already holds,
    plus a transient one-image buffer, not a second full list. Because
    `images` (the base) is untouched, the caller can call this again for a
    second, third, etc. sibling group (round 44 adds `random_perspective`
    alongside round 40's `random_shift`) against the exact same base -- see
    `_run_select_phase` for the accumulation pattern -- rather than being
    limited to one terminal stage the way a doubling implementation would be.
    """
    rng = np.random.default_rng(seed)
    new_features = [extract_features(Image.fromarray(single_image_augment_fn(img, rng))) for img in images]
    print(f"{stage_name}: {len(images)} -> +{len(images)} sibling images (features extracted one at a time)")
    return np.stack(new_features)


def _select_thresholds(y_true: np.ndarray, dme_proba: np.ndarray, min_recall: float = 0.85) -> dict:
    """Picks candidate DME probability cutoffs from a precision-recall curve.

    Caller must feed this data the model's threshold choice will not later be
    graded against -- see _cv_threshold_analysis, which is why this function
    itself takes bare arrays rather than reaching into the test set.
    """
    y_binary = (y_true == "DME").astype(int)
    precisions, recalls, thresholds = precision_recall_curve(y_binary, dme_proba)
    f1s = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        precisions[:-1] + recalls[:-1],
        out=np.zeros_like(thresholds),
        where=(precisions[:-1] + recalls[:-1]) > 0,
    )
    best_f1_threshold = float(thresholds[int(np.argmax(f1s))])

    high_recall_candidates = [i for i in range(len(thresholds)) if recalls[i] >= min_recall]
    high_recall_threshold = (
        float(thresholds[max(high_recall_candidates, key=lambda i: precisions[i])]) if high_recall_candidates else None
    )
    return {"best_f1_threshold": best_f1_threshold, "high_recall_threshold": high_recall_threshold}


def _metrics_at_threshold(y_true: np.ndarray, dme_proba: np.ndarray, threshold: float) -> dict:
    preds = np.where(dme_proba >= threshold, "DME", "NORMAL")
    return {
        "threshold": float(threshold),
        # labels=CLASSES (matching confusion_matrix below) guarantees a "DME" key is
        # always present, even if a small patient-grouped fold happens to contain no
        # true or predicted DME -- main() indexes ['DME'] unconditionally below, and
        # without this, sklearn simply omits the key and that indexing raises KeyError.
        "classification_report": classification_report(
            y_true, preds, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=CLASSES).tolist(),
    }


def _cv_threshold_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_dme_proba: np.ndarray,
    estimator: object,
    groups_train: np.ndarray | None = None,
) -> dict:
    """Picks DME decision-threshold operating points using only the training
    split (via 4-fold out-of-fold predictions), then reports how those exact
    thresholds perform on the untouched test set.

    Round 5 originally chose thresholds by running precision_recall_curve
    directly against the test set -- the same set used to report the final
    numbers. That is test-set leakage: it picks the threshold that looks best
    on the specific 223 test images, not one expected to generalize. Round 6
    fixes this by selecting thresholds out-of-fold on the 890 training images
    only; the test set is then used exactly once, to grade (not pick) those
    thresholds. Round 10 adds patient grouping to that same CV (groups_train),
    for the same reason the main train/test split needs it -- see
    _group_aware_split.
    """
    cv = (
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        if groups_train is not None
        else StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    )
    oof_proba = cross_val_predict(estimator, X_train, y_train, groups=groups_train, cv=cv, method="predict_proba")
    train_classes = sorted(set(y_train))
    oof_dme_proba = oof_proba[:, train_classes.index("DME")]

    thresholds = _select_thresholds(y_train, oof_dme_proba)
    result = {
        "selection_method": "thresholds chosen via 4-fold out-of-fold CV on the training split only; "
        "metrics below are the test set graded at those thresholds, not used to pick them",
        "default_threshold": 0.5,
        "best_f1_operating_point": _metrics_at_threshold(y_test, test_dme_proba, thresholds["best_f1_threshold"]),
    }
    result["high_recall_operating_point"] = (
        _metrics_at_threshold(y_test, test_dme_proba, thresholds["high_recall_threshold"])
        if thresholds["high_recall_threshold"] is not None
        else None
    )
    return result


def _multi_seed_worker_evaluate(
    data_dir: str,
    seed: int,
    estimator_name: str,
    rotation_augment: bool,
    flip_augment: bool,
    brightness_augment: bool,
    shift_augment: bool,
    perspective_augment: bool,
) -> dict:
    """Evaluates ONE split seed: load, split, augment, extract, fit, grade
    against the held-out fold. Called from a fresh subprocess per seed by
    `_multi_seed_evaluation` below -- see that function's docstring for why.
    """
    raw_images, labels, split_groups = _load_real_dataset(data_dir)
    if not raw_images:
        raise SystemExit(f"No images matched between OCT.csv and OCT/OCT*/*.jpg under {data_dir}")
    train_idx, test_idx = _group_aware_split(labels, split_groups, random_state=seed)
    train_images = [raw_images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    if rotation_augment:
        train_images, train_labels, _ = _rotation_augment(train_images, train_labels)
    if flip_augment:
        train_images, train_labels, _ = _flip_augment(train_images, train_labels)
    if brightness_augment:
        train_images, train_labels, _ = _brightness_contrast_augment(train_images, train_labels)

    X_train_seed = np.stack([extract_features(Image.fromarray(img)) for img in train_images])
    y_train_seed = np.array(train_labels)
    base_labels_seed = np.array(train_labels)
    # Sibling stages (round 40's shift, round 44's perspective): each derives its
    # own group straight from the same post-rotation+flip+brightness base images,
    # never doubling train_images itself -- see
    # _extract_sibling_augmentation_features's docstring for the OOM this avoids.
    if shift_augment:
        X_shift_seed = _extract_sibling_augmentation_features(
            train_images, random_shift, seed=55, stage_name="Shift-augmented training split (round 40)"
        )
        X_train_seed = np.concatenate([X_train_seed, X_shift_seed])
        y_train_seed = np.concatenate([y_train_seed, base_labels_seed])
    if perspective_augment:
        X_persp_seed = _extract_sibling_augmentation_features(
            train_images, random_perspective, seed=99, stage_name="Perspective-augmented training split (round 44)"
        )
        X_train_seed = np.concatenate([X_train_seed, X_persp_seed])
        y_train_seed = np.concatenate([y_train_seed, base_labels_seed])
    del train_images

    X_test_seed = np.stack([extract_features(Image.fromarray(raw_images[i])) for i in test_idx])
    y_all = np.array(labels)

    model = _candidate_estimators()[estimator_name]
    model.fit(X_train_seed, y_train_seed)
    preds = model.predict(X_test_seed)
    rep = classification_report(y_all[test_idx], preds, output_dict=True, zero_division=0)
    result: dict = {"seed": seed, "accuracy": rep["accuracy"]}
    # A grouped split can, on this small dataset, draw a test fold with zero true
    # DME cases -- "DME" key is then absent from rep entirely. That's different
    # from the model genuinely scoring 0 precision/recall on a fold that DID have
    # DME cases; silently defaulting both to 0.0 would understate the real mean/std
    # by mixing in seeds that had nothing to score. The caller excludes these seeds.
    if "DME" in rep:
        result["dme_precision"] = rep["DME"]["precision"]
        result["dme_recall"] = rep["DME"]["recall"]
    return result


def _multi_seed_evaluation(
    data_dir: str,
    estimator_name: str,
    seeds: tuple[int, ...] = (42, 7, 99, 123, 2024),
    rotation_augment: bool = False,
    flip_augment: bool = False,
    brightness_augment: bool = False,
    shift_augment: bool = False,
    perspective_augment: bool = False,
) -> dict:
    """Refits the winning model type across several train/test split seeds and
    reports mean/std, instead of trusting the single split's numbers at face
    value. Round 14 found DME precision on this small, patient-grouped dataset
    swings from 0.47 to 0.72 depending on which split random_state is used --
    a single split's numbers alone can mislead. The shipped checkpoint still
    comes from one fixed split (random_state=42) for reproducibility; this is
    purely a characterization exercise, saved to the model card for context.

    When `rotation_augment` is set (round 23), each seed's training split is
    rotation-augmented before fitting, matching whatever recipe actually
    produces the shipped checkpoint -- round 23 originally left this
    characterization describing the pre-round-23 recipe even after the
    augmented recipe shipped, which would have made this field describe a
    model no longer in production. `flip_augment` (round 29),
    `brightness_augment` (round 30), and `shift_augment`/`perspective_augment`
    (rounds 40/44) do the same for their respective stages, applied in the
    same order as the real training pipeline (rotation, then flip, then
    brightness/contrast, then the shift and perspective sibling groups).

    Round 30 found this function's own 5-seed loop -- stacked on top of an
    already-large select_best_model phase within the same process -- was the
    proven cause of two separate OOM kills once the augmented training set
    tripled in size (rotation+flip+brightness, ~17500 images): numpy/BLAS/
    libsvm don't reliably return freed memory to the OS between repeated
    large fits in one long-running process, so RSS climbs seed over seed
    until the kernel kills it. Each seed now runs `_multi_seed_worker_evaluate`
    in its own fresh subprocess (via `python -m app.ml.train
    --multi-seed-worker-seed N`) instead of in-process, so the OS fully
    reclaims memory between seeds -- the same fix already validated for this
    round's throwaway experiment script. Costs re-loading the dataset from
    disk per seed (a few seconds, not the bottleneck) and 5x the feature
    extraction of the non-augmented path, same as before.
    """

    def _summary(values: list[float]) -> dict:
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": [float(v) for v in values]}

    accuracies: list[float] = []
    dme_precisions: list[float] = []
    dme_recalls: list[float] = []
    skipped_seeds: list[int] = []
    for seed in seeds:
        cmd = [
            sys.executable,
            "-m",
            "app.ml.train",
            "--data-dir",
            data_dir,
            "--multi-seed-worker-seed",
            str(seed),
            "--multi-seed-worker-estimator",
            estimator_name,
        ]
        if not rotation_augment:
            cmd.append("--no-rotation-augment")
        if not flip_augment:
            cmd.append("--no-flip-augment")
        if not brightness_augment:
            cmd.append("--no-brightness-augment")
        if not shift_augment:
            cmd.append("--no-shift-augment")
        if not perspective_augment:
            cmd.append("--no-perspective-augment")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # The worker prints exactly one JSON line (as its last stdout line);
        # sklearn/joblib warnings some environments emit go to stderr, not stdout.
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        accuracies.append(result["accuracy"])
        if "dme_precision" in result:
            dme_precisions.append(result["dme_precision"])
            dme_recalls.append(result["dme_recall"])
        else:
            skipped_seeds.append(seed)

    return {
        "seeds": list(seeds),
        "rotation_augment": rotation_augment,
        "flip_augment": flip_augment,
        "brightness_augment": brightness_augment,
        "shift_augment": shift_augment,
        "perspective_augment": perspective_augment,
        "test_accuracy": _summary(accuracies),
        "dme_precision": _summary(dme_precisions),
        "dme_recall": _summary(dme_recalls),
        "seeds_skipped_no_dme_in_test_fold": skipped_seeds,
    }


def select_best_model(
    X: np.ndarray, y: np.ndarray, cv_folds: int = 4, groups: np.ndarray | None = None
) -> tuple[str, object, dict[str, float]]:
    cv = (
        StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        if groups is not None
        else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    )
    # probability=False here (round 44 perf fix): cross_val_score's balanced_accuracy
    # scoring only ever calls .predict(), never .predict_proba(), so SVC's Platt-
    # scaling calibration (an internal 5-fold CV inside every .fit()) is pure
    # overhead during selection -- see _candidate_estimators' docstring. The winner
    # is re-fetched with probability=True below, since the caller needs real
    # probability estimates once this candidate is actually the one getting fit.
    candidates = _candidate_estimators(probability=False)
    scores: dict[str, float] = {}
    for name, estimator in candidates.items():
        try:
            fold_scores = cross_val_score(estimator, X, y, groups=groups, cv=cv, scoring="balanced_accuracy")
        except ValueError as exc:
            # e.g. svm_rbf_pca30's fixed PCA(n_components=30) needs at least 30 samples
            # in every fold -- fine on the real dataset (folds in the hundreds) but not
            # on a small synthetic/smoke-test run. Skip the candidate rather than crash
            # the whole training run over one estimator that doesn't fit this dataset.
            print(f"  {name}: skipped ({exc})")
            continue
        scores[name] = float(fold_scores.mean())
        print(f"  {name}: cv balanced accuracy = {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")

    if not scores:
        raise RuntimeError("No candidate estimator could be cross-validated on this dataset")
    best_name = max(scores, key=scores.get)
    # clone() gives a fresh unfitted copy -- but of the real (probability=True)
    # estimator, not the probability=False one used for scoring above, since the
    # caller fits this one for real and needs predict_proba downstream (DME
    # threshold selection, calibration metrics).
    return best_name, clone(_candidate_estimators()[best_name]), scores


def _run_synthetic_training(args: argparse.Namespace) -> None:
    """Original single-process training flow, kept simple and unchanged for the
    synthetic smoke-test path: the dataset is tiny (--synthetic-per-class, a
    few hundred images by default), rotation/flip/brightness augmentation are
    always gated off for it (`data_source == "synthetic"`), and multi-seed
    evaluation and the round-16 full-dataset refit never run for it either --
    none of round 30's OOM/subprocess-isolation concerns apply here, so this
    stays a plain in-process run rather than the phase-subprocess machinery
    `_run_real_training` below needs for the real dataset's much larger scale.
    """
    print(
        "No --data-dir given: training on synthetic procedural data. "
        "This checkpoint is a pipeline smoke test only and is NOT trained on real patients."
    )
    raw_images, labels = generate_dataset(n_per_class=args.synthetic_per_class, augment=not args.no_augment)
    data_source = "synthetic"
    split_groups = [str(i) for i in range(len(labels))]
    train_idx, test_idx = _group_aware_split(labels, split_groups)
    images_train = [raw_images[i] for i in train_idx]
    images_test = [raw_images[i] for i in test_idx]
    labels_train = [labels[i] for i in train_idx]
    labels_test = [labels[i] for i in test_idx]

    if args.minority_oversample > 1:
        rng = np.random.default_rng(42)
        before = Counter(labels_train)
        images_train, labels_train, _ = _oversample_minority(
            images_train, labels_train, rng, max_factor=args.minority_oversample
        )
        print(f"Oversampled minority classes in training split: {dict(before)} -> {dict(Counter(labels_train))}")
    n_train_after_oversampling = len(images_train)

    X_train = np.stack([extract_features(Image.fromarray(img)) for img in images_train])
    X_test = np.stack([extract_features(Image.fromarray(img)) for img in images_test])
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    print("Selecting best model via cross-validation on the training split:")
    best_name, best_estimator, cv_scores = select_best_model(X_train, y_train)
    print(f"Selected: {best_name} (cv balanced accuracy = {cv_scores[best_name]:.3f})")

    model = OCTClassifier(best_estimator)
    model.fit(X_train, y_train)

    preds = model.clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    print(f"Held-out test accuracy: {test_accuracy:.3f}")
    print(classification_report(y_test, preds))

    threshold_analysis = None
    calibration_metrics = None
    if "DME" in model.clf.classes_:
        dme_class_idx = list(model.clf.classes_).index("DME")
        test_dme_proba = model.clf.predict_proba(X_test)[:, dme_class_idx]
        y_test_binary = (y_test == "DME").astype(int)
        calibration_metrics = {
            "brier_score": float(brier_score_loss(y_test_binary, test_dme_proba)),
            "log_loss": float(log_loss(y_test_binary, test_dme_proba)),
            "note": (
                "round 8 tried CalibratedClassifierCV (sigmoid/isotonic) against these numbers: isotonic improves "
                "both (Brier 0.046 vs 0.053, log loss 0.14 vs 0.20 here) but makes the shipped high-recall "
                "threshold operating point worse (precision 0.58 vs 0.68 at recall 0.76 vs 0.79); sigmoid is a "
                "roughly neutral wash on both. Better average-case calibration doesn't guarantee a better specific "
                "operating point -- not shipped, see README round 8."
            ),
        }
        print(f"Calibration: Brier={calibration_metrics['brier_score']:.4f} LogLoss={calibration_metrics['log_loss']:.3f}")
        print("Selecting DME decision thresholds via out-of-fold CV on the training split (not the test set):")
        threshold_analysis = _cv_threshold_analysis(X_train, y_train, y_test, test_dme_proba, best_estimator)
        best = threshold_analysis["best_f1_operating_point"]
        print(
            f"  best-F1 cutoff = {best['threshold']:.3f} "
            f"(test precision {best['classification_report']['DME']['precision']:.2f} / "
            f"recall {best['classification_report']['DME']['recall']:.2f})"
        )
        high_recall = threshold_analysis["high_recall_operating_point"]
        if high_recall:
            print(
                f"  recall>=0.85 (on train CV) cutoff = {high_recall['threshold']:.3f} "
                f"(test precision {high_recall['classification_report']['DME']['precision']:.2f} / "
                f"recall {high_recall['classification_report']['DME']['recall']:.2f})"
            )
    else:
        print(
            f"WARNING: training split has no DME examples (classes seen: {list(model.clf.classes_)}) -- "
            "skipping DME threshold/calibration analysis. This usually indicates a data-loading or "
            "labeling problem rather than an expected outcome; the shipped checkpoint's metrics.json "
            "will have null dme_threshold_analysis/dme_calibration_metrics fields as a result."
        )

    model.save(args.out)
    print(f"Saved checkpoint to {args.out}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "n_samples": len(raw_images),
        "n_train_samples_after_oversampling": n_train_after_oversampling,
        "minority_oversample_factor": args.minority_oversample,
        "rotation_augment_enabled": False,
        "n_train_samples_after_rotation_augment": n_train_after_oversampling,
        "flip_augment_enabled": False,
        "n_train_samples_after_flip_augment": n_train_after_oversampling,
        "brightness_augment_enabled": False,
        "n_train_samples_after_brightness_augment": n_train_after_oversampling,
        "shift_augment_enabled": False,
        "n_train_samples_after_shift_augment": n_train_after_oversampling,
        "perspective_augment_enabled": False,
        "n_train_samples_after_perspective_augment": n_train_after_oversampling,
        "classes": CLASSES,
        "selected_model": best_name,
        "cv_balanced_accuracy_by_model": cv_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASSES).tolist(),
        "trained_on_real_patient_data": False,
        "dme_threshold_analysis": threshold_analysis,
        "dme_calibration_metrics": calibration_metrics,
        "multi_seed_evaluation": None,
        "shipped_checkpoint_trained_on_full_dataset": False,
    }
    metrics_path = Path(args.out).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model card to {metrics_path}")


def _run_select_phase(args: argparse.Namespace) -> dict:
    """Phase 1 of the subprocess-isolated real-data pipeline (round 30): load,
    oversample+augment the training split, select the best model type, fit it
    on the single split, evaluate on the untouched held-out test set, and
    compute calibration/threshold diagnostics. Invoked as its own fresh
    subprocess (`python -m app.ml.train --phase select`) by the orchestrator
    in main() -- see _multi_seed_evaluation's docstring for why: this phase's
    own peak memory (~11GB, confirmed via two OOM kills with dmesg once the
    augmented training set reached ~17500 images) must not stack with the
    multi-seed and final-refit phases that follow, and the only way to
    guarantee that is for the process holding it to actually exit rather than
    just idle while waiting on child subprocesses.

    This raises the training-set size at which OOM recurs; it does not
    eliminate the failure mode. select_best_model alone runs ~21 sequential
    fits (6 candidates x 4 CV folds, plus the final fit) inside this one
    process with no further isolation -- the same numpy/BLAS/libsvm memory-
    retention behavior blamed for the multi-seed OOM applies here too, just
    with more budget before it bites.

    Round 40 hit exactly the predicted failure: adding a 4th augmentation
    stage (shift) via the same doubling pattern as rotation/flip/brightness
    pushed the ~17560-image training split to ~35000 raw grayscale images at
    native resolution -- confirmed via dmesg (Killed process ..., anon-rss at
    the machine's 15GB ceiling) mid-augmentation, before select_best_model
    even started. The fix wasn't isolating select_best_model's candidate
    loop (that would only have helped if the augmented *feature matrix* were
    the problem; it's tiny). The real cost was the raw *pixel* list: neither
    shift nor anything added after it needs its augmented copies' raw
    pixels, only their features -- so each is applied via
    `_extract_sibling_augmentation_features`, which extracts one new
    "sibling" group's features one image at a time, straight from a fixed
    post-rotation+flip+brightness base image list, and never doubles the
    raw-image list at all (the same fix round 38 used for its own throwaway
    experiment script, now applied to the real pipeline).

    Round 44 generalized this from a single terminal stage to an arbitrary
    number of parallel sibling groups: shift (round 40) and perspective
    (round 44) are each derived independently from the same fixed base
    images/labels/groups captured before either is added, and concatenated
    onto X_train/y_train/labels_train/groups_train by the caller -- see
    `_extract_sibling_augmentation_features`'s docstring. A future 5th
    stage that should sit alongside shift and perspective (rather than
    chain onto either of their outputs) just needs the same
    `base_images_train`-derived call added as another sibling block; only a
    stage that must genuinely chain onto a *previous* augmented copy (not
    the shared base) would need the older doubling `_apply_augmentation_stage`
    pattern instead.

    Returns a JSON-serializable dict of everything the orchestrator needs for
    the final model card. Does NOT save a checkpoint unless minority-oversample
    is active (a non-default experimental flag) -- in that case no later phase
    refits on the full dataset (mirroring the original single-process
    behavior's X_all gating), so this phase's single-split model IS the final
    artifact and must be saved here, since its fitted model object won't
    survive past this process exiting.
    """
    raw_images, labels, groups = _load_real_dataset(args.data_dir)
    if not raw_images:
        raise SystemExit(f"No images matched between OCT.csv and OCT/OCT*/*.jpg under {args.data_dir}")
    n_patients = len(set(groups))
    print(f"Loaded {len(raw_images)} real images from {args.data_dir} ({n_patients} unique patients)")

    train_idx, test_idx = _group_aware_split(labels, groups)
    images_train = [raw_images[i] for i in train_idx]
    images_test = [raw_images[i] for i in test_idx]
    labels_train = [labels[i] for i in train_idx]
    labels_test = [labels[i] for i in test_idx]
    groups_train = np.array(groups)[train_idx]

    if args.minority_oversample > 1:
        rng = np.random.default_rng(42)
        before = Counter(labels_train)
        groups_train_list = list(groups_train)
        images_train, labels_train, groups_train_list = _oversample_minority(
            images_train, labels_train, rng, groups=groups_train_list, max_factor=args.minority_oversample
        )
        groups_train = np.array(groups_train_list)
        print(f"Oversampled minority classes in training split: {dict(before)} -> {dict(Counter(labels_train))}")
    n_train_after_oversampling = len(images_train)

    X_train_raw = np.stack([extract_features(Image.fromarray(img)) for img in images_train])
    X_test = np.stack([extract_features(Image.fromarray(img)) for img in images_test])
    y_train_raw = np.array(labels_train)
    y_test = np.array(labels_test)

    X_train, y_train = X_train_raw, y_train_raw
    rotation_augmented = not args.no_rotation_augment
    if rotation_augmented:
        images_train, labels_train, groups_train, X_train, y_train = _apply_augmentation_stage(
            images_train, labels_train, groups_train, X_train, _rotation_augment, "Rotation-augmented training split (round 23)"
        )
    n_train_after_rotation_augment = len(images_train)

    flip_augmented = not args.no_flip_augment
    if flip_augmented:
        images_train, labels_train, groups_train, X_train, y_train = _apply_augmentation_stage(
            images_train, labels_train, groups_train, X_train, _flip_augment, "Flip-augmented training split (round 29)"
        )
    n_train_after_flip_augment = len(images_train)

    brightness_augmented = not args.no_brightness_augment
    if brightness_augmented:
        images_train, labels_train, groups_train, X_train, y_train = _apply_augmentation_stage(
            images_train,
            labels_train,
            groups_train,
            X_train,
            _brightness_contrast_augment,
            "Brightness/contrast-augmented training split (round 30)",
        )
    n_train_after_brightness_augment = len(images_train)

    # Sibling groups (round 40's shift, round 44's perspective): each derives its
    # own group straight from the same post-rotation+flip+brightness base images,
    # never doubling images_train itself -- see
    # _extract_sibling_augmentation_features's docstring for the OOM this avoids.
    base_images_train = images_train
    base_labels_train = list(labels_train)
    base_groups_train = groups_train

    shift_augmented = not args.no_shift_augment
    if shift_augmented:
        X_shift = _extract_sibling_augmentation_features(
            base_images_train, random_shift, seed=55, stage_name="Shift-augmented training split (round 40)"
        )
        X_train = np.concatenate([X_train, X_shift])
        y_train = np.concatenate([y_train, np.array(base_labels_train)])
        labels_train = labels_train + base_labels_train
        groups_train = np.concatenate([groups_train, base_groups_train])
    n_train_after_shift_augment = len(labels_train)

    perspective_augmented = not args.no_perspective_augment
    if perspective_augmented:
        X_persp = _extract_sibling_augmentation_features(
            base_images_train,
            random_perspective,
            seed=99,
            stage_name="Perspective-augmented training split (round 44)",
        )
        X_train = np.concatenate([X_train, X_persp])
        y_train = np.concatenate([y_train, np.array(base_labels_train)])
        labels_train = labels_train + base_labels_train
        groups_train = np.concatenate([groups_train, base_groups_train])
    n_train_after_perspective_augment = len(labels_train)

    # images_train (== base_images_train) no longer matches y_train/X_train once a
    # sibling stage above extended labels/features without extending the raw-pixel
    # list (that's the whole memory-saving point) -- freed here rather than left
    # sitting at ~13.5GB across the ~21 sequential fits select_best_model is about
    # to run.
    del images_train, base_images_train

    print("Selecting best model via cross-validation on the training split:")
    best_name, best_estimator, cv_scores = select_best_model(X_train, y_train, groups=groups_train)
    print(f"Selected: {best_name} (cv balanced accuracy = {cv_scores[best_name]:.3f})")

    model = OCTClassifier(best_estimator)
    model.fit(X_train, y_train)

    preds = model.clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    print(f"Held-out test accuracy: {test_accuracy:.3f}")
    print(classification_report(y_test, preds))

    threshold_analysis = None
    calibration_metrics = None
    if "DME" in model.clf.classes_:
        dme_class_idx = list(model.clf.classes_).index("DME")
        test_dme_proba = model.clf.predict_proba(X_test)[:, dme_class_idx]
        y_test_binary = (y_test == "DME").astype(int)
        calibration_metrics = {
            "brier_score": float(brier_score_loss(y_test_binary, test_dme_proba)),
            # log_loss accepts a 1D array of positive-class (y_true==1, i.e. DME) probabilities for binary y_true.
            "log_loss": float(log_loss(y_test_binary, test_dme_proba)),
            "note": (
                "round 8 tried CalibratedClassifierCV (sigmoid/isotonic) against these numbers: isotonic improves "
                "both (Brier 0.046 vs 0.053, log loss 0.14 vs 0.20 here) but makes the shipped high-recall "
                "threshold operating point worse (precision 0.58 vs 0.68 at recall 0.76 vs 0.79); sigmoid is a "
                "roughly neutral wash on both. Better average-case calibration doesn't guarantee a better specific "
                "operating point -- not shipped, see README round 8."
            ),
        }
        print(f"Calibration: Brier={calibration_metrics['brier_score']:.4f} LogLoss={calibration_metrics['log_loss']:.3f}")

        print("Selecting DME decision thresholds via out-of-fold CV on the training split (not the test set):")
        threshold_analysis = _cv_threshold_analysis(
            X_train, y_train, y_test, test_dme_proba, best_estimator, groups_train=groups_train
        )
        best = threshold_analysis["best_f1_operating_point"]
        print(
            f"  best-F1 cutoff = {best['threshold']:.3f} "
            f"(test precision {best['classification_report']['DME']['precision']:.2f} / "
            f"recall {best['classification_report']['DME']['recall']:.2f})"
        )
        high_recall = threshold_analysis["high_recall_operating_point"]
        if high_recall:
            print(
                f"  recall>=0.85 (on train CV) cutoff = {high_recall['threshold']:.3f} "
                f"(test precision {high_recall['classification_report']['DME']['precision']:.2f} / "
                f"recall {high_recall['classification_report']['DME']['recall']:.2f})"
            )
    else:
        print(
            f"WARNING: training split has no DME examples (classes seen: {list(model.clf.classes_)}) -- "
            "skipping DME threshold/calibration analysis. This usually indicates a data-loading or "
            "labeling problem rather than an expected outcome; the shipped checkpoint's metrics.json "
            "will have null dme_threshold_analysis/dme_calibration_metrics fields as a result."
        )

    if args.minority_oversample > 1:
        # No later phase refits on 100% of data when oversampling is active (the
        # orchestrator's run_full_pipeline gate) -- this phase's single-split
        # model IS the final artifact, so save it here rather than losing the
        # fitted object when this process exits.
        model.save(args.out)
        print(f"Saved checkpoint to {args.out}")

    return {
        "data_source": "real:OCT-AND-EYE-FUNDUS-DATASET",
        "n_samples": len(raw_images),
        "n_train_samples_after_oversampling": n_train_after_oversampling,
        "rotation_augment_enabled": rotation_augmented,
        "n_train_samples_after_rotation_augment": n_train_after_rotation_augment,
        "flip_augment_enabled": flip_augmented,
        "n_train_samples_after_flip_augment": n_train_after_flip_augment,
        "brightness_augment_enabled": brightness_augmented,
        "n_train_samples_after_brightness_augment": n_train_after_brightness_augment,
        "shift_augment_enabled": shift_augmented,
        "n_train_samples_after_shift_augment": n_train_after_shift_augment,
        "perspective_augment_enabled": perspective_augmented,
        "n_train_samples_after_perspective_augment": n_train_after_perspective_augment,
        "selected_model": best_name,
        "cv_balanced_accuracy_by_model": cv_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASSES).tolist(),
        "dme_threshold_analysis": threshold_analysis,
        "dme_calibration_metrics": calibration_metrics,
    }


def _run_final_refit_phase(args: argparse.Namespace) -> None:
    """Phase 2 of the subprocess-isolated real-data pipeline (round 30, round
    16 originally): reload the full dataset, apply the same augmentation
    recipe to 100% of it, fit the given (already-selected, by
    `--estimator-name`) model type, and save it as the shipped checkpoint.
    Invoked as its own fresh subprocess by the orchestrator in main() -- see
    _run_select_phase's docstring for why this can't just continue in the
    process that ran selection.

    Re-extracts features for the raw (unaugmented) images from scratch rather
    than reusing the select phase's already-computed ones, since those don't
    survive that phase's process exiting -- a few seconds of redundant I/O,
    traded for the memory isolation that's the whole point of this split.
    """
    raw_images, labels, groups = _load_real_dataset(args.data_dir)
    if not raw_images:
        raise SystemExit(f"No images matched between OCT.csv and OCT/OCT*/*.jpg under {args.data_dir}")
    full_images, full_labels = raw_images, labels
    X_full = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y_full = np.array(labels)

    print("Retraining final checkpoint on 100% of the data (round 16) for shipping:")
    if not args.no_rotation_augment:
        full_images, full_labels, _, X_full, y_full = _apply_augmentation_stage(
            full_images, full_labels, None, X_full, _rotation_augment, "Rotation-augmented full dataset for final refit"
        )
    if not args.no_flip_augment:
        full_images, full_labels, _, X_full, y_full = _apply_augmentation_stage(
            full_images, full_labels, None, X_full, _flip_augment, "Flip-augmented full dataset for final refit"
        )
    if not args.no_brightness_augment:
        full_images, full_labels, _, X_full, y_full = _apply_augmentation_stage(
            full_images,
            full_labels,
            None,
            X_full,
            _brightness_contrast_augment,
            "Brightness/contrast-augmented full dataset for final refit",
        )
    # Sibling groups (round 40's shift, round 44's perspective): each derives
    # its own group straight from the same post-rotation+flip+brightness base
    # images, never doubling full_images itself -- see
    # _extract_sibling_augmentation_features's docstring for the OOM this avoids.
    base_full_images = full_images
    base_full_labels = list(full_labels)

    if not args.no_shift_augment:
        X_shift_full = _extract_sibling_augmentation_features(
            base_full_images, random_shift, seed=55, stage_name="Shift-augmented full dataset for final refit"
        )
        X_full = np.concatenate([X_full, X_shift_full])
        y_full = np.concatenate([y_full, np.array(base_full_labels)])

    if not args.no_perspective_augment:
        X_persp_full = _extract_sibling_augmentation_features(
            base_full_images,
            random_perspective,
            seed=99,
            stage_name="Perspective-augmented full dataset for final refit",
        )
        X_full = np.concatenate([X_full, X_persp_full])
        y_full = np.concatenate([y_full, np.array(base_full_labels)])

    del full_images, base_full_images

    estimator = _candidate_estimators()[args.estimator_name]
    model = OCTClassifier(estimator)
    model.fit(X_full, y_full)
    model.save(args.out)
    print(f"Saved checkpoint to {args.out}")


def _phase_subprocess_cmd(args: argparse.Namespace) -> list[str]:
    """Common flags shared by every phase subprocess the orchestrator launches.

    Includes --minority-oversample whenever it's non-default, even though
    --phase final-refit ignores it (that phase never oversamples the full
    dataset) -- harmless today because _run_real_training only ever launches
    final-refit when minority_oversample == 1 (see run_full_pipeline below),
    so the flag is never actually forwarded in that case. Noted here so a
    future change to that gating doesn't silently start masking a real
    oversample-vs-full-refit policy conflict.
    """
    cmd = [sys.executable, "-m", "app.ml.train", "--data-dir", args.data_dir, "--out", args.out]
    if args.no_rotation_augment:
        cmd.append("--no-rotation-augment")
    if args.no_flip_augment:
        cmd.append("--no-flip-augment")
    if args.no_brightness_augment:
        cmd.append("--no-brightness-augment")
    if args.no_shift_augment:
        cmd.append("--no-shift-augment")
    if args.no_perspective_augment:
        cmd.append("--no-perspective-augment")
    if args.minority_oversample != 1:
        cmd += ["--minority-oversample", str(args.minority_oversample)]
    return cmd


def _run_real_training(args: argparse.Namespace) -> None:
    """Thin orchestrator for the real-dataset training pipeline (round 30):
    launches the select phase, then (if eligible -- see below) multi-seed
    evaluation and the final-refit phase, each as its own fresh subprocess
    with live (uncaptured) output so progress still streams normally. This
    process itself never touches a feature matrix or a fitted model, so its
    own memory stays trivial throughout -- the fix for the OOM kills round 30
    hit twice. See _run_select_phase's docstring for what this does and
    doesn't guarantee: each phase's own peak memory is still unbounded, this
    just stops phases from stacking on top of each other.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        select_output_path = tmp.name
    try:
        subprocess.run(
            _phase_subprocess_cmd(args) + ["--phase", "select", "--phase-output", select_output_path],
            check=True,
        )
        select_result = json.loads(Path(select_output_path).read_text())
    finally:
        Path(select_output_path).unlink(missing_ok=True)

    best_name = select_result["selected_model"]
    # Mirrors the original single-process X_all gating: no full-dataset refit
    # (and no multi-seed characterization of it) when minority-oversample is
    # active, since that's a non-default experimental path this project found
    # doesn't help -- see README and _oversample_minority's docstring.
    run_full_pipeline = args.minority_oversample == 1

    multi_seed_metrics = None
    shipped_on_full_dataset = False
    if run_full_pipeline:
        print("Evaluating across multiple train/test splits (round 15) to characterize typical performance:")
        multi_seed_metrics = _multi_seed_evaluation(
            args.data_dir,
            best_name,
            rotation_augment=select_result["rotation_augment_enabled"],
            flip_augment=select_result["flip_augment_enabled"],
            brightness_augment=select_result["brightness_augment_enabled"],
            shift_augment=select_result["shift_augment_enabled"],
            perspective_augment=select_result["perspective_augment_enabled"],
        )
        print(
            f"  test accuracy: {multi_seed_metrics['test_accuracy']['mean']:.3f} "
            f"(+/- {multi_seed_metrics['test_accuracy']['std']:.3f})"
        )
        print(
            f"  DME precision: {multi_seed_metrics['dme_precision']['mean']:.3f} "
            f"(+/- {multi_seed_metrics['dme_precision']['std']:.3f})"
        )
        print(
            f"  DME recall: {multi_seed_metrics['dme_recall']['mean']:.3f} "
            f"(+/- {multi_seed_metrics['dme_recall']['std']:.3f})"
        )

        subprocess.run(
            _phase_subprocess_cmd(args) + ["--phase", "final-refit", "--estimator-name", best_name],
            check=True,
        )
        shipped_on_full_dataset = True

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_samples_after_oversampling": select_result["n_train_samples_after_oversampling"],
        "minority_oversample_factor": args.minority_oversample,
        "rotation_augment_enabled": select_result["rotation_augment_enabled"],
        "n_train_samples_after_rotation_augment": select_result["n_train_samples_after_rotation_augment"],
        "flip_augment_enabled": select_result["flip_augment_enabled"],
        "n_train_samples_after_flip_augment": select_result["n_train_samples_after_flip_augment"],
        "brightness_augment_enabled": select_result["brightness_augment_enabled"],
        "n_train_samples_after_brightness_augment": select_result["n_train_samples_after_brightness_augment"],
        "shift_augment_enabled": select_result["shift_augment_enabled"],
        "n_train_samples_after_shift_augment": select_result["n_train_samples_after_shift_augment"],
        "perspective_augment_enabled": select_result["perspective_augment_enabled"],
        "n_train_samples_after_perspective_augment": select_result["n_train_samples_after_perspective_augment"],
        "classes": CLASSES,
        "trained_on_real_patient_data": True,
        **{
            k: select_result[k]
            for k in (
                "data_source",
                "n_samples",
                "selected_model",
                "cv_balanced_accuracy_by_model",
                "test_accuracy",
                "classification_report",
                "confusion_matrix",
                "dme_threshold_analysis",
                "dme_calibration_metrics",
            )
        },
        "multi_seed_evaluation": multi_seed_metrics,
        "shipped_checkpoint_trained_on_full_dataset": shipped_on_full_dataset,
    }
    metrics_path = Path(args.out).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model card to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None, help="Path to a OCT-AND-EYE-FUNDUS-DATASET checkout")
    parser.add_argument("--synthetic-per-class", type=int, default=250)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation of synthetic images")
    parser.add_argument(
        "--minority-oversample",
        type=int,
        default=1,
        help="Top up minority classes (e.g. DME) with augmented copies up to this multiple of their original "
        "count, capped at the majority class size. Default 1 (disabled): on this dataset, oversampling did not "
        "raise DME recall (same cases missed at every factor tried) and lowered precision -- see README.",
    )
    parser.add_argument(
        "--no-rotation-augment",
        action="store_true",
        help="Disable rotation-augmented training (round 23, real data only): by default, each real "
        "training image is supplemented with copies rotated by -10/-5/+5/+10 degrees so the model "
        "doesn't collapse on a mildly rotated photo (e.g. a phone photo of a screen/printout) -- see "
        "README round 19/23. Measured to improve both the unrotated baseline and rotation robustness, "
        "so it ships enabled by default; this flag reverts to the round 19-22 baseline behavior.",
    )
    parser.add_argument(
        "--no-flip-augment",
        action="store_true",
        help="Disable flip-augmented training (round 29, real data only): by default, each (possibly "
        "rotation-augmented) real training image is supplemented with a horizontally-mirrored copy -- "
        "see README round 26/29. Measured across 5 split seeds against the current shipped recipe: DME "
        "recall improved in 5/5 seeds, accuracy in 4/5, precision in 4/5, so it ships enabled by default; "
        "this flag reverts to the round 23-28 baseline behavior.",
    )
    parser.add_argument(
        "--no-brightness-augment",
        action="store_true",
        help="Disable brightness/contrast-augmented training (round 30, real data only): by default, each "
        "(rotation+flip-augmented) real training image is supplemented with a randomized brightness/contrast "
        "perturbed copy -- see README round 30. Measured across 5 split seeds against the current shipped "
        "recipe: accuracy improved in 4/5 seeds, DME precision in 4/5 (the largest single-round precision "
        "gain so far), DME recall stayed flat within noise, so it ships enabled by default; this flag "
        "reverts to the round 29 baseline behavior.",
    )
    parser.add_argument(
        "--no-shift-augment",
        action="store_true",
        help="Disable shift-augmented training (round 40, real data only): by default, each "
        "(rotation+flip+brightness-augmented) real training image is supplemented with a randomly "
        "translated copy, padded with black rather than wrapped -- see README round 40. Measured across "
        "5 split seeds against the current shipped recipe: accuracy improved in 4/5 seeds (flat in the "
        "5th), DME precision in 4/5 (flat in the 5th), DME recall in 2/5 (flat in 2, down in 1 by less "
        "than the metric's own cross-seed std) -- never down on accuracy or precision, so it ships "
        "enabled by default; this flag reverts to the round 30 baseline behavior.",
    )
    parser.add_argument(
        "--no-perspective-augment",
        action="store_true",
        help="Disable perspective/keystone-warp-augmented training (round 44, real data only): by default, "
        "each (rotation+flip+brightness-augmented) real training image is supplemented with a copy warped "
        "by displacing its 4 corners inward by up to 6%% of width/height (a phone photo of a screen/printout "
        "is rarely taken perfectly perpendicular), padded with black rather than stretched -- see README "
        "round 44. Measured across 5 split seeds against the current shipped recipe (base+shift): accuracy "
        "improved in 4/5 seeds (down in 1 by less than the metric's own cross-seed std), DME precision up "
        "substantially in 3/5 (down negligibly in 2), DME recall up substantially in 3/5 (down modestly in "
        "2, losses smaller than the gains) -- the strongest and cleanest result of any augmentation round so "
        "far, so it ships enabled by default; this flag reverts to the round 40 baseline behavior.",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument(
        "--multi-seed-worker-seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # internal: subprocess entry point for _multi_seed_evaluation, see round 30
    )
    parser.add_argument("--multi-seed-worker-estimator", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase",
        choices=["select", "final-refit"],
        default=None,
        help=argparse.SUPPRESS,  # internal: subprocess entry points for _run_real_training, see round 30
    )
    parser.add_argument("--phase-output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--estimator-name", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # These internal subprocess-entry-point flags are always paired by their
    # own caller (_multi_seed_evaluation, _run_real_training), but each pairs
    # a cheap flag with an expensive phase -- load/augment/fit can run for
    # minutes to hours before the missing partner flag would otherwise surface
    # as a confusing KeyError/TypeError deep inside the phase. Fail fast
    # instead, before any of that work starts. All three also need --data-dir
    # (a test writing a bare `--multi-seed-worker-seed ... --multi-seed-worker-
    # estimator ...` invocation caught this: without it, _load_real_dataset's
    # `Path(None)` raises a raw TypeError instead of an actionable message).
    if args.multi_seed_worker_seed is not None and args.multi_seed_worker_estimator is None:
        parser.error("--multi-seed-worker-seed requires --multi-seed-worker-estimator")
    if args.phase == "select" and args.phase_output is None:
        parser.error("--phase select requires --phase-output")
    if args.phase == "final-refit" and args.estimator_name is None:
        parser.error("--phase final-refit requires --estimator-name")
    if (args.multi_seed_worker_seed is not None or args.phase is not None) and args.data_dir is None:
        parser.error("--multi-seed-worker-seed/--phase requires --data-dir")

    if args.multi_seed_worker_seed is not None:
        # Isolated single-seed evaluation, invoked as a fresh subprocess by
        # _multi_seed_evaluation -- print one JSON line and exit before any of
        # the normal training flow below runs.
        result = _multi_seed_worker_evaluate(
            args.data_dir,
            args.multi_seed_worker_seed,
            args.multi_seed_worker_estimator,
            rotation_augment=not args.no_rotation_augment,
            flip_augment=not args.no_flip_augment,
            brightness_augment=not args.no_brightness_augment,
            shift_augment=not args.no_shift_augment,
            perspective_augment=not args.no_perspective_augment,
        )
        print(json.dumps(result))
        return

    if args.phase == "select":
        result = _run_select_phase(args)
        Path(args.phase_output).write_text(json.dumps(result))
        return

    if args.phase == "final-refit":
        _run_final_refit_phase(args)
        return

    if not args.data_dir:
        _run_synthetic_training(args)
        return

    _run_real_training(args)


if __name__ == "__main__":
    main()
