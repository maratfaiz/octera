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
from app.ml.augmentation import rotate as rotate_image
from app.ml.features import extract_features
from app.ml.model import CLASSES, DEFAULT_CHECKPOINT_PATH, OCTClassifier
from app.ml.synthetic_dataset import generate_dataset


def _candidate_estimators(random_state: int = 42) -> dict[str, object]:
    # Each estimator is wrapped in a StandardScaler so the handful of domain
    # features (band thickness, dark-zone fraction, ...) get comparable weight
    # to the 2304 raw-pixel features instead of being drowned out -- and the
    # saved checkpoint carries the scaler, so inference stays consistent.
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
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        ),
        # Round 11: the 1578-dim HOG+domain vector against ~700-900 patient-grouped
        # training images is a high-dim/low-N regime prone to overfitting, especially
        # for SVM. PCA(30) ahead of svm_rbf measured cv balanced accuracy 0.879 vs
        # 0.816-0.831 for every un-reduced or MLP+PCA combination tried -- see README.
        "svm_rbf_pca30": make_pipeline(
            StandardScaler(),
            PCA(n_components=30, random_state=random_state),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
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
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
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


def _multi_seed_evaluation(
    raw_images: list[np.ndarray],
    labels: list[str],
    split_groups: list[str],
    estimator_factory,
    seeds: tuple[int, ...] = (42, 7, 99, 123, 2024),
    rotation_augment: bool = False,
    flip_augment: bool = False,
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
    model no longer in production. `flip_augment` (round 29) does the same
    for mirrored copies, applied on top of rotation augmentation. Costs 5x
    the feature extraction of the non-augmented path (each seed's augmented
    training images are re-extracted from scratch, since the augmented set
    differs per seed), but this function is already documented as a
    characterization exercise, not a hot path.
    """
    y_all = np.array(labels)

    def _summary(values: list[float]) -> dict:
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": [float(v) for v in values]}

    accuracies: list[float] = []
    dme_precisions: list[float] = []
    dme_recalls: list[float] = []
    skipped_seeds: list[int] = []
    for seed in seeds:
        train_idx, test_idx = _group_aware_split(labels, split_groups, random_state=seed)
        train_images = [raw_images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        if rotation_augment:
            train_images, train_labels, _ = _rotation_augment(train_images, train_labels)
        if flip_augment:
            train_images, train_labels, _ = _flip_augment(train_images, train_labels)

        X_train_seed = np.stack([extract_features(Image.fromarray(img)) for img in train_images])
        y_train_seed = np.array(train_labels)
        X_test_seed = np.stack([extract_features(Image.fromarray(raw_images[i])) for i in test_idx])

        model = estimator_factory()
        model.fit(X_train_seed, y_train_seed)
        preds = model.predict(X_test_seed)
        rep = classification_report(y_all[test_idx], preds, output_dict=True, zero_division=0)
        accuracies.append(rep["accuracy"])
        # A grouped split can, on this small dataset, draw a test fold with zero true
        # DME cases -- "DME" key is then absent from rep entirely. That's different
        # from the model genuinely scoring 0 precision/recall on a fold that DID have
        # DME cases; silently defaulting both to 0.0 would understate the real mean/std
        # by mixing in seeds that had nothing to score. Exclude those seeds instead.
        if "DME" in rep:
            dme_precisions.append(rep["DME"]["precision"])
            dme_recalls.append(rep["DME"]["recall"])
        else:
            skipped_seeds.append(seed)

    return {
        "seeds": list(seeds),
        "rotation_augment": rotation_augment,
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
    candidates = _candidate_estimators()
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
    # clone() gives a fresh unfitted copy of the SAME estimator that was already
    # built above -- cheaper and clearer than reconstructing all 5 candidate
    # pipelines again just to keep 1.
    return best_name, clone(candidates[best_name]), scores


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
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    args = parser.parse_args()

    groups: list[str] | None
    if args.data_dir:
        raw_images, labels, groups = _load_real_dataset(args.data_dir)
        if not raw_images:
            raise SystemExit(f"No images matched between OCT.csv and OCT/OCT*/*.jpg under {args.data_dir}")
        data_source = "real:OCT-AND-EYE-FUNDUS-DATASET"
        n_patients = len(set(groups))
        print(f"Loaded {len(raw_images)} real images from {args.data_dir} ({n_patients} unique patients)")
    else:
        print(
            "No --data-dir given: training on synthetic procedural data. "
            "This checkpoint is a pipeline smoke test only and is NOT trained on real patients."
        )
        data_source = "synthetic"
        raw_images, labels = generate_dataset(n_per_class=args.synthetic_per_class, augment=not args.no_augment)
        groups = None  # no patient concept for procedurally generated images

    # A patient's images (both eyes, repeat visits) never span both splits -- see
    # _group_aware_split and README round 10. Synthetic images have no group
    # concept, so each gets its own trivial one-sample group (a no-op constraint).
    split_groups = groups if groups is not None else [str(i) for i in range(len(labels))]
    train_idx, test_idx = _group_aware_split(labels, split_groups)
    images_train = [raw_images[i] for i in train_idx]
    images_test = [raw_images[i] for i in test_idx]
    labels_train = [labels[i] for i in train_idx]
    labels_test = [labels[i] for i in test_idx]
    groups_train = np.array(split_groups)[train_idx] if groups is not None else None

    if args.minority_oversample > 1:
        rng = np.random.default_rng(42)
        before = Counter(labels_train)
        groups_train_list = list(groups_train) if groups_train is not None else None
        images_train, labels_train, groups_train_list = _oversample_minority(
            images_train, labels_train, rng, groups=groups_train_list, max_factor=args.minority_oversample
        )
        groups_train = np.array(groups_train_list) if groups_train_list is not None else None
        print(f"Oversampled minority classes in training split: {dict(before)} -> {dict(Counter(labels_train))}")
    n_train_after_oversampling = len(images_train)

    # Unaugmented features -- always needed for X_all (the round-16 "ship on 100%
    # of data" refit when rotation augmentation is disabled below), regardless of
    # what candidate selection and the final single-split fit end up using.
    X_train_raw = np.stack([extract_features(Image.fromarray(img)) for img in images_train])
    X_test = np.stack([extract_features(Image.fromarray(img)) for img in images_test])
    y_train_raw = np.array(labels_train)
    y_test = np.array(labels_test)

    # Round 23: rotation-augmented training (see _rotation_augment). Originally
    # applied only after candidate selection, matching every prior round's
    # methodology -- but round 27 found that selecting a model/hyperparameter on
    # data different from what's actually fit afterward can pick a measurably
    # worse choice (PCA(30), tuned in round 11 pre-augmentation, loses to
    # PCA(75) once evaluated on the augmented data it's really trained on).
    # Augmentation now happens before selection, so both selection and the
    # final fit see the exact same data.
    rotation_augmented = data_source != "synthetic" and not args.no_rotation_augment
    if rotation_augmented:
        groups_train_list = list(groups_train) if groups_train is not None else None
        images_train, labels_train, groups_train_list = _rotation_augment(
            images_train, labels_train, groups_train_list
        )
        groups_train = np.array(groups_train_list) if groups_train_list is not None else None
        print(
            f"Rotation-augmented training split (round 23): "
            f"{n_train_after_oversampling} -> {len(images_train)} images"
        )
        # _rotation_augment always keeps the original images unchanged as a prefix
        # before appending rotated copies, so X_train_raw already covers
        # images_train[:n_train_after_oversampling] -- only the newly-appended
        # rotated copies need feature extraction, not the whole set.
        X_new = np.stack(
            [extract_features(Image.fromarray(img)) for img in images_train[n_train_after_oversampling:]]
        )
        X_train = np.concatenate([X_train_raw, X_new])
        y_train = np.array(labels_train)
    else:
        X_train, y_train = X_train_raw, y_train_raw
    n_train_after_rotation_augment = len(images_train)

    # Round 29: flip-augmented training (see _flip_augment), applied on top of
    # whatever rotation augmentation produced above -- same before-selection
    # placement and same reasoning as round 27/28 for rotation augmentation,
    # so selection and the final fit again see identical data.
    flip_augmented = data_source != "synthetic" and not args.no_flip_augment
    if flip_augmented:
        groups_train_list = list(groups_train) if groups_train is not None else None
        images_train, labels_train, groups_train_list = _flip_augment(images_train, labels_train, groups_train_list)
        groups_train = np.array(groups_train_list) if groups_train_list is not None else None
        print(f"Flip-augmented training split (round 29): {n_train_after_rotation_augment} -> {len(images_train)} images")
        # Same reuse pattern as rotation augmentation above: _flip_augment keeps
        # the input unchanged as a prefix, so X_train already covers it -- only
        # the newly-appended mirrored tail needs extracting.
        X_new = np.stack(
            [extract_features(Image.fromarray(img)) for img in images_train[n_train_after_rotation_augment:]]
        )
        X_train = np.concatenate([X_train, X_new])
        y_train = np.array(labels_train)
    n_train_after_flip_augment = len(images_train)

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

    X_all: np.ndarray | None = None
    y_all: np.ndarray | None = None
    if data_source != "synthetic" and args.minority_oversample == 1:
        X_all = np.empty((len(raw_images), X_train_raw.shape[1]), dtype=X_train_raw.dtype)
        X_all[train_idx] = X_train_raw
        X_all[test_idx] = X_test
        y_all = np.array(labels)

    multi_seed_metrics = None
    if X_all is not None:
        print("Evaluating across multiple train/test splits (round 15) to characterize typical performance:")
        multi_seed_metrics = _multi_seed_evaluation(
            raw_images,
            labels,
            split_groups,
            lambda: clone(best_estimator),
            rotation_augment=rotation_augmented,
            flip_augment=flip_augmented,
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

    shipped_on_full_dataset = False
    if X_all is not None:
        # All metrics above (test_accuracy, threshold_analysis, calibration_metrics,
        # multi_seed_metrics) were measured on a model trained on ~80% of the data --
        # that's the honest way to estimate generalization. But this dataset is small
        # (1113 images total) and every labeled image is precious, so the checkpoint
        # actually shipped is refit on 100% of it (round 16), standard practice once
        # validation is done: more real training data should only help, not hurt, a
        # model that already generalized well on the held-out estimates above. The
        # metrics in this model card describe the held-out-validated *approach*, not
        # a measurement of this exact final artifact.
        print("Retraining final checkpoint on 100% of the data (round 16) for shipping:")
        full_images, full_labels = raw_images, labels
        X_full, y_full = X_all, y_all
        if rotation_augmented:
            # Same reuse as the earlier augmentation step: _rotation_augment keeps
            # the input unchanged as a prefix, and X_full already has its features
            # (assembled above in the same order) -- only the new rotated tail needs
            # extracting.
            n_before = len(full_images)
            full_images, full_labels, _ = _rotation_augment(full_images, full_labels)
            X_new_full = np.stack([extract_features(Image.fromarray(img)) for img in full_images[n_before:]])
            X_full = np.concatenate([X_full, X_new_full])
            y_full = np.array(full_labels)
            print(f"Rotation-augmented full dataset for final refit: {n_before} -> {len(full_images)} images")
        if flip_augmented:
            # Round 29: same reuse pattern, applied on top of the rotation-augmented
            # full set (or the raw full set if rotation augmentation is disabled).
            n_before = len(full_images)
            full_images, full_labels, _ = _flip_augment(full_images, full_labels)
            X_new_full = np.stack([extract_features(Image.fromarray(img)) for img in full_images[n_before:]])
            X_full = np.concatenate([X_full, X_new_full])
            y_full = np.array(full_labels)
            print(f"Flip-augmented full dataset for final refit: {n_before} -> {len(full_images)} images")
        model = OCTClassifier(clone(best_estimator))
        model.fit(X_full, y_full)
        shipped_on_full_dataset = True

    model.save(args.out)
    print(f"Saved checkpoint to {args.out}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "n_samples": len(raw_images),
        "n_train_samples_after_oversampling": n_train_after_oversampling,
        "minority_oversample_factor": args.minority_oversample,
        "rotation_augment_enabled": rotation_augmented,
        "n_train_samples_after_rotation_augment": n_train_after_rotation_augment,
        "flip_augment_enabled": flip_augmented,
        "n_train_samples_after_flip_augment": n_train_after_flip_augment,
        "classes": CLASSES,
        "selected_model": best_name,
        "cv_balanced_accuracy_by_model": cv_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASSES).tolist(),
        "trained_on_real_patient_data": data_source != "synthetic",
        "dme_threshold_analysis": threshold_analysis,
        "dme_calibration_metrics": calibration_metrics,
        "multi_seed_evaluation": multi_seed_metrics,
        "shipped_checkpoint_trained_on_full_dataset": shipped_on_full_dataset,
    }
    metrics_path = Path(args.out).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model card to {metrics_path}")


if __name__ == "__main__":
    main()
