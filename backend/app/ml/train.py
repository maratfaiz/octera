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

from app.ml.augmentation import augment as augment_image
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
    right at test time. Union-find over (patient_id, content_hash) pairs merges any
    patient IDs that ever share a duplicate image into one group.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_hash: dict[str, list[str]] = {}
    for pid, content_hash in zip(patient_ids, content_hashes):
        by_hash.setdefault(content_hash, []).append(pid)
    for pids in by_hash.values():
        for other in pids[1:]:
            union(pids[0], other)

    return [find(pid) for pid in patient_ids]


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
    images: list[np.ndarray], labels: list[str], rng: np.random.Generator, max_factor: int = 4
) -> tuple[list[np.ndarray], list[str]]:
    """Tops up under-represented classes with augmented copies of their own images.

    MLPClassifier (the model round 3 selected) has no `class_weight` knob, so this
    was tried as a way to address DME's ~15% share of the dataset from the data
    side. Empirically (round 4) it does not raise DME recall -- the same held-out
    cases are missed at every factor tried -- and lowers precision, so it ships
    disabled by default (see the --minority-oversample flag and the README).
    Capped at `max_factor`x the original count per class so it stays a top-up,
    not a full rebalance to majority size.
    """
    counts = Counter(labels)
    majority_count = max(counts.values())
    images_by_class = {cls: [img for img, lbl in zip(images, labels) if lbl == cls] for cls in counts}

    out_images, out_labels = list(images), list(labels)
    for cls, count in counts.items():
        target = min(majority_count, count * max_factor)
        cls_images = images_by_class[cls]
        for i in range(target - count):
            out_images.append(augment_image(cls_images[i % len(cls_images)], rng))
            out_labels.append(cls)
    return out_images, out_labels


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
        "classification_report": classification_report(y_true, preds, output_dict=True),
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


def select_best_model(
    X: np.ndarray, y: np.ndarray, cv_folds: int = 4, groups: np.ndarray | None = None
) -> tuple[str, object, dict[str, float]]:
    cv = (
        StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        if groups is not None
        else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    )
    scores: dict[str, float] = {}
    for name, estimator in _candidate_estimators().items():
        fold_scores = cross_val_score(estimator, X, y, groups=groups, cv=cv, scoring="balanced_accuracy")
        scores[name] = float(fold_scores.mean())
        print(f"  {name}: cv balanced accuracy = {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")

    best_name = max(scores, key=scores.get)
    return best_name, _candidate_estimators()[best_name], scores


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
        images_train, labels_train = _oversample_minority(images_train, labels_train, rng, args.minority_oversample)
        print(f"Oversampled minority classes in training split: {dict(before)} -> {dict(Counter(labels_train))}")
        # Augmented copies aren't tied to a well-defined patient group; skip group-aware
        # internal CV rather than mis-group them (this flag ships disabled by default).
        groups_train = None

    X_train = np.stack([extract_features(Image.fromarray(img)) for img in images_train])
    X_test = np.stack([extract_features(Image.fromarray(img)) for img in images_test])
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

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

    model.save(args.out)
    print(f"Saved checkpoint to {args.out}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "n_samples": len(raw_images),
        "n_train_samples_after_oversampling": len(images_train),
        "minority_oversample_factor": args.minority_oversample,
        "classes": CLASSES,
        "selected_model": best_name,
        "cv_balanced_accuracy_by_model": cv_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASSES).tolist(),
        "trained_on_real_patient_data": data_source != "synthetic",
        "dme_threshold_analysis": threshold_analysis,
        "dme_calibration_metrics": calibration_metrics,
    }
    metrics_path = Path(args.out).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model card to {metrics_path}")


if __name__ == "__main__":
    main()
