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
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    }


def _load_real_dataset(data_dir: str) -> tuple[list[np.ndarray], list[str]]:
    """Loads the OCT-AND-EYE-FUNDUS-DATASET layout: <data_dir>/OCT.csv +
    <data_dir>/OCT/OCT*/<Name>.jpg, labeled by the CSV's DME column.
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
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = image_index.get(row["Name"])
            if path is None:
                continue
            images.append(np.array(Image.open(path).convert("L")))
            labels.append("DME" if row["DME"].strip() == "1" else "NORMAL")
    return images, labels


def select_best_model(X: np.ndarray, y: np.ndarray, cv_folds: int = 4) -> tuple[str, object, dict[str, float]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores: dict[str, float] = {}
    for name, estimator in _candidate_estimators().items():
        fold_scores = cross_val_score(estimator, X, y, cv=cv, scoring="balanced_accuracy")
        scores[name] = float(fold_scores.mean())
        print(f"  {name}: cv balanced accuracy = {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")

    best_name = max(scores, key=scores.get)
    return best_name, _candidate_estimators()[best_name], scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None, help="Path to a OCT-AND-EYE-FUNDUS-DATASET checkout")
    parser.add_argument("--synthetic-per-class", type=int, default=250)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation of synthetic images")
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    args = parser.parse_args()

    if args.data_dir:
        raw_images, labels = _load_real_dataset(args.data_dir)
        if not raw_images:
            raise SystemExit(f"No images matched between OCT.csv and OCT/OCT*/*.jpg under {args.data_dir}")
        data_source = "real:OCT-AND-EYE-FUNDUS-DATASET"
        print(f"Loaded {len(raw_images)} real images from {args.data_dir}")
    else:
        print(
            "No --data-dir given: training on synthetic procedural data. "
            "This checkpoint is a pipeline smoke test only and is NOT trained on real patients."
        )
        data_source = "synthetic"
        raw_images, labels = generate_dataset(n_per_class=args.synthetic_per_class, augment=not args.no_augment)

    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

    model.save(args.out)
    print(f"Saved checkpoint to {args.out}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "n_samples": len(raw_images),
        "classes": CLASSES,
        "selected_model": best_name,
        "cv_balanced_accuracy_by_model": cv_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASSES).tolist(),
        "trained_on_real_patient_data": data_source != "synthetic",
    }
    metrics_path = Path(args.out).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model card to {metrics_path}")


if __name__ == "__main__":
    main()
