"""Trains the OCTera diagnosis classifier.

Without --data-dir, trains on procedurally generated synthetic images as a
smoke test of the pipeline only -- the resulting checkpoint is NOT clinically
valid. Point --data-dir at a Kermany OCT2017-style layout
(``<data-dir>/train/<CLASS>/*.jpeg`` with CLASS in NORMAL/CNV/DME/DRUSEN,
e.g. https://www.kaggle.com/datasets/paultimothymooney/kermany2018) to train
on real data.

Picks the best of a few candidate scikit-learn classifiers via cross-
validation, then reports held-out test metrics and writes a model-card
JSON (metrics.json) alongside the checkpoint.

Usage:
    python -m app.ml.train --data-dir /path/to/OCT2017
    python -m app.ml.train  # synthetic smoke test
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from app.ml.features import extract_features
from app.ml.model import CLASSES, DEFAULT_CHECKPOINT_PATH, OCTClassifier
from app.ml.synthetic_dataset import generate_dataset


def _candidate_estimators(random_state: int = 42) -> dict[str, object]:
    return {
        "mlp_128_64": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=600, random_state=random_state
        ),
        "mlp_64": MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", alpha=1e-4, max_iter=600, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


def _load_real_dataset(data_dir: str) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    root = Path(data_dir) / "train"
    for cls in CLASSES:
        folder = root / cls
        if not folder.exists():
            continue
        for file in list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
            images.append(np.array(Image.open(file)))
            labels.append(cls)
    return images, labels


def select_best_model(X: np.ndarray, y: np.ndarray, cv_folds: int = 4) -> tuple[str, object, dict[str, float]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores: dict[str, float] = {}
    for name, estimator in _candidate_estimators().items():
        fold_scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
        scores[name] = float(fold_scores.mean())
        print(f"  {name}: cv accuracy = {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")

    best_name = max(scores, key=scores.get)
    return best_name, _candidate_estimators()[best_name], scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None, help="Path to a Kermany OCT2017-style dataset")
    parser.add_argument("--synthetic-per-class", type=int, default=250)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation of synthetic images")
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    args = parser.parse_args()

    if args.data_dir:
        raw_images, labels = _load_real_dataset(args.data_dir)
        if not raw_images:
            raise SystemExit(f"No images found under {args.data_dir}/train/<CLASS>/")
        data_source = f"real:{args.data_dir}"
        print(f"Loaded {len(raw_images)} real images from {args.data_dir}")
    else:
        print(
            "No --data-dir given: training on synthetic procedural data. "
            "This checkpoint is a pipeline smoke test only and is NOT clinically valid."
        )
        data_source = "synthetic"
        raw_images, labels = generate_dataset(n_per_class=args.synthetic_per_class, augment=not args.no_augment)

    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Selecting best model via cross-validation on the training split:")
    best_name, best_estimator, cv_scores = select_best_model(X_train, y_train)
    print(f"Selected: {best_name} (cv accuracy = {cv_scores[best_name]:.3f})")

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
        "cv_accuracy_by_model": cv_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASSES).tolist(),
        "clinically_valid": data_source != "synthetic",
    }
    metrics_path = Path(args.out).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model card to {metrics_path}")


if __name__ == "__main__":
    main()
