from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN"]

DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "artifacts" / "oct_classifier.joblib"


def default_estimator() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        max_iter=600,
        random_state=42,
    )


class OCTClassifier:
    """Thin wrapper around a scikit-learn classifier for OCT B-scan classification.

    Any scikit-learn-compatible classifier (MLP, RandomForest, SVM, ...) can be
    passed in -- train.py picks the best one via cross-validation.
    """

    def __init__(self, clf: Any | None = None):
        self.clf = clf or default_estimator()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OCTClassifier":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path: str | Path) -> "OCTClassifier":
        return cls(joblib.load(path))
