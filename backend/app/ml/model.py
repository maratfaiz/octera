from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN"]

DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "artifacts" / "oct_classifier.joblib"


class OCTClassifier:
    """Thin wrapper around a scikit-learn MLP for OCT B-scan classification."""

    def __init__(self, clf: MLPClassifier | None = None):
        self.clf = clf or MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=400,
            random_state=42,
        )

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
