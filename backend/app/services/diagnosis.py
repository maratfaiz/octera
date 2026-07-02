"""Diagnosis module.

Runs the trained classifier from app/ml/ over the four disease categories
covered by the Kermany OCT2017 dataset layout: NORMAL, CNV (choroidal
neovascularization), DME (diabetic macular edema) and DRUSEN. Falls back to
a deterministic pseudo-random stub when no trained checkpoint is present, so
the API keeps working end to end without the ML dependency wired up.

The checkpoint shipped in app/ml/artifacts/ was trained on synthetic
procedural images (see app/ml/synthetic_dataset.py) as a smoke test of the
pipeline only -- it is NOT trained on real patient data and must not be used
for anything beyond demonstrating the mechanism. Retrain on a real dataset
via `python -m app.ml.train --data-dir <path-to-kermany-oct2017>` before
relying on its output.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from app.ml import inference as ml_inference
from app.ml.model import DEFAULT_CHECKPOINT_PATH

CLASS_LABELS_RU = {
    "NORMAL": "Без признаков патологии",
    "CNV": "Хориоидальная неоваскуляризация",
    "DME": "Диабетический макулярный отек",
    "DRUSEN": "Друзы (ранняя возрастная макулярная дегенерация)",
}


@dataclass
class Diagnosis:
    code: str
    label: str
    probability: float


def _fallback_diagnoses(image_path: str) -> list[Diagnosis]:
    seed = int(hashlib.sha256(image_path.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=len(CLASS_LABELS_RU))
    probabilities = np.exp(logits) / np.exp(logits).sum()
    return [
        Diagnosis(code=code, label=label, probability=round(float(p), 3))
        for (code, label), p in zip(CLASS_LABELS_RU.items(), probabilities)
    ]


def predict_diagnoses(image_path: str) -> list[Diagnosis]:
    if not DEFAULT_CHECKPOINT_PATH.exists():
        diagnoses = _fallback_diagnoses(image_path)
    else:
        probabilities = ml_inference.predict(image_path, DEFAULT_CHECKPOINT_PATH)
        diagnoses = [
            Diagnosis(code=code, label=CLASS_LABELS_RU[code], probability=round(float(p), 3))
            for code, p in probabilities.items()
        ]
    return sorted(diagnoses, key=lambda d: d.probability, reverse=True)
