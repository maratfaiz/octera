"""Diagnosis module.

Runs the trained classifier from app/ml/ to detect diabetic macular edema
(DME) vs. no DME. Falls back to a deterministic pseudo-random stub when no
trained checkpoint is present, so the API keeps working end to end without
the ML dependency wired up.

The shipped checkpoint (app/ml/artifacts/oct_classifier.joblib) is trained
on the "Dataset of Eye Fundus and OCT Images for the study of Diabetic
Macular Edema and Diabetic Retinopathy" (Hughes Cano, Olivares Pinto &
Thebault -- CONACYT/UNAM/IMO/APEC/INDEREB), i.e. real de-identified patient
OCT scans with DME diagnosed by retinal ophthalmologists -- not synthetic
data. That said, it is a simple pixel-downscale + shallow classifier
trained on ~1,100 images from one study; it has NOT gone through clinical
validation and must not be used for anything beyond demonstrating the
mechanism. See app/ml/train.py and app/ml/artifacts/metrics.json for
training details and measured accuracy.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from app.ml import inference as ml_inference
from app.ml.model import DEFAULT_CHECKPOINT_PATH

CLASS_LABELS_RU = {
    "NORMAL": "Без признаков патологии",
    "DME": "Диабетический макулярный отек",
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
