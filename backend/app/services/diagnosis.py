"""Diagnosis module.

Stub implementation that derives plausible diagnosis probabilities from
segmentation features. Swap the body of `predict_diagnoses` with a trained
multi-label classifier without changing the return contract.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

DISEASE_CATALOG = [
    ("normal", "Без признаков патологии"),
    ("dme", "Диабетический макулярный отек"),
    ("amd", "Возрастная макулярная дегенерация"),
    ("erm", "Эпиретинальная мембрана"),
    ("glaucoma", "Глаукома"),
    ("serous_pigment_epithelial_detachment", "Серозная отслойка пигментного эпителия"),
]


@dataclass
class Diagnosis:
    code: str
    label: str
    probability: float


def predict_diagnoses(image_path: str, layer_thickness_um: dict[str, float]) -> list[Diagnosis]:
    seed = int(hashlib.sha256(image_path.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    avg_thickness = sum(layer_thickness_um.values()) / max(len(layer_thickness_um), 1)
    logits = rng.normal(loc=0.0, scale=1.0, size=len(DISEASE_CATALOG))
    logits[0] += 1.5 if avg_thickness < 70 else -0.5  # bias towards "normal" for thinner retina

    probabilities = np.exp(logits) / np.exp(logits).sum()

    diagnoses = [
        Diagnosis(code=code, label=label, probability=round(float(prob), 3))
        for (code, label), prob in zip(DISEASE_CATALOG, probabilities)
    ]
    return sorted(diagnoses, key=lambda d: d.probability, reverse=True)
