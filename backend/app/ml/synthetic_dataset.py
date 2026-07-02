"""Procedural synthetic OCT-like images used only to smoke-test the training
pipeline when a real dataset is not available. These are NOT real retinal
scans and a model trained solely on them must never be used for anything
beyond verifying the pipeline works end to end.
"""

import numpy as np

from app.ml.model import CLASSES

IMG_SIZE = 96


def _base_layers(rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0, 1, IMG_SIZE)[:, None]
    phase = rng.uniform(0, 1)
    bands = np.sin(y * 14 + phase) * 0.15 + 0.5
    img = np.tile(bands, (1, IMG_SIZE))
    img += rng.normal(0, 0.03, size=img.shape)
    return img


def _add_blob(img: np.ndarray, rng: np.random.Generator, cx: float, cy: float, radius: float, intensity: float) -> np.ndarray:
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-dist_sq / (2 * radius**2)) * intensity
    return img + blob


def _generate_sample(label: str, rng: np.random.Generator) -> np.ndarray:
    img = _base_layers(rng)

    if label == "CNV":
        cx, cy = rng.uniform(0.3, 0.7) * IMG_SIZE, rng.uniform(0.35, 0.55) * IMG_SIZE
        img = _add_blob(img, rng, cx, cy, radius=rng.uniform(6, 10), intensity=rng.uniform(0.35, 0.5))
    elif label == "DME":
        cx, cy = rng.uniform(0.3, 0.7) * IMG_SIZE, rng.uniform(0.4, 0.6) * IMG_SIZE
        img = _add_blob(img, rng, cx, cy, radius=rng.uniform(14, 20), intensity=-rng.uniform(0.3, 0.45))
    elif label == "DRUSEN":
        for _ in range(rng.integers(3, 7)):
            cx = rng.uniform(0.1, 0.9) * IMG_SIZE
            cy = rng.uniform(0.55, 0.7) * IMG_SIZE
            img = _add_blob(img, rng, cx, cy, radius=rng.uniform(2, 4), intensity=rng.uniform(0.25, 0.4))
    # NORMAL: base layers only.

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def generate_dataset(n_per_class: int = 250, seed: int = 42) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for label in CLASSES:
        for _ in range(n_per_class):
            images.append(_generate_sample(label, rng))
            labels.append(label)
    return images, labels
