from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.augmentation import random_brightness_contrast, rotate
from app.ml.features import extract_features
from app.ml.model import OCTClassifier

_cache: dict[str, OCTClassifier] = {}

# Test-time augmentation: average predicted probabilities over the uploaded
# scan plus 4 augmented views (same amplitude as the training-time recipe in
# app/ml/train.py). Verified on 5 seeds at the clinical screening threshold
# (see README round 35) to never lower DME recall versus a single pass, at
# the cost of ~5x feature extraction per request. Fixed seed so the same
# upload always yields the same prediction.
_TTA_SEED = 0


def _get_model(checkpoint_path: str | Path) -> OCTClassifier:
    key = str(checkpoint_path)
    if key not in _cache:
        _cache[key] = OCTClassifier.load(checkpoint_path)
    return _cache[key]


def _tta_views(gray: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(_TTA_SEED)
    return [gray, rotate(gray, -5), rotate(gray, 5), np.fliplr(gray), random_brightness_contrast(gray, rng)]


def predict(image_path: str | Path, checkpoint_path: str | Path) -> dict[str, float]:
    model = _get_model(checkpoint_path)
    with Image.open(image_path) as image:
        gray = np.array(image.convert("L"))
    features = np.stack([extract_features(Image.fromarray(view)) for view in _tta_views(gray)])
    probabilities = model.predict_proba(features).mean(axis=0)
    return {str(cls): prob for cls, prob in zip(model.clf.classes_, probabilities.tolist())}
