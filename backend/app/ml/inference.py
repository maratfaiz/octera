from pathlib import Path

from PIL import Image

from app.ml.features import extract_features
from app.ml.model import OCTClassifier

_cache: dict[str, OCTClassifier] = {}


def _get_model(checkpoint_path: str | Path) -> OCTClassifier:
    key = str(checkpoint_path)
    if key not in _cache:
        _cache[key] = OCTClassifier.load(checkpoint_path)
    return _cache[key]


def predict(image_path: str | Path, checkpoint_path: str | Path) -> dict[str, float]:
    model = _get_model(checkpoint_path)
    with Image.open(image_path) as image:
        features = extract_features(image).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    return {str(cls): prob for cls, prob in zip(model.clf.classes_, probabilities.tolist())}
