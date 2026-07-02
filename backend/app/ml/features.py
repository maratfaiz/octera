"""Turns a raw OCT image into a fixed-size feature vector for the classifier."""

import numpy as np
from PIL import Image

IMAGE_SIZE = 48


def extract_features(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    return pixels.flatten()
