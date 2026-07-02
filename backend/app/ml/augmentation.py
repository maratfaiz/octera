"""Lightweight image augmentation shared by synthetic and real training data.

Kept dependency-free (numpy only) so it works without torchvision/albumentations.
"""

import numpy as np


def random_flip(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        return np.fliplr(img)
    return img


def random_brightness_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    brightness = rng.uniform(-0.15, 0.15)
    contrast = rng.uniform(0.8, 1.2)
    out = (img.astype(np.float32) / 255.0 - 0.5) * contrast + 0.5 + brightness
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def speckle_noise(img: np.ndarray, rng: np.random.Generator, amount: float = 0.15) -> np.ndarray:
    """Multiplicative speckle noise, closer to real OCT scan noise than additive gaussian."""
    noise = rng.normal(1.0, amount, size=img.shape)
    out = img.astype(np.float32) * noise
    return np.clip(out, 0, 255).astype(np.uint8)


def random_shift(img: np.ndarray, rng: np.random.Generator, max_frac: float = 0.08) -> np.ndarray:
    height, width = img.shape
    dy = int(rng.uniform(-max_frac, max_frac) * height)
    dx = int(rng.uniform(-max_frac, max_frac) * width)
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = random_flip(img, rng)
    img = random_shift(img, rng)
    img = random_brightness_contrast(img, rng)
    img = speckle_noise(img, rng)
    return img
