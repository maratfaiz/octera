"""Lightweight image augmentation shared by synthetic and real training data.

Kept dependency-free of heavy vision libraries (torchvision/albumentations);
`rotate` uses PIL, already a project dependency for image I/O elsewhere.
"""

import io

import numpy as np
from PIL import Image


def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    return np.array(Image.fromarray(img).rotate(degrees, resample=Image.BILINEAR, fillcolor=0))


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


def jpeg_artifacts(img: np.ndarray, rng: np.random.Generator, quality_range: tuple[int, int] = (30, 70)) -> np.ndarray:
    """Re-encodes through a low-quality JPEG round-trip, introducing real
    blocking/ringing compression artifacts -- not tried before (round 39
    exploratory check, see README) despite matching the exact "phone photo of
    a screen/printout" scenario rounds 19/23 (rotation) and 30 (brightness)
    already built robustness for: a photo of a physical printout or another
    screen is very plausibly re-compressed harder than the training dataset's
    own direct digital exports.
    """
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


def random_shift(img: np.ndarray, rng: np.random.Generator, max_frac: float = 0.08) -> np.ndarray:
    """Translates the image, padding the vacated edge with black (0) rather
    than wrapping content around from the opposite edge.

    An earlier version used np.roll (circular wrap), which doesn't model any
    real acquisition variation: a real off-center photo/crop shows more dark
    background on one side and loses content on the other, it doesn't teleport
    pixels from the far edge to the near one. Round-tripping through a wrap
    would hand the classifier a wrap seam artifact that never occurs in a real
    upload -- caught in code review, fixed to pad instead (round 40 then
    tested this fixed version as a real-data augmentation candidate).
    """
    height, width = img.shape
    dy = int(rng.uniform(-max_frac, max_frac) * height)
    dx = int(rng.uniform(-max_frac, max_frac) * width)

    out = np.zeros_like(img)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = random_flip(img, rng)
    img = random_shift(img, rng)
    img = random_brightness_contrast(img, rng)
    img = speckle_noise(img, rng)
    return img
