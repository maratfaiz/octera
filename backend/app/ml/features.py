"""Turns a raw OCT image into a fixed-size feature vector for the classifier.

Two feature groups are concatenated:

1. HOG (Histogram of Oriented Gradients) over the image resized to 64x64.
   HOG captures the layered/edge texture of a retinal B-scan far better than
   raw pixel intensities -- on this dataset it roughly halved the DME false-
   alarm rate versus raw pixels (see the model card / commit history). It is a
   classic, CPU-cheap strong feature for medical-image texture.
2. Domain features derived from the same signal analysis the segmentation
   module uses. Diabetic macular edema is, by definition, retinal thickening
   plus fluid pockets, so features that measure the retinal band's extent and
   the amount of locally dark (hyporeflective) tissue give the classifier the
   clinically relevant signal directly. On their own they don't beat pixels,
   but combined with HOG they add a small, measured gain.

Both groups are computed here (no file I/O, no side effects) so training and
inference always see the exact same representation.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.feature import hog

HOG_SIZE = 64

# Kept in sync with app/services/segmentation.py's darkness heuristic.
_DARKNESS_OFFSET = 0.20


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window)
    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode="same")


def _retinal_band(gray: np.ndarray) -> tuple[int, int]:
    """Rough vertical extent of the retinal tissue: rows whose mean brightness
    is above a fraction of the peak row brightness. Thickening (a DME sign)
    widens this band.
    """
    row_profile = _smooth(gray.mean(axis=1), window=max(3, gray.shape[0] // 40))
    threshold = 0.35 * row_profile.max()
    bright_rows = np.where(row_profile > threshold)[0]
    if bright_rows.size == 0:
        return 0, gray.shape[0]
    return int(bright_rows.min()), int(bright_rows.max())


def _domain_features(gray: np.ndarray) -> np.ndarray:
    height = gray.shape[0]
    top, bottom = _retinal_band(gray)
    band = gray[top:bottom, :]

    band_top_frac = top / height
    band_bottom_frac = bottom / height
    band_thickness_frac = (bottom - top) / height

    if band.size == 0:
        dark_area_frac = 0.0
        dark_zone_count_norm = 0.0
    else:
        row_baseline = _smooth(band.mean(axis=1), window=max(3, band.shape[0] // 20))[:, None]
        dark_mask = band < (row_baseline - _DARKNESS_OFFSET)
        dark_area_frac = float(dark_mask.mean())
        _, num_zones = ndimage.label(dark_mask)
        dark_zone_count_norm = min(num_zones / 50.0, 1.0)

    row_profile = _smooth(gray.mean(axis=1), window=max(3, height // 40))
    profile_std = float(row_profile.std())
    profile_max = float(row_profile.max())

    return np.array(
        [
            band_top_frac,
            band_bottom_frac,
            band_thickness_frac,
            dark_area_frac,
            dark_zone_count_norm,
            profile_std,
            profile_max,
            float(gray.mean()),
            float(gray.std()),
            float((gray > 0.6).mean()),  # bright-pixel fraction (highly reflective layers)
        ],
        dtype=np.float32,
    )


def _hog_features(gray: np.ndarray) -> np.ndarray:
    small = np.asarray(
        Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize((HOG_SIZE, HOG_SIZE)),
        dtype=np.float32,
    ) / 255.0
    return hog(
        small,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).astype(np.float32)


def extract_features(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return np.concatenate([_hog_features(gray), _domain_features(gray)])
