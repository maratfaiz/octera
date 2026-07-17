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

from app.ml.signal_utils import dark_mask_in_band, flatten_band, smooth, tissue_extent

HOG_SIZE = 64

# Kept in sync with app/services/segmentation.py's darkness heuristic.
_DARKNESS_OFFSET = 0.20


def _domain_features(gray: np.ndarray) -> np.ndarray:
    height = gray.shape[0]
    # Per-column tissue extent (round 24; previously a single row-profile
    # averaged across the whole image width, like segmentation.py's boundary
    # detection before round 23 -- see that module's docstring for why a
    # single global profile smears together depths that don't correspond to
    # the same anatomy on a curved/rotated real scan).
    top, bottom = tissue_extent(gray)

    band_top_frac = float(top.mean()) / height
    band_bottom_frac = float(bottom.mean()) / height
    band_thickness_frac = float((bottom - top).mean()) / height

    flat, valid = flatten_band(gray, top, bottom)
    if not valid.any():
        dark_area_frac = 0.0
        dark_zone_count_norm = 0.0
    else:
        dark_mask = dark_mask_in_band(flat, valid, _DARKNESS_OFFSET)
        dark_area_frac = float(dark_mask.sum()) / float(valid.sum())
        _, num_zones = ndimage.label(dark_mask)
        dark_zone_count_norm = min(num_zones / 50.0, 1.0)

    row_profile = smooth(gray.mean(axis=1), window=max(3, height // 40))
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
