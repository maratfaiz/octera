"""Shared 1D/2D intensity-profile signal-processing primitives for OCT B-scans.

Used by both app.ml.features (classifier domain features) and
app.services.segmentation (layer/pathology visualization) -- kept in one
place instead of two independently-maintained copies. They used to be
hand-duplicated (features.py's simpler, single-global-profile retinal band
estimate vs. segmentation.py's per-column tissue extent), kept in sync only
by a comment -- exactly the kind of drift risk this module removes.
"""

import numpy as np


def smooth(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window)
    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode="same")


def tissue_extent(gray: np.ndarray, threshold_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Per-column top/bottom of the actual tissue signal, as opposed to background.

    A column's own brightest point is a reliable anchor even when a dark gap
    (e.g. a cystoid cavity) sits between it and the next bright structure, so
    this stays robust in cases where fine-grained, boundary-by-boundary layer
    tracking can lose its way.
    """
    height, width = gray.shape
    window = max(3, height // 40)
    min_run = max(2, height // 50)
    top = np.zeros(width)
    bottom = np.full(width, float(height))
    for col in range(width):
        profile = smooth(gray[:, col], window)
        peak = profile.max()
        if peak <= 0:
            continue
        above = profile >= (peak * threshold_frac)
        # Require `min_run` consecutive rows above threshold, not just one -- a
        # single noisy/bright pixel right at the very top edge would otherwise
        # register as "tissue starts here", pulling background into the band.
        run_lengths = np.convolve(above.astype(int), np.ones(min_run, dtype=int), mode="valid")
        sustained = np.where(run_lengths >= min_run)[0]
        if sustained.size:
            top[col] = float(sustained[0])
            bottom[col] = float(sustained[-1] + min_run)
    smooth_window = max(3, width // 30)
    return smooth(top, smooth_window), smooth(bottom, smooth_window)


def flatten_band(gray: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shifts each column so row 0 = its own top boundary ("retina flattening"),
    removing curvature so a simple row-wise baseline/threshold is meaningful
    again. Returns (flattened values, validity mask), both shaped
    (max_band_height, width).
    """
    height, width = gray.shape
    top_i = np.clip(np.round(top).astype(int), 0, height)
    bottom_i = np.clip(np.round(bottom).astype(int), 0, height)
    band_heights = np.maximum(bottom_i - top_i, 0)
    max_h = int(band_heights.max()) if band_heights.size else 0
    max_h = max(max_h, 1)

    flat = np.zeros((max_h, width), dtype=gray.dtype)
    valid = np.zeros((max_h, width), dtype=bool)
    for col in range(width):
        t, b = top_i[col], bottom_i[col]
        if b <= t:
            continue
        seg = gray[t:b, col]
        flat[: len(seg), col] = seg
        valid[: len(seg), col] = True
    return flat, valid
