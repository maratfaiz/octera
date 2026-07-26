"""Shared 1D/2D intensity-profile signal-processing primitives for OCT B-scans.

Used by both app.ml.features (classifier domain features) and
app.services.segmentation (layer/pathology visualization) -- kept in one
place instead of two independently-maintained copies. They used to be
hand-duplicated (features.py's simpler, single-global-profile retinal band
estimate vs. segmentation.py's per-column tissue extent), kept in sync only
by a comment -- exactly the kind of drift risk this module removes.

Caution when tuning: the two call sites have different needs -- the
classifier wants a stable, reproducible numeric summary (any change here
changes what the shipped checkpoint was trained on and requires a retrain,
see app/ml/train.py), while the visualization side wants to look right on a
given image. A change made purely to improve how the segmentation overlay
looks (e.g. adjusting threshold_frac or the smoothing windows below) will
silently also change the classifier's features -- retrain and re-validate
(see README's ML rounds) rather than assuming a visualization tweak is free.
"""

import numpy as np


def smooth(profile: np.ndarray, window: int) -> np.ndarray:
    if len(profile) == 0:
        return profile
    # np.convolve(..., mode="same") returns an array of length max(len(profile),
    # len(kernel)), NOT len(profile), whenever the kernel is longer than the
    # profile -- silently breaking every caller's index alignment (and, for
    # 2D callers that broadcast the result against an array shaped by
    # len(profile), raising a shape-mismatch error outright). Clamping the
    # window to at most len(profile) guarantees the output always matches the
    # input length.
    window = min(max(3, window), len(profile))
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
    has_tissue = np.zeros(width, dtype=bool)
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
            has_tissue[col] = True
    smooth_window = max(3, width // 30)
    top = smooth(top, smooth_window)
    bottom = smooth(bottom, smooth_window)
    # A column with no sustained bright run (background/vignetting at the image
    # edge, not real tissue) never overwrote its (0, height) placeholder above --
    # left as-is, flatten_band would read that as "the entire column is valid
    # tissue", pulling background noise into the row-wise darkness baseline and
    # producing false-positive pathology-zone flags right at the image edges
    # (observed empirically on real uploaded photos with vignetted borders).
    # Zero these out to an empty (invalid) band instead.
    top = np.where(has_tissue, top, 0.0)
    bottom = np.where(has_tissue, bottom, 0.0)
    return top, bottom


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


def _row_baseline(flat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row-wise brightness baseline at each depth of a flattened tissue band
    (see flatten_band) -- the local "expected brightness at this depth"
    reference both dark_mask_in_band and bright_mask_in_band compare each
    pixel against, factored out so the two share one implementation instead
    of (as an earlier version had it) two independently-maintained copies of
    the same row-averaging/smoothing logic that differed only in the final
    comparison direction.
    """
    row_count = valid.sum(axis=1)
    row_sum = np.where(valid, flat, 0.0).sum(axis=1)
    row_mean = np.divide(row_sum, row_count, out=np.zeros_like(row_sum), where=row_count > 0)
    return smooth(row_mean, window=max(3, flat.shape[0] // 20))[:, None]


def dark_mask_in_band(flat: np.ndarray, valid: np.ndarray, darkness_offset: float) -> np.ndarray:
    """Boolean mask of pixels in a flattened tissue band (see flatten_band) that
    are notably darker (hyporeflective) than the row-wise brightness baseline
    at that depth. Shared by segmentation.py's pathology-zone detector and
    features.py's classifier domain features -- both need the same "is this
    pixel darker than its local surroundings" answer and used to carry two
    independently-maintained copies of this exact formula.
    """
    if not valid.any():
        return np.zeros_like(valid, dtype=bool)
    return valid & (flat < (_row_baseline(flat, valid) - darkness_offset))


def bright_mask_in_band(flat: np.ndarray, valid: np.ndarray, brightness_offset: float) -> np.ndarray:
    """Boolean mask of pixels in a flattened tissue band that are notably
    brighter (hyperreflective) than the row-wise baseline at that depth --
    the mirror image of dark_mask_in_band, for pathology categories that
    present as bright material rather than optically-empty fluid (e.g.
    subretinal hyperreflective material, see segmentation.py).
    """
    if not valid.any():
        return np.zeros_like(valid, dtype=bool)
    return valid & (flat > (_row_baseline(flat, valid) + brightness_offset))
