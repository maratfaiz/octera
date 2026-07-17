"""Unit tests for app/ml/signal_utils.py -- the shared primitives used by both
segmentation.py (layer/pathology visualization) and features.py (classifier
domain features). Added after a code review found smooth() could silently
change array length and tissue_extent() could silently treat a background/
vignette column as fully valid tissue -- neither was covered by any test.
"""

import numpy as np

from app.ml.signal_utils import smooth, tissue_extent


def test_smooth_preserves_length_even_when_window_exceeds_profile_length():
    for n in [0, 1, 2, 3, 5, 50]:
        profile = np.arange(n, dtype=float)
        out = smooth(profile, window=10)
        assert len(out) == n


def test_tissue_extent_marks_a_background_column_as_invalid():
    height, width = 100, 10
    gray = np.full((height, width), 0.6, dtype=np.float32)
    gray[:, 0] = 0.0  # a fully black column -- background/vignette, not tissue

    top, bottom = tissue_extent(gray)

    assert top[0] == 0.0 and bottom[0] == 0.0
    # A normal bright column should still be recognized as (mostly) valid tissue.
    assert bottom[5] > top[5]


def test_tissue_extent_finds_a_normal_bright_band():
    height, width = 100, 30
    gray = np.zeros((height, width), dtype=np.float32)
    gray[30:60, :] = 0.8

    top, bottom = tissue_extent(gray)

    # Check interior columns only -- the width-wise smoothing pass tapers near
    # the array edges (an inherent, expected property of convolve, not a bug).
    assert np.all(top[5:-5] < 35)
    assert np.all(bottom[5:-5] > 55)
