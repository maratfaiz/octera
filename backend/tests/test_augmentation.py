import numpy as np

from app.ml.augmentation import (
    augment,
    gaussian_blur,
    jpeg_artifacts,
    random_brightness_contrast,
    random_flip,
    random_shift,
    speckle_noise,
)


def test_augment_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 1, size=(96, 96)) * 255).astype(np.uint8)

    out = augment(img, rng)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_individual_transforms_stay_in_valid_range():
    rng = np.random.default_rng(1)
    img = (rng.uniform(0, 1, size=(64, 64)) * 255).astype(np.uint8)

    for transform in (
        random_flip,
        random_brightness_contrast,
        speckle_noise,
        random_shift,
        jpeg_artifacts,
        gaussian_blur,
    ):
        out = transform(img, rng)
        assert out.shape == img.shape
        assert out.min() >= 0 and out.max() <= 255


def test_jpeg_artifacts_actually_changes_pixels():
    """A meaningfully lossy round-trip should perturb at least some pixels --
    if this ever passed with an unchanged image, the JPEG encode/decode step
    would have silently become a no-op (e.g. quality clamped to 100 or the
    buffer never actually re-read).
    """
    rng = np.random.default_rng(2)
    img = (rng.uniform(0, 1, size=(96, 96)) * 255).astype(np.uint8)

    out = jpeg_artifacts(img, rng, quality_range=(10, 10))

    assert not np.array_equal(out, img)


def test_gaussian_blur_actually_changes_pixels():
    """A meaningful blur radius should perturb at least some pixels -- if this
    ever passed with an unchanged image, the filter would have silently
    become a no-op (e.g. radius clamped to 0 or the filter never applied).
    """
    rng = np.random.default_rng(3)
    img = (rng.uniform(0, 1, size=(96, 96)) * 255).astype(np.uint8)

    out = gaussian_blur(img, rng, radius_range=(2.0, 2.0))

    assert not np.array_equal(out, img)


def test_random_shift_pads_instead_of_wrapping():
    """Regression test: an earlier version used np.roll (circular wrap),
    which teleports content from the far edge to the near one -- an artifact
    that never occurs in a real off-center photo/crop. A real shift pads the
    vacated edge with black and genuinely loses the content that shifted out
    of frame, rather than smearing it around to the opposite side.
    """
    height, width = 40, 40
    img = np.full((height, width), 255, dtype=np.uint8)  # solid bright frame

    class _FixedRng:
        def uniform(self, low, high):
            return high  # always shift by the maximum allowed fraction, deterministically

    out = random_shift(img, _FixedRng(), max_frac=0.5)

    # Shifted down-and-right by half the image: the vacated top-left region
    # must be black padding, not wrapped-around bright content from the
    # opposite (bottom-right) edge.
    assert (out[:20, :20] == 0).all()
    assert (out[20:, 20:] == 255).all()
