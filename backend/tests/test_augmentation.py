import numpy as np

from app.ml.augmentation import (
    augment,
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

    for transform in (random_flip, random_brightness_contrast, speckle_noise, random_shift, jpeg_artifacts):
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
