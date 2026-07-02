import numpy as np

from app.ml.augmentation import augment, random_brightness_contrast, random_flip, random_shift, speckle_noise


def test_augment_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 1, size=(96, 96)) * 255).astype(np.uint8)

    out = augment(img, rng)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_individual_transforms_stay_in_valid_range():
    rng = np.random.default_rng(1)
    img = (rng.uniform(0, 1, size=(64, 64)) * 255).astype(np.uint8)

    for transform in (random_flip, random_brightness_contrast, speckle_noise, random_shift):
        out = transform(img, rng)
        assert out.shape == img.shape
        assert out.min() >= 0 and out.max() <= 255
