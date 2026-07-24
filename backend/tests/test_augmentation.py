import numpy as np

from app.ml.augmentation import (
    augment,
    gaussian_blur,
    jpeg_artifacts,
    random_brightness_contrast,
    random_flip,
    random_perspective,
    random_shift,
    random_vignette,
    random_zoom,
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
        random_perspective,
        random_vignette,
        random_zoom,
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


def test_random_perspective_actually_changes_pixels():
    """A meaningful warp should perturb at least some pixels -- if this ever
    passed with an unchanged image, the coefficient solve or the transform
    call would have silently become a no-op (e.g. all corner offsets zero).
    """
    rng = np.random.default_rng(4)
    img = (rng.uniform(0, 1, size=(96, 96)) * 255).astype(np.uint8)

    out = random_perspective(img, rng, max_warp_frac=0.1)

    assert not np.array_equal(out, img)


def test_random_perspective_pads_vacated_region_with_black():
    """The vacated region left by the warp must be padded black (matching
    random_shift's "lost content" reasoning), not wrapped, stretched, or left
    uninitialized.

    A fixed RNG returning the same value for all 8 calls (one per corner's x
    and y offset) degenerates the warp into a pure translation -- every
    source corner shifts by the identical (+max_warp_frac*w, +max_warp_frac*h)
    -- which makes the outcome exactly predictable: output pixel (x, y) reads
    input pixel (x + offset, y + offset), so only the far bottom-right strip
    (offset pixels wide/tall) samples outside the source image and must be
    black; everything else stays the solid source color.
    """
    height, width = 100, 100
    img = np.full((height, width), 255, dtype=np.uint8)  # solid bright frame

    class _FixedRng:
        def uniform(self, low, high):
            return high

    max_warp_frac = 0.2
    out = random_perspective(img, _FixedRng(), max_warp_frac=max_warp_frac)
    offset = int(max_warp_frac * width)

    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert (out[: height - offset, : width - offset] == 255).all()
    assert (out[height - offset :, :] == 0).all()
    assert (out[:, width - offset :] == 0).all()


def test_random_vignette_actually_changes_pixels():
    """A meaningful vignette strength should perturb at least some pixels --
    if this ever passed with an unchanged image, the falloff computation
    would have silently become a no-op (e.g. strength clamped to 0).
    """
    rng = np.random.default_rng(5)
    img = np.full((96, 96), 200, dtype=np.uint8)

    out = random_vignette(img, rng, strength_range=(0.4, 0.4))

    assert not np.array_equal(out, img)


def test_random_vignette_darkens_corners_more_than_center():
    """The whole point of a vignette is a smooth radial falloff -- corners
    (farthest from center) must darken strictly more than the center pixel,
    not uniformly or in reverse. A fixed strength makes the exact falloff at
    the center (1.0, unchanged) and at the farthest corner (1 - strength)
    precisely predictable.
    """
    height, width = 100, 100
    img = np.full((height, width), 200, dtype=np.uint8)

    class _FixedRng:
        def uniform(self, low, high):
            return low

    strength = 0.4
    out = random_vignette(img, _FixedRng(), strength_range=(strength, strength))

    center = out[height // 2, width // 2]
    corner = out[0, 0]

    assert corner < center
    assert center == 200  # falloff is 1.0 exactly at the center
    assert corner == round(200 * (1 - strength))


def test_random_zoom_actually_changes_pixels():
    """A meaningful zoom (in or out) should perturb a non-uniform image --
    if this ever passed with an unchanged image, the resize/crop/paste logic
    would have silently become a no-op (e.g. scale clamped to 1.0).
    """
    rng = np.random.default_rng(6)
    img = np.linspace(0, 255, 96 * 96, dtype=np.uint8).reshape(96, 96)

    out = random_zoom(img, rng, scale_range=(1.2, 1.2))

    assert not np.array_equal(out, img)


def test_random_zoom_out_pads_vacated_border_with_black():
    """Zooming out (scale < 1) must pad the vacated border with black
    (matching random_shift/random_perspective's "lost content" convention),
    not wrap, stretch, or leave it uninitialized.
    """
    height, width = 100, 100
    img = np.full((height, width), 255, dtype=np.uint8)

    class _FixedRng:
        def uniform(self, low, high):
            return low

    scale = 0.6
    out = random_zoom(img, _FixedRng(), scale_range=(scale, scale))
    new_height = round(height * scale)
    new_width = round(width * scale)
    top = (height - new_height) // 2
    left = (width - new_width) // 2

    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert (out[top : top + new_height, left : left + new_width] == 255).all()
    assert (out[:top, :] == 0).all()
    assert (out[top + new_height :, :] == 0).all()
    assert (out[:, :left] == 0).all()
    assert (out[:, left + new_width :] == 0).all()


def test_random_zoom_in_crops_to_original_size():
    """Zooming in (scale > 1) must crop the resized image's center back down
    to the original canvas size -- output shape must never change regardless
    of zoom direction.
    """
    height, width = 100, 100
    img = np.full((height, width), 255, dtype=np.uint8)

    class _FixedRng:
        def uniform(self, low, high):
            return high

    out = random_zoom(img, _FixedRng(), scale_range=(1.3, 1.3))

    assert out.dtype == np.uint8
    assert out.shape == img.shape


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
