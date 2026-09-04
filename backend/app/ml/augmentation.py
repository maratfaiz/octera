"""Lightweight image augmentation shared by synthetic and real training data.

Kept dependency-free of heavy vision libraries (torchvision/albumentations);
`rotate` uses PIL, already a project dependency for image I/O elsewhere.
"""

import io

import numpy as np
from PIL import Image, ImageFilter


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


def gaussian_blur(img: np.ndarray, rng: np.random.Generator, radius_range: tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
    """Applies a randomized Gaussian blur, using PIL's built-in filter (no new
    dependency) -- not tried before (round 42 exploratory check, see README)
    despite matching the same "phone photo of a screen/printout" scenario
    rounds 19/23 (rotation), 30 (brightness), and 40 (shift) already built
    robustness for: a real photo taken by hand is plausibly out of focus or
    slightly motion-blurred, unlike the dataset's own direct digital exports.
    """
    radius = float(rng.uniform(radius_range[0], radius_range[1]))
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def _perspective_coeffs(dst: list[tuple[float, float]], src: list[tuple[float, float]]) -> np.ndarray:
    """Solves for the 8 coefficients PIL's Image.transform(..., Image.PERSPECTIVE, ...)
    needs to map each `dst` (output canvas) corner to the corresponding `src`
    (source image) corner -- the standard homography linear system, not
    provided by PIL itself.
    """
    matrix = []
    for (x, y), (sx, sy) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -sx * x, -sx * y])
        matrix.append([0, 0, 0, x, y, 1, -sy * x, -sy * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b)


def random_perspective(img: np.ndarray, rng: np.random.Generator, max_warp_frac: float = 0.06) -> np.ndarray:
    """Applies a mild random perspective (keystone) warp -- not tried before
    (round 44 exploratory check, see README) despite matching the same "phone
    photo of a screen/printout" scenario rounds 19/23 (rotation) and 40
    (shift) already built robustness for: a real handheld photo is rarely
    taken perfectly perpendicular to the screen/printout, which produces a
    trapezoidal keystone distortion that pure rotation doesn't model (rotation
    keeps parallel lines parallel; a perspective warp doesn't).

    Each of the 4 image corners is independently displaced inward by up to
    `max_warp_frac` of the image's own width/height, then that perturbed
    quadrilateral is mapped back onto the full output canvas -- the vacated
    regions fill with black (PIL's fillcolor), the same "lost content, more
    dark background" reasoning as random_shift rather than wrapping or
    stretching artifacts.
    """
    height, width = img.shape
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    src = [
        (x + rng.uniform(-max_warp_frac, max_warp_frac) * width, y + rng.uniform(-max_warp_frac, max_warp_frac) * height)
        for x, y in corners
    ]
    coeffs = _perspective_coeffs(corners, src)
    out = Image.fromarray(img).transform(
        (width, height), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=0
    )
    return np.array(out)


def random_vignette(img: np.ndarray, rng: np.random.Generator, strength_range: tuple[float, float] = (0.15, 0.45)) -> np.ndarray:
    """Applies a radial vignette (darkened corners/edges) -- not tried before
    (round 47 exploratory check, see README) despite matching the same "phone
    photo of a screen/printout" scenario that already motivated rotation
    (round 19/23), shift (round 40), and perspective (round 44): a real
    handheld phone camera's own lens/sensor falloff, plus uneven ambient
    lighting on the photographed surface, commonly darkens the corners and
    edges relative to the center. Unlike the pixel-level noise/blur
    augmentations this project already tried and found unhelpful (speckle
    noise round 38, JPEG artifacts round 39, Gaussian blur rounds 42/45),
    this is a geometric/lighting artifact -- the same category as the three
    augmentations that have actually shipped.

    Darkens each pixel by a factor that falls off quadratically with its
    normalized distance from the image center (1.0 at the center, down to
    `1 - strength` at the farthest corner), rather than a hard circular mask,
    for a smooth falloff matching real vignetting rather than a visible edge.
    """
    height, width = img.shape
    strength = float(rng.uniform(strength_range[0], strength_range[1]))
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist
    falloff = 1.0 - strength * normalized_dist**2
    out = img.astype(np.float64) * falloff
    return np.clip(out, 0, 255).astype(np.uint8)


def random_zoom(img: np.ndarray, rng: np.random.Generator, scale_range: tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """Applies a random zoom in/out, re-centered to the original canvas size
    -- not tried before (round 48 exploratory check, see README) despite
    matching the same "phone photo of a screen/printout" scenario that
    already motivated rotation (round 19/23), shift (round 40), and
    perspective (round 44): a real handheld photo is rarely taken from
    exactly the same distance every time, so the photographed content
    plausibly appears larger or smaller within the frame. This is a distinct
    geometric transform from the three that shipped -- rotation turns,
    shift translates, perspective skews, but none of them change scale.

    Zooming in (scale > 1) crops the resized image's center back down to the
    original size, discarding content that fell outside the frame; zooming
    out (scale < 1) pastes the smaller resized image onto a black canvas of
    the original size, centered -- the same "lost/vacated content is black,
    not wrapped or stretched" convention as random_shift and
    random_perspective.
    """
    height, width = img.shape
    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    resized = np.array(Image.fromarray(img).resize((new_width, new_height), resample=Image.BILINEAR))

    if scale >= 1.0:
        top = (new_height - height) // 2
        left = (new_width - width) // 2
        return resized[top : top + height, left : left + width]

    out = np.zeros((height, width), dtype=np.uint8)
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    out[top : top + new_height, left : left + new_width] = resized
    return out


def random_glare(
    img: np.ndarray,
    rng: np.random.Generator,
    strength_range: tuple[float, float] = (0.3, 0.7),
    radius_frac_range: tuple[float, float] = (0.15, 0.35),
) -> np.ndarray:
    """Adds a soft-edged specular highlight (glare) blob at a random position
    -- a new category, not tried before, despite matching the same "phone
    photo of a screen/printout" scenario that already motivated rotation
    (round 19/23), shift (round 40), perspective (round 44), and vignette
    (round 47): a glossy phone or monitor screen commonly reflects a window,
    room light, or camera flash back at the lens, producing a bright,
    roughly circular highlight somewhere in the frame. Unlike vignette (round
    47, rejected) -- which uniformly *darkens* every pixel by a fixed
    center-relative falloff -- glare is strictly additive (light only adds,
    never subtracts) and localized to a random position rather than always
    centered on the image, so it is a meaningfully different hypothesis
    despite the shared "lighting artifact" category.

    The highlight's center is drawn uniformly across the full canvas
    (including partly off-frame, since a real reflection isn't guaranteed to
    be centered in shot), and its brightness falls off as a Gaussian of the
    distance from that center, so it reads as a smooth highlight rather than
    a hard-edged disc.
    """
    height, width = img.shape
    strength = float(rng.uniform(strength_range[0], strength_range[1]))
    radius = float(rng.uniform(radius_frac_range[0], radius_frac_range[1])) * min(height, width)
    center_x = float(rng.uniform(0, width))
    center_y = float(rng.uniform(0, height))
    y, x = np.ogrid[:height, :width]
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    sigma = radius / 2.0
    glare = strength * 255.0 * np.exp(-dist_sq / (2 * sigma**2))
    out = img.astype(np.float64) + glare
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = random_flip(img, rng)
    img = random_shift(img, rng)
    img = random_brightness_contrast(img, rng)
    img = speckle_noise(img, rng)
    return img
