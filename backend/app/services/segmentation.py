"""Retinal layer segmentation module.

Approximates retinal layer boundaries via 1D intensity-profile peak
detection along the B-scan's depth axis (classical "A-scan" analysis,
the technique OCT software used before deep-learning segmentation). It runs
on the real pixel data of the uploaded image -- unlike a per-pixel model,
which needs boundary-labeled ground truth we don't have. Swap this module
for a trained U-Net / nnU-Net once labeled segmentation data is available;
keep the same return contract (heatmap path + per-layer thickness in
microns).

Boundaries are tracked per-column (one candidate position per A-scan),
not from a single profile averaged across the whole image width. A real
retina is rarely flat across a B-scan -- it dips at the fovea and follows
the eye's natural curvature -- so a single global profile smears together
depths that don't correspond to the same anatomy, producing meaningless
boundaries on anything but a perfectly flat test image. Each boundary is
tracked outward from a "seed" column with the clearest banded signal,
snapping to the nearest locally-detected peak so the curve follows the
actual retinal shape instead of a straight line.

The pixel-to-micron scale factor is a placeholder -- no scan calibration
metadata is available from a plain JPEG/PNG upload, so treat absolute
thickness values as illustrative, not measured.

Also flags candidate "pathology zones": patches inside the retinal band
that are notably darker (hyporeflective) than their local surroundings.
Fluid, cysts and serous detachment all appear optically empty (dark) on
OCT, so local-contrast thresholding is a real, classical way to surface
candidate regions -- but it is a generic darkness detector, not a trained
classifier that tells cyst apart from fluid apart from a shadow artifact.
Present it to the user as "algorithm-flagged zones for review", never as a
diagnosis. Detection runs on a "flattened" version of the retinal band
(each column shifted so its own top boundary sits at row 0) so the local
brightness baseline isn't distorted by the band's curvature either.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

LAYERS = [
    "nfl_gcl",  # nerve fiber layer / ganglion cell layer
    "ipl_inl",  # inner plexiform / inner nuclear layer
    "opl_onl",  # outer plexiform / outer nuclear layer
    "photoreceptor",
    "rpe",
]

# Human-readable Russian labels for LAYERS, for anywhere a clinician-facing view
# (report text, UI) needs to show layer thickness -- the bare codes above are an
# internal shorthand, not something to put in front of a doctor.
LAYER_LABELS_RU = {
    "nfl_gcl": "Слой нервных волокон / ганглиозных клеток (NFL/GCL)",
    "ipl_inl": "Внутренний плексиформный / внутренний ядерный слой (IPL/INL)",
    "opl_onl": "Наружный плексиформный / наружный ядерный слой (OPL/ONL)",
    "photoreceptor": "Слой фоторецепторов",
    "rpe": "Пигментный эпителий сетчатки (RPE)",
}

# One overlay color per boundary line (needs len(LAYERS) + 1 of these).
BOUNDARY_COLORS = [
    (255, 220, 60),
    (255, 140, 60),
    (255, 90, 90),
    (255, 90, 200),
    (150, 120, 255),
    (90, 180, 255),
]

ASSUMED_UM_PER_PIXEL = 2.0

# Heuristic pathology-zone detection parameters.
DARKNESS_OFFSET = 0.20  # a zone must be this much darker than the local layer baseline
MIN_ZONE_AREA_DIVISOR = 3000  # min blob area, as a fraction of the retinal band's pixel count


@dataclass
class SegmentationOutput:
    map_path: str
    layer_thickness_um: dict[str, float]
    pathology_map_path: str
    pathology_zone_count: int


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude -= magnitude.min()
    peak = magnitude.max()
    if peak > 0:
        magnitude /= peak
    return magnitude


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window)
    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode="same")


def _find_peaks(profile: np.ndarray, min_distance: int) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
            elif profile[i] > profile[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _column_peaks(gray: np.ndarray, col: int, window: int, min_distance: int) -> list[int]:
    profile = _smooth(gray[:, col], window)
    return _find_peaks(profile, min_distance)


def _seed_column(gray: np.ndarray, window: int, min_distance: int, needed: int) -> int | None:
    """Picks the column with the clearest banded signal to start tracking from.

    Sampled rather than exhaustive for speed on wide images -- a coarse scan is
    enough to find a reasonably representative starting point.
    """
    height, width = gray.shape
    step = max(1, width // 60)
    best_col, best_score = None, -1.0
    for col in range(0, width, step):
        profile = _smooth(gray[:, col], window)
        peaks = _find_peaks(profile, min_distance)
        if len(peaks) < needed:
            continue
        score = float(sum(profile[p] for p in peaks))
        if score > best_score:
            best_score, best_col = score, col
    return best_col


def _track_boundaries(gray: np.ndarray) -> np.ndarray:
    """Returns a (len(LAYERS) + 1, width) array: one tracked row-position per
    boundary per column, following the retina's actual curvature.

    Falls back to flat, evenly-spaced boundaries (broadcast across every
    column) when the image has no distinguishable banded structure at all
    (e.g. a uniform test image) -- callers always get a full set.
    """
    height, width = gray.shape
    needed = len(LAYERS) + 1
    window = max(3, height // 40)
    min_distance = max(2, height // 20)
    snap_tolerance = max(4, height // 8)

    seed_col = _seed_column(gray, window, min_distance, needed)
    if seed_col is None:
        flat = np.linspace(0, height - 1, needed)
        return np.tile(flat[:, None], (1, width))

    seed_peaks = _column_peaks(gray, seed_col, window, min_distance)
    seed_profile = _smooth(gray[:, seed_col], window)
    top_by_prominence = np.argsort([seed_profile[p] for p in seed_peaks])[-needed:]
    seed_positions = sorted(np.array(seed_peaks)[top_by_prominence].tolist())

    boundaries = np.zeros((needed, width), dtype=float)
    boundaries[:, seed_col] = seed_positions

    def _advance(order: range, prev_positions: list[float]) -> None:
        prev = list(prev_positions)
        for col in order:
            peaks = _column_peaks(gray, col, window, min_distance)
            new_positions = []
            for p in prev:
                if peaks:
                    nearest = min(peaks, key=lambda x: abs(x - p))
                    new_positions.append(nearest if abs(nearest - p) <= snap_tolerance else p)
                else:
                    new_positions.append(p)
            # Boundaries can't cross -- enforce non-decreasing order defensively
            # against a noisy column snapping a boundary past its neighbor.
            for i in range(1, len(new_positions)):
                new_positions[i] = max(new_positions[i], new_positions[i - 1] + 1)
            boundaries[:, col] = new_positions
            prev = new_positions

    _advance(range(seed_col + 1, width), seed_positions)
    _advance(range(seed_col - 1, -1, -1), seed_positions)

    smooth_window = max(3, width // 30)
    for i in range(needed):
        boundaries[i] = _smooth(boundaries[i], smooth_window)
    for i in range(1, needed):
        boundaries[i] = np.maximum(boundaries[i], boundaries[i - 1] + 1)
    return np.clip(boundaries, 0, height - 1)


def _tissue_extent(gray: np.ndarray, threshold_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Per-column top/bottom of the actual tissue signal, as opposed to background.

    Used to bound pathology-zone search independently of the fine-grained,
    boundary-by-boundary layer tracking above -- a large dark pathological gap
    (e.g. a cystoid cavity spanning much of the retina's visible height) can
    make _track_boundaries lose track of individual layer borders inside it,
    but the overall top/bottom extent of "there is retinal tissue here" is a
    much simpler, more robust question this answers independently: a column's
    own brightest point is still a reliable anchor even when a dark gap sits
    between it and the next bright structure.
    """
    height, width = gray.shape
    window = max(3, height // 40)
    min_run = max(2, height // 50)
    top = np.zeros(width)
    bottom = np.full(width, float(height))
    for col in range(width):
        profile = _smooth(gray[:, col], window)
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
    return _smooth(top, smooth_window), _smooth(bottom, smooth_window)


def _flatten_band(gray: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shifts each column so row 0 = its own top boundary ("retina flattening"),
    removing curvature so a simple row-wise baseline is meaningful again.
    Returns (flattened values, validity mask) both shaped (max_band_height, width).
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


def _detect_pathology_zones(gray: np.ndarray, boundaries) -> tuple[np.ndarray, int]:
    height, width = gray.shape
    top = np.broadcast_to(np.asarray(boundaries[0], dtype=float), (width,))
    bottom = np.broadcast_to(np.asarray(boundaries[-1], dtype=float), (width,))
    mask = np.zeros_like(gray, dtype=bool)

    flat, valid = _flatten_band(gray, top, bottom)
    if not valid.any():
        return mask, 0

    row_count = valid.sum(axis=1)
    row_sum = np.where(valid, flat, 0.0).sum(axis=1)
    row_mean = np.divide(row_sum, row_count, out=np.zeros_like(row_sum), where=row_count > 0)
    row_baseline = _smooth(row_mean, window=max(3, flat.shape[0] // 20))[:, None]

    dark_mask = valid & (flat < (row_baseline - DARKNESS_OFFSET))
    labeled, num_features = ndimage.label(dark_mask)
    min_area = max(15, int(valid.sum()) // MIN_ZONE_AREA_DIVISOR)
    max_width = 0.4 * width  # real fluid/cyst pockets are localized blobs, not
    # bands spanning most of the scan's width (which is usually a vitreous/layer
    # transition artifact rather than a discrete pathological zone).

    top_i = np.clip(np.round(top).astype(int), 0, height)
    zone_count = 0
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() < min_area:
            continue
        cols = np.where(component.any(axis=0))[0]
        width_span = cols.max() - cols.min() + 1
        if width_span > max_width:
            continue
        # Map the flattened component back to original (row, col) coordinates.
        rows_flat, cols_flat = np.where(component)
        orig_rows = rows_flat + top_i[cols_flat]
        valid_rows = orig_rows < height
        mask[orig_rows[valid_rows], cols_flat[valid_rows]] = True
        zone_count += 1

    return mask, zone_count


def _draw_pathology_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = (np.clip(gray, 0, 1) * 255).astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    overlay_color = np.array([255, 80, 60], dtype=np.float32)
    alpha = 0.45
    rgb[mask] = rgb[mask] * (1 - alpha) + overlay_color * alpha
    return rgb.astype(np.uint8)


def _draw_layer_overlay(gray: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Tints each tracked layer band with its own translucent color directly on
    the scan, plus a color-coded legend strip naming each layer -- a labeled
    readout closer to what a clinician expects from OCT device software,
    instead of a bare edge map. Bands (not boundary lines) are colored so the
    legend maps 1:1 to what's shown -- len(LAYERS) named regions, not
    len(LAYERS) + 1 boundary lines that don't correspond to a single layer each.
    """
    height, width = gray.shape
    base = (np.clip(gray, 0, 1) * 255).astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)

    row_idx = np.arange(height)[:, None]
    alpha = 0.35
    for i, layer in enumerate(LAYERS):
        color = np.array(BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)], dtype=np.float32)
        top = boundaries[i][None, :]
        bottom = boundaries[i + 1][None, :]
        band_mask = (row_idx >= top) & (row_idx < bottom)
        rgb[band_mask] = rgb[band_mask] * (1 - alpha) + color * alpha

    legend_h = 26
    canvas = np.zeros((height + legend_h, width, 3), dtype=np.uint8)
    canvas[:height] = rgb.astype(np.uint8)
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    x = 6
    for i, layer in enumerate(LAYERS):
        color = BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
        draw.rectangle([x, height + 8, x + 10, height + 18], fill=color)
        label = layer.upper()
        draw.text((x + 14, height + 6), label, fill=(255, 255, 255))
        x += 14 + 8 * len(label) + 16

    return np.array(img)


def segment_layers(image_path: str) -> SegmentationOutput:
    with Image.open(image_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    boundaries = _track_boundaries(gray)

    overlay = _draw_layer_overlay(gray, boundaries)
    map_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_segmentation.png"))
    Image.fromarray(overlay, mode="RGB").save(map_path)

    thickness = {
        layer: round(float(np.mean(bottom - top)) * ASSUMED_UM_PER_PIXEL, 1)
        for layer, top, bottom in zip(LAYERS, boundaries, boundaries[1:])
    }

    tissue_top, tissue_bottom = _tissue_extent(gray)
    pathology_mask, zone_count = _detect_pathology_zones(gray, [tissue_top, tissue_bottom])
    pathology_overlay = _draw_pathology_overlay(gray, pathology_mask)
    pathology_map_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_pathology.png"))
    Image.fromarray(pathology_overlay, mode="RGB").save(pathology_map_path)

    return SegmentationOutput(
        map_path=map_path,
        layer_thickness_um=thickness,
        pathology_map_path=pathology_map_path,
        pathology_zone_count=zone_count,
    )
