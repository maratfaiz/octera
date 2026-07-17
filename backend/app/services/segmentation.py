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

from app.ml.signal_utils import dark_mask_in_band, flatten_band, smooth, tissue_extent

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

# Compact standard abbreviations for the on-image legend (_draw_layer_overlay) --
# the full LAYER_LABELS_RU phrases are for text (report, thickness table), not a
# space-constrained legend; these match the bare-abbreviation convention real
# OCT device software uses directly on the scan (e.g. "NFL/GCL", not the raw
# Python identifier "nfl_gcl").
LAYER_SHORT_LABELS = {
    "nfl_gcl": "NFL/GCL",
    "ipl_inl": "IPL/INL",
    "opl_onl": "OPL/ONL",
    "photoreceptor": "PHOTORECEPTOR",
    "rpe": "RPE",
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
    profile = smooth(gray[:, col], window)
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
        profile = smooth(gray[:, col], window)
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
    seed_profile = smooth(gray[:, seed_col], window)
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
        boundaries[i] = smooth(boundaries[i], smooth_window)
    for i in range(1, needed):
        boundaries[i] = np.maximum(boundaries[i], boundaries[i - 1] + 1)
    return np.clip(boundaries, 0, height - 1)


def _detect_pathology_zones(gray: np.ndarray, boundaries) -> tuple[np.ndarray, int]:
    height, width = gray.shape
    top = np.broadcast_to(np.asarray(boundaries[0], dtype=float), (width,))
    bottom = np.broadcast_to(np.asarray(boundaries[-1], dtype=float), (width,))
    mask = np.zeros_like(gray, dtype=bool)

    flat, valid = flatten_band(gray, top, bottom)
    if not valid.any():
        return mask, 0

    dark_mask = dark_mask_in_band(flat, valid, DARKNESS_OFFSET)
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
    for i in range(len(LAYERS)):
        color = np.array(BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)], dtype=np.float32)
        top = boundaries[i][None, :]
        bottom = boundaries[i + 1][None, :]
        band_mask = (row_idx >= top) & (row_idx < bottom)
        rgb[band_mask] = rgb[band_mask] * (1 - alpha) + color * alpha

    # Lay out the legend first (on a throwaway draw context, just to measure
    # real text widths via textlength) and wrap entries onto as many rows as
    # the image's actual width needs -- a narrow upload (a tight crop, or this
    # module's own 200-300px test fixtures) would otherwise have later labels
    # (up to and including "RPE") drawn entirely off-canvas and never visible.
    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    swatch_w, gap, margin = 14, 16, 6
    entries = [(LAYER_SHORT_LABELS[layer], BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]) for i, layer in enumerate(LAYERS)]
    rows: list[list[tuple[str, tuple[int, int, int], int]]] = [[]]
    x = margin
    for label, color in entries:
        entry_w = swatch_w + int(measurer.textlength(label)) + gap
        if x + entry_w > width and rows[-1]:
            rows.append([])
            x = margin
        rows[-1].append((label, color, x))
        x += entry_w
    row_h = 20
    legend_h = margin + len(rows) * row_h

    canvas = np.zeros((height + legend_h, width, 3), dtype=np.uint8)
    canvas[:height] = rgb.astype(np.uint8)
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    for row_i, row in enumerate(rows):
        y = height + margin + row_i * row_h
        for label, color, x in row:
            draw.rectangle([x, y + 2, x + 10, y + 12], fill=color)
            draw.text((x + swatch_w, y), label, fill=(255, 255, 255))

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

    tissue_top, tissue_bottom = tissue_extent(gray)
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
