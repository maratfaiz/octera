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
    "nfl",  # nerve fiber layer
    "gcl_ipl",  # ganglion cell / inner plexiform layer
    "inl",  # inner nuclear layer
    "opl",  # outer plexiform layer
    "onl",  # outer nuclear layer
    "is_os",  # photoreceptor inner/outer segments
    "rpe",  # retinal pigment epithelium
    "choroid",
]

# Human-readable Russian labels for LAYERS, for anywhere a clinician-facing view
# (report text, UI) needs to show layer thickness -- the bare codes above are an
# internal shorthand, not something to put in front of a doctor.
#
# Split into 8 bands (round 36) instead of the previous 5 merged ones, closer to
# standard OCT-device layer nomenclature. Caveat that matters for how much to
# trust any single boundary: there is no per-layer ground truth for this
# dataset (only a whole-scan DME/NORMAL label), so accuracy of each individual
# boundary isn't validated the way the DME classifier's accuracy is -- treat
# thin/low-contrast boundaries (ELM in particular was left merged into IS/OS
# rather than split out, since it's a single faint line most heuristic
# peak-detection can lose track of on a noisy scan) as illustrative anatomy,
# not a measured clinical reading.
LAYER_LABELS_RU = {
    "nfl": "Слой нервных волокон (NFL)",
    "gcl_ipl": "Слой ганглиозных клеток / внутренний плексиформный слой (GCL/IPL)",
    "inl": "Внутренний ядерный слой (INL)",
    "opl": "Наружный плексиформный слой (OPL)",
    "onl": "Наружный ядерный слой (ONL)",
    "is_os": "Слой фоторецепторов (IS/OS)",
    "rpe": "Пигментный эпителий сетчатки (RPE)",
    "choroid": "Хориоидея",
}

# Compact standard abbreviations for the on-image legend (_draw_layer_overlay) --
# the full LAYER_LABELS_RU phrases are for text (report, thickness table), not a
# space-constrained legend; these match the bare-abbreviation convention real
# OCT device software uses directly on the scan (e.g. "GCL/IPL", not the raw
# Python identifier "gcl_ipl").
LAYER_SHORT_LABELS = {
    "nfl": "NFL",
    "gcl_ipl": "GCL/IPL",
    "inl": "INL",
    "opl": "OPL",
    "onl": "ONL",
    "is_os": "IS/OS",
    "rpe": "RPE",
    "choroid": "CHOROID",
}

# One overlay color per layer band (_draw_layer_overlay indexes this mod
# len(LAYERS), so it's safe even if the two counts ever drift).
BOUNDARY_COLORS = [
    (255, 220, 60),
    (255, 170, 60),
    (255, 120, 70),
    (255, 90, 130),
    (220, 90, 220),
    (150, 120, 255),
    (90, 180, 255),
    (90, 220, 200),
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


def _seed_column(
    gray: np.ndarray, window: int, min_distance: int, needed: int, peaks_at
) -> int | None:
    """Picks the column with the clearest banded signal to start tracking from.

    Sampled rather than exhaustive for speed on wide images -- a coarse scan is
    enough to find a reasonably representative starting point. Uses `peaks_at`
    (a per-column cache shared with the caller's later full-width tracking
    pass) so a sampled column's peaks aren't recomputed once tracking reaches
    that same column again.
    """
    height, width = gray.shape
    step = max(1, width // 60)
    best_col, best_score = None, -1.0
    for col in range(0, width, step):
        peaks = peaks_at(col)
        if len(peaks) < needed:
            continue
        profile = smooth(gray[:, col], window)
        score = float(sum(profile[p] for p in peaks))
        if score > best_score:
            best_score, best_col = score, col
    return best_col


def _track_boundaries(gray: np.ndarray) -> np.ndarray:
    """Returns a (len(LAYERS) + 1, width) array: one tracked row-position per
    boundary per column, following the retina's actual curvature.

    The outermost two boundaries (index 0 = ILM/vitreoretinal interface,
    index -1 = outer edge of the choroid signal) are pinned to
    `signal_utils.tissue_extent` rather than independently rediscovered here --
    that function is already the shared, tested "real tissue vs. background"
    detector used by the pathology-zone detector and the classifier's domain
    features (see signal_utils.py's module docstring). An earlier version of
    this function re-derived its own top/bottom from column-peak prominence
    with no floor on brightness; splitting into more/thinner layers (round 36:
    5 layers -> 8) shrank the smoothing window enough that faint background
    noise above the retina started registering as a legitimate peak, so the
    tracked "top" boundary drifted up into empty vitreous space, and the
    labeled band for that boundary swallowed real background in the overlay.
    Confining peak search to inside the tissue band removes that failure mode
    at the source instead of just tuning parameters around it.

    Falls back to flat, evenly-spaced boundaries within the tissue band (or
    the whole image height, if no tissue is detected at all) when the image
    has no distinguishable inner banded structure -- callers always get a
    full set.
    """
    height, width = gray.shape
    needed = len(LAYERS) + 1
    inner_needed = needed - 2  # excludes the two tissue_extent-pinned boundaries

    tissue_top, tissue_bottom = tissue_extent(gray)
    has_tissue = bool((tissue_bottom > tissue_top).any())
    if not has_tissue:
        flat = np.linspace(0, height - 1, needed)
        return np.tile(flat[:, None], (1, width))

    window = max(3, height // (needed * 6))
    min_distance = max(2, height // (needed * 4))
    snap_tolerance = max(4, height // (needed * 2))

    # Cached across both the seed-column search and the full-width tracking
    # pass below -- the sampled columns _seed_column scores are a subset of
    # the columns _advance() will visit anyway, so this avoids recomputing
    # the same column's peaks twice.
    peaks_cache: dict[int, list[int]] = {}

    def peaks_at(col: int) -> list[int]:
        if col not in peaks_cache:
            top_i, bottom_i = int(tissue_top[col]), int(tissue_bottom[col])
            if bottom_i <= top_i:
                peaks_cache[col] = []
            else:
                profile = smooth(gray[top_i:bottom_i, col], window)
                peaks_cache[col] = [p + top_i for p in _find_peaks(profile, min_distance)]
        return peaks_cache[col]

    boundaries = np.zeros((needed, width), dtype=float)
    boundaries[0] = tissue_top
    boundaries[-1] = tissue_bottom

    if inner_needed > 0:
        seed_col = _seed_column(gray, window, min_distance, inner_needed, peaks_at)
        if seed_col is None:
            for col in range(width):
                boundaries[1:-1, col] = np.linspace(tissue_top[col], tissue_bottom[col], needed)[1:-1]
        else:
            seed_peaks = peaks_at(seed_col)
            seed_profile = smooth(gray[:, seed_col], window)
            top_by_prominence = np.argsort([seed_profile[p] for p in seed_peaks])[-inner_needed:]
            seed_positions = sorted(np.array(seed_peaks)[top_by_prominence].tolist())

            boundaries[1:-1, seed_col] = seed_positions

            def _advance(order: range, prev_positions: list[float]) -> None:
                prev = list(prev_positions)
                for col in order:
                    peaks = peaks_at(col)
                    new_positions = []
                    for p in prev:
                        if peaks:
                            nearest = min(peaks, key=lambda x: abs(x - p))
                            new_positions.append(nearest if abs(nearest - p) <= snap_tolerance else p)
                        else:
                            new_positions.append(p)
                    # Boundaries can't cross -- enforce non-decreasing order defensively
                    # against a noisy column snapping a boundary past its neighbor, and
                    # clamp back inside this column's own tissue band.
                    lo, hi = tissue_top[col], tissue_bottom[col]
                    for i in range(1, len(new_positions)):
                        new_positions[i] = max(new_positions[i], new_positions[i - 1] + 1)
                    new_positions = [float(np.clip(p, lo, hi)) for p in new_positions]
                    boundaries[1:-1, col] = new_positions
                    prev = new_positions

            _advance(range(seed_col + 1, width), seed_positions)
            _advance(range(seed_col - 1, -1, -1), seed_positions)

    smooth_window = max(3, width // 30)
    for i in range(needed):
        boundaries[i] = smooth(boundaries[i], smooth_window)
    boundaries[0] = tissue_top
    boundaries[-1] = tissue_bottom
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


def _detect_fovea_column(boundaries: np.ndarray) -> int | None:
    """Column of the foveal pit -- the point where the ILM boundary (the
    retina's top edge, `boundaries[0]`) dips deepest, since the fovea is a
    genuine anatomical thinning of the inner retina, not a fixed image
    position. Restricted to the central 60% of the scan width (real B-scans
    are usually centered on the fovea, and this avoids a false hit on
    vignetting/edge tracking noise near the image border).

    Returns None when there's no dip clearly beyond tracking noise -- e.g. a
    peripheral (non-macular) scan genuinely has no foveal pit in frame, and
    claiming one would be a fabricated finding, not a detected one.
    """
    width = boundaries.shape[1]
    height_scale = boundaries[-1].mean() - boundaries[0].mean()
    lo, hi = int(width * 0.2), int(width * 0.8)
    if hi <= lo or height_scale <= 0:
        return None
    ilm = boundaries[0]
    segment = ilm[lo:hi]
    col = lo + int(np.argmax(segment))
    dip = float(ilm[col] - np.median(ilm))
    if dip < max(4.0, 0.08 * height_scale):
        return None
    return col


def _draw_fovea_marker(img: Image.Image, col: int, top_row: int, bottom_row: int) -> None:
    draw = ImageDraw.Draw(img)
    for y in range(top_row, bottom_row, 6):
        draw.line([(col, y), (col, min(y + 3, bottom_row))], fill=(255, 255, 255), width=1)
    label = "Fovea"
    text_y = max(0, top_row - 16)
    text_w = int(draw.textlength(label))
    draw.text((min(max(col - text_w // 2, 2), img.width - text_w - 2), text_y), label, fill=(255, 255, 255))


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

    fovea_col = _detect_fovea_column(boundaries)
    if fovea_col is not None:
        _draw_fovea_marker(img, fovea_col, int(boundaries[0][fovea_col]), int(boundaries[-1][fovea_col]))

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
