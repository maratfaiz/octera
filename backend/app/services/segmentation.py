"""Retinal layer segmentation module.

Approximates retinal layer boundaries via 1D intensity-profile peak
detection along the B-scan's depth axis (classical "A-scan" analysis,
the technique OCT software used before deep-learning segmentation). It runs
on the real pixel data of the uploaded image -- unlike a per-pixel model,
which needs boundary-labeled ground truth we don't have. Swap this module
for a trained U-Net / nnU-Net once labeled segmentation data is available;
keep the same return contract (heatmap path + per-layer thickness in
microns).

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
diagnosis.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
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


def _layer_boundaries(gray: np.ndarray) -> list[int]:
    height = gray.shape[0]
    needed = len(LAYERS) + 1

    row_profile = _smooth(gray.mean(axis=1), window=max(3, height // 40))
    min_distance = max(2, height // 20)
    peaks = _find_peaks(row_profile, min_distance)

    if len(peaks) < needed:
        # Not enough distinguishable bands (e.g. a flat/uniform test image);
        # fall back to evenly spaced boundaries so callers always get a full set.
        return list(np.linspace(0, height - 1, needed, dtype=int))

    prominences = row_profile[peaks]
    top_by_prominence = np.argsort(prominences)[-needed:]
    return sorted(np.array(peaks)[top_by_prominence].tolist())


def _detect_pathology_zones(gray: np.ndarray, boundaries: list[int]) -> tuple[np.ndarray, int]:
    top, bottom = boundaries[0], boundaries[-1]
    band = gray[top:bottom, :]
    mask = np.zeros_like(gray, dtype=bool)
    if band.size == 0:
        return mask, 0

    row_baseline = _smooth(band.mean(axis=1), window=max(3, band.shape[0] // 20))[:, None]
    dark_mask = band < (row_baseline - DARKNESS_OFFSET)

    labeled, num_features = ndimage.label(dark_mask)
    min_area = max(15, band.size // MIN_ZONE_AREA_DIVISOR)
    max_width = 0.4 * band.shape[1]  # real fluid/cyst pockets are localized blobs, not
    # bands spanning most of the scan's width (which is usually a vitreous/layer
    # transition artifact rather than a discrete pathological zone).

    zone_count = 0
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() < min_area:
            continue
        cols = np.where(component.any(axis=0))[0]
        width = cols.max() - cols.min() + 1
        if width > max_width:
            continue
        mask[top:bottom, :][component] = True
        zone_count += 1

    return mask, zone_count


def _draw_pathology_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = (np.clip(gray, 0, 1) * 255).astype(np.float32)
    rgb = np.stack([base, base, base], axis=-1)
    overlay_color = np.array([255, 80, 60], dtype=np.float32)
    alpha = 0.45
    rgb[mask] = rgb[mask] * (1 - alpha) + overlay_color * alpha
    return rgb.astype(np.uint8)


def segment_layers(image_path: str) -> SegmentationOutput:
    with Image.open(image_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    heatmap = (_gradient_magnitude(gray) * 255).astype(np.uint8)
    map_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_segmentation.png"))
    Image.fromarray(heatmap, mode="L").save(map_path)

    boundaries = _layer_boundaries(gray)
    thickness = {
        layer: round((bottom - top) * ASSUMED_UM_PER_PIXEL, 1)
        for layer, top, bottom in zip(LAYERS, boundaries, boundaries[1:])
    }

    pathology_mask, zone_count = _detect_pathology_zones(gray, boundaries)
    pathology_overlay = _draw_pathology_overlay(gray, pathology_mask)
    pathology_map_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_pathology.png"))
    Image.fromarray(pathology_overlay, mode="RGB").save(pathology_map_path)

    return SegmentationOutput(
        map_path=map_path,
        layer_thickness_um=thickness,
        pathology_map_path=pathology_map_path,
        pathology_zone_count=zone_count,
    )
