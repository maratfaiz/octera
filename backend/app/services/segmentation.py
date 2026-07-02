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
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

LAYERS = [
    "nfl_gcl",  # nerve fiber layer / ganglion cell layer
    "ipl_inl",  # inner plexiform / inner nuclear layer
    "opl_onl",  # outer plexiform / outer nuclear layer
    "photoreceptor",
    "rpe",
]

ASSUMED_UM_PER_PIXEL = 2.0


@dataclass
class SegmentationOutput:
    map_path: str
    layer_thickness_um: dict[str, float]


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

    return SegmentationOutput(map_path=map_path, layer_thickness_um=thickness)
