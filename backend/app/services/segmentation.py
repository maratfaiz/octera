"""Retinal layer segmentation module.

Stub implementation that produces a deterministic placeholder heatmap and
plausible layer-thickness measurements from the input image. Swap the body
of `segment_layers` with a trained segmentation model (e.g. U-Net / nnU-Net)
without changing the return contract.
"""

import hashlib
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


@dataclass
class SegmentationOutput:
    map_path: str
    layer_thickness_um: dict[str, float]


def _seeded_rng(image_path: str) -> np.random.Generator:
    seed = int(hashlib.sha256(image_path.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def segment_layers(image_path: str) -> SegmentationOutput:
    rng = _seeded_rng(image_path)

    with Image.open(image_path) as img:
        width, height = img.size

    # Placeholder heatmap: a smooth gradient standing in for a model attention
    # map, saved next to the source image so the frontend has something to render.
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    gradient = np.outer(np.sin(y * np.pi), np.cos(x * np.pi))
    gradient_range = gradient.max() - gradient.min()
    heatmap = ((gradient - gradient.min()) / (gradient_range + 1e-6) * 255).astype(np.uint8)

    map_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_segmentation.png"))
    Image.fromarray(heatmap, mode="L").save(map_path)

    thickness = {layer: round(float(rng.uniform(20, 120)), 1) for layer in LAYERS}
    return SegmentationOutput(map_path=map_path, layer_thickness_um=thickness)
