"""Quality-check module.

Stub implementation using basic image statistics (resolution, sharpness).
Swap the body of `check_quality` with a trained classifier (noise / blur /
scan-artifact detection) without changing the return contract.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MIN_RESOLUTION = 256
SHARPNESS_THRESHOLD = 8.0


@dataclass
class QualityReport:
    score: float
    issues: list[str] = field(default_factory=list)


def _sharpness(gray: np.ndarray) -> float:
    # Variance of a simple discrete Laplacian kernel as a sharpness proxy.
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def check_quality(image_path: str) -> QualityReport:
    issues: list[str] = []

    with Image.open(image_path) as img:
        width, height = img.size
        gray = np.asarray(img.convert("L"), dtype=np.float32)

    if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
        issues.append("low_resolution")

    sharpness = _sharpness(gray)
    if sharpness < SHARPNESS_THRESHOLD:
        issues.append("blurry")

    if float(gray.mean()) < 15 or float(gray.mean()) > 240:
        issues.append("poor_exposure")

    score = max(0.0, min(1.0, 1.0 - 0.25 * len(issues)))
    return QualityReport(score=score, issues=issues)
