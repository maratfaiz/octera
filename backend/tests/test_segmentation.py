import numpy as np
from PIL import Image

from app.services.segmentation import (
    LAYERS,
    PATHOLOGY_LABELS_RU,
    _detect_fovea_column,
    _detect_pathology_findings,
    _detect_pathology_zones,
    _draw_layer_overlay,
    _track_boundaries,
    segment_layers,
)


def _synthetic_retina(height: int, width: int, dip_col: int | None) -> np.ndarray:
    """A bright banded retina over a dark background, with an optional foveal
    dip (a deeper trough in the band around `dip_col`), for testing boundary
    tracking and fovea detection without needing a real scan.
    """
    cols = np.arange(width)
    if dip_col is not None:
        top = 60 + 25 * np.exp(-(((cols - dip_col) / 20) ** 2))
    else:
        top = np.full(width, 60.0)
    band_height = 60
    gray = np.full((height, width), 0.03, dtype=np.float32)
    depth = np.arange(height)[:, None]
    # A few internal bright/dark stripes so boundary tracking has real peaks
    # to find, not just one flat plateau.
    stripe = 0.5 + 0.35 * np.sin((depth - top[None, :]) / 6.0)
    band_mask = (depth >= top[None, :]) & (depth < (top[None, :] + band_height))
    gray = np.where(band_mask, stripe, gray).astype(np.float32)
    return gray


def test_segment_layers_flat_image_falls_back_to_even_spacing(tmp_path):
    img = Image.new("L", (300, 300), color=128)
    path = tmp_path / "flat.png"
    img.save(path)

    result = segment_layers(str(path))

    assert set(result.layer_thickness_um.keys()) == set(LAYERS)
    assert all(v >= 0 for v in result.layer_thickness_um.values())


def test_segment_layers_reflects_real_banded_structure(tmp_path):
    height, width = 400, 200
    y = np.linspace(0, 1, height)[:, None]
    bands = (np.sin(y * 10 * np.pi) * 0.5 + 0.5) * 255
    img_array = np.tile(bands, (1, width)).astype(np.uint8)
    path = tmp_path / "banded.png"
    Image.fromarray(img_array, mode="L").save(path)

    flat_path = tmp_path / "flat.png"
    Image.new("L", (width, height), color=128).save(flat_path)

    banded_result = segment_layers(str(path))
    flat_result = segment_layers(str(flat_path))

    # A genuinely banded image should not produce the same evenly-spaced
    # fallback thickness as a flat, featureless image.
    assert banded_result.layer_thickness_um != flat_result.layer_thickness_um


def test_pathology_map_is_generated_alongside_segmentation(tmp_path):
    img = Image.new("L", (300, 300), color=128)
    path = tmp_path / "flat.png"
    img.save(path)

    result = segment_layers(str(path))

    assert {f.key for f in result.pathology_findings} == set(PATHOLOGY_LABELS_RU)
    assert all(f.zone_count >= 0 for f in result.pathology_findings)
    assert (tmp_path / "flat_pathology.png").exists()
    assert result.pathology_map_path == str(tmp_path / "flat_pathology.png")


def test_detect_pathology_findings_splits_by_category():
    """Round 37: a dark blob in the inner retina (above the IS/OS-RPE
    boundary) should register as intraretinal_fluid, not subretinal_fluid,
    and vice versa -- otherwise the two categories are just cosmetic
    relabeling of the same single detector rather than a real split.

    Uses a flat (untextured) band rather than `_synthetic_retina`'s sine-wave
    stripes: those stripes are meant to give boundary tracking real peaks to
    follow, but a sine wave's own troughs are dark enough relative to the
    smoothed row baseline to register as false-positive zones in both bands,
    which would make this test meaningless. A fully flat, untextured band
    also has no internal peaks for _track_boundaries to find, so inner
    boundaries fall back to evenly spacing tissue_extent's own top/bottom --
    which tapers near the image's left/right edges (a known, pre-existing
    smoothing artifact, see the round-36 boundary-containment test above), so
    assertions here only look at the carved column span (120:130), not the
    whole-image zone count which the edges can pollute independently of the
    category split under test.
    """
    height, width = 200, 300
    band_top, band_height = 60, 60
    gray = np.full((height, width), 0.03, dtype=np.float32)
    gray[band_top : band_top + band_height, :] = 0.6
    boundaries = _track_boundaries(gray)

    inner_row = int((boundaries[0].mean() + boundaries[5].mean()) / 2)
    gray_intraretinal = gray.copy()
    gray_intraretinal[inner_row - 5 : inner_row + 5, 120:130] = 0.05
    _, masks = _detect_pathology_findings(gray_intraretinal, boundaries)
    assert masks["intraretinal_fluid"][:, 120:130].any()
    assert not masks["subretinal_fluid"][:, 120:130].any()

    outer_row = int((boundaries[6].mean() + boundaries[8].mean()) / 2)
    gray_subretinal = gray.copy()
    gray_subretinal[outer_row - 3 : outer_row + 3, 120:130] = 0.05
    _, masks = _detect_pathology_findings(gray_subretinal, boundaries)
    assert masks["subretinal_fluid"][:, 120:130].any()
    assert not masks["intraretinal_fluid"][:, 120:130].any()


def test_detect_pathology_zones_finds_a_localized_dark_blob():
    height, width = 200, 200
    band = np.full((height, width), 0.6, dtype=np.float32)
    band[80:120, 80:120] = 0.1  # compact, notably darker patch well over the min-area threshold

    mask, count = _detect_pathology_zones(band, [0, height])

    assert count >= 1
    assert mask[80:120, 80:120].any()


def test_detect_pathology_zones_ignores_a_uniform_band():
    height, width = 200, 200
    rng = np.random.default_rng(0)
    band = np.full((height, width), 0.6, dtype=np.float32) + rng.normal(0, 0.01, (height, width)).astype(np.float32)

    _, count = _detect_pathology_zones(band, [0, height])

    assert count == 0


def test_layer_overlay_legend_wraps_instead_of_running_off_narrow_images():
    """A prior version of the legend used a fixed-width single row with hardcoded
    per-character spacing: on any image narrower than ~450px (including this
    module's own 200-300px test fixtures above), later labels -- up to and
    including "RPE" -- were drawn entirely off the canvas and never visible.
    """
    height, width = 300, 200
    gray = np.random.default_rng(0).random((height, width)).astype(np.float32)
    boundaries = np.tile(np.linspace(0, height - 1, len(LAYERS) + 1)[:, None], (1, width))

    overlay = _draw_layer_overlay(gray, boundaries)

    # Legend needs more than one row at this width, so the canvas must be taller
    # than just the image plus a single legend row.
    assert overlay.shape[0] > height + 30


def test_tracked_boundaries_stay_inside_the_tissue_band_not_the_background():
    """Regression test for round 36: an earlier version re-derived the top/bottom
    boundaries from column-peak prominence with no floor on brightness, and
    splitting into more/thinner layers shrank the smoothing window enough that
    background noise above the retina started registering as a peak -- the
    tracked top boundary drifted up into empty background space. The fix pins
    the outer two boundaries to signal_utils.tissue_extent instead.
    """
    height, width = 200, 300
    gray = _synthetic_retina(height, width, dip_col=None)

    boundaries = _track_boundaries(gray)

    # Every boundary in the central columns must stay within (a small
    # tolerance of) the true synthetic band [60, 120] -- not drift into the
    # dark background above (rows < 60) or below (rows >= 120) it. The
    # outermost ~20 columns are excluded: tissue_extent's own edge smoothing
    # tapers there regardless of this fix (shared with the classifier's
    # features, see signal_utils.py), which is a separate, pre-existing
    # characteristic this test isn't about.
    central = boundaries[:, 20:-20]
    assert central.min() >= 60 - 5
    assert central.max() <= 120 + 5


def test_detect_fovea_column_finds_the_dip():
    height, width = 200, 300
    gray = _synthetic_retina(height, width, dip_col=150)

    boundaries = _track_boundaries(gray)
    fovea_col = _detect_fovea_column(boundaries)

    assert fovea_col is not None
    assert abs(fovea_col - 150) <= 20


def test_detect_fovea_column_returns_none_without_a_real_dip():
    height, width = 200, 300
    gray = _synthetic_retina(height, width, dip_col=None)

    boundaries = _track_boundaries(gray)
    fovea_col = _detect_fovea_column(boundaries)

    assert fovea_col is None
