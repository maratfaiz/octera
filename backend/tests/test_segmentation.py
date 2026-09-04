import numpy as np
from PIL import Image

from app.ml.signal_utils import dark_mask_in_band
from app.services.segmentation import (
    DARKNESS_OFFSET,
    LAYERS,
    PATHOLOGY_LABELS_RU,
    _RPE_BOTTOM_IDX,
    _detect_drusen_zones,
    _detect_fovea_column,
    _detect_pathology_findings,
    _detect_pathology_zones,
    _draw_layer_overlay,
    _legend_font,
    _track_boundaries,
    segment_layers,
)


def _dark_mask_fn(flat, valid):
    return dark_mask_in_band(flat, valid, DARKNESS_OFFSET)


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

    mask, count = _detect_pathology_zones(band, [0, height], _dark_mask_fn)

    assert count >= 1
    assert mask[80:120, 80:120].any()


def test_detect_pathology_zones_ignores_a_uniform_band():
    height, width = 200, 200
    rng = np.random.default_rng(0)
    band = np.full((height, width), 0.6, dtype=np.float32) + rng.normal(0, 0.01, (height, width)).astype(np.float32)

    _, count = _detect_pathology_zones(band, [0, height], _dark_mask_fn)

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


def test_pathology_categories_cover_the_is_os_band_no_gap():
    """Regression test: PATHOLOGY_BOUNDARY_RANGES used to split at boundary
    indices (0, 5) and (6, 8), leaving the IS/OS band -- boundaries[5] to
    boundaries[6] -- uncovered by either category. A dark zone entirely
    inside it (e.g. subretinal fluid that has pushed up against the
    photoreceptor layer) was silently never reported by either. Caught in
    code review (three independent review angles flagged the same gap), not
    by the original test_detect_pathology_findings_splits_by_category, which
    never placed a blob inside this specific band. Fixed by having both
    ranges share boundary index 5 (IS/OS's own anatomical split point)
    instead of skipping it.
    """
    height, width = 200, 300
    band_top, band_height = 60, 60
    gray = np.full((height, width), 0.03, dtype=np.float32)
    gray[band_top : band_top + band_height, :] = 0.6
    boundaries = _track_boundaries(gray)

    is_os_row = int((boundaries[5].mean() + boundaries[6].mean()) / 2)
    gray_gap = gray.copy()
    gray_gap[is_os_row - 3 : is_os_row + 3, 120:130] = 0.05

    _, masks = _detect_pathology_findings(gray_gap, boundaries)

    assert masks["subretinal_fluid"][:, 120:130].any()


def test_detect_pathology_findings_flags_bright_subretinal_material_not_dark_fluid():
    """subretinal_hyperreflective_material is bright_mask_in_band's mirror
    image of subretinal_fluid's dark_mask_in_band, scoped to the same
    boundary range -- a bright patch there must register as SHRM, not as
    fluid (which only fires on darker-than-baseline pixels).
    """
    height, width = 200, 300
    band_top, band_height = 60, 60
    gray = np.full((height, width), 0.03, dtype=np.float32)
    gray[band_top : band_top + band_height, :] = 0.6
    boundaries = _track_boundaries(gray)

    is_os_row = int((boundaries[5].mean() + boundaries[6].mean()) / 2)
    gray_bright = gray.copy()
    gray_bright[is_os_row - 3 : is_os_row + 3, 120:130] = 0.95  # brighter than the 0.6 baseline

    _, masks = _detect_pathology_findings(gray_bright, boundaries)

    assert masks["subretinal_hyperreflective_material"][:, 120:130].any()
    assert not masks["subretinal_fluid"][:, 120:130].any()


def test_detect_drusen_zones_finds_a_localized_rpe_bulge():
    """Drusen bulge the RPE-choroid boundary outward -- unlike the intensity-
    based categories above, this must fire from the boundary curve's own
    shape, not from pixel brightness.
    """
    height, width = 200, 300
    boundaries = np.zeros((len(LAYERS) + 1, width), dtype=float)
    boundaries[0] = 60.0
    boundaries[-1] = 140.0
    for i in range(1, len(LAYERS)):
        boundaries[i] = 60.0 + i * 10.0
    boundaries[_RPE_BOTTOM_IDX, 100:130] += 15.0  # a real, localized outward bulge

    mask, count = _detect_drusen_zones(height, boundaries)

    assert count >= 1
    assert mask[:, 100:130].any()
    assert not mask[:, :50].any()
    assert not mask[:, 250:].any()


def test_detect_drusen_zones_ignores_flat_boundary_edge_smoothing_artifact():
    """Regression test: an earlier version compared the RPE-choroid boundary
    directly to `smooth`'s output with no edge margin. `smooth`'s boxcar
    convolution implicitly zero-pads past the array edges, which drags the
    smoothed baseline down near columns 0 and width-1 regardless of image
    content -- a perfectly flat boundary (no real bulge anywhere) still fired
    false "drusen" near both edges, over a span of roughly half the
    smoothing window. Every real image's RPE boundary reaches both edges, so
    unfixed this would have false-flagged drusen on nearly every scan.
    """
    height, width = 200, 300
    boundaries = np.zeros((len(LAYERS) + 1, width), dtype=float)
    boundaries[0] = 60.0
    boundaries[-1] = 140.0
    for i in range(1, len(LAYERS)):
        boundaries[i] = 60.0 + i * 10.0  # perfectly flat -- no real bulge anywhere

    mask, count = _detect_drusen_zones(height, boundaries)

    assert count == 0
    assert not mask.any()


def test_detect_pathology_zones_min_area_reference_uses_provided_pixel_count():
    """Regression test: round 37 started calling _detect_pathology_zones once
    per category over a narrower sub-band instead of once over the whole
    retinal band. Without min_area_reference_pixels, each sub-band's smaller
    valid.sum() would silently lower the min-blob-size noise floor purely
    because of how the scope got divided -- not because of any change in
    image content. min_area_reference_pixels lets a caller calibrate against
    a caller-chosen reference (the full band) instead.
    """
    height, width = 100, 100
    band = np.full((height, width), 0.6, dtype=np.float32)
    band[40:44, 40:44] = 0.1  # a small, compact 16px blob

    _, count_default = _detect_pathology_zones(band, [0, height], _dark_mask_fn)
    _, count_with_large_reference = _detect_pathology_zones(
        band, [0, height], _dark_mask_fn, min_area_reference_pixels=10_000_000
    )

    assert count_default >= 1
    assert count_with_large_reference == 0


def test_tracked_boundaries_never_push_the_bottom_past_true_tissue_bottom():
    """Regression test: at a column where the tissue band's thickness changes
    sharply, cross-column smoothing of an inner boundary could push it past
    that column's own tissue_bottom. The monotonic-order enforcement pass
    would then push boundaries[-1] (just reset to the true tissue_bottom) back
    out past the real tissue edge to satisfy boundaries[-1] > boundaries[-2],
    silently breaking _track_boundaries's own documented invariant that
    boundaries[-1] always equals tissue_extent's bottom. Caught in code
    review with a reproduction on a sharp interior thickness cliff -- the
    existing boundary-containment test only excludes the image's outer edge
    columns and never exercised an interior cliff like this one.
    """
    from app.ml.signal_utils import tissue_extent

    height, width = 250, 300
    top = 30
    bottom = np.where(np.arange(width) < 150, 200, 90)  # sharp cliff at col 150
    depth = np.arange(height)[:, None]
    stripe = 0.5 + 0.35 * np.sin((depth - top) / 6.0)
    band_mask = (depth >= top) & (depth < bottom[None, :])
    gray = np.where(band_mask, stripe, 0.03).astype(np.float32)

    tissue_top, tissue_bottom = tissue_extent(gray)
    boundaries = _track_boundaries(gray)

    assert (boundaries[-1] <= tissue_bottom + 1).all()


def test_legend_font_renders_cyrillic_glyphs():
    """Regression test: PIL's default bitmap font (used by ImageDraw.text
    when no font is given) has no Cyrillic glyphs -- every character
    silently renders as an unreadable box. Round 37's pathology-map legend
    ("Выявлены"/"Не выявлены") was the first Cyrillic text ever drawn onto a
    saved image in this module (the layer-overlay legend before it only used
    ASCII short labels like "RPE"), so this went unnoticed until caught by
    actually looking at the rendered PNG, not just running the existing
    tests. Guards against the bundled font asset (app/assets/fonts/) being
    deleted or moved without updating the loader.
    """
    font = _legend_font(13)
    bbox = font.getbbox("Выявлены")
    assert bbox is not None
    assert bbox[2] - bbox[0] > 20  # a real glyph run, not a collapsed/missing one
