import numpy as np
from PIL import Image

from app.services.segmentation import LAYERS, segment_layers


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
