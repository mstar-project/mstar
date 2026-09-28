"""``Model.load_image`` decodes any PNG/JPEG mode down to 3 channels, the
convention every image model's ``process_prompt`` relies on."""

import numpy as np
import torch
import torchvision
from PIL import Image

from mstar.model.base import Model


class _Stub:
    pass


def _load(path) -> torch.Tensor:
    return Model.load_image(_Stub(), str(path), "cpu").data


def test_rgba_composites_alpha_over_white(tmp_path):
    path = tmp_path / "rgba.png"
    img = Image.new("RGBA", (2, 1), (0, 0, 0, 0))  # transparent black
    img.putpixel((1, 0), (255, 0, 0, 255))  # opaque red
    img.save(path)

    out = _load(path)

    assert out.shape[0] == 3
    assert torch.allclose(out[:, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(out[:, 0, 1], torch.tensor([1.0, 0.0, 0.0]))


def test_grayscale_replicates_to_three_channels(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (2, 2), 51).save(path)

    out = _load(path)

    assert out.shape[0] == 3
    assert torch.allclose(out, torch.full((3, 2, 2), 51 / 255))


def test_gray_alpha_fully_transparent_is_white(tmp_path):
    path = tmp_path / "la.png"
    Image.new("LA", (2, 2), (0, 0)).save(path)

    out = _load(path)

    assert out.shape[0] == 3
    assert torch.allclose(out, torch.ones(3, 2, 2))


def test_palette_image_resolves_true_colors(tmp_path):
    path = tmp_path / "pal.png"
    img = Image.new("P", (4, 4))
    img.putpalette([255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 6))
    img.putpixel((0, 0), 0)  # red
    img.putpixel((3, 3), 1)  # blue
    img.save(path)

    out = _load(path)

    assert out.shape[0] == 3
    assert torch.allclose(out[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out[:, 3, 3], torch.tensor([0.0, 0.0, 1.0]))


def test_palette_transparency_index_becomes_white(tmp_path):
    path = tmp_path / "pal_trans.png"
    img = Image.new("P", (4, 4))
    img.putpalette([255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 6))
    img.putpixel((0, 0), 0)  # red, but index 0 is the transparent color below
    img.putpixel((3, 3), 1)  # blue, stays opaque
    img.save(path, transparency=0)

    out = _load(path)

    assert torch.allclose(out[:, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(out[:, 3, 3], torch.tensor([0.0, 0.0, 1.0]))


def test_16bit_grayscale_clips_to_8bit_range(tmp_path):
    path = tmp_path / "i16.png"
    Image.new("I;16", (2, 2), 60000).save(path)

    out = _load(path)

    assert out.shape == (3, 2, 2)
    assert out.dtype == torch.float32
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_rgb_is_bit_identical_to_plain_decode_png_and_jpeg(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    png_path = tmp_path / "rgb.png"
    jpg_path = tmp_path / "rgb.jpg"
    Image.fromarray(arr, "RGB").save(png_path)
    Image.fromarray(arr, "RGB").save(jpg_path, quality=90)

    for path in (png_path, jpg_path):
        out = _load(path)
        expected = torchvision.io.decode_image(str(path)).float() / 255.0
        assert out.shape[0] == 3
        assert torch.equal(out, expected)
