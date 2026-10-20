"""The base image loader hands every model three channels, whatever the file holds."""
import io

import numpy as np
import pytest
import torch
from PIL import Image

from mstar.model.base import Model

RGB = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)


def _save(path, mode, fmt):
    Image.fromarray(RGB).convert(mode).save(path, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return str(path)


CASES = [("RGBA", "PNG", "rgba.png"), ("CMYK", "JPEG", "cmyk.jpg"), ("LA", "PNG", "gray_alpha.png"),
         ("L", "PNG", "gray.png"), ("P", "PNG", "palette.png"), ("RGB", "PNG", "rgb.png")]


@pytest.mark.parametrize(("mode", "fmt", "name"), CASES)
def test_image_loads_as_rgb(tmp_path, mode, fmt, name):
    path = _save(tmp_path / name, mode, fmt)
    out = Model.load_image(None, path, "cpu").data
    with Image.open(path) as image:
        expected = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    assert out.shape == (3, 48, 64)
    torch.testing.assert_close(out, expected, atol=1 / 255, rtol=0)


@pytest.mark.parametrize(("mode", "fmt", "name"), CASES)
def test_qwen_image_processor_accepts_it(tmp_path, mode, fmt, name):
    """Qwen3-Omni's image processor refused 4- and 2-channel images ("Unable
    to infer channel dimension format", a 400) and 1-channel ones (its
    3-channel mean, a 400 or a 500 depending on the transformers version)."""
    transformers = pytest.importorskip("transformers")
    out = Model.load_image(None, _save(tmp_path / name, mode, fmt), "cpu").data
    # Qwen3OmniModel.process_prompt's conversion: uint8, HWC for 1 or 3 channels
    img = (out * 255.0).clamp(0, 255).to(torch.uint8)
    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    processed = transformers.Qwen2VLImageProcessor()(images=[img.contiguous().numpy()], return_tensors="pt")
    assert processed["pixel_values"].shape[0] > 0


# a dark drawing on a transparent background: the transparent pixels hold black
DRAWING = np.zeros((48, 64, 4), np.uint8)
DRAWING[8:40, 8:56] = (30, 30, 30, 255)
DRAWING[20:28, :] = (200, 20, 20, 128)  # half transparent
DRAWING[44:, :8] = (0, 0, 200, 64)


def _paletted(path, fmt):
    """The drawing as a palette image whose index 0 (black) is transparent."""
    image = Image.new("P", (64, 48), 0)
    image.putpalette([0, 0, 0, 30, 30, 30, 200, 20, 20])
    image.paste(1, (8, 8, 56, 40))
    image.paste(2, (0, 20, 64, 28))
    image.save(path, format=fmt, transparency=0)
    return str(path)


def _trns_color(path, mode):
    """A gray or RGB PNG whose black is transparent through a tRNS chunk."""
    image = Image.fromarray(DRAWING[..., :3]).convert(mode)
    image.save(path, format="PNG", transparency=0 if mode == "L" else (0, 0, 0))
    return str(path)


def _save_drawing(path, mode, fmt):
    extra = {"lossless": True} if fmt == "WEBP" else {}
    Image.fromarray(DRAWING, "RGBA").convert(mode).save(path, format=fmt, **extra)
    return str(path)


TRANSPARENT = {
    "rgba.png": lambda p: _save_drawing(p, "RGBA", "PNG"),
    "gray_alpha.png": lambda p: _save_drawing(p, "LA", "PNG"),
    "rgba.webp": lambda p: _save_drawing(p, "RGBA", "WEBP"),
    "palette_trns.png": lambda p: _paletted(p, "PNG"),
    "palette_trns.gif": lambda p: _paletted(p, "GIF"),
    "gray_trns.png": lambda p: _trns_color(p, "L"),
    "rgb_trns.png": lambda p: _trns_color(p, "RGB"),
}


def _reference_on_white(path):
    """qwen_vl_utils' ``to_rgb`` for RGBA: paste on white with the alpha as the mask."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.split()[3])
    return torch.from_numpy(np.array(white)).permute(2, 0, 1).float() / 255.0


@pytest.mark.parametrize("name", list(TRANSPARENT))
def test_transparency_is_composited_on_white(tmp_path, name):
    """Dropping the alpha turned a transparent background into black, so a dark
    drawing reached the model as a black square."""
    path = TRANSPARENT[name](tmp_path / name)
    out = Model.load_image(None, path, "cpu").data
    assert out.shape == (3, 48, 64)
    assert torch.equal(out[:, 0, 60], torch.ones(3))  # a fully transparent pixel
    torch.testing.assert_close(out, _reference_on_white(path), atol=1 / 255, rtol=0)


@pytest.mark.parametrize(("color_type", "channels"), [(6, 4), (4, 2)])  # RGBA, gray+alpha
def test_a_16_bit_png_with_alpha_is_composited_on_white(tmp_path, color_type, channels):
    from test_media_decode_limits import _png16

    samples = np.zeros((12, 20, channels), np.uint16)
    samples[:, 10:, :-1] = 30 * 257
    samples[:, 10:, -1] = 65535
    out = Model.load_image(None, _png16(tmp_path / "deep.png", samples, color_type), "cpu").data
    assert out.shape == (3, 12, 20)
    assert torch.equal(out[:, :, :10], torch.ones(3, 12, 10))
    torch.testing.assert_close(out[:, :, 10:], torch.full((3, 12, 10), 30 / 255))


@pytest.mark.parametrize("fmt", ["PNG", "GIF"])  # GIF goes through Pillow, PNG through torchvision
def test_corrupt_image_is_a_client_error_without_the_path(tmp_path, fmt):
    buf = io.BytesIO()
    Image.fromarray(RGB).save(buf, format=fmt)
    path = tmp_path / f"secret-name.{fmt.lower()}"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])  # the header survives, the pixels don't
    with pytest.raises(ValueError, match="corrupt or truncated") as info:
        Model.load_image(None, str(path), "cpu")
    assert "secret-name" not in str(info.value)
