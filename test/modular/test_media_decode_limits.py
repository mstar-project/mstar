"""Request media is refused by what it decodes to, before the decode allocates it."""
import struct
import sys
import types
import zlib
from dataclasses import dataclass

import pytest
import torch

from mstar.model.base import Model
from mstar.model.media_limits import DEFAULT_MAX_IMAGE_PIXELS, VideoBudget, check_image_size


def _png(path, width, height, *, rows=None):
    """An RGB PNG; ``rows=0`` writes the header and no pixel data."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = (b"\x00" + b"\x00" * (width * 3)) * (height if rows is None else rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return str(path)


@pytest.fixture
def no_decode(monkeypatch):
    """Fail the test if the loader reaches the decoder."""
    import torchvision

    def refuse(*_args, **_kwargs):
        raise AssertionError("decoded an image that should have been refused from its header")

    monkeypatch.setattr(torchvision.io, "decode_image", refuse)


def test_small_image_loads(tmp_path):
    out = Model.load_image(None, _png(tmp_path / "ok.png", 32, 24), "cpu")
    assert tuple(out.data.shape) == (3, 24, 32)


@pytest.mark.parametrize("side", [10_000, 30_000])
def test_header_over_the_limit_is_refused_before_decoding(tmp_path, no_decode, side):
    # 10000^2 is between Pillow's warning and error thresholds; 30000^2 is past both.
    path = _png(tmp_path / "bomb.png", side, side, rows=0)
    assert side * side > DEFAULT_MAX_IMAGE_PIXELS
    with pytest.raises(ValueError, match="limit"):
        Model.load_image(None, path, "cpu")


def test_limit_follows_the_env(tmp_path, no_decode, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "10000")
    with pytest.raises(ValueError, match="10000 pixels"):
        Model.load_image(None, _png(tmp_path / "big.png", 200, 200), "cpu")


def test_animated_gif_counts_every_frame(tmp_path, monkeypatch):
    from PIL import Image

    frames = [Image.new("RGB", (100, 100), (i * 60, 0, 0)) for i in range(3)]
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50)
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "25000")
    with pytest.raises(ValueError, match="3 frames"):
        check_image_size(str(path))
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "30000")
    check_image_size(str(path))


def test_unreadable_image_is_a_client_error_without_the_path(tmp_path):
    path = tmp_path / "secret-name.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError) as info:
        check_image_size(str(path))
    assert "secret-name" not in str(info.value)


def test_waypoint_and_flux2_loaders_check_the_header(tmp_path, no_decode):
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.waypoint.waypoint_model import WaypointModel

    path = _png(tmp_path / "bomb.png", 30_000, 30_000, rows=0)
    for cls in (WaypointModel, Flux2KleinModel):
        with pytest.raises(ValueError, match="limit"):
            cls.load_image(None, path, "cpu")


@pytest.mark.parametrize("value", ["0", "-1", "lots"])
def test_bad_env_value_is_an_error(monkeypatch, value):
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", value)
    with pytest.raises(ValueError):
        VideoBudget()


@dataclass
class _Metadata:
    num_frames: int = 1000


def test_base_video_loader_stops_at_the_budget(monkeypatch):
    pulled = []

    class FakeDecoder:
        def __init__(self, *_args, **_kwargs):
            self.metadata = _Metadata()

        def __iter__(self):
            for i in range(1000):
                pulled.append(i)
                yield torch.zeros(3, 64, 64, dtype=torch.uint8)

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = FakeDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 4))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        Model.load_video(None, "clip.mp4", "cpu")
    assert len(pulled) == 5


def _mp4(path, frames, side=64):
    av = pytest.importorskip("av")
    import numpy as np

    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width = stream.height = side
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((side, side, 3), dtype=np.uint8), format="rgb24")
        for _ in range(frames):
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return str(path)


def test_cosmos3_pyav_fallback_stops_at_the_budget(tmp_path, monkeypatch):
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    path = _mp4(tmp_path / "clip.mp4", frames=10)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)  # take the PyAV path
    model = object.__new__(Cosmos3Model)
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 4))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        model.load_video(path, "cpu")
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 10))
    assert model.load_video(path, "cpu").data.shape[0] == 10
