"""Request media is refused by what it decodes to, before the decode allocates it."""
import struct
import sys
import types
import zlib
from dataclasses import dataclass

import pytest
import torch

from mstar.model.base import Model
from mstar.model.media_limits import (
    DEFAULT_MAX_IMAGE_PIXELS,
    VideoBudget,
    check_image_size,
    probe_video,
    read_image,
    read_video_whole_budgeted,
)


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


def _gif_lzw(n_pixels):
    """``n_pixels`` of colour 0, coded with a clear code often enough that the code width stays 3 bits."""
    codes = []
    for i in range(n_pixels):
        if i % 2 == 0:
            codes.append(4)
        codes.append(0)
    codes.append(5)
    bits = nbits = 0
    out = bytearray()
    for code in codes:
        bits |= code << nbits
        nbits += 3
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits:
        out.append(bits & 0xFF)
    blocks = bytearray([2])
    for at in range(0, len(out), 255):
        blocks += bytes([len(out[at:at + 255])]) + out[at:at + 255]
    return bytes(blocks + b"\x00")


def _gif(path, canvas, frame_sides):
    """A GIF with a ``canvas``-square logical screen and one frame per side in ``frame_sides``."""
    data = bytearray(b"GIF89a" + struct.pack("<HH", canvas, canvas) + bytes([0xF1, 0, 0]))
    data += b"\x00\x00\x00\xff\xff\xff" * 2
    for side in frame_sides:
        data += b"\x21\xf9\x04\x00\x05\x00\x00\x00"
        data += b"\x2c" + struct.pack("<HHHH", 0, 0, side, side) + b"\x00" + _gif_lzw(side * side)
    path.write_bytes(bytes(data + b"\x3b"))
    return str(path)


@pytest.fixture
def no_decode(monkeypatch):
    """Fail the test if the loader reaches torchvision's decoder."""
    import torchvision

    def refuse(*_args, **_kwargs):
        raise AssertionError("torchvision decoded a file that should not have reached it")

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


def test_gif_is_read_by_pillow_as_its_first_frame(tmp_path, no_decode):
    out = Model.load_image(None, _gif(tmp_path / "anim.gif", 16, [16, 16, 16]), "cpu")
    assert tuple(out.data.shape) == (3, 16, 16)


def test_gif_frames_larger_than_the_screen_never_reach_torchvision(tmp_path, no_decode):
    path = _gif(tmp_path / "odd.gif", 16, [16, 64, 64])
    assert tuple(read_image(path).shape) == (3, 16, 16)
    assert tuple(read_image(path, rgb=True).shape) == (3, 16, 16)


def test_gif_first_frame_past_the_screen_is_counted(tmp_path, no_decode, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "1000")
    with pytest.raises(ValueError, match="limit"):
        read_image(_gif(tmp_path / "wide.gif", 16, [64]))


def test_refused_image_reaches_the_client_as_a_400(tmp_path, no_decode):
    """The preprocess loop answers whatever ``_process_input`` raises with
    ``_fail_request``: a refused image becomes a 400 error chunk that states
    the limit, not the server path."""
    import queue

    from mstar.api_server.data_worker import PreprocessWorkerThread
    from mstar.api_server.request_types import PreprocessInput

    path = _png(tmp_path / "upload-name.png", 30_000, 30_000, rows=0)
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.in_flight_requests = set()
    wt.request_output_state = {}
    wt.device = "cpu"
    wt.model = types.SimpleNamespace(load_image=lambda p, device: Model.load_image(None, p, device))
    request = PreprocessInput(
        request_id="r1", text="describe it", file_paths={"image": [path]},
        input_modalities=["image", "text"], output_modalities=["text"], model_kwargs={},
    )
    with pytest.raises(ValueError) as info:
        wt._process_input(request)
    wt._fail_request("r1", info.value, "preprocessing")
    chunk = wt.out_queue.get_nowait()
    assert chunk.modality == "error"
    assert chunk.metadata["status"] == 400
    assert b"the limit is" in chunk.data
    assert b"upload-name" not in chunk.data


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
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS must be a positive integer"):
        VideoBudget()


@pytest.mark.parametrize("name", ["MSTAR_MAX_IMAGE_PIXELS", "MSTAR_MAX_VIDEO_PIXELS"])
def test_bad_env_value_stops_the_data_worker_at_startup(monkeypatch, name):
    from mstar.api_server.data_worker import PreprocessWorker

    monkeypatch.setenv(name, "lots")
    with pytest.raises(ValueError, match=name):
        PreprocessWorker(model=None)


@dataclass
class _Metadata:
    num_frames: int = 1000
    height: int = 64
    width: int = 64


def _fake_torchcodec(monkeypatch, pulled, metadata=None):
    class FakeDecoder:
        def __init__(self, *_args, **_kwargs):
            self.metadata = metadata or _Metadata()

        def __iter__(self):
            for i in range(self.metadata.num_frames):
                pulled.append(i)
                yield torch.zeros(3, 64, 64, dtype=torch.uint8)

        def __len__(self):
            return self.metadata.num_frames

        def get_frames_at(self, indices):
            pulled.extend(indices)
            return torch.zeros(len(indices), 3, self.metadata.height, self.metadata.width, dtype=torch.uint8)

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = FakeDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)


def test_base_video_loader_stops_at_the_budget(monkeypatch):
    pulled = []
    _fake_torchcodec(monkeypatch, pulled)
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 4))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        Model.load_video(None, "clip.mp4", "cpu")
    assert len(pulled) == 5


def test_vjepa2_refuses_sampled_frames_over_the_budget_before_decoding(monkeypatch):
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model

    pulled = []
    _fake_torchcodec(monkeypatch, pulled, _Metadata(num_frames=1000, height=64, width=64))
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 63))
    model = object.__new__(VJepa2Model)
    model.config = types.SimpleNamespace(frames_per_clip=64)
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        model.load_video("clip.mp4", "cpu")
    assert pulled == []


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


def test_probe_reads_the_header(tmp_path, monkeypatch):
    path = _mp4(tmp_path / "clip.mp4", frames=10)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)  # the PyAV path
    frames, fps, height, width = probe_video(path)
    assert (round(fps), height, width) == (30, 64, 64)
    assert 9 <= frames <= 11


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


def test_qwen3_omni_checks_sampled_frames_before_fetch_video(monkeypatch):
    try:
        from mstar.model.qwen3_omni import qwen3_omni_model
    except ImportError as exc:
        pytest.skip(f"Qwen3-Omni does not import here: {exc}")

    fetched = []
    vision = types.ModuleType("qwen_omni_utils.v2_5.vision_process")
    vision.fetch_video = lambda *a, **k: fetched.append(a)
    vision.smart_nframes = lambda ele, total_frames, video_fps: 64
    vision.VIDEO_READER_BACKENDS = {"torchvision": fetched.append}
    for name in ("qwen_omni_utils", "qwen_omni_utils.v2_5"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils.v2_5.vision_process", vision)
    monkeypatch.setattr(qwen3_omni_model, "probe_video", lambda _path: (10_000, 30.0, 4320, 7680))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        qwen3_omni_model.Qwen3OmniModel.load_video(None, "clip.mp4", "cpu")
    assert fetched == []
    assert vision.VIDEO_READER_BACKENDS["torchvision"] is read_video_whole_budgeted


def test_the_whole_clip_reader_counts_frames_as_it_decodes(tmp_path, monkeypatch):
    """qwen-omni-utils' torchvision reader decodes the whole clip, not the
    sampled frames the Qwen3-Omni loader checks; its replacement stops at the
    budget whatever the header says."""
    path = _mp4(tmp_path / "clip.mp4", frames=10)
    vision = types.ModuleType("qwen_omni_utils.v2_5.vision_process")
    vision.smart_nframes = lambda ele, total_frames, video_fps: 4
    for name in ("qwen_omni_utils", "qwen_omni_utils.v2_5"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils.v2_5.vision_process", vision)
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 4))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        read_video_whole_budgeted({"video": path})
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 20))
    video, metadata, sample_fps = read_video_whole_budgeted({"video": f"file://{path}"})
    assert video.shape == (4, 3, 64, 64) and video.dtype == torch.uint8
    assert metadata["total_num_frames"] == 10 and metadata["frames_indices"].tolist() == [0, 3, 6, 9]
    assert round(metadata["fps"]) == 30 and sample_fps == pytest.approx(4 / 10 * metadata["fps"])
