"""Request media is refused by what it decodes to, before the decode allocates it."""
import importlib.util
import pathlib
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
    MIN_FRAME_PIXELS,
    RequestMediaBudget,
    VideoBudget,
    check_image_size,
    check_media_limits,
    probe_video,
    read_image,
)

# by path: importing the package would import the Qwen3-Omni model, which needs flashinfer
_spec = importlib.util.spec_from_file_location(
    "qwen3_omni_video_reader",
    pathlib.Path(__file__).parents[2] / "mstar" / "model" / "qwen3_omni" / "video_reader.py",
)
_video_reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_video_reader)
read_video_whole_budgeted = _video_reader.read_video_whole_budgeted


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


def _png16(path, samples, color_type):
    """A 16-bit PNG of ``samples`` (``[H, W, C]`` uint16): color type 0 gray, 2 RGB."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    height, width = samples.shape[:2]
    raw = b"".join(b"\x00" + samples[y].astype(">u2").tobytes() for y in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, color_type, 0, 0, 0))
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
    wt._keyed_labels = {}
    wt._live_requests = {}
    wt.model = types.SimpleNamespace(
        load_image=lambda p, device: Model.load_image(None, p, device), max_input_files=dict,
    )
    request = PreprocessInput(
        request_id="r1", text="describe it", file_paths={"image": [path]},
        input_modalities=["image", "text"], output_modalities=["text"], model_kwargs={},
    )
    with pytest.raises(ValueError) as info:
        wt._process_input(request)
    wt._fail_request("r1", info.value, "preprocessing", abort=False)
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


def test_model_image_loaders_check_the_header(tmp_path, no_decode):
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel
    from mstar.model.waypoint.waypoint_model import WaypointModel

    path = _png(tmp_path / "bomb.png", 30_000, 30_000, rows=0)
    for cls in (WaypointModel, Flux2KleinModel, LingBotWorldModel):
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
    height: int | None = 64
    width: int | None = 64
    average_fps: float = 30.0


def _fake_torchcodec(monkeypatch, pulled, metadata=None, opened=None, key_frames=None):
    class FakeDecoder:
        def __init__(self, source=None, *_args, **kwargs):
            self.source = source
            self.metadata = metadata or _Metadata()
            if opened is not None:
                opened.append(kwargs)

        def __iter__(self):
            for i in range(self.metadata.num_frames):
                pulled.append(i)
                yield torch.zeros(3, 64, 64, dtype=torch.uint8)

        def __len__(self):
            return self.metadata.num_frames

        def __getitem__(self, i):
            pulled.append(i)
            return torch.zeros(3, 64, 64, dtype=torch.uint8)

        def get_frames_at(self, indices):
            if isinstance(self.source, str):
                pathlib.Path(self.source).read_bytes()  # a decoder on a path reads the file to decode
            pulled.extend(indices)
            # every pixel of a frame holds its index (mod 256), to tell the frames apart
            frames = (torch.tensor(indices, dtype=torch.int64) % 256).to(torch.uint8)
            return frames.view(-1, 1, 1, 1).expand(-1, 3, self.metadata.height, self.metadata.width)

    if key_frames is not None:  # what _get_key_frame_indices returns, or the exception it raises
        def _get_key_frame_indices(self):
            if isinstance(key_frames, Exception):
                raise key_frames
            return key_frames

        FakeDecoder._get_key_frame_indices = _get_key_frame_indices
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


def test_vjepa2_refuses_a_video_without_load_video_metadata():
    """A video tensor it can't decode is an error, not a request without a video."""
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model

    model = VJepa2Model("unused", skip_weight_loading=True)
    with pytest.raises(ValueError, match="load_video"):
        model.process_prompt(None, ["video"], ["video"], tensors={"video_inputs": [torch.zeros(8, 3, 64, 64)]})


@pytest.mark.parametrize("kwargs, decoded", [
    ({"rollout_horizon": 0}, 64),
    ({"rollout_horizon": 4}, 8),
    # the encoder-only walk encodes the whole clip, whatever horizon the request names
    ({"rollout_horizon": 4, "skip_predictor": True}, 64),
])
def test_vjepa2_ac_decodes_only_the_frames_its_walk_encodes(monkeypatch, tmp_path, kwargs, decoded):
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model

    pulled = []
    _fake_torchcodec(monkeypatch, pulled, _Metadata(num_frames=1000, height=64, width=64))
    model = VJepa2Model("unused", skip_weight_loading=True, predictor_kind="ac")
    upload = tmp_path / "clip.mp4"
    upload.write_bytes(b"not decoded by the fake")
    loaded = model.load_video(str(upload), "cpu")
    assert pulled == []
    upload.unlink()  # the API server deletes an upload on a timer that doesn't wait for the decode
    out = model.process_prompt(
        None, ["video"], ["video"],
        tensors={"video_inputs": [loaded.data]}, input_metadata={"video_inputs": [loaded.metadata]},
        actions=[[0.0] * 7] * 4, states=[[0.0] * 7] * 4, **kwargs,
    ).new_input_tensors
    assert pulled == loaded.metadata["sampled_indices"][:decoded]
    assert out["video_frames"][0].shape[0] == decoded
    assert out["video_inputs"] == []


@pytest.mark.parametrize("key_frames, decoders", [
    (torch.tensor([0, 10, 20, 30, 40]), 3),  # one per group of GOPs; the index's decoder decodes the first
    ([0, 10, 20, 30, 40], 3),  # the list torchcodec's annotation promises
    (RuntimeError("private API changed"), 1),  # an unusable index: one decoder, not an error
])
def test_gop_split_decode_returns_the_frames_in_the_order_asked(monkeypatch, key_frames, decoders):
    """Sampled frames grouped by keyframe interval decode on separate decoders, each frame
    once, and come back in the order asked for, duplicates included; without a usable
    keyframe index they decode on one decoder."""
    from mstar.model.components import video_decode

    pulled, opened = [], []
    _fake_torchcodec(monkeypatch, pulled, opened=opened, key_frames=key_frames)
    monkeypatch.setattr(video_decode, "_max_decoders", lambda: 3)
    indices = [41, 3, 22, 3, 15, 0, 35, 22, 8]
    frames = video_decode.decode_frames_at(torch.zeros(16, dtype=torch.uint8), indices)
    assert frames[:, 0, 0, 0].tolist() == indices
    assert sorted(pulled) == sorted(set(indices) if decoders > 1 else indices)
    assert len(opened) == decoders


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


def test_a_tiny_frame_counts_as_the_minimum(monkeypatch):
    """Millions of 2x2 frames fit 2^30 pixels but cost about 2 KB each once
    decoded; every frame is charged at least MIN_FRAME_PIXELS."""
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(MIN_FRAME_PIXELS * 100))
    budget = VideoBudget()
    budget.add(2, 2, 100)
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        budget.add(2, 2)
    budget = VideoBudget()
    budget.add(64, 64, 100)  # a frame at the minimum counts as itself
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        budget.add(64, 64)


def test_cosmos3_pyav_fallback_counts_tiny_frames(tmp_path, monkeypatch):
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    path = _mp4(tmp_path / "tiny.mp4", frames=30, side=16)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)  # take the PyAV path
    model = object.__new__(Cosmos3Model)
    # 30 frames of 16x16 are 7,680 pixels by area, but 30 minimum frames are over 20
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(MIN_FRAME_PIXELS * 20))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        model.load_video(path, "cpu")


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
    from mstar.model.qwen3_omni.video_reader import read_video_whole_budgeted as installed
    assert vision.VIDEO_READER_BACKENDS["torchvision"] is installed


def test_qwen3_omni_without_a_header_count_decodes_with_the_counting_reader(tmp_path, monkeypatch):
    """A clip whose header gives no frame count (an MKV without one) cannot be
    charged before a sampled reader decodes it, so that reader must not run:
    fetch_video's fallback, the whole-clip reader, counts every frame."""
    try:
        from mstar.model.qwen3_omni import qwen3_omni_model
    except ImportError as exc:
        pytest.skip(f"Qwen3-Omni does not import here: {exc}")

    sampled = []

    def decord(ele):
        sampled.append(ele["video"])  # decodes its sampled frames with no count of its own
        return torch.zeros(4, 3, 64, 64, dtype=torch.uint8), {}, 2.0

    def fetch_video(ele, **_kwargs):
        # qwen-omni-utils 0.0.9: the selected reader, else its torchvision reader
        try:
            video, _metadata, sample_fps = vision.VIDEO_READER_BACKENDS["decord"](ele)
        except Exception:  # noqa: BLE001
            video, _metadata, sample_fps = vision.VIDEO_READER_BACKENDS["torchvision"](ele)
        return video, sample_fps

    vision = types.ModuleType("qwen_omni_utils.v2_5.vision_process")
    vision.fetch_video = fetch_video
    vision.smart_nframes = lambda ele, total_frames, video_fps: 4
    vision.VIDEO_READER_BACKENDS = {"decord": decord, "torchvision": None}
    for name in ("qwen_omni_utils", "qwen_omni_utils.v2_5"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils.v2_5.vision_process", vision)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)  # probe with PyAV

    mkv = _mp4(tmp_path / "clip.mkv", frames=10)
    assert probe_video(mkv) is None
    load_video = qwen3_omni_model.Qwen3OmniModel.load_video
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 4))
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        load_video(None, mkv, "cpu")
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 20))
    assert tuple(load_video(None, mkv, "cpu").data.shape) == (4, 3, 64, 64)
    assert sampled == []

    # a header with a frame count is charged first, and the sampled reader then runs
    mp4 = _mp4(tmp_path / "clip.mp4", frames=10)
    assert probe_video(mp4) is not None
    load_video(None, mp4, "cpu")
    assert sampled == [mp4]
    assert vision.VIDEO_READER_BACKENDS["decord"].__wrapped__ is decord


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


@pytest.mark.parametrize("fmt", ["BMP", "TIFF"])
def test_formats_torchvision_cannot_read_are_a_client_error(tmp_path, no_decode, fmt):
    from PIL import Image

    path = tmp_path / f"img.{fmt.lower()}"
    Image.new("RGB", (32, 24)).save(path, format=fmt)
    with pytest.raises(ValueError, match="not in a format"):
        Model.load_image(None, str(path), "cpu")


def _flux2():
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel

    model = object.__new__(Flux2KleinModel)
    model._config = types.SimpleNamespace(ref_image_max_area=1024 * 1024, spatial_alignment=16)
    return model


@pytest.mark.parametrize("fmt", ["BMP", "TIFF"])
def test_flux2_still_reads_the_formats_pillow_decodes(tmp_path, fmt):
    """FLUX.2 klein decodes reference images with Pillow, so the API's BMP and
    TIFF uploads keep loading there; only the torchvision loaders refuse them."""
    from PIL import Image

    path = tmp_path / f"ref.{fmt.lower()}"
    Image.new("RGB", (64, 64), (200, 20, 30)).save(path, format=fmt)
    out = _flux2().load_image(str(path), "cpu")
    assert tuple(out.data.shape) == (3, 64, 64)
    assert out.data[:, 0, 0].mul(255).round().tolist() == [200, 20, 30]


@pytest.mark.parametrize("fmt", ["BMP", "TIFF"])
def test_flux2_checks_the_header_of_the_formats_pillow_decodes(tmp_path, monkeypatch, fmt):
    from PIL import Image

    path = tmp_path / f"ref.{fmt.lower()}"
    Image.new("RGB", (64, 64)).save(path, format=fmt)
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", str(64 * 64 - 1))
    with pytest.raises(ValueError, match="limit"):
        _flux2().load_image(str(path), "cpu")


def test_animated_webp_is_read_by_pillow_as_its_first_frame(tmp_path, no_decode):
    from PIL import Image

    path = tmp_path / "anim.webp"
    first, second = Image.new("RGB", (32, 24), (200, 0, 0)), Image.new("RGB", (32, 24), (0, 0, 200))
    first.save(path, save_all=True, append_images=[second], duration=100, lossless=True)
    out = read_image(str(path))
    assert tuple(out.shape) == (3, 24, 32)
    assert out[:, 0, 0].tolist() == [200, 0, 0]


def test_a_raised_image_limit_is_not_masked_by_pillow(tmp_path, monkeypatch):
    """Pillow refuses past twice its own limit (about 179M pixels) unless the
    startup check lifts it to ``MSTAR_MAX_IMAGE_PIXELS``."""
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    path = _png(tmp_path / "wide.png", 20_000, 10_000, rows=0)
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", str(400_000_000))
    check_media_limits()
    assert Image.MAX_IMAGE_PIXELS == 400_000_000
    assert check_image_size(path) == ("PNG", False)


def test_a_lowered_image_limit_leaves_pillow_alone(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    before = Image.MAX_IMAGE_PIXELS
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "1000")
    check_media_limits()
    assert Image.MAX_IMAGE_PIXELS == before


def _preprocess_thread(model):
    import queue

    from mstar.api_server.data_worker import PreprocessWorkerThread

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.in_flight_requests = set()
    wt.request_output_state = {}
    wt.device = "cpu"
    wt._keyed_labels = {}
    wt._live_requests = {}
    wt.model = model
    return wt


def _image_request(paths):
    from mstar.api_server.request_types import PreprocessInput

    return PreprocessInput(
        request_id="r1", text="describe them", file_paths={"image": paths},
        input_modalities=["image", "text"], output_modalities=["text"], model_kwargs={},
    )


@pytest.fixture
def decoded(monkeypatch):
    """Stand in for torchvision's decoder and record each file it is given."""
    import torchvision

    paths = []

    def decode(path, mode=None):
        paths.append(path)
        return torch.zeros(3, 1, 1, dtype=torch.uint8)

    monkeypatch.setattr(torchvision.io, "decode_image", decode)
    return paths


def test_images_in_one_request_share_the_image_limit(tmp_path, decoded):
    """Eight 9000x9000 images each fit the default limit; together they are
    refused at the second, before it is decoded."""
    paths = [_png(tmp_path / f"copy{i}.png", 9_000, 9_000, rows=0) for i in range(8)]
    wt = _preprocess_thread(types.SimpleNamespace(
        load_image=lambda p, device: Model.load_image(None, p, device), max_input_files=dict,
    ))
    with pytest.raises(ValueError, match="images decode to more than .* together") as info:
        wt._process_input(_image_request(paths))
    assert decoded == paths[:1]
    wt._fail_request("r1", info.value, "preprocessing", abort=False)
    assert wt.out_queue.get_nowait().metadata["status"] == 400


def test_images_up_to_the_limit_together_all_load(tmp_path, decoded, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", str(3 * 64 * 64))
    paths = [_png(tmp_path / f"img{i}.png", 64, 64, rows=0) for i in range(3)]
    budget = RequestMediaBudget(num_files=3)
    for path in paths:
        with budget.file():
            Model.load_image(None, path, "cpu")
    assert decoded == paths and budget.used["image"] == 3 * 64 * 64


def test_too_many_files_are_refused_before_any_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_MEDIA_FILES", "2")
    loaded = []
    wt = _preprocess_thread(types.SimpleNamespace(load_image=lambda p, device: loaded.append(p), max_input_files=dict))
    paths = [_png(tmp_path / f"img{i}.png", 8, 8) for i in range(3)]
    with pytest.raises(ValueError, match="attaches 3 media files; the limit is 2"):
        wt._process_input(_image_request(paths))
    assert loaded == []


def test_a_video_is_charged_its_largest_check_not_their_sum(monkeypatch):
    """A header estimate and the decode after it count the same frames: one
    file is charged the larger, and the next file adds to it."""
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 100))
    budget = RequestMediaBudget(num_files=2)
    with budget.file():
        VideoBudget().add(64, 64, 60)
        decode = VideoBudget()
        for _ in range(60):
            decode.add(64, 64)
    assert budget.used["video"] == 64 * 64 * 60
    with budget.file(), pytest.raises(ValueError, match="videos decode to more than .* together"):
        VideoBudget().add(64, 64, 60)


def test_outside_a_request_each_file_is_checked_alone(monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 100))
    for _ in range(3):
        VideoBudget().add(64, 64, 60)


def test_vjepa2_without_frame_size_in_the_header_decodes_one_frame_to_check(monkeypatch):
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model

    pulled = []
    _fake_torchcodec(monkeypatch, pulled, _Metadata(num_frames=1000, height=None, width=None))
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 63))
    model = object.__new__(VJepa2Model)
    model.config = types.SimpleNamespace(frames_per_clip=64)
    with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
        model.load_video("clip.mp4", "cpu")
    assert pulled == [0]


@pytest.mark.parametrize("cpus,expected", [(1, 1), (3, 3), (64, 8)])
def test_video_decodes_thread_within_the_usable_cpus(monkeypatch, cpus, expected):
    """FFmpeg threads per decode follow the CPUs the process's plan leaves
    unpinned, at most 8, and not the API server's small preprocessing pool."""
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model
    from mstar.utils import cpus as cpu_plan

    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(cpu_plan, "_plan", cpu_plan.CpuPlan(frozenset(range(cpus)), cpus))
    monkeypatch.setenv("MSTAR_MAX_VIDEO_PIXELS", str(64 * 64 * 4))
    opened = []
    _fake_torchcodec(monkeypatch, [], opened=opened)
    vjepa2 = object.__new__(VJepa2Model)
    vjepa2.config = types.SimpleNamespace(frames_per_clip=64)
    for load in (lambda: Model.load_video(None, "clip.mp4", "cpu"), lambda: vjepa2.load_video("clip.mp4", "cpu")):
        with pytest.raises(ValueError, match="MSTAR_MAX_VIDEO_PIXELS"):
            load()
    assert [kw["num_ffmpeg_threads"] for kw in opened] == [expected, expected]


def test_probe_reads_torchcodec_metadata_without_a_scan(monkeypatch):
    opened = []
    _fake_torchcodec(monkeypatch, [], _Metadata(num_frames=10), opened=opened)
    assert probe_video("clip.mp4") == (10, 30.0, 64, 64)
    assert opened == [{"seek_mode": "approximate"}]


@pytest.mark.parametrize("color_type,channels", [(2, 3), (0, 1)])
def test_a_16_bit_png_reads_as_uint8(tmp_path, color_type, channels):
    """torchvision decodes a 16-bit PNG to uint16; read_image keeps the high
    byte, so the base loader's divide by 255 stays in [0, 1]."""
    import numpy as np
    from PIL import Image

    samples = np.random.default_rng(0).integers(0, 65536, size=(12, 20, channels), dtype=np.uint16)
    path = _png16(tmp_path / "deep.png", samples, color_type)
    image = read_image(path)
    assert image.dtype == torch.uint8 and image.shape == (channels, 12, 20)
    assert torch.equal(image, torch.from_numpy((samples >> 8).astype(np.uint8)).permute(2, 0, 1))
    if channels == 3:
        assert torch.equal(image, torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1))
    loaded = Model.load_image(None, path, "cpu").data
    assert 0.0 <= loaded.min() and loaded.max() <= 1.0
