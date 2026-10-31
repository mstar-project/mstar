"""The API server must not treat a request with no reported outputs as
fully delivered, and a request that fails preprocessing must reach the
client as an error instead of leaving it to hit the server timeout: a 400
for an upload that won't decode, a 500 for the server's own fault.

An empty output_chunk_counts dict made received_final_chunks vacuously true, so a
request whose walk emitted nothing (or whose completion message raced ahead
of every result) completed instantly with zero chunks instead of holding for
late results.
"""

import importlib
import io
import queue
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.model.base import Model


def _worker_with(received_chunk_counts):
    worker = PreprocessWorker.__new__(PreprocessWorker)
    worker.received_chunk_counts = received_chunk_counts
    return worker


def test_empty_output_chunk_counts_is_not_done():
    worker = _worker_with({"r1": {}})
    assert worker.received_final_chunks("r1", {}) is False


def test_output_chunk_counts_wait_for_registration_then_complete():
    worker = _worker_with({"r1": {}})
    # Completion reported but the terminal chunk hasn't registered yet.
    assert worker.received_final_chunks("r1", {"video_output": 1}) is False
    worker.received_chunk_counts["r1"]["video_output"] = 1
    assert worker.received_final_chunks("r1", {"video_output": 1}) is True


def _preprocess_failure(model, file_paths=None):
    """Run one request through the preprocess loop and return its error chunk
    and the rids force-cleaned."""
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.in_queue = queue.Queue()
    wt._is_pending = lambda key: True
    wt.out_queue = queue.Queue()
    wt.result_tensor_queue = queue.Queue()
    wt.cleanup_request_queue = queue.Queue()
    wt.abort_request_queue = queue.Queue()
    wt.reads_done_queue = queue.Queue()
    wt.discard_tensor_queue = queue.Queue()
    wt.stop_event = threading.Event()
    wt.timing_requested = threading.Event()
    wt.communicator = SimpleNamespace(
        get_all_new_messages=lambda: [], send=lambda *args: None,
    )
    cleaned = []
    wt.tensor_manager = SimpleNamespace(
        force_cleanup_request=cleaned.append,
        has_inflight_reads=lambda rid: False,
        get_ready_tensors=lambda: {},
        take_failed_reads=lambda: [],
    )
    wt.model = model
    wt.device = "cpu"
    wt.tensor_uuid_to_metadata_per_request = {}
    wt.request_model_kwargs = {}
    wt.in_flight_requests = set()
    wt._draining_rids = set()
    wt._reads_done_sent = set()
    wt._live_requests = {}
    wt._keyed_labels = {}
    wt._client_edges = frozenset()

    wt.in_queue.put(PreprocessInput(
        request_id="r1", text="x", file_paths=file_paths,
        input_modalities=[*(file_paths or {}), "text"], output_modalities=["video"],
        model_kwargs={},
    ))
    thread = threading.Thread(target=wt.run, daemon=True)
    thread.start()
    try:
        chunk = wt.out_queue.get(timeout=10)
    finally:
        wt.stop_event.set()
        thread.join(timeout=10)
    assert chunk.request_id == "r1"
    assert chunk.modality == "error"
    return chunk, cleaned


def test_preprocess_failure_emits_error_chunk():
    """A request rejected during preprocessing (e.g. an invalid model kwarg)
    must produce an "error" chunk so the waiting client is released."""

    class _RejectingModel:
        def process_prompt(self, *args, **kwargs):
            raise ValueError("bad knob")

    chunk, cleaned = _preprocess_failure(_RejectingModel())
    assert b"bad knob" in chunk.data
    assert chunk.metadata["status"] == 400
    assert cleaned == ["r1"]


class _BaseLoaders:
    """``Model``'s own loaders, for the models that don't override them."""

    max_input_files = Model.max_input_files
    load_image = Model.load_image
    load_audio = Model.load_audio
    load_video = Model.load_video


def _bare(module, cls, **attrs):
    model = object.__new__(getattr(importlib.import_module(f"mstar.model.{module}"), cls))
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def _flux2_klein():
    from mstar.model.flux2_klein.config import Flux2KleinConfig

    return _bare("flux2_klein.flux2_klein_model", "Flux2KleinModel", _config=Flux2KleinConfig())


def _qwen3_tts():
    from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig

    return _bare("qwen3_tts.qwen3_tts_model", "Qwen3TTSModel", config=Qwen3TTSModelConfig())


MODELS = {
    "base": _BaseLoaders,
    "waypoint": lambda: _bare("waypoint.waypoint_model", "WaypointModel"),
    "flux2_klein": _flux2_klein,
    "lingbot_world": lambda: _bare("lingbot_world.lingbot_world_model", "LingBotWorldModel"),
    "chatterbox": lambda: _bare("chatterbox.chatterbox_model", "ChatterboxModel"),
    "omnivoice": lambda: _bare(
        "omnivoice.omnivoice_model", "OmniVoiceModel", config=SimpleNamespace(sample_rate=24_000),
    ),
    "fish_s2_pro": lambda: _bare(
        "fish_s2_pro.fish_s2_pro_model", "FishS2ProModel",
        config=SimpleNamespace(codec=SimpleNamespace(sample_rate=44_100)),
    ),
    "qwen3_tts": _qwen3_tts,
    "vjepa2": lambda: _bare("vjepa2.vjepa2_model", "VJepa2Model", config=SimpleNamespace(frames_per_clip=64)),
}


def _image(fmt):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


FILES = {
    "image.png": lambda: _image("PNG"),
    # the header survives, the pixels don't
    "truncated.png": lambda: _image("PNG")[:4000],
    "truncated.jpg": lambda: _image("JPEG")[:2000],
    "garbage.wav": lambda: b"RIFF\x24\x00\x00\x00WAVE" + b"not audio" * 400,
    "garbage.m4a": lambda: b"\x00\x00\x00\x18ftypM4A " + b"not audio" * 400,
    "garbage.mp4": lambda: b"\x00\x00\x00\x18ftypmp42" + b"not a video" * 400,
}


def _refuse(*args, **kwargs):
    """What torchcodec's decoders raise for a file FFmpeg can't read."""
    raise RuntimeError("Invalid data found when processing input")


def _allocate(*args, **kwargs):
    return torch.empty(1 << 62, dtype=torch.uint8)


def _torchcodec(monkeypatch, decoder, ffmpeg_major_version=7):
    """A torchcodec that imports and decodes with ``decoder``. One whose
    ``ffmpeg_major_version`` is None found no FFmpeg (0.16 imports without it)."""
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = decoders.VideoDecoder = decoder
    package = types.ModuleType("torchcodec")
    package.decoders, package.ffmpeg_major_version = decoders, ffmpeg_major_version
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)


def _no_ffmpeg_at_import(monkeypatch, tmp_path):
    """torchcodec 0.10 and earlier without FFmpeg: every import raises."""
    package = tmp_path / "site" / "torchcodec"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise RuntimeError("Could not load libtorchcodec")\n')
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    for name in ("torchcodec", "torchcodec.decoders"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def _no_ffmpeg_at_decode(monkeypatch, tmp_path):
    _torchcodec(monkeypatch, _refuse, ffmpeg_major_version=None)


def _allocation_fails(monkeypatch, tmp_path):
    import torchvision

    monkeypatch.setattr(torchvision.io, "decode_image", _allocate)
    _torchcodec(monkeypatch, _allocate)


def _upload_removed(monkeypatch, tmp_path):
    _torchcodec(monkeypatch, _refuse)


def _path(tmp_path, name):
    path = tmp_path / name
    if name in FILES:
        path.write_bytes(FILES[name]())
    return str(path)


@pytest.mark.parametrize(("model", "modality", "name"), [
    ("base", "image", "truncated.png"),  # torchvision
    ("waypoint", "image", "truncated.jpg"),
    ("flux2_klein", "image", "truncated.jpg"),  # Pillow
    ("lingbot_world", "image", "truncated.png"),
    ("base", "audio", "garbage.m4a"),  # libsndfile, then torchcodec
    ("chatterbox", "audio", "garbage.m4a"),
    ("omnivoice", "audio", "garbage.m4a"),
    ("fish_s2_pro", "audio", "garbage.wav"),  # libsndfile alone
    ("qwen3_tts", "audio", "garbage.wav"),
    ("base", "video", "garbage.mp4"),  # torchcodec
    ("vjepa2", "video", "garbage.mp4"),
])
def test_an_upload_that_will_not_decode_is_a_400(tmp_path, monkeypatch, model, modality, name):
    loader = MODELS[model]()
    _torchcodec(monkeypatch, _refuse)
    chunk, cleaned = _preprocess_failure(loader, {modality: [_path(tmp_path, name)]})
    assert chunk.metadata["status"] == 400, chunk.data
    assert b"could not decode" in chunk.data.lower() and str(tmp_path).encode() not in chunk.data
    assert cleaned == ["r1"]


@pytest.mark.parametrize(("model", "modality", "name", "fault"), [
    ("base", "video", "garbage.mp4", _no_ffmpeg_at_import),
    ("chatterbox", "audio", "garbage.m4a", _no_ffmpeg_at_import),
    ("base", "audio", "garbage.m4a", _no_ffmpeg_at_decode),
    ("vjepa2", "video", "garbage.mp4", _no_ffmpeg_at_decode),
    ("base", "image", "image.png", _allocation_fails),
    ("base", "audio", "garbage.m4a", _allocation_fails),
    ("chatterbox", "audio", "garbage.m4a", _allocation_fails),
    ("base", "video", "garbage.mp4", _allocation_fails),
    ("base", "video", "removed.mp4", _upload_removed),
])
def test_a_server_fault_while_decoding_an_upload_is_a_500(tmp_path, monkeypatch, model, modality, name, fault):
    loader = MODELS[model]()
    path = _path(tmp_path, name)
    fault(monkeypatch, tmp_path)
    chunk, _ = _preprocess_failure(loader, {modality: [path]})
    assert chunk.metadata["status"] == 500, chunk.data
