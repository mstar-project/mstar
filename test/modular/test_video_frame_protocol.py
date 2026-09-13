"""Native raw-video-frame protocol and SDK contract."""

from __future__ import annotations

import asyncio
import queue
import runpy
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")
pytest.importorskip("requests")

from mstar.api_server import entrypoint  # noqa: E402
from mstar.api_server.data_worker import (  # noqa: E402
    PreprocessWorkerThread,
    _video_frame_metadata,
)
from mstar.api_server.entrypoint import SUPPORTED_MODALITIES, APIServer  # noqa: E402
from mstar.api_server.request_types import PreprocessInput, ResultTensors  # noqa: E402
from mstar.client import MStarClient, VideoFrameChunk  # noqa: E402
from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.graph.loop_indices import NestedLoopIndices  # noqa: E402
from mstar.model.waypoint.submodules import PRIME_WALK, ROLLOUT_WALK  # noqa: E402
from mstar.model.waypoint.waypoint_model import WaypointModel  # noqa: E402


def _metadata(**overrides):
    values = {
        "width": 3,
        "height": 2,
        "fps": 60,
        "pixel_format": "rgb24",
        "frame_index": 0,
        "frame_count": 4,
    }
    values.update(overrides)
    return values


def test_sdk_video_frame_chunk_is_a_zero_copy_shaped_view():
    raw = bytes(range(4 * 2 * 3 * 3))

    event = MStarClient._to_event(
        {
            "modality": "video_frame",
            "bytes": raw,
            "metadata": _metadata(frame_index=8),
        }
    )

    assert isinstance(event, VideoFrameChunk)
    assert (event.frame_index, event.frame_count, event.fps) == (8, 4, 60.0)
    frames = event.to_numpy()
    assert frames.shape == (4, 2, 3, 3)
    assert frames.dtype == np.uint8
    assert not frames.flags.owndata
    assert not frames.flags.writeable
    assert np.shares_memory(frames, np.frombuffer(raw, dtype=np.uint8))
    assert frames[0, 0, 0].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("metadata", "data", "message"),
    [
        ({}, b"", "missing required"),
        (_metadata(), bytearray(72), "immutable bytes"),
        (_metadata(pixel_format="bgr24"), bytes(72), "pixel_format"),
        (_metadata(fps=float("nan")), bytes(72), "finite positive"),
        (_metadata(frame_index=-1), bytes(72), "frame_index"),
        (_metadata(), bytes(71), "payload length"),
    ],
)
def test_sdk_video_frame_chunk_validates_metadata_and_payload(metadata, data, message):
    with pytest.raises(ValueError, match=message):
        VideoFrameChunk(data, metadata)


def test_sdk_rejects_nonstreaming_raw_frames_before_http():
    client = MStarClient("http://unused")
    with pytest.raises(ValueError, match="requires stream=True"):
        client.generate(output_modalities=("video_frame",), stream=False)


def test_api_core_rejects_nonstreaming_raw_frames_before_preprocessing():
    server = APIServer.__new__(APIServer)
    assert "video_frame" in SUPPORTED_MODALITIES

    with pytest.raises(ValueError, match="requires streaming=True"):
        server.submit_request(
            input_modalities=["image"],
            output_modalities=["video_frame"],
            streaming=False,
        )

    with pytest.raises(ValueError, match="output-only"):
        server.submit_request(
            input_modalities=["video_frame"],
            output_modalities=["text"],
            streaming=True,
        )


def test_native_endpoint_reports_nonstreaming_raw_frames_as_bad_request(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", object())

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            entrypoint.generate(
                request=SimpleNamespace(),
                text=None,
                files=None,
                input_modalities=None,
                output_modalities="video_frame",
                streaming=False,
                model_kwargs=None,
                request_id=None,
            )
        )

    assert raised.value.status_code == 400
    assert "requires streaming=true" in str(raised.value.detail)


def test_native_endpoint_rejects_raw_frames_as_input(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", object())

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            entrypoint.generate(
                request=SimpleNamespace(),
                text=None,
                files=None,
                input_modalities="video_frame",
                output_modalities="text",
                streaming=True,
                model_kwargs=None,
                request_id=None,
            )
        )

    assert raised.value.status_code == 400
    assert raised.value.detail == "'video_frame' is an output-only modality"


def test_server_metadata_is_canonical_and_counts_raw_frames():
    tensor = torch.zeros((4, 2, 3, 3), dtype=torch.uint8)
    metadata = _video_frame_metadata(
        tensor,
        fps=60,
        frame_index=12,
        metadata={"width": 999, "producer": "decoder"},
    )
    assert metadata == {
        "producer": "decoder",
        "width": 3,
        "height": 2,
        "fps": 60,
        "pixel_format": "rgb24",
        "frame_index": 12,
        "frame_count": 4,
    }


class _FrameModel:
    def postprocess(self, tensor, modality, request_kwargs=None):
        assert modality == "video_frame"
        return tensor.cpu().contiguous().numpy().tobytes()

    def get_output_frame_rate(self, modality, request_kwargs=None):
        assert modality == "video_frame"
        assert request_kwargs == {"world": "test"}
        return 60


class _ReadyTensorManager:
    def __init__(self, tensors_by_request):
        self.tensors_by_request = tensors_by_request
        self.ready = tensors_by_request
        self.dereferenced = []
        self.cleaned = []

    def get_ready_tensors(self):
        ready, self.ready = self.ready, {}
        return {
            request_id: [
                GraphEdge(
                    next_node="api",
                    name="video_frame_output",
                    tensor_info=[SimpleNamespace(uuid=name) for name in tensors],
                )
            ]
            for request_id, tensors in ready.items()
        }

    def start_read_tensors(self, request_id, graph_edges):
        pass

    def get_tensor(self, request_id, uuid):
        return self.tensors_by_request[request_id][uuid]

    def dereference(self, request_id, uuid):
        self.dereferenced.append((request_id, uuid))

    def cleanup_request(self, request_id):
        self.cleaned.append(request_id)

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass


def _set_output_order_state(worker, tensors_by_request):
    loop_indices = NestedLoopIndices(
        loop_name_order=["rollout_loop"],
        loop_indices={"rollout_loop": 0},
        wg_fwd_pass_idx=0,
    )
    worker.tensor_uuid_to_output_order_per_request = {
        request_id: {
            name: (sequence, loop_indices)
            for sequence, name in enumerate(tensors)
        }
        for request_id, tensors in tensors_by_request.items()
    }
    worker.request_next_output_sequence = {
        request_id: len(tensors)
        for request_id, tensors in tensors_by_request.items()
    }
    worker.request_next_emit_sequence = {
        request_id: 0 for request_id in tensors_by_request
    }
    worker.request_pending_output_chunks = {
        request_id: {} for request_id in tensors_by_request
    }


def test_data_worker_emits_complete_metadata_and_monotonic_frame_indices():
    tensors = {
        "first": torch.arange(72, dtype=torch.uint8).reshape(4, 2, 3, 3),
        "second": torch.arange(72, dtype=torch.uint8).reshape(4, 2, 3, 3),
    }
    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.tensor_manager = _ReadyTensorManager({"request": tensors})
    worker.model = _FrameModel()
    worker.out_queue = queue.Queue()
    worker.request_model_kwargs = {"request": {"world": "test"}}
    worker.request_output_frame_indices = {"request": 0}
    worker.tensor_uuid_to_metadata_per_request = {"request": {name: {"producer": "decoder"} for name in tensors}}
    _set_output_order_state(worker, {"request": tensors})

    assert worker._process_read_tensors() is True
    chunks = [worker.out_queue.get_nowait(), worker.out_queue.get_nowait()]

    assert [chunk.modality for chunk in chunks] == ["video_frame", "video_frame"]
    assert [chunk.metadata["frame_index"] for chunk in chunks] == [0, 4]
    assert all(chunk.metadata["frame_count"] == 4 for chunk in chunks)
    assert all(chunk.metadata["pixel_format"] == "rgb24" for chunk in chunks)
    assert all(chunk.metadata["producer"] == "decoder" for chunk in chunks)
    assert worker.request_output_frame_indices["request"] == 8
    assert worker.tensor_manager.dereferenced == [
        ("request", "first"),
        ("request", "second"),
    ]


def test_data_worker_tracks_interleaved_frame_indices_per_request():
    frame = torch.zeros((4, 2, 3, 3), dtype=torch.uint8)
    tensors = {
        "request-a": {"a-first": frame, "a-second": frame},
        "request-b": {"b-first": frame},
    }
    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.tensor_manager = _ReadyTensorManager(tensors)
    worker.tensor_manager.ready = {
        "request-a": {"a-first": frame},
        "request-b": {"b-first": frame},
    }
    worker.model = _FrameModel()
    worker.out_queue = queue.Queue()
    worker.request_model_kwargs = {
        "request-a": {"world": "test"},
        "request-b": {"world": "test"},
    }
    worker.request_output_frame_indices = {"request-a": 0, "request-b": 0}
    worker.tensor_uuid_to_metadata_per_request = {
        request_id: {name: {} for name in request_tensors} for request_id, request_tensors in tensors.items()
    }
    _set_output_order_state(worker, tensors)

    assert worker._process_read_tensors() is True
    worker.tensor_manager.ready = {"request-a": {"a-second": frame}}
    assert worker._process_read_tensors() is True

    chunks = [worker.out_queue.get_nowait() for _ in range(3)]
    assert [chunk.request_id for chunk in chunks] == [
        "request-a",
        "request-b",
        "request-a",
    ]
    assert [chunk.metadata["frame_index"] for chunk in chunks] == [0, 0, 4]
    assert worker.request_output_frame_indices == {
        "request-a": 8,
        "request-b": 4,
    }


def test_data_worker_reorders_async_completions_before_frame_emission():
    first = torch.zeros((4, 2, 3, 3), dtype=torch.uint8)
    second = torch.ones((4, 2, 3, 3), dtype=torch.uint8)
    tensors = {"request": {"first": first, "second": second}}
    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.tensor_manager = _ReadyTensorManager(tensors)
    worker.tensor_manager.ready = {}
    worker.model = _FrameModel()
    worker.out_queue = queue.Queue()
    worker.request_model_kwargs = {"request": {"world": "test"}}
    worker.request_output_frame_indices = {"request": 0}
    worker.tensor_uuid_to_metadata_per_request = {}
    worker.tensor_uuid_to_output_order_per_request = {"request": {}}
    worker.request_next_output_sequence = {"request": 0}
    worker.request_next_emit_sequence = {"request": 0}
    worker.request_pending_output_chunks = {"request": {}}

    for iteration, name in enumerate(("first", "second")):
        worker._read_result_tensor(ResultTensors(
            request_id="request",
            modality="video_frame",
            graph_edge=GraphEdge(
                next_node="api",
                name="video_frame",
                tensor_info=[SimpleNamespace(uuid=name)],
            ),
            loop_indices=NestedLoopIndices(
                loop_name_order=["rollout_loop"],
                loop_indices={"rollout_loop": iteration},
                wg_fwd_pass_idx=1,
            ),
        ))

    worker.tensor_manager.ready = {"request": {"second": second}}
    assert worker._process_read_tensors() is True
    assert worker.out_queue.empty()

    worker.tensor_manager.ready = {"request": {"first": first}}
    assert worker._process_read_tensors() is True
    chunks = [worker.out_queue.get_nowait(), worker.out_queue.get_nowait()]
    assert [chunk.metadata["frame_index"] for chunk in chunks] == [0, 4]
    assert chunks[0].data == first.numpy().tobytes()
    assert chunks[1].data == second.numpy().tobytes()


def test_data_worker_cleanup_drops_all_frame_protocol_state():
    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.tensor_manager = _ReadyTensorManager({})
    worker.tensor_uuid_to_metadata_per_request = {
        "reused": {"old": {"producer": "decoder"}},
        "other": {"keep": {}},
    }
    worker.request_model_kwargs = {
        "reused": {"world": "old"},
        "other": {"world": "keep"},
    }
    worker.request_output_frame_indices = {"reused": 24, "other": 8}
    _set_output_order_state(worker, {
        "reused": {"old": object()},
        "other": {"keep": object()},
    })

    worker._cleanup_request_state("reused")

    assert worker.tensor_manager.cleaned == ["reused"]
    assert "reused" not in worker.tensor_uuid_to_metadata_per_request
    assert "reused" not in worker.request_model_kwargs
    assert "reused" not in worker.request_output_frame_indices
    assert "reused" not in worker.tensor_uuid_to_output_order_per_request
    assert "reused" not in worker.request_next_output_sequence
    assert "reused" not in worker.request_next_emit_sequence
    assert "reused" not in worker.request_pending_output_chunks
    assert worker.request_output_frame_indices == {"other": 8}

    worker.model = SimpleNamespace(process_prompt=lambda *args, **kwargs: {})
    worker.device = "cpu"
    worker.enable_prof = False
    worker.communicator = SimpleNamespace(send=lambda *args: None)
    worker._process_input(
        PreprocessInput(
            request_id="reused",
            text=None,
            file_paths=None,
            input_modalities=["image"],
            output_modalities=["video_frame"],
            model_kwargs={"world": "new"},
        )
    )

    assert worker.request_output_frame_indices["reused"] == 0
    assert worker.request_model_kwargs["reused"] == {"world": "new"}


def test_rollout_harness_requires_exactly_four_frames_per_step_from_index_zero():
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    check_rollout = harness["_check"]

    def chunk(frame_index):
        return VideoFrameChunk(bytes(4 * 2 * 3 * 3), _metadata(frame_index=frame_index))

    valid = [chunk(0), chunk(4)]
    assert check_rollout(valid, num_steps=2, height=2, width=3) == []

    missing = check_rollout(valid[:1], num_steps=2, height=2, width=3)
    assert "expected 2 video chunks, got 1" in missing
    assert "expected exactly 8 generated frames" in missing

    bad_index = [chunk(0), chunk(5)]
    assert any(
        "frame_index" in failure
        for failure in check_rollout(
            bad_index,
            num_steps=2,
            height=2,
            width=3,
        )
    )


@pytest.mark.parametrize(
    ("name", "model_variant", "height", "width", "tokens", "checkpoint_name"),
    [
        ("360p", "waypoint-1.5-1b-360p", 360, 640, 128, "Waypoint-1.5-1B-360P"),
        ("720p", "waypoint-1.5-1b-720p", 720, 1280, 512, "Waypoint-1.5-1B"),
    ],
)
def test_rollout_harness_variant_controls_config_and_checkpoint_default(
    tmp_path,
    name,
    model_variant,
    height,
    width,
    tokens,
    checkpoint_name,
):
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    variant = harness["VARIANTS"][name]
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "model": "waypoint",
                "model_kwargs": {"variant": "stale", "compile_dit": True},
                "max_seq_len": 999,
            }
        )
    )
    output = tmp_path / "run.yaml"

    harness["_run_config"](
        base,
        variant,
        Path("custom/checkpoint"),
        Path("custom/ae"),
        output,
        worlds=2,
    )
    generated = yaml.safe_load(output.read_text())

    assert (variant.model_variant, variant.height, variant.width) == (
        model_variant,
        height,
        width,
    )
    assert variant.tokens_per_frame == tokens
    assert variant.checkpoint_dir.name == checkpoint_name
    assert generated["model_kwargs"] == {
        "variant": model_variant,
        "compile_dit": True,
        "checkpoint_dir": "custom/checkpoint",
        "ae_path": "custom/ae",
    }
    assert generated["max_seq_len"] == tokens
    assert generated["max_concurrent_requests"] == 2
    assert generated["resources"]["kv"]["num_worlds"] == 2


def test_rollout_harness_hub_config_omits_local_overrides_and_forwards_cache(tmp_path):
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    variant = harness["VARIANTS"]["360p"]
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "model": "waypoint",
                "model_kwargs": {
                    "checkpoint_dir": "stale-checkpoint",
                    "ae_path": "stale-ae",
                    "compile_dit": True,
                },
            }
        )
    )
    output = tmp_path / "run.yaml"

    harness["_run_config"](base, variant, None, None, output, worlds=2)
    generated = yaml.safe_load(output.read_text())
    command = harness["_server_command"](
        output,
        8123,
        tmp_path,
        "DEBUG",
        90.0,
        tmp_path / "hub-cache",
        True,
    )

    assert generated["model_kwargs"] == {
        "compile_dit": True,
        "variant": "waypoint-1.5-1b-360p",
    }
    assert generated["max_concurrent_requests"] == 2
    assert generated["resources"]["kv"]["num_worlds"] == 2
    assert command[command.index("--cache-dir") + 1] == str(tmp_path / "hub-cache")
    assert "--enable-nvtx" in command


@pytest.mark.parametrize("name", ["360p", "720p"])
def test_rollout_harness_resizes_seed_to_variant_with_pillow(tmp_path, name):
    image_module = pytest.importorskip("PIL.Image")
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    variant = harness["VARIANTS"][name]
    source = tmp_path / "source.jpg"
    output = tmp_path / "seed.png"
    image_module.new("RGB", (19, 11), color=(1, 2, 3)).save(source)

    harness["_seed_png"](source, variant, output)

    with image_module.open(output) as seed:
        assert seed.format == "PNG"
        assert seed.mode == "RGB"
        assert seed.size == (variant.width, variant.height)


def test_rollout_harness_submits_and_consumes_typed_sdk_stream(tmp_path):
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    frame = VideoFrameChunk(bytes(4 * 2 * 3 * 3), _metadata())

    class Client:
        kwargs = None

        def stream(self, **kwargs):
            self.kwargs = kwargs
            return iter([frame])

    client = Client()
    seed = tmp_path / "seed.png"
    seed.write_bytes(b"PNG")

    chunks = harness["_rollout"](client, seed, num_steps=1, request_id="rid", rng_seed=17)

    assert chunks == [frame]
    assert client.kwargs == {
        "images": seed,
        "input_modalities": ("image",),
        "output_modalities": ("video_frame",),
        "request_id": "rid",
        "num_steps": 1,
        "actions": [{"mouse": [-12.0, 0.0], "buttons": [0], "scroll": 0.0}],
        "seed": 17,
    }


def test_rollout_harness_concurrent_pair_uses_separate_clients_and_starts_together(tmp_path):
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    rollout_spec = harness["RolloutSpec"]
    clients = []
    active = 0
    lock = threading.Lock()
    both_active = threading.Event()

    class Client:
        def __init__(self):
            clients.append(self)

        def stream(self, **kwargs):
            def events():
                nonlocal active
                with lock:
                    active += 1
                    if active == 2:
                        both_active.set()
                assert both_active.wait(timeout=2), "the peer stream was not active"
                yield VideoFrameChunk(bytes([kwargs["seed"]]) * 72, _metadata())

            return events()

    seed = tmp_path / "seed.png"
    seed.write_bytes(b"PNG")
    results = harness["_concurrent_rollouts"](
        Client,
        seed,
        1,
        (rollout_spec("A", "rid-a", 1), rollout_spec("B", "rid-b", 2)),
    )

    assert len(clients) == 2
    assert results["A"][0].data == bytes([1]) * 72
    assert results["B"][0].data == bytes([2]) * 72


def test_rollout_harness_requires_worker_schedule_interleaving_and_cleanup_markers():
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    interleaving_failure = harness["_interleaving_failure"]
    log = "\n".join(
        [
            "DEBUG Executing: dit graph_walk=prime ('rid-a',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-a',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-b',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-a',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-b',)",
            "INFO Request cleanup complete: rid-a",
            "INFO Request cleanup complete: rid-b",
        ]
    )

    assert interleaving_failure(log, ("rid-a", "rid-b")) is None
    assert harness["_execution_count_failure"](log, ("rid-a", "rid-b"), 2) is None
    assert harness["_cleaned_request_ids"](log) == {"rid-a", "rid-b"}
    serial = "\n".join(
        [
            "DEBUG Executing: dit graph_walk=rollout ('rid-a',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-a',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-b',)",
            "DEBUG Executing: dit graph_walk=rollout ('rid-b',)",
        ]
    )
    assert "did not contain A/B/A" in interleaving_failure(serial, ("rid-a", "rid-b"))
    assert "expected 3" in harness["_execution_count_failure"](
        serial, ("rid-a", "rid-b"), 3
    )


def test_rollout_harness_parses_and_filters_memory_telemetry(monkeypatch):
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    assert harness["_parse_pss_kib"]("Rss: 12 kB\nPss: 7 kB\n") == 7
    assert harness["_parse_nvidia_smi_processes"]("101, 512\n202, 128.5\n") == [
        (101, 512.0),
        (202, 128.5),
    ]

    completed = SimpleNamespace(returncode=0, stdout="101, 512\n202, 128\n", stderr="")
    monkeypatch.setattr(harness["subprocess"], "run", lambda *args, **kwargs: completed)
    monkeypatch.setattr(harness["os"], "getpgid", lambda pid: 77 if pid == 101 else 88)

    assert harness["_process_group_gpu_mib"](77, 2) == 512.0
    with pytest.raises(RuntimeError, match="no compute process"):
        harness["_process_group_gpu_mib"](99, 2)
    with pytest.raises(ValueError, match="unusable"):
        harness["_parse_nvidia_smi_processes"]("101, N/A\n")


def test_rollout_harness_memory_plateau_excludes_warmup_and_rejects_growth():
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))
    memory_sample = harness["MemorySample"]
    wave_memory = harness["WaveMemory"]
    stable = [
        memory_sample(1.0, "quiet", 100.0, 1000.0),
        memory_sample(2.0, "quiet", 101.0, 1000.0),
        memory_sample(3.0, "quiet", 100.5, 1000.0),
    ]
    unstable = stable[:-1] + [memory_sample(3.0, "quiet", 104.0, 1000.0)]

    assert harness["_stable_memory_plateau"](stable) == (101.0, 1000.0)
    assert harness["_stable_memory_plateau"](unstable) is None

    measured = [
        wave_memory("wave-2", 400.0, 2000.0, 200.0, 1500.0),
        wave_memory("wave-3", 410.0, 2010.0, 220.0, 1510.0),
    ]
    assert harness["_bounded_memory_failures"](measured, host_growth_mib=128.0, gpu_growth_mib=64.0) == []
    failures = harness["_bounded_memory_failures"](
        measured[:-1] + [wave_memory("wave-3", 410.0, 2010.0, 329.0, 1565.0)],
        host_growth_mib=128.0,
        gpu_growth_mib=64.0,
    )
    assert len(failures) == 2


def test_rollout_harness_shutdown_signals_only_the_api_parent(monkeypatch):
    harness = runpy.run_path(str(Path(__file__).parents[1] / "waypoint" / "serve_rollout.py"))

    class Process:
        pid = 101
        signals = []

        def poll(self):
            return None

        def send_signal(self, sig):
            self.signals.append(sig)

        def wait(self, timeout):
            assert timeout == 60
            return 0

    proc = Process()
    monkeypatch.setattr("os.killpg", lambda *_: pytest.fail("normal shutdown killed the group"))

    harness["_shutdown"](proc)

    assert proc.signals == [signal.SIGINT]


def test_waypoint_has_no_encoded_video_adapter(monkeypatch):
    from mstar.api_server.openai import router
    from mstar.api_server.openai.adapters import get_adapter

    assert get_adapter("waypoint") is None
    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(model_name="waypoint"))
    _api, _model_name, _adapter, error = router._resolve("supports_videos")
    assert error.status_code == 404


def test_waypoint_emits_only_generated_raw_frame_chunks():
    model = WaypointModel(skip_weight_loading=True)
    walks = model.get_graph_walk_graphs()

    assert walks[PRIME_WALK].sections[-1].outputs == []
    edge = walks[ROLLOUT_WALK].section.sections[-1].outputs[0]
    assert edge.output_modality == "video_frame"
    assert model.get_output_frame_rate() == 60.0

    with pytest.raises(ValueError, match="video_frame"):
        model.process_prompt(
            None,
            ["image"],
            ["video"],
            tensors=None,
            num_steps=1,
            actions=[{}],
        )
