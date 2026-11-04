"""CPU checks for the PersonaPlex benchmark integration, not model parity."""

import asyncio
import base64
import hashlib
import json
import socket
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmark.base import ModelType, PersonaPlex, RequestType, Status
from benchmark.personaplex.dataset import LocalAudioDataset
from benchmark.personaplex.serving import verify_ownership_observations
from benchmark.request import OurSystem, aggregate_metrics


def _manifest(tmp_path):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(bytes(48000))
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps([{"id": "english", "path": "input.wav", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])
    )
    return manifest


def test_personaplex_local_manifest_uses_persona_and_checks_digest(tmp_path):
    model = ModelType.PERSONAPLEX.inst()
    assert model.get_tokenizer() is None and model.get_supported_modalities() == {RequestType.A2S}
    assert model.get_model_kwargs(RequestType.A2S) == {"voice": "NATF2", "greedy": True}
    manifest = _manifest(tmp_path)
    dataset = LocalAudioDataset(manifest, 3, RequestType.A2S)
    assert len(dataset.get_requests()) == 3
    assert all(r.prompt == PersonaPlex.DEFAULT_PERSONA for r in dataset.get_requests())
    (tmp_path / "input.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        LocalAudioDataset(manifest, 1, RequestType.A2S)


def test_personaplex_actual_benchmark_adapter_counts_pcm_and_rtf(tmp_path):
    """Real multipart request/NDJSON response; fake PCM, no worker/model claim."""
    import aiohttp
    from aiohttp import web

    async def exercise():
        async def generate(request):
            data = await request.post()
            assert data["text"] == PersonaPlex.DEFAULT_PERSONA
            assert data["output_modalities"] == "audio"
            assert "audio" in data["input_modalities"].split(",")
            assert json.loads(data["model_kwargs"])["greedy"] is True
            assert data["files"].file.read()
            return web.Response(
                text=json.dumps(
                    {
                        "modality": "audio",
                        "data": base64.b64encode(bytes(48000)).decode(),
                        "metadata": {"sample_rate": 24000, "num_channels": 1},
                    }
                )
                + "\n",
                content_type="application/x-ndjson",
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            req = LocalAudioDataset(_manifest(tmp_path), 1, RequestType.A2S)[0]
            async with aiohttp.ClientSession() as session:
                metric = await OurSystem().send_request(session, req, f"http://127.0.0.1:{port}", 1, PersonaPlex())
            assert metric.status == Status.SUCCESS and metric.output_bytes["audio"] == 48000
            agg = aggregate_metrics([metric], wall_time=2, model=PersonaPlex())
            assert agg.audio_duration_mean_s == 1
            assert agg.audio_seconds_throughput == 0.5
            assert agg.rtf.mean == metric.e2e_latency
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_failing_saved_comparison_reports_fail_and_nonzero(tmp_path):
    agent = np.zeros((1, 8, 2), dtype=np.int64)
    changed = agent.copy()
    changed[0, 0, 1] = 1
    np.savez(
        tmp_path / "pipeline.npz",
        mstar_agent=changed,
        nvidia_agent=agent,
        mstar_text=np.zeros((2, 1)),
        nvidia_text=np.zeros((2, 1)),
        mstar_pcm=np.zeros(20),
        nvidia_pcm=np.zeros(20),
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"checks": {"output_frame_mismatches": [1], "user_codes": {"different": 0}}})
    )
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.personaplex.compare", str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "FAIL"


def test_matching_comparison_fails_when_its_negative_control_was_not_caught(tmp_path):
    from benchmark.personaplex.compare import audit

    agent = np.zeros((1, 8, 2), dtype=np.int64)
    np.savez(
        tmp_path / "pipeline.npz",
        mstar_agent=agent,
        nvidia_agent=agent,
        mstar_text=np.zeros((2, 1)),
        nvidia_text=np.zeros((2, 1)),
        mstar_pcm=np.zeros(20),
        nvidia_pcm=np.zeros(20),
    )
    checks = {"output_frame_mismatches": [], "user_codes": {"different": 0}}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"checks": checks}))
    assert audit(tmp_path)["status"] == "PASS"
    for detected, status in ((True, "PASS"), (False, "FAIL")):
        summary.write_text(json.dumps({"checks": {**checks, "negative_control": {"detected": detected}}}))
        result = audit(tmp_path)
        assert result["status"] == status and result["negative_control"] == {"detected": detected}


def test_negative_control_rope_rotates_split_halves_in_both_layouts():
    import math

    import torch

    from benchmark.personaplex.controls import HalfSplitRope

    q, k = torch.randn(2, 1, 4, 8), torch.randn(2, 1, 4, 8)
    control = HalfSplitRope(num_layers=1, rope_theta=10000.0)
    control.offset = 5
    angles = 5 * torch.exp(torch.arange(4) * (-math.log(10000.0) * 2 / 8))
    cos, sin = torch.cos(angles), torch.sin(angles)

    def rotate_half(x):
        x1, x2 = x[..., :4], x[..., 4:]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    for got, x in zip(control.rope(q, k), (q, k), strict=True):
        torch.testing.assert_close(got, rotate_half(x))
    for got, x in zip(control.rope(q[:, 0], k[:, 0]), (q, k), strict=True):  # token-major
        torch.testing.assert_close(got, rotate_half(x)[:, 0])


def test_negative_control_changes_the_model_output():
    """Unlike a uniform position offset, which RoPE's relative form leaves exactly invariant."""
    import torch

    from benchmark.personaplex.controls import HalfSplitRope
    from mstar.model.personaplex import config as cfg
    from mstar.model.personaplex.components.temporal import DenseTemporalCache
    from mstar.model.personaplex.weights import PersonaPlexLM

    torch.manual_seed(0)
    tiny = cfg.LMArchConfig(dim=32, num_heads=2, num_layers=2, ffn_hidden=40, depformer_dim=16, depformer_heads=2,
                            depformer_layers=2, depformer_ffn_hidden=24)
    temporal = PersonaPlexLM(tiny).eval().temporal
    x = torch.randn(4, 1, tiny.dim)
    hidden = {}
    with torch.no_grad():
        for name, backend in (("served", DenseTemporalCache(2, 10000.0)), ("control", HalfSplitRope(2, 10000.0))):
            for t in range(4):
                hidden[name], _ = temporal.forward_embeddings(x[t], backend)
                backend.advance()
    assert (hidden["served"] - hidden["control"]).abs().max() > 1e-2


def test_first_reference_decision_skips_forced_choices_and_reports_both_margins():
    import torch

    from benchmark.personaplex.controls import first_reference_decision

    def record(phase, text, depth, text_forced=False):
        depth_logits = torch.zeros(1, 16, 4)
        for d, token in depth.items():
            depth_logits[0, d, token] = 2.0
        return {"phase": phase, "text_logits": torch.tensor([[[[0.0, 3.0, 1.0, 0.0]]]]) if text == 1
                else torch.tensor([[[[0.0, 1.0, 3.0, 0.0]]]]),
                "provided": torch.tensor([[[text_forced]]]), "depth_logits": depth_logits,
                "depth_provided": torch.zeros(1, 16, dtype=torch.bool)}

    default = [record("conditioning", 1, {}), record("generation", 1, {}, text_forced=True),
               record("generation", 1, {3: 1})]
    math = [record("conditioning", 2, {}), record("generation", 2, {}, text_forced=True),
            record("generation", 1, {3: 2})]
    found = first_reference_decision(default, math)
    assert (found["record"], found["kind"], found["depth_step"]) == (2, "depth", 3)
    assert found["default"]["tokens"][0] == 1 and found["math"]["tokens"][0] == 2
    assert found["default"]["margin"] == found["math"]["margin"] == 2.0
    assert first_reference_decision(default, default) is None


def test_arena_lifecycle_cannot_pass_without_precleanup_decoder_observations():
    with pytest.raises(AssertionError, match="Missing decoder observations"):
        verify_ownership_observations([], completed=1)
    rows = [
        {"event": "after_decoder_inputs_consumed", "rid": 1, "time": 10, "tensors": []},
        {"event": "before_hard_cleanup", "rid": 1, "time": 11, "tensors": []},
    ]
    verify_ownership_observations(rows, completed=1)
    rows[0]["time"] = 12
    with pytest.raises(AssertionError, match="has no later hard cleanup"):
        verify_ownership_observations(rows, completed=1)


def _ownership(event, rid, time, dims=(), entity="worker_0", name=None):
    return {"event": event, "rid": rid, "time": time, "entity": entity,
            "tensors": [{"name": name, "dims": list(d)} for d in dims]}


def test_ownership_gate_pairs_recycled_handles_with_their_own_cleanup():
    # Handle 0 serves three requests in turn; the API process logs request UUIDs.
    rows = [
        _ownership("after_decoder_inputs_consumed", 0, 10, [(1, 8, 16), (26,)]),
        _ownership("before_hard_cleanup", 0, 11),
        _ownership("before_hard_cleanup", "9f1c-uuid", 11.5, entity="api_server_preprocess_worker"),
        _ownership("before_hard_cleanup", 0, 20),  # a cancelled request: no decoder observation
        _ownership("after_decoder_inputs_consumed", 0, 30, [(1, 8, 16), (1,)]),
        _ownership("before_hard_cleanup", 0, 31),
    ]
    verify_ownership_observations(rows, completed=2)


def test_ownership_gate_allows_frames_in_flight_until_the_last_decoder_step():
    """A streamed request's earlier chunks leave the LM's newest frame in flight; its last must not."""
    streaming = [
        _ownership("after_decoder_inputs_consumed", 0, 10, [(1, 8, 16), (1, 8, 1)]),
        _ownership("after_decoder_inputs_consumed", 0, 11, [(1, 8, 16), (1, 8, 1)]),
        _ownership("after_decoder_inputs_consumed", 0, 12, [(1, 8, 16)]),
        _ownership("before_hard_cleanup", 0, 13),
    ]
    verify_ownership_observations(streaming, completed=1)
    leaked = streaming[:2] + streaming[3:]
    with pytest.raises(AssertionError, match="still hold agent frames"):
        verify_ownership_observations(leaked, completed=1)


def test_ownership_gate_needs_decoder_evidence_for_every_completed_request():
    """Two completions, but decoder observations (several chunks) for only one request's lifetime."""
    rows = [
        _ownership("after_decoder_inputs_consumed", 0, 10, [(1, 8, 16)]),
        _ownership("after_decoder_inputs_consumed", 0, 11, [(1, 8, 16)]),
        _ownership("before_hard_cleanup", 0, 12),
        _ownership("before_hard_cleanup", 1, 13, [(1, 8, 1)]),
    ]
    with pytest.raises(AssertionError, match="Missing decoder observations: 1 request lifetimes for 2"):
        verify_ownership_observations(rows, completed=2)
    verify_ownership_observations(rows, completed=1)


def test_ownership_gate_recognises_unnamed_agent_frames_by_shape():
    rows = [_ownership("after_decoder_inputs_consumed", 0, 10, [(1, 8, 16), (1, 8, 1)]),
            _ownership("before_hard_cleanup", 0, 11)]
    with pytest.raises(AssertionError, match="still hold agent frames"):
        verify_ownership_observations(rows, completed=1)
    rows[0]["tensors"] = [{"name": "agent_frame", "dims": None}]
    with pytest.raises(AssertionError, match="still hold agent frames"):
        verify_ownership_observations(rows, completed=1)


def test_rust_ownership_observer_fires_through_shm_arena_cleanup(tmp_path, monkeypatch):
    import runpy
    import types
    from pathlib import Path

    from mstar.api_server.entrypoint import APIServer
    from mstar.communication.arena import ArenaShmCommunicationManager
    from mstar.communication.tensors import TensorCommunicationManager
    from mstar.engine.engine import Engine

    class StubRustRuntime:
        def add_request(self, request_id, *args, **kwargs):
            return 7

        def cleanup_consumed_inputs(self, node_name, rids, wg_ids):
            return [], []

    rust = types.ModuleType("mstar.graph.runtime.rust")
    rust.RustGraphRuntime = StubRustRuntime
    monkeypatch.setitem(sys.modules, "mstar.graph.runtime.rust", rust)
    # The observer rebinds these class attributes; monkeypatch restores them.
    for cls, name in (
        (TensorCommunicationManager, "cleanup_collectable"),
        (TensorCommunicationManager, "force_cleanup_request"),
        (Engine, "remove_request"),
        (APIServer, "submit_request"),
    ):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    arena_calls = []
    monkeypatch.setattr(ArenaShmCommunicationManager, "cleanup_collectable",
                        lambda self, collectable, registered: arena_calls.append(collectable))
    monkeypatch.setenv("PERSONAPLEX_OWNERSHIP_OBSERVE", "1")
    monkeypatch.setenv("PERSONAPLEX_OWNERSHIP_ROOT", str(tmp_path))
    runpy.run_path(str(Path("benchmark/personaplex/observe/sitecustomize.py")))

    store = types.SimpleNamespace(
        get_all_uuids=lambda rid: [5], get_info=lambda uuid: None,
        _rust=types.SimpleNamespace(ref_count=lambda uuid: 1, stored_counts=lambda: (1, 1)),
    )
    manager = types.SimpleNamespace(tensor_store=store, uuid_to_edge_name={}, my_entity_id="worker_0")
    assert StubRustRuntime().add_request("wire-request") == 7
    StubRustRuntime().cleanup_consumed_inputs("MimiDecoder", [7], [0])
    ArenaShmCommunicationManager.cleanup_collectable(manager, [5], [True])

    assert arena_calls == [[5]]
    rows = [json.loads(line) for path in tmp_path.glob("ownership-events-*.jsonl")
            for line in path.read_text().splitlines()]
    assert [(row["event"], row["rid"]) for row in rows] == [("after_decoder_inputs_consumed", 7)]
    assert rows[0]["request_id"] == "wire-request"


def test_ownership_observer_reads_the_real_rust_tensor_store(tmp_path, monkeypatch):
    """A stub store hid that the observer read attributes the Rust store no longer has."""
    pytest.importorskip("mstar_rust")
    import runpy
    import types

    import torch

    from mstar.api_server.entrypoint import APIServer
    from mstar.communication.arena import ArenaShmCommunicationManager
    from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
    from mstar.communication.tensors import TensorCommunicationManager
    from mstar.engine.engine import Engine
    from mstar.graph.base import TensorPointerInfo
    from mstar.graph.runtime.rust import RustGraphRuntime

    for cls, name in (
        (TensorCommunicationManager, "cleanup_collectable"),
        (TensorCommunicationManager, "force_cleanup_request"),
        (ArenaShmCommunicationManager, "cleanup_collectable"),
        (RustGraphRuntime, "cleanup_consumed_inputs"),
        (RustGraphRuntime, "add_request"),
        (Engine, "remove_request"),
        (APIServer, "submit_request"),
    ):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    monkeypatch.setenv("PERSONAPLEX_OWNERSHIP_OBSERVE", "1")
    monkeypatch.setenv("PERSONAPLEX_OWNERSHIP_ROOT", str(tmp_path))
    observer = runpy.run_path(str(Path("benchmark/personaplex/observe/sitecustomize.py")))

    store = TensorStore(bookkeeping=RustTensorBookkeeping())
    info = TensorPointerInfo(dims=[1, 8, 1], dtype=torch.int64, nbytes=64, address=0, stride=[1, 1, 1], uuid=5,
                             source_session_id="s", source_entity="w0")
    store.put_tensor(7, 5, torch.zeros(1, 8, 1, dtype=torch.int64), info)
    store.increment_ref(5, n=2)
    manager = types.SimpleNamespace(tensor_store=store, uuid_to_edge_name={5: "agent_frame"}, my_entity_id="worker_0")
    observer["snapshot"](manager, "before_hard_cleanup", 7)

    (row,) = [json.loads(line) for path in tmp_path.glob("ownership-events-*.jsonl")
              for line in path.read_text().splitlines()]
    assert row["total_tensors"] == 1
    assert row["tensors"] == [{"uuid": 5, "name": "agent_frame", "refs": 2, "dims": [1, 8, 1]}]


def test_ownership_gate_correlates_http_ids_and_recycled_handles():
    rows = []
    for wire_id, start in (("first", 10), ("second", 20)):
        rows.extend([
            {**_ownership("after_decoder_inputs_consumed", 0, start), "request_id": wire_id},
            {**_ownership("before_hard_cleanup", 0, start + 1), "request_id": wire_id},
        ])
    verify_ownership_observations(rows, 2, ["first", "second"])
    # A cancelled third request cannot substitute for a missing successful HTTP request.
    with pytest.raises(AssertionError, match="Completed HTTP requests"):
        verify_ownership_observations(rows, 2, ["first", "missing"])
    rows[-1]["request_id"] = "wrong-lifetime"
    with pytest.raises(AssertionError, match="inconsistent wire"):
        verify_ownership_observations(rows, 2, ["first", "second"])


def test_engine_cleanup_gate_requires_wire_id_state_and_page_release():
    from benchmark.personaplex.serving import verify_engine_cleanup

    row = {"event": "after_engine_cleanup", "request_id": "http-id",
           "remaining_node_states": [], "reclaimable_pages": {"lm_kv": 0}}
    verify_engine_cleanup([row], ["http-id"])
    with pytest.raises(AssertionError, match="Missing engine cleanup"):
        verify_engine_cleanup([row], ["another-http-id"])
    with pytest.raises(AssertionError, match="Node state survives"):
        verify_engine_cleanup([{**row, "remaining_node_states": ["MimiDecoder"]}], ["http-id"])
    with pytest.raises(AssertionError, match="reclaimable KV"):
        verify_engine_cleanup([{**row, "reclaimable_pages": {"lm_kv": 1}}], ["http-id"])


def test_worker_ids_resolve_through_observed_api_submission_keys():
    from benchmark.personaplex.serving import correlate_request_ids

    rows = [
        {"event": "request_identity", "entity": "api_server", "request_id": "client#literal",
         "submission_key": "opaque-key-one"},
        {**_ownership("after_decoder_inputs_consumed", 0, 10), "request_id": "opaque-key-one"},
        {**_ownership("before_hard_cleanup", 0, 11), "request_id": "opaque-key-one"},
        {"event": "request_identity", "entity": "api_server", "request_id": "second",
         "submission_key": "opaque-key-two"},
        {**_ownership("after_decoder_inputs_consumed", 0, 20), "request_id": "opaque-key-two"},
        {**_ownership("before_hard_cleanup", 0, 21), "request_id": "opaque-key-two"},
    ]
    correlated = correlate_request_ids(rows)
    assert correlated[0]["request_id"] == "client#literal"
    assert correlated[0]["internal_request_id"] == "opaque-key-one"
    verify_ownership_observations(correlated, 2, ["client#literal", "second"])
    # An unrelated lifetime cannot stand in for an HTTP completion when its mapping is absent.
    with pytest.raises(AssertionError, match="Completed HTTP requests"):
        verify_ownership_observations(correlate_request_ids(rows[1:]), 2, ["client#literal", "second"])


def test_cancelled_stream_may_hold_inflight_frames_but_completed_stream_must_not():
    rows = [
        {**_ownership("after_decoder_inputs_consumed", 0, 10, [(1, 8, 1)]), "request_id": "cancelled"},
        {**_ownership("before_hard_cleanup", 0, 11), "request_id": "cancelled"},
        {**_ownership("after_decoder_inputs_consumed", 0, 20), "request_id": "completed"},
        {**_ownership("before_hard_cleanup", 0, 21), "request_id": "completed"},
    ]
    verify_ownership_observations(rows, 1, ["completed"])
    rows[2]["tensors"] = [{"dims": [1, 8, 1]}]
    with pytest.raises(AssertionError, match="still hold agent frames"):
        verify_ownership_observations(rows, 1, ["completed"])


def test_post_audio_cancellation_closes_a_real_http_stream(tmp_path):
    """Exercise the client disconnect boundary; the test server is not a worker/model."""
    import re
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    from benchmark.personaplex.serving import Lifecycle

    closed = threading.Event()
    events = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            wire_id = re.search(rb'name="request_id"\r\n\r\n([^\r]+)', body)[1].decode()
            payload = json.dumps({"modality": "audio", "data": base64.b64encode(bytes(1920)).decode(),
                                  "metadata": {"sample_rate": 24000, "num_channels": 1}}).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()
            self.connection.settimeout(5)
            assert self.connection.recv(1) == b""  # The client closed without waiting for the stream's end.
            events.append({"event": "before_hard_cleanup", "request_id": wire_id, "time": time.time()})
            closed.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    lifecycle = Lifecycle(SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", root=tmp_path), [], True)
    lifecycle.events = lambda: events
    lifecycle.drain = lambda *_: closed.wait(5)
    manifest = _manifest(tmp_path)
    entry = LocalAudioDataset(manifest, 1, RequestType.A2S).entries[0]
    try:
        lifecycle.cancel_after_audio(entry)
        assert closed.is_set()
        assert lifecycle.rows[0]["after_first_audio"] is True
        assert lifecycle.rows[0]["first_chunk_samples"] == 960
        assert events[0]["request_id"] == lifecycle.rows[0]["request_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_timing_summary_measures_frames_against_the_80ms_budget():
    from benchmark.personaplex.timing import median_runs, summarize

    out = summarize([70.0, 90.0, 80.0, 100.0, 60.0], audio_seconds=0.4, total_seconds=0.6)
    assert out["frames"] == 5 and out["frame_ms_p50"] == 80.0 and out["frame_ms_max"] == 100.0
    assert out["frames_within_80ms"] == 0.6 and out["rtf"] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        summarize([], audio_seconds=1.0, total_seconds=1.0)
    merged = median_runs([{"rtf": 1.0, "note": "a"}, {"rtf": 3.0, "note": "b"}, {"rtf": 2.0, "note": "c"}])
    assert merged["rtf"] == 2.0 and merged["repeats"] == 3 and "note" not in merged


def test_stage_deadline_is_optional_and_keeps_the_stop_reserve():
    from benchmark.personaplex.suite import stage_deadline

    assert stage_deadline(100.0, None, smoke=False) == float("inf")
    assert stage_deadline(100.0, 3600, smoke=False) == 100.0 + 3600 - 900
    assert stage_deadline(100.0, 1200, smoke=True) == 100.0 + 1200 - 120


def test_missing_english_clip_blocks_only_its_parity_cases():
    from benchmark.personaplex.suite import missing_clip_results

    rows = missing_clip_results([{"id": "existing-222"}], smoke=False)
    assert set(rows) == {"parity-english-dense", "parity-english-resources"}
    assert all(row["status"] == "BLOCKED" for row in rows.values())
    assert missing_clip_results([{"id": "existing-222"}, {"id": "english"}], smoke=False) == {}
    assert missing_clip_results([{"id": "short-control"}], smoke=True) == {}


def test_rust_extension_from_directory_or_installed_module(tmp_path, monkeypatch):
    import importlib.util
    import types

    from benchmark.personaplex.suite import rust_extension_files

    built = tmp_path / "ext"
    built.mkdir()
    (built / "mstar_rust.cpython-312-x86_64-linux-gnu.so").write_bytes(b"x")
    assert [p.name for p in rust_extension_files({"extension": str(built)})] == [
        "mstar_rust.cpython-312-x86_64-linux-gnu.so"]
    with pytest.raises(FileNotFoundError, match="No mstar_rust"):
        rust_extension_files({"extension": str(tmp_path)})
    package = tmp_path / "site" / "mstar_rust"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "mstar_rust.cpython-312-x86_64-linux-gnu.so").write_bytes(b"y")
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: types.SimpleNamespace(origin=str(package / "__init__.py")))
    assert [p.parent.name for p in rust_extension_files({})] == ["mstar_rust"]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(FileNotFoundError, match="not importable"):
        rust_extension_files({"extension": None})


def test_resume_continues_the_original_run_kind(tmp_path):
    from benchmark.personaplex.suite import run_kind

    assert run_kind(tmp_path, "smoke") == "smoke"
    assert run_kind(tmp_path, "resume") == "smoke"  # not silently upgraded to a full run
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="Existing results"):
        run_kind(tmp_path, "full")
    assert run_kind(tmp_path, "resume") == "smoke"  # the refused run did not overwrite run.json
    with pytest.raises(ValueError, match="Nothing to resume"):
        run_kind(tmp_path / "empty", "resume")


def test_finished_stages_are_reused_only_for_the_same_command():
    from benchmark.personaplex.suite import reusable

    smoke_command = ["python", "-m", "benchmark.personaplex.warm", "--smoke"]
    full_command = ["python", "-m", "benchmark.personaplex.warm"]
    passed = {"status": "PASS", "command": smoke_command}
    assert reusable(passed, smoke_command, retry_failed=False)
    assert not reusable(passed, full_command, retry_failed=False)  # a smoke PASS is not a full PASS
    failed = {"status": "FAIL", "command": full_command}
    assert reusable(failed, full_command, retry_failed=False)
    assert not reusable(failed, full_command, retry_failed=True)
    assert not reusable({"status": "BLOCKED", "reason": "x"}, full_command, retry_failed=False)


def test_model_file_identity_hashes_small_files_and_stats_large_ones(tmp_path):
    from benchmark.personaplex.suite import file_identity

    voice = tmp_path / "NATF2.pt"
    voice.write_bytes(b"voice-a")
    first = file_identity(voice)
    voice.write_bytes(b"voice-b")  # same size, replaced in place
    assert "sha256" in first and file_identity(voice) != first
    large = file_identity(voice, hash_limit=1)
    assert set(large) == {"bytes", "mtime_ns"} and large["bytes"] == 7


def test_word_error_rate_ignores_case_and_punctuation():
    from benchmark.personaplex.judge import word_error_rate

    assert word_error_rate("Hello, there!", "hello there") == 0
    assert word_error_rate("pack light for japan", "pack lights for japan") == 0.25
    assert word_error_rate("one two three", "one three") == pytest.approx(1 / 3)
    assert word_error_rate("one", "one two three") == 2
    assert word_error_rate("", "anything") is None


def test_judge_passes_only_when_every_stage_was_transcribed():
    from benchmark.personaplex.judge import summarize

    judged = {"transcripts": {"mstar": "hi", "nvidia": "hi"}, "agent_wer_mstar_vs_nvidia": 0.0}
    assert summarize({"a": judged}, "asr")["status"] == "PASS"
    blocked = summarize({"a": judged, "b": {"missing": True}}, "asr")
    assert blocked["status"] == "BLOCKED" and "b" in blocked["reason"]
    assert summarize({}, "asr")["status"] == "BLOCKED"


def test_server_close_stops_workers_whose_launcher_already_exited(tmp_path):
    """The launcher can die first; its spawned workers must still be stopped."""
    import os
    import signal
    import time

    from benchmark.personaplex.serving import Server

    child_pid = tmp_path / "child.pid"
    launcher = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys; "
         "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
         f"open({str(child_pid)!r}, 'w').write(str(c.pid))"],
        start_new_session=True,
    )
    launcher.wait(timeout=30)  # the launcher exits; its child keeps running in the same session
    pid = int(child_pid.read_text())
    os.kill(pid, 0)  # alive
    server = Server.__new__(Server)
    server.process = launcher
    try:
        server.close()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("worker outlived Server.close()")
    finally:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_sessions_playback_counts_underruns_and_shifts_later_deadlines():
    from benchmark.personaplex.sessions import playback

    rate = 24000
    on_time = [(1.0, rate), (1.5, rate), (2.0, 0)]  # each second of audio arrives before it is needed
    assert playback(on_time, rate=rate, buffer_s=0.5) == {"underruns": 0, "stall_s": 0.0}
    late = [(1.0, rate), (3.0, rate), (3.2, rate)]  # needed at 2.5, arrives 3.0; playback resumes from 3.0
    result = playback(late, rate=rate, buffer_s=0.5)
    assert result["underruns"] == 1 and abs(result["stall_s"] - 0.5) < 1e-9
    assert playback([], rate=rate) == {"underruns": 0, "stall_s": 0.0}


def test_sessions_onset_is_found_within_chunks_and_respects_overlap():
    from benchmark.personaplex.sessions import response_onset, speech_segments

    rate = 24000
    track = np.zeros(rate * 6, dtype=np.float32)
    track[int(3.37 * rate) : int(4.5 * rate)] = 0.3  # starts mid-way through a 320 ms chunk
    segments = speech_segments(track, rate)
    assert len(segments) == 1 and abs(segments[0][0] - 3.36) < 0.021
    onset = response_onset(segments, speech_end=3.0)
    assert not onset["already_speaking_at_speech_end"] and abs(onset["onset_s"] - 0.36) < 0.021
    # The agent began before the user finished: a negative onset, not ~0.
    early = response_onset([(2.6, 4.0)], speech_end=3.0)
    assert abs(early["onset_s"] + 0.4) < 1e-9 and not early["already_speaking_at_speech_end"]
    # Talking long before speech end and still going: reported as such, and the next segment is the onset.
    running = response_onset([(0.5, 3.5), (4.2, 5.0)], speech_end=3.0)
    assert running["already_speaking_at_speech_end"] and abs(running["onset_s"] - 1.2) < 1e-9
    assert speech_segments(np.zeros(100, dtype=np.float32), rate) == []
    blip = np.zeros(rate, dtype=np.float32)
    blip[: int(0.05 * rate)] = 0.5  # shorter than the minimum voiced duration
    assert speech_segments(blip, rate) == []


def test_sessions_detector_finds_speech_in_tracks_with_little_or_no_silence():
    """A mostly voiced reply has no quiet percentile to calibrate on; the threshold must not need one."""
    from benchmark.personaplex.sessions import speech_segments

    rate = 24000
    constant = np.full(rate, 0.3, dtype=np.float32)
    assert speech_segments(constant, rate) == [(0.0, 1.0)]
    mostly = np.full(rate * 2, 0.3, dtype=np.float32)
    mostly[: int(0.2 * rate)] = 0.002  # 10% idle at PersonaPlex's measured idle level
    segments = speech_segments(mostly, rate)
    assert len(segments) == 1 and abs(segments[0][0] - 0.2) < 0.021
    assert speech_segments(np.full(rate, 0.008, dtype=np.float32), rate) == []  # idle noise stays silent


def test_sessions_cell_is_invalid_and_exits_nonzero_when_a_session_fails(tmp_path, monkeypatch):
    from benchmark.personaplex import sessions

    clip = tmp_path / "user.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(np.zeros(4800, dtype="<i2").tobytes())

    def fake_session(url, audio, voice, timeout, out):
        failing = len(calls) == 1
        calls.append(1)
        samples = np.zeros(4800 if not failing else 1920, dtype=np.float32)
        out.update(wall_s=0.1, arrivals=[(0.05, len(samples))], pcm=samples)
        if failing:
            out["error"] = "RuntimeError('stream cut')"

    calls = []
    monkeypatch.setattr(sessions, "run_session", fake_session)
    cell = sessions.run_cell("http://x", str(clip), 2, 0.1, "NATF2", 0.32, 60)
    assert cell["completed"] == 1 and not cell["valid"]
    monkeypatch.setattr(sys, "argv", ["sessions", "--url", "http://x", "--audio", str(clip), "--speech-end", "0.1",
                                      "--sessions", "2", "--warmup", "0", "--output", str(tmp_path / "out.json")])
    calls.clear()
    assert sessions.main() == 1
    assert json.loads((tmp_path / "out.json").read_text())["cells"][0]["valid"] is False


def test_serving_config_override_copies_the_yaml_with_one_batch_size(tmp_path):
    import yaml

    from benchmark.personaplex.serving import server_config

    source = tmp_path / "personaplex.yaml"
    source.write_text("model: personaplex\nmodel_kwargs:\n  voice_dir: /voices\n  max_batch_size: 8\n")
    settings = {"server_config": str(source)}
    assert server_config(settings, tmp_path, None) == str(source)
    derived = yaml.safe_load(Path(server_config(settings, tmp_path, 1)).read_text())
    assert derived["model_kwargs"] == {"voice_dir": "/voices", "max_batch_size": 1}
    assert yaml.safe_load(source.read_text())["model_kwargs"]["max_batch_size"] == 8


def test_judge_is_blocked_without_an_asr_model():
    from benchmark.personaplex.suite import judge_blocked

    assert judge_blocked({"asr_model": "/models/whisper"}) == {}
    assert judge_blocked({})["judge"]["status"] == "BLOCKED"


def _session_row(ttfa, stall, complete=True, samples=24000):
    return {"complete": complete, "time_to_first_audio_s": ttfa, "stall_s": stall, "samples": samples}


def test_goodput_counts_only_sessions_meeting_both_limits():
    from benchmark.personaplex.sessions import slo_summary

    rows = [
        _session_row(0.1, 0.0),  # meets both
        _session_row(0.1, 0.01),  # stall at the limit misses it
        _session_row(0.2, 0.0),  # first audio at the limit misses it
        _session_row(0.1, 0.0, complete=False),  # an incomplete session never counts
    ]
    summary = slo_summary(rows, wall=2.0)
    assert (summary["met_slo"], summary["met_stall"], summary["met_ttfa"]) == (1, 2, 2)
    assert summary["goodput_sessions_per_s"] == 0.5
    assert summary["goodput_audio_s_per_s"] == 0.5
    assert summary["stall_only_goodput_audio_s_per_s"] == 1.0
    assert summary["ttfa_s"]["max"] == 0.2  # latencies of completed sessions only


def test_goodput_of_a_cell_with_no_completed_session_is_zero():
    from benchmark.personaplex.sessions import slo_summary

    summary = slo_summary([_session_row(None, 0.0, complete=False, samples=0)], wall=1.0)
    assert summary["met_slo"] == 0 and summary["goodput_audio_s_per_s"] == 0
    assert summary["ttfa_s"]["p50"] is None


def test_sessions_split_evenly_over_client_processes():
    from benchmark.personaplex.sessions import split_sessions

    assert split_sessions(1, 1) == [1]
    assert split_sessions(1, 4) == [1]
    assert split_sessions(128, 8) == [16] * 8
    assert split_sessions(10, 4) == [3, 3, 2, 2]


def test_client_processes_return_every_session(tmp_path):
    """Real spawned clients against a closed port: each session comes back recording its failure."""
    from benchmark.personaplex.sessions import run_sessions

    clip = tmp_path / "user.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(np.zeros(1920, dtype="<i2").tobytes())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    results, wall = run_sessions(f"http://127.0.0.1:{port}", str(clip), "NATF2", 30, 3, client_procs=2)
    assert len(results) == 3 and wall > 0
    assert all("error" in row and len(row["pcm"]) == 0 for row in results)


def test_sweep_yaml_sizes_the_pool_for_the_clip_and_lifts_the_full_length_cap(tmp_path):
    import yaml

    from benchmark.personaplex.sweep import sweep_config
    from mstar.model.personaplex.personaplex_model import PersonaPlexModel

    path = sweep_config("configs/personaplex.yaml", 128, 222, tmp_path / "config.yaml")
    config = yaml.safe_load(Path(path).read_text())
    assert config["model_kwargs"]["max_batch_size"] == 128
    assert "max_concurrent_requests" not in config
    # (222 + 128) forwards = 3 pages a request, plus the sink page
    assert config["resources"]["lm_kv"] == {"page_size": 128, "max_num_pages": 385}
    PersonaPlexModel(skip_weight_loading=True).validate_config_yaml(config, path)


def _wave(valid=True, audio=10.0, requests=1.0, met=1, sessions=1, ttfa=0.1, stall=0.0):
    return {"valid": valid, "sessions": sessions, "completed": sessions, "audio_seconds_throughput": audio,
            "requests_per_s": requests, "goodput_sessions_per_s": requests, "goodput_audio_s_per_s": audio,
            "stall_only_goodput_audio_s_per_s": audio, "met_slo": met, "met_stall": met, "met_ttfa": met,
            "rows": [_session_row(ttfa, stall) for _ in range(sessions)]}


def test_sweep_row_takes_the_median_wave_and_pools_latencies():
    from benchmark.personaplex.sweep import aggregate

    cell = aggregate(2, 30.0, [_wave(audio=1.0, sessions=2, met=2), _wave(audio=9.0, sessions=2, met=2, ttfa=0.5),
                               _wave(audio=2.0, sessions=2, met=1)])
    assert cell["status"] == "PASS" and cell["audio_s_per_s"] == 2.0
    assert cell["sessions"] == 6 and cell["met_slo"] == 5
    assert cell["ttfa_s"]["max"] == 0.5
    assert aggregate(2, 30.0, [_wave(), _wave(valid=False)])["status"] == "FAIL"


def test_sweep_summary_reports_the_highest_concurrency_where_every_session_met_the_slo():
    from benchmark.personaplex.sweep import aggregate, summarize

    cells = [aggregate(1, 1.0, [_wave()]), aggregate(4, 1.0, [_wave(sessions=4, met=4)]),
             aggregate(8, 1.0, [_wave(sessions=8, met=7)]), {"concurrency": 16, "status": "FAIL", "error": "oom"},
             {"concurrency": 32, "status": "NOT RUN", "reason": "Stopped"}]
    assert summarize(cells)["highest_concurrency_meeting_slo"] == 4
    assert summarize(cells[3:])["highest_concurrency_meeting_slo"] is None


def test_sweep_stops_at_a_failed_cell_and_exits_nonzero(tmp_path, monkeypatch):
    from benchmark.personaplex import sweep

    clip = tmp_path / "user.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(np.zeros(1920, dtype="<i2").tobytes())
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"server_config": "configs/personaplex.yaml"}))

    def fake_cell(settings, args, concurrency, clip_frames):
        if concurrency == 4:
            raise RuntimeError("Server exited 1; inspect server.log")
        return sweep.aggregate(concurrency, 1.0, [_wave()])

    monkeypatch.setattr(sweep, "run_concurrency", fake_cell)
    monkeypatch.setattr(sys, "argv", ["sweep", "--settings", str(settings), "--audio", str(clip),
                                      "--sessions", "1", "4", "8", "--boot-rounds", "0",
                                      "--output", str(tmp_path / "out")])
    assert sweep.main() == 1
    cells = json.loads((tmp_path / "out" / "sweep.json").read_text())["cells"]
    assert [(c["concurrency"], c["status"]) for c in cells] == [(1, "PASS"), (4, "FAIL"), (8, "NOT RUN")]
    assert "Server exited 1" in cells[1]["error"]
    assert "concurrency" in (tmp_path / "out" / "sweep.csv").read_text()


def test_time_to_first_audio_splits_at_the_first_text_event():
    from benchmark.personaplex.sessions import ttfa_split

    assert ttfa_split(2.5, 2.0) == {"setup_s": 2.0, "first_audio_after_setup_s": 0.5}
    # No claim where the marker is missing or came after the audio.
    none = {"setup_s": None, "first_audio_after_setup_s": None}
    assert ttfa_split(2.0, 2.5) == none and ttfa_split(2.0, None) == none and ttfa_split(None, 1.0) == none


def test_a_session_records_its_first_text_event_even_when_empty(monkeypatch):
    from benchmark.personaplex import sessions
    from mstar.client import client as client_module
    from mstar.client.types import AudioChunk, TextChunk

    class FakeClient:
        def __init__(self, url, timeout):
            pass

        def generate(self, **kwargs):
            yield TextChunk("", {})  # a padding token's frame
            yield TextChunk("hi", {})
            yield AudioChunk(np.zeros(1920, dtype="<i2").tobytes())

    monkeypatch.setattr(client_module, "MStarClient", FakeClient)
    out = {}
    sessions.run_session("http://x", "clip.wav", "NATF2", 60, out)
    assert "error" not in out
    assert 0 <= out["first_text_s"] <= out["arrivals"][0][0]
    assert out["arrivals"][0][1] == 1920
