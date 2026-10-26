"""CPU checks for the PersonaPlex benchmark integration, not model parity."""

import asyncio
import base64
import hashlib
import json
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
    reused = rows[:1] + [_ownership("after_decoder_inputs_consumed", 0, 10.5)] + rows[1:2]
    with pytest.raises(AssertionError, match="observed twice without a hard cleanup"):
        verify_ownership_observations(reused, completed=2)


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

    from mstar.communication.arena import ArenaShmCommunicationManager
    from mstar.communication.tensors import TensorCommunicationManager
    from mstar.graph.runtime.python import PythonGraphRuntime

    class StubRustRuntime:
        def cleanup_consumed_inputs(self, node_name, rids, wg_ids):
            return [], []

    rust = types.ModuleType("mstar.graph.runtime.rust")
    rust.RustGraphRuntime = StubRustRuntime
    monkeypatch.setitem(sys.modules, "mstar.graph.runtime.rust", rust)
    # The observer rebinds these class attributes; monkeypatch restores them.
    for cls, name in (
        (TensorCommunicationManager, "cleanup_collectable"),
        (TensorCommunicationManager, "force_cleanup_request"),
        (PythonGraphRuntime, "cleanup_consumed_inputs"),
    ):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    arena_calls = []
    monkeypatch.setattr(ArenaShmCommunicationManager, "cleanup_collectable",
                        lambda self, collectable, registered: arena_calls.append(collectable))
    monkeypatch.setenv("PERSONAPLEX_OWNERSHIP_OBSERVE", "1")
    monkeypatch.setenv("PERSONAPLEX_OWNERSHIP_ROOT", str(tmp_path))
    monkeypatch.setenv("PERSONAPLEX_LIVE_BACKEND", "rust")
    runpy.run_path(str(Path("benchmark/personaplex/observe/sitecustomize.py")))

    store = types.SimpleNamespace(
        get_all_uuids=lambda rid: [5], get_info=lambda uuid: None,
        bookkeeping=types.SimpleNamespace(_ref_info={}), _uuid_to_rid={5: 7},
    )
    manager = types.SimpleNamespace(tensor_store=store, uuid_to_edge_name={}, my_entity_id="worker_0")
    StubRustRuntime().cleanup_consumed_inputs("MimiDecoder", [7], [0])
    ArenaShmCommunicationManager.cleanup_collectable(manager, [5], [True])

    assert arena_calls == [[5]]
    rows = [json.loads(line) for path in tmp_path.glob("ownership-events-*.jsonl")
            for line in path.read_text().splitlines()]
    assert [(row["event"], row["rid"]) for row in rows] == [("after_decoder_inputs_consumed", 7)]


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
