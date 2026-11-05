"""CPU checks for the PersonaPlex benchmark integration, not model parity."""

import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from benchmark.base import ModelType, PersonaPlex, RequestType, Status
from benchmark.personaplex.dataset import LocalAudioDataset
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


def test_post_audio_cancellation_closes_a_real_http_stream(tmp_path):
    """Exercise the client disconnect boundary; the test server is not a worker/model."""
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    from benchmark.personaplex.serving import Lifecycle

    closed = threading.Event()
    seen = []

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
            seen.append(wire_id)
            closed.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    lifecycle = Lifecycle(SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", root=tmp_path), [], True)
    manifest = _manifest(tmp_path)
    entry = LocalAudioDataset(manifest, 1, RequestType.A2S).entries[0]
    try:
        lifecycle.cancel_after_audio(entry)
        assert closed.wait(5)
        assert lifecycle.rows[0]["after_first_audio"] is True
        assert lifecycle.rows[0]["first_chunk_samples"] == 960
        assert seen == [lifecycle.rows[0]["request_id"]]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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


class _FakeClient:
    def __init__(self, alive=False, exitcode=1):
        self.alive, self.exitcode, self.stopped = alive, exitcode, []

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.stopped.append("terminate")
        self.alive = False

    def kill(self):
        self.stopped.append("kill")
        self.alive = False


