"""CPU checks for the PersonaPlex benchmark integration, not model parity."""

import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import wave

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
