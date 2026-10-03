"""Zonos2 serving path against the real checkpoint: one colocated server, real requests.

Skips without CUDA, ``ZONOS2_CACHE_DIR`` (env or repo ``.env``), or a cached Whisper
(large-v3-turbo, else small.en). Audio is judged by Whisper round-trip, not by shape.
Boot takes several minutes; run it inside an allocation.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

requests = pytest.importorskip("requests")
jiwer = pytest.importorskip("jiwer")

REPO = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 44100
BOOT_TIMEOUT_S = 20 * 60


def _cache_dir() -> str | None:
    if os.environ.get("ZONOS2_CACHE_DIR"):
        return os.environ["ZONOS2_CACHE_DIR"]
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ZONOS2_CACHE_DIR":
                return value.strip()
    return None


CACHE_DIR = _cache_dir()
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not CACHE_DIR or not Path(CACHE_DIR).is_dir(), reason="ZONOS2_CACHE_DIR not set",
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="zonos2_it_")
    env = dict(os.environ)
    env.setdefault("HF_MODULES_CACHE", os.path.join(CACHE_DIR, "hf_modules"))
    log = open(os.path.join(tmp, "server.log"), "w")
    proc = subprocess.Popen(
        [
            sys.executable, "mstar/api_server/entrypoint.py",
            "--config", "configs/zonos2_colocated.yaml",
            "--cache-dir", CACHE_DIR,
            "--socket-path-prefix", f"{tmp}/sock/",
            "--upload-dir", f"{tmp}/uploads/",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--mooncake-port", str(_free_port()),
            "--tensor-comm-protocol", "SHM",
        ],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + BOOT_TIMEOUT_S
    while True:
        if proc.poll() is not None:
            pytest.fail(f"server exited with {proc.returncode}; see {log.name}")
        try:
            if requests.get(f"{url}/health", timeout=5).ok:
                break
        except requests.ConnectionError:
            pass
        if time.time() > deadline:
            proc.kill()
            pytest.fail(f"server not ready after {BOOT_TIMEOUT_S} s; see {log.name}")
        time.sleep(5)
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def _wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def tts(url, text, ref_wav: bytes | None = None, **model_kwargs) -> tuple[bytes, int]:
    """Stream one request; return (pcm, status), with any in-band error's status."""
    data = {
        "text": text, "output_modalities": "audio",
        "model_kwargs": json.dumps(model_kwargs),
    }
    files = None
    if ref_wav is not None:
        data["input_modalities"] = "audio,text"
        files = {"files": ("ref.wav", ref_wav, "application/octet-stream")}
    pcm, status = b"", 200
    with requests.post(f"{url}/generate", data=data, files=files, stream=True,
                       timeout=600) as resp:
        if resp.status_code != 200:
            return b"", resp.status_code
        for line in resp.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("modality") == "error":
                status = int((msg.get("metadata") or {}).get("status", 500))
            elif msg.get("modality") == "audio" and msg.get("data"):
                pcm += base64.b64decode(msg["data"])
    return pcm, status


@pytest.fixture(scope="module")
def asr():
    from transformers import pipeline

    for name in ("openai/whisper-large-v3-turbo", "openai/whisper-small.en"):
        try:
            pipe = pipeline("automatic-speech-recognition", model=name, device="cuda",
                            model_kwargs={"local_files_only": True})
            break
        except OSError:
            continue
    else:
        pytest.skip("no Whisper checkpoint in the local HF cache")

    def transcribe(pcm: bytes) -> str:
        import torchaudio.functional as AF

        wav = torch.frombuffer(bytearray(pcm), dtype=torch.int16).float() / 32768.0
        wav = AF.resample(wav, SAMPLE_RATE, 16000)
        return pipe({"raw": wav.numpy(), "sampling_rate": 16000})["text"]

    return transcribe


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def _wer(ref: str, hyp: str) -> float:
    return jiwer.wer(_norm(ref), _norm(hyp))


def _corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Pooled WER; one misheard word in one sentence must not fail a test."""
    return jiwer.wer([_norm(r) for r in refs], [_norm(h) for h in hyps])


SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
]


def test_text_round_trips_through_asr(server, asr):
    hyps = []
    for text in SENTENCES:
        pcm, status = tts(server, text)
        assert status == 200 and pcm, text
        hyps.append(asr(pcm))
    assert _corpus_wer(SENTENCES, hyps) <= 0.15, hyps


def test_concurrent_requests_each_speak_their_own_text(server, asr):
    with ThreadPoolExecutor(len(SENTENCES)) as pool:
        results = list(pool.map(lambda t: tts(server, t), SENTENCES))
    hyps = []
    for text, (pcm, status) in zip(SENTENCES, results, strict=True):
        assert status == 200 and pcm, text
        hyps.append(asr(pcm))
        assert _wer(text, hyps[-1]) <= 0.5, (text, hyps[-1])  # crossed rows score ~1
    assert _corpus_wer(SENTENCES, hyps) <= 0.15, hyps


def test_clone_round_trips_through_asr(server, asr):
    ref = _wav_bytes(tts(server, SENTENCES[1])[0])
    texts = [SENTENCES[0], SENTENCES[2], SENTENCES[3]]
    hyps = []
    for text in texts:
        pcm, status = tts(server, text, ref_wav=ref)
        assert status == 200 and pcm, text
        hyps.append(asr(pcm))
    assert _corpus_wer(texts, hyps) <= 0.15, hyps


def test_one_frame_budget_returns_one_frame(server):
    pcm, status = tts(server, "Hello.", max_output_tokens=1)
    assert status == 200
    assert len(pcm) == 512 * 2  # one DAC hop of int16


def test_zero_frame_budget_is_rejected(server):
    _, status = tts(server, "Hello.", max_output_tokens=0)
    assert status == 400


def test_tiny_reference_clip_fails_alone(server):
    """A <=16 ms clip gets a 400 and does not fail the clones batched with it."""
    good = _wav_bytes(tts(server, SENTENCES[3])[0])
    tiny = _wav_bytes(b"\x00\x00" * 400)
    with ThreadPoolExecutor(5) as pool:
        futs = [pool.submit(tts, server, "Hi there.", tiny)]
        futs += [pool.submit(tts, server, "Hi there.", good) for _ in range(4)]
        results = [f.result() for f in futs]
    assert results[0][1] == 400
    for pcm, status in results[1:]:
        assert status == 200 and pcm


@pytest.fixture(scope="module")
def burst(server):
    """More requests than the 256 sampler slots per step, all at once."""
    n = 300
    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(
            lambda i: tts(server, f"Number {i}.", max_output_tokens=40), range(n),
        ))


def test_burst_over_buffer_capacity_leaves_the_engine_serving(server, burst, asr):
    # The resize bug left a dead CUDA context: every later request failed.
    pcm, status = tts(server, SENTENCES[0])
    assert status == 200 and pcm
    assert _wer(SENTENCES[0], asr(pcm)) <= 0.5


@pytest.mark.xfail(reason="shared API server: result delivery TTL (15 s) expires under a "
                          "300-stream burst although the engine finishes every request")
def test_burst_over_buffer_capacity_delivers_every_response(burst):
    assert all(status == 200 and pcm for pcm, status in burst)
