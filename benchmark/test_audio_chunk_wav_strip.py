"""Regression test: streamed WAV-container audio chunks must not splice
their RIFF headers into the accumulated PCM stream.

sglang-omni encodes every /v1/chat/completions audio delta as a *complete*
WAV file (sglang_omni/client/audio.py::audio_to_base64 defaults to
output_format="wav"). Writing those bytes verbatim into the PCM buffer
embeds a 44-byte header every chunk, which is audible as a periodic click.
"""

import base64
import io
import struct
import time
import wave

import numpy as np

from benchmark.base import RequestType
from benchmark.request import RequestMetrics

SAMPLE_RATE = 24000


def _wav_chunk(pcm: bytes) -> str:
    """Encode int16 PCM as a complete WAV file, base64'd — what sglang-omni sends."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _new_metrics() -> RequestMetrics:
    m = RequestMetrics(request_id="0", type=RequestType.T2S)
    m.start_time = time.monotonic()
    return m


def test_wav_container_chunks_are_stripped_to_pcm():
    rng = np.random.default_rng(0)
    chunks = [
        (rng.random(1000) * 2 - 1).astype(np.float32).__mul__(32767).astype(np.int16).tobytes()
        for _ in range(3)
    ]

    m = _new_metrics()
    for c in chunks:
        m.record_output_chunk(modality="audio", data_b64=_wav_chunk(c))

    got = m._audio_pcm.getvalue()
    want = b"".join(chunks)
    assert got == want, (
        f"expected {len(want)} PCM bytes, got {len(got)} "
        f"({(len(got) - len(want))} extra = spliced WAV headers)"
    )


def test_raw_pcm_chunks_pass_through_unchanged():
    """Systems that stream raw PCM (no RIFF header) must be untouched."""
    pcm = np.arange(500, dtype=np.int16).tobytes()
    m = _new_metrics()
    m.record_output_chunk(modality="audio", data_b64=base64.b64encode(pcm).decode())
    assert m._audio_pcm.getvalue() == pcm


def test_byte_metrics_exclude_wav_headers():
    """output_bytes/duration must count audio, not container overhead."""
    pcm = np.zeros(2400, dtype=np.int16).tobytes()  # 0.1 s @ 24 kHz
    m = _new_metrics()
    m.record_output_chunk(modality="audio", data_b64=_wav_chunk(pcm))
    assert m.output_bytes["audio"] == len(pcm)
