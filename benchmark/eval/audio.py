"""Audio helpers: WAV parsing (including M*'s open-ended streaming WAV),
PCM conversion, resampling, a playback-stall model and an energy VAD.

Only numpy and the standard library are needed; ``resample`` uses
torchaudio when it is importable (it is a core M* dependency).
"""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WavFormat:
    sample_rate: int
    channels: int
    bits_per_sample: int
    data_offset: int


def parse_wav_header(data: bytes) -> WavFormat | None:
    """The format of a RIFF/WAVE prefix, or None when ``data`` does not start
    with a complete header up to the ``data`` chunk.

    A streamed WAV has an unknown length, so its RIFF and data sizes are
    placeholders (0 or 0xFFFFFFFF); only the offset of the samples matters.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            if body + 16 > len(data):
                return None
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[body : body + 16])
            fmt = (rate, channels, bits)
        elif cid == b"data":
            if fmt is None:
                return None
            return WavFormat(fmt[0], fmt[1], fmt[2], body)
        pos = body + size + (size & 1)
    return None


def pcm16_to_float(pcm: bytes, channels: int = 1) -> np.ndarray:
    """Little-endian PCM16 bytes to float32 in [-1, 1], downmixed to mono."""
    usable = len(pcm) - len(pcm) % (2 * channels)
    x = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def write_wav(path: str | Path, samples: np.ndarray | bytes, sample_rate: int) -> None:
    """Mono PCM16 WAV from float samples or PCM16 bytes."""
    pcm = samples if isinstance(samples, (bytes, bytearray)) else float_to_pcm16(samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """(mono float32, sample_rate). PCM16 WAV through the standard library,
    anything else through soundfile when it is installed."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() == 2:
                return pcm16_to_float(w.readframes(w.getnframes()), w.getnchannels()), w.getframerate()
    except (wave.Error, EOFError):
        pass
    import soundfile as sf

    x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(rate)


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or x.size == 0:
        return np.asarray(x, dtype=np.float32)
    try:
        import torch
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), src_rate, dst_rate).numpy()
    except ImportError:
        n = round(len(x) * dst_rate / src_rate)
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


@dataclass(frozen=True)
class StallStats:
    count: int
    total_s: float
    longest_s: float


def playback_stalls(
    chunks: list[tuple[float, float]], prebuffer_s: float = 0.0, min_stall_s: float = 0.0
) -> StallStats:
    """Underruns of a player fed by a stream.

    ``chunks`` are ``(arrival_time_s, audio_seconds)`` in arrival order. The
    player starts ``prebuffer_s`` after the first chunk arrives and plays in
    real time; whenever its buffer runs dry before the next chunk arrives,
    that gap is a stall. Gaps shorter than ``min_stall_s`` are ignored.
    """
    if not chunks:
        return StallStats(0, 0.0, 0.0)
    play_until = chunks[0][0] + prebuffer_s
    count, total, longest = 0, 0.0, 0.0
    for arrival, dur in chunks:
        if arrival > play_until:
            gap = arrival - play_until
            if gap > min_stall_s:
                count += 1
                total += gap
                longest = max(longest, gap)
            play_until = arrival
        play_until += dur
    return StallStats(count, total, longest)


def frame_rms_db(x: np.ndarray, sample_rate: int, frame_s: float = 0.02) -> np.ndarray:
    """RMS level per frame in dBFS (floor at -120 dB)."""
    hop = max(1, int(round(frame_s * sample_rate)))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(x[: n * hop], dtype=np.float64).reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-6))).astype(np.float32)


def speech_segments(
    x: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    frame_s: float = 0.02,
    min_speech_s: float = 0.1,
    min_silence_s: float = 0.3,
) -> list[tuple[float, float]]:
    """Energy-VAD speech segments ``[(start_s, end_s), ...]``.

    Frames above ``threshold_db`` are speech; silences shorter than
    ``min_silence_s`` are bridged and segments shorter than ``min_speech_s``
    are dropped. Deterministic, which makes latency numbers reproducible; a
    neural VAD can replace it per suite.
    """
    active = frame_rms_db(x, sample_rate, frame_s) > threshold_db
    segs: list[list[float]] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segs.append([start * frame_s, i * frame_s])
            start = None
    if start is not None:
        segs.append([start * frame_s, len(active) * frame_s])
    merged: list[list[float]] = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < min_silence_s:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(s, e) for s, e in merged if e - s >= min_speech_s]
