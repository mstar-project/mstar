"""Waveform preprocessing for Omnilingual ASR, in the reference's order.

The reference pipeline (omnilingual_asr ``models/inference/pipeline.py`` and
``datasets/utils/audio.py``) decodes the file at its own rate (libsndfile,
float32), resamples every channel to 16 kHz with
``torchaudio.functional.resample``, averages the channels, normalizes the
utterance to zero mean and unit variance (``layer_norm`` over the whole
waveform, eps 1e-5) and rejects clips over 40 s. ``load_waveform`` and
``preprocess`` do the same, with the same ops, so an upload reaches the encoder
bit-identical to the reference's input.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from mstar.api_server.media_io import (
    AudioRefusedError,
    AudioTooLongError,
    check_resample_rate,
    torchcodec_samples_capped,
)
from mstar.model.omnilingual_asr.config import MAX_AUDIO_SECONDS, MIN_AUDIO_SAMPLES, SAMPLE_RATE


def decode_file(path: str) -> tuple[torch.Tensor, int]:
    """Float32 samples at the file's own rate: ``(time,)`` for mono, ``(time,
    channels)`` otherwise, as libsndfile (the reference's decoder) returns them.

    libsndfile through ``soundfile`` first; then, for a WAV file, a reader here
    that scales samples as libsndfile does, so a server without soundfile still
    decodes WAV exactly; then FFmpeg through ``torchcodec`` for the containers
    neither reads. The two library decoders stop just past
    ``MAX_AUDIO_SECONDS`` and refuse a longer file with
    :class:`~mstar.api_server.media_io.AudioTooLongError`, whatever its header
    declares."""
    errors = []
    try:
        import soundfile as sf

        with sf.SoundFile(path) as sound:
            sr = sound.samplerate
            try:
                check_resample_rate(sr, SAMPLE_RATE)
            except ValueError as exc:
                raise AudioRefusedError(str(exc)) from None
            limit = int(MAX_AUDIO_SECONDS * sr)
            # one read, as the reference's `sf.read` does: a read split inside
            # an MP3 frame moves the samples after it
            audio = sound.read(limit + 1, dtype="float32", always_2d=False)
        if len(audio) > limit:
            raise AudioTooLongError(MAX_AUDIO_SECONDS)
        return torch.from_numpy(np.ascontiguousarray(audio)), int(sr)
    except AudioRefusedError:
        raise
    except Exception as exc:  # noqa: BLE001 (missing, or not a libsndfile container)
        errors.append(f"soundfile: {exc}")
    with open(path, "rb") as f:
        head = f.read(12)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        try:
            return _decode_pcm_wav(path)
        except ValueError as exc:
            errors.append(f"wav: {exc}")
    try:
        samples = torchcodec_samples_capped(path, MAX_AUDIO_SECONDS)
        data = samples.data  # (channels, time)
        data = data[0] if data.shape[0] == 1 else data.transpose(0, 1).contiguous()
        return data.float(), int(samples.sample_rate)
    except AudioRefusedError:
        raise
    except Exception as exc:  # noqa: BLE001 (missing, or FFmpeg cannot read it)
        errors.append(f"torchcodec: {exc}")
    # the decoders name the file; the client only needs to know it was theirs
    detail = "; ".join(errors).replace(str(path), "<upload>").replace(repr(str(path))[1:-1], "<upload>")
    raise ValueError(f"cannot decode the audio file ({detail}); install soundfile (mstar-ai[audio])")


_WAVE_PCM, _WAVE_FLOAT, _WAVE_EXTENSIBLE = 1, 3, 0xFFFE


def _decode_pcm_wav(path: str) -> tuple[torch.Tensor, int]:
    """PCM (8/16/24/32-bit) and IEEE float (32/64-bit) WAV, scaled as
    libsndfile scales them to float32."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file; install soundfile (mstar-ai[audio]) for other formats")
    fmt = samples = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise ValueError("WAV file without fmt or data chunk")
    if len(fmt) < 16:
        raise ValueError("WAV fmt chunk is too short")
    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _WAVE_EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    if channels < 1:
        raise ValueError(f"WAV file with {channels} channels")
    width = bits // 8
    supported = (tag == _WAVE_FLOAT and width in (4, 8)) or (tag == _WAVE_PCM and width in (1, 2, 3, 4))
    if not supported:
        raise ValueError(f"unsupported WAV encoding (format {tag}, {bits} bits); install soundfile")
    samples = samples[: len(samples) - len(samples) % (width * channels)]
    if tag == _WAVE_FLOAT:
        audio = np.frombuffer(samples, dtype="<f4" if width == 4 else "<f8").astype(np.float32)
    elif width == 1:
        audio = (np.frombuffer(samples, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(samples, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (raw[:, 0] << 8 | raw[:, 1] << 16 | raw[:, 2] << 24) >> 8
        audio = ints.astype(np.float32) / 8388608.0
    else:
        audio = np.frombuffer(samples, dtype="<i4").astype(np.float32) / 2147483648.0
    audio = audio.reshape(-1, channels) if channels > 1 else audio
    return torch.from_numpy(np.ascontiguousarray(audio)), int(rate)


def to_16k_mono(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample each channel to 16 kHz, then average the channels, as the
    reference does (resample, then ``convert_to_mono``'s in-order sum and divide)."""
    if audio.dim() > 2:
        raise ValueError(f"audio must be (time,) or (time, channels); got shape {tuple(audio.shape)}")
    check_resample_rate(sample_rate, SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        # the reference's heuristic: the longer axis is time
        transpose = audio.dim() == 2 and audio.shape[0] > audio.shape[1]
        if transpose:
            audio = audio.transpose(1, 0)
        audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
        if transpose:
            audio = audio.transpose(1, 0)
    if audio.dim() == 2:
        mono = audio[:, 0].clone()
        for i in range(1, audio.shape[1]):
            mono += audio[:, i]
        mono /= audio.shape[1]
        audio = mono
    return audio


def load_waveform(path: str) -> torch.Tensor:
    """A file's 16 kHz mono float32 waveform, not yet normalized."""
    audio, sample_rate = decode_file(path)
    return to_16k_mono(audio.float(), sample_rate)


def check_length(num_samples: int) -> None:
    seconds = num_samples / SAMPLE_RATE
    if seconds > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"audio is {seconds:.2f} s; Omnilingual ASR CTC transcribes at most {MAX_AUDIO_SECONDS:g} s "
            "per request (/v1/audio/transcriptions cuts longer uploads into windows)"
        )
    if num_samples < MIN_AUDIO_SAMPLES:
        raise ValueError(f"audio is {num_samples} samples; at least {MIN_AUDIO_SAMPLES} (25 ms at 16 kHz) are needed")


@torch.no_grad()
def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance over the whole utterance, in float32
    (the reference's ``layer_norm(waveform, waveform.shape)``)."""
    waveform = waveform.float()
    return F.layer_norm(waveform, waveform.shape)


def preprocess(waveform: torch.Tensor) -> torch.Tensor:
    """16 kHz mono waveform -> the encoder input: length-checked, normalized."""
    if waveform.dim() != 1:
        raise ValueError(f"expected a 1-D 16 kHz waveform; got shape {tuple(waveform.shape)}")
    check_length(waveform.shape[0])
    return normalize(waveform)
