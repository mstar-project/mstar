"""Every loader that resamples client audio refuses a sample rate whose
resampling kernel is unbounded (``media_io.check_resample_rate``) before it
resamples, and still decodes every common rate.

torchaudio's sinc kernel holds about ``(rate / g) * (target / g)`` taps,
``g = gcd(rate, target)``: a header claiming 48,001 Hz costs 1.15G taps at
24 kHz. The refusal tests replace ``torchaudio.functional.resample`` with a
function that fails, so they detect the call instead of allocating.

Each loader runs with ``soundfile`` when it is installed and with a standard
library stand-in otherwise (``soundfile`` is the optional ``audio`` extra).
"""

import math
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from mstar.api_server import media_io

COMMON_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000)
PULL_DOWN_RATES = (44056, 47952, 48048)
TARGETS = (16000, 24000)
BAD_RATE = 48001


def _wave_soundfile() -> types.ModuleType:
    """The parts of ``soundfile`` the loaders use, for 16-bit PCM WAV."""
    module = types.ModuleType("soundfile")

    class SoundFileError(RuntimeError):
        pass

    def info(path):
        with wave.open(str(path), "rb") as w:
            return types.SimpleNamespace(
                samplerate=w.getframerate(), frames=w.getnframes(), channels=w.getnchannels(),
            )

    def _frames(w, frames, always_2d):
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes() if frames < 0 else frames)
        data = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).reshape(-1, channels)
        return data if always_2d or channels > 1 else data[:, 0]

    def read(path, dtype="float32", always_2d=False):
        with wave.open(str(path), "rb") as w:
            return _frames(w, -1, always_2d), w.getframerate()

    class SoundFile:
        def __init__(self, path):
            self._w = wave.open(str(path), "rb")
            self.samplerate, self.channels = self._w.getframerate(), self._w.getnchannels()
            self.frames = self._w.getnframes()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._w.close()

        def read(self, frames=-1, dtype="float32", always_2d=False):
            return _frames(self._w, frames, always_2d)

    def write(path, data, samplerate):
        pcm = np.round(np.clip(np.asarray(data, dtype=np.float32), -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
            w.setsampwidth(2)
            w.setframerate(samplerate)
            w.writeframes(pcm.tobytes())

    module.SoundFileError = SoundFileError
    module.info, module.read, module.write, module.SoundFile = info, read, write, SoundFile
    return module


@pytest.fixture(params=["soundfile", "wave"])
def sf(request, monkeypatch):
    if request.param == "soundfile":
        return pytest.importorskip("soundfile")
    stand_in = _wave_soundfile()
    monkeypatch.setitem(sys.modules, "soundfile", stand_in)
    return stand_in


@pytest.fixture
def no_resample(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("resampled from an unsupported rate")

    monkeypatch.setattr(torchaudio.functional, "resample", refuse)


def _clip(sf, path: Path, rate: int, seconds: float = 1.0) -> str:
    t = np.arange(int(rate * seconds)) / rate
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
    return str(path)


def _media_io(path):
    return torch.as_tensor(media_io.decode_audio(path, 16000))


def _audio_features(path):
    from mstar.model.components.audio_features import load_speech_file

    return load_speech_file(path, 16000, media_io.MAX_AUDIO_SECONDS)


def _qwen3_tts(path):
    from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

    model = object.__new__(Qwen3TTSModel)
    model.config = Qwen3TTSModelConfig()
    return model.load_audio(path, "cpu").data


def _chatterbox(path):
    from mstar.model.chatterbox.chatterbox_model import ChatterboxModel

    return ChatterboxModel._decode_audio(path)


def _nemotron_duplex(path):
    from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

    return object.__new__(NemotronDuplexModel).load_audio(path, "cpu").data


LOADERS = [
    pytest.param(_media_io, 16000, id="media_io.decode_audio"),
    pytest.param(_audio_features, 16000, id="audio_features.load_speech_file"),
    pytest.param(_qwen3_tts, 24000, id="qwen3_tts"),
    pytest.param(_chatterbox, 24000, id="chatterbox"),
    pytest.param(_nemotron_duplex, 16000, id="nemotron_duplex"),
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("rate", COMMON_RATES + PULL_DOWN_RATES)
def test_common_and_pull_down_rates_pass_the_bound(rate, target):
    media_io.check_resample_rate(rate, target)


@pytest.mark.parametrize("rate", [BAD_RATE, 100003, 0, -16000, 768_001, 10**9])
@pytest.mark.parametrize("target", TARGETS)
def test_unbounded_or_out_of_range_rates_are_refused(rate, target):
    with pytest.raises(ValueError, match=f"unsupported sample rate {rate} Hz; resample the audio to a standard rate"):
        media_io.check_resample_rate(rate, target)


def test_the_bound_counts_the_kernel_torchaudio_builds():
    from torchaudio.functional.functional import _get_sinc_resample_kernel

    for rate, target in [(44100, 16000), (47952, 16000), (22050, 16000), (8000, 24000), (16000, 16000)]:
        kernel, _ = _get_sinc_resample_kernel(rate, target, math.gcd(rate, target))
        assert media_io.resample_kernel_size(rate, target) == kernel.numel(), (rate, target, torchaudio.__version__)
    # (target / g) * (rate / g) alone is under 2**24 here, the padded kernel is not
    assert 655_325 * 16_000 // math.gcd(655_325, 16_000) ** 2 <= media_io.MAX_RESAMPLE_KERNEL
    with pytest.raises(ValueError, match="unsupported sample rate 655325 Hz"):
        media_io.check_resample_rate(655_325, 16_000)


@pytest.mark.parametrize("load,target", LOADERS)
def test_a_coprime_header_rate_is_refused_before_resampling(load, target, sf, no_resample, tmp_path):
    path = _clip(sf, tmp_path / "odd.wav", BAD_RATE)
    with pytest.raises(ValueError, match=f"unsupported sample rate {BAD_RATE} Hz"):
        load(path)


@pytest.mark.parametrize("rate", COMMON_RATES)
@pytest.mark.parametrize("load,target", LOADERS)
def test_common_rates_still_decode(load, target, rate, sf, tmp_path):
    audio = load(_clip(sf, tmp_path / f"r{rate}.wav", rate))
    assert audio.ndim == 1 and abs(audio.shape[0] - target) <= 1
    assert 0.4 < float(audio.abs().max()) < 0.6


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions: an upload over the model's clip length is decoded
# (and resampled) when mstar-server asks the bridge to plan its windows
# ---------------------------------------------------------------------------

def _plan(upload: Path, *ops: str) -> dict:
    """The bridge's answer to the last of ``ops`` on ``upload``'s Whisper windows."""
    import asyncio

    from mstar.api_server.openai.serving_transcriptions import CUT_FRAME_SECONDS, CUT_SEARCH_SECONDS
    from mstar.api_server.rust_frontend import RustFrontendBridge

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.sent, bridge._planners = [], {}
    bridge._send = bridge.sent.append
    bridge.server = types.SimpleNamespace(upload_dir=str(upload.parent))
    bridge.loop = asyncio.new_event_loop()
    try:
        for op in ops:
            bridge._windows({"op": op, "plan": 1, "rid": "r", "sub": 0, "path": str(upload), "max_seconds": 30.0,
                             "balanced": False, "search_seconds": CUT_SEARCH_SECONDS,
                             "frame_seconds": CUT_FRAME_SECONDS, "whole_seconds": 30.0})
            bridge.loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(bridge.loop)))
    finally:
        bridge.loop.close()
    return bridge.sent[-1]["metadata"]


def _long_upload(sf, tmp_path: Path, rate: int) -> Path:
    # longer than Whisper's 30 s clip, so the bridge decodes it to cut windows
    (tmp_path / "uploads").mkdir()
    return Path(_clip(sf, tmp_path / "uploads" / "long.wav", rate, seconds=31.0))


def test_transcription_upload_at_a_coprime_rate_is_a_400(sf, no_resample, tmp_path):
    upload = _long_upload(sf, tmp_path, BAD_RATE)
    error = _plan(upload, "plan")["error"]
    assert error["status"] == 400, error
    assert f"unsupported sample rate {BAD_RATE} Hz" in error["message"]
    assert list(upload.parent.iterdir()) == []


@pytest.mark.parametrize("error", [RuntimeError, ImportError])
def test_transcription_upload_that_fails_to_decode_is_removed(error, sf, tmp_path, monkeypatch):
    # header readable, body not: soundfile's read raises and the torchcodec
    # fallback raises something other than ValueError
    def broken(path, sample_rate, max_seconds):
        raise error("corrupt stream")

    monkeypatch.setattr(media_io, "decode_audio", broken)
    upload = _long_upload(sf, tmp_path, 44100)
    assert _plan(upload, "plan")["error"]["status"] == 500
    assert list(upload.parent.iterdir()) == []


def test_transcription_upload_at_a_common_rate_is_windowed(sf, tmp_path):
    answer = _plan(_long_upload(sf, tmp_path, 44100), "plan", "fixed")
    assert "error" not in answer, answer
    assert len(answer["windows"]) == 2
