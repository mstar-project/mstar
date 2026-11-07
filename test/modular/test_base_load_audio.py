"""``Model.load_audio`` decodes common containers without FFmpeg.

The Dynamo base image ships a video-only FFmpeg 8 that torchcodec loads
first, so every audio upload failed there with "Invalid data found when
processing input". The fake torchcodec below fails the same way.
"""
import sys
import types

import numpy as np
import pytest
import torch

from mstar.model.base import Model


class _VideoOnlyFFmpegDecoder:
    calls: list[str] = []

    def __init__(self, path, **_kwargs):
        type(self).calls.append(path)
        raise RuntimeError(f"Could not open input file: {path} Invalid data found when processing input")


@pytest.fixture
def video_only_ffmpeg(monkeypatch):
    _VideoOnlyFFmpegDecoder.calls = []
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = _VideoOnlyFFmpegDecoder
    package = types.ModuleType("torchcodec")
    package.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return _VideoOnlyFFmpegDecoder


def _tone(rate, seconds=0.5, channels=1):
    t = np.arange(int(rate * seconds)) / rate
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.stack([mono] * channels, axis=1)


@pytest.mark.parametrize(
    ("ext", "subtype"), [("wav", "PCM_16"), ("flac", "PCM_16"), ("ogg", "VORBIS"), ("mp3", "MPEG_LAYER_III")]
)
@pytest.mark.parametrize(("rate", "channels"), [(16_000, 1), (24_000, 2)])
def test_common_containers_decode_without_ffmpeg(tmp_path, video_only_ffmpeg, ext, subtype, rate, channels):
    sf = pytest.importorskip("soundfile")
    if subtype not in sf.available_subtypes(ext.upper() if ext != "mp3" else "MP3"):
        pytest.skip(f"this libsndfile cannot write {subtype}")
    path = tmp_path / f"clip.{ext}"
    sf.write(path, _tone(rate, channels=channels), rate, subtype=subtype)

    out = Model.load_audio(None, str(path), "cpu")

    assert video_only_ffmpeg.calls == []
    assert out.metadata == {"sample_rate": 16_000, "num_channels": 1}
    assert out.data.dtype == torch.float32 and out.data.dim() == 1
    assert abs(out.data.numel() - 8_000) <= 1_200  # MP3 adds encoder delay and padding
    peak = out.data.abs().max().item()
    assert 0.4 < peak < 0.6


def test_16k_wav_samples_are_the_file_samples(tmp_path, video_only_ffmpeg):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "clip.wav"
    sf.write(path, _tone(16_000), 16_000, subtype="FLOAT")
    out = Model.load_audio(None, str(path), "cpu")
    assert torch.equal(out.data, torch.from_numpy(_tone(16_000)[:, 0]))


@pytest.mark.parametrize("rate", [100_003, 48_001])
def test_a_crafted_header_rate_is_refused_before_resampling(tmp_path, video_only_ffmpeg, monkeypatch, rate):
    """torchaudio's kernel for a rate coprime to 16 kHz holds about rate * 16000
    floats (6.4 GB at 100,003 Hz). FFmpeg resampled on this path before, so the
    libsndfile path must refuse such a header before torchaudio sees it."""
    sf = pytest.importorskip("soundfile")
    import torchaudio

    def no_resample(*_args, **_kwargs):
        raise AssertionError("a crafted rate reached torchaudio's resampler")

    monkeypatch.setattr(torchaudio.functional, "resample", no_resample)
    path = tmp_path / "crafted.wav"
    sf.write(path, _tone(16_000), rate, subtype="PCM_16")
    with pytest.raises(ValueError, match=f"unsupported sample rate {rate} Hz"):
        Model.load_audio(None, str(path), "cpu")


def test_other_containers_still_go_to_torchcodec(tmp_path, video_only_ffmpeg):
    path = tmp_path / "clip.m4a"
    path.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
    with pytest.raises(RuntimeError, match="Invalid data"):
        Model.load_audio(None, str(path), "cpu")
    assert video_only_ffmpeg.calls == [str(path)]


def test_omnivoice_reads_a_wav_reference_without_ffmpeg(tmp_path, video_only_ffmpeg):
    """The cloning reference, end to end: the API check, then the loader at the codec rate."""
    sf = pytest.importorskip("soundfile")
    import base64

    from mstar.api_server.openai import adapters
    from mstar.api_server.openai.protocol import SpeechRequest
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    clip = tmp_path / "ref.wav"
    sf.write(clip, _tone(24_000, seconds=1.0), 24_000, subtype="PCM_16")
    uploads = tmp_path / "uploads"
    uploads.mkdir(mode=0o700)
    req = SpeechRequest(input="hi", ref_audio=base64.b64encode(clip.read_bytes()).decode(), ref_text="hi")
    (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]

    model = types.SimpleNamespace(config=types.SimpleNamespace(sample_rate=24_000))
    out = OmniVoiceModel.load_audio(model, path, "cpu")

    assert video_only_ffmpeg.calls == []
    assert out.metadata == {"sample_rate": 24_000, "num_channels": 1}
    assert out.data.numel() == 24_000
