"""Qwen3-TTS reference clips whose header gives no sample count (a FLAC written
to a pipe), for which libsndfile reports 2**63 - 1 frames."""
from pathlib import Path

import numpy as np
import pytest

from mstar.api_server import media_io
from mstar.model import media_limits
from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

soundfile = pytest.importorskip("soundfile")

RATE = 24000
# libsndfile cannot decode the last frame of such a FLAC; its own encoder writes 4096 samples a frame
LAST_FRAME = 4096


def _model():
    model = object.__new__(Qwen3TTSModel)
    model.config = Qwen3TTSModelConfig(tts_model_type="base")
    return model


def _headerless_flac(path, seconds, channels=1):
    """A real FLAC whose STREAMINFO carries no sample count."""
    wave = (np.sin(np.arange(int(seconds * RATE)) / 20) * 8000).astype(np.int16)
    samples = np.stack([wave, wave // 2], axis=1)[:, :channels]
    soundfile.write(str(path), samples, RATE, format="FLAC", subtype="PCM_16")
    data = bytearray(path.read_bytes())
    # STREAMINFO starts at byte 8; its 36-bit sample count ends at byte 25 (the
    # low nibble of byte 21 holds its top bits), and 26..41 is the MD5
    data[21] &= 0xF0
    data[22:42] = bytes(20)
    path.write_bytes(bytes(data))
    assert soundfile.info(str(path)).frames == 2**63 - 1
    return str(path), samples.astype(np.float32) / 32768


@pytest.mark.parametrize("channels", [1, 2])
def test_load_audio_decodes_a_clip_without_a_sample_count(tmp_path, channels):
    path, samples = _headerless_flac(tmp_path / "piped.flac", 5, channels)
    audio = _model().load_audio(path, "cpu").data
    assert 5 * RATE - LAST_FRAME <= audio.numel() <= 5 * RATE
    np.testing.assert_allclose(audio.numpy(), samples[: audio.numel()].mean(axis=1), atol=1e-6)


def test_load_audio_refuses_a_long_clip_without_a_sample_count_after_about_30_s(tmp_path):
    path, _ = _headerless_flac(tmp_path / "long.flac", 45)
    with pytest.raises(ValueError, match=r"reference clip is 30\.\d\d s; it must be between 1 and 30 s"):
        _model().load_audio(path, "cpu")


def test_the_api_server_accepts_a_clip_without_a_sample_count(tmp_path):
    path, _ = _headerless_flac(tmp_path / "piped.flac", 5)
    media_io.check_ref_audio(path, "soundfile")
    assert Path(path).exists()


def test_the_api_server_counts_a_long_clip_without_a_sample_count_only_past_its_limit(tmp_path, monkeypatch):
    path, _ = _headerless_flac(tmp_path / "long.flac", 75)
    counted = []
    count = media_limits.count_sound_file_seconds

    def counting(*args):
        counted.append(count(*args))
        return counted[-1]

    monkeypatch.setattr(media_limits, "count_sound_file_seconds", counting)
    with pytest.raises(media_io.RefAudioTooLongError):
        media_io.check_ref_audio(path, "soundfile")
    assert len(counted) == 1 and 60 < counted[0] <= 60 + LAST_FRAME / RATE
    assert not Path(path).exists()


@pytest.mark.parametrize("decoder", ["soundfile", "soundfile+torchcodec"])
def test_the_api_server_refuses_a_clip_with_no_samples(tmp_path, decoder):
    path = tmp_path / "empty.wav"
    soundfile.write(str(path), np.zeros(0, dtype=np.float32), RATE)
    with pytest.raises(ValueError, match="ref_audio must be audio"):
        media_io.check_ref_audio(str(path), decoder)
    assert not path.exists()
