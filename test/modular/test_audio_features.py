"""The shared Whisper-family audio front end against HF's feature extractor.

Log-mel features to ~1e-4 for 80- and 128-bin banks, the frame-count
convention (``samples // hop``), padding/trimming to the 30 s window, the
sinusoidal position table, and the FFmpeg-free file loader.
"""

import numpy as np
import pytest
import torch

from mstar.model.components.audio_features import (
    LogMelSpectrogram,
    load_audio_file,
    sinusoid_positions,
    slaney_mel_filter_bank,
)


def test_mel_filter_bank_matches_hf():
    audio_utils = pytest.importorskip("transformers.audio_utils")
    for n_mels in (80, 128):
        hf = audio_utils.mel_filter_bank(
            num_frequency_bins=201, num_mel_filters=n_mels, min_frequency=0.0, max_frequency=8000.0,
            sampling_rate=16_000, norm="slaney", mel_scale="slaney",
        )
        ours = slaney_mel_filter_bank(201, n_mels, 16_000)
        assert ours.shape == (n_mels, 201)
        assert np.allclose(ours, hf.T, atol=1e-8)


@pytest.mark.parametrize("num_mel_bins", [80, 128])
def test_log_mel_matches_hf_feature_extractor(num_mel_bins):
    transformers = pytest.importorskip("transformers")
    fe = transformers.WhisperFeatureExtractor(feature_size=num_mel_bins)
    mel = LogMelSpectrogram(num_mel_bins=num_mel_bins)
    torch.manual_seed(0)
    wave = torch.randn(int(16_000 * 3.37)) * 0.1
    hf = torch.from_numpy(fe(wave.numpy(), sampling_rate=16_000, return_tensors="np")["input_features"][0])
    ours = mel(mel.pad_or_trim(wave))
    assert ours.shape == hf.shape == (num_mel_bins, 3000)
    assert torch.allclose(ours, hf, atol=2e-4), (ours - hf).abs().max()


def test_log_mel_variable_length_matches_hf_unpadded():
    """Qwen3-ASR feeds the unpadded clip: HF with ``padding=False`` and no
    truncation yields ``samples // hop`` frames, as we do."""
    transformers = pytest.importorskip("transformers")
    fe = transformers.WhisperFeatureExtractor(feature_size=128)
    mel = LogMelSpectrogram()
    torch.manual_seed(1)
    wave = torch.randn(16_000 * 4 + 123) * 0.1
    hf = fe(wave.numpy(), sampling_rate=16_000, return_tensors="np", padding=False, truncation=False)
    expected = torch.from_numpy(hf["input_features"][0])
    ours = mel(wave)
    assert ours.shape == expected.shape == (128, wave.numel() // 160)
    assert torch.allclose(ours, expected, atol=2e-4)


def test_pad_or_trim_and_batching():
    mel = LogMelSpectrogram()
    assert mel.pad_or_trim(torch.ones(16_000 * 40)).shape[-1] == 480_000
    assert mel.pad_or_trim(torch.ones(16_000)).shape[-1] == 480_000
    assert mel.num_frames(16_000 * 3 + 37) == (16_000 * 3 + 37) // 160
    batched = mel(torch.stack([mel.pad_or_trim(torch.randn(16_000)), mel.pad_or_trim(torch.randn(32_000))]))
    assert batched.shape == (2, 128, 3000)
    # the dynamic-range clamp is per clip, so a loud batch-mate leaves a quiet clip alone
    quiet, loud = torch.randn(16_000) * 1e-3, torch.randn(16_000)
    alone = mel(mel.pad_or_trim(quiet))
    together = mel(torch.stack([mel.pad_or_trim(quiet), mel.pad_or_trim(loud)]))[0]
    assert torch.allclose(alone, together)


def test_sinusoid_positions_match_hf():
    modeling = pytest.importorskip("transformers.models.whisper.modeling_whisper")
    assert torch.allclose(sinusoid_positions(50, 64), modeling.sinusoids(50, 64), atol=1e-6)
    with pytest.raises(ValueError):
        sinusoid_positions(10, 63)


def test_load_audio_file_decodes_and_resamples(tmp_path):
    sf = pytest.importorskip("soundfile")
    t = np.arange(8000) / 8000.0
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    sf.write(tmp_path / "a.flac", stereo, 8000)
    wave = load_audio_file(str(tmp_path / "a.flac"), 16_000)
    assert wave.dtype == torch.float32 and wave.dim() == 1
    assert abs(wave.numel() - 16_000) <= 16
    sf.write(tmp_path / "b.wav", stereo[:, :1], 16_000)
    assert load_audio_file(str(tmp_path / "b.wav")).numel() == 8000
