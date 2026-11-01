"""CPU parity of the M* ECAPA-TDNN speaker encoder and mel front end (Qwen3-TTS Base).

Oracle: the ``qwen_tts`` reference package. Both encoders start from the same
random weights (the M* module loads the reference state dict verbatim, which
also pins the checkpoint key layout), so any drift is an implementation
difference, not initialization noise.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from mstar.model.qwen3_tts.components.speaker_encoder import Qwen3TTSSpeakerEncoder, speaker_mel
from mstar.model.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig


def _reference_modules():
    # The reference package imports only next to the dependencies it pins: a
    # ``--no-deps`` install lacks librosa and sox (ModuleNotFoundError), and under
    # transformers 5 its ``check_model_inputs`` decorators raise TypeError. Serving
    # needs neither, so only the oracle tests skip there.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pysox probes for a SoX binary on import
        try:
            from qwen_tts.core.models.configuration_qwen3_tts import (
                Qwen3TTSSpeakerEncoderConfig as RefConfig,
            )
            from qwen_tts.core.models.modeling_qwen3_tts import (
                Qwen3TTSSpeakerEncoder as RefEncoder,
            )
            from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        except (ImportError, TypeError) as exc:
            pytest.skip(f"qwen-tts reference does not import here: {exc!r}")
    return RefConfig, RefEncoder, mel_spectrogram


def test_speaker_encoder_matches_reference_on_shared_weights():
    RefConfig, RefEncoder, _ = _reference_modules()
    torch.manual_seed(0)
    config = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, enc_channels=(32, 32, 32, 32, 96), enc_se_channels=16,
                                          enc_attention_channels=16)
    reference = RefEncoder(RefConfig(
        enc_dim=64, enc_channels=[32, 32, 32, 32, 96], enc_se_channels=16, enc_attention_channels=16,
    )).eval()
    ours = Qwen3TTSSpeakerEncoder(config).eval()
    # Identical parameter names: the checkpoint's ``speaker_encoder.*`` keys
    # load without a remap.
    assert set(dict(ours.named_parameters())) == set(dict(reference.named_parameters()))
    ours.load_state_dict(reference.state_dict())

    mels = torch.randn(3, 97, config.mel_dim)
    with torch.no_grad():
        expected = reference(mels)
        actual = ours(mels)
    assert actual.shape == (3, 64)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_mel_front_end_matches_reference():
    _, _, mel_spectrogram = _reference_modules()
    config = Qwen3TTSSpeakerEncoderConfig(enc_dim=2048)
    torch.manual_seed(1)
    wave = (torch.randn(2, 24000) * 0.3).clamp(-1, 1)
    expected = mel_spectrogram(
        wave, n_fft=config.n_fft, num_mels=config.mel_dim, sampling_rate=config.sample_rate,
        hop_size=config.hop_size, win_size=config.win_size, fmin=config.fmin, fmax=config.fmax,
    ).transpose(1, 2)
    actual = speaker_mel(config)(wave).transpose(1, 2)
    assert actual.shape == expected.shape == (2, 24000 // config.hop_size, config.mel_dim)
    # the reference's own librosa filters and op sequence: bit-equal on CPU
    assert torch.equal(actual, expected)


