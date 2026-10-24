"""Higgs audio tower: a short clip padded to the chunk width and masked
(``valid_frames``) encodes as the unpadded clip does."""

import pytest
import torch

from mstar.model.higgs_audio.components.audio_tower import HiggsAudioFeatureProjector, HiggsAudioTower
from mstar.model.higgs_audio.config import HiggsAudioModelConfig


@pytest.mark.parametrize("width", [3, 4, 7, 37, 64, 99, 100])
def test_masked_padded_tower_matches_unpadded(width):
    torch.manual_seed(0)
    config = HiggsAudioModelConfig(
        hidden_size=48, audio_d_model=32, audio_encoder_layers=2, audio_encoder_attention_heads=4,
        audio_encoder_ffn_dim=64, audio_num_mel_bins=16, audio_max_source_positions=64,
    )
    tower, projector = HiggsAudioTower(config).eval(), HiggsAudioFeatureProjector(config).eval()
    frames = 100
    clip = torch.randn(1, config.audio_num_mel_bins, width)
    other = torch.randn(1, config.audio_num_mel_bins, frames)
    padded = torch.cat([torch.nn.functional.pad(clip, (0, frames - width)), other])
    with torch.no_grad():
        want = projector(tower(clip))[0]
        got = projector(tower(padded, torch.tensor([width, frames])))
        full = projector(tower(other))[0]
    n = config.encoder_output_length(width)
    assert n == want.shape[0]
    assert torch.allclose(got[0, :n], want[:n], atol=1e-5, rtol=1e-4)
    assert torch.allclose(got[1], full, atol=1e-5, rtol=1e-4)
