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


@pytest.mark.parametrize("width,frames", [(10, 11), (37, 38), (64, 64)])
def test_a_short_clip_embeds_the_frames_its_prompt_counts(width, frames):
    """A clip under one chunk: the encoder node emits exactly what the tower
    computes over the clip's ``frames`` (the feature extractor's mask sum,
    which the prompt's audio placeholders are counted from and which can be
    one past the mel width; that frame is the zero padding), masked on the
    padded, batched path. A width of 2 mod 8 is where the extra frame adds an
    embedding."""
    from types import SimpleNamespace

    from mstar.model.higgs_audio.config import AUDIO_FEATURE_LENS_KEY
    from mstar.model.higgs_audio.submodules import HiggsAudioEncoderSubmodule

    torch.manual_seed(0)
    config = HiggsAudioModelConfig(
        hidden_size=48, audio_d_model=32, audio_encoder_layers=2, audio_encoder_attention_heads=4,
        audio_encoder_ffn_dim=64, audio_num_mel_bins=16, audio_max_source_positions=64,
        chunk_size_seconds=1.0,
    )
    tower, projector = HiggsAudioTower(config).eval(), HiggsAudioFeatureProjector(config).eval()
    sub = HiggsAudioEncoderSubmodule(tower, projector, config, graphs=False)
    clip = torch.randn(1, config.audio_num_mel_bins, width)
    fwd = SimpleNamespace(step_metadata={AUDIO_FEATURE_LENS_KEY: [frames]})
    engine_inputs = SimpleNamespace(request_ids=["r"], piecewise_runners={})
    with torch.no_grad():
        row = sub.prepare_inputs("prefill_audio", fwd, {"audio_features": [clip]})
        batch = sub.preprocess("prefill_audio", engine_inputs, [row])
        got = sub.forward_batched("prefill_audio", engine_inputs, **batch)["r"]["audio_embeds"][0]
        want = projector(tower(torch.nn.functional.pad(clip, (0, frames - width))))[0]
    assert got.shape[0] == config.encoder_output_length(frames) == want.shape[0]
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
