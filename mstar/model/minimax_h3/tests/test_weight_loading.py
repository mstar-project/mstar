"""The transformer loader and the snapshot's config reader, on a tiny random checkpoint:
no diffusers needed, so these run in CI's CPU job while the parity modules skip there."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.minimax_h3 import weight_loader
from mstar.model.minimax_h3.config import PARTITION, H3Config
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG, random_checkpoint, write_snapshot_configs


def test_loader_refuses_a_partial_checkpoint(tmp_path):
    state = random_checkpoint(CFG)
    del state["blocks.1.mlp.fc2.weight"]
    (tmp_path / PARTITION / "transformer").mkdir(parents=True)
    save_file(state, str(tmp_path / PARTITION / "transformer" / "model.safetensors"))
    with pytest.raises(RuntimeError, match="unloaded"):
        weight_loader.build_transformer(H3Config(transformer=CFG), tmp_path, "cpu", dtype=torch.float32)


def test_config_reads_the_checkpoint_json(tmp_path):
    write_snapshot_configs(tmp_path, CFG)
    config = H3Config.from_snapshot(tmp_path)
    assert config.transformer == CFG  # the unknown "_class_name" ignored, lists as tuples
    assert (config.video_shift, config.audio_shift) == (11.0, 2.5)
    assert config.image_token_id == 151655 and config.mrope_section == (24, 20, 20)
    assert config.text.hidden_size == 5120 and not config.text.fused_qkv
    assert config.text.max_sequence_length == config.text_max_positions == 262_144
    assert config.video_vae.latents_std == (2.0,) * 24 and config.video_vae.decoder_layers == 36
    assert config.video_encoder.channel_mult == (1, 2, 2, 4, 4, 8)
    assert config.audio_vae.sample_rate == 32000
