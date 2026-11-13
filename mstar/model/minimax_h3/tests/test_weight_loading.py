"""The transformer loader and the snapshot's config reader, on a tiny random checkpoint:
no diffusers needed (but for one check against its converter), so these run in CI's CPU job
while the parity modules skip there."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.minimax_h3 import weight_loader
from mstar.model.minimax_h3.config import PARTITION, H3Config
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG, random_checkpoint, to_diffusers, write_snapshot_configs


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


def _write(state: dict, path, name: str = "diffusion_pytorch_model.safetensors") -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(path / name))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_diffusers_layout_transformer_loads_as_the_native_one(tmp_path, dtype):
    """FastH3 ships the transformer in diffusers' layout, every tensor bf16: mapped back, the DiT and
    the text conditioner hold what FL2VA's layout gives (the fp32 modules the same values,
    upcast), and the VSA gates (NaN here) are never read."""
    state = {k: v if k == "rope.inv_freq" else v.to(dtype) for k, v in random_checkpoint(CFG).items()}
    _write(state, tmp_path / PARTITION / "transformer", "model.safetensors")
    _write(to_diffusers(state, CFG), tmp_path / "fasth3" / "transformer")
    config = H3Config(transformer=CFG)
    for build in (weight_loader.build_transformer, weight_loader.build_text_conditioner):
        native = build(config, tmp_path, "cpu", dtype=torch.bfloat16)
        distilled = build(config, tmp_path, "cpu", dtype=torch.bfloat16, dit_source=tmp_path / "fasth3" / "transformer")
        ours, want = distilled.state_dict(), native.state_dict()
        assert ours.keys() == want.keys() and len(want) > 10
        for name, tensor in want.items():
            assert ours[name].dtype == tensor.dtype and torch.equal(ours[name], tensor), name


def test_a_diffusers_layout_transformer_with_a_missing_or_unknown_key_is_refused(tmp_path):
    state = to_diffusers(random_checkpoint(CFG), CFG)
    source = tmp_path / "transformer"
    for edit in ({"transformer_blocks.1.ff.net.2.weight": None}, {"transformer_blocks.0.attn.to_extra.weight": None}):
        broken = dict(state)
        for key in edit:
            if key in broken:
                del broken[key]
            else:
                broken[key] = torch.zeros(1)
        _write(broken, source)
        with pytest.raises(RuntimeError):
            weight_loader.build_transformer(H3Config(transformer=CFG), tmp_path, "cpu", dtype=torch.float32,
                                            dit_source=source)


def test_the_restated_conversion_is_diffusers_own():
    """``to_diffusers`` (what the loader inverts) against diffusers' converter, names and values."""
    single_file = pytest.importorskip("diffusers.loaders.single_file_utils")
    convert = getattr(single_file, "convert_minimax_h3_transformer_checkpoint_to_diffusers", None)
    if convert is None:
        pytest.skip("this diffusers has no MiniMax-H3 converter")
    state = random_checkpoint(CFG)
    config = {"num_attention_heads": CFG.num_attention_heads, "attention_head_dim": CFG.attention_head_dim}
    theirs = convert(dict(state), config, qkv_layout="interleaved")
    ours = to_diffusers(state, CFG, vsa_gates=False)
    assert theirs.keys() == ours.keys()
    for name, tensor in theirs.items():
        assert torch.equal(tensor, ours[name]), name


def test_a_vsa_distilled_transformer_loads_its_gates(tmp_path):
    """With ``vsa_gate`` (a release distilled with VSA) each block's ``to_gate_compress`` loads as
    ``attn.gate_compress``, and a missing one is refused."""
    import dataclasses

    state = random_checkpoint(CFG)
    diffusers = to_diffusers(state, CFG)
    gates = {k: torch.randn_like(v, dtype=torch.float32) for k, v in diffusers.items() if "to_gate_compress" in k}
    assert len(gates) == CFG.num_layers
    _write({**diffusers, **gates}, tmp_path / "transformer")
    config = H3Config(transformer=dataclasses.replace(CFG, vsa_gate=True))
    dit = weight_loader.build_transformer(config, tmp_path, "cpu", dtype=torch.float32,
                                          dit_source=tmp_path / "transformer")
    for i, block in enumerate(dit.blocks):
        gate = gates[f"transformer_blocks.{i}.attn.to_gate_compress.weight"]
        assert torch.equal(block.attn.gate_compress.weight, gate)
    _write(state, tmp_path / "stripped" / "transformer")
    with pytest.raises(RuntimeError):
        weight_loader.build_transformer(config, tmp_path, "cpu", dtype=torch.float32,
                                        dit_source=tmp_path / "stripped" / "transformer")
