"""Build the ported Praxis towers from the checkpoint's own dimensions."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from mstar.model.praxis1.components.action_expert import ActionExpert
from mstar.model.praxis1.components.decoder import DecoderTower
from mstar.model.praxis1.components.encoder import EncoderTower, TrunkStack
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError


class PraxisBackbone(nn.Module):
    def __init__(self, encoder: EncoderTower, decoder: DecoderTower) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def _section(source: Mapping, name: str) -> Mapping:
    value = source.get(name)
    if not isinstance(value, Mapping):
        raise Praxis1ConfigError(f"Praxis checkpoint lacks {name!r} settings")
    return value


def _core(config: Praxis1Config, tower: Mapping, backbone: Mapping) -> TrunkStack:
    dense = tower.get("dense_blocks")
    if not isinstance(dense, list) or dense != list(range(len(dense))):
        raise Praxis1ConfigError("Praxis dense_blocks must be an initial prefix")
    rope = _section(backbone, "rope")
    return TrunkStack(
        depth=int(tower["depth"]), num_dense=len(dense), dim=config.dim,
        head_dim=config.head_dim, num_kv_heads=config.num_kv_heads,
        dense_hidden=config.dense_ffn_hidden, moe_hidden=config.expert_hidden,
        num_experts=config.num_routed_experts, top_k=config.top_k_routed,
        routed_scaling_factor=config.routed_scaling_factor,
        rope_effective_dim=int(rope["effective_dim"]), rope_base=int(rope["base"]),
        rope_table_size=int(rope["efficient_cache_max"]),
        moe_dispatch_chunk_size=config.dispatch_chunk_size,
        mod_dim=None if tower.get("modulation_dim") is None else int(tower["modulation_dim"]),
        modulation_dtype=str(backbone["modulation_dtype"]),
    )


def build_backbone(config: Praxis1Config) -> PraxisBackbone:
    backbone = _section(config.raw, "backbone")
    enc, dec = _section(backbone, "encoder"), _section(backbone, "decoder")
    if int(enc["depth"]) != config.encoder_depth or int(dec["depth"]) != config.decoder_depth:
        raise Praxis1ConfigError("Praxis tower depth differs from the parsed checkpoint config")
    if tuple(enc["tap_blocks"]) != config.tap_blocks:
        raise Praxis1ConfigError("Praxis tap blocks differ from the parsed checkpoint config")
    enc_inputs, dec_inputs = _section(enc, "input_projections"), _section(dec, "input_projections")
    if (int(_section(enc_inputs, "TextEncodingTensor")["c_in"]) != config.text_hidden_dim
            or int(_section(enc_inputs, "VideoTensor")["c_in"]) != config.latent_channels * 4
            or int(_section(dec_inputs, "VideoTensor")["c_in"]) != config.latent_channels):
        raise Praxis1ConfigError("Praxis tower input widths disagree with the VAE and text encoder")
    patch = tuple(int(x) for x in dec["unpatch_factor"])
    if patch != config.patch_size:
        raise Praxis1ConfigError("Praxis decoder unpatch factor disagrees with the video patch size")
    encoder = EncoderTower(
        _core(config, enc, backbone), dim=config.dim, text_dim=config.text_hidden_dim,
        video_dim=config.latent_channels * 4, time_dim=config.timestep_dim,
        num_registers=config.num_registers, tap_blocks=config.tap_blocks,
        positive_zero_residual_add=bool(enc["positive_zero_residual_add"]),
    )
    decoder = DecoderTower(
        _core(config, dec, backbone), dim=config.dim, video_dim=config.latent_channels,
        time_dim=config.timestep_dim, num_registers=config.num_registers,
        patch_size=(patch[1], patch[2]),
        positive_zero_residual_add=bool(dec["positive_zero_residual_add"]),
    )
    return PraxisBackbone(encoder, decoder)


def build_expert(config: Praxis1Config) -> ActionExpert:
    expert = _section(config.raw, "action_expert")
    for key, expected in (
        ("rope_cross", False), ("rope_self", False), ("absolute_positions", True),
        ("action_pos_table", False), ("future_chunk_offset", "zero"),
    ):
        if expert.get(key) != expected:
            raise Praxis1ConfigError(f"Praxis expert {key}={expert.get(key)!r} is not ported")
    rope = _section(expert, "rope")
    return ActionExpert(
        action_patch_dim=config.action_patch_dim, proprio_patch_dim=config.proprio_patch_dim,
        trunk_dim=config.trunk_dim, dim=config.expert_dim, num_layers=config.expert_layers,
        num_heads=config.expert_heads, head_dim=config.expert_head_dim,
        num_future_per_chunk=int(expert["num_future_per_chunk"]),
        max_chunks=int(expert["max_chunks"]), max_action_tokens=config.max_action_tokens,
        tap_norm=bool(expert["tap_norm"]),
        clean_only_conditioning=bool(expert["clean_only_conditioning"]),
        rope=rope, global_order=tuple(expert["global_order"]),
        head_plus_zero=bool(expert["head_plus_zero"]),
        layout_stamp=int(expert["layout_stamp"]),
    )
