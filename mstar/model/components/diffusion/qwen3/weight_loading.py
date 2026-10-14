"""Checkpoint rules for the Qwen3 prompt encoder.

What is specific to loading a Qwen3 LM checkpoint into
``qwen3.encoder.Qwen3HiddenStateEncoder``: the fused-QKV sharding rules, the
``model.`` key prefix, which keys the encoder never runs, and building the
module from a config. The model-agnostic streaming/materialising half is
``components.diffusion.weight_loading``.
"""

from __future__ import annotations

import re

from mstar.model.components.diffusion.qwen3.encoder import Qwen3EncoderConfig, Qwen3HiddenStateEncoder
from mstar.model.loader.base import StackedParamRule

QKV_RULES_LM = [
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
]


def remap_text_encoder_key(name: str) -> str:
    return name[len("model."):] if name.startswith("model.") else name


def text_encoder_skip(text_config: Qwen3EncoderConfig):
    """Predicate for the LM checkpoint keys the encoder never runs: the LM head, the
    final norm, and every decoder layer past the deepest tapped one."""
    needed = text_config.num_layers_needed
    layer_re = re.compile(r"^model\.layers\.(\d+)\.")

    def skip(name: str) -> bool:
        if name.startswith("lm_head.") or name == "model.norm.weight":
            return True
        m = layer_re.match(name)
        return m is not None and int(m.group(1)) >= needed

    return skip


def make_text_encoder(text_config: Qwen3EncoderConfig) -> Qwen3HiddenStateEncoder:
    return Qwen3HiddenStateEncoder(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_heads=text_config.num_attention_heads,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        rms_norm_eps=text_config.rms_norm_eps,
        rope_theta=text_config.rope_theta,
        tap_layers=text_config.hidden_state_layers,
    )
