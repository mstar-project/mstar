"""Checkpoint rules for the Qwen3 prompt encoder.

What is specific to loading a Qwen3 LM checkpoint into
``qwen3.encoder.Qwen3HiddenStateEncoder``: the fused-QKV sharding rules, the
key prefix (``model.`` for a Qwen3 LM, ``model.language_model.`` for the
language model of a Qwen3-VL checkpoint), which keys the encoder never runs,
and building the module from a config. The model-agnostic
streaming/materialising half is ``components.diffusion.weight_loading``.
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

QWEN3_PREFIX = "model."
QWEN3_VL_PREFIX = "model.language_model."


def remap_text_encoder_key(name: str) -> str:
    return name[len(QWEN3_PREFIX):] if name.startswith(QWEN3_PREFIX) else name


def remap_qwen3_vl_text_key(name: str) -> str:
    return name[len(QWEN3_VL_PREFIX):] if name.startswith(QWEN3_VL_PREFIX) else name


def text_encoder_skip(text_config: Qwen3EncoderConfig, prefix: str = QWEN3_PREFIX):
    """Predicate for the LM checkpoint keys the encoder never runs: the LM head, the
    final norm, every decoder layer past the deepest tapped one and, in a Qwen3-VL
    checkpoint, the vision tower."""
    needed = text_config.num_layers_needed
    layer_re = re.compile(rf"^{re.escape(prefix)}layers\.(\d+)\.")

    def skip(name: str) -> bool:
        if name.startswith(("lm_head.", "model.visual.")) or name == f"{prefix}norm.weight":
            return True
        m = layer_re.match(name)
        return m is not None and int(m.group(1)) >= needed

    return skip


def qwen3_vl_text_skip(text_config: Qwen3EncoderConfig):
    return text_encoder_skip(text_config, QWEN3_VL_PREFIX)


def qkv_rules(text_config: Qwen3EncoderConfig) -> list[StackedParamRule]:
    """The stacked rules routing q/k/v into the fused projection, or none when the
    encoder keeps the checkpoint's three."""
    return QKV_RULES_LM if text_config.fused_qkv else []


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
        fused_qkv=text_config.fused_qkv,
    )
