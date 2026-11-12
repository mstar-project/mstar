"""Checkpoint rules for the Qwen prompt encoder.

What is specific to loading a Qwen LM checkpoint into
``qwen.encoder.QwenHiddenStateEncoder``: the fused-QKV sharding rules, the key prefix
(``model.`` for a Qwen LM and a Qwen2.5-VL checkpoint's language model as published,
``model.language_model.`` for the language model of a checkpoint transformers 5 saves), and
which keys the encoder never runs. The model-agnostic streaming/materialising half is
``components.diffusion.weight_loading``.
"""

from __future__ import annotations

import re

from mstar.model.components.diffusion.qwen.encoder import QwenEncoderConfig
from mstar.model.loader.base import StackedParamRule

QKV_RULES_LM = [
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
]

# the language model's prefixes, the longer first
_LM_PREFIXES = ("model.language_model.", "model.")
_LAYER = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.")


def remap_text_encoder_key(name: str) -> str:
    for prefix in _LM_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def text_encoder_skip(text_config: QwenEncoderConfig):
    """Predicate for the checkpoint keys the encoder never runs: the LM head, a VL
    checkpoint's vision tower, every decoder layer past the deepest tapped one, and the
    final norm unless a tap is the last layer."""
    needed = text_config.num_layers_needed
    final_norms = () if text_config.taps_final_norm else tuple(f"{p}norm.weight" for p in _LM_PREFIXES)

    def skip(name: str) -> bool:
        if name.startswith(("lm_head.", "model.visual.", "visual.")) or name in final_norms:
            return True
        m = _LAYER.match(name)
        return m is not None and int(m.group(1)) >= needed

    return skip


def qkv_rules(text_config: QwenEncoderConfig) -> list[StackedParamRule]:
    """The stacked rules routing q/k/v into the fused projection, or none when the
    encoder keeps the checkpoint's three."""
    return QKV_RULES_LM if text_config.fused_qkv else []
