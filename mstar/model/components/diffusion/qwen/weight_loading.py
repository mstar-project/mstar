"""Checkpoint rules for the Qwen prompt encoder.

What is specific to loading a Qwen LM checkpoint into
``qwen.encoder.QwenHiddenStateEncoder``: the fused-QKV sharding rules, the key prefix
(``model.`` for a Qwen LM and a Qwen2.5-VL checkpoint's language model as published,
``model.language_model.`` for the language model of a checkpoint transformers 5 saves), and
which keys the encoder never runs; and the Qwen3-VL vision tower whose tokens it takes. The
model-agnostic streaming/materialising half is ``components.diffusion.weight_loading``.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from mstar.model.components.diffusion.qwen.encoder import QwenEncoderConfig
from mstar.model.components.diffusion.weight_loading import iter_transformers_component, materialize
from mstar.model.components.qwen3_vit import VisionRotaryEmbedding
from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel
from mstar.model.loader.base import StackedParamRule, load_weights_into

QKV_RULES_LM = [
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
]

# the language model's prefixes, the longer first
_LM_PREFIXES = ("model.language_model.", "model.")
_QWEN3_VL_VISION_PREFIX = "model.visual."
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


def build_qwen3_vl_tower(source: Path, device, what: str, dtype: torch.dtype = torch.bfloat16) -> QwenVisionModel:
    """A Qwen3-VL checkpoint's vision tower with its deepstack mergers, from the shards in ``source``
    (``model.visual.``). Its rotary ``inv_freq`` is derived, fp32 as the reference keeps it, so
    completeness is over the persistent state."""
    source = Path(source)
    config = QwenVisionConfig.from_hf(source)
    with torch.device("meta"):
        tower = QwenVisionModel(config)
    materialize(tower, dtype, device)
    tower.rotary_pos_emb.register_buffer(
        "inv_freq", VisionRotaryEmbedding(config).inv_freq.to(device), persistent=False,
    )
    prefix = _QWEN3_VL_VISION_PREFIX
    weights = iter_transformers_component(source, device, skip=lambda n: not n.startswith(prefix))
    loaded = load_weights_into(tower, weights, name_remapper=lambda n: n[len(prefix):])
    missing = sorted(set(tower.state_dict()) - loaded)
    if missing:
        raise RuntimeError(f"{what}: {len(missing)} unloaded parameters {missing[:5]}")
    return tower.eval()
