"""Resource keys and walk names of the text LM node, and the language-model families a
checkpoint's ``model_type`` selects."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from torch import nn

LLM_NODE = "LLM"
PREFILL_WALK = "prefill"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"

KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"


class LMFamily(NamedTuple):
    """What a checkpoint's ``model_type`` brings: its config, its module and its weight names.

    ``config`` maps the HF ``config.json`` dict to a config with the attention geometry the
    resources are declared from (``num_hidden_layers``, ``num_attention_heads``,
    ``num_key_value_heads``, ``head_dim``, ``rope_theta``, ``vocab_size``). ``build`` makes the
    language model, which has ``embed(ids)``, ``forward(embeds, *, label)``, ``logits(hidden)``
    and ``load_weights(pairs)``. ``weight_name`` maps a checkpoint key to the module's, or None
    to skip it.
    """
    config: Callable[[dict], object]
    build: Callable[..., nn.Module]
    weight_name: Callable[[str], str | None]


def _qwen3_config(hf: dict):
    from mstar.model.components.qwen3_lm import Qwen3LMConfig

    # what the shared Qwen3 decoder does not implement is refused, not ignored
    if hf.get("rope_scaling") not in (None, {}):
        raise ValueError(f"rope_scaling {hf['rope_scaling']!r} is not supported for Qwen3 text models")
    if hf.get("use_sliding_window"):
        raise ValueError("sliding-window attention is not supported for Qwen3 text models")
    if hf.get("attention_bias"):
        raise ValueError("attention_bias is not supported for Qwen3 text models")
    if hf.get("hidden_act", "silu") != "silu":
        raise ValueError(f"hidden_act {hf['hidden_act']!r} is not supported for Qwen3 text models")
    return Qwen3LMConfig.from_hf(hf)


def _qwen3_build(config, *, comm_group=None, moe_parallel=None) -> nn.Module:
    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3MoeLM

    keys = dict(attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=comm_group)
    if config.num_experts:
        return Qwen3MoeLM(config, moe_parallel=moe_parallel, **keys)
    return Qwen3DenseLM(config, **keys)


def _hf_causal_lm_name(name: str) -> str | None:
    """``Qwen3ForCausalLM``-style keys: ``model.`` dropped, ``lm_head.weight`` kept."""
    if name.startswith("model."):
        return name.removeprefix("model.")
    return name if name.startswith("lm_head.") else None


_QWEN3 = LMFamily(config=_qwen3_config, build=_qwen3_build, weight_name=_hf_causal_lm_name)

# HF ``model_type`` -> family
LM_FAMILIES: dict[str, LMFamily] = {
    "qwen3": _QWEN3,
    "qwen3_moe": _QWEN3,
}
