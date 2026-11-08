"""Walks, resource keys and the checkpoint config of the dense Qwen3 text model."""
from __future__ import annotations

from mstar.model.components.qwen3_lm import Qwen3LMConfig

LLM_NODE = "LLM"
PREFILL_WALK = "prefill"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"

KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"


def qwen3_lm_config(hf: dict) -> Qwen3LMConfig:
    """``Qwen3ForCausalLM``'s ``config.json``. What the shared Qwen3 decoder
    doesn't implement is refused, not ignored."""
    if hf.get("model_type") != "qwen3":
        raise ValueError(f"not a dense Qwen3 checkpoint: model_type {hf.get('model_type')!r}")
    if hf.get("rope_scaling") not in (None, {}):
        raise ValueError(f"rope_scaling {hf['rope_scaling']!r} is not supported for Qwen3")
    # transformers 5 writes RoPE under rope_parameters, and reads it over a top-level rope_theta
    rope = hf.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"rope_parameters {rope!r} (scaled RoPE) is not supported for Qwen3")
    if "rope_theta" in rope:
        hf = {**hf, "rope_theta": rope["rope_theta"]}
    if hf.get("use_sliding_window"):
        raise ValueError("sliding-window attention is not supported for Qwen3")
    if hf.get("attention_bias"):
        raise ValueError("attention_bias is not supported for Qwen3")
    if hf.get("hidden_act", "silu") != "silu":
        raise ValueError(f"hidden_act {hf['hidden_act']!r} is not supported for Qwen3")
    return Qwen3LMConfig.from_hf(hf)


def checkpoint_dtype(hf: dict) -> str:
    """The weight dtype ``config.json`` names (``torch_dtype``, or ``dtype``)."""
    return str(hf.get("torch_dtype") or hf.get("dtype") or "bfloat16")
