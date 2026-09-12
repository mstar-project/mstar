"""Hugging Face checkpoint mapping for Qwen3-VL-30B-A3B.

The published checkpoint wraps the text tower as ``model.language_model.*``
and the vision tower as ``model.visual.*``. The text keys land on
``QwenVLForCausalLM``'s ``model.*`` / ``lm_head.*`` parameter paths. Expert
weights are already fused as ``experts.{gate_up_proj,down_proj}``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights
from mstar.model.loader.iterators import iter_safetensors_shards


def remap_qwen_vl_text_key(name: str) -> str | None:
    """Map a checkpoint key onto ``QwenVLForCausalLM`` parameter names.

    Returns ``None`` to drop vision / RoPE-buffer keys so they are not
    silently copied into a text parameter of the same suffix.
    """
    if name.startswith("model.language_model."):
        rest = name.removeprefix("model.language_model.")
        if "rotary_emb" in rest:
            return None
        return rest if rest.startswith("lm_head.") else f"model.{rest}"
    if name.startswith("lm_head."):
        return name
    return None


def remap_qwen_vl_vision_key(name: str) -> str | None:
    """Strip the HF vision-tower prefix.  Returns ``None`` for text keys."""
    if "rotary_emb" in name:
        return None
    if name.startswith("model.visual."):
        return name.removeprefix("model.visual.")
    return None


def require_complete_weight_load(module: nn.Module, loaded: set[str], component: str) -> None:
    missing = sorted(set(dict(module.named_parameters())) - loaded)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(f"QwenVL {component} checkpoint load missed {len(missing)} parameters: {preview}{suffix}")


def iter_qwen_vl_text_weights(local_dir: str, device: torch.device | str) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream Qwen3-VL's ``model.*`` and ``lm_head.*`` tensors."""
    yield from iter_safetensors_shards(local_dir, device=device, prefix="model.language_model.")
    yield from iter_safetensors_shards(local_dir, device=device, prefix="lm_head.")


def iter_qwen_vl_vision_weights(local_dir: str, device: torch.device | str) -> Iterator[tuple[str, torch.Tensor]]:
    yield from iter_safetensors_shards(local_dir, device=device, prefix="model.visual.")


def load_qwen_vl_text_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    return load_hf_weights(
        module,
        weights,
        stacked_params=LLAMA_STACKED_PARAMS,
        name_remapper=remap_qwen_vl_text_key,
    )


def load_qwen_vl_vision_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    return load_hf_weights(module, weights, name_remapper=remap_qwen_vl_vision_key)
