"""Varlen attention for encoders when flash_attn isn't installed.

Qwen3-Omni's audio/vision towers and BAGEL's ViT pack several segments into
one sequence and use flash_attn's varlen kernel to attend within each one.
There's no flash_attn wheel for ROCm, so there we use AITER's varlen kernel
(the same one the dense attention resource uses). With neither, callers fall
back to SDPA.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import logging

import torch

logger = logging.getLogger(__name__)

# attn_implementation name for varlen_attention. HF only passes cu_seqlens to
# implementations with "flash" in the name, others get one call per segment.
HF_VARLEN_ATTENTION = "flash_varlen_mstar"


def flash_attn_installed() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


@functools.cache
def varlen_attention_available() -> bool:
    """True if :func:`varlen_attention` has a kernel here."""
    from mstar.platforms import current_platform

    return current_platform.is_rocm() and importlib.util.find_spec("aiter") is not None


@torch.library.custom_op("mstar::varlen_attention", mutates_args=())
def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Same contract as flash_attn_varlen_func, with q [Tq, Hq, D] and k/v [Tk, Hk, D].
    Causal masks are aligned bottom-right."""
    from aiter import flash_attn_varlen_func

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q.to(torch.int32), cu_seqlens_k.to(torch.int32),
        max_seqlen_q, max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out[0] if isinstance(out, tuple) else out


@varlen_attention.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, softmax_scale=None):
    return q.new_empty(*q.shape[:-1], v.shape[-1])


def hf_encoder_attn_implementation() -> str:
    """attn_implementation for an HF encoder that packs segments with cu_seqlens.
    FA2 if flash_attn is installed, else :func:`varlen_attention`, else SDPA."""
    if flash_attn_installed():
        return "flash_attention_2"
    if varlen_attention_available():
        _register_hf_varlen_attention()
        return HF_VARLEN_ATTENTION
    logger.warning(
        "No flash_attn or varlen attention kernel here, HF encoders will run SDPA "
        "one segment at a time"
    )
    return "sdpa"


def build_hf_encoder(model_cls, config):
    """``model_cls._from_config`` with :func:`hf_encoder_attn_implementation`.
    Works on a copy since ``_from_config`` writes into the config it gets."""
    config = copy.deepcopy(config)
    implementation = hf_encoder_attn_implementation()
    if implementation != HF_VARLEN_ATTENTION:
        return model_cls._from_config(config, attn_implementation=implementation)
    # HF's init treats any "flash" name as flash_attn or a Hub kernel and
    # rejects ours, so build on SDPA and switch afterwards.
    model = model_cls._from_config(config, attn_implementation="sdpa")
    model.config._attn_implementation = HF_VARLEN_ATTENTION
    return model


@functools.cache
def _register_hf_varlen_attention() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(HF_VARLEN_ATTENTION, _hf_varlen_forward)


def _hf_varlen_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | torch.Tensor | None = None,
    max_length_k: int | torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HF attention interface for the packed path, like its flash_attention_2.
    Takes [1, H, T, D] and returns [1, T, H, D]."""
    if cu_seq_lens_q is None or attention_mask is not None or query.shape[0] != 1:
        raise ValueError(f"{HF_VARLEN_ATTENTION} serves only packed sequences described by cu_seqlens")
    if dropout and module.training:
        raise ValueError(f"{HF_VARLEN_ATTENTION} is inference-only")
    if cu_seq_lens_k is None:
        cu_seq_lens_k, max_length_k = cu_seq_lens_q, max_length_q
    causal = module.is_causal if is_causal is None else is_causal
    out = varlen_attention(
        query[0].transpose(0, 1), key[0].transpose(0, 1), value[0].transpose(0, 1),
        cu_seq_lens_q, cu_seq_lens_k,
        int(max_length_q), int(max_length_k),
        causal=bool(causal),
        softmax_scale=scaling,
    )
    return out.unsqueeze(0), None
