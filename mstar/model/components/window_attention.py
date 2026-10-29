"""Bidirectional attention over a batch of equal-length windows.

An encoder that attends within fixed windows (Whisper's 1500 frames) is a dense
``[batch, seq, heads, head_dim]`` problem: no mask, no cache, nothing to plan.
Off ROCm this is SDPA, as the encoders called it before. On AMD GPUs SDPA runs
AOTriton's flash kernel; AITER's FMHA (the kernel family the ragged attention
resource runs there) is faster at these shapes, and it reads the projections'
``[batch, seq, heads, head_dim]`` views as they are, so the transposes around
SDPA go as well. ``benchmark/asr_perf/whisper_encoder.py`` times the encoder
either way.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

from mstar.platforms.dispatch import PlatformDispatch

logger = logging.getLogger(__name__)

# what AITER's FMHA forward takes
_AITER_DTYPES = (torch.bfloat16, torch.float16)
_AITER_MAX_HEAD_DIM = 256


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


@torch.library.custom_op("mstar::aiter_window_attention", mutates_args=())
def _aiter_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from aiter import flash_attn_func

    out = flash_attn_func(q, k, v)
    out = out[0] if isinstance(out, tuple) else out
    # the layout the fake below promises
    return out.contiguous()


@_aiter_window_attention.register_fake
def _(q, k, v):
    return q.new_empty(*q.shape[:-1], v.shape[-1])


def _rocm() -> Callable[..., torch.Tensor]:
    if importlib.util.find_spec("aiter") is None:
        logger.warning("AITER is not installed: window attention runs SDPA (AOTriton) on this GPU")
        return _sdpa

    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.dtype in _AITER_DTYPES and q.shape[-1] <= _AITER_MAX_HEAD_DIM and q.stride(-1) == 1:
            return _aiter_window_attention(q, k, v)
        _warn_sdpa(q.dtype, q.shape[-1])
        return _sdpa(q, k, v)

    return attend


@functools.cache
def _warn_sdpa(dtype: torch.dtype, head_dim: int) -> None:
    logger.warning("window attention: AITER's FMHA does not take %s with head_dim %d; running SDPA", dtype, head_dim)


_window_attention = PlatformDispatch("mstar::window_attention", rocm=_rocm, default=lambda: _sdpa)


def window_attention_kernel() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """This platform's unmasked attention within each window: ``q``, ``k``,
    ``v`` and the result are ``[batch, seq, heads, head_dim]`` (views with a
    unit last stride are fine), scaled by ``head_dim ** -0.5``.

    A layer binds it when it is built: picking it from inside a compiled
    forward would break the graph (the pick takes a lock dynamo can't trace).
    """
    return _window_attention.resolve()
