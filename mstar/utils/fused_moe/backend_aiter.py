"""AITER's fused MoE as a FusedMoEBackend, for AMD Instinct GPUs.

aiter.fused_moe runs CK two-stage or asm kernels on weights pre-shuffled into
16x16 tiles, so prepare_weights shuffles both expert tensors once after
loading. AITER picks kernels per token count from its tuned tables, or from
heuristics for sizes they don't cover. Keep AITER_ONLINE_TUNE unset when
serving, otherwise it tunes in-process the first time it sees new sizes.
Its output isn't bitwise reproducible across calls, unlike Triton's.

Only imported when a model picks this backend. Nothing here runs on CUDA.
"""

from __future__ import annotations

import importlib.util

import torch
from torch import nn

from mstar.utils.fused_moe.backend import FusedMoEBackend
from mstar.utils.fused_moe.config import MoEProblem

# CK's two-stage MoE GEMMs tile the intermediate dimension by 128.
_INTER_ALIGN = 128


@torch.library.custom_op("mstar::aiter_fused_moe", mutates_args=())
def aiter_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    from aiter.fused_moe import fused_moe

    # AITER picks kernels based on this flag, which may not survive a compiled
    # graph. These are always the weights prepare_weights shuffled.
    w1.is_shuffled = True
    w2.is_shuffled = True
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)


@aiter_fused_moe.register_fake
def _(hidden_states, w1, w2, topk_weights, topk_ids):
    return torch.empty_like(hidden_states)


class AiterFusedMoE(FusedMoEBackend):
    name = "aiter"

    def __init__(self, problem: MoEProblem):
        from mstar.platforms import current_platform

        super().__init__(problem)
        if not current_platform.is_rocm():
            raise ValueError(f"MoE backend 'aiter' runs on AMD GPUs only, not {current_platform.name}")
        if importlib.util.find_spec("aiter") is None:
            raise ValueError("MoE backend 'aiter' needs the aiter package")
        if problem.intermediate_size % _INTER_ALIGN:
            raise ValueError(
                f"MoE backend 'aiter': the per-rank expert intermediate size "
                f"{problem.intermediate_size} is not a multiple of {_INTER_ALIGN}; use 'triton'"
            )
        self._prepared = False

    def prepare_weights(self, w1: nn.Parameter, w2: nn.Parameter) -> None:
        if self._prepared:
            return
        from aiter.ops.shuffle import shuffle_weight

        for w in (w1, w2):
            if w.dtype != torch.bfloat16:
                raise ValueError(f"MoE backend 'aiter' takes bf16 expert weights, got {w.dtype}")
            w.data = shuffle_weight(w.data, layout=(16, 16))
        self._prepared = True

    def __call__(self, hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=False):
        if skip_invalid:
            raise NotImplementedError(
                "MoE backend 'aiter' can't skip padding rows or another rank's experts "
                "(ep_config parallel: ep, or mask_padding); use 'triton'"
            )
        if not self._prepared:
            raise RuntimeError("AiterFusedMoE: prepare_weights was not called after loading")
        return aiter_fused_moe(
            hidden_states, w1, w2, topk_weights.to(torch.float32), topk_ids.to(torch.int32),
        )
