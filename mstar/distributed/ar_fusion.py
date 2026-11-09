"""Fused tensor-parallel all-reduce + residual add + RMSNorm.

Replaces the NCCL all-reduce, add and norm after a row-parallel projection
with FlashInfer's one-kernel ``allreduce_fusion`` (~4-6us vs ~11us unfused at
decode shapes). It runs behind a ``torch.library`` op so dynamo and CUDA-graph
capture see one opaque node; the workspace is reached through an int-keyed
registry.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.distributed.communication import CommGroup

logger = logging.getLogger(__name__)

# ws_id -> FlashInfer AllReduceFusionWorkspace
_WORKSPACES: list = []
# ws_id -> the NCCL group ``allreduce_flashinfer`` falls back to for messages its workspace can't take
_FALLBACK_GROUPS: dict = {}
# (ws_id, world, rows, cols, dtype) -> whether the workspace takes the message
_FITS: dict = {}

# "auto" picks MNNVL whenever FlashInfer's capability vote passes, but mapping
# its multicast buffer can still fail (CUDA_ERROR_INVALID_VALUE inside some
# containers). The TRT-LLM backend uses IPC handles, not multicast, so it is
# tried before giving up on fusion.
_BACKENDS = ("auto", "trtllm")


def create_workspace(
    group: CommGroup, max_tokens: int, hidden: int, dtype: torch.dtype,
    backends: tuple[str, ...] = _BACKENDS,
) -> int | None:
    """Collective over the TP ``group``: every member calls it, in the same order.
    Returns the id for ``allreduce_add_rmsnorm``, or None if FlashInfer cannot
    build a workspace here (caller keeps the unfused path). ``backends``: the
    ones tried, in order (the MoE finalize fusion takes only "trtllm")."""
    try:
        import flashinfer.comm as fc
        from flashinfer.comm.comm_backend import TorchDistBackend
    except Exception:
        logger.exception("FlashInfer all-reduce fusion unavailable; "
                         "keeping NCCL all-reduce + separate norm")
        return None
    for backend in backends:
        ws = None
        try:
            ws = fc.create_allreduce_fusion_workspace(
                backend=backend, world_size=group.world_size, rank=group.rank,
                max_token_num=max_tokens, hidden_dim=hidden, dtype=dtype,
                comm_backend=TorchDistBackend(group=group.cpu_group), group=group.cpu_group,
            )
        except Exception:
            logger.warning("FlashInfer all-reduce fusion backend %r failed",
                           backend, exc_info=True)
        # Every rank must use the same backend, or the next collective
        # deadlocks: keep one only if it built on all of them.
        try:
            agreed = group.agree([int(ws is not None)])[0]
        except RuntimeError:
            # A rank failed between FlashInfer's own collectives, so this vote
            # met a different one; the gloo timeout closes the group on every
            # member, and every rank falls back here.
            logger.exception("TP ranks lost sync building the fusion workspace; "
                             "keeping NCCL all-reduce + separate norm")
            if ws is not None:
                ws.destroy()
            return None
        if agreed:
            _WORKSPACES.append(ws)
            logger.info("all-reduce fusion workspace: %s (backend %r), up to "
                        "%d tokens x %d", type(ws).__name__, backend,
                        max_tokens, hidden)
            return len(_WORKSPACES) - 1
        if ws is not None:
            ws.destroy()
    logger.warning("no FlashInfer all-reduce fusion backend works here; "
                   "keeping NCCL all-reduce + separate norm")
    return None


@torch.library.custom_op("mstar::allreduce_add_rmsnorm", mutates_args=())
def allreduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float, weight_bias: float, ws_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``r = allreduce(x) + residual``; returns ``(rmsnorm(r), r)``, with the
    norm scaling by ``weight + weight_bias``."""
    import flashinfer.comm as fc

    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    fc.allreduce_fusion(
        input=x, workspace=_WORKSPACES[ws_id],
        pattern=fc.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True, residual_in=residual, residual_out=residual_out,
        norm_out=norm_out, rms_gamma=weight, rms_eps=eps, fp32_acc=True,
        weight_bias=weight_bias,
        # Complete at the end: with an early trigger a PDL-launched consumer
        # can read the one-shot Lamport buffer before it is committed (see
        # flashinfer-ai/flashinfer#1223).
        trigger_completion_at_end=True,
    )
    return norm_out, residual_out


@allreduce_add_rmsnorm.register_fake
def _(x, residual, weight, eps, weight_bias, ws_id):
    return torch.empty_like(x), torch.empty_like(residual)


@torch.library.custom_op("mstar::allreduce_moe_finalize_rmsnorm", mutates_args=())
def allreduce_moe_finalize_rmsnorm(
    permuted: torch.Tensor, expanded_idx: torch.Tensor, scales: torch.Tensor, residual: torch.Tensor,
    weight: torch.Tensor, eps: float, weight_bias: float, ws_id: int,
) -> torch.Tensor:
    """``rmsnorm(allreduce(finalize(permuted)) + residual)`` in one kernel (FlashInfer's
    ``kMoEFinalizeARResidualRMSNorm`` on a TRT-LLM workspace): each rank's routed experts' unfinalized output
    (``permuted`` rows of the MoE's permuted buffer, ``expanded_idx [tokens, top_k]`` into it, ``scales`` the
    routing weights) summed over its top-k, reduced over the group, normed. ``residual [tokens, hidden]``."""
    import flashinfer.comm as fc

    norm_out = torch.empty_like(residual)
    fc.allreduce_fusion(
        input=permuted, workspace=_WORKSPACES[ws_id],
        pattern=fc.AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        # no PDL: this path takes no trigger_completion_at_end, and an early trigger lets a PDL-launched
        # consumer read the one-shot Lamport buffer before it is committed (flashinfer-ai/flashinfer#1223)
        launch_with_pdl=False, residual_in=residual, residual_out=torch.empty_like(residual),
        norm_out=norm_out, rms_gamma=weight, rms_eps=eps, weight_bias=weight_bias,
        expanded_idx_to_permuted_idx=expanded_idx, expert_scale_factor=scales,
    )
    return norm_out


@allreduce_moe_finalize_rmsnorm.register_fake
def _(permuted, expanded_idx, scales, residual, weight, eps, weight_bias, ws_id):
    return torch.empty_like(residual)


@torch.library.custom_op("mstar::allreduce_oneshot", mutates_args=())
def allreduce_oneshot(x: torch.Tensor, ws_id: int) -> torch.Tensor:
    """``allreduce(x)`` with the same one-kernel collective, and no norm."""
    import flashinfer.comm as fc

    out = torch.empty_like(x)
    fc.allreduce_fusion(
        input=x, workspace=_WORKSPACES[ws_id],
        pattern=fc.AllReduceFusionPattern.kAllReduce,
        launch_with_pdl=True, output=out, fp32_acc=True,
        trigger_completion_at_end=True,
    )
    return out


@allreduce_oneshot.register_fake
def _(x, ws_id):
    return torch.empty_like(x)


@torch.library.custom_op("mstar::allreduce_flashinfer", mutates_args=())
def allreduce_flashinfer(x: torch.Tensor, ws_id: int, world_size: int) -> torch.Tensor:
    """``allreduce(x)``: FlashInfer's one-shot kernel where workspace ``ws_id`` takes the message,
    else NCCL over the group registered for it. The choice is made here, per call, so a compiled
    caller (an opaque op to Dynamo) keeps the one-shot kernel instead of tracing the NCCL branch.
    It follows the message's shape and dtype only, so every rank of the group takes the same one."""
    key = (ws_id, world_size, *x.shape, x.dtype)
    fits = _FITS.get(key)
    if fits is None:
        fits = _FITS[key] = bool(_WORKSPACES[ws_id].is_buffer_size_sufficient(world_size, *x.shape, x.dtype))
    if fits:
        return allreduce_oneshot(x, ws_id)
    import torch.distributed as dist

    out = x.clone()
    dist.all_reduce(out, group=_FALLBACK_GROUPS[ws_id])
    return out


@allreduce_flashinfer.register_fake
def _(x, ws_id, world_size):
    return torch.empty_like(x)
