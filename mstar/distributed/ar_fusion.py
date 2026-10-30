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

# ws_id -> FlashInfer AllReduceFusionWorkspace (None once destroyed)
_WORKSPACES: list = []

# "auto" picks MNNVL whenever FlashInfer's capability vote passes, but mapping
# its multicast buffer can still fail (CUDA_ERROR_INVALID_VALUE inside some
# containers). The TRT-LLM backend uses IPC handles, not multicast, so it is
# tried before giving up on fusion.
_BACKENDS = ("auto", "trtllm")


def create_workspace(
    group: CommGroup, max_tokens: int, hidden: int, dtype: torch.dtype,
) -> int | None:
    """Collective over the TP ``group``: every member calls it, in the same order.
    Returns the id for ``allreduce_add_rmsnorm``, or None if FlashInfer cannot
    build a workspace here (caller keeps the unfused path)."""
    try:
        import flashinfer.comm as fc
        from flashinfer.comm.comm_backend import TorchDistBackend
    except Exception:
        logger.exception("FlashInfer all-reduce fusion unavailable; "
                         "keeping NCCL all-reduce + separate norm")
        return None
    for backend in _BACKENDS:
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


def destroy_workspace(ws_id: int) -> None:
    """Free a workspace ``create_workspace`` built. Every member of its group
    frees it at the same point, once no captured graph still uses it."""
    _WORKSPACES[ws_id].destroy()
    _WORKSPACES[ws_id] = None


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
