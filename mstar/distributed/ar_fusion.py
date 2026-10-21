"""Fused tensor-parallel all-reduce + residual add + RMSNorm.

Replaces the NCCL all-reduce, add and norm after a row-parallel projection
with FlashInfer's one-kernel ``allreduce_fusion`` (~4-6us vs ~11us unfused at
decode shapes). It runs behind a ``torch.library`` op so dynamo and CUDA-graph
capture see one opaque node; the workspace is reached through an int-keyed
registry.
"""
from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ws_id -> FlashInfer AllReduceFusionWorkspace
_WORKSPACES: list = []

# "auto" picks MNNVL whenever FlashInfer's capability vote passes, but mapping
# its multicast buffer can still fail (CUDA_ERROR_INVALID_VALUE inside some
# containers). The TRT-LLM backend uses IPC handles, not multicast, so it is
# tried before giving up on fusion.
_BACKENDS = ("auto", "trtllm")


def _all_ranks(ok: bool, cpu_group) -> bool:
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())


def create_workspace(
    rank: int, world_size: int, cpu_group, max_tokens: int, hidden: int,
    dtype: torch.dtype,
) -> int | None:
    """Collective over the TP group: every member calls it, in the same order.
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
                backend=backend, world_size=world_size, rank=rank,
                max_token_num=max_tokens, hidden_dim=hidden, dtype=dtype,
                comm_backend=TorchDistBackend(group=cpu_group), group=cpu_group,
            )
        except Exception:
            logger.warning("FlashInfer all-reduce fusion backend %r failed",
                           backend, exc_info=True)
        # Every rank must use the same backend, or the next collective
        # deadlocks: keep one only if it built on all of them.
        try:
            agreed = _all_ranks(ws is not None, cpu_group)
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
        # vLLM's allreduce_rms_fusion and flashinfer-ai/flashinfer#1223).
        trigger_completion_at_end=True,
    )
    return norm_out, residual_out


@allreduce_add_rmsnorm.register_fake
def _(x, residual, weight, eps, weight_bias, ws_id):
    return torch.empty_like(x), torch.empty_like(residual)


@torch.library.custom_op("mstar::allreduce", mutates_args=())
def allreduce(x: torch.Tensor, ws_id: int) -> torch.Tensor:
    """``allreduce(x)`` alone, by the workspace's one-shot kernel."""
    import flashinfer.comm as fc

    out = torch.empty_like(x)
    fc.allreduce_fusion(
        input=x, workspace=_WORKSPACES[ws_id], pattern=fc.AllReduceFusionPattern.kAllReduce,
        launch_with_pdl=True, output=out, fp32_acc=True, trigger_completion_at_end=True,
    )
    return out


@allreduce.register_fake
def _(x, ws_id):
    return torch.empty_like(x)

