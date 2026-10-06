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

logger = logging.getLogger(__name__)

# ws_id -> FlashInfer AllReduceFusionWorkspace
_WORKSPACES: list = []


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

        ws = fc.create_allreduce_fusion_workspace(
            backend="auto", world_size=world_size, rank=rank,
            max_token_num=max_tokens, hidden_dim=hidden, dtype=dtype,
            comm_backend=TorchDistBackend(group=cpu_group), group=cpu_group,
        )
    except Exception:
        logger.exception("FlashInfer all-reduce fusion workspace unavailable; "
                         "keeping NCCL all-reduce + separate norm")
        return None
    _WORKSPACES.append(ws)
    logger.info("all-reduce fusion workspace: %s, up to %d tokens x %d",
                type(ws).__name__, max_tokens, hidden)
    return len(_WORKSPACES) - 1


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
