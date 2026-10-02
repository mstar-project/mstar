"""Fused tensor-parallel all-reduce + residual add + RMSNorm.

A row-parallel projection's output is a per-rank partial sum. The layer then
adds it to the residual stream and normalises the result for the next block.
Done separately that is an NCCL all-reduce, an add and a norm kernel per
projection -- three launches and three trips through HBM for a tensor that
is only ``[tokens, hidden]``. FlashInfer's ``allreduce_fusion`` does all of
it in one kernel over a peer-mapped workspace, which at decode shapes is
~4-6us against ~11us for the unfused sequence.

The kernel runs behind a ``torch.library`` op so dynamo and the CUDA-graph
capture see one opaque node; the workspace, which is a Python object, is
reached through a small registry keyed by an int.
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
    """Collective over the TP group: every member must call it, in the same
    order. Returns the id to pass to ``allreduce_add_rmsnorm``, or None if
    FlashInfer cannot build a workspace on this topology (the caller then
    keeps the unfused path)."""
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
