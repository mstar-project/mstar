"""Fused tensor-parallel all-reduce + residual add + RMSNorm.

Replaces the NCCL all-reduce, add and norm after a row-parallel projection
with FlashInfer's one-kernel ``allreduce_fusion`` (~4-6us vs ~11us unfused at
decode shapes). On ROCm the same pattern runs on AITER's custom all-reduce
(``fused_ar_rms``), as vLLM does there. It runs behind a ``torch.library`` op
so dynamo and CUDA-graph capture see one opaque node; the workspace is reached
through an int-keyed registry.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ws_id -> FlashInfer AllReduceFusionWorkspace, or AITER CustomAllreduce on ROCm
_WORKSPACES: list = []
_IS_ROCM = torch.version.hip is not None


def create_workspace(
    rank: int, world_size: int, cpu_group, max_tokens: int, hidden: int,
    dtype: torch.dtype,
) -> int | None:
    """Collective over the TP group: every member calls it, in the same order.
    Returns the id for ``allreduce_add_rmsnorm``, or None if FlashInfer cannot
    build a workspace here (caller keeps the unfused path)."""
    if _IS_ROCM:
        return _create_aiter_workspace(cpu_group, max_tokens, hidden, dtype)
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


def _create_aiter_workspace(
    cpu_group, max_tokens: int, hidden: int, dtype: torch.dtype,
) -> int | None:
    # FlashInfer-AMD has no allreduce_fusion. AITER stages inputs through an IPC
    # buffer and its two-stage kernel needs twice the input; 128 MiB is vLLM's size.
    nbytes = max_tokens * hidden * torch.finfo(dtype).bits // 8
    try:
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        ca = CustomAllreduce(cpu_group, torch.cuda.current_device(),
                             max_size=max(128 << 20, 2 * nbytes))
    except Exception:
        logger.exception("AITER custom all-reduce unavailable; "
                         "keeping RCCL all-reduce + separate norm")
        return None
    if ca.disabled:
        logger.warning("AITER custom all-reduce disabled for this group; "
                       "keeping RCCL all-reduce + separate norm")
        return None
    _WORKSPACES.append(ca)
    logger.info("all-reduce fusion workspace: AITER CustomAllreduce, up to %d tokens x %d",
                max_tokens, hidden)
    return len(_WORKSPACES) - 1


def _aiter_one_stage(ca, x: torch.Tensor) -> bool:
    """vLLM's choice between AITER's one- and two-stage kernels (rocm_aiter_ops)."""
    hidden = x.shape[-1]
    nbytes = x.numel() * x.element_size()
    if ca.world_size == 2:
        size_ok = True
    elif ca.fully_connected and ca.world_size <= 4:
        size_ok = nbytes < 256 * 1024
    elif ca.fully_connected and ca.world_size <= 8:
        size_ok = nbytes < 128 * 1024
    else:
        size_ok = False
    return hidden % 8 == 0 and hidden // 8 <= 1024 and x.shape[0] <= 80 and size_ok


@torch.library.custom_op("mstar::allreduce_add_rmsnorm", mutates_args=())
def allreduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float, weight_bias: float, ws_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``r = allreduce(x) + residual``; returns ``(rmsnorm(r), r)``, with the
    norm scaling by ``weight + weight_bias``."""
    if _IS_ROCM:
        # gemma_norm scales by (1 + weight), the only nonzero weight_bias in use
        assert weight_bias in (0.0, 1.0), weight_bias
        ca = _WORKSPACES[ws_id]
        return ca.fused_ar_rms(
            x, residual, w=weight.to(x.dtype), eps=eps,
            use_1stage=_aiter_one_stage(ca, x), gemma_norm=weight_bias == 1.0,
        )
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
