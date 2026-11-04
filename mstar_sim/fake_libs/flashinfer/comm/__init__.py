"""``flashinfer.comm``: the one-kernel all-reduce + residual add + RMSNorm.

``create_allreduce_fusion_workspace`` builds a workspace on every rank, so
``mstar.distributed.ar_fusion`` takes its fused path as on a GPU. A fused call
returns its outputs' shapes and is priced as the NCCL all-reduce of its input
plus the norm.
"""

from __future__ import annotations

import enum

import torch

from mstar_sim import clock

# FlashInfer's member names (functional form: they are not snake_case)
AllReduceFusionPattern = enum.IntEnum("AllReduceFusionPattern", [
    "kAllReduce", "kARResidualRMSNorm", "kARResidualRMSNormFP8Quant",
    "kARResidualRMSNormFP4Quant", "kARResidualRMSNormOutFP8Quant", "kARResidualRMSNormOutFP4Quant",
], start=0)


def _missing(name: str):
    raise NotImplementedError(f"mstar_sim: flashinfer.comm.{name} has no stand-in yet")


class AllReduceFusionWorkspace:
    """What a workspace is to its caller: its group's size and limits."""

    def __init__(self, backend, world_size: int, rank: int, max_token_num: int, hidden_dim: int, dtype):
        self.backend, self.world_size, self.rank = backend, world_size, rank
        self.max_token_num, self.hidden_dim, self.dtype = max_token_num, hidden_dim, dtype

    def is_buffer_size_sufficient(self, tp_size: int, num_tokens: int, hidden_dim: int, dtype=None,
                                  use_oneshot=None) -> bool:
        """A message fits when its bytes fit the workspace's ``max_token_num x hidden_dim``."""
        size = torch.empty((), dtype=dtype or self.dtype).element_size()
        ws = torch.empty((), dtype=self.dtype).element_size()
        return num_tokens * hidden_dim * size <= self.max_token_num * self.hidden_dim * ws

    def destroy(self) -> None:
        pass


def create_allreduce_fusion_workspace(backend="auto", world_size=1, rank=0, max_token_num=0, hidden_dim=0,
                                      dtype=torch.bfloat16, comm_backend=None, group=None, **kwargs):
    return AllReduceFusionWorkspace(backend, world_size, rank, max_token_num, hidden_dim, dtype)


def _norm_cost(input, *args, **kwargs):
    return 5.0 * input.numel(), 4.0 * input.numel() * input.element_size()


@clock.kernel_fn("flashinfer.allreduce_fusion_norm", cost=_norm_cost)
def _norm(input, residual_out, norm_out):
    return None  # writes residual_out and norm_out in place


def allreduce_fusion(input, workspace, pattern=AllReduceFusionPattern.kARResidualRMSNorm, *, residual_in=None,
                     residual_out=None, norm_out=None, rms_gamma=None, rms_eps=1e-6, **kwargs):
    if pattern not in (AllReduceFusionPattern.kAllReduce, AllReduceFusionPattern.kARResidualRMSNorm):
        _missing(f"allreduce_fusion(pattern={pattern!r})")
    from mstar_sim import cost

    if not clock.FAST:
        clock.charge(cost.collective_seconds("all_reduce", workspace.world_size,
                                             input.numel() * input.element_size(),
                                             graph=clock.capturing() is not None))
    _norm(input, residual_out, norm_out)
    return norm_out if norm_out is not None else input
