"""``flashinfer.comm``: the one-kernel all-reduce fusion has no stand-in.

Building a workspace raises, so ``mstar.distributed.ar_fusion`` keeps its
NCCL all-reduce + separate norm path."""

from __future__ import annotations

import enum

# FlashInfer's member names (functional form: they are not snake_case)
AllReduceFusionPattern = enum.IntEnum("AllReduceFusionPattern", [
    "kAllReduce", "kARResidualRMSNorm", "kARResidualRMSNormFP8Quant",
    "kARResidualRMSNormFP4Quant", "kARResidualRMSNormOutFP8Quant", "kARResidualRMSNormOutFP4Quant",
], start=0)


def _missing(name: str):
    raise NotImplementedError(f"mstar_sim: flashinfer.comm.{name} has no stand-in yet")


def create_allreduce_fusion_workspace(*args, **kwargs):
    _missing("create_allreduce_fusion_workspace")


def allreduce_fusion(*args, **kwargs):
    _missing("allreduce_fusion")
