"""Kernel ops and their tuned configs, kept as data.

An op is a stable kernel signature, such as ``moe_gemm`` (the grouped expert GEMMs of
:func:`mstar.utils.fused_moe.fused_experts`). It names the static dimensions its configs are keyed
by (its shape) and the dynamic one its rows are keyed by (its bucket, e.g. tokens), and lists its
implementations.

Which config runs at which token counts lives in JSON tables, one file per (op, GPU model or
architecture) under ``mstar/kernels/tables/``, plus any directory listed in
``MSTAR_KERNEL_TABLES``. ``python -m benchmark.kernels.sweep`` times candidate configs at a
model's shapes on the local GPU, checks each against a reference output, and writes the winners.
The code that launches an op looks its shape up once per process and keeps its built-in config
where no table has it, so a new table changes no model code and adds nothing per step.
"""

from mstar.kernels.registry import Impl, KernelOp, Target, Tolerance, get_op, op_names, register
from mstar.kernels.table import Choice, DeviceKey, TableEntry, device_key, table_dirs

__all__ = [
    "Choice",
    "DeviceKey",
    "Impl",
    "KernelOp",
    "TableEntry",
    "Target",
    "Tolerance",
    "device_key",
    "get_op",
    "op_names",
    "register",
    "table_dirs",
]
