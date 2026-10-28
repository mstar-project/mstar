"""Kernel ops, the backends that run them, and their tuned launch configs kept as data.

An op is a stable kernel signature, such as ``moe_gemm`` (the grouped expert GEMMs of
:func:`mstar.utils.fused_moe.fused_experts`). It names the static dimensions that select a
tuned config (its shape) and the dynamic one its rows are keyed by (its bucket, e.g. tokens).

Tuned configs are JSON tables, one file per (op, device) under ``mstar/kernels/tables/``,
plus any directory listed in ``MSTAR_KERNEL_TABLES``. ``python -m benchmark.kernels.sweep``
measures candidates at a model's shapes on the local GPU and writes them. The code that
launches an op looks its config up once per shape and process, so a new table changes no
model code and adds nothing per step.
"""

from mstar.kernels.registry import Backend, KernelOp, get_op, op_names, register
from mstar.kernels.table import DeviceKey, TableEntry, device_key, table_dirs

__all__ = [
    "Backend",
    "DeviceKey",
    "KernelOp",
    "TableEntry",
    "device_key",
    "get_op",
    "op_names",
    "register",
    "table_dirs",
]
