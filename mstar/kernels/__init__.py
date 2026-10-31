"""Kernel ops and their tuned configs, kept as data.

An op (``registry``) names the dimensions its configs are keyed by; its configs live in JSON
tables per op and GPU (``table``, ``mstar/kernels/tables/``), which
``python -m benchmark.kernels.sweep`` writes. The code that launches an op looks its shape up once
per process and keeps its built-in config where no table has it, so a new table changes no model
code and adds nothing per step.
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
