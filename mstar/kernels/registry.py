"""Kernel ops by name, their backends, and the once-per-shape table lookup.

An op is declared next to its kernels and registered under a stable name::

    MOE_GEMM = register(KernelOp(
        name="moe_gemm", shape=("E", "N", "K", "top_k"), bucket="M",
        backends=(Backend("triton", _triton_unavailable),),
    ))

The code that launches it asks ``MOE_GEMM.lookup(backend, dtype, shape, device)`` for the
tuned entry and keeps its default config where there is none. A lookup reads the tables once
per (op, backend, dtype, shape, device) and process, and logs what it found, or the sweep
command that would tune it. Its inputs, the package's tables, ``MSTAR_KERNEL_TABLES`` and
the GPU model, are the same on every rank of a node, so every rank picks the same config.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from mstar.kernels.table import TableEntry, device_key, dtype_name, find_entry, load_table

logger = logging.getLogger(__name__)

# Ops whose module registers them on import, so a tool can find one by name alone.
_OP_MODULES = {
    "moe_gemm": "mstar.utils.fused_moe.kernels",
}

_OPS: dict[str, KernelOp] = {}
_LOOKUPS: dict[tuple, TableEntry | None] = {}

# (device, dtype, shape) -> why the backend can't run it there, or None when it can
Availability = Callable[[torch.device, torch.dtype, Mapping[str, int]], "str | None"]


@dataclass(frozen=True)
class Backend:
    """One implementation of an op. ``check`` says why it can't run a (device, dtype, shape),
    or returns None when it can; without one it runs everywhere its op does."""

    name: str
    check: Availability | None = None

    def unavailable(self, device: torch.device, dtype: torch.dtype, shape: Mapping[str, int]) -> str | None:
        return None if self.check is None else self.check(device, dtype, shape)


@dataclass(frozen=True)
class KernelOp:
    """A stable kernel signature: the static dimensions that select a tuned entry (``shape``),
    the dynamic one its rows are keyed by (``bucket``, None for one row per shape), and its
    backends in order of preference."""

    name: str
    shape: tuple[str, ...]
    bucket: str | None
    backends: tuple[Backend, ...]
    doc: str = ""

    def backend(self, name: str) -> Backend:
        for backend in self.backends:
            if backend.name == name:
                return backend
        raise KeyError(f"op {self.name!r} has no backend {name!r}; it has "
                       f"{[b.name for b in self.backends]}")

    def shape_key(self, shape: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
        if set(shape) != set(self.shape):
            raise ValueError(f"op {self.name!r} is keyed by {self.shape}, got {sorted(shape)}")
        return tuple(sorted((name, int(value)) for name, value in shape.items()))

    def format_shape(self, shape: Mapping[str, int]) -> str:
        """``E=128,N=1536,K=2048,top_k=8``: in declaration order, as the sweep takes it."""
        return ",".join(f"{name}={int(shape[name])}" for name in self.shape)

    def sweep_command(self, dtype: torch.dtype | str, shape: Mapping[str, int]) -> str:
        return (f"python -m benchmark.kernels.sweep {self.name} --dtype {dtype_name(dtype)} "
                f"--shape {self.format_shape(shape)}")

    def lookup(
        self, backend: str, dtype: torch.dtype | str, shape: Mapping[str, int],
        device: torch.device | int | None = None,
    ) -> TableEntry | None:
        """The tuned entry for ``shape`` on ``device`` (the current CUDA device by default), or
        None where no table has one. Resolved and logged once per process; the caller keeps the
        result (or its row) rather than calling this per step."""
        index = _device_index(device)
        key = (self.name, backend, dtype_name(dtype), self.shape_key(shape), index)
        if key in _LOOKUPS:
            return _LOOKUPS[key]
        self.backend(backend)
        dev = None if index is None else device_key(index)
        entry = None if dev is None else find_entry(self.name, dev, key[2], key[3], backend)
        _LOOKUPS[key] = entry
        what = f"{self.name} {key[2]} {self.format_shape(shape)} ({backend})"
        if entry is not None:
            swept = entry.provenance
            logger.info("kernel table: %s from %s (swept on %s, %s)", what, entry.source,
                        swept.get("device"), swept.get("date"))
        elif dev is not None:
            logger.info("kernel table: no %s entry for %s; default configs. To tune it: %s",
                        what, dev.name, self.sweep_command(dtype, shape))
        return entry


def _device_index(device: torch.device | int | None) -> int | None:
    """The CUDA device index a lookup is for, or None off CUDA/ROCm."""
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        device = device.index
    if device is None:
        return torch.cuda.current_device() if torch.cuda.is_available() else None
    return int(device)


def register(op: KernelOp) -> KernelOp:
    """Add ``op`` under its name. A second op with the same name is a bug, not an override."""
    existing = _OPS.get(op.name)
    if existing is not None and existing != op:
        raise ValueError(f"kernel op {op.name!r} is already registered")
    _OPS[op.name] = op
    return op


def get_op(name: str) -> KernelOp:
    """The op registered as ``name``, importing the module that declares it if needed."""
    if name not in _OPS and name in _OP_MODULES:
        importlib.import_module(_OP_MODULES[name])
    try:
        return _OPS[name]
    except KeyError:
        raise KeyError(f"no kernel op {name!r}; known: {op_names()}") from None


def op_names() -> list[str]:
    return sorted(set(_OPS) | set(_OP_MODULES))


def reset_lookups() -> None:
    """Forget resolved lookups and loaded tables, so the next lookup reads the files again
    (tests, and a sweep that just wrote one). Code already compiled or captured keeps the
    configs it resolved: a process serves the tables it started with."""
    _LOOKUPS.clear()
    load_table.cache_clear()
