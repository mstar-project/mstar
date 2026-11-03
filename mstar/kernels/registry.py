"""Kernel ops by name: their implementations, and the once-per-shape table lookup.

An op is declared next to its kernels and registered under a stable name. It names the static
dimensions its tuned configs are keyed by (its shape) and the dynamic one its rows are keyed by
(its bucket, a token count), and lists its implementations::

    MOE_GEMM = register(KernelOp(
        name="moe_gemm", shape=("E", "N", "K", "top_k"), bucket="M",
        impls=(Impl("triton", check=_triton_unavailable, tolerance=Tolerance(1e-2, 1e-3)),),
    ))

An implementation says why it can't run a :class:`Target`, and how far a candidate config's output
may sit from the reference output its sweep compares it with. The code that launches an op asks
:meth:`KernelOp.lookup` for the tuned entry of its (dtype, shape) on its GPU and keeps its
built-in config where there is none. A lookup reads the tables once per process and logs what it
found, or the sweep command that would tune the shape. Its inputs (the package's tables,
``MSTAR_KERNEL_TABLES`` and the GPU model) are the same on every rank of a node, so the ranks of
a lockstep instance pick the same configs.
"""

from __future__ import annotations

import bisect
import importlib
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from mstar.kernels.table import Choice, Rows, TableEntry, device_key, dtype_name, find_entries, load_table

logger = logging.getLogger(__name__)

# Ops whose module registers them on import, so a tool can find one by name alone.
_OP_MODULES = {
    "moe_gemm": "mstar.utils.fused_moe.kernels",
    "moe_decode": "mstar.utils.fused_moe.decode",
}

_OPS: dict[str, KernelOp] = {}
_LOOKUPS: dict[tuple, TableEntry | None] = {}
_RESET_HOOKS: list[Callable[[], None]] = []


@dataclass(frozen=True)
class Target:
    """What an op runs: one dtype and shape on one device."""

    device: torch.device
    dtype: torch.dtype
    shape: Mapping[str, int]


@dataclass(frozen=True)
class Tolerance:
    """How close a candidate's output must sit to the reference output for the sweep to time it:
    every element finite, and max |out - ref| <= rtol * max |ref| + atol."""

    rtol: float
    atol: float = 0.0

    def error(self, out: torch.Tensor, ref: torch.Tensor) -> float:
        """The worst deviation as a fraction of what is allowed: at most 1 passes."""
        out, ref = out.float(), ref.float()
        if out.shape != ref.shape or not bool(torch.isfinite(out).all()):
            return math.inf
        worst = float((out - ref).abs().max()) if out.numel() else 0.0
        allowed = self.rtol * (float(ref.abs().max()) if ref.numel() else 0.0) + self.atol
        return worst / allowed if allowed > 0 else (0.0 if worst == 0 else math.inf)


EXACT = Tolerance(0.0)

# why an implementation can't run a target, or None when it can
Check = Callable[[Target], "str | None"]


@dataclass(frozen=True)
class Impl:
    """One implementation of an op, named by the table rows that configure it. ``check`` says why
    it can't run a target, None when it can (no check: wherever its op runs). ``tolerance`` is how
    far a candidate's output may sit from the reference output its sweep compares it with."""

    name: str
    check: Check | None = None
    tolerance: Tolerance = EXACT
    doc: str = ""

    def unavailable(self, target: Target) -> str | None:
        return None if self.check is None else self.check(target)


def describe_rows(rows: Rows) -> str:
    """``1-2 triton {...}, 3+ triton {...}``: the token counts each row takes."""
    if len(rows) == 1:
        return rows[0][1].describe()
    parts, low = [], 1
    for i, (key, choice) in enumerate(rows):
        span = f"{low}+" if i == len(rows) - 1 else (f"{low}" if key == low else f"{low}-{key}")
        parts.append(f"{span} {choice.describe()}")
        low = (key or 0) + 1
    return ", ".join(parts)


def row_bounds(rows: Rows) -> tuple[int, ...]:
    """The rows' upper bounds, all but the ``"*"`` row's: ``m`` tokens run row
    ``bisect_left(bounds, m)``."""
    return tuple(key for key, _ in rows[:-1])


@dataclass(frozen=True)
class KernelOp:
    """A stable kernel signature: the static dimensions that key its tuned configs (``shape``),
    the dynamic one its rows are keyed by (``bucket``, None for one row per shape), and the
    implementations its rows may name."""

    name: str
    shape: tuple[str, ...]
    bucket: str | None
    impls: tuple[Impl, ...]
    doc: str = ""

    def __post_init__(self) -> None:
        names = [impl.name for impl in self.impls]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"op {self.name!r} needs implementations with distinct names, got {names}")

    def impl(self, name: str) -> Impl:
        for impl in self.impls:
            if impl.name == name:
                return impl
        raise KeyError(f"op {self.name!r} has no implementation {name!r}; it has "
                       f"{[impl.name for impl in self.impls]}")

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

    def target(self, dtype: torch.dtype | str, shape: Mapping[str, int],
               device: torch.device | int | str | None = None) -> Target:
        self.shape_key(shape)
        return Target(_device(device), getattr(torch, dtype_name(dtype)),
                      {name: int(shape[name]) for name in self.shape})

    def unavailable(self, target: Target) -> dict[str, str]:
        """Why each implementation that can't run ``target`` can't."""
        reasons = {impl.name: impl.unavailable(target) for impl in self.impls}
        return {name: reason for name, reason in reasons.items() if reason is not None}

    def check_rows(self, rows: Rows, where: str) -> None:
        """Rows name this op's implementations, ascend, and end with the ``"*"`` row, which takes
        every count above the others. A table that stopped at a numeric row would hand its last
        row every larger count, which nobody measured."""
        if not rows:
            raise ValueError(f"{where}: {self.name} needs at least one row")
        for _, choice in rows:
            self.impl(choice.impl)
        keys = [key for key, _ in rows]
        if keys[-1] is not None or None in keys[:-1]:
            raise ValueError(f"{where}: {self.name}'s last row, and only it, must be '*'")
        if keys[:-1] != sorted(set(keys[:-1])):
            raise ValueError(f"{where}: {self.name}'s rows must ascend, one per count")
        if self.bucket is None and len(keys) > 1:
            raise ValueError(f"{where}: {self.name} has no bucket, so its '*' row is its only row")

    def row_for(self, rows: Rows, m: int) -> Choice:
        """The choice of ``rows`` that ``m`` tokens run."""
        return rows[bisect.bisect_left(row_bounds(rows), m)][1]

    def lookup(self, dtype: torch.dtype | str, shape: Mapping[str, int],
               device: torch.device | int | str | None = None) -> TableEntry | None:
        """The tuned entry for (dtype, shape) on ``device`` (the current GPU by default), or None
        where the caller's built-in configs apply: off GPU, and where no table has the shape. The
        ``MSTAR_KERNEL_TABLES`` directories come first, then the package's; in each, the GPU
        model's file, then its architecture's. Looked up and logged once per process; a caller
        keeps what it needs from the entry rather than looking it up per step."""
        dev = _device(device)
        if dev.type != "cuda":
            return None
        key = (self.name, dtype_name(dtype), self.shape_key(shape), dev)
        if key not in _LOOKUPS:
            _LOOKUPS[key] = self._lookup(dtype_name(dtype), shape, dev)
        return _LOOKUPS[key]

    def _lookup(self, dtype: str, shape: Mapping[str, int], device: torch.device) -> TableEntry | None:
        dev = device_key(device)
        if dev is None:
            return None
        what = f"{self.name} {dtype} {self.format_shape(shape)}"
        entry = next(iter(find_entries(self.name, dev, dtype, self.shape_key(shape))), None)
        if entry is None:
            logger.info("kernel table: no %s entry for %s; built-in configs. To tune it: %s",
                        what, dev.name, self.sweep_command(dtype, shape))
            return None
        self.check_rows(entry.rows, entry.source)
        logger.info("kernel table: %s on %s: %s (from %s, swept on %s, %s)", what, dev.name,
                    describe_rows(entry.rows), entry.source, entry.provenance.get("device"),
                    entry.provenance.get("date"))
        return entry


def _device(device: torch.device | int | str | None) -> torch.device:
    """The device a lookup is for: an index or an index-less ``cuda`` means that GPU or the
    current one; None, the current GPU, or the CPU without one."""
    if isinstance(device, int):
        return torch.device("cuda", device)
    if device is None:
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


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


def on_reset(hook: Callable[[], None]) -> Callable[[], None]:
    """Run ``hook`` in :func:`reset_lookups`: for a module's own cache of what it looked up."""
    _RESET_HOOKS.append(hook)
    return hook


def reset_lookups() -> None:
    """Forget lookups and loaded tables, so the next lookup reads the files again (tests, and a
    sweep that just wrote one). Code already compiled or captured keeps the configs it looked
    up: a process serves the tables it started with."""
    _LOOKUPS.clear()
    load_table.cache_clear()
    for hook in _RESET_HOOKS:
        hook()
