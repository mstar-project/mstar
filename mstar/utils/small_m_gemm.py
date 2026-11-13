"""Small-M GEMMs for decode: ``x (M, K) @ w (N, K).T`` through the backend a measured table names for that
shape and token count, cuBLAS otherwise.

At decode a projection's M is a capture bucket's row count (1..64), where cuBLAS spends a few microseconds of
launch and small-wave overhead on every call; FlashInfer's small-M bf16 kernels (``tinygemm``, ``tgv``, the
``cute-dsl`` direct / split-K ones, the Kimi-K3 router GEMM) beat it on some shapes of some GPUs and lose on
others, so the choice is a table measured on the GPU (``benchmark/kernels/k31_dense_gemm.py --table-out``),
keyed by compute capability, ``(N, K, out dtype)`` and M. A shape or M the table doesn't name takes cuBLAS,
as ``F.linear`` / ``torch.mm`` would. ``linear`` is a custom op, opaque to torch.compile and captured whole.
"""
from __future__ import annotations

import json
from collections.abc import Callable

import torch
import torch.nn.functional as F

_OUT = {"bf16": torch.bfloat16, "fp32": torch.float32}
# capability -> {(n, k, out): [(m, backend)] ascending in m}: a row count takes the first m at or above it
_TABLES: dict[tuple[int, int], dict[tuple[int, int, str], list[tuple[int, str]]]] = {}


def _torch(x, w, out, out_dtype):
    if out_dtype is torch.float32 and x.dtype != torch.float32:
        y = torch.mm(x, w.t(), out_dtype=torch.float32)
        return y if out is None else out.copy_(y)
    return F.linear(x, w) if out is None else torch.mm(x, w.t(), out=out)


def _fi(backend: str) -> Callable:
    def run(x, w, out, out_dtype):
        import flashinfer

        return flashinfer.mm_bf16(x, w.t(), out=out, out_dtype=out_dtype or x.dtype, backend=backend)
    return run


def _tinygemm(x, w, out, out_dtype):
    from flashinfer.gemm import tinygemm_bf16

    y = out if out is not None else x.new_empty(x.shape[0], w.shape[0])
    tinygemm_bf16(x, w, y)
    return y


def _router_n896(x, w, out, out_dtype):
    from flashinfer.gemm import mm_M1_16_K7168_N896

    y = out if out is not None else x.new_empty(x.shape[0], w.shape[0], dtype=torch.float32)
    mm_M1_16_K7168_N896(x, w.t(), y, launch_with_pdl=False)
    return y


# name -> (fn, the output dtypes it writes, the largest M it takes or None)
BACKENDS: dict[str, tuple[Callable, tuple[torch.dtype, ...], int | None]] = {
    "torch": (_torch, (torch.bfloat16, torch.float32), None),
    "fi_cublaslt": (_fi("cublaslt"), (torch.bfloat16, torch.float32), None),
    "fi_tinygemm": (_tinygemm, (torch.bfloat16,), None),
    "fi_tgv": (_fi("tgv"), (torch.bfloat16,), None),
    "fi_cute_dsl": (_fi("cute-dsl"), (torch.bfloat16,), 32),
    "fi_router_n896": (_router_n896, (torch.float32,), 16),
}


def _capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def load_table(path: str) -> None:
    """Add a table ``benchmark/kernels/k31_dense_gemm.py --table-out`` wrote: ``{"capability": [major, minor],
    "entries": [{"n", "k", "out", "m", "backend"}, ...]}``, replacing that capability's entries for the
    shapes it names."""
    with open(path) as f:
        table = json.load(f)
    cap = tuple(table["capability"])
    rows: dict[tuple[int, int, str], list[tuple[int, str]]] = {}
    for e in table["entries"]:
        if e["backend"] not in BACKENDS:
            raise ValueError(f"small_m_gemm table {path}: unknown backend {e['backend']!r}")
        rows.setdefault((int(e["n"]), int(e["k"]), e["out"]), []).append((int(e["m"]), e["backend"]))
    current = _TABLES.setdefault(cap, {})
    for key, entries in rows.items():
        current[key] = sorted(entries)


def backend_for(m: int, n: int, k: int, out_dtype: torch.dtype, device: torch.device) -> str:
    """The table's backend for an ``(m, k) x (n, k)`` product, or ``torch``."""
    if device.type != "cuda" or not _TABLES:
        return "torch"
    out = "fp32" if out_dtype is torch.float32 else "bf16"
    for m_max, name in _TABLES.get(_capability(device), {}).get((n, k, out), ()):
        if m <= m_max:
            _, dtypes, limit = BACKENDS[name]
            return name if out_dtype in dtypes and (limit is None or m <= limit) else "torch"
    return "torch"


def _run(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None, out_dtype: torch.dtype | None):
    out_dtype = out_dtype or (out.dtype if out is not None else x.dtype)
    name = backend_for(x.shape[0], w.shape[0], x.shape[1], out_dtype, x.device)
    return BACKENDS[name][0](x, w, out, None if name == "torch" and out_dtype == x.dtype else out_dtype)


@torch.library.custom_op("mstar::small_m_mm", mutates_args=())
def _small_m_mm(x: torch.Tensor, w: torch.Tensor, fp32_out: bool) -> torch.Tensor:
    return _run(x, w, None, torch.float32 if fp32_out else None)


@_small_m_mm.register_fake
def _(x, w, fp32_out):
    return x.new_empty(x.shape[0], w.shape[0], dtype=torch.float32 if fp32_out else x.dtype)


@torch.library.custom_op("mstar::small_m_mm_out", mutates_args=("out",))
def _small_m_mm_out(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    _run(x, w, out, out.dtype)


def linear(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype | None = None,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ w.T`` for ``x (..., K)`` and ``w (N, K)``: ``F.linear`` without a table entry (any device, any
    dtype), the table's backend with one. ``out_dtype=torch.float32`` returns fp32 products of bf16 inputs;
    ``out`` (2-D, contiguous) takes the result in place."""
    if not _TABLES or not x.is_cuda:
        if out is not None:
            return torch.mm(x, w.t(), out=out)
        if out_dtype is torch.float32 and x.dtype != torch.float32:
            return torch.mm(x.reshape(-1, x.shape[-1]), w.t(), out_dtype=torch.float32).view(*x.shape[:-1], -1)
        return F.linear(x, w)
    rows = x.reshape(-1, x.shape[-1])
    if not rows.is_contiguous():
        rows = rows.contiguous()
    if out is not None:
        _small_m_mm_out(rows, w, out)
        return out
    y = _small_m_mm(rows, w, out_dtype is torch.float32 and x.dtype != torch.float32)
    return y.view(*x.shape[:-1], w.shape[0])
