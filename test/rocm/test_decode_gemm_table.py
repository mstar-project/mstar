"""Every entry of the device's ``decode_gemm`` table on an AMD GPU: the kernel that serves a shape, and each other
kernel the table keeps for it, meets the op's bar against an fp32 reference on the device (a vocab head's weight is
2 GB), and the served one replays from a captured graph to what it computes eagerly."""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

from mstar.kernels.numerics import Tolerance, compare
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"
# what serves a gate/up left unfused is its linear's own entry, checked there
RUNNABLE = {"aiter", "triton", "hipblaslt", "hipblaslt_tuned", "fused"}


def _entries():
    from mstar.platforms.rocm_gemm import DECODE_GEMM

    shapes = (DECODE_GEMM.table() or {}).get("shapes", {})
    return [(key, impl) for key, entry in sorted(shapes.items()) for impl in entry["candidates"] if impl in RUNNABLE]


def _shape(key):
    parts = {part.rstrip("0123456789"): int(part.lstrip("abcdefghijklmnopqrstuvwxyz")) for part in key.split("-")}
    return parts["rows"], parts["n"], parts["k"], "gated" in parts


@functools.lru_cache(maxsize=1)
def _weight(n, k):
    g = torch.Generator(device=DEV).manual_seed(n + k)
    return (torch.randn(n, k, device=DEV, dtype=torch.bfloat16, generator=g) * k ** -0.5).contiguous()


def _kernel(impl, params):
    """``impl`` at the table's ``params``, as ``fn(x, w)``."""
    from mstar.kernels.triton import gemv

    knobs = {**gemv.DEFAULT_KNOBS, **{k: v for k, v in params.items() if k in gemv.KNOBS}}
    if impl == "fused":
        return lambda x, w: gemv.decode_gemv_silu_mul(x, w, knobs)
    if impl == "triton":
        return lambda x, w: gemv.decode_gemv(x, w, None, knobs)
    if impl == "aiter":
        return lambda x, w: torch.ops.mstar.skinny_gemm(x, w, None)
    if impl == "hipblaslt_tuned":
        import aiter

        aiter.hipb_create_extension()
        return lambda x, w: aiter.hipb_mm(x, w.t(), int(params["solution"]))
    return F.linear


def _reference(x, w, gated):
    out = x.float() @ w.float().T
    if gated:
        gate, up = out.chunk(2, dim=-1)
        return F.silu(gate) * up
    return out


@pytest.mark.parametrize(("key", "impl"), _entries() if current_platform.is_rocm() else [])
def test_each_kernel_the_table_holds_meets_the_bar(key, impl):
    from mstar.platforms.rocm_gemm import DECODE_GEMM

    rows, n, k, gated = _shape(key)
    shape = {"rows": rows, "n": n, "k": k, **({"gated": 1} if gated else {})}
    if impl == "fused" and not gated:
        pytest.skip("a fused kernel only serves a gated entry")
    fn = _kernel(impl, DECODE_GEMM.params(shape, impl))
    g = torch.Generator(device=DEV).manual_seed(rows)
    x = torch.randn(rows, k, device=DEV, dtype=torch.bfloat16, generator=g)
    w = _weight(n, k)
    eager = fn(x, w)
    result = compare(eager, _reference(x, w, gated), Tolerance.accumulated())
    assert result.ok, result.summary()
    if DECODE_GEMM.choose(shape).impl != impl:
        return
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn(x, w)
    graph.replay()
    torch.cuda.synchronize()
    result = compare(out, _reference(x, w, gated), Tolerance.accumulated())
    assert result.ok, f"replay: {result.summary()}"


def test_a_shape_the_table_does_not_name_falls_to_its_class():
    from mstar.platforms.rocm_gemm import gemv_impl

    assert gemv_impl(1, 96, 64) == "aiter"
    assert gemv_impl(8, 96, 64) == "hipblaslt"
