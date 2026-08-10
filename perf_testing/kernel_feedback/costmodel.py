"""Analytic per-kernel work -> speed-of-light time against a GPU spec.

The speed-of-light (SOL) of a kernel is the classic roofline lower bound on the
work it *actually performed*:

    sol_ms = max(flops / peak_tflops, traffic_bytes / mem_bandwidth)

clamped to never exceed the observed time (a noisy estimate must not make the
"optimal" time slower than reality — same monotonicity rule VibeSim uses).
``observed - sol`` is then the *kernel-quality* headroom (a better kernel for
the same work), while for pure data-movement kernels the SOL itself is
*fusion/elimination* headroom: their necessary floor is ~0 because a fused
producer/consumer would never materialize the intermediate.
"""

from __future__ import annotations

from .hardware import GpuSpec
from .inductor_dump import ExternCall, TritonKernel

# No kernel completes meaningfully faster than this (launch/teardown floor).
# Below it a kernel is latency-bound: recoverable only by eliminating the
# launch (fusion/batching), not by a better implementation of the kernel.
LAUNCH_FLOOR_MS = 0.0015

# ATen ops that move/reshape/materialize data but do no arithmetic. A fused
# kernel whose op set is contained here has a ~0 necessary floor.
MOVEMENT_ATEN = {
    "_to_copy", "view", "clone", "copy", "copy_", "cat", "slice", "select",
    "squeeze", "unsqueeze", "permute", "expand", "transpose", "t", "reshape",
    "contiguous", "repeat", "stack", "split", "alias", "detach", "fill", "full",
    "zeros", "zeros_like", "ones", "empty", "arange", "constant_pad_nd",
    "slice_scatter", "select_scatter", "index", "index_select", "gather",
    "scatter", "lift_fresh", "clamp", "type_as", "to",
}

# Eager aten:: op names (from profiler attribution) that are pure movement.
MOVEMENT_EAGER_OPS = {
    "aten::copy_", "aten::to", "aten::_to_copy", "aten::contiguous",
    "aten::clone", "aten::fill_", "aten::zero_", "aten::zeros", "aten::cat",
    "aten::masked_fill_", "aten::index_put_", "aten::repeat", "aten::full",
}

_ARITH_OPS = {
    "mul", "add", "sub", "div", "pow", "rsqrt", "sqrt", "exp", "log", "sin",
    "cos", "tanh", "sigmoid", "silu", "gelu", "neg", "where", "maximum",
    "minimum", "erf", "reciprocal", "abs", "relu", "mean", "sum", "amax",
    "amin", "var", "scatter_add", "cumsum", "softmax", "_softmax", "addcmul",
    "floor", "ceil", "round", "fmod", "remainder", "floor_divide", "rem",
}


def is_movement_kernel(k: TritonKernel) -> bool:
    return bool(k.aten_ops) and not (k.aten_ops - MOVEMENT_ATEN)


def triton_flops_estimate(k: TritonKernel) -> int:
    """~1 flop per output element per arithmetic node. Reductions count their
    input elements. Coarse on purpose: fused pointwise/reduction chains sit far
    below any GPU's ridge point, so bandwidth decides their SOL regardless."""
    flops = 0
    max_in = max((t.numel for t in k.inputs.values()), default=0)
    for node, op in k.op_of_node.items():
        base = op.split(".")[0]
        if base not in _ARITH_OPS:
            continue
        numel = k.node_tensors[node].numel
        if base in ("mean", "sum", "amax", "amin", "var", "softmax", "_softmax", "cumsum"):
            numel = max(numel, max_in)
        flops += numel
    return flops


def sol_ms(flops: float | None, traffic_bytes: float | None, spec: GpuSpec | None,
           dtype: str = "bf16") -> float | None:
    """Roofline lower bound in ms for one kernel invocation; None if unmodeled."""
    if spec is None or (not flops and not traffic_bytes):
        return None
    compute_ms = 0.0
    if flops:
        peak = spec.peak_tflops(dtype)
        if peak:
            compute_ms = flops / (peak * 1e12) * 1e3
    memory_ms = 0.0
    if traffic_bytes:
        memory_ms = traffic_bytes / (spec.mem_bandwidth_gbps * 1e9) * 1e3
    return max(compute_ms, memory_ms)


def extern_sol_ms(call: ExternCall, spec: GpuSpec | None) -> float | None:
    if call.flops is None:
        return None
    return sol_ms(call.flops, call.traffic_bytes, spec, call.dtype)


def kernel_kind(key: str) -> str:
    k = key.lower()
    if key.startswith("triton_"):
        return "triton"
    if "conv" in k or "nchwtonhwc" in k or "implicit_gemm" in k:
        return "conv"
    if any(x in k for x in ("nvjet", "cublas", "cutlass", "splitkreduce", "gemm")):
        return "gemm"
    if any(x in k for x in ("flashinfer", "fmha", "flash", "scaled_dot")):
        return "attention"
    if any(x in k for x in ("memcpy", "memset", "graphlaunch")):
        return "memcpy"
    return "eager"
