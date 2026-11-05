"""How long each kernel takes on the simulated GPU.

Every aten op the simulated device runs costs ``op_seconds``; a stand-in for a
fused kernel (FlashInfer, Triton) costs ``kernel_seconds``. Both price a kernel's body

    min_kernel_s + max(flops / (peak[mode] * eff_c) + ramp_s, bytes / (bw * eff_m))

where ``peak``, ``bw``, ``min_kernel_s`` and the launch gap come from the GPU's
spec (:mod:`mstar_sim.gpus`), ``eff_c``/``eff_m`` are the GPU's efficiency
factors times the kernel's (``KERNEL_EFFICIENCY``, matched by substring of the
op or kernel name), and ``ramp_s`` is the time a compute-bound kernel loses
filling and draining its tile pipeline. A price never holds a launch gap:
:func:`mstar_sim.clock.charge` adds the GPU's eager gap (``launch_seconds()``) per
kernel launched on a stream, and a graph replay ``graph_node_seconds()`` per node
and ``graph_launch_seconds()`` once, so a price means the same eagerly and
captured. This is the roofline of
``torch.utils._runtime_estimation`` (PyTorch's ``RuntimeEstimator``) with a
latency floor and a ramp added; unlike it, FLOPs are not halved (the
``flop_registry`` counts 2 per multiply-add, as datasheet peaks do) and bytes
come from ``numel * element_size`` rather than storage sizes, so a view of a
large cache costs only what it touches.

FLOPs come from ``torch.utils.flop_counter.flop_registry`` (mm, addmm, bmm,
baddbmm, convolution, the SDPA kernels), plus ``_int_mm``, ``_grouped_mm`` and
the CPU flash-attention op; a causal self-attention counts half. Every other op
is memory-bound. Bytes are the distinct input and output tensors, with three
corrections: an in-place op reads and writes its mutated input (a write-only op
such as ``copy_`` or ``fill_`` only writes it); a gather (``embedding``,
``index_select``, ``index``) reads only as much of its source as it outputs; an
in-place scatter (``index_put_``, ``index_copy_``, ``scatter_``) touches only the
rows it writes, not the whole destination (a KV-cache write costs its slot, not
the cache). Host<->device copies are charged by the emulator with
``copy_seconds``. Ops that launch no kernel on CUDA (views, aliases, ``empty``, metadata
queries, ``sym_*``) cost nothing.

With ``MSTAR_SIM_COST=tables`` (the default when NVIDIA AIConfigurator's data
directory exists, see :mod:`mstar_sim.tables`), a GEMM (``mm``, ``addmm``,
``bmm``, ``baddbmm``, ``_scaled_mm``) and a fused kernel that reports its
geometry (attention, GDN, causal conv) cost their time interpolated from
latencies measured on the real GPU, floored at the family's smallest measured
kernel; what the tables don't cover costs the roofline. ``MSTAR_SIM_COST=roofline``
keeps the roofline everywhere.

A profile DB (``configure(gpu, profile_db=path)``, written by
``python -m mstar_sim.profile``) holds measured kernel times on the real GPU.
An op whose key is in it costs the measured time; an op of a measured family
(the same op and shapes except the leading token dimension) costs its roofline
times the measured/roofline ratio interpolated over that dimension; anything
else costs the tables' time, else the roofline. ``start_trace(path)`` appends every distinct key the
simulator charges, with what is needed to replay it, to a JSONL file that the
profiler reads.

``op_seconds`` runs for every op the simulated device launches, so its result
is cached per (op, input and output shapes and dtypes, non-numeric arguments),
in a cache of at most ``CACHE_ENTRIES`` (least recently used out first); the
numeric value of a scalar argument does not change a kernel's cost.
"""

from __future__ import annotations

import bisect
import collections
import json
import os
import threading
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import flop_registry

from mstar_sim import clock, gpus, tables

FORMAT = "mstar_sim.profile/1"
aten = torch.ops.aten


@dataclass(frozen=True)
class Efficiency:
    """A kernel's efficiency relative to the GPU's large-GEMM and large-copy factors."""

    compute: float = 1.0
    memory: float = 1.0
    ramp_s: float = 3e-6  # cuBLAS bf16 1024^3 on H100 takes ~8 us, roofline + floor ~5 us


# First substring match of the op or kernel name wins. Edit before configure().
KERNEL_EFFICIENCY: dict[str, Efficiency] = {
    # FlashAttention-2/3 and FlashInfer reach ~50-75% of peak at long sequences and much less
    # at short ones (few CTAs per head): FA3 at 512 causal tokens ~200 TF on H100. Decode
    # attention streams the KV cache at ~75% of HBM bandwidth.
    "attention": Efficiency(compute=0.75, memory=0.9, ramp_s=7e-6),
    "attn": Efficiency(compute=0.75, memory=0.9, ramp_s=7e-6),
    # Radix sort and top-k make several passes over their input.
    "sort": Efficiency(memory=0.15),
    "topk": Efficiency(memory=0.3),
}
_DEFAULT_EFFICIENCY = Efficiency()

# aten ops (by name, any overload) that launch no kernel on CUDA, beyond func.is_view.
_FREE = frozenset({
    "_unsafe_view", "lift_fresh", "unsafe_split", "unsafe_split_with_sizes", "unsafe_chunk",
    "empty", "empty_like", "empty_strided", "empty_permuted", "new_empty", "new_empty_strided",
    "resize_", "resize_as_", "set_", "record_stream", "resolve_conj", "resolve_neg",
    "detach_", "squeeze_", "unsqueeze_", "t_", "transpose_", "swapdims_", "swapaxes_", "as_strided_",
    "size", "stride", "numel", "dim", "storage_offset", "is_contiguous", "is_strides_like_format",
    "is_non_overlapping_and_dense", "is_same_size", "is_pinned", "is_complex", "is_floating_point",
    "_nnz", "_assert_tensor_metadata", "_has_compatible_shallow_copy_type", "_local_scalar_dense",
    "sym_size", "sym_stride", "sym_numel", "sym_storage_offset",
})
_GATHER = frozenset({"embedding", "index_select", "index", "_unsafe_index", "gather", "take"})
_SCATTER = frozenset({
    "index_put_", "_index_put_impl_", "index_copy_", "index_add_", "index_fill_", "scatter_",
    "scatter_add_", "scatter_reduce_", "masked_scatter_", "put_",
})
_WRITE_ONLY = frozenset({
    "copy_", "fill_", "zero_", "normal_", "uniform_", "random_", "bernoulli_", "exponential_",
    "geometric_", "cauchy_", "log_normal_",
})


def _grouped_mm_flops(a, b, *args, out_val=None, **kwargs) -> int:
    (a, b) = (a.shape, b.shape)
    if len(a) == 3 and len(b) == 3:  # (G, M, K) x (G, K, N)
        return 2 * a[0] * a[1] * a[2] * b[2]
    return 2 * a[-2] * a[-1] * b[-1]  # (T, K) x (G, K, N) or (M, K) x (K, N): each row meets one group


_EXTRA_FLOPS = {
    "_int_mm": flop_registry[aten.mm],
    "_grouped_mm": _grouped_mm_flops,
    "_scaled_grouped_mm": _grouped_mm_flops,
    "_scaled_dot_product_flash_attention_for_cpu": flop_registry[aten._scaled_dot_product_flash_attention],
}

_FP8 = {getattr(torch, n) for n in dir(torch) if n.startswith("float8_")}
_DTYPE_NAMES = {
    "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "fp8": torch.float8_e4m3fn,
    "int8": torch.int8, "fp64": torch.float64,
}


@dataclass(frozen=True)
class _OpInfo:
    free: bool
    name: str = ""
    flops: object = None  # flop formula, or None for a memory-bound op
    causal: int | None = None  # positional index of an is_causal argument
    mutable: frozenset = frozenset()  # positional indices and kwarg names of mutated arguments
    write_only: bool = False
    gather: bool = False
    scatter: bool = False
    efficiency: Efficiency = _DEFAULT_EFFICIENCY


class _State:
    gpu: gpus.GPU = gpus.lookup("h100")
    db: dict[str, float] | None = None
    families: dict[str, tuple[list[int], list[float]]] = {}
    tables: tables.Tables | None = None


_S = _State()
_INFO: dict = {}
# a model whose shapes change step to step (a TTS, a prefill mix) keeps adding keys: bounded,
# so a long run's memory stays flat
CACHE_ENTRIES = 1 << 13


class Cache(collections.OrderedDict):
    """A memo of at most ``CACHE_ENTRIES``, the least recently used out first."""

    def get(self, key, default=None):
        try:
            self.move_to_end(key)
        except KeyError:
            return default
        return self[key]

    def put(self, key, value):
        self[key] = value
        if len(self) > CACHE_ENTRIES:
            self.popitem(last=False)
        return value


_CACHE = Cache()
_TRACE: _Trace | None = None


def configure(gpu: str, profile_db: str | None = None, model: str | None = None) -> None:
    """Simulate ``gpu`` (an alias such as "h100" or a torch device name), preferring the
    measured times in ``profile_db`` where it has them. ``model`` is "tables" or
    "roofline" (default: ``MSTAR_SIM_COST``, else tables when their data exists)."""
    spec = gpus.lookup(gpu)
    db, families = (None, {}) if profile_db is None else _load_db(profile_db, spec)
    root = tables.data_dir()
    model = model or os.environ.get("MSTAR_SIM_COST") or (
        "tables" if os.path.isdir(root) and not clock.FAST else "roofline")  # --fast prices nothing
    if model not in ("tables", "roofline"):
        raise ValueError(f"MSTAR_SIM_COST must be 'tables' or 'roofline', not {model!r}")
    if model == "tables" and not os.path.isdir(root):
        raise ValueError(f"MSTAR_SIM_COST=tables but AIConfigurator data is not at {root!r}: "
                         f"{tables.FETCH}, or set MSTAR_SIM_AIC_DATA")
    _S.gpu, _S.db, _S.families = spec, db, families
    _S.tables = tables.Tables(spec, root) if model == "tables" else None
    _INFO.clear()
    _CACHE.clear()


def kernel_seconds(name: str, *, flops: float = 0.0, bytes: float = 0.0, dtype=None,
                   features: dict | None = None) -> float:
    """Device time of one fused kernel ``name`` doing ``flops`` over ``bytes`` of DRAM traffic.

    ``features`` is the geometry a measured table needs, by kernel family:

    - attention (``flashinfer.batch_paged_attention``, ``.batch_prefill_ragged``,
      ``.trtllm_batch_decode``): ``rows`` [(queries, keys, visible pairs)] per
      request, ``heads``, ``kv_heads``, ``head_dim``, ``kv_dtype``;
    - GDN (``flashinfer.gdn_prefill``, ``.gdn_decode``): ``batch``, ``tokens``,
      ``k_heads``, ``v_heads``, ``head_k_dim``, ``head_v_dim``;
    - causal conv (``triton.causal_conv1d_fn``, ``_update``): ``batch``, ``tokens``, ``channels``.
    """
    gpu = _S.gpu
    dtype = _DTYPE_NAMES.get(dtype, dtype) if isinstance(dtype, str) else (dtype or torch.bfloat16)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    body = _roofline(gpu, _efficiency(name), flops, bytes, _mode(dtype, tensor_cores=False))
    measured = None
    if _S.tables is not None and features is not None and name in _TABLE_KERNELS:
        measured = _TABLE_KERNELS[name](_S.tables, features)
    if _S.db is not None or _TRACE is not None:
        spec = {"kernel": name, "flops": float(flops), "bytes": float(bytes), "dtype": str(dtype)[6:]}
        key = "kernel:" + json.dumps(spec, sort_keys=True, separators=(",", ":"))
        if _TRACE is not None:
            _TRACE.record(key, spec, body)
        if _S.db is not None and key in _S.db:
            measured = _S.db[key]
    return body if measured is None else measured


def op_seconds(func, args, kwargs, out) -> float:
    """Device time of the aten op ``func(*args, **kwargs) -> out`` on the simulated GPU.

    Tensors are read only through ``.shape``, ``.dtype``, ``.numel()`` and
    ``.element_size()``, so a wrapper subclass works."""
    info = _INFO.get(func)
    if info is None:
        info = _INFO[func] = _op_info(func)
    if info.free:
        return 0.0
    kwargs = kwargs or {}
    key = (
        func,
        tuple([(a.shape, a.dtype) if isinstance(a, torch.Tensor) else _sig(a) for a in args]),
        _sig(kwargs) if kwargs else None,
        (out.shape, out.dtype) if isinstance(out, torch.Tensor) else _sig(out),
    )
    body = _CACHE.get(key)
    if body is None:
        body = _CACHE.put(key, _measured_or_roofline(func, info, args, kwargs, out))
    return body


def launch_seconds() -> float:
    """The device gap before each kernel launched on a stream (eagerly)."""
    return _S.gpu.launch_s


def graph_node_seconds() -> float:
    """The device gap each kernel node adds to a CUDA-graph replay."""
    return _S.gpu.graph_node_s


def graph_launch_seconds() -> float:
    """The cost of one CUDA-graph replay launch, before its first node starts."""
    return _S.gpu.graph_launch_s


def copy_seconds(nbytes: int, kind: str) -> float:
    """Device time of a ``cudaMemcpyAsync`` of ``nbytes``; ``kind`` is "h2d", "d2h" or "d2d"."""
    gpu = _S.gpu
    if kind == "d2d":  # reads and writes HBM
        return gpu.min_kernel_s + 2 * nbytes / (gpu.hbm_bw * gpu.memory_eff)
    if kind in ("h2d", "d2h"):
        return gpu.copy_latency_s + nbytes / (gpu.host_link_bw * gpu.memory_eff)
    raise ValueError(f"copy kind must be h2d, d2h or d2d, not {kind!r}")


def start_trace(path: str) -> None:
    """Append each distinct op and kernel the simulator charges from now on to ``path`` (JSONL)."""
    global _TRACE
    stop_trace()
    _TRACE = _Trace(path)
    _CACHE.clear()  # so every distinct op misses the cache once and is recorded


def stop_trace() -> None:
    global _TRACE
    if _TRACE is not None:
        _TRACE.file.close()
        _TRACE = None


def roofline_seconds(func, args, kwargs, out) -> float:
    """The roofline part of ``op_seconds`` (no launch gap, no profile DB)."""
    return _roofline_op(_op_info(func), args, kwargs or {}, out)


def spec_key(spec: dict, lead: int | None = None) -> str:
    """The DB key of a traced op spec. Scalar float values don't change cost, so they are
    blanked; with ``lead``, tensor dims 0 equal to it are too (the op's family key)."""
    return json.dumps(_keyify(spec, lead), separators=(",", ":"))


def encode(x):
    """A JSON form of an aten argument that :mod:`mstar_sim.profile` can rebuild."""
    if isinstance(x, torch.Tensor):
        return {"tensor": list(x.shape), "dtype": str(x.dtype)[6:]}
    if isinstance(x, (list, tuple)):
        return [encode(i) for i in x]
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (torch.dtype, torch.memory_format, torch.layout)):
        return {"torch": str(x)[6:]}
    if isinstance(x, torch.device):
        return {"device": str(x)}
    if isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return encode(x.node.hint if hasattr(x.node, "hint") else int(x))
    return {"opaque": type(x).__name__}


# --- internals ---------------------------------------------------------------


def _op_info(func) -> _OpInfo:
    name = func._opname
    if func.is_view or name in _FREE or name.startswith("sym_") or func.namespace in ("prim", "profiler"):
        return _OpInfo(free=True)
    schema = func._schema.arguments
    mutable = frozenset(
        a.name if a.kwarg_only else i
        for i, a in enumerate(schema)
        if a.alias_info is not None and a.alias_info.is_write
    )
    causal = next((i for i, a in enumerate(schema) if a.name == "is_causal" and not a.kwarg_only), None)
    return _OpInfo(
        free=False,
        name=name,
        flops=flop_registry.get(func._overloadpacket) or _EXTRA_FLOPS.get(name),
        causal=causal,
        mutable=mutable,
        write_only=name in _WRITE_ONLY,
        gather=name in _GATHER,
        scatter=name in _SCATTER,
        efficiency=_efficiency(str(func)),
    )


def _efficiency(name: str) -> Efficiency:
    return next((e for pattern, e in KERNEL_EFFICIENCY.items() if pattern in name), _DEFAULT_EFFICIENCY)


def _roofline(gpu: gpus.GPU, eff: Efficiency, flops: float, nbytes: float, mode: str) -> float:
    t = nbytes / (gpu.hbm_bw * gpu.memory_eff * eff.memory)
    if flops > 0:
        t = max(t, flops / (gpu.peak(mode) * gpu.compute_eff * eff.compute) + eff.ramp_s)
    return gpu.min_kernel_s + t


def _mode(dtype, tensor_cores: bool, conv: bool = False) -> str:
    if dtype in (torch.bfloat16, torch.float16):
        return "bf16"
    if dtype == torch.float32:
        if conv:
            return "tf32" if torch.backends.cudnn.allow_tf32 else "fp32"
        return "tf32" if tensor_cores and torch.get_float32_matmul_precision() != "highest" else "fp32"
    if dtype in _FP8:
        return "fp8"
    if dtype in (torch.int8, torch.uint8):
        return "int8"
    if dtype == torch.float64:
        return "fp64"
    return "fp32"


def _sig(x):
    if isinstance(x, torch.Tensor):
        return (x.shape, x.dtype)
    if isinstance(x, (list, tuple)):
        return tuple(map(_sig, x))
    if type(x) is float or type(x) is int:
        return type(x)
    if isinstance(x, dict):
        return tuple((k, _sig(v)) for k, v in x.items())
    return x


def _tensors(x, found: list) -> list:
    if isinstance(x, torch.Tensor):
        found.append(x)
    elif isinstance(x, (list, tuple)):
        for i in x:
            _tensors(i, found)
    return found


def _nbytes(t) -> int:
    return t.numel() * t.element_size()


def _roofline_op(info: _OpInfo, args, kwargs, out) -> float:
    gpu = _S.gpu
    outs = _tensors(out, [])
    out_bytes = sum(map(_nbytes, outs))
    seen: set[int] = set()
    nbytes = 0
    dst = None
    widest = 0  # elements of the largest tensor other than a scatter's destination
    first = None
    items = [(a, i in info.mutable, i == 0) for i, a in enumerate(args)]
    items += [(a, k in info.mutable, False) for k, a in kwargs.items()]
    for value, mutated, is_self in items:
        write_only = info.write_only or not is_self and mutated  # out= arguments are only written
        for t in _tensors(value, []):
            if id(t) in seen:
                continue
            seen.add(id(t))
            first = t if first is None else first
            if info.scatter and is_self:
                dst = t
                continue
            widest = max(widest, t.numel())
            b = _nbytes(t)
            if info.gather and is_self:
                b = min(b, out_bytes)
            elif mutated and not write_only:
                b *= 2
            nbytes += b
    nbytes += sum(_nbytes(t) for t in outs if id(t) not in seen)
    if dst is not None:
        nbytes += 2 * widest * dst.element_size()
    flops = 0
    mode = "fp32"
    if info.flops is not None:
        flops = info.flops(*args, **kwargs, out_val=out)
        if info.causal is not None:
            causal = args[info.causal] if len(args) > info.causal else kwargs.get("is_causal", False)
            if causal and args[0].shape[-2] == args[1].shape[-2]:
                flops //= 2
        mode = _mode(first.dtype, tensor_cores=True, conv="conv" in info.name)
    return _roofline(gpu, info.efficiency, flops, nbytes, mode)


def _measured_or_roofline(func, info: _OpInfo, args, kwargs, out) -> float:
    body = _roofline_op(info, args, kwargs, out)
    table = _table_gemm(info.name, args) if _S.tables is not None else None
    if _S.db is None and _TRACE is None:
        return body if table is None else table
    spec = {"op": str(func), "args": encode(args), "kwargs": encode(kwargs)}
    key = spec_key(spec)
    if _TRACE is not None:
        _TRACE.record(key, spec, body)
    if _S.db is None:
        return body if table is None else table
    if key in _S.db:
        return _S.db[key]
    lead = _lead(spec)
    family = _S.families.get(spec_key(spec, lead)) if lead is not None else None
    if family is None:
        return body if table is None else table
    leads, ratios = family
    i = bisect.bisect_left(leads, lead)
    if i == 0 or i == len(leads):
        return body * ratios[min(i, len(leads) - 1)]
    w = (lead - leads[i - 1]) / (leads[i] - leads[i - 1])
    return body * (ratios[i - 1] + w * (ratios[i] - ratios[i - 1]))


# aten GEMMs the tables answer: positions of (a, b) in the arguments
_GEMMS = {"mm": (0, 1), "addmm": (1, 2), "bmm": (0, 1), "baddbmm": (1, 2), "_scaled_mm": (0, 1)}


def _table_gemm(name: str, args) -> float | None:
    where = _GEMMS.get(name)
    if where is None:
        return None
    a, b = args[where[0]], args[where[1]]
    if a.dtype in (torch.bfloat16, torch.float16) and b.dtype == a.dtype:
        dtype = "bf16"
    elif name == "_scaled_mm" and a.dtype in _FP8:
        dtype = "fp8"
    else:
        return None
    if a.dim() == 2 and b.dim() == 2:
        return _S.tables.gemm_seconds(a.shape[0], b.shape[1], a.shape[1], dtype)
    if a.dim() == 3 and b.dim() == 3:
        return _S.tables.gemm_seconds(a.shape[1], b.shape[2], a.shape[2], dtype, batch=a.shape[0])
    return None


def _table_attention(t: tables.Tables, f: dict) -> float | None:
    """A mixed batch costs its generation rows (one query each) plus its context rows."""
    kv_dtype = "bf16" if f["kv_dtype"] in (torch.bfloat16, torch.float16) else \
        "fp8" if f["kv_dtype"] in _FP8 else None
    if kv_dtype is None or f.get("head_dim") is None:
        return None
    geometry = (f["heads"], f["kv_heads"], f["head_dim"], kv_dtype)
    total = 0.0
    for phase, rows in (("generation", [r for r in f["rows"] if r[0] == 1]),
                        ("context", [r for r in f["rows"] if r[0] > 1])):
        if not rows:
            continue
        q, kv, pairs = (sum(col) for col in zip(*rows, strict=True))
        if phase == "generation":
            kv = pairs  # the keys a windowed row reads
        t_phase = t.attention_seconds(phase, len(rows), q, kv, pairs, *geometry)
        if t_phase is None:
            return None
        total += t_phase
    return total


def _table_gdn(kernel: str):
    return lambda t, f: t.gdn_seconds(kernel, **f)


_TABLE_KERNELS = {
    "flashinfer.batch_paged_attention": _table_attention,
    "flashinfer.batch_prefill_ragged": _table_attention,
    "flashinfer.trtllm_batch_decode": _table_attention,
    "flashinfer.gdn_prefill": _table_gdn("chunk_gated_delta_rule"),
    "flashinfer.gdn_decode": _table_gdn("fused_recurrent_gated_delta_rule_packed_decode"),
    "triton.causal_conv1d_fn": _table_gdn("causal_conv1d_fn"),
    "triton.causal_conv1d_update": _table_gdn("causal_conv1d_update"),
}


def collective_seconds(op: str, num_gpus: int, nbytes: int, graph: bool = True) -> float:
    """Device time of a collective ``op`` ("all_reduce", "all_gather", "reduce_scatter",
    "alltoall") of ``nbytes`` (16-bit) across ``num_gpus`` GPUs of one node: measured
    (NCCL, SGLang custom all-reduce) when the tables cover it, else a link-bandwidth bound."""
    t = _S.tables.collective_seconds(op, num_gpus, nbytes, graph) if _S.tables is not None else None
    if t is None:
        if num_gpus <= 1:
            return 0.0
        link = tables.LINK_BW.get(_S.gpu.name, _S.gpu.host_link_bw)
        share = 2.0 * (num_gpus - 1) / num_gpus if op == "all_reduce" else (num_gpus - 1) / num_gpus
        t = _S.gpu.min_kernel_s + share * nbytes / link
    return t


def _keyify(x, lead):
    if isinstance(x, dict):
        if isinstance(x.get("tensor"), list) and lead is not None and x["tensor"][:1] == [lead]:
            return {**x, "tensor": [-1, *x["tensor"][1:]]}
        return {k: _keyify(v, lead) for k, v in x.items()}
    if isinstance(x, list):
        return [_keyify(i, lead) for i in x]
    return "f" if isinstance(x, float) else x


def _lead(spec: dict) -> int | None:
    """The leading (token) dimension of an op spec: dim 0 of its first tensor argument."""
    stack = list(reversed(spec["args"]))
    while stack:
        x = stack.pop()
        if isinstance(x, dict) and isinstance(x.get("tensor"), list):
            return x["tensor"][0] if x["tensor"] else None
        if isinstance(x, list):
            stack.extend(reversed(x))
    return None


def _load_db(path: str, gpu: gpus.GPU) -> tuple[dict[str, float], dict]:
    with open(path) as f:
        data = json.load(f)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} profile DB")
    if data["gpu"] != gpu.name:
        raise ValueError(f"profile DB {path} was measured on {data['gpu']!r}, not {gpu.name!r}")
    db = {key: e["seconds"] for key, e in data["entries"].items()}
    points: dict[str, list[tuple[int, float]]] = {}
    for e in data["entries"].values():
        spec = e["spec"]
        lead = _lead(spec) if "op" in spec else None
        if lead is not None and e.get("roofline", 0) > 0:
            points.setdefault(spec_key(spec, lead), []).append((lead, e["seconds"] / e["roofline"]))
    families = {}
    for fam, pts in points.items():
        pts.sort()
        families[fam] = ([m for m, _ in pts], [r for _, r in pts])
    return db, families


class _Trace:
    def __init__(self, path: str) -> None:
        self.file = open(path, "a")  # noqa: SIM115 - closed by stop_trace
        self.seen: set[str] = set()
        self.lock = threading.Lock()

    def record(self, key: str, spec: dict, seconds: float) -> None:
        if key in self.seen:
            return
        with self.lock:
            if key in self.seen:
                return
            self.seen.add(key)
            self.file.write(json.dumps({"key": key, "spec": spec, "seconds": seconds}) + "\n")
            self.file.flush()
