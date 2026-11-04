"""A CUDA device emulated on the CPU, for GPU-time simulation.

Nothing is computed. A cuda tensor keeps real values only when it is a small
integer tensor (positions, page tables, lengths: what mstar plans with); every
other one is a ``meta`` tensor, shape and dtype only. Each kernel is still
launched on the simulated stream and charged its predicted time, so the
scheduler, worker and engine run against a GPU's timing without its math. A
host read of a value that was never computed (a sampled token, say) reads
zeros.

``SimTensor`` is a wrapper tensor that reports ``device='cuda:N'`` and keeps
its payload (a meta tensor, or a CPU one) in ``_data``. Torch functions called on one are
intercepted in Python (``__torch_function__``), before any C++ code could ask
for a CUDA device guard this build doesn't have: the arguments are unwrapped,
the function runs on the payloads, and its tensor results are wrapped again.
Under that call an observing dispatch mode sees each aten op: it charges the
op's simulated GPU time to the current stream (:mod:`mstar_sim.clock`,
:mod:`mstar_sim.cost`) or, while a graph is being captured, records it on the
graph's tape. ``SimMode`` (a torch-function mode on every thread) turns
``device='cuda'`` arguments, ``.to('cuda')``/``.cuda()`` and pinned-memory
requests into CPU ones and wraps their results, so CUDA code runs unchanged.

Behaviour a real GPU has, kept on purpose:

- reading a value on the host (``.item()``, ``.tolist()``, ``bool(t)``,
  ``.cpu()``, a blocking D2H copy) waits until the device would have finished
  the current stream;
- doing so while a graph is being captured raises, as CUDA does;
- an op mixing a cuda tensor with a non-scalar CPU tensor raises, as CUDA does
  (except copies and indexing, which CUDA allows).
"""

from __future__ import annotations

import contextlib
import threading
import weakref

import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

from mstar_sim import clock


def tree_flatten(obj):
    """The leaves of an op's arguments; plain tuples, lists and dicts walked
    directly (torch's pytree costs ~40 us an op), anything else through pytree."""
    out: list = []
    stack = [obj]
    while stack:
        x = stack.pop()
        t = type(x)
        if t is tuple or t is list:
            stack.extend(reversed(x))
        elif t is dict:
            stack.extend(reversed(list(x.values())))
        elif t in _LEAVES or isinstance(x, torch.Tensor):
            out.append(x)
        else:
            out.extend(_pytree.tree_flatten(x)[0])
    return out, None


def tree_map(fn, obj):
    t = type(obj)
    if t is tuple:
        return tuple(tree_map(fn, x) for x in obj)
    if t is list:
        return [tree_map(fn, x) for x in obj]
    if t is dict:
        return {k: tree_map(fn, v) for k, v in obj.items()}
    if t in _LEAVES or isinstance(obj, torch.Tensor):
        return fn(obj)
    return _pytree.tree_map(fn, obj)


_LEAVES = {int, float, bool, str, type(None), torch.dtype, torch.device, torch.memory_format,
           torch.layout, slice, type(Ellipsis)}

CPU = torch.device("cpu")
T = torch.Tensor

META = torch.device("meta")
# integer tensors up to this size keep their values
SMALL_INT = 1 << 20
# set in a worker process: there, large float CPU tensors (a model built on
# the CPU before it moves to cuda) are meta too, so no weight is materialized
WORKER = False
CPU_META_FLOATS = 1 << 16


def keeps_values(dtype: torch.dtype, numel: int) -> bool:
    """Whether a cuda tensor of this kind holds real values."""
    return not (dtype.is_floating_point or dtype.is_complex) and numel <= SMALL_INT


def _is_meta(x) -> bool:
    return isinstance(x, torch.Tensor) and x.device.type == "meta"


def host_value(t: torch.Tensor) -> torch.Tensor:
    """``t``'s values for a host read: zeros for one that was never computed."""
    return torch.zeros(t.shape, dtype=t.dtype) if t.device.type == "meta" else t


def to_device_storage(x: torch.Tensor) -> torch.Tensor:
    """What a cuda copy of CPU (or meta) tensor ``x`` stores."""
    if x.device.type == "meta" or not keeps_values(x.dtype, x.numel()):
        return torch.empty_like(x, device=META)
    return x.clone()


def is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.split(":")[0] == "cuda"
    return False


def _index(device) -> int:
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and device.index is not None:
        return device.index
    if isinstance(device, int):
        return device
    from mstar_sim import cuda

    return cuda.current_device()


class SimTensor(torch.Tensor):
    """A tensor on the emulated GPU; its payload is ``self._data``."""

    _data: torch.Tensor

    @staticmethod
    def __new__(cls, data: torch.Tensor, index: int = 0):
        out = torch.Tensor._make_wrapper_subclass(
            cls, data.size(), strides=data.stride(), storage_offset=data.storage_offset(),
            dtype=data.dtype, layout=data.layout, device=torch.device("cuda", index),
            requires_grad=data.requires_grad,
        )
        out._data = data
        _track(out, data)
        return out

    def __repr__(self, *, tensor_contents=None):
        return f"SimTensor({self._data!r}, device='{self.device}')"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in _METADATA:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))
        return _function(func, args, kwargs or {})

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Reached only by C++ code that bypassed __torch_function__.
        index = _check_devices(func, args, kwargs or {})
        return _run(func, args, kwargs or {}, index)


# simulated device memory: bytes of the distinct storages SimTensors hold
_STORAGES: dict[int, list[int]] = {}
ALLOCATED = [0, 0]  # current, peak
# reentrant: a finalizer can run (garbage collection) while this thread holds it
_track_lock = threading.RLock()


def _release(key: int) -> None:
    with _track_lock:
        entry = _STORAGES.get(key)
        if entry is not None:
            entry[1] -= 1
            if entry[1] == 0:
                del _STORAGES[key]
                ALLOCATED[0] -= entry[0]


def _track(wrapper, data: torch.Tensor) -> None:
    """Count ``data``'s storage while it lives. The finalizer hangs on the payload,
    not the wrapper: module moves swap wrappers, and a swap refuses weakrefs."""
    if data._base is not None or getattr(data, "_sim_tracked", False):
        return  # a view: its base holds (and counts) the storage
    data._sim_tracked = True
    storage = data.untyped_storage()
    key = storage._cdata
    with _track_lock:
        entry = _STORAGES.get(key)
        if entry is None:
            entry = _STORAGES[key] = [storage.nbytes(), 0, _fake_range(storage.nbytes())]
            ALLOCATED[0] += entry[0]
            ALLOCATED[1] = max(ALLOCATED[1], ALLOCATED[0])
        entry[1] += 1
    weakref.finalize(data, _release, key)


# Value-less storages get addresses in [2^60, 2^62): unique per storage, never
# reused, and non-canonical on x86-64 and arm64, so code that compares or offsets
# pointers sees distinct storages, and code that reads through one faults loudly
# (``faulthandler`` prints the Python stack) instead of reading host memory.
_FAKE_NEXT = [1 << 60]
_FAKE_UNTRACKED: dict[int, int] = {}


def _fake_range(nbytes: int) -> int:
    with _track_lock:
        base = _FAKE_NEXT[0]
        _FAKE_NEXT[0] += -(-max(nbytes, 1) // 512) * 512 + 512
    return base


def fake_base(storage) -> int:
    """The fake address of a value-less storage."""
    entry = _STORAGES.get(storage._cdata)
    if entry is not None:
        return entry[2]
    base = _FAKE_UNTRACKED.get(storage._cdata)
    if base is None:
        base = _FAKE_UNTRACKED[storage._cdata] = _fake_range(storage.nbytes())
    return base


class _FakeStorage:
    """A value-less tensor's ``untyped_storage()``: the meta storage with its fake
    address, so callers that key storages by ``data_ptr()`` see them as distinct.
    Handing it to C++ (``set_``) raises."""

    def __init__(self, storage):
        self._storage = storage

    def data_ptr(self) -> int:
        return fake_base(self._storage)

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def __len__(self):
        return len(self._storage)


def _unwrap(x):
    if isinstance(x, SimTensor):
        return x._data
    if isinstance(x, torch.device) and x.type == "cuda":
        return CPU
    if isinstance(x, str) and is_cuda_device(x):
        return "cpu"
    return x


def _shallow(args, kwargs):
    """The tensors among an op's arguments, one list deep (``cat``'s list): cheap."""
    for a in args:
        if isinstance(a, torch.Tensor):
            yield a
        elif type(a) is list or type(a) is tuple:
            for b in a:
                if isinstance(b, torch.Tensor):
                    yield b
    for a in kwargs.values():
        if isinstance(a, torch.Tensor):
            yield a


def _rewrap(raw, inputs, index):
    """Wrap the call's new CPU tensors; give back inputs it returned as they were.
    ``inputs`` are the call's tensor arguments (cuda ones and plain)."""
    by_id = {}
    for x in inputs:
        by_id[id(x._data) if isinstance(x, SimTensor) else id(x)] = x

    def one(x):
        if not isinstance(x, torch.Tensor) or isinstance(x, SimTensor):
            return x
        seen = by_id.get(id(x))
        if seen is not None:
            return seen
        return SimTensor(x, index)

    return tree_map(one, raw)


def host_wait() -> None:
    """A host read of device data: wait for the current stream, refuse under capture."""
    if clock.capturing() is not None:
        raise RuntimeError(
            "CUDA error: operation not permitted when stream is capturing "
            "(mstar_sim: a host read of a cuda tensor during graph capture)"
        )
    clock.wait_until(clock.current_timeline().free_at)


def _check_devices(func, args, kwargs, flat=None) -> int:
    """The cuda index of the call; raise for a cuda/cpu mix CUDA would refuse."""
    index, cpu_tensor = None, False
    for x in (flat if flat is not None else tree_flatten((args, kwargs))[0]):
        if isinstance(x, SimTensor):
            if index is None:
                index = x.device.index
        elif isinstance(x, torch.Tensor) and x.dim() > 0 and x.device.type == "cpu":
            cpu_tensor = True
    if cpu_tensor and index is not None and func is not None and func not in _MIXED_OK:
        where = [
            f"{'cuda' if isinstance(x, SimTensor) else x.device.type}{tuple(x.shape)}"
            for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)
        ]
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two "
            f"devices, cuda:{index} and cpu! (mstar_sim: {getattr(func, '__name__', func)} "
            f"over {', '.join(where)})"
        )
    return index or 0


# ---------------------------------------------------------------- the op loop


class _Observe(TorchDispatchMode):
    """Runs each aten op of a call on shapes (or, over small ints, values) and
    charges it, or records its cost while a graph is captured. ``autocast``:
    the CUDA autocast dtype in force at the call (inside the handler autocast
    reads as off)."""

    def __init__(self, autocast: torch.dtype | None = None):
        super().__init__()
        self.autocast = autocast

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        launched = clock.vnow()  # the kernel is launched now; the emulator's time is not device time
        with contextlib.nullcontext() if clock.FAST else clock.computing():
            if self.autocast is not None:
                args, kwargs = _autocast(func, args, kwargs, self.autocast)
            if any(t.device.type == "meta" for t in _shallow(args, kwargs)):
                # a value-less operand: the op runs on shapes only
                args, kwargs = tree_map(_as_meta, (args, kwargs))
                out = meta_call(func, args, kwargs, list(_shallow(args, kwargs)))
            else:
                out = func(*args, **kwargs)
        if clock.FAST or func.is_view or func in _FREE or clock.in_kernel():
            return out  # unpriced, or inside a stand-in (its own cost is charged)
        from mstar_sim import cost

        seconds = cost.op_seconds(func, args, kwargs, out)
        tape = clock.capturing()
        if tape is not None:
            tape.entries.append(seconds)
        elif seconds:
            clock.charge(seconds, launched)
        return out


# CUDA autocast's cast policy (aten/src/ATen/autocast_mode.cpp), applied by hand:
# the payloads are meta or CPU tensors, which CUDA autocast never sees. Matched
# at the aten level the dispatch mode sees (``linear`` arrives as ``mm``).
_AUTOCAST_LOW = {
    "mm", "addmm", "bmm", "baddbmm", "addbmm", "mv", "addmv", "addr", "matmul", "linear",
    "convolution", "_convolution", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
    "conv_transpose2d", "conv_transpose3d", "prelu", "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention_for_cpu",
}
_AUTOCAST_FP32 = {
    "acos", "asin", "cosh", "erfinv", "exp", "expm1", "log", "log10", "log2", "log1p",
    "reciprocal", "rsqrt", "sinh", "tan", "pow", "softplus", "native_layer_norm",
    "native_group_norm", "logsumexp", "cdist", "renorm", "binary_cross_entropy_with_logits",
}
# fp32 unless the call names an output dtype
_AUTOCAST_FP32_UNLESS_DTYPE = {"_softmax", "_log_softmax", "sum", "prod", "cumsum", "cumprod",
                               "linalg_vector_norm"}
_AUTOCAST_TARGET: dict = {}


def _autocast(func, args, kwargs, low: torch.dtype):
    rule = _AUTOCAST_TARGET.get(func)
    if rule is None:
        name = func.overloadpacket.__name__
        rule = _AUTOCAST_TARGET[func] = (
            "low" if name in _AUTOCAST_LOW else "fp32" if name in _AUTOCAST_FP32
            else "fp32_opt" if name in _AUTOCAST_FP32_UNLESS_DTYPE else "")
    if not rule or (rule == "fp32_opt" and kwargs.get("dtype") is not None):
        return args, kwargs
    if rule == "low":
        to = low
        eligible = (torch.float32,)
    else:
        to, eligible = torch.float32, (torch.float16, torch.bfloat16)

    def cast(x):
        return x.to(to) if isinstance(x, torch.Tensor) and x.dtype in eligible else x

    return tree_map(cast, args), tree_map(cast, kwargs)


def _as_meta(x):
    if isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dim() > 0:
        return x.to(META)
    return x


_FREE = {
    torch.ops.aten.empty.memory_format, torch.ops.aten.empty_strided.default,
    torch.ops.aten.empty_like.default, torch.ops.aten.lift_fresh.default,
    torch.ops.aten._local_scalar_dense.default, torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride.int, torch.ops.aten.sym_numel.default,
    torch.ops.aten.sym_storage_offset.default, torch.ops.aten.is_same_size.default,
    torch.ops.aten.set_.source_Tensor,
}


# ---------------------------------------------------------------- shape cache
# A value-less op still costs torch's meta kernel, which for elementwise and
# indexing ops is Python (~150-250 us). Its result depends only on the input
# shapes, strides, dtypes and the non-tensor arguments, so each signature is
# run once and later calls build the outputs (~2 us each).
_SHAPES: dict = {}
_UNCACHEABLE = object()


def _signature(func, uargs, ukwargs, tensors):
    leaves = []
    for x in tree_flatten((uargs, ukwargs))[0]:
        if isinstance(x, torch.Tensor):
            continue
        if type(x) is slice:
            x = ("slice", x.start, x.stop, x.step)
        leaves.append(x)
    try:
        key = (func, tuple((t.shape, t.stride(), t.dtype, t.device.type) for t in tensors), tuple(leaves))
        hash(key)
        return key
    except TypeError:
        return None


def _describe(raw, tensors):
    if isinstance(raw, torch.Tensor):
        for i, t in enumerate(tensors):
            if raw is t:
                return ("arg", i)
        if raw.device.type != "meta":
            return _UNCACHEABLE
        return ("t", tuple(raw.shape), raw.stride(), raw.dtype)
    if type(raw) in (tuple, list):
        parts = [_describe(r, tensors) for r in raw]
        return _UNCACHEABLE if any(p is _UNCACHEABLE for p in parts) else (type(raw), parts)
    if raw is None or type(raw) in (int, float, bool):
        return ("v", raw)
    return _UNCACHEABLE


def _build(desc, tensors):
    kind = desc[0]
    if kind == "t":
        return torch.empty_strided(desc[1], desc[2], dtype=desc[3], device=META)
    if kind == "arg":
        return tensors[desc[1]]
    if kind == "v":
        return desc[1]
    return desc[0](_build(d, tensors) for d in desc[1])


def meta_call(func, uargs, ukwargs, tensors):
    """``func`` over value-less operands, its output shapes cached per signature.

    Ops whose output aliases an input are not cached: a view rebuilt from the
    cache would be a new storage (counted again, a different address), and an
    in-place view op (``transpose_``, inside ``stft``) changes its argument's
    own shape, which a cache hit would skip. On meta tensors they are cheap."""
    aliases = (getattr(func, "is_view", False) or func in _VIEWS
               or torch.Tag.inplace_view in getattr(func, "tags", ()))
    key = None if aliases else _signature(func, uargs, ukwargs, tensors)
    if key is not None:
        desc = _SHAPES.get(key)
        if desc is not None:
            return _build(desc, tensors)
    raw = func(*uargs, **ukwargs)
    if key is not None:
        desc = _describe(raw, tensors)
        if desc is not _UNCACHEABLE:
            _SHAPES[key] = desc
    return raw

# Fast path: torch functions that are one kernel (or a view) are
# priced at the function level, without the per-aten-op dispatch mode, which
# costs more host time than the op. Heavier functions (matmul, attention,
# norms) still go through ``_Observe`` so each of their aten ops is priced.
_VIEWS = {
    T.view, T.reshape, T.narrow, T.permute, T.transpose, T.unsqueeze, T.squeeze, T.expand,
    T.flatten, T.unflatten, T.view_as, T.expand_as, T.t, T.split, T.chunk, T.unbind,
    T.select, T.diagonal, T.detach, T.as_strided, T.__getitem__,
    torch.narrow, torch.unsqueeze, torch.squeeze, torch.reshape, torch.transpose,
    torch.permute, torch.split, torch.chunk, torch.flatten, torch.unbind,
}
_ONE_KERNEL = {
    T.index_select, torch.index_select, T.gather, torch.gather, T.fill_, T.zero_, T.clone,
    torch.clone, T.contiguous, T.add, T.sub, T.mul, T.div, T.__add__, T.__sub__, T.__mul__,
    T.__truediv__, T.__radd__, T.__rsub__, T.__rmul__, T.__floordiv__, T.__mod__, T.add_,
    T.sub_, T.mul_, T.div_, T.masked_fill, T.masked_fill_, torch.where, torch.cat, torch.stack,
    T.cumsum, torch.cumsum, T.clamp, T.clamp_, T.clamp_min, T.clamp_max, T.long, T.int,
    T.float, T.bfloat16, T.half, T.bool, T.index_copy_, T.index_put_, T.scatter_, T.scatter,
    T.eq, T.ne, T.lt, T.le, T.gt, T.ge, T.__eq__, T.__ne__, T.__lt__, T.__gt__, T.__le__,
    T.__ge__, T.__and__, T.__or__, T.__xor__, T.__invert__, T.__neg__, T.sum, torch.sum,
    T.argmax, torch.argmax, T.any, T.all, torch.zeros_like, torch.empty_like, torch.ones_like,
    torch.full_like, T.repeat_interleave, T.new_zeros, T.new_empty, T.new_full, T.new_ones,
    T.__setitem__, T.copy_, T.masked_scatter_, T.index_fill_, torch.maximum, torch.minimum,
    T.abs, T.neg, T.sigmoid, T.exp, T.log, T.sqrt, T.rsqrt, T.pow, T.__pow__, T.sign,
}
# ones whose result dtype CUDA autocast changes: under it, priced per aten op
_AUTOCAST_FN = {T.exp, T.log, T.rsqrt, T.pow, T.__pow__, T.sum, torch.sum, T.cumsum, torch.cumsum}
_FAST_COST: dict = {}


def _is_view_index(index) -> bool:
    items = index if type(index) is tuple else (index,)
    return not any(isinstance(i, torch.Tensor) or type(i) in (list, bool) for i in items)


def _fast(func, args, kwargs, flat, index):
    """``func`` as one kernel (or none): run on shapes, price by its traffic."""
    uargs, ukwargs = tree_map(_unwrap, args), tree_map(_unwrap, kwargs)
    tensors = [t for t in _shallow(uargs, ukwargs)]
    if any(t.device.type == "meta" for t in tensors):
        if any(t.device.type == "cpu" for t in tensors):
            uargs, ukwargs = tree_map(_as_meta, (uargs, ukwargs))
            tensors = [t for t in _shallow(uargs, ukwargs)]
        raw = meta_call(func, uargs, ukwargs, tensors)
    else:
        raw = func(*uargs, **ukwargs)  # small ints: real, and cheap
        if func in _UNINITIALIZED:
            raw.zero_()
    view = func in _VIEWS and (func is not T.__getitem__ or _is_view_index(args[1]))
    if not view and not clock.FAST:
        outs = [o for o in (raw if type(raw) in (tuple, list) else (raw,)) if isinstance(o, torch.Tensor)]
        key = (func, tuple((t.shape, t.dtype) for t in tensors), tuple((o.shape, o.dtype) for o in outs))
        seconds = _FAST_COST.get(key)
        if seconds is None:
            from mstar_sim import cost

            nbytes = sum(t.numel() * t.element_size() for t in (*tensors, *outs))
            seconds = _FAST_COST[key] = cost.kernel_seconds(
                f"torch.{getattr(func, '__name__', 'op')}", flops=0.0, bytes=float(nbytes))
        clock.charge(seconds)
    return _rewrap(raw, [x for x in flat if isinstance(x, torch.Tensor)], index)


def _run(func, args, kwargs, index, flat=None):
    if flat is None:
        flat = tree_flatten((args, kwargs))[0]
    if ((func in _ONE_KERNEL or func in _VIEWS) and not clock.in_kernel()
            and not (func in _AUTOCAST_FN and torch.is_autocast_enabled("cuda"))):
        return _fast(func, args, kwargs, flat, index)
    uargs, ukwargs = tree_map(_unwrap, args), tree_map(_unwrap, kwargs)
    autocast = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
    with _Observe(autocast):
        raw = func(*uargs, **ukwargs)
    return _rewrap(raw, [x for x in flat if isinstance(x, torch.Tensor)], index)


def _is_custom_op(func) -> bool:
    ns = getattr(func, "namespace", None)
    return ns is not None and ns not in ("aten", "prim", "prims")


def _custom_impl(func):
    """The Python body a custom op runs on CUDA (its cuda kernel, else its default)."""
    from torch._library import custom_ops

    name = getattr(func, "_qualified_op_name", None) or func._schema.name
    opdef = custom_ops.OPDEFS.get(name.replace(".", "::", 1) if "::" not in name else name)
    if opdef is None:
        return None
    return opdef._backend_fns.get("cuda") or opdef._init_fn


def _function(func, args, kwargs):
    if clock.FAST:
        return _function_body(func, args, kwargs)
    # the interposer's own time is not the host's: the device stamps exclude it
    token = clock.compute_begin()
    try:
        return _function_body(func, args, kwargs)
    finally:
        clock.compute_end(token)


def _function_body(func, args, kwargs):
    handler = _METHODS.get(func)
    if handler is not None:
        return handler(*args, **kwargs)
    if getattr(func, "__name__", "") == "__get__":
        # attribute getters (device, shape, dtype, data, ...) read the wrapper itself
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
    if func in _HOST_READS:
        host_wait()
    device = kwargs.get("device")
    if device is not None and not is_cuda_device(device) and not isinstance(device, int):
        # a host allocation shaped like a cuda tensor (``zeros_like(x, device="cpu")``)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*tree_map(_unwrap, args), **kwargs)
    if _is_custom_op(func):
        impl = _custom_impl(func)
        if impl is not None:
            # the body runs inside SimMode's handler, where the mode is off: its factory calls
            # (``torch.empty(..., device=x.device)``) need the mode back
            with SimMode():
                return impl(*args, **kwargs)
    flat = tree_flatten((args, kwargs))[0]
    index = _check_devices(func, args, kwargs, flat)
    return _run(func, args, kwargs, index, flat)


# ---------------------------------------------------------------- graph replay


def replay(tape: clock.Tape) -> None:
    """Run a captured graph: one launch charging its kernels' time back to back."""
    from mstar_sim import cost

    clock.STATS["replays"] += 1
    if clock.FAST:
        return
    launched = clock.vnow()
    with clock.computing():
        total, nodes = 0.0, 0
        for entry in tape.entries:
            seconds = entry if type(entry) is float else clock.graph_node_seconds(entry)
            total += seconds
            nodes += seconds > 0 or type(entry) is not float
        clock.charge(total + nodes * cost.graph_node_seconds() + cost.graph_launch_seconds(), launched)


# ------------------------------------------------------------ special methods


def _copy_seconds(nbytes: int, kind: str) -> float:
    if clock.FAST:
        return 0.0
    from mstar_sim import cost

    return cost.copy_seconds(nbytes, kind)


def _to(self, *args, **kwargs):
    kwargs = dict(kwargs)
    copy = kwargs.pop("copy", False)
    if len(args) >= 4 and isinstance(args[3], bool):  # to(device, dtype, non_blocking, copy)
        args, copy = args[:3], args[3]
    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    opts = {"dtype": dtype, "non_blocking": non_blocking, "copy": copy}
    if memory_format is not None:
        opts["memory_format"] = memory_format
    if isinstance(self, SimTensor):
        if device is None or device.type == "cuda":
            raw = self._data.to(**opts)
            return self if raw is self._data else SimTensor(raw, self.device.index)
        if device.type == "cpu":
            if not non_blocking:
                host_wait()
            value = host_value(self._data)
            return value.to(copy=True, **{k: v for k, v in opts.items() if k != "copy"})
        return self._data.to(device, **opts)
    if is_cuda_device(device):
        if self.device.type not in ("cpu", "meta"):
            raise RuntimeError(f"mstar_sim: can't move a {self.device} tensor to cuda")
        with torch._C.DisableTorchFunctionSubclass():
            src = self.to(dtype=dtype) if dtype is not None and dtype != self.dtype else self
        data = to_device_storage(src)
        clock.charge(_copy_seconds(data.numel() * data.element_size(), "h2d"))
        return SimTensor(data, _index(device))
    with torch._C.DisableTorchFunctionSubclass():
        return T.to(self, *args, copy=copy, **kwargs)


def _cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
    target = torch.device("cuda", _index(device if device is not None else "cuda"))
    return _to(self, target, non_blocking=non_blocking)


def _cpu(self, memory_format=torch.preserve_format):
    host_wait()
    return host_value(self._data).clone(memory_format=memory_format)


def _host_read(name):
    def read(self, *args, **kwargs):
        host_wait()
        return getattr(host_value(self._data), name)(*args, **kwargs)

    return read


def _data_ptr(self) -> int:
    """A real address, or for a value-less tensor its storage's fake one."""
    data = self._data
    if data.device.type != "meta":
        return data.data_ptr()
    return fake_base(data.untyped_storage()) + data.storage_offset() * data.element_size()


def _copy_(self, src, non_blocking=False):
    if isinstance(self, SimTensor):
        if not isinstance(src, SimTensor):
            clock.charge(_copy_seconds(src.numel() * src.element_size(), "h2d"))
            if not _is_meta(self._data) and not _is_meta(src):
                self._data.copy_(src, non_blocking=non_blocking)
            return self
        return _run(T.copy_, (self, src), {"non_blocking": non_blocking}, self.device.index)
    # device-to-host into a CPU tensor
    if not non_blocking:
        host_wait()
    if not _is_meta(self):
        self.copy_(host_value(src._data))
    return self


_METHODS = {
    T.to: _to,
    T.cuda: _cuda,
    T.cpu: _cpu,
    T.copy_: _copy_,
    T.tolist: _host_read("tolist"),
    T.numpy: _host_read("numpy"),
    T.item: _host_read("item"),
    T.__bool__: _host_read("__bool__"),
    T.__int__: _host_read("__int__"),
    T.__float__: _host_read("__float__"),
    T.__index__: _host_read("__index__"),
    T.data_ptr: _data_ptr,
    T.untyped_storage: lambda self: (_FakeStorage(self._data.untyped_storage()) if _is_meta(self._data)
                                     else self._data.untyped_storage()),
    T.is_pinned: lambda self, *a, **k: False,
    T.pin_memory: lambda self, *a, **k: self,
    T.share_memory_: lambda self: self,
    T.get_device: lambda self: self.device.index,
    T.__reduce_ex__: lambda self, proto: (_rebuild, (self._data, self.device.index)),
    torch._has_compatible_shallow_copy_type: lambda a, b: type(a) is type(b),
}
_HOST_READS = {torch.equal}
# copies, indexing, and load_state_dict(assign=True) into a module built on the CPU
_MIXED_OK = {T.copy_, torch.Tensor.__getitem__, torch.Tensor.__setitem__, T.index_put_,
             torch.index_put, T.to, T.__getitem__, T.module_load}


def _rebuild(data, index):
    return SimTensor(data, index)


_PLAIN = {T.to, T.cuda, T.copy_, T.module_load, T.pin_memory, T.is_pinned}


def _plain(func, args, kwargs):
    """Calls on CPU tensors that name cuda or pinned memory."""
    if func is T.to:
        return _to(*args, **kwargs)
    if func is T.cuda:
        return _cuda(*args, **kwargs)
    if func is T.copy_ and isinstance(args[1], SimTensor) and not isinstance(args[0], SimTensor):
        return _copy_(*args, **kwargs)
    if func is T.copy_ and _is_meta(args[1]) and not _is_meta(args[0]):
        return args[0]  # a header-only weight into a host tensor: keeps its values
    if func is T.module_load and _is_meta(args[1]) and not _is_meta(args[0]) \
            and not (kwargs.get("assign") or (len(args) > 2 and args[2])):
        return args[0].detach()  # the same, through load_state_dict's parameter swap
    if func is T.pin_memory:
        return args[0]
    if func is T.is_pinned:
        return True
    return NotImplemented


def _emulated(fn, args, kwargs):
    """Emulator work: the host's own time stops at this call."""
    if clock.FAST:
        return fn(*args, **kwargs)
    depth = clock.compute_begin()
    try:
        return fn(*args, **kwargs)
    finally:
        clock.compute_end(depth)


def _create(func, args, kwargs, device):
    """A factory call on the cuda device: meta unless small ints."""
    index = _index(device)
    target = CPU
    shape = func(*args, **dict(kwargs, device=META))
    if isinstance(shape, T) and not keeps_values(shape.dtype, shape.numel()):
        target = META
    with _Observe():
        out = func(*args, **dict(kwargs, device=target))
    if target is CPU and func in _UNINITIALIZED:
        out.zero_()  # a never-computed value reads as zeros, not as the host heap's garbage
    if func in (torch.tensor, torch.as_tensor, torch.from_numpy) and isinstance(out, T):
        clock.charge(_copy_seconds(out.numel() * out.element_size(), "h2d"))
    return tree_map(
        lambda x: SimTensor(x, index) if isinstance(x, T) and not isinstance(x, SimTensor) else x, out)


class SimMode(TorchFunctionMode):
    """Routes ``device='cuda'`` and pinned memory to the CPU and wraps results."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in _METADATA:  # shapes and dtypes: the wrapper answers, nothing to route
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        if func is torch.device and len(args) == 1 and type(args[0]) is int and not kwargs:
            return torch.device("cuda", args[0])  # a bare index names the accelerator: the emulated GPU
        if kwargs.get("pin_memory"):  # no pinned allocator without a real accelerator
            kwargs = dict(kwargs, pin_memory=False)
        device = kwargs.get("device")
        if func in _PASSTHROUGH or any(type(a) is SimTensor for a in _shallow(args, kwargs)):
            # a call over cuda tensors: SimTensor handles it, devices included
            return _emulated(func, args, kwargs)
        if func in _PLAIN:
            out = _emulated(_plain, (func, args, kwargs), {})
            if out is not NotImplemented:
                return out
        if is_cuda_device(device) or (isinstance(device, int) and func in _FACTORIES):
            return _emulated(_create, (func, args, kwargs, device), {})
        if WORKER and func in _ALLOCATORS and device in (None, "cpu", CPU):
            shape = func(*args, **dict(kwargs, device=META))
            if isinstance(shape, T) and shape.is_floating_point() and shape.numel() >= CPU_META_FLOATS:
                return shape  # e.g. a weight of a model built on the CPU
        return func(*args, **kwargs)


# metadata the wrapper itself holds (no data, no device): answered directly
_METADATA = {
    T.numel, T.nelement, T.element_size, T.dim, T.ndimension, T.size, T.stride,
    T.storage_offset, T.is_contiguous, T.is_floating_point, T.is_complex, T.__len__,
    T.shape.__get__, T.dtype.__get__, T.device.__get__, T.ndim.__get__, T.is_cuda.__get__,
    T.layout.__get__, T.requires_grad.__get__, T.is_leaf.__get__, T.itemsize.__get__,
    T.nbytes.__get__, T.is_meta.__get__, T.is_sparse.__get__, T.is_quantized.__get__,
    T.names.__get__, T.grad_fn.__get__,
}

# parse a device argument rather than create on it
_PASSTHROUGH = {torch._C._nn._parse_to}
_FACTORIES = {torch.empty, torch.zeros, torch.ones, torch.full, torch.arange, torch.tensor,
              torch.randn, torch.rand, torch.randint, torch.empty_strided, torch.as_tensor}
# allocations whose values are undefined: on the emulated GPU they read as zeros
_UNINITIALIZED = {torch.empty, torch.empty_strided, torch.empty_like, T.new_empty, T.new_empty_strided}
# fill-only allocations (no host data in them), which a worker may leave value-less
_ALLOCATORS = {torch.empty, torch.zeros, torch.ones, torch.full, torch.randn, torch.rand,
               torch.empty_strided}


# --------------------------------------------------------------- installation

_installed = False


def install() -> None:
    """Enter ``SimMode`` on this thread and on every thread started after."""
    global _installed
    if _installed:
        return
    _installed = True
    import faulthandler

    faulthandler.enable()  # a read through a fake address prints the Python stack
    # Moving a module to cuda keeps each Parameter object, as on a GPU (where a
    # CPU and a CUDA tensor share an impl type and torch sets `.data`): tied
    # weights stay tied and references taken before the move stay live.
    torch.__future__.set_swap_module_params_on_conversion(True)
    SimMode().__enter__()
    bootstrap = threading.Thread._bootstrap_inner

    def bootstrap_inner(self):
        SimMode().__enter__()
        bootstrap(self)

    threading.Thread._bootstrap_inner = bootstrap_inner
