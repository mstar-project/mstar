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
    """Whether a cuda tensor of this kind holds real values: small ints, and small
    tensors of any dtype (a frame rate, a scale staged from the host)."""
    return numel <= SMALL_ANY or (not (dtype.is_floating_point or dtype.is_complex) and numel <= SMALL_INT)


# tensors up to this many elements keep their values, any dtype
SMALL_ANY = 1 << 12


def _is_meta(x) -> bool:
    return isinstance(x, torch.Tensor) and x.device.type == "meta"


# never-computed values so far: host reads of value-less tensors (zeros were
# returned), and value-holding tensors that stand in for a kernel's output (a
# zero-filled empty, a stand-in's or a Triton launch's small ints). A check on
# values after one is a check the emulator can't answer (``mstar_sim.pytest_plugin``)
VALUELESS_READS = [0]


def uncomputed() -> None:
    VALUELESS_READS[0] += 1


# stand-in kernels run so far: a test that expected a kernel's own error check
# (pytest.raises around a launch) can't see it from a stand-in
KERNELS_RUN = [0]
# ops run on zeros because their output shape depends on values never computed:
# any failure after one is the emulator's, not mstar's
VALUE_SHAPES = [0]


def host_value(t: torch.Tensor) -> torch.Tensor:
    """``t``'s values for a host read: zeros for one that was never computed."""
    if t.device.type != "meta":
        return t
    VALUELESS_READS[0] += 1
    return torch.zeros(t.shape, dtype=t.dtype)


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


# Simulated device memory, as torch's caching allocator accounts it. Each device has a
# default pool and the private pools graphs capture into. A pool's ``live`` bytes are
# its tensors' storages; ``reserved`` is what it holds from the card: its high-water
# mark, until the cache is emptied (a freed block is reused within its pool, never
# by another). ``graphs`` counts the graphs using a private pool; one no graph uses
# gives its free bytes back on ``empty_cache``, and goes once nothing in it lives.
# ``recording``: the thread whose allocations on its device go to the pool (a capture's,
# or a ``_cuda_beginAllocateToPool`` caller's), or None.
DEFAULT_POOL = (0, 0)


class Pool:
    __slots__ = ("device", "handle", "live", "reserved", "graphs", "recording")

    def __init__(self, device: int, handle: tuple):
        self.device, self.handle = device, handle
        self.live = self.reserved = self.graphs = 0
        self.recording: int | None = None


POOLS: dict[tuple[int, tuple], Pool] = {}
_STORAGES: dict[int, list] = {}  # storage key: [bytes, references, fake address range, pool or None]
ALLOCATED: dict[int, list[int]] = {}  # device index: [current, peak] bytes in tensors
RESERVED_PEAK: dict[int, int] = {}
# reentrant: a finalizer can run (garbage collection) while this thread holds it
_track_lock = threading.RLock()


def pool(index: int, handle=DEFAULT_POOL) -> Pool:
    key = (index, tuple(handle))
    with _track_lock:
        found = POOLS.get(key)
        if found is None:
            found = POOLS[key] = Pool(*key)
        return found


def reserved(index: int) -> int:
    with _track_lock:
        return sum(p.reserved for p in POOLS.values() if p.device == index)


def release_cached(index: int) -> None:
    """``empty_cache``: the default pool, and private pools no graph uses, give back
    what their tensors don't hold."""
    with _track_lock:
        for key, p in list(POOLS.items()):
            if p.device == index and (p.handle == DEFAULT_POOL or p.graphs == 0):
                p.reserved = p.live
                if p.handle != DEFAULT_POOL and p.live == 0 and p.recording is None:
                    del POOLS[key]


def _release(key: int) -> None:
    with _track_lock:
        entry = _STORAGES.get(key)
        if entry is not None:
            entry[1] -= 1
            if entry[1] == 0:
                del _STORAGES[key]
                if entry[3] is not None:
                    entry[3].live -= entry[0]
                    ALLOCATED[entry[3].device][0] -= entry[0]


def _allocate(index: int, nbytes: int) -> Pool:
    """The pool a new storage of ``nbytes`` on device ``index`` comes from: the one this
    thread records into on that device, else the default. Growing a pool past the card
    (its memory fraction) raises out of memory, after emptying the cache once, as torch's
    allocator retries."""
    thread = threading.get_ident()
    into = next((p for p in POOLS.values() if p.recording == thread and p.device == index), None) or pool(index)
    if into.live + nbytes > into.reserved:
        from mstar_sim import cuda

        grow = _segment(nbytes)
        limit = cuda.memory_limit(index)
        if reserved(index) + grow > limit:
            release_cached(index)
            if into.live + nbytes <= into.reserved:
                return into
            if reserved(index) + grow > limit:
                held = ", ".join(f"{'default' if p.handle == DEFAULT_POOL else p.handle}: {p.reserved / 2**20:.0f}"
                                 for p in POOLS.values() if p.device == index and p.reserved)
                raise gpu_error(torch.cuda.OutOfMemoryError(
                    f"CUDA out of memory. Tried to allocate {nbytes / 2**20:.2f} MiB. GPU {index} has a total "
                    f"capacity of {limit / 2**30:.2f} GiB, of which {(limit - reserved(index)) / 2**20:.2f} MiB "
                    f"is free (mstar_sim: MiB reserved per pool: {held})"))
        into.reserved += grow
        RESERVED_PEAK[index] = max(RESERVED_PEAK.get(index, 0), reserved(index))
    return into


def _segment(nbytes: int) -> int:
    """The segment torch's caching allocator reserves from the card for a block it has no room
    for (``CUDACachingAllocator``'s ``get_allocation_size``): 2 MiB for a small block (up to
    1 MiB), 20 MiB up to 10 MiB, else the block rounded up to 2 MiB."""
    if nbytes <= 1 << 20:
        return 2 << 20
    if nbytes < 10 << 20:
        return 20 << 20
    return -(-nbytes // (2 << 20)) * (2 << 20)


def _track(wrapper, data: torch.Tensor) -> None:
    """Count ``data``'s storage while it lives. The finalizer hangs on the payload,
    not the wrapper: module moves swap wrappers, and a swap refuses weakrefs. A
    storage opened from another process (CUDA IPC) is that process's memory."""
    if data._base is not None or getattr(data, "_sim_tracked", False):
        return  # a view: its base holds (and counts) the storage
    storage = data.untyped_storage()
    key = storage._cdata
    with _track_lock:
        entry = _STORAGES.get(key)
        if entry is None:
            # a block is a multiple of 512 bytes, as the caching allocator rounds it
            nbytes = -(-storage.nbytes() // 512) * 512
            entry = _STORAGES[key] = [nbytes, 0, _fake_range(nbytes), None]
            if not getattr(storage, "_sim_imported", False) and not clock.in_kernel():
                _count(entry, wrapper.device.index)
        entry[1] += 1
    data._sim_tracked = True
    weakref.finalize(data, _release, key)


def _count(entry: list, index: int) -> None:
    into = entry[3] = _allocate(index, entry[0])
    into.live += entry[0]
    used = ALLOCATED.setdefault(into.device, [0, 0])
    used[0] += entry[0]
    used[1] = max(used[1], used[0])


def count_storages(out) -> None:
    """A stand-in kernel's outputs are its allocations; the tensors it made them with
    inside (``clock.in_kernel``) are not counted, as the real kernel has none."""
    for t in tree_flatten(out)[0]:
        if isinstance(t, SimTensor):
            storage = t._data.untyped_storage()
            with _track_lock:
                entry = _STORAGES.get(storage._cdata)
                if entry is not None and entry[3] is None and not getattr(storage, "_sim_imported", False):
                    _count(entry, t.device.index)


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


# torch.cuda.set_sync_debug_mode: 0 default, 1 warn, 2 error
SYNC_DEBUG = [0]
_SHARED = b"mstar_sim:"


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

    def _share_cuda_(self):
        """A CUDA IPC handle naming this storage; another emulated process opens it
        (``UntypedStorage._new_shared_cuda``) as a value-less storage of its size."""
        nbytes = self._storage.nbytes()
        handle = _SHARED + f"{self.data_ptr()}:{nbytes}".encode()
        return (0, handle, nbytes, 0, b"", 0, b"", False)


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


def gpu_error(error: BaseException) -> BaseException:
    """``error``, which the emulator raises where the GPU stack would (CUDA, the caching
    allocator, a kernel library's own check): tagged, so it reads as mstar's, not as a gap
    in the emulator (``mstar_sim.pytest_plugin``)."""
    error._mstar_sim_gpu = True
    return error


def refuse_while_capturing(what: str) -> None:
    """CUDA refuses a synchronizing call (``what``) on a capturing stream."""
    if clock.capturing() is not None:
        raise gpu_error(RuntimeError(
            f"CUDA error: operation not permitted when stream is capturing (mstar_sim: {what})"))


def host_wait() -> None:
    """A host read of device data: wait for the current stream, refuse under capture
    (and, under ``torch.cuda.set_sync_debug_mode``, warn or refuse as CUDA does)."""
    if SYNC_DEBUG[0]:
        message = "called a synchronizing CUDA operation (mstar_sim: a host read of a cuda tensor)"
        if SYNC_DEBUG[0] >= 2:
            raise gpu_error(RuntimeError(message))
        import warnings

        warnings.warn(message, stacklevel=3)
    refuse_while_capturing("a host read of a cuda tensor")
    clock.wait_until(clock.current_timeline().free_at)
    check_fault()


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
        raise gpu_error(RuntimeError(
            "Expected all tensors to be on the same device, but found at least two "
            f"devices, cuda:{index} and cpu! (mstar_sim: {getattr(func, '__name__', func)} "
            f"over {', '.join(where)})"
        ))
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
        in_bounds = _check_bounds(func, args)
        with contextlib.nullcontext() if clock.FAST else clock.computing():
            if self.autocast is not None:
                args, kwargs = _autocast(func, args, kwargs, self.autocast)
            if not in_bounds or FAULT[0] or any(t.device.type == "meta" for t in _shallow(args, kwargs)):
                # a value-less operand (or a faulted device): the op runs on shapes only
                (args, kwargs), land = _join(args, kwargs)
                out = land(meta_call(func, args, kwargs, list(_shallow(args, kwargs))))
            elif func in _ALLOCATES:
                # an allocation, not a kernel (a graph doesn't re-run it): undefined values read
                # as zeros, never as the host's leftover bytes
                out = func(*args, **kwargs).zero_()
            elif clock.capturing() is not None and not func.is_view:
                out = _captured_values(func, args, kwargs)
            else:
                out = func(*args, **kwargs)
        if clock.FAST or func.is_view or clock.in_kernel():
            return out  # unpriced, or inside a stand-in (its own cost is charged)
        from mstar_sim import cost

        seconds = cost.op_seconds(func, args, kwargs, out)
        tape = clock.capturing()
        if tape is not None:
            tape.nodes.append((seconds, None))
        elif seconds:
            clock.charge(seconds, launched)
        return out


# A device-side assert: an index out of bounds in a kernel. As on CUDA it is asynchronous
# and sticky: the launch returns, the faulting kernel and every later one compute nothing,
# and once the device has got there (its stream's time at the launch) every call fails, a
# sync or host read first among them. So a server that ran one doesn't serve on.
FAULT: list = ["", 0.0]  # what faulted, and the device time it did


def check_fault() -> None:
    """Raise the device's fault, once the caller's time has reached it."""
    if FAULT[0] and clock.vnow() >= FAULT[1]:
        raise gpu_error(RuntimeError(f"CUDA error: device-side assert triggered (mstar_sim: {FAULT[0]})"))
_TAKES_NEGATIVE = {"index", "index_put", "index_put_", "_index_put_impl_", "take", "index_fill", "index_fill_"}


def _check_bounds(func, args) -> bool:
    """Whether the indices a kernel reads (values the emulator keeps) fall inside the
    tensor they index, as the CUDA kernel asserts; one that doesn't faults the device.
    Captured, the kernel runs at each replay, on that step's indices (an in-place
    ``clamp_`` before it has re-run by then): the check goes into the graph."""
    name = func.overloadpacket.__name__
    pairs = []
    if name == "embedding":
        pairs = [(args[1], args[0].shape[0])]
    elif name in ("index_select", "gather", "scatter", "scatter_", "scatter_add", "scatter_add_",
                  "index_copy", "index_copy_", "index_add", "index_add_", "index_fill", "index_fill_"):
        dim = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
        pairs = [(args[2], args[0].shape[dim])] if len(args) > 2 and args[0].dim() else []
    elif name == "take":
        pairs = [(args[1], args[0].numel())]
    elif name in _TAKES_NEGATIVE and len(args) > 1 and isinstance(args[1], (list, tuple)):
        dim = 0
        for ix in args[1]:
            if isinstance(ix, torch.Tensor) and ix.dtype in (torch.bool, torch.uint8):
                dim += ix.dim()  # a mask indexes as many dimensions as it has
                continue
            if dim < args[0].dim():
                pairs.append((ix, args[0].shape[dim]))
            dim += 1
    pairs = [(ix, size) for ix, size in pairs if isinstance(ix, torch.Tensor) and ix.device.type == "cpu"
             and ix.numel() and ix.dtype != torch.bool and not ix.is_floating_point()]
    if not pairs:
        return True
    if clock.capturing() is not None:
        clock.on_replay(lambda: _in_bounds(name, pairs))
        return True
    return _in_bounds(name, pairs)


def _in_bounds(name: str, pairs) -> bool:
    for ix, size in pairs:
        low = -size if name in _TAKES_NEGATIVE else 0
        bad = (ix < low) | (ix >= size)
        if bool(bad.any()):
            if not FAULT[0]:
                FAULT[:] = [f"{name}: index {int(ix[bad][0])} is out of bounds for size {size}",
                            clock.current_timeline().free_at]
            return False
    return True


# torch functions with an index operand, priced per aten op when an index holds values
# (so each is bounds-checked)
_INDEXING = {T.index_select, torch.index_select, T.gather, torch.gather, T.index_copy_, T.index_put_,
             T.scatter_, T.scatter, T.__getitem__, T.__setitem__, T.index_fill_}


def _real_index(flat) -> bool:
    return any(type(x) is SimTensor and not x.is_floating_point() and x._data.device.type == "cpu"
               and x._data.dim() > 0 for x in flat) or any(
        type(x) is torch.Tensor and x.device.type == "cpu" and not x.is_floating_point() and x.dim() > 0
        for x in flat)


# composite ops CUDA autocast casts whole, before they decompose
_F = torch.nn.functional
_AUTOCAST_COMPOSITES = {_F.scaled_dot_product_attention, torch.matmul, torch.einsum}  # with _AUTOCAST_LOW's below


def _istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False,
           onesided=None, length=None, return_complex=False):
    """istft's output shape: ATen's own reads its window envelope on the host."""
    frames = input.shape[-1]
    hop = hop_length or n_fft // 4
    n = length if length is not None else n_fft + hop * (frames - 1) - (2 * (n_fft // 2) if center else 0)
    dtype = input.dtype if return_complex else (input.real.dtype if input.is_complex() else input.dtype)
    out = torch.empty((*input.shape[:-2], n), dtype=dtype, device=META)
    from mstar_sim import cost

    clock.charge(cost.kernel_seconds("torch.istft", flops=0.0,
                                     bytes=float(input.numel() * input.element_size() * 2)))
    return out


# composite functions that read device values on the host in their CPU/meta
# decomposition: their output shapes, computed directly
_COMPOSITE_STAND_INS = {torch.istft: _istft, T.istft: _istft}


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
_AUTOCAST_COMPOSITES |= {getattr(_F, name) for name in _AUTOCAST_LOW if hasattr(_F, name)}
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


def _join(args, kwargs, ints: bool = True):
    """Kept values beside a value-less operand join it: the op runs on shapes, and its
    result has no values. 0-dim integer and bool tensors stay (a size, a count or a mask a
    shape depends on), as do integer operands when not ``ints`` (a composite's indices,
    checked at the aten op). Returns the joined arguments and ``land(out)``: a kept tensor
    the op wrote (in place, or as ``out=``) holds never-computed values now, zeros, and is
    what the op returns in place of its copy."""
    copies = {}

    def join(x):
        if isinstance(x, torch.Tensor) and x.device.type == "cpu" and (
                _floating(x) or (ints and x.dim() > 0)):
            copy = x.to(META)
            copies[id(copy)] = x
            return copy
        return x

    def land(out):
        def back(t):
            kept = copies.get(id(t)) if isinstance(t, torch.Tensor) else None
            if kept is None:
                return t
            kept.zero_()
            uncomputed()
            return kept

        return tree_map(back, out) if copies else out

    return tree_map(join, (args, kwargs)), land


def _writes(func, kwargs) -> bool:
    """Whether a torch function writes its first operand or an ``out=`` (by its name, as
    torch names in-place methods); at the aten level, an output that is an operand is one."""
    name = getattr(func, "__name__", "")
    return "out" in kwargs or name == "__setitem__" or name.startswith("__i") or (
        name.endswith("_") and not name.endswith("__"))


def _floating(x) -> bool:
    return x.dtype.is_floating_point or x.dtype.is_complex


_ALLOCATES = {
    torch.ops.aten.empty.memory_format, torch.ops.aten.empty_strided.default,
    torch.ops.aten.empty_like.default, torch.ops.aten.new_empty.default,
    torch.ops.aten.new_empty_strided.default,
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
    try:
        raw = func(*uargs, **ukwargs)
    except NotImplementedError:
        # no shape-only kernel, as for ops whose output shape depends on the
        # values (bincount, nonzero): run it on what the values read as, zeros
        return _on_zeros(func, uargs, ukwargs)
    if key is not None:
        desc = _describe(raw, tensors)
        if desc is not _UNCACHEABLE:
            _SHAPES[key] = desc
    return raw


# the most input bytes an op without a shape-only kernel is run on, as zeros
_ZEROS_LIMIT = 1 << 28


def _on_zeros(func, uargs, ukwargs):
    metas = [t for t in tree_flatten((uargs, ukwargs))[0] if _is_meta(t)]
    if sum(t.numel() * t.element_size() for t in metas) > _ZEROS_LIMIT:
        raise NotImplementedError(f"mstar_sim: {func} has no shape-only kernel and its inputs are too "
                                  "large to run on zeros")
    VALUE_SHAPES[0] += 1
    raw = func(*tree_map(_host_arg, uargs), **tree_map(_host_arg, ukwargs))

    def back(x):
        if isinstance(x, torch.Tensor) and x.device.type == "cpu" and (_floating(x) or x.numel() > SMALL_INT):
            return x.to(META)  # derived from never-computed values: only a shape's ints are kept
        return x

    return tree_map(back, raw)

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
    if any(t.device.type == "meta" for t in tensors) or (FAULT[0] and func not in _VIEWS):
        (uargs, ukwargs), land = _join(uargs, ukwargs)
        tensors = [t for t in _shallow(uargs, ukwargs)]
        raw = meta_call(func, uargs, ukwargs, tensors)
        if _writes(func, kwargs):
            raw = land(raw)
    else:
        if clock.capturing() is not None and func not in _VIEWS and func not in _UNINITIALIZED:
            # captured: deferred to the replays per aten op, and priced there
            with _Observe(torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None):
                raw = func(*uargs, **ukwargs)
            return _rewrap(raw, [x for x in flat if isinstance(x, torch.Tensor)], index)
        raw = func(*uargs, **ukwargs)  # small ints: real, and cheap
        if func in _UNINITIALIZED:
            raw.zero_()
            uncomputed()
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
            and not (func in _AUTOCAST_FN and torch.is_autocast_enabled("cuda"))
            and not (func in _INDEXING and _real_index(flat))):
        return _fast(func, args, kwargs, flat, index)
    uargs, ukwargs = tree_map(_unwrap, args), tree_map(_unwrap, kwargs)
    shallow = list(_shallow(uargs, ukwargs))
    land = None
    if any(t.device.type == "meta" for t in shallow) and any(t.device.type == "cpu" for t in shallow):
        # a composite checks its operands share a device before it decomposes
        # (stft's window, weight_norm's g): kept floats beside a value-less one join it
        (uargs, ukwargs), land = _join(uargs, ukwargs, ints=False)
        land = land if _writes(func, kwargs) else None
    standin = _COMPOSITE_STAND_INS.get(func)
    if standin is not None and any(_is_meta(t) for t in tree_flatten((uargs, ukwargs))[0]):
        return _rewrap(standin(*uargs, **ukwargs), [x for x in flat if isinstance(x, torch.Tensor)], index)
    if torch.is_autocast_enabled("cuda") and func in _AUTOCAST_COMPOSITES:
        # CUDA autocast casts these as a whole, before they decompose (sdpa's dtype check)
        low = torch.get_autocast_dtype("cuda")
        uargs, ukwargs = tree_map(lambda x: x.to(low) if isinstance(x, torch.Tensor) and x.dtype == torch.float32
                                  else x, (uargs, ukwargs))
    if isinstance(func, torch._ops.HigherOrderOperator):
        # (flex_attention) its own dispatch runs it on the payloads; a dispatch mode
        # would need a rule per operator, so its inner ops go unpriced
        raw = func(*uargs, **ukwargs)
        return _rewrap(raw, [x for x in flat if isinstance(x, torch.Tensor)], index)
    autocast = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
    with _Observe(autocast):
        raw = func(*uargs, **ukwargs)
    return _rewrap(land(raw) if land else raw, [x for x in flat if isinstance(x, torch.Tensor)], index)


def _is_custom_op(func) -> bool:
    ns = getattr(func, "namespace", None)  # a higher-order op has a namespace but no schema
    return ns is not None and ns not in ("aten", "prim", "prims") and hasattr(func, "_schema")


def _custom_impl(func):
    """The Python body a custom op runs on CUDA (its cuda kernel, else its default)."""
    from torch._library import custom_ops

    name = getattr(func, "_qualified_op_name", None) or func._schema.name
    opdef = custom_ops.OPDEFS.get(name.replace(".", "::", 1) if "::" not in name else name)
    if opdef is None:
        return None
    return opdef._backend_fns.get("cuda") or opdef._init_fn


def _function(func, args, kwargs):
    check_fault()
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
    if (getattr(func, "__name__", "") == "__set__" and getattr(func, "__self__", None) is _DATA_DESCRIPTOR
            and isinstance(args[0], SimTensor) and isinstance(args[1], SimTensor)):
        # `param.data = other`: the payload moves too, so the old storage is freed and the
        # parameter reads (and aliases) the new one, as on a GPU
        with torch._C.DisableTorchFunctionSubclass():
            func(*args, **kwargs)
        args[0]._data = args[1]._data
        return None
    if getattr(func, "__name__", "") == "__get__":
        # attribute getters (device, shape, dtype, data, ...) read the wrapper itself
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
    if func in _HOST_READS:
        host_wait()
    if func in _HOST_VALUED:
        host_wait()
        return func(*tree_map(_host_arg, args), **tree_map(_host_arg, kwargs))
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


def _captured_values(func, args, kwargs):
    """An aten op over values the emulator keeps, met while capturing. As on CUDA, capture
    runs nothing: its new outputs are graph buffers, shaped on meta, and each replay runs it
    (into them, or in place). An in-place metadata op (``unsqueeze_``, ``set_``) changes
    shapes once, at capture. An op whose output shape depends on the values (``nonzero``, a
    mask index) is refused, as CUDA refuses the host sync it needs."""
    if torch.Tag.inplace_view in func.tags:
        return func(*args, **kwargs)
    real = {}

    def shaped(x):
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
            return x
        m = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=META)
        real[id(m)] = x
        return m

    try:
        out = func(*tree_map(shaped, args), **tree_map(shaped, kwargs))
    except (NotImplementedError, RuntimeError) as e:
        if not isinstance(e, NotImplementedError) and torch.Tag.dynamic_output_shape not in func.tags:
            raise
        refuse_while_capturing(f"{func.overloadpacket.__name__}'s output shape depends on its input values")

    def buffer(m):
        if not isinstance(m, torch.Tensor):
            return m
        if id(m) in real:
            return real[id(m)]
        return torch.empty_strided(m.size(), m.stride(), dtype=m.dtype).zero_()

    out = tree_map(buffer, out)

    def rerun():
        fresh = func(*args, **kwargs)
        for dst, src in zip(tree_flatten(out)[0], tree_flatten(fresh)[0], strict=True):
            if isinstance(dst, torch.Tensor) and dst is not src:
                dst.copy_(src)

    clock.on_replay(rerun)
    return out


def replay(tape: clock.Tape) -> None:
    """Run a captured graph's nodes in order (their work on kept values, and the
    pricing of each kernel on the values before it), as one launch charging its
    kernels' time back to back."""
    from mstar_sim import cost

    check_fault()
    clock.STATS["replays"] += 1
    launched = clock.vnow()  # re-running the work on kept values is the emulator's time, not the host's
    total, kernels = 0.0, 0
    with contextlib.nullcontext() if clock.FAST else clock.computing():
        for seconds, effect in tape.nodes:
            if effect is not None and not FAULT[0]:  # the device faulted: nothing after it runs
                effect()
            if seconds is None or clock.FAST:
                continue
            priced = seconds if type(seconds) is float else clock.graph_node_seconds(seconds)
            total += priced
            kernels += priced > 0 or type(seconds) is not float
        if not clock.FAST:
            clock.charge(total + kernels * cost.graph_node_seconds() + cost.graph_launch_seconds(), launched)


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
            return _run(T.to, (self,), opts, self.device.index)  # a cast is a kernel, captured as one
        if device.type == "cpu":
            _d2h(self)
            if not non_blocking:
                host_wait()
            value = host_value(self._data)
            return value.to(copy=True, **{k: v for k, v in opts.items() if k != "copy"})
        return self._data.to(device, **opts)
    if is_cuda_device(device):
        if self.device.type not in ("cpu", "meta"):
            raise RuntimeError(f"mstar_sim: can't move a {self.device} tensor to cuda")
        if not non_blocking:
            refuse_while_capturing("a blocking host-to-device copy")
        with torch._C.DisableTorchFunctionSubclass():
            src = self.to(dtype=dtype) if dtype is not None and dtype != self.dtype else self
        data = to_device_storage(src)
        if data.device.type == "cpu":  # captured: reads its host buffer at each replay, not now
            clock.on_replay(lambda: data.copy_(self))
        clock.charge(_copy_seconds(data.numel() * data.element_size(), "h2d"))
        return SimTensor(data, _index(device))
    if _is_meta(self) and (device is None or device.type in ("cpu", "meta")):
        # a header-only weight cast or moved on the host stays value-less
        return torch.empty_like(self, dtype=dtype or self.dtype, device=META)
    with torch._C.DisableTorchFunctionSubclass():
        return T.to(self, *args, copy=copy, **kwargs)


def _d2h(src) -> None:
    """A device-to-host copy runs on the current stream, after what is queued there."""
    clock.charge(_copy_seconds(src.numel() * src.element_size(), "d2h"))


def _cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
    target = torch.device("cuda", _index(device if device is not None else "cuda"))
    return _to(self, target, non_blocking=non_blocking)


def _cpu(self, memory_format=torch.preserve_format):
    _d2h(self)
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
            if not non_blocking:
                refuse_while_capturing("a blocking host-to-device copy")
            clock.charge(_copy_seconds(src.numel() * src.element_size(), "h2d"))
            if not _is_meta(self._data) and not _is_meta(src):
                dst = self._data
                if clock.capturing() is None:
                    dst.copy_(src, non_blocking=non_blocking)
                else:  # captured: reads its host buffer at each replay, not now
                    clock.on_replay(lambda: dst.copy_(src))
            return self
        return _run(T.copy_, (self, src), {"non_blocking": non_blocking}, self.device.index)
    # device-to-host into a CPU tensor
    _d2h(src)
    if not non_blocking:
        host_wait()
    if not _is_meta(self):
        if clock.capturing() is None:
            self.copy_(host_value(src._data))
        elif not _is_meta(src._data):  # captured: lands at each replay (a host mailbox), not now
            dst, data = self, src._data
            clock.on_replay(lambda: dst.copy_(data))
    return self


_METHODS = {
    T.to: _to,
    T.cuda: _cuda,
    T.cpu: _cpu,
    T.copy_: _copy_,
    T.tolist: _host_read("tolist"),
    T.numpy: _host_read("numpy"),
    T.item: _host_read("item"),
    T.__format__: _host_read("__format__"),
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
_DATA_DESCRIPTOR = torch._C.TensorBase.__dict__["data"]
# functions that answer the host with a Python value: computed on the host values
# (zeros for what was never computed), since a meta tensor has none
_HOST_VALUED = {torch.equal, T.equal, torch.allclose, T.allclose}
_READS = {T.tolist, T.item, T.numpy, T.__bool__, T.__int__, T.__float__, T.__index__, T.__format__}
# copies, indexing, and load_state_dict(assign=True) into a module built on the CPU
_MIXED_OK = {T.copy_, torch.Tensor.__getitem__, torch.Tensor.__setitem__, T.index_put_,
             torch.index_put, T.to, T.__getitem__, T.module_load}


def _host_arg(x):
    if isinstance(x, SimTensor):
        return host_value(x._data)
    return host_value(x) if _is_meta(x) else x


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
    check_fault()
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
    from_host = func in (torch.tensor, torch.as_tensor, torch.from_numpy)
    if from_host:
        refuse_while_capturing(f"torch.{func.__name__} copies host data synchronously")
    shape = func(*args, **dict(kwargs, device=META))
    if isinstance(shape, T) and not keeps_values(shape.dtype, shape.numel()):
        target = META
    with _Observe():
        out = func(*args, **dict(kwargs, device=target))
    if target is CPU and func in _UNINITIALIZED:
        out.zero_()  # a never-computed value reads as zeros, not as the host heap's garbage
        uncomputed()
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
        if args and _is_meta(args[0]) and (func in _READS or func in _HOST_VALUED):
            return func(*tree_map(_host_arg, args), **tree_map(_host_arg, kwargs))
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
        tensors = [t for t in _shallow(args, kwargs) if isinstance(t, T)]
        if any(t.device.type == "meta" for t in tensors) and any(t.device.type == "cpu" for t in tensors):
            # host work mixing a never-computed value (a header-only weight) with real data:
            # the value reads as zeros, as any host read of it does
            return func(*tree_map(_host_arg, args), **tree_map(_host_arg, kwargs))
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


# The ``torch.ops`` namespaces mstar may call and how each runs on the emulated GPU
# (read by the contract test in tests/test_device.py). Compiled CUDA extensions are
# not built on a CPU box, so mstar takes its torch path for them.
OPS_NAMESPACES = {
    "aten": "dispatched and priced",
    "prims": "dispatched and priced",
    "mstar": "Python custom ops: their bodies run on the emulated GPU",
    "_C": "mstar's compiled CUDA extension: not built here, mstar takes its torch path",
    "_mstar_moe_C": "mstar's compiled MoE extension: not built here, mstar takes its torch path",
    "_xpu_C": "XPU only",
    "symm_mem": "symmetric memory needs NVSHMEM: absent here, mstar keeps NCCL",
}
# Python packages for CUDA mstar may import, absent on the emulated GPU, and why
# that is right
ABSENT_MODULES = {
    "cuda": "cuda-python: the KV page copier falls back to its torch copy, which is priced",
}


def _swap_or_assign() -> None:
    """Moving a module swaps its parameters in place (so tied weights stay tied), but a
    swap refuses a tensor something else holds (weight_norm's hooks): that module alone
    is then moved or loaded by assignment, as torch does without the swap, on its thread
    alone (the swap setting reads per thread)."""
    from torch import nn

    future, assigning = torch.__future__, threading.local()
    swaps = future.get_swap_module_params_on_conversion
    future.get_swap_module_params_on_conversion = lambda: swaps() and not getattr(assigning, "on", False)

    def by_assignment(call, *args):
        assigning.on = True
        try:
            return call(*args)
        finally:
            assigning.on = False

    apply, load = nn.Module._apply, nn.Module._load_from_state_dict

    def apply_one(self, fn, recurse=True):
        try:
            return apply(self, fn, recurse)
        except RuntimeError as e:  # its own parameters (a child retries its own)
            if not str(e).startswith("_apply(): Couldn't swap"):
                raise
            return by_assignment(apply, self, fn, False)

    def load_one(self, state_dict, prefix, metadata, strict, missing, unexpected, errors):
        marks = len(missing), len(unexpected), len(errors)
        load(self, state_dict, prefix, metadata, strict, missing, unexpected, errors)
        if any(e.startswith("While swapping") for e in errors[marks[2]:]):
            for found, mark in zip((missing, unexpected, errors), marks, strict=True):
                del found[mark:]
            by_assignment(load, self, state_dict, prefix, metadata, strict, missing, unexpected, errors)

    nn.Module._apply = apply_one
    nn.Module._load_from_state_dict = load_one


def _relax_blockwise_scaled_mm_meta() -> None:
    """torch 2.10's meta check for block-scaled fp8 GEMMs (``_scaled_mm_v2``, 1x128 x 128x128
    blocks) floors ``N / 128``, so a weight whose last 128-row block is partial (GLM-5.2's
    fused q_a + kv_a) fails it, though the CUDA kernel takes it. Here the check only
    decides the output's shape: ``[M, N]``."""
    import torch._meta_registrations as mr

    check = getattr(mr, "_check_scaled_mm_sizes_v2", None)
    if check is None or getattr(check, "_mstar_sim", False):
        return

    def relaxed(self, mat2, *args, out_dtype=None, **kwargs):
        try:
            return check(self, mat2, *args, out_dtype=out_dtype, **kwargs)
        except RuntimeError as e:
            if "blockwise scaling" not in str(e):
                raise
            return torch.empty(self.shape[0], mat2.shape[1], dtype=out_dtype or self.dtype, device=self.device)

    relaxed._mstar_sim = True
    mr._check_scaled_mm_sizes_v2 = relaxed


# --------------------------------------------------------------- installation

_installed = False


def install() -> None:
    """Enter ``SimMode`` on this thread and on every thread started after."""
    global _installed
    if _installed:
        return
    _installed = True
    new_shared = torch.UntypedStorage._new_shared_cuda

    def _new_shared_cuda(*args):
        handle = args[1] if len(args) > 1 else b""
        if isinstance(handle, bytes) and handle.startswith(_SHARED):
            storage = torch.UntypedStorage(int(handle.split(b":")[-1]), device=META)
            storage._sim_imported = True  # the exporter's memory, not this process's
            return storage
        return new_shared(*args)

    torch.UntypedStorage._new_shared_cuda = staticmethod(_new_shared_cuda)
    _relax_blockwise_scaled_mm_meta()
    import faulthandler

    # a read through a fake address prints its thread's Python stack (all threads'
    # from inside a signal handler is not safe while they run: install.thread_dumps)
    faulthandler.enable(all_threads=False)
    # Moving a module to cuda keeps each Parameter object, as on a GPU (where a
    # CPU and a CUDA tensor share an impl type and torch sets `.data`): tied
    # weights stay tied and references taken before the move stay live.
    torch.__future__.set_swap_module_params_on_conversion(True)
    _swap_or_assign()
    SimMode().__enter__()
    bootstrap = threading.Thread._bootstrap_inner

    def bootstrap_inner(self):
        SimMode().__enter__()
        bootstrap(self)

    threading.Thread._bootstrap_inner = bootstrap_inner
