"""``torch.cuda`` / ``torch.accelerator`` for the emulated GPU.

Streams and events run on :mod:`mstar_sim.clock`; graphs record a
:class:`~mstar_sim.clock.Tape` and replay it through
:func:`mstar_sim.device.replay`. Memory figures are the caching allocator's
accounting of the simulated tensors' storages (``mstar_sim.device.Pool``) against
the card's capacity (``MSTAR_SIM_MEMORY_GB``, by default the card's).
"""

from __future__ import annotations

import collections
import itertools
import os
import threading
import time
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from mstar_sim import clock

_tls = threading.local()
_ids = itertools.count(1)


class Spec(SimpleNamespace):
    pass


def _spec() -> Spec:
    try:
        from mstar_sim import gpus

        gpu = gpus.lookup(os.environ.get("MSTAR_SIM_GPU", "h100"))
    except (ImportError, AttributeError):  # the spec table is optional for the device itself
        gpu = SimpleNamespace(name="NVIDIA H100 80GB HBM3", capability=(9, 0), sm_count=132,
                              memory_bytes=80 << 30)
    mem_gb = os.environ.get("MSTAR_SIM_MEMORY_GB")
    total = int(float(mem_gb) * (1 << 30)) if mem_gb else gpu.memory_bytes
    return Spec(gpu=gpu, total=total)


SPEC: Spec | None = None


def spec() -> Spec:
    global SPEC
    if SPEC is None:
        SPEC = _spec()
    return SPEC


def device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip() != ""])
    return int(os.environ.get("MSTAR_SIM_NUM_GPUS", "8"))


def current_device() -> int:
    return getattr(_tls, "device", 0)


def set_device(device) -> None:
    if isinstance(device, (str, torch.device)):
        device = torch.device(device).index or 0
    _tls.device = int(device)


class device:  # noqa: N801  (torch.cuda.device)
    def __init__(self, device):
        self.idx = -1 if device is None else _as_index(device)

    def __enter__(self):
        self.prev = current_device()
        if self.idx >= 0:
            set_device(self.idx)

    def __exit__(self, *exc):
        set_device(self.prev)
        return False


def _as_index(device) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    return torch.device(device).index or 0


def get_device_name(device=None) -> str:
    return spec().gpu.name


def get_device_capability(device=None) -> tuple[int, int]:
    return tuple(spec().gpu.capability)


def get_device_properties(device=None):
    gpu = spec().gpu
    major, minor = gpu.capability
    return SimpleNamespace(
        name=gpu.name, total_memory=spec().total, major=major, minor=minor,
        multi_processor_count=gpu.sm_count, is_integrated=False, is_multi_gpu_board=False,
        L2_cache_size=getattr(gpu, "l2_bytes", 50 << 20), warp_size=32,
        max_threads_per_multi_processor=2048, regs_per_multiprocessor=65536,
        uuid=f"mstar-sim-{_as_index(device)}", pci_bus_id=_as_index(device), pci_device_id=0,
        pci_domain_id=0, gcnArchName="",
    )


# ------------------------------------------------------------------ memory


def _used(device) -> list[int]:
    """``[current, peak]`` bytes of a simulated device's live tensors (their storages)."""
    from mstar_sim import device as dev

    return dev.ALLOCATED.setdefault(_as_index(device), [0, 0])


def memory_allocated(device=None) -> int:
    return _used(device)[0]


def max_memory_allocated(device=None) -> int:
    return _used(device)[1]


def memory_reserved(device=None) -> int:
    """What the caching allocator holds from the card: every pool's high-water mark
    (``mstar_sim.device.Pool``)."""
    from mstar_sim import device as dev

    return dev.reserved(_as_index(device))


def max_memory_reserved(device=None) -> int:
    from mstar_sim import device as dev

    return dev.RESERVED_PEAK.get(_as_index(device), 0)


def reset_peak_memory_stats(device=None) -> None:
    from mstar_sim import device as dev

    used = _used(device)
    used[1] = used[0]
    dev.RESERVED_PEAK[_as_index(device)] = memory_reserved(device)


def empty_cache() -> None:
    from mstar_sim import device as dev

    for index in range(device_count()):
        dev.release_cached(index)


def mem_get_info(device=None) -> tuple[int, int]:
    total = spec().total
    return total - memory_reserved(device), total


_fraction: dict[int, float] = {}


def set_per_process_memory_fraction(fraction: float, device=None) -> None:
    _fraction[_as_index(device)] = fraction


def get_per_process_memory_fraction(device=None) -> float:
    return _fraction.get(_as_index(device), 1.0)


def memory_limit(index: int) -> int:
    """The bytes this process may reserve on device ``index``: past it, out of memory."""
    return int(spec().total * get_per_process_memory_fraction(index))


def memory_stats(device=None) -> dict:
    """The figures the emulator keeps; every other stat torch's allocator reports reads 0."""
    return collections.defaultdict(int, {"allocated_bytes.all.current": memory_allocated(device),
                                         "allocated_bytes.all.peak": max_memory_allocated(device),
                                         "reserved_bytes.all.current": memory_reserved(device),
                                         "reserved_bytes.all.peak": max_memory_reserved(device),
                                         "num_alloc_retries": 0, "num_ooms": 0})


def memory_snapshot(*_, **__) -> list[dict]:
    """One segment per pool (no blocks): what reads a pool's size off the snapshot sees it."""
    from mstar_sim import device as dev

    return [{"device": p.device, "segment_pool_id": p.handle, "total_size": p.reserved,
             "allocated_size": p.live, "active_size": p.live, "requested_size": p.live,
             "stream": 0, "segment_type": "large", "address": 0, "blocks": []}
            for p in list(dev.POOLS.values()) if p.reserved]


# graph pools, as the caching allocator's (torch._C._cuda_*AllocateToPool, _cuda_releasePool)


def _begin_allocate_to_pool(device, mempool_id, *_) -> None:
    """This thread's allocations on ``device`` go to the pool (a capture's stream filter, here
    its thread), until it ends."""
    from mstar_sim import device as dev

    with dev._track_lock:
        p = dev.pool(_as_index(device), mempool_id)
        p.graphs += 1
        p.recording = threading.get_ident()


def _end_allocate_to_pool(device, mempool_id) -> None:
    from mstar_sim import device as dev

    with dev._track_lock:
        p = dev.POOLS.get((_as_index(device), tuple(mempool_id)))
        if p is None or p.recording is None:
            raise dev.gpu_error(RuntimeError(f"mstar_sim: pool {tuple(mempool_id)} is not recording"))
        p.recording = None


def _release_pool(device, mempool_id) -> None:
    from mstar_sim import device as dev

    with dev._track_lock:
        p = dev.POOLS.get((_as_index(device), tuple(mempool_id)))
        if p is not None:
            p.graphs -= 1


# ------------------------------------------------------------ streams, events


class Stream:
    """A stream of the emulated GPU: a :class:`clock.Timeline`."""

    def __init__(self, device=None, priority: int = 0, **_):
        self.device = torch.device("cuda", _as_index(device) if device is not None else current_device())
        self.device_index = self.device.index
        self.device_type = "cuda"
        self.priority = priority
        self.stream_id = next(_ids)
        self.cuda_stream = self.stream_id
        self.timeline = clock.Timeline(f"stream{self.stream_id}")
        _STREAMS[self.stream_id] = self

    def wait_event(self, event: Event) -> None:
        if event is not None and event.t is not None:
            self.timeline.free_at = max(self.timeline.free_at, event.t)

    def wait_stream(self, stream: Stream) -> None:
        self.timeline.free_at = max(self.timeline.free_at, stream.timeline.free_at)

    def record_event(self, event: Event | None = None) -> Event:
        event = event if event is not None else Event()
        event.record(self)
        return event

    def query(self) -> bool:
        _check_fault()
        return time.perf_counter() >= self.timeline.free_at

    def synchronize(self) -> None:
        clock.wait_until(self.timeline.free_at)
        _check_fault()

    def __enter__(self):
        # as torch's THPStream_enter: the saved context lives on the stream object,
        # so a second enter (nested, or from another thread) raises
        if getattr(self, "_ctx", None) is not None:
            from mstar_sim.device import gpu_error

            raise gpu_error(RuntimeError("Stream's context should not be initialized."))
        self._ctx = stream(self)
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        ctx, self._ctx = self._ctx, None
        return ctx.__exit__(*exc)

    def __repr__(self):
        return f"<mstar_sim Stream {self.stream_id} on {self.device}>"


_defaults: dict[int, Stream] = {}


def default_stream(device=None) -> Stream:
    idx = _as_index(device)
    s = _defaults.get(idx)
    if s is None:
        s = _defaults[idx] = Stream(idx)
        # a process drives one GPU (its worker's), so every index's default
        # stream is the one timeline kernels are charged to by default
        s.timeline = clock.DEFAULT
    return s


def current_stream(device=None) -> Stream:
    s = getattr(_tls, "stream", None)
    return s if s is not None else default_stream(device)


def set_stream(s: Stream | None) -> None:
    _tls.stream = s
    clock.set_current_timeline(None if s is None else s.timeline)


class stream:  # noqa: N801  (torch.cuda.stream / StreamContext)
    def __init__(self, s: Stream | None):
        self.s = s

    def __enter__(self):
        self.prev = getattr(_tls, "stream", None)
        if self.s is not None:
            set_stream(self.s)
        return self.s

    def __exit__(self, *exc):
        if self.s is not None:
            set_stream(self.prev)
        return False


StreamContext = stream


class Event:
    """A point on a stream's timeline."""

    def __init__(self, device=None, *, enable_timing: bool = False, blocking: bool = False,
                 interprocess: bool = False, external: bool = False, **_):
        if isinstance(device, bool):  # torch.cuda.Event(enable_timing) positional
            enable_timing, device = device, None
        self.enable_timing = enable_timing
        self.t: float | None = None
        self.stamp: float | None = None
        self.device = torch.device("cuda", current_device())

    def record(self, stream: Stream | None = None) -> None:
        _check_fault()
        s = stream if stream is not None else current_stream()
        # complete once the stream drains past it: on an idle stream that is now, for
        # every thread (the recording thread's own virtual time would skew a waiter on
        # another); the timestamp, for elapsed_time, is when the GPU reaches it
        self.t = s.timeline.free_at
        self.stamp = max(clock.vnow(), self.t)

    def wait(self, stream: Stream | None = None) -> None:
        (stream if stream is not None else current_stream()).wait_event(self)

    def query(self) -> bool:
        _check_fault()
        return self.t is None or time.perf_counter() >= self.t

    def synchronize(self) -> None:
        if self.t is not None:
            clock.wait_until(self.t)
        _check_fault()

    def elapsed_time(self, end: Event) -> float:
        return (end.stamp - self.stamp) * 1000.0


def synchronize(device=None) -> None:
    streams = [s.timeline.free_at for s in _all_streams()]
    clock.wait_until(max([clock.DEFAULT.free_at, *streams]))
    _check_fault()


def _check_fault() -> None:
    """A device-side assert the device has reached fails every CUDA call after it."""
    from mstar_sim import device

    device.check_fault()




def _all_streams():
    return list(_STREAMS.values())


# every live stream by its handle (``cuda_stream``): what a device-wide sync
# waits on, and what an ``ExternalStream`` of a raw handle names
_STREAMS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def ExternalStream(stream_ptr: int, device=None, **_):  # noqa: N802  (torch.cuda.ExternalStream)
    stream = _STREAMS.get(stream_ptr)
    if stream is None:
        raise RuntimeError(f"mstar_sim: no emulated stream has the handle {stream_ptr}")
    return stream


# ------------------------------------------------------------------- graphs


def graph_pool_handle():
    # torch's is a (int, int) id
    return (next(_ids), 0)


class CUDAGraph:
    """A captured graph: its tape, and the private pool its allocations come from,
    which it holds until reset (torch's ``use_count``)."""

    def __init__(self, keep_graph: bool = False):
        self.tape: clock.Tape | None = None
        self._pool = None
        self._device = None
        self._prev = None
        self._holds_pool = False

    def capture_begin(self, pool=None, capture_error_mode: str = "global"):
        self._pool = tuple(pool) if pool is not None else graph_pool_handle()
        self._device = current_device()
        _begin_allocate_to_pool(self._device, self._pool)
        self._holds_pool = True
        self.tape = clock.Tape()
        self._prev = clock.set_capturing(self.tape)

    def capture_end(self):
        clock.set_capturing(self._prev)
        _end_allocate_to_pool(self._device, self._pool)

    def instantiate(self):
        pass

    def replay(self):
        from mstar_sim import device

        device.replay(self.tape)

    def reset(self):
        self.tape = None
        if self._holds_pool:
            self._holds_pool = False
            _release_pool(self._device, self._pool)

    def __del__(self):
        try:
            self.reset()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def pool(self):
        return self._pool

    def raw_cuda_graph(self) -> int:
        from mstar_sim import libcuda

        if getattr(self, "_handle", None) is None:
            self._handle = libcuda.register(self)
        return self._handle

    def enable_debug_mode(self):
        pass

    def debug_dump(self, path):
        pass


class graph:  # noqa: N801  (torch.cuda.graph)
    def __init__(self, cuda_graph: CUDAGraph, pool=None, stream: Stream | None = None,
                 capture_error_mode: str = "global"):
        self.cuda_graph = cuda_graph
        self.pool = pool
        self.capture_stream = stream if stream is not None else Stream()
        self.capture_error_mode = capture_error_mode

    def __enter__(self):
        import gc

        synchronize()
        gc.collect()  # as torch does: what the capture can reuse is freed first
        empty_cache()
        self.ctx = globals()["stream"](self.capture_stream)
        self.ctx.__enter__()
        self.cuda_graph.capture_begin(self.pool, capture_error_mode=self.capture_error_mode)

    def __exit__(self, *exc):
        self.cuda_graph.capture_end()
        self.ctx.__exit__(*exc)
        return False


def is_current_stream_capturing() -> bool:
    return clock.capturing() is not None


@contextmanager
def _noop_ctx(*_, **__):
    yield


# the CUDA names mstar may reach that are not stand-ins, and why that is right on
# the emulated GPU (read by the contract test in tests/test_device.py)
PASSTHROUGH = {
    "torch.cuda.OutOfMemoryError": "an exception class, nothing to emulate",
    "torch.cuda.nvtx": "a namespace; install() makes its range and mark functions no-ops",
}


def _noop(*_, **__):
    return None


# ------------------------------------------------------------------ install


_TorchGenerator = torch.Generator


class Generator(_TorchGenerator):
    """``torch.Generator``, which on ``cuda:N`` is a CPU generator reporting that
    device: what it seeds is never computed, but seeding and its state work."""

    def __new__(cls, device="cpu"):
        dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        gen = _TorchGenerator.__new__(cls, "cpu" if dev.type == "cuda" else dev)
        gen._sim_device = torch.device("cuda", dev.index or 0) if dev.type == "cuda" else None
        return gen

    @property
    def device(self):
        return self._sim_device or _TorchGenerator.device.__get__(self)


def _set_sync_debug_mode(mode) -> None:
    from mstar_sim import device

    names = {"default": 0, "warn": 1, "error": 2}
    device.SYNC_DEBUG[0] = names[mode] if isinstance(mode, str) else int(mode)


def _get_sync_debug_mode() -> int:
    from mstar_sim import device

    return device.SYNC_DEBUG[0]


TORCH_C = {
    "_cuda_clearCublasWorkspaces": _noop,
    "_cuda_set_sync_debug_mode": _set_sync_debug_mode,
    "_cuda_get_sync_debug_mode": _get_sync_debug_mode,
    "_cuda_sleep": _noop,  # a spin of N cycles; nothing to wait for
    "_cuda_getDevice": current_device,
    # a raw ``cudaStream_t``: the stream's id (no raw CUDA call takes it here, libcudart is refused)
    "_cuda_getCurrentRawStream": lambda device=None: current_stream().cuda_stream,
    "_cuda_beginAllocateToPool": _begin_allocate_to_pool,
    "_cuda_endAllocateToPool": _end_allocate_to_pool,
    "_cuda_releasePool": _release_pool,
}


def install() -> None:
    """Make ``torch.cuda`` and ``torch.accelerator`` describe the emulated GPU."""
    tc = torch.cuda
    for name, value in {
        "is_available": lambda: True,
        "device_count": device_count,
        "current_device": current_device,
        "set_device": set_device,
        "device": device,
        "get_device_name": get_device_name,
        "get_device_capability": get_device_capability,
        "get_device_properties": get_device_properties,
        "mem_get_info": mem_get_info,
        "memory_allocated": memory_allocated,
        "memory_reserved": memory_reserved,
        "max_memory_allocated": max_memory_allocated,
        "max_memory_reserved": max_memory_reserved,
        "reset_peak_memory_stats": reset_peak_memory_stats,
        "memory_stats": memory_stats,
        "memory_snapshot": memory_snapshot,
        "empty_cache": empty_cache,
        "set_per_process_memory_fraction": set_per_process_memory_fraction,
        "get_per_process_memory_fraction": get_per_process_memory_fraction,
        "Stream": Stream,
        "ExternalStream": ExternalStream,
        "Event": Event,
        "current_stream": current_stream,
        "default_stream": default_stream,
        "set_stream": set_stream,
        "stream": stream,
        "StreamContext": stream,
        "synchronize": synchronize,
        "CUDAGraph": CUDAGraph,
        "graph": graph,
        "graph_pool_handle": graph_pool_handle,
        "is_current_stream_capturing": is_current_stream_capturing,
        "init": _noop,
        "_lazy_init": _noop,
        "is_initialized": lambda: True,
        "ipc_collect": _noop,
        "get_arch_list": lambda: ["sm_90", "sm_100"],
    }.items():
        setattr(tc, name, value)
    for name in ("CUDAGraph", "graph", "graph_pool_handle", "is_current_stream_capturing"):
        setattr(tc.graphs, name, getattr(tc, name))
    tc.memory.memory_stats = memory_stats  # what torch.cuda.memory_summary reads
    tc.nvtx.range_push = _noop
    tc.nvtx.range_pop = _noop
    tc.nvtx.mark = _noop
    tc.nvtx.range = _noop_ctx
    torch.version.cuda = torch.version.cuda or "12.8"
    for name, value in TORCH_C.items():
        setattr(torch._C, name, value)
    from mstar_sim import libcuda

    libcuda.install()

    acc = torch.accelerator
    for name, value in {
        "is_available": lambda: True,
        "current_accelerator": lambda check_available=False: torch.device("cuda"),
        "device_count": device_count,
        "current_device_index": current_device,
        "current_device_idx": current_device,
        "set_device_index": set_device,
        "set_device_idx": set_device,
        "device_index": device,
        "current_stream": lambda device=None: current_stream(device),
        "set_stream": set_stream,
        "synchronize": synchronize,
        "memory_allocated": memory_allocated,
        "max_memory_allocated": max_memory_allocated,
        "memory_reserved": memory_reserved,
        "max_memory_reserved": max_memory_reserved,
        "reset_peak_memory_stats": reset_peak_memory_stats,
        "empty_cache": empty_cache,
        "get_memory_info": mem_get_info,
    }.items():
        setattr(acc, name, value)
    torch.Stream = Stream
    torch.Event = Event
    torch.Generator = Generator
    # torch.cuda's seeding and RNG-state functions (and dynamo's RNG save) use these
    tc.default_generators = tuple(Generator(f"cuda:{i}") for i in range(device_count()))
    for name in ("_lazy_init", "current_device", "device_count", "is_initialized"):
        setattr(tc.random, name, getattr(tc, name))  # bound there at its import
    # the C++ binding refuses an emulated stream before any override sees it, and the
    # caching allocator's cross-stream lifetime has nothing to defer here
    torch.Tensor.record_stream = lambda self, stream: None
