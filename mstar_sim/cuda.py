"""``torch.cuda`` / ``torch.accelerator`` for the emulated GPU.

Streams and events run on :mod:`mstar_sim.clock`; graphs record a
:class:`~mstar_sim.clock.Tape` and replay it through
:func:`mstar_sim.device.replay`. Memory figures are the simulated tensors'
storages against the card's capacity (``MSTAR_SIM_MEMORY_GB``, by default the
card's).
"""

from __future__ import annotations

import itertools
import os
import threading
import time
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


def memory_allocated(device=None) -> int:
    """Bytes of the simulated device's live tensors (their storages)."""
    from mstar_sim import device as dev

    return dev.ALLOCATED[0]


def max_memory_allocated(device=None) -> int:
    from mstar_sim import device as dev

    return dev.ALLOCATED[1]


def reset_peak_memory_stats(device=None) -> None:
    from mstar_sim import device as dev

    dev.ALLOCATED[1] = dev.ALLOCATED[0]


def mem_get_info(device=None) -> tuple[int, int]:
    total = spec().total
    return max(0, total - memory_allocated()), total


_fraction = 1.0


def set_per_process_memory_fraction(fraction: float, device=None) -> None:
    global _fraction
    _fraction = fraction


def get_per_process_memory_fraction(device=None) -> float:
    return _fraction


def memory_stats(device=None) -> dict:
    used = memory_allocated()
    return {"allocated_bytes.all.current": used, "reserved_bytes.all.current": used,
            "allocated_bytes.all.peak": max_memory_allocated()}


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
        return time.perf_counter() >= self.timeline.free_at

    def synchronize(self) -> None:
        clock.wait_until(self.timeline.free_at)

    def __enter__(self):
        # as torch's THPStream_enter: the saved context lives on the stream object,
        # so a second enter (nested, or from another thread) raises
        if getattr(self, "_ctx", None) is not None:
            raise RuntimeError("Stream's context should not be initialized.")
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
        self.device = torch.device("cuda", current_device())

    def record(self, stream: Stream | None = None) -> None:
        s = stream if stream is not None else current_stream()
        self.t = max(clock.vnow(), s.timeline.free_at)

    def wait(self, stream: Stream | None = None) -> None:
        (stream if stream is not None else current_stream()).wait_event(self)

    def query(self) -> bool:
        return self.t is None or time.perf_counter() >= self.t

    def synchronize(self) -> None:
        if self.t is not None:
            clock.wait_until(self.t)

    def elapsed_time(self, end: Event) -> float:
        return (end.t - self.t) * 1000.0


def synchronize(device=None) -> None:
    streams = [s.timeline.free_at for s in _all_streams()]
    clock.wait_until(max([clock.DEFAULT.free_at, *streams]))


_live_streams: list = []


def _all_streams():
    return [current_stream(), *_defaults.values()]


# ------------------------------------------------------------------- graphs


class _Pool:
    pass


def graph_pool_handle():
    # torch's is a (int, int) id
    return (next(_ids), 0)


class CUDAGraph:
    def __init__(self, keep_graph: bool = False):
        self.tape: clock.Tape | None = None
        self._pool = None
        self._prev = None

    def capture_begin(self, pool=None, capture_error_mode: str = "global"):
        self._pool = pool if pool is not None else graph_pool_handle()
        self.tape = clock.Tape()
        self._prev = clock.set_capturing(self.tape)

    def capture_end(self):
        clock.set_capturing(self._prev)

    def instantiate(self):
        pass

    def replay(self):
        from mstar_sim import device

        device.replay(self.tape)

    def reset(self):
        self.tape = None

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
        synchronize()
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


def _noop(*_, **__):
    return None


# ------------------------------------------------------------------ install


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
        "memory_reserved": memory_allocated,
        "max_memory_allocated": max_memory_allocated,
        "max_memory_reserved": max_memory_allocated,
        "reset_peak_memory_stats": reset_peak_memory_stats,
        "memory_stats": memory_stats,
        "memory_snapshot": lambda *a, **k: [],
        "empty_cache": _noop,
        "set_per_process_memory_fraction": set_per_process_memory_fraction,
        "get_per_process_memory_fraction": get_per_process_memory_fraction,
        "Stream": Stream,
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
        "manual_seed": _noop,
        "manual_seed_all": _noop,
        "seed": _noop,
        "seed_all": _noop,
        "ipc_collect": _noop,
        "get_arch_list": lambda: ["sm_90", "sm_100"],
    }.items():
        setattr(tc, name, value)
    for name in ("CUDAGraph", "graph", "graph_pool_handle", "is_current_stream_capturing"):
        setattr(tc.graphs, name, getattr(tc, name))
    tc.nvtx.range_push = _noop
    tc.nvtx.range_pop = _noop
    tc.nvtx.mark = _noop
    tc.nvtx.range = _noop_ctx
    torch.version.cuda = torch.version.cuda or "12.8"
    torch._C._cuda_clearCublasWorkspaces = _noop
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
        "memory_reserved": memory_allocated,
        "max_memory_reserved": max_memory_allocated,
        "reset_peak_memory_stats": reset_peak_memory_stats,
        "empty_cache": _noop,
        "get_memory_info": mem_get_info,
    }.items():
        setattr(acc, name, value)
    torch.Stream = Stream
    torch.Event = Event
