"""Named CUDA streams, and the two-branch fork used inside graph captures.

Code asks for a stream by what it is *for* -- ``get_stream(PLAN, device)`` --
and a layout maps those roles onto physical streams. Every role other than
``DEFAULT`` is off the compute stream and fenced against it with events, so
roles can share a physical stream without deadlocking; sharing only orders
their work FIFO. The layout comes from ``MSTAR_STREAMS``:

* ``aux`` (default): three streams at most -- ``default`` for compute,
  ``offload`` for KV offload / reload (so a multi-MB copy never queues a
  step's pre-plan behind it), and ``aux`` for every other role. ``shared``
  is an alias.
* ``split``: one physical stream per role.
* ``role=physical,...``: explicit, e.g. ``plan=aux,send=aux``; unlisted
  roles get a stream of their own name.

Physical streams are created lazily, once per (device, name) per process,
on the first ``get`` of a role that maps to them -- a deployment that never
offloads never creates ``offload`` -- instead of once per object that wants
one, so a worker's stream count stays fixed however many graph runners, pools
or transports it builds.

What runs on a side stream matters more than how many there are: work on the
SMs there (a kernel, or a device-to-device copy) while a CUDA graph replays
slows every graph node in the process from then on. Side-stream work beside a
live graph should be H2D / D2H copies only -- see ``mstar.utils.h2d``.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable

import torch

# The compute stream: whatever the device's default stream is.
DEFAULT = "default"
# Pre-planning kernels and memcpys for the next step (CudaGraphRunner).
PLAN = "plan"
# D2H of the sampled rows ``check_stop`` reads (Worker).
CHECK_STOP = "check_stop"
# Tensor transport: outgoing D2H and incoming H2D (SHM / arena managers).
SEND = "send"
RECV = "recv"
# Background READs of the Mooncake transfer engine.
TRANSFER_READ = "transfer_read"
# KV-cache offload / reload copies (CpuPagePool).
KV_OFFLOAD = "kv_offload"
# The second branch of a ``Fork`` during graph capture. Only live while a
# capture runs -- at replay the branch is graph nodes, not stream work.
FORK = "fork"

_AUX = "aux"
_OFFLOAD = "offload"


def _parse_layout(spec: str) -> Callable[[str], str]:
    spec = spec.strip()
    if spec in ("", "aux", "shared"):
        return lambda role: _OFFLOAD if role == KV_OFFLOAD else _AUX
    if spec == "split":
        return lambda role: role
    mapping = {}
    for item in spec.split(","):
        role, sep, phys = item.partition("=")
        if not sep or not role.strip() or not phys.strip():
            raise ValueError(
                f"MSTAR_STREAMS entry {item!r} is not role=stream; expected "
                "'split', 'shared' or e.g. 'plan=aux,send=aux'")
        mapping[role.strip()] = phys.strip()
    if mapping.get(DEFAULT, DEFAULT) != DEFAULT:
        raise ValueError("MSTAR_STREAMS cannot move the default role off the "
                         "default stream")
    return lambda role: mapping.get(role, role)


class StreamManager:
    """The process's named streams. Thread-safe; use ``get_stream``."""

    def __init__(self, layout: str | None = None):
        spec = os.environ.get("MSTAR_STREAMS", "aux") if layout is None else layout
        self._physical_of = _parse_layout(spec)
        self._streams: dict[tuple[int, str], torch.cuda.Stream] = {}
        self._lock = threading.Lock()

    def physical_name(self, role: str) -> str:
        return DEFAULT if role == DEFAULT else self._physical_of(role)

    def get(self, role: str, device=None) -> torch.cuda.Stream | None:
        """The stream for ``role`` on ``device`` (default: the current CUDA
        device), or None when CUDA is unavailable or ``device`` is the CPU."""
        if not torch.cuda.is_available():
            return None
        if device is not None and torch.device(device).type != "cuda":
            return None
        dev = torch.device("cuda", torch.cuda.current_device()) \
            if device is None or torch.device(device).index is None \
            else torch.device(device)
        phys = self.physical_name(role)
        if phys == DEFAULT:
            return torch.cuda.default_stream(dev)
        key = (dev.index, phys)
        stream = self._streams.get(key)
        if stream is None:
            with self._lock:
                stream = self._streams.get(key)
                if stream is None:
                    stream = self._streams[key] = torch.cuda.Stream(device=dev)
        return stream

    def num_streams(self) -> int:
        """Physical side streams created so far (the default one excluded)."""
        return len(self._streams)


_MANAGER: StreamManager | None = None
_MANAGER_LOCK = threading.Lock()


def stream_manager() -> StreamManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = StreamManager()
    return _MANAGER


def get_stream(role: str, device=None) -> torch.cuda.Stream | None:
    """Shorthand for ``stream_manager().get(role, device)``."""
    return stream_manager().get(role, device)


class Fork:
    """Two independent branches of a captured step on two streams.

    Where a layer has two branches that do not depend on each other, running
    them on two streams lets their kernels overlap. ``run`` does that only
    while a CUDA graph is being captured, where the fork and join become graph
    dependencies and the graph's pool keeps both branches' tensors alive;
    eagerly it calls the two functions in order, so results are identical.
    ``MSTAR_AUX_STREAM=0`` keeps every capture on one stream. The branch
    stream is the shared ``FORK`` role, not one per ``Fork``.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self._enabled = (os.environ.get("MSTAR_AUX_STREAM", "1") == "1"
                         if enabled is None else enabled)
        self._fork: torch.cuda.Event | None = None
        self._join: torch.cuda.Event | None = None

    def run(self, fn0: Callable[[], Any], fn1: Callable[[], Any]) -> tuple[Any, Any]:
        """``fn0`` on the current stream and ``fn1`` on the fork stream,
        joined before returning, while a capture is in progress; otherwise
        both in order on the current stream."""
        if not (self._enabled and torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()):
            return fn0(), fn1()
        if self._fork is None:
            self._fork = torch.cuda.Event()
            self._join = torch.cuda.Event()
        side = get_stream(FORK)
        self._fork.record()
        r0 = fn0()
        with torch.cuda.stream(side):
            self._fork.wait()
            r1 = fn1()
            self._join.record()
        self._join.wait()
        return r0, r1


def reset_device_scheduling(device) -> None:
    """Create and destroy a throwaway CUDA context on ``device``.

    Kernels (or device-to-device copies) from a side stream running beside a
    CUDA-graph replay can leave the GPU in a mode CUDA graph nodes have delayed
    launch, until the device's scheduling setup is rebuilt, e.g., another
    context arriving on the GPU.
    """
    import ctypes

    if not torch.cuda.is_available():
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(index)
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cu.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_int()
    ctx = ctypes.c_void_p()
    if cu.cuDeviceGet(ctypes.byref(handle), index) == 0 and \
            cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, handle.value) == 0:
        cu.cuCtxDestroy_v2(ctx)
    # back onto torch's primary context for this thread
    torch.cuda.set_device(index)
