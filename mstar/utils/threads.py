"""Starting threads, and moving the threads a step passes through onto their CPUs.

A step passes between the worker's main thread, its GPU thread and its plan
thread several times (submit, launch, done, commit, plan ready), and each
handoff blocks one thread and wakes another. On a host with deep idle states a
thread woken on another core first waits for that core to leave C6 (130-175 µs
on the B300 host this was measured on); woken on a core that is already
running, it starts at once. The API server's result threads (the message
thread, and the Rust frontend's bridge) hand every step's tokens on the same
way. ``step_thread_core`` (``mstar.utils.cpus``) pins them.

A new thread starts with its creator's CPU mask, so a thread or pool that a
pinned thread starts would share its CPU. ``Thread`` and ``ThreadPool`` start
theirs on the process mask; ruff refuses a bare ``threading.Thread`` or
``ThreadPoolExecutor`` under ``mstar/`` (``banned-api`` in pyproject.toml).

A ``schedutil`` governor sets a CPU's clock from how busy it looks, so threads
that take turns stay near the lowest clock although together they fill a core.
``hold_clock_from_knob`` tells it what a process's threads need instead.
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import threading
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor

from mstar.utils import knobs
from mstar.utils.cpus import PINNABLE

logger = logging.getLogger(__name__)

# ``os`` has no ``sched_setattr``
_SCHED_SETATTR = {"x86_64": 314, "aarch64": 274}
_SCHED_KEEP_POLICY, _SCHED_KEEP_PARAMS, _SCHED_UTIL_CLAMP_MIN = 0x08, 0x10, 0x20
_UCLAMP_SYSTEM_CAP = "/proc/sys/kernel/sched_util_clamp_min"


class _SchedAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32), ("policy", ctypes.c_uint32), ("flags", ctypes.c_uint64),
        ("nice", ctypes.c_int32), ("priority", ctypes.c_uint32),
        ("runtime", ctypes.c_uint64), ("deadline", ctypes.c_uint64), ("period", ctypes.c_uint64),
        ("util_min", ctypes.c_uint32), ("util_max", ctypes.c_uint32),
    ]


def _set_util_min(value: int) -> str | None:
    number = _SCHED_SETATTR.get(platform.machine())
    if number is None:
        return f"no sched_setattr number for {platform.machine()}"
    attr = _SchedAttr(
        size=ctypes.sizeof(_SchedAttr), util_min=value,
        flags=_SCHED_KEEP_POLICY | _SCHED_KEEP_PARAMS | _SCHED_UTIL_CLAMP_MIN,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(number, 0, ctypes.byref(attr), 0) != 0:
        return f"sched_setattr failed: {os.strerror(ctypes.get_errno())}"
    return None


def hold_clock(floor: int) -> str | None:
    """Ask for this thread's CPU at no less than ``floor`` of 1024 of its clock
    whenever it runs (uclamp's ``util_min``), or say why not. Needs no
    privileges. Threads and processes this one starts afterwards inherit it. The
    hint does nothing under a governor that isn't reading utilization, and a
    cgroup's ``cpu.uclamp.max`` can cap it unseen."""
    try:
        with open(_UCLAMP_SYSTEM_CAP) as f:
            cap = int(f.read())
    except (OSError, ValueError) as e:
        return f"this kernel has no uclamp ({e})"
    if cap < floor:
        return f"{_UCLAMP_SYSTEM_CAP} caps it at {cap}"
    return _set_util_min(floor)


def hold_clock_from_knob(who: str) -> None:
    """``knobs: uclamp_min`` for this process: call it on its main thread, before
    it starts the threads that should have it."""
    floor = knobs.uclamp_min()
    if not floor:
        return
    refused = hold_clock(floor)
    if refused is None:
        logger.info("%s: uclamp_min %d applied", who, floor)
    else:
        logger.warning("%s: uclamp_min %d not applied: %s", who, floor, refused)


def release_clock() -> None:
    """Drop what this process inherited from the one that started it."""
    _set_util_min(0)


# ATen splits an op into chunks of at least this many elements per thread
_INTRA_OP_GRAIN = 32768

# the process mask, recorded by the first pin, for the threads started later
_process_mask: frozenset[int] | None = None


def pin_thread(cpus: Collection[int], process_mask: Collection[int]) -> None:
    """Confine the calling thread to ``cpus``.

    A thread's intra-op (OpenMP) pool is started by its first parallel op
    with the mask the thread has then, and a pool confined to one CPU runs a
    parallel op many times slower. So go back to the process mask first (the
    main thread, already confined, starts the GPU and plan threads), start the
    pool there with one op as wide as the pool, and only then move.
    """
    import torch

    global _process_mask
    _process_mask = frozenset(process_mask)
    os.sched_setaffinity(0, _process_mask)
    torch.empty(torch.get_num_threads() * _INTRA_OP_GRAIN, dtype=torch.uint8).fill_(0)
    os.sched_setaffinity(0, set(cpus))


def _to_process_mask() -> None:
    if PINNABLE and _process_mask is not None:
        os.sched_setaffinity(0, _process_mask)


class Thread(threading.Thread):
    """A ``threading.Thread`` that runs on the process mask, whatever its
    creator's and whatever its ``run`` is: a target, an override that doesn't
    call ``super().run()``, a mixin's. Wrapped at ``start`` so that calling
    ``run`` directly stays on the caller's thread and mask."""

    def start(self) -> None:
        run = self.run

        def on_process_mask() -> None:
            try:
                _to_process_mask()
                run()
            finally:
                # as Thread.run drops its target: no cycle through the bound method
                del self.run

        self.run = on_process_mask
        super().start()


def _on_process_mask(initializer: Callable[..., object] | None) -> Callable[..., None]:
    def init(*args) -> None:
        _to_process_mask()
        if initializer is not None:
            initializer(*args)
    return init


class ThreadPool(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose threads run on the process mask before
    ``initializer``, whatever the mask of the thread that started them."""

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
        initializer: Callable[..., object] | None = None,
        initargs: tuple = (),
    ):
        super().__init__(max_workers, thread_name_prefix, _on_process_mask(initializer), initargs)
