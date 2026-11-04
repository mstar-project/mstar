"""The simulated device's clock, and the tape a captured graph replays.

Every stream of the simulated GPU has a timeline on the host's
``time.perf_counter`` clock. A kernel launched at host time ``t`` starts at
``max(t, stream.free_at)`` and ends ``cost`` seconds later, so the host runs
ahead of the device as it does on a real GPU. An event takes its stream's
``free_at`` when recorded; waiting on it (``synchronize``, ``query``, a
device-to-host copy) sleeps the host until the device would have got there.

While a thread captures a graph, the kernels it launches are appended to that
graph's :class:`Tape` instead of being charged; each replay charges them as
one launch. :func:`kernel_fn` makes a stand-in (FlashInfer, Triton, mstar's
own fused kernels) one kernel of its own cost.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

_tls = threading.local()
# MSTAR_SIM_FAST=1 (``--fast``): host control flow only. No kernel is priced or
# queued, so a host read never waits; the clock is the real one, so mstar's own
# timeouts still fire. Graph capture, host-read refusal under capture and value
# semantics are unchanged.
FAST = os.environ.get("MSTAR_SIM_FAST") == "1"


@dataclass
class Timeline:
    """One stream's queue on the simulated device."""

    name: str
    free_at: float = 0.0
    kernels: int = 0
    busy: float = 0.0

    def launch(self, seconds: float, launched: float | None = None) -> float:
        start = max(vnow() if launched is None else launched, self.free_at)
        self.free_at = start + seconds
        self.kernels += 1
        self.busy += seconds
        return self.free_at


_LOCK = threading.Lock()
DEFAULT = Timeline("default")
STATS = {"host_wait": 0.0, "host_compute": 0.0, "replays": 0}


# --- host time spent in the emulator.
# A real GPU launch returns at once; here the launching thread also runs the
# emulator (shape inference, pricing). That time is the thread's "debt" since
# it last synchronised with the device: launches and event records are stamped
# `vnow() = now - debt`, as if the emulator cost nothing, and a host wait
# sleeps only for device time still left, then clears the debt.


def vnow() -> float:
    """The host's time net of the emulator: frozen while the emulator runs, so a
    launch is stamped when the call that made it began."""
    if getattr(_tls, "computing", 0):
        return _tls.since - getattr(_tls, "debt", 0.0)
    return time.perf_counter() - getattr(_tls, "debt", 0.0)


def compute_begin() -> int:
    """``computing()`` without a context manager, for the per-op hot path:
    returns a token for :func:`compute_end`."""
    depth = getattr(_tls, "computing", 0)
    _tls.computing = depth + 1
    if depth == 0:
        _tls.since = time.perf_counter()
    return depth


def compute_end(depth: int) -> None:
    _tls.computing = depth
    if depth == 0:
        dt = time.perf_counter() - _tls.since
        _tls.debt = getattr(_tls, "debt", 0.0) + dt
        STATS["host_compute"] += dt


@contextmanager
def computing():
    """Mark host time spent in the emulator (outermost only)."""
    depth = compute_begin()
    try:
        yield
    finally:
        compute_end(depth)


def wait_until(t: float) -> None:
    """Sleep the host until device time ``t``, then settle the thread's debt."""
    delay = t - time.perf_counter()
    if delay > 0:
        STATS["host_wait"] += delay
        time.sleep(delay)
    _tls.debt = 0.0
    if getattr(_tls, "computing", 0):
        _tls.since = time.perf_counter()  # the wait is the device's time, not the emulator's


def current_timeline() -> Timeline:
    return getattr(_tls, "timeline", None) or DEFAULT


def set_current_timeline(timeline: Timeline | None) -> Timeline | None:
    previous = getattr(_tls, "timeline", None)
    _tls.timeline = timeline
    return previous


def in_kernel() -> bool:
    return getattr(_tls, "in_kernel", 0) > 0


def charge(seconds: float, launched: float | None = None) -> float:
    """Queue ``seconds`` of device work, launched at host time ``launched`` (now
    by default), on the current stream; returns its end. While capturing, the
    work goes on the tape instead."""
    if FAST:
        return time.perf_counter()
    tape = capturing()
    if tape is not None:
        tape.entries.append(seconds)
        return time.perf_counter()
    with _LOCK:
        return current_timeline().launch(seconds, launched)


# --------------------------------------------------------------------- capture


@dataclass
class Tape:
    """What a captured graph launches, in order: each kernel's cost in seconds,
    or a callable pricing it at replay time (a kernel whose cost depends on
    the values in its arguments, such as attention over the latest plan)."""

    entries: list[float | Callable[[], float]] = field(default_factory=list)


def capturing() -> Tape | None:
    return getattr(_tls, "tape", None)


def set_capturing(tape: Tape | None) -> Tape | None:
    previous = getattr(_tls, "tape", None)
    _tls.tape = tape
    return previous


def _flag_values(out, written):
    """A stand-in's value-holding outputs (small ints: a sampled token) hold no
    kernel's values; nor do value-holding tensors it was handed to write."""
    from mstar_sim import device

    device.KERNELS_RUN[0] += 1
    if any(isinstance(t, device.SimTensor) and t._data.device.type != "meta"
           for t in device.tree_flatten((out, written))[0]):
        device.uncomputed()
    return out


def kernel_fn(name: str, cost: Callable[..., tuple] | None = None, dtype=None, static: bool = False,
              writes_args: bool = False):
    """Make a stand-in one kernel of the simulated GPU.

    The decorated function computes nothing: it returns what the kernel would
    (its ``out`` arguments, or empty tensors of the right shape), and the
    tensor ops it uses for that are not charged. ``cost`` (same arguments)
    returns ``(flops, bytes)``, or ``(flops, bytes, features)`` with the
    geometry a measured table needs (:func:`mstar_sim.cost.kernel_seconds`).
    ``static``: the cost depends only on argument shapes, so a captured graph
    prices it once rather than on every replay. ``writes_args``: it writes its
    arguments in place, with no telling which (a Triton launch)."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from mstar_sim import cost as cost_model

            if in_kernel():
                return fn(*args, **kwargs)
            if FAST:
                return _flag_values(fn(*args, **kwargs), args if writes_args else ())

            def seconds() -> float:
                flops, nbytes, *features = cost(*args, **kwargs) if cost is not None else (0.0, 0.0)
                return cost_model.kernel_seconds(name, flops=flops, bytes=nbytes, dtype=dtype,
                                                 features=features[0] if features else None)

            launched = vnow()
            _tls.in_kernel = getattr(_tls, "in_kernel", 0) + 1
            try:
                with computing():
                    out = _flag_values(fn(*args, **kwargs), args if writes_args else ())
            finally:
                _tls.in_kernel -= 1
            tape = capturing()
            with computing():
                if tape is None:
                    charge(seconds(), launched)
                else:
                    tape.entries.append(graph_node_seconds(seconds) if static else seconds)
            return out

        wrapper.__mstar_sim_kernel__ = name
        return wrapper

    return decorate


def graph_node_seconds(seconds: Callable[[], float]) -> float:
    """``seconds()`` priced as a graph node (no eager launch gap)."""
    previous = set_capturing(Tape())
    try:
        return seconds()
    finally:
        set_capturing(previous)
