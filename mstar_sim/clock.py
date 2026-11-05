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

import contextlib
import functools
import inspect
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
# MSTAR_SIM_TIME_SCALE=K (``--time-scale K``): every kernel takes K times its predicted
# time, so a slower GPU (or a host that outruns it) can be staged for timing-shaped bugs
TIME_SCALE = float(os.environ.get("MSTAR_SIM_TIME_SCALE", "1"))


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
STATS = {"host_wait": 0.0, "host_compute": 0.0, "replays": 0, "calls": 0}


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
    did_more()
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


def charge(seconds: float, launched: float | None = None, kernels: int = 1) -> float:
    """Queue ``kernels`` kernels whose bodies take ``seconds`` of device time in all, launched
    at host time ``launched`` (now by default), on the current stream: each adds the GPU's
    eager launch gap. Returns their end. While capturing, they go on the tape instead (a
    replay adds its own gaps)."""
    recording = getattr(_tls, "recording", None)
    if recording is not None:
        recording[0] += seconds
        recording[1] += kernels
    if FAST:
        return time.perf_counter()
    tape = capturing()
    if tape is not None:  # unscaled: its replay is charged here
        tape.seconds += seconds
        tape.kernels += kernels
        return time.perf_counter()
    from mstar_sim import cost

    with _LOCK:
        return current_timeline().launch((seconds + kernels * cost.launch_seconds()) * TIME_SCALE, launched)


@contextmanager
def recording():
    """What this thread charges inside: ``[seconds, kernels, more]``, where ``more`` says it
    did more than queue kernels of fixed cost (:func:`did_more`), so charging the same
    again can't stand for a repeat of it. Nested, the outer one sees the inner's too."""
    outer = getattr(_tls, "recording", None)
    mine = _tls.recording = [0.0, 0, False]
    try:
        yield mine
    finally:
        _tls.recording = outer
        if outer is not None:
            outer[0] += mine[0]
            outer[1] += mine[1]
            outer[2] = outer[2] or mine[2]


def did_more() -> None:
    """The call being recorded did more than queue kernels: it left work for a graph's
    replays (an effect, a kernel priced at replay), waited for the device, or checked
    for a capture."""
    recording = getattr(_tls, "recording", None)
    if recording is not None:
        recording[2] = True


# --------------------------------------------------------------------- capture


@dataclass
class Tape:
    """What a captured graph launches. ``seconds`` and ``kernels``: the bodies and count of
    its kernels whose cost is fixed at capture, summed as they are captured. Each node is
    ``(cost, effect)``: a callable pricing a kernel at replay time (one whose cost depends
    on the values in its arguments, such as attention over the latest plan), or None; and
    the work it does on values the emulator keeps (small tensors, host copies to and from
    them), re-run at each replay, or None. A replay runs the nodes in order, so a cost reads
    the values the nodes before it left. ``collectives``: what it launches on other ranks'
    groups, in order, as ``(group, descriptor)``; at the capture's end, what each replay
    meets on instead (:func:`mstar_sim.dist.seal`)."""

    seconds: float = 0.0
    kernels: int = 0
    nodes: list[tuple[Callable[[], float] | None, Callable[[], None] | None]] = field(default_factory=list)
    collectives: list[tuple[object, list[int]]] = field(default_factory=list)


def on_replay(effect: Callable[[], None]) -> None:
    """While capturing, re-run ``effect`` at each replay of the graph."""
    tape = capturing()
    if tape is not None:
        did_more()
        tape.nodes.append((None, effect))


def capturing() -> Tape | None:
    return getattr(_tls, "tape", None)


def set_capturing(tape: Tape | None) -> Tape | None:
    previous = getattr(_tls, "tape", None)
    _tls.tape = tape
    return previous


def _flag_values(out, written):
    """No kernel computed a stand-in's kept outputs, or the kept tensors it was handed to
    write (at each replay, if captured). Its outputs are the kernel's allocations (what it
    used to make them is not)."""
    from mstar_sim import device

    device.KERNELS_RUN[0] += 1
    device.count_storages(out)
    kept = [t._data for t in device.tree_flatten((out, written))[0]
            if isinstance(t, device.SimTensor) and t._data.device.type != "meta"]

    def flag():
        for t in kept:
            device.set_never_computed(t)

    if capturing() is None:
        flag()
    elif kept:
        on_replay(flag)
    return out


def kernel_fn(name: str, cost: Callable[..., tuple | Callable[[], tuple]] | None = None, dtype=None,
              writes_args: bool | tuple[str, ...] = False):
    """Make a stand-in one kernel of the simulated GPU.

    The decorated function computes nothing: it returns what the kernel would
    (its ``out`` arguments, or empty tensors of the right shape), and the
    tensor ops it uses for that are not charged, nor count as allocations.
    Captured, those ops on kept values run at each replay, like any others. ``cost`` (same arguments)
    returns ``(flops, bytes)``, or ``(flops, bytes, features)`` with the
    geometry a measured table needs (:func:`mstar_sim.cost.kernel_seconds`),
    priced once; or a callable returning that, for a cost that depends on the
    values its kernel reads (a plan's lengths), priced at every replay of a
    graph that captured it. The callable is all a graph keeps of the call, so
    it should hold what it reads, not the kernel's tensors. ``writes_args``: the
    arguments it writes in place, by name, or True for any of them, with no telling
    which (a Triton launch)."""

    def decorate(fn):
        sig = inspect.signature(fn)

        def written(args, kwargs) -> tuple:
            if isinstance(writes_args, bool):
                return args if writes_args else ()
            bound = sig.bind(*args, **kwargs).arguments
            return tuple(bound.get(name) for name in writes_args)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from mstar_sim import cost as cost_model

            if in_kernel():
                return fn(*args, **kwargs)

            def price(terms) -> float:
                flops, nbytes, *features = terms
                return cost_model.kernel_seconds(name, flops=flops, bytes=nbytes, dtype=dtype,
                                                 features=features[0] if features else None)

            launched = vnow()
            _tls.in_kernel = getattr(_tls, "in_kernel", 0) + 1
            with contextlib.nullcontext() if FAST else computing():
                try:
                    out = fn(*args, **kwargs)
                finally:
                    _tls.in_kernel -= 1
                out = _flag_values(out, written(args, kwargs))
            if FAST:
                return out
            tape = capturing()
            with computing():
                terms = cost(*args, **kwargs) if cost is not None else (0.0, 0.0)
                if tape is not None and callable(terms):
                    did_more()
                    tape.nodes.append((lambda: price(terms()), None))
                else:
                    charge(price(terms() if callable(terms) else terms), launched)
            return out

        wrapper.__mstar_sim_kernel__ = name
        return wrapper

    return decorate
