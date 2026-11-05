"""The simulated device's clock, and the tape a captured graph replays.

Every stream of the simulated GPU has a timeline on the process's clock
(:func:`now`), which is the host's time less the emulator's own. A kernel launched
at host time ``t`` starts at ``max(t, stream.free_at)`` and ends ``cost`` seconds
later, so the host runs ahead of the device as it does on a real GPU. An event
takes its stream's ``free_at`` when recorded; waiting on it (``synchronize``,
``query``, a device-to-host copy) holds the host until the device would have got
there, and a process with nothing else to run jumps there.

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
        start = max(now() if launched is None else launched, self.free_at)
        self.free_at = start + seconds
        self.kernels += 1
        self.busy += seconds
        return self.free_at


_LOCK = threading.Lock()
DEFAULT = Timeline("default")
STATS = {"replays": 0, "calls": 0, "emulator_s": 0.0, "run_delay_s": 0.0, "blocked_s": 0.0, "jumped_s": 0.0,
         "device_waited_s": 0.0, "thread_time_faster": 0}


# --- the process's clock, which every time source in the process reads (mstar_sim.waits patches
# them): ``now() = real - S - B + J``.
# - S, the emulator's time. A region (an intercepted call) with no other counted thread running
#   holds the clock still: all its wall time is the emulator's, whatever took it. One overlapping
#   another running thread costs that thread nothing: S grows by its CPU time, live (so no read
#   goes lower than one before it), and by the run delay the box descheduled it for.
# - B, the real time every thread of the process spent blocked in a known wait.
# - J, the moves: where every thread is blocked and one waits for the device, the clock jumps to
#   the device's time instead of sleeping for it (mstar_sim.waits).
# A thread's own ``thread_time`` is its CPU time less its emulator CPU time, plus the device time
# it waited (a CUDA sync spins). In ``--fast`` the clock is the real one and none of this runs.

_real = time.perf_counter  # the box's clock, whatever is patched later
_real_thread_time = time.thread_time
_real_process_time = time.process_time
# reentrant: a free (``mstar_sim.memory``) runs a region from a finalizer, a queue's put from GC
LOCK = threading.RLock()
# (S closed + B - J, the real time all threads blocked or None): swapped whole, under LOCK
_base: tuple[float, float | None] = (0.0, None)
RUNNING = [0]  # counted threads not in a known wait (mstar_sim.waits)
# each thread inside a region: its CPU clock, its CPU time at entry, and the real time at entry
# if it was the only thread running then, else None
_OPEN: dict[int, tuple[int, float, float | None]] = {}
_last = [0.0]
# a region whose wall time passed its CPU time by more than this was likely descheduled
_DESCHEDULED = 20e-6
_MAX_LOST = 0.05


def now() -> float:
    """The process's clock (see above); the real one under ``--fast``. Never goes back."""
    if FAST:
        return _real()
    offset, frozen = _base
    real = _real()
    t = (real if frozen is None else frozen) - offset
    if _OPEN:
        for clk, cpu, alone in list(_OPEN.values()):
            t -= time.clock_gettime(clk) - cpu if alone is None else (real if frozen is None else frozen) - alone
    if t < _last[0]:
        return _last[0]
    _last[0] = t
    return t


def _open() -> None:
    clk = getattr(_tls, "clk", None)
    if clk is None:
        clk = _tls.clk = time.pthread_getcpuclockid(threading.get_ident())
    real = _tls.wall = _real()
    _OPEN[threading.get_ident()] = (clk, _real_thread_time(), real if RUNNING[0] <= 1 else None)


def _close() -> None:
    global _base
    ident = threading.get_ident()
    _, entered, alone = _OPEN[ident]
    wall = _real() - _tls.wall
    delay = 0.0
    if alone is None or RUNNING[0] > 1:
        lost = wall - (_real_thread_time() - entered) + getattr(_tls, "lost", 0.0)
        delay = _run_delay(lost) if lost > _DESCHEDULED else 0.0
    with LOCK:  # S grows before the open entry goes: a reader between sees less, not more
        cpu = _real_thread_time() - entered  # last: this bookkeeping is the emulator's too
        if alone is not None and RUNNING[0] <= 1:
            delay = _real() - alone - cpu
        _base = (_base[0] + cpu + delay, _base[1])
        del _OPEN[ident]
        STATS["emulator_s"] += cpu
        STATS["run_delay_s"] += delay
    _tls.emulated = getattr(_tls, "emulated", 0.0) + cpu


def compute_begin() -> int:
    """``computing()`` without a context manager, for the per-op hot path:
    returns a token for :func:`compute_end`."""
    depth = getattr(_tls, "computing", 0)
    _tls.computing = depth + 1
    if depth == 0 and not FAST:
        _open()
    return depth


def compute_end(depth: int) -> None:
    if depth == 0 and not FAST:
        _close()  # still inside: a free in here would otherwise open a region over this one
    _tls.computing = depth


def pause() -> None:
    """This thread blocks: a region it is in stops here (a wait isn't the emulator's time)."""
    if getattr(_tls, "computing", 0) and not FAST:
        _close()


def resume() -> None:
    if getattr(_tls, "computing", 0) and not FAST:
        _open()


def _run_delay(lost: float) -> float:
    """The time this thread waited for a CPU since it last asked (``run_delay`` in
    ``/proc/thread-self/schedstat``), up to ``lost``: waiting for the GIL is not in it. The
    kernel books a wait when the thread next runs, so what a region lost may show only at a
    later one: the rest of ``lost`` carries over, up to ``_MAX_LOST``."""
    try:
        fd = os.open("/proc/thread-self/schedstat", os.O_RDONLY)
        try:
            delay = int(os.read(fd, 64).split()[1]) / 1e9
        finally:
            os.close(fd)
    except (OSError, IndexError, ValueError):
        return 0.0
    last = getattr(_tls, "run_delay", delay)
    _tls.run_delay = delay
    claimed = max(0.0, min(lost, delay - last))
    _tls.lost = min(lost - claimed, _MAX_LOST)
    return claimed


@contextmanager
def computing():
    """Mark host time spent in the emulator (outermost only)."""
    depth = compute_begin()
    try:
        yield
    finally:
        compute_end(depth)


def freeze() -> None:
    """Every thread is blocked: the clock stands still (B accrues). Under LOCK."""
    global _base
    if _base[1] is None:
        _base = (_base[0], _real())


def thaw() -> None:
    """A thread runs again: the clock runs on from where it stood. Under LOCK."""
    global _base
    offset, frozen = _base
    if frozen is not None:
        blocked = _real() - frozen
        _base = (offset + blocked, None)
        STATS["blocked_s"] += blocked


def jump(to: float) -> None:
    """Move the clock forward to ``to`` (no further: it never goes back). Under LOCK."""
    global _base
    ahead = to - now()
    if ahead > 0:
        _base = (_base[0] - ahead, _base[1])
        STATS["jumped_s"] += ahead


def thread_time() -> float:
    """``time.thread_time``: this thread's CPU time less its emulator CPU time, plus the device
    time it waited (a CUDA sync spins). It counts ``thread_time_faster``, the times it ran
    ahead of :func:`now` (an op that released the GIL inside a region: the self-check)."""
    entry = _OPEN.get(threading.get_ident())
    cpu = _real_thread_time() if entry is None else entry[1]
    t = cpu - getattr(_tls, "emulated", 0.0) + getattr(_tls, "waited", 0.0)
    clock, last = now(), getattr(_tls, "checked", None)
    if last is not None and t - last[0] > clock - last[1] + 1e-4:
        STATS["thread_time_faster"] += 1
    _tls.checked = (t, clock)
    return t


def process_time() -> float:
    """``time.process_time``: the process's CPU time less the emulator's, plus the device time
    its threads waited."""
    open_cpu = sum(time.clock_gettime(entry[0]) - entry[1] for entry in list(_OPEN.values()))
    return _real_process_time() - STATS["emulator_s"] - open_cpu + STATS["device_waited_s"]


def wait_until(t: float) -> None:
    """Wait for the device to reach ``t`` (:func:`mstar_sim.waits.device_wait`)."""
    did_more()
    if FAST or now() >= t:
        return
    from mstar_sim import waits

    waits.device_wait(t)


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
        return now()
    tape = capturing()
    if tape is not None:  # unscaled: its replay is charged here
        tape.seconds += seconds
        tape.kernels += kernels
        return now()
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

            launched = now()
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
