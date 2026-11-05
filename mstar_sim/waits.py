"""The process's waits, so its clock jumps where a GPU run's host would only have waited.

Every thread the process starts is counted, running until it enters a wait this module
knows (the registry): a Condition's waiter lock (Event, Queue, Future, Semaphore, Barrier,
a thread's start), a ``queue.SimpleQueue`` get (an executor's idle thread), ``time.sleep``,
a selector's select (asyncio), mstar's Rust poll, a wait for the device, and a rank's wait
for its peers. A thread in any other wait (a plain lock, a join, a socket) counts as
running. While no counted thread runs, the clock stands still (:func:`mstar_sim.clock.freeze`).

Each wake point marks the thread it wakes running before the waker can block: a waiter
lock's release (a notify), a ``SimpleQueue`` put (by count, as C wakes any getter), an
``os.eventfd_write`` on the fd a poll waits on (mstar's ``EventWakeup``), and ``Thread.start``
for the thread it starts. So the process is never seen idle while a woken thread has yet to
run.

One thread of the emulator's own, the mover, ends waits on deadlines: a thread waiting for
the device is woken once the clock reaches the device's time, by a real-time timer while
another thread runs, or at once by a jump of the clock (:func:`mstar_sim.clock.jump`) when
every counted thread is blocked. A wait with a host deadline (a sleep, a timeout) counts as
blocked and holds on until the clock reaches its deadline.

The processes of a deployment agree through its table (:mod:`mstar_sim.table`), by Shadow's
two clauses. Blocked, a process follows the bound (how far the others let it go) up to its
next deadline: mstar's sleep loops poll what other processes write, so no host deadline goes
past it, though a device deadline needs no one. With nothing to bound it, a host deadline
becomes a move only once its timeout has passed in real time (it is then due: :func:`_until`),
so idle processes keep real time rather than spin ahead. Unknown channels (HTTP, a pipe) have
no counter, so a move on the bound waits for the deployment's state to stay still for a settle
window. Running, a thread that reads the clock more than W past the bound waits there
(:func:`read`), blocked, until the bound rises. A thread at a collective's meet stands still,
as NCCL's host doesn't wait: its process stands aside from the bound and takes no moves.

Threads named ``mstar_sim-*`` are the emulator's own and never counted. ``--fast`` installs
none of this.
"""

from __future__ import annotations

import _queue
import _thread
import math
import os
import queue
import selectors
import sys
import threading
import time
from contextlib import contextmanager

from mstar_sim import clock, table

LOCK = clock.LOCK
DEVICE, HOST, MEET, AHEAD = "device", "host", "meet", "ahead"
W = 0.5e-3  # how far past the bound a running process may read the clock
SETTLE = 200e-6  # how long the deployment's state stays still before a move on the bound
_MAX_SLEEP = 0.05  # the mover looks at least this often
_real_lock = _thread.allocate_lock
_real_sleep = time.sleep
_real_eventfd_write = os.eventfd_write
# a device wait woken this much after its time by the mover's timer counts as a late resume
_LATE = 1e-3
# how often a due wait the mover can't end (a select, a poll) looks at the clock
_SLICE = 1e-3
OWN = "mstar_sim-"  # the emulator's own threads' names start with this

_counted: set[int] = set()  # threads the registry counts
_running = clock.RUNNING
_waits: set[Wait] = set()
_fds: dict[int, Wait] = {}  # an fd a poll waits on: its eventfd_write wakes the poll
_installed = [False]
TABLE: table.Table
_ahead = [math.inf]  # past this a clock read looks at the bound
_quiet = [0, 0.0]  # the deployment's state, and the real time it last changed
_wakefd = [-1]  # the mover's wake for a select or a poll, which no lock can end
_poked: set[Wait] = set()
REPORT = {"slept_s": 0.0, "late_resumes": 0, "jump_blockers": {}, "ahead_stalls": 0, "settle_overruns": 0,
          "meet_skew_s": 0.0}


class Wait:
    """A blocked thread's entry: what it waits for (``kind``: the device, a host deadline, or
    None for nothing in particular), its deadline on the clock, and how to end it early."""

    __slots__ = ("kind", "deadline", "counts", "woken", "due", "timed_out", "release", "thread", "fd", "real")

    def __init__(self, kind, deadline, counts, release, fd):
        self.kind, self.deadline, self.counts, self.release, self.fd = kind, deadline, counts, release, fd
        self.woken = self.due = self.timed_out = False
        self.thread = threading.get_ident()
        self.real = clock._real()


def _block_one() -> None:
    _running[0] -= 1
    if _running[0] == 0:
        clock.freeze()
        _publish()


def _run_one() -> None:
    _running[0] += 1
    if _running[0] == 1:
        clock.thaw()
        _publish()


def _publish() -> None:
    """This process's state changed: its next event is the earliest deadline it holds."""
    pending = [w for w in _waits if not w.woken]
    TABLE.publish(clock.now(), _running[0] > 0, min((w.deadline for w in pending if w.deadline is not None),
                                                    default=math.inf), any(w.kind is MEET for w in pending))


def enter(kind, deadline=None, release=None, fd=None, counts=True) -> Wait | None:
    """This thread enters a wait (None when the registry isn't installed). ``release`` ends it
    early (the mover's delivery); ``fd``, an fd whose ``eventfd_write`` wakes it; ``counts``
    False for a wait whose thread is counted blocked elsewhere (a queue's getters)."""
    if not _installed[0]:
        return None
    w = Wait(kind, deadline, counts and threading.get_ident() in _counted, release, fd)
    clock.pause()
    with LOCK:
        _waits.add(w)
        if fd is not None:
            _fds[fd] = w
        if w.counts:
            _block_one()
        if kind is MEET:
            _publish()
        if kind is DEVICE or kind is MEET:
            kick()
    return w


def wake(w: Wait) -> bool:
    """Mark ``w``'s thread running (a waker, before it can block itself). False if something
    woke it already."""
    with LOCK:
        if w.woken:
            return False
        w.woken = True
        if w.counts:
            _run_one()
        return True


def leave(w: Wait | None) -> None:
    """This thread's wait ``w`` is over."""
    if w is None:
        return
    with LOCK:
        _waits.discard(w)
        if w.fd is not None and _fds.get(w.fd) is w:
            del _fds[w.fd]
        if w in _poked:
            _poked.discard(w)
            if not _poked:
                try:
                    os.eventfd_read(_wakefd[0])
                except BlockingIOError:
                    pass
        if not w.woken:
            w.woken = True
            if w.counts:
                _run_one()
        if w.kind is MEET:
            _publish()
    clock.resume()


def _until(w: Wait | None, timeout, attempt):
    """``attempt(seconds)``, a wait ending in real time (falsy if it timed out), for wait ``w``:
    retried until it ends or the clock reaches ``w``'s deadline (none: tried once). After its
    real timeout the deadline is due (the mover may then move the clock to it, bound or no
    bound). The mover ends it (``w.release``, ``w.timed_out``) when the clock gets there; a wait
    it can't end looks every ``_SLICE``."""
    result = attempt(timeout)
    while not result and w is not None and w.deadline is not None:
        with LOCK:
            t = clock.now()
            if w.timed_out or t >= w.deadline:
                break
            if not w.due:
                w.due = True
                kick()
        result = attempt(w.deadline - t if w.release is not None else min(w.deadline - t, _SLICE))
    return result


@contextmanager
def blocked(kind=None, deadline=None, fd=None, release=None):
    """A wait whose waker this module doesn't see (a gloo collective, a Rust poll): the thread
    counts as blocked until it returns."""
    w = enter(kind, deadline, release, fd)
    try:
        yield w
    finally:
        leave(w)


def kick() -> None:
    """Have the mover look again."""
    TABLE.kick()


def _poke() -> None:
    """End a select or a poll (through the eventfd they watch beside their own)."""
    _real_eventfd_write(_wakefd[0], 1)


# ------------------------------------------------------------------- the device


def device_wait(t: float) -> None:
    """Hold this thread until the device reaches ``t`` on the clock: the mover wakes it then,
    jumping the clock there if nothing else in the process runs."""
    began = clock.now()
    lock = _real_lock()
    lock.acquire()
    w = enter(DEVICE, t, release=lock.release)
    if w is None:
        while clock.now() < t:
            _real_sleep(t - clock.now())
    else:
        try:
            lock.acquire()
        finally:
            leave(w)
    waited = max(clock.now(), t) - began
    clock._tls.waited = getattr(clock._tls, "waited", 0.0) + waited
    clock.STATS["device_waited_s"] += waited


def _mover() -> None:
    while True:
        word = TABLE.word()  # before it looks: a change while it does ends the wait at once
        with LOCK:
            timeout = _move()
        TABLE.wait(word, timeout)


def _move() -> float:
    """Release the threads the bound has caught up with; with every counted thread blocked,
    move the clock to the process's next event; end the waits that are due. Returns how long
    in real time it may sleep, the deployment's changes aside. Under LOCK."""
    pending = [w for w in _waits if not w.woken]
    if _running[0]:
        TABLE.tick(clock.now())  # a thread busy in native code (an import) reads no clock
    bound = TABLE.bound()
    _ahead[0] = bound + W
    for w in pending:
        if w.kind is AHEAD and bound + W >= w.deadline and wake(w):
            w.release()
    if any(w.kind is MEET for w in pending):
        return _MAX_SLEEP  # a rank at a meet takes no moves, and its deadlines wait for the meet's end
    timeout = _MAX_SLEEP
    idle = _running[0] == 0
    if idle:
        device = min((w.deadline for w in pending if w.kind is DEVICE), default=math.inf)
        hosts = [w for w in pending if w.kind is HOST and w.deadline is not None]
        host = min((w.deadline for w in hosts), default=math.inf)
        target = device
        if device > host or device == math.inf:
            due = min((w.deadline for w in hosts if w.due), default=math.inf)
            target = min(host, bound) if bound < math.inf else due
            if target < math.inf and target > clock.now():
                waited = _settled()
                if waited < SETTLE:
                    target = math.inf
                    timeout = SETTLE - waited
        if target < math.inf and target > clock.now():
            clock.jump(target)
            TABLE.moved(target)
    t = clock.now()
    for w in pending:
        if w.kind is not DEVICE and w.kind is not HOST or w.deadline is None or w.woken:
            continue
        if w.deadline > t:
            if w.kind is DEVICE:
                timeout = min(timeout, w.deadline - t)
        elif wake(w):
            if w.kind is HOST:
                w.timed_out = True
            elif not idle:
                _slept(w, t)
            if w.release is _poke:
                _poked.add(w)
            if w.release is not None:
                w.release()
    return timeout


def _settled() -> float:
    """The real time the deployment's state has stood still."""
    epochs, real = TABLE.epochs(), clock._real()
    if epochs != _quiet[0]:
        if real - _quiet[1] < SETTLE:
            REPORT["settle_overruns"] += 1
        _quiet[:] = [epochs, real]
    return real - _quiet[1]


def read() -> float:
    """A clock read by mstar: publishes a running clock, and holds a thread that has got more
    than W past the bound, until the bound rises (clause 2). It is blocked meanwhile, so its
    process's clock stands still once every thread is, and the slowest process is never held."""
    t = clock.now()
    TABLE.tick(t)
    if t > _ahead[0] and threading.get_ident() in _counted and not getattr(clock._tls, "computing", 0):
        bound = TABLE.bound()
        _ahead[0] = bound + W
        if t > bound + W:
            lock = _real_lock()
            lock.acquire()
            w = enter(AHEAD, t, release=lock.release)
            REPORT["ahead_stalls"] += 1
            try:
                lock.acquire()
            finally:
                leave(w)
            t = clock.now()
    return t


def _slept(w: Wait, t: float) -> None:
    """A device wait ended by the timer, while other threads ran: what kept the clock from
    jumping (each running counted thread's call site), and how late it ended."""
    REPORT["slept_s"] += clock._real() - w.real
    if t - w.deadline > _LATE:
        REPORT["late_resumes"] += 1
    blocked = {x.thread for x in _waits if x.counts and not x.woken}
    names = {th.ident: th.name for th in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        if (ident in _counted and ident not in blocked and ident != w.thread
                and frame.f_code is not SimpleQueue.get.__code__):  # (a queue counts its own getters)
            where = f"{names.get(ident, ident)} {frame.f_code.co_filename}:{frame.f_lineno} ({frame.f_code.co_name})"
            REPORT["jump_blockers"][where] = REPORT["jump_blockers"].get(where, 0) + 1


# ------------------------------------------------------------------ wake points


class _WaiterLock:
    """``threading``'s lock, as ``Condition.wait`` makes one per waiter: blocking on it is a
    registered wait, and its release (the notify) marks the waiter running first."""

    __slots__ = ("_lock", "_wait")

    def __init__(self):
        self._lock = _real_lock()
        self._wait = None

    def acquire(self, blocking=True, timeout=-1):
        if self._lock.acquire(False):
            return True
        if not blocking:
            return False
        w = enter(HOST if timeout >= 0 else None, clock.now() + timeout if timeout >= 0 else None,
                  release=self._lock.release)
        if w is None:
            return self._lock.acquire(True, timeout)
        self._wait = w
        try:  # (a notify between the first try and here released it already)
            got = self._lock.acquire(False) or _until(w, timeout, lambda t: self._lock.acquire(True, t))
        finally:
            self._wait = None
            leave(w)
        return got and not w.timed_out

    def release(self):
        w = self._wait
        if w is not None:
            wake(w)
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def _at_fork_reinit(self):
        self._lock._at_fork_reinit()
        self._wait = None

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()


class SimpleQueue(_queue.SimpleQueue):
    """``queue.SimpleQueue`` (an executor's work queue) whose blocked getters the registry
    counts: as C wakes any one getter, a queue counts max(0, getters - items) blocked, and a
    put changes that before it can wake anyone. ``put`` stays reentrant (a GC callback puts)."""

    def __init__(self):
        self._getters = self._items = self._blocked = 0

    def _recount(self) -> None:  # under LOCK
        blocked = max(0, self._getters - self._items)
        for _ in range(self._blocked - blocked):
            _run_one()
        for _ in range(blocked - self._blocked):
            _block_one()
        self._blocked = blocked

    def put(self, item, block=True, timeout=None):
        with LOCK:
            self._items += 1
            self._recount()
            super().put(item)

    def put_nowait(self, item):
        self.put(item)

    def get(self, block=True, timeout=None):
        counts = block and _installed[0] and threading.get_ident() in _counted
        if counts:
            with LOCK:
                self._getters += 1
                self._recount()
        try:
            if counts and timeout is not None:
                w = enter(HOST, clock.now() + timeout, counts=False)  # (counted blocked as a getter)
                try:
                    got = _until(w, timeout, self._get_within)
                finally:
                    leave(w)
                if not got:
                    raise queue.Empty
                item = got[0]
            else:
                item = super().get(block, timeout)
        except BaseException:
            if counts:
                with LOCK:
                    self._getters -= 1
                    self._recount()
            raise
        with LOCK:
            self._items -= 1
            if counts:
                self._getters -= 1
            self._recount()
        return item

    def get_nowait(self):
        return self.get(False)

    def _get_within(self, seconds):
        try:
            return (super().get(True, seconds),)
        except queue.Empty:
            return None


def sleep(seconds) -> None:
    """``time.sleep``: a host deadline (:func:`_until`)."""
    if seconds <= 0:
        return _real_sleep(seconds)
    lock = _real_lock()
    lock.acquire()
    w = enter(HOST, clock.now() + seconds, release=lock.release)
    try:
        _until(w, seconds, lambda t: lock.acquire(True, t)) if w is not None else _real_sleep(seconds)
    finally:
        leave(w)


def eventfd_write(fd, value) -> None:
    """``os.eventfd_write``: on an fd a poll waits on, the poller runs from here on."""
    w = _fds.get(fd)
    if w is not None:
        wake(w)
    _real_eventfd_write(fd, value)


def _selecting(select):
    def wrapped(self, timeout=None):
        if not _installed[0] or (timeout is not None and timeout <= 0):
            return select(self, timeout)
        if not getattr(self, "_mstar_sim_watches", False):
            # on the poll object, out of the selector's map: its owner loops on what is in that,
            # and a ready fd the map doesn't hold comes back as no event at all
            self._mstar_sim_watches = True
            if isinstance(self, selectors.SelectSelector):
                self._readers.add(_wakefd[0])
            else:
                self._selector.register(_wakefd[0], self._EVENT_READ)
        with blocked(HOST if timeout is not None else None, None if timeout is None else clock.now() + timeout,
                     release=_poke) as w:
            ready = _until(w, timeout, lambda seconds: select(self, seconds))
            _caught_up(w, ready)
            return ready

    return wrapped


def _caught_up(w: Wait | None, ready) -> None:
    """A wait ended by a channel the emulator can't see may beat the mover's settle window: the
    sender's clock may be past this one's, so it moves up to the others' latest clock (no further
    than the bound or its deadline) before it runs, and what it was sent doesn't land in its past."""
    if ready and w is not None:
        with LOCK:
            if _running[0] == 0:
                bound = TABLE.bound()
                clock.jump(min(bound, TABLE.latest(), math.inf if w.deadline is None else w.deadline))


def polling(poll, fd, timeout: float):
    """A poll for another process (mstar's Rust poll): ``poll(seconds)`` returns falsy at its
    timeout. ``fd``, the eventfd it also wakes on, if any (:func:`eventfd_write`)."""
    with blocked(HOST, clock.now() + timeout, fd=fd, release=_poke) as w:
        ready = _until(w, timeout, poll)
        _caught_up(w, ready)
        return ready


def poked() -> bool:
    """Whether a poll's wake was the mover's, not its owner's."""
    return bool(_poked)


def wakefd() -> int:
    return _wakefd[0]


# ---------------------------------------------------------------- installation


def install() -> None:
    """Patch every time source and wait of the process onto the clock, and start the mover.
    Before mstar (or asyncio, concurrent.futures) is imported."""
    if _installed[0] or clock.FAST:
        return
    global TABLE
    _installed[0] = True
    _counted.add(threading.get_ident())
    _running[0] = 1
    # a helper of the interpreter's that sits in a read of its own would hold the deployment back
    helper = "multiprocessing.resource_tracker" in " ".join(sys.orig_argv)
    TABLE = table.Table(None if helper else table.path_for(os.environ.get("MSTAR_SIM_NODE")))
    _wakefd[0] = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    epoch = time.time() - clock._real()  # the same in every process: wall time is one time
    start = TABLE.start_at()
    if start is not None:
        clock.start_at(start)
    TABLE.publish(clock.now(), True)
    for name, fn in {
        "perf_counter": read, "monotonic": read, "time": lambda: epoch + read(),
        "thread_time": clock.thread_time, "process_time": clock.process_time, "sleep": sleep,
    }.items():
        setattr(time, name, fn)
        if name != "sleep":
            setattr(time, name + "_ns", lambda fn=fn: int(fn() * 1e9))
    threading._time = read
    queue.time = read  # bound at import: ``from time import monotonic as time``
    threading._allocate_lock = _WaiterLock
    queue.SimpleQueue = SimpleQueue
    os.eventfd_write = eventfd_write
    for cls in (selectors.EpollSelector, selectors.PollSelector, selectors.SelectSelector):
        cls.select = _selecting(cls.select)

    start, bootstrap = threading.Thread.start, threading.Thread._bootstrap_inner

    def start_counted(self):
        if self.name.startswith(OWN):
            return start(self)
        with LOCK:  # running before it exists, so its starter's wait for it can't be jumped
            _run_one()
        self._mstar_sim_counted = True
        try:
            start(self)
        except BaseException:
            with LOCK:
                _block_one()
            raise

    def bootstrap_counted(self):
        counted = getattr(self, "_mstar_sim_counted", False)
        if counted:
            _counted.add(threading.get_ident())
        try:
            bootstrap(self)
        finally:
            if counted:
                with LOCK:
                    _counted.discard(threading.get_ident())
                    _block_one()

    threading.Thread.start = start_counted
    threading.Thread._bootstrap_inner = bootstrap_counted
    os.register_at_fork(after_in_child=_after_fork)
    threading.Thread(target=_mover, name=OWN + "mover", daemon=True).start()


def _after_fork() -> None:
    """A forked child has only the forking thread, running, and a slot of its own to find."""
    global TABLE
    LOCK._at_fork_reinit()
    TABLE = table.Table(table.path_for(os.environ.get("MSTAR_SIM_NODE")))
    _counted.clear()
    _counted.add(threading.get_ident())
    _waits.clear()
    _fds.clear()
    _running[0] = 1
    for ident in list(clock._OPEN):
        if ident != threading.get_ident():
            del clock._OPEN[ident]
    clock.thaw()
    TABLE.publish(clock.now(), True)
    threading.Thread(target=_mover, name=OWN + "mover", daemon=True).start()
