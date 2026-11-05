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
blocked and holds on until the clock reaches its deadline. Once its timeout has passed in real
time its deadline is due: with every counted thread blocked, the mover moves the clock to the
earliest deadline due, device or host (:func:`_until`). A host deadline so never takes the
clock past real time.

Threads named ``mstar_sim-*`` are the emulator's own and never counted. ``--fast`` installs
none of this.
"""

from __future__ import annotations

import _queue
import _thread
import os
import queue
import selectors
import sys
import threading
import time
from contextlib import contextmanager

from mstar_sim import clock

LOCK = clock.LOCK
DEVICE, HOST = "device", "host"
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
_kick = _real_lock()
_installed = [False]
REPORT = {"slept_s": 0.0, "late_resumes": 0, "jump_blockers": {}}


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
        kick()


def _run_one() -> None:
    if _running[0] == 0:
        clock.thaw()
    _running[0] += 1


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
        if kind is DEVICE:
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
        if not w.woken:
            w.woken = True
            if w.counts:
                _run_one()
    clock.resume()


def _until(w: Wait | None, timeout, attempt):
    """``attempt(seconds)``, a wait ending in real time (falsy if it timed out), for wait ``w``:
    retried until it ends or the clock reaches ``w``'s deadline (none: tried once). After its
    real timeout the deadline is due (the mover may move the clock to it), and the mover ends
    it (``w.release``, ``w.timed_out``) or, for a wait it can't end, it looks every ``_SLICE``."""
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
def blocked(kind=None, deadline=None, fd=None):
    """A wait whose waker this module doesn't see (a gloo collective, a Rust poll): the thread
    counts as blocked until it returns."""
    w = enter(kind, deadline, fd=fd)
    try:
        yield w
    finally:
        leave(w)


def kick() -> None:
    """Have the mover look again."""
    try:
        _kick.release()
    except RuntimeError:  # already kicked
        pass


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
        with LOCK:
            timeout = _move()
        _kick.acquire(True, -1 if timeout is None else timeout)


def _move() -> float | None:
    """End the device waits that are due, jumping first if every counted thread is blocked;
    the real time until the next one is due (None: nothing to wait for). Under LOCK."""
    due = [w for w in _waits if not w.woken and (w.kind is DEVICE or w.due)]
    if not due:
        return None
    idle = _running[0] == 0
    if idle:
        clock.jump(min(w.deadline for w in due))
    t, later = clock.now(), None
    for w in due:
        if w.deadline > t:
            if w.kind is DEVICE:
                later = w.deadline if later is None else min(later, w.deadline)
        elif wake(w):
            if w.kind is not DEVICE:
                w.timed_out = True
            elif not idle:
                _slept(w, t)
            if w.release is not None:
                w.release()
    return None if later is None else later - t


def _slept(w: Wait, t: float) -> None:
    """A device wait ended by the timer, while other threads ran: what kept the clock from
    jumping (each running counted thread's call site), and how late it ended."""
    REPORT["slept_s"] += clock._real() - w.real
    if t - w.deadline > _LATE:
        REPORT["late_resumes"] += 1
    blocked = {x.thread for x in _waits if x.counts and not x.woken}
    names = {th.ident: th.name for th in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        if ident in _counted and ident not in blocked and ident != w.thread:
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
        if timeout is not None and timeout <= 0:
            return select(self, timeout)
        with blocked(HOST if timeout is not None else None, None if timeout is None else clock.now() + timeout) as w:
            return _until(w, timeout, lambda t: select(self, t))

    return wrapped


def polling(poll, fd, timeout: float):
    """A poll for another process (mstar's Rust poll): ``poll(seconds)`` returns falsy at its
    timeout. ``fd``, the eventfd it also wakes on, if any (:func:`eventfd_write`)."""
    with blocked(HOST, clock.now() + timeout, fd=fd) as w:
        return _until(w, timeout, poll)


# ---------------------------------------------------------------- installation


def install() -> None:
    """Patch every time source and wait of the process onto the clock, and start the mover.
    Before mstar (or asyncio, concurrent.futures) is imported."""
    if _installed[0] or clock.FAST:
        return
    _installed[0] = True
    _counted.add(threading.get_ident())
    _running[0] = 1
    _kick.acquire()
    epoch = time.time() - clock.now()
    for name, fn in {
        "perf_counter": clock.now, "monotonic": clock.now, "time": lambda: epoch + clock.now(),
        "thread_time": clock.thread_time, "process_time": clock.process_time, "sleep": sleep,
    }.items():
        setattr(time, name, fn)
        if name != "sleep":
            setattr(time, name + "_ns", lambda fn=fn: int(fn() * 1e9))
    threading._time = clock.now
    queue.time = clock.now  # bound at import: ``from time import monotonic as time``
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
    """A forked child has only the forking thread, running."""
    LOCK._at_fork_reinit()
    _kick._at_fork_reinit()
    _kick.acquire()
    _counted.clear()
    _counted.add(threading.get_ident())
    _waits.clear()
    _fds.clear()
    _running[0] = 1
    for ident in list(clock._OPEN):
        if ident != threading.get_ident():
            del clock._OPEN[ident]
    clock.thaw()
    threading.Thread(target=_mover, name=OWN + "mover", daemon=True).start()
