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
so idle processes keep real time rather than spin ahead. Running, a thread that reads the clock
more than W past the bound waits there (:func:`read`), blocked, until the bound rises. A thread at
a collective's meet stands still, as NCCL's host doesn't wait: it takes no moves, and stands aside
from the bound until its group has all come (:meth:`mstar_sim.table.Table.bound`).

The kernel says what the registry can't. A process whose threads are in waits it doesn't know is
*parked* (counts as blocked) when, between two of the mover's looks, nothing read the clock, no
CPU time was used, and no task is on a CPU or waiting for one (:func:`_look`), so no such wait
paces the deployment. And a process says it is blocked, and takes a move on the bound, only once
its mover has seen no task of its own on a CPU (:func:`_quiet`): a message on its way always has
a runnable thread on its path, so nothing is moved past one in flight. The settle window is left
for a third process that read the receiver's next event just before that.

A native child no patch reaches (the Rust frontend, ffmpeg: :func:`watch`) counts as one more
thread of the process that started it, running while it uses or waits for a CPU, so no one passes
its work, which takes real time on the clock, and idle it holds no one back. It reads real time
itself, and is never held.

Threads named ``mstar_sim-*`` are the emulator's own and never counted. ``--fast`` installs
none of this.
"""

from __future__ import annotations

import _queue
import _thread
import atexit
import ctypes
import math
import os
import queue
import selectors
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager

from mstar_sim import clock, table

LOCK = clock.LOCK
DEVICE, HOST, MEET, AHEAD = "device", "host", "meet", "ahead"
W = 0.5e-3  # how far past the bound a running process may read the clock
SETTLE = 200e-6  # how long the deployment's state stays still before a move on the bound
_MAX_SLEEP = 0.05  # the mover looks at least this often: a peer's death wakes no one
HELD_LIMIT = 60.0  # real seconds the bound may stand still under a held thread before the emulator gives up
_real_lock = _thread.allocate_lock
_real_sleep = time.sleep
_real_eventfd_write = os.eventfd_write
# a device wait woken this much after its time by the mover's timer counts as a late resume
_LATE = 1e-3
OWN = "mstar_sim-"  # the emulator's own threads' names start with this

_counted: set[int] = set()  # threads the registry counts
_running = clock.RUNNING
_waits: set[Wait] = set()
_fds: dict[int, Wait] = {}  # an fd a poll waits on: its eventfd_write wakes the poll
_installed = [False]
TABLE: table.Table
_ahead = [math.inf]  # past this a clock read looks at the bound
_seen = [-1]  # the deployment's state when _ahead was last worked out
_settling = [0, 0.0]  # the deployment's state, and the real time it last changed
_wakefd = [-1]  # the mover's wake for a select or a poll, which no lock can end
_poked: set[Wait] = set()
_told_blocked = [False]  # the table says this process is blocked
_native: dict = {}  # each watched child: its CPU clock, its last CPU time, its tasks
_native_busy = [False]  # one is working, which counts as a running thread
_looked = [0.0, 0.0]  # the mover's last look: when, and what the process's other threads had used
_moved = [0.0, 0.0]  # the last move on the bound: when, and how far
_CPU_ALLOWANCE = 1e-3  # CPU time a look lets pass as nothing: zmq's and gloo's own threads ticking over
_shut = [0.0]  # the real time the gate found a task on a CPU, with nothing running (0.0: it hasn't)
_GATE_LIMIT = 1.0  # real seconds a message can't stay on its way, though a loaded box descheduled it
_meet = [0]  # the group word of the meet a thread of this process is in (dist)
REPORT = {"slept_s": 0.0, "late_resumes": 0, "jump_blockers": {}, "ahead_stalls": 0, "late_deliveries": 0,
          "meet_skew_s": 0.0, "mover_s": 0.0, "scan_s": 0.0, "scans": 0, "settle_deferrals": 0, "gate_waits": 0,
          "parks": 0, "unparks_by_look": 0}


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
        kick()  # the mover tells the table, once no task of this process is on a CPU


def _run_one(thaw: bool = True) -> None:
    """A thread runs. ``thaw`` False: the mover's delivery, whose thread has yet to get a CPU. It
    counts as running, so nothing jumps over it, but the clock waits for it (:func:`leave`):
    how long the box takes to run it is no host time of mstar's."""
    _running[0] += 1
    if thaw:
        clock.thaw()
    if _running[0] == 1 and _told_blocked[0] and not clock.PARKED[0]:
        _told_blocked[0] = False
        _publish()


def _publish() -> None:
    """This process's state changed: its next event is the earliest deadline it holds."""
    pending = [w for w in _waits if not w.woken]
    TABLE.publish(clock.now(), not _told_blocked[0], min((w.deadline for w in pending if w.deadline is not None),
                                                         default=math.inf),
                  _meet[0] if any(w.kind is MEET for w in pending) else 0)


def enter(kind, deadline=None, release=None, fd=None) -> Wait | None:
    """This thread enters a wait (None when the registry isn't installed). ``release`` ends it
    early (the mover's delivery); ``fd``, an fd whose ``eventfd_write`` wakes it."""
    if not _installed[0]:
        return None
    w = Wait(kind, deadline, threading.get_ident() in _counted, release, fd)
    clock.pause()
    with LOCK:
        clock.READS[0] += 1
        if clock.PARKED[0]:
            _unpark()
        _waits.add(w)
        if fd is not None:
            _fds[fd] = w
        if w.counts:
            _block_one()
        if kind is MEET or (deadline is not None and _told_blocked[0]):
            _publish()
        if kind is DEVICE or kind is MEET:
            kick()
    return w


def wake(w: Wait, thaw: bool = True) -> bool:
    """Mark ``w``'s thread running (a waker, before it can block itself). False if something
    woke it already."""
    with LOCK:
        if w.woken:
            return False
        w.woken = True
        if w.counts:
            _run_one(thaw)
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
        if clock.PARKED[0]:
            _unpark()
        if not w.woken:
            w.woken = True
            if w.counts:
                _run_one()
        if w.kind is MEET or (w.deadline is not None and _told_blocked[0]):
            _publish()  # a blocked process's next event went with it: peers' bounds hang on that
        if _running[0]:
            clock.thaw()
    clock.resume()


def _until(w: Wait | None, timeout, attempt):
    """``attempt(seconds)``, a wait ending in real time (falsy if it timed out), for wait ``w``:
    retried until it ends or the clock reaches ``w``'s deadline (none: tried once). After its
    real timeout the deadline is due (the mover may then move the clock to it, bound or no
    bound). The mover ends it (``w.release``, ``w.timed_out``) when the clock gets there."""
    result = attempt(timeout)
    while not result and w is not None and w.deadline is not None:
        with LOCK:
            t = clock.now()
            if w.timed_out or t >= w.deadline:
                break
            if not w.due:
                w.due = True
                kick()
        result = attempt(w.deadline - t)
    return result


@contextmanager
def blocked(kind=None, deadline=None, fd=None, release=None, group: int = 0):
    """A wait whose waker this module doesn't see (a gloo collective, a Rust poll): the thread
    counts as blocked until it returns. ``group``: a meet's (:func:`mstar_sim.table.group_word`)."""
    if group:
        _meet[0] = group
    w = enter(kind, deadline, release, fd)
    try:
        yield w
    finally:
        leave(w)


def kick() -> None:
    """Have this process's mover look again."""
    TABLE.kick_me()


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
    """Every deadline hangs on this thread, so it doesn't die quietly."""
    _mover_tid[0] = str(threading.get_native_id())
    try:
        while True:
            word = TABLE.word()  # before it looks: a change while it does ends the wait at once
            with LOCK:
                began = clock._real_thread_time()
                timeout = _move()
                spent = clock._real_thread_time() - began
                if _running[0] > _native_busy[0]:  # it took the GIL from a Python thread (not from a child)
                    clock.spent(spent)
                REPORT["mover_s"] += spent
            TABLE.wait(word, timeout)
    except Exception:
        traceback.print_exc()
        clock.abort()


def _move() -> float:
    """Release the threads the bound has caught up with; look at whether the process is as idle as
    the registry can't tell; with no thread running, tell the table so, and move the clock to the
    process's next event; end the waits that are due. Returns how long in real time it may
    sleep, the deployment's changes aside. Under LOCK."""
    pending = [w for w in _waits if not w.woken]
    if not _told_blocked[0]:
        TABLE.tick(clock.now())  # a thread busy in native code (an import) reads no clock
    bound = TABLE.bound()
    _ahead[0] = bound + W
    _seen[0] = -1  # (a read that worked out a lower one just now, on newer state, looks again)
    for w in pending:
        if w.kind is AHEAD and bound + W >= w.deadline and wake(w, thaw=False):
            w.release()
    meet = any(w.kind is MEET for w in pending)  # a rank at a meet follows no one; what is due still ends
    timeout = _MAX_SLEEP
    _look()
    if _native:
        _watch_children()
    idle = _running[0] == 0 or clock.PARKED[0]
    if _native and (_native_busy[0] or (idle and any(w.deadline is not None for w in pending))):
        timeout = 2 * SETTLE  # a move is pending on a child: look often
    if not idle:
        _shut[0] = 0.0
    if idle:
        quiet = _gate()
        if quiet and not _told_blocked[0]:
            _told_blocked[0] = True
            _publish()
        elif not quiet:
            REPORT["gate_waits"] += 1
            timeout = SETTLE
    if idle:
        device = min((w.deadline for w in pending if w.kind is DEVICE), default=math.inf)
        hosts = [w for w in pending if w.kind is HOST and w.deadline is not None]
        host = min((w.deadline for w in hosts), default=math.inf)
        target = math.inf
        if device < math.inf and (device <= host or bound == math.inf or meet):
            target = min(device, host)  # a device deadline needs no one, and a sleep on its way ends as it passes
        elif not meet:
            due = min((w.deadline for w in hosts if w.due), default=math.inf)
            target = min(host, bound) if bound < math.inf else due
            if bound < host < math.inf:  # the bound, running, wakes no one as it rises
                timeout = min(timeout, max(host - bound, SETTLE))
            if target < math.inf and target > clock.now():
                waited = _settled()
                if not quiet:
                    target = math.inf
                elif waited < SETTLE:
                    REPORT["settle_deferrals"] += 1
                    target = math.inf
                    timeout = SETTLE - waited
        if target < math.inf and target > clock.now():
            _moved[:] = [clock._real(), target - clock.now()]
            clock.jump(target)
            TABLE.moved(target)
    t = clock.now()
    for w in pending:
        if w.kind is not DEVICE and w.kind is not HOST or w.deadline is None or w.woken:
            continue
        if w.deadline > t:
            if w.kind is DEVICE:
                timeout = min(timeout, w.deadline - t)
        elif wake(w, thaw=False):
            if w.kind is HOST:
                w.timed_out = True
            elif not idle:
                _slept(w, t)
            if w.release is _poke:
                _poked.add(w)
            if w.release is not None:
                w.release()
    return timeout


class _Tasks:
    """A process's tasks: is one on a CPU, or waiting for one, or in the kernel? A task the box
    has descheduled reads R, so the answer doesn't depend on load. A cached fd a task."""

    def __init__(self, pid):
        self.root, self.fds = f"/proc/{pid}/task", {}

    def on_cpu(self, skip: str | None = None) -> bool:
        try:
            tids = os.listdir(self.root)
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError:
            return True  # it can't tell: better a move delayed than one past a message
        for tid in self.fds.keys() - set(tids):
            os.close(self.fds.pop(tid))
        found = False
        for tid in tids:
            if tid == skip or found:
                continue
            try:
                fd = self.fds.get(tid)
                if fd is None:
                    fd = self.fds[tid] = os.open(f"{self.root}/{tid}/stat", os.O_RDONLY)
                stat = os.pread(fd, 128, 0)
                found = stat[stat.rindex(b")") + 2] in b"RD"
            except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
                pass  # a task that has just gone
            except OSError:
                found = True
        return found

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


_own = _Tasks("self")
_mover_tid = [""]
_libc = ctypes.CDLL(None)


def _quiet() -> bool:
    """Nothing of this process can be on its way to a peer. A send wakes the next thread on its path
    inside the syscall, so a message in flight has a runnable thread until the receiver has it."""
    began = clock._real()
    working = _own.on_cpu(_mover_tid[0])
    REPORT["scans"] += 1
    REPORT["scan_s"] += clock._real() - began
    return not working


def _gate() -> bool:
    """Whether a process with no thread running may say so. It waits for :func:`_quiet`, but a task
    on a CPU for ``_GATE_LIMIT`` is no message: a native spinner would hold the gate for ever, and
    the slot, running at a clock that stands still, would hold every peer to the abort."""
    if _quiet():
        _shut[0] = 0.0
        return True
    _shut[0] = _shut[0] or clock._real()
    return clock._real() - _shut[0] > _GATE_LIMIT


def _watch_children() -> None:
    """A child at a CPU counts as a running thread of this process. Under LOCK."""
    began = clock._real()
    working = any(child[2].on_cpu() for child in list(_native.values()))
    REPORT["scan_s"] += clock._real() - began
    if working != _native_busy[0]:
        _native_busy[0] = working
        _run_one() if working else _block_one()


def _others_cpu() -> float:
    """The CPU time this process's other threads and its children have used: the mover asks."""
    cpu = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID) - clock._real_thread_time()
    for pid, child in list(_native.items()):
        try:
            child[1] = time.clock_gettime(child[0])
        except OSError:  # it has gone
            child[2].close()
            del _native[pid]
            if not _native and _native_busy[0]:
                _native_busy[0] = False
                _block_one()
            continue
        cpu += child[1]
    return cpu


def _look() -> None:
    """At most once a ``_MAX_SLEEP``: a process with running threads that has shown no sign of life
    since the last look is parked; a parked one showing any is not. It takes all three signs: a
    Python thread gives the GIL up for the mover to look, so at that moment it is off CPU in a
    busy process, and only the CPU time it used since says so. Under LOCK."""
    real = clock._real()
    if real - _looked[0] < _MAX_SLEEP or _running[0] == 0:
        return
    reads, clock.READS[0] = clock.READS[0], 0
    cpu = _others_cpu()
    used = cpu - _looked[1]
    _looked[:] = [real, cpu]
    busy = reads or used > _CPU_ALLOWANCE or clock._OPEN or not _quiet()
    if busy and clock.PARKED[0]:
        REPORT["unparks_by_look"] += 1
        _unpark()
    elif not busy and not clock.PARKED[0]:
        REPORT["parks"] += 1
        clock.PARKED[0] = True
        clock.freeze()


def _unpark() -> None:
    """A parked process shows a sign of life: its threads are running after all. Under LOCK."""
    _catch_up()
    clock.PARKED[0] = False
    clock.thaw()
    if _told_blocked[0]:
        _told_blocked[0] = False
        _publish()


def _catch_up() -> None:
    """A process about to run with no thread awake moves up to the others' latest clock (no further
    than the bound or its own earliest deadline), as something it can't see may have woken it,
    and what it was sent then doesn't land in its past. With no one to follow, it was waiting on
    something outside the deployment: that took the real time it stood still. Under LOCK."""
    latest = TABLE.latest() or clock.now() + clock.stood_still()
    pending = min((w.deadline for w in _waits if not w.woken and w.deadline is not None and w.kind is not AHEAD),
                  default=math.inf)
    clock.jump(min(TABLE.bound(), latest, pending))


def delivered() -> None:
    """A poll found a message. Within a millisecond of a move past W, it may have been passed."""
    if clock._real() - _moved[0] < 1e-3 and _moved[1] > W:
        REPORT["late_deliveries"] += 1


def watch(pid: int) -> None:
    """Count native child ``pid`` as a thread of this process, running while it uses a CPU."""
    clk = ctypes.c_int()
    if _libc.clock_getcpuclockid(pid, ctypes.byref(clk)) == 0:
        with LOCK:
            _native[pid] = [clk.value, 0.0, _Tasks(pid)]


def _settled() -> float:
    """The real time the deployment's state has stood still."""
    epochs, real = TABLE.epochs(), clock._real()
    if epochs != _settling[0]:
        _settling[:] = [epochs, real]
    return real - _settling[1]


def read() -> float:
    """A clock read by mstar: publishes a running clock, and holds a thread that has got more
    than W past the bound, until the bound rises (clause 2), and reads again: what it returns is
    always a time it checked. Not a read made inside this module's own bookkeeping (a GC callback
    that runs while LOCK is held): the mover needs LOCK to let it go."""
    clock.READS[0] += 1
    if clock.PARKED[0] and not LOCK._is_owned():
        _unpark_now()
    while True:
        t = clock.now()
        TABLE.tick(t)
        epochs = TABLE.epochs()
        if epochs != _seen[0]:  # a change of state anywhere can lower the bound: look again
            _seen[0] = epochs
            _ahead[0] = -math.inf
        if (t <= _ahead[0] or threading.get_ident() not in _counted or getattr(clock._tls, "computing", 0)
                or LOCK._is_owned()):
            return t
        bound = TABLE.bound()
        _ahead[0] = bound + W
        if t <= _ahead[0]:
            return t
        _hold(t)


def _hold(t: float) -> None:
    """This thread waits for the bound to reach ``t``. It is blocked meanwhile, so its process's
    clock stands still once every thread is, and the slowest process is never held."""
    lock = _real_lock()
    lock.acquire()
    w = enter(AHEAD, t, release=lock.release)
    REPORT["ahead_stalls"] += 1
    try:  # the mover lets it go; it looks too, as the mover is gone at interpreter exit
        bound, end = TABLE.bound(), clock._real() + HELD_LIMIT
        while not lock.acquire(True, _MAX_SLEEP):
            was, bound = bound, TABLE.bound()
            if bound + W >= t:
                break
            if bound > was:
                end = clock._real() + HELD_LIMIT
            elif clock._real() > end:
                _held_too_long(t)
    finally:
        leave(w)


def _held_too_long(t: float) -> None:
    """The deployment's clock has stopped: an emulator stall, which is never wrong timing and
    never a hang. Abort as a rank whose collectives don't pair does, saying who holds the bound."""
    print(f"mstar_sim: pid {os.getpid()} held {HELD_LIMIT:.0f} s at clock {t:.6f}, past the bound "
          f"{TABLE.bound():.6f}; the deployment's processes:\n{TABLE.describe()}", file=sys.stderr, flush=True)
    clock.abort()


def _slept(w: Wait, t: float) -> None:
    """A device wait ended by the timer, while other threads ran: what kept the clock from
    jumping (each running counted thread's call site), and how late it ended."""
    REPORT["slept_s"] += clock._real() - w.real
    if t - w.deadline > _LATE:
        REPORT["late_resumes"] += 1
    blocked = {x.thread for x in _waits if x.counts and not x.woken}
    names = {ident: th.name for ident, th in list(threading._active.items())}  # (not enumerate: it takes a lock)
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
        with LOCK:  # a notify comes before this, and is seen here, or finds the wait listed
            if self._lock.acquire(False):
                return True
            w = self._wait = enter(HOST if timeout >= 0 else None, clock.now() + timeout if timeout >= 0 else None,
                                   release=self._lock.release)
        try:
            got = _until(w, timeout, lambda t: self._lock.acquire(True, t))
        finally:
            self._wait = None
            leave(w)
        return got and not w.timed_out

    def release(self):
        with LOCK:
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
        # (a timed get is a wait this module doesn't know: nothing in mstar makes one)
        counts = block and timeout is None and _installed[0] and threading.get_ident() in _counted
        if counts:
            with LOCK:
                self._getters += 1
                self._recount()
        try:
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
            _woken_unseen(ready)
            return ready

    return wrapped


def _woken_unseen(ready) -> None:
    """A channel the emulator can't see ended a wait, with no other thread awake."""
    if ready:
        with LOCK:
            if _running[0] == 0:
                _catch_up()


def polling(poll, fd, timeout: float):
    """A poll for another process (mstar's Rust poll): ``poll(seconds)`` returns falsy at its
    timeout. ``fd``, the eventfd it also wakes on, if any (:func:`eventfd_write`)."""
    with blocked(HOST, clock.now() + timeout, fd=fd, release=_poke) as w:
        ready = _until(w, timeout, poll)
        _woken_unseen(ready)
        return ready




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
    TABLE = table.Table(table.path_for(os.environ.get("MSTAR_SIM_NODE")))
    _wakefd[0] = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    epoch = time.time() - clock._real()  # the same in every process: wall time is one time
    start = TABLE.start_at()
    if start is not None:
        clock.start_at(start)
    TABLE.publish(clock.now(), True)
    clock.UNPARK[0] = _unpark_now
    atexit.register(_leave)
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

    popen = subprocess.Popen.__init__

    def popen_watched(self, args, *rest, **kwargs):
        popen(self, args, *rest, **kwargs)
        command = args if isinstance(args, (str, bytes, os.PathLike)) else args[0]
        path = shutil.which(os.fsdecode(command)) or os.fsdecode(command)
        if os.path.realpath(path) != os.path.realpath(sys.executable):  # a Python has a slot of its own
            watch(self.pid)

    subprocess.Popen.__init__ = popen_watched
    threading.Thread.start = start_counted
    threading.Thread._bootstrap_inner = bootstrap_counted
    os.register_at_fork(after_in_child=_after_fork)
    threading.Thread(target=_mover, name=OWN + "mover", daemon=True).start()


def _unpark_now() -> None:
    with LOCK:
        if clock.PARKED[0]:
            _unpark()


def _leave() -> None:
    TABLE.tick(clock.now())
    TABLE.leave()  # (this process's table, after a fork)


def _after_fork() -> None:
    """A forked child has only the forking thread, running, and a slot of its own to find."""
    global TABLE
    LOCK._at_fork_reinit()
    TABLE = table.Table(table.path_for(os.environ.get("MSTAR_SIM_NODE")))
    _counted.clear()
    _counted.add(threading.get_ident())
    _waits.clear()
    _fds.clear()
    _native.clear()
    _running[0] = 1
    _told_blocked[0] = _native_busy[0] = clock.PARKED[0] = False
    for ident in list(clock._OPEN):
        if ident != threading.get_ident():
            del clock._OPEN[ident]
    clock.thaw()
    TABLE.publish(clock.now(), True)
    threading.Thread(target=_mover, name=OWN + "mover", daemon=True).start()
