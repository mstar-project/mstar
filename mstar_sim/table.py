"""The deployment's clock table: what its processes need of each other to agree on one time.

A shared file, ``/dev/shm/mstar_sim_clock_<uid>_<name>``, named by the deployment
(``MSTAR_SIM_NODE``, which the launcher takes from the port in its command). A process
with no name keeps a private one. Each process holds one slot, by an OFD lock on its first
byte, which the kernel drops when the process dies, however it dies. Only its owner writes a
slot, and every field is one aligned word: a reader needs no snapshot, as a clock only rises
and a deadline is written before the flag that says to read it.

A slot publishes its clock, its next event (the earliest deadline of a blocked process, ``inf``
if none), whether it runs, whether a thread of it is at a meet, and when it began to exit.

:meth:`Table.bound` is how far this process may go: the minimum over the other live slots of
their clock if running, else their next event. Each slot's mover has a futex word of its own in
its slot; a change of state wakes every live slot's.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import math
import mmap
import os
import struct
import time
import zlib

SLOTS = 64
_WORDS = 8  # a slot's 64-bit words
_CLOCK, _NEXT, _RUNNING, _MEET, _EPOCH, _RANK, _WAKE, _LEFT = range(8)
_HEADER = 8  # words before the first slot: the second is how many slots were ever used
_LIVE_FOR = 0.05  # real seconds a look at who is alive stands: a peer's death takes this long to see
_real = time.perf_counter  # (bound here, before the emulator patches it)
_SIZE = (_HEADER + SLOTS * _WORDS) * 8
_SYS_FUTEX = {"x86_64": 202, "aarch64": 98}[os.uname().machine]
_FUTEX_WAIT, _FUTEX_WAKE = 0, 1
_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long
_F_OFD_GETLK, _F_OFD_SETLK, _F_OFD_SETLKW = 36, 37, 38


class _Timespec(ctypes.Structure):
    _fields_ = [("sec", ctypes.c_long), ("nsec", ctypes.c_long)]


def _flock(kind: int, start: int) -> bytes:
    return struct.pack("hhqqi", kind, os.SEEK_SET, start, 1, 0)


class Table:
    """This process's view of the table (``path`` None: private to it)."""

    def __init__(self, path: str | None):
        self.path = path
        self._fd = -1
        while path is not None:  # one process at a time claims, and resets a dead run's table
            self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.fcntl(self._fd, _F_OFD_SETLKW, _flock(fcntl.F_WRLCK, 0))
            if os.fstat(self._fd).st_nlink:
                os.ftruncate(self._fd, _SIZE)
                break
            os.close(self._fd)  # the last one out took this one with it
        self._map = mmap.mmap(self._fd, _SIZE)
        self._q = memoryview(self._map).cast("Q")
        self._d = memoryview(self._map).cast("d")
        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        self._looked = -math.inf
        self.live: list[int] = []  # the others, as of the last look
        try:
            self.me = self._claim()
            self._mine = _HEADER + self.me * _WORDS
            self.live = self._live(SLOTS)  # (a dead run's count can't be trusted)
            if not self.live:
                self._map[:] = bytes(_SIZE)  # a dead run's table: a restart on its port starts clean
            self._q[1] = max(self._q[1], self.me + 1)
            self._d[self._mine + _CLOCK] = 0.0
            self._d[self._mine + _NEXT] = math.inf
            self._q[self._mine + _RUNNING] = 1
            self._q[self._mine + _MEET] = 0
            self._q[self._mine + _RANK] = 0
            self._d[self._mine + _LEFT] = 0.0
        finally:
            if self._fd >= 0:
                fcntl.fcntl(self._fd, _F_OFD_SETLK, _flock(fcntl.F_UNLCK, 0))

    def _claim(self) -> int:
        if self._fd < 0:
            return 0
        for slot in range(SLOTS):
            try:
                fcntl.fcntl(self._fd, _F_OFD_SETLK, _flock(fcntl.F_WRLCK, self._lock_byte(slot)))
                return slot
            except OSError:
                continue
        raise SystemExit(f"mstar_sim: the deployment's clock table has no free slot of {SLOTS}")

    @staticmethod
    def _lock_byte(slot: int) -> int:
        return (_HEADER + slot * _WORDS) * 8

    def _live(self, slots: int | None = None) -> list[int]:
        """The other processes' slots, among the first ``slots`` (by default all that were used): a
        held lock is a live process. (One ``fcntl`` a slot, so a look stands for ``_LIVE_FOR``, by
        which time a mover looks again anyway.)"""
        if self._fd < 0:
            return []
        if _real() - self._looked < _LIVE_FOR:
            return self.live
        self._looked = _real()
        found = []
        for slot in range(slots or self._q[1]):
            if slot != self.me:
                got = fcntl.fcntl(self._fd, _F_OFD_GETLK, _flock(fcntl.F_WRLCK, self._lock_byte(slot)))
                if struct.unpack("hhqqi", got)[0] != fcntl.F_UNLCK:
                    found.append(slot)
        return found

    def start_at(self) -> float | None:
        """Where a new process's clock starts: level with the deployment, so a respawn or a late
        client lands in its present (None: this is the first)."""
        bound = self.bound()
        return bound if math.isfinite(bound) else self.latest() or None

    def latest(self) -> float:
        """The latest clock among the others (0.0: none has published)."""
        return max((self._clock(slot) for slot in self.live), default=0.0)

    def _clock(self, slot: int) -> float:
        """A slot's clock: one exiting runs on at real rate, as nothing reads it now."""
        base = _HEADER + slot * _WORDS
        left = self._d[base + _LEFT]
        return self._d[base + _CLOCK] + (_real() - left if left else 0.0)

    def publish(self, clock: float, running: bool, next_event: float = math.inf, meet: int = 0) -> None:
        """This slot's state changed: tell the others (a new epoch, and every mover wakes).
        ``meet``: the group word (:func:`group_word`) of a meet a thread of it is in, or 0. Each
        field is one word, read without a snapshot: so a slot claims running before anything
        else changes, and claims blocked only after its next event is in place, and a reader
        between two stores errs low."""
        self._q[1] = max(self._q[1], self.me + 1)
        if running:
            self._q[self._mine + _RUNNING] = 1
        self._d[self._mine + _CLOCK] = clock
        self._d[self._mine + _NEXT] = next_event
        self._q[self._mine + _MEET] = meet
        if not running:
            self._q[self._mine + _RUNNING] = 0
        self._q[self._mine + _EPOCH] += 1
        self.kick()

    def rank(self) -> None:
        """This process is a rank of collectives (:meth:`bound`)."""
        self._q[self._mine + _RANK] = 1

    def tick(self, clock: float) -> None:
        """A running clock's latest reading: no change of state, so no one is woken."""
        self._d[self._mine + _CLOCK] = clock

    def moved(self, clock: float) -> None:
        """This process's clock moved: others may follow, but its state is what it was."""
        self._d[self._mine + _CLOCK] = clock
        self.kick()

    def epochs(self) -> int:
        """The sum of the slots' epochs: it stands still while no state changes."""
        return sum(self._q[_HEADER + slot * _WORDS + _EPOCH] for slot in range(self._q[1]))

    def bound(self) -> float:
        """How far this process may go. A rank at a meet whose group has yet to all arrive
        stands aside, as it may wait on someone this process holds back; once all have, it
        needs no one, and counts at its clock (one still running counts anyway). To another
        rank, a rank at a meet always stands aside: nothing bounds rank against rank."""
        bound = math.inf
        live = self.live = self._live()
        rank = self._q[self._mine + _RANK]
        meets = [self._q[_HEADER + slot * _WORDS + _MEET] for slot in live]  # one reading of them all
        for slot, meet in zip(live, meets, strict=True):
            base = _HEADER + slot * _WORDS
            if not self._d[base + _CLOCK]:  # (0: it has yet to publish)
                continue
            if self._d[base + _LEFT]:
                bound = min(bound, self._clock(slot))
            elif meet:
                if rank or not (self._q[base + _RUNNING] or meets.count(meet) >= meet & 0xFF):
                    continue
                bound = min(bound, self._d[base + _CLOCK])
            elif self._q[base + _RUNNING]:
                bound = min(bound, self._d[base + _CLOCK])
            else:
                bound = min(bound, self._d[base + _NEXT])
        return bound

    def describe(self) -> str:
        """Every live slot's state, for a stall that has to be reported."""
        lines = []
        for slot in [self.me, *self._live()]:
            base = _HEADER + slot * _WORDS
            lines.append(f"  slot {slot}{' (this process)' if slot == self.me else ''}: clock "
                         f"{self._d[base + _CLOCK]:.6f}, {'running' if self._q[base + _RUNNING] else 'blocked'}, next "
                         f"event {self._d[base + _NEXT]:.6f}{', at a meet' if self._q[base + _MEET] else ''}"
                         f"{', exiting' if self._d[base + _LEFT] else ''}")
        return "\n".join(lines)

    def leave(self) -> None:
        """A clean exit, before the process's last real time. It is still a peer until it dies,
        when the kernel drops its slot: it counts as running at real rate, as nothing reads its
        clock now. The last slot out takes the table with it (whoever claims a table with no live
        slot resets it, should a death leave one)."""
        if self._fd >= 0:
            fcntl.fcntl(self._fd, _F_OFD_SETLKW, _flock(fcntl.F_WRLCK, 0))
            self._d[self._mine + _LEFT] = _real()
            self._looked = -math.inf
            if all(self._d[_HEADER + slot * _WORDS + _LEFT] for slot in self._live()):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.path)
            fcntl.fcntl(self._fd, _F_OFD_SETLK, _flock(fcntl.F_UNLCK, 0))

    def _wake(self, slot: int) -> None:
        """Wake a slot's mover. Its word moves by an amount of this slot's, so two writers racing
        can't leave it where its mover last read it."""
        word = _HEADER + slot * _WORDS + _WAKE
        self._q[word] = (self._q[word] + 1 + (self.me << 16)) & 0xFFFFFFFF
        _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._base + word * 8), _FUTEX_WAKE, 1 << 30, None, None, 0)

    def kick(self) -> None:
        """Wake every mover."""
        for slot in (self.me, *self.live):
            self._wake(slot)

    def kick_me(self) -> None:
        """Wake this process's mover."""
        self._wake(self.me)

    def word(self) -> int:
        return self._q[self._mine + _WAKE]

    def wait(self, word: int, timeout: float | None) -> None:
        """Sleep until this slot's word moves on from ``word`` (read before looking at the state),
        or for ``timeout`` real seconds."""
        spec = None
        if timeout is not None:
            spec = _Timespec(int(timeout), int((timeout % 1) * 1e9))
        _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._base + (self._mine + _WAKE) * 8), _FUTEX_WAIT,
                      word & 0xFFFFFFFF, None if spec is None else ctypes.byref(spec), None, 0)


def group_word(ranks) -> int:
    """A meet's group, the same in every process: its ranks' checksum, and their count in the
    low byte."""
    return (zlib.crc32(repr(sorted(ranks)).encode()) << 8) | min(len(ranks), 0xFF)


def path_for(name: str | None) -> str | None:
    return None if not name else f"/dev/shm/mstar_sim_clock_{os.getuid()}_{name}"
