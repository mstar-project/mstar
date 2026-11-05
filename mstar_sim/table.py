"""The deployment's clock table: what its processes need of each other to agree on one time.

A shared file, ``/dev/shm/mstar_sim_clock_<uid>_<name>``, named by the deployment
(``MSTAR_SIM_NODE``, which the launcher takes from the port in its command). A process
with no name keeps a private one. Each process holds one slot, by an OFD lock on its first
byte, which the kernel drops when the process dies, however it dies. Only its owner writes a
slot, and every field is one aligned word: a reader needs no snapshot, as a clock only rises
and a deadline is written before the flag that says to read it.

A slot publishes its clock, its next event (the earliest deadline of a blocked process, ``inf``
if none), whether it runs, whether it is in a meet, what it sent each peer, and the sum it had
taken of what was sent to it. Messages in flight to slot *j* are what the live slots sent it
less what it took: per receiver, so a message to one process freezes no other, and a dead
receiver's go with it. Sends the emulator can't see (the Rust runtime's own) are in neither
number: a receiver takes all it was sent so far each time a poll finds its socket empty.

:meth:`Table.bound` is how far this process may go: the minimum over the other live slots of
their clock if running or if a message is in flight to them, else their next event. A slot in
a meet stands aside. One futex word wakes every process's mover on any change of state.
"""

from __future__ import annotations

import ctypes
import fcntl
import math
import mmap
import os
import struct

SLOTS = 64
NAMES = 8  # communicator ids a process can answer to
_WORDS = 8 + NAMES + SLOTS  # a slot's 64-bit words
_CLOCK, _NEXT, _RUNNING, _MEET, _EPOCH, _TAKEN, _PID = range(7)
_NAME, _SENT = 8, 8 + NAMES
_HEADER = 8  # words before the first slot: the futex word, then how many slots were ever used
_SIZE = (_HEADER + SLOTS * _WORDS) * 8
_SYS_FUTEX, _FUTEX_WAIT, _FUTEX_WAKE = 202, 0, 1
_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long
_F_OFD_GETLK, _F_OFD_SETLK = 36, 37


class _Timespec(ctypes.Structure):
    _fields_ = [("sec", ctypes.c_long), ("nsec", ctypes.c_long)]


def _flock(kind: int, start: int) -> bytes:
    return struct.pack("hhqqi", kind, os.SEEK_SET, start, 1, 0)


def name_hash(name: str) -> int:
    """FNV-1a, odd so never an empty field: the same in every process (``hash`` isn't)."""
    h = 1469598103934665603
    for b in name.encode():
        h = ((h ^ b) * 1099511628211) & (2**64 - 1)
    return h | 1


class Table:
    """This process's view of the table (``path`` None: private to it)."""

    def __init__(self, path: str | None):
        if path is None:
            self._fd = -1
            self._map = mmap.mmap(-1, _SIZE)
        else:
            self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            os.ftruncate(self._fd, _SIZE)
            self._map = mmap.mmap(self._fd, _SIZE)
        self._q = memoryview(self._map).cast("Q")
        self._d = memoryview(self._map).cast("d")
        self._word = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        self.me = self._claim()
        self._mine = _HEADER + self.me * _WORDS
        self._q[1] = SLOTS
        if not self._live():
            self._map[:] = bytes(_SIZE)  # a dead run's table: a restart on its port starts clean
        self._q[1] = max(self._q[1], self.me + 1)
        self.live: list[int] = []  # as of the last bound
        self._peers: dict[str, int] = {}
        self._names = 0
        self._d[self._mine + _CLOCK] = 0.0
        self._d[self._mine + _NEXT] = math.inf
        self._q[self._mine + _RUNNING] = 1
        self._q[self._mine + _MEET] = 0
        self._q[self._mine + _PID] = os.getpid()
        for i in range(NAMES):
            self._q[self._mine + _NAME + i] = 0
        for i in range(SLOTS):
            self._q[self._mine + _SENT + i] = 0
        self.live = self._live()
        self.take(self.sent_to_me())  # a respawn starts level with what its senders sent before

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

    def _live(self) -> list[int]:
        """The other processes' slots: a held lock is a live process."""
        if self._fd < 0:
            return []
        found = []
        for slot in range(self._q[1]):
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
        return max((self._d[_HEADER + slot * _WORDS + _CLOCK] for slot in self.live), default=0.0)

    def publish(self, clock: float, running: bool, next_event: float = math.inf, meet: bool = False) -> None:
        """This slot's state changed: tell the others (a new epoch, and every mover wakes)."""
        self._q[1] = max(self._q[1], self.me + 1)
        self._d[self._mine + _CLOCK] = clock
        self._d[self._mine + _NEXT] = next_event
        self._q[self._mine + _MEET] = meet
        self._q[self._mine + _RUNNING] = running
        self._q[self._mine + _EPOCH] += 1
        self.kick()

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
        bound = math.inf
        live = self.live = self._live()
        for slot in live:
            base = _HEADER + slot * _WORDS
            if self._q[base + _MEET] or not self._d[base + _CLOCK]:  # (0: it has yet to publish)
                continue
            if self._q[base + _RUNNING] or self._in_flight(slot, base, live):
                bound = min(bound, self._d[base + _CLOCK])
            else:
                bound = min(bound, self._d[base + _NEXT])
        return bound

    def _in_flight(self, slot: int, base: int, live: list[int]) -> bool:
        sent = self._q[self._mine + _SENT + slot]
        for other in live:
            sent += self._q[_HEADER + other * _WORDS + _SENT + slot]
        return sent > self._q[base + _TAKEN]

    def answer_to(self, name: str) -> None:
        """Messages to communicator id ``name`` come here."""
        if self._names < NAMES:
            self._q[self._mine + _NAME + self._names] = name_hash(name)
            self._names += 1

    def sent(self, name: str) -> None:
        """A message to communicator id ``name`` is on its way."""
        slot = self._peers.get(name)
        if slot is None:
            wanted = name_hash(name)
            for other in self.live:
                base = _HEADER + other * _WORDS + _NAME
                if wanted in (self._q[base + i] for i in range(NAMES)):
                    slot = self._peers[name] = other
                    break
            else:
                return  # not (yet) a process of this deployment
        self._q[self._mine + _SENT + slot] += 1
        self._q[self._mine + _EPOCH] += 1

    def sent_to_me(self) -> int:
        return sum(self._q[_HEADER + slot * _WORDS + _SENT + self.me] for slot in self.live)

    def take(self, sent: int) -> None:
        """All of ``sent`` (:meth:`sent_to_me`, read before a poll found nothing) has arrived."""
        if sent != self._q[self._mine + _TAKEN]:
            self._q[self._mine + _TAKEN] = sent
            self._q[self._mine + _EPOCH] += 1

    def kick(self) -> None:
        """Wake every mover. Each slot steps the word by a different amount, so two writers
        racing can't leave it where a waiter last read it."""
        self._q[0] = (self._q[0] + 1 + (self.me << 16)) & 0xFFFFFFFF
        _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._word), _FUTEX_WAKE, 1 << 30, None, None, 0)

    def word(self) -> int:
        return self._q[0]

    def wait(self, word: int, timeout: float | None) -> None:
        """Sleep until the word moves on from ``word`` (read before looking at the state), or
        for ``timeout`` real seconds."""
        spec = None
        if timeout is not None:
            spec = _Timespec(int(timeout), int((timeout % 1) * 1e9))
        _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._word), _FUTEX_WAIT, word & 0xFFFFFFFF,
                      None if spec is None else ctypes.byref(spec), None, 0)


def path_for(name: str | None) -> str | None:
    return None if not name else f"/dev/shm/mstar_sim_clock_{os.getuid()}_{name}"
