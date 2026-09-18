"""PageAllocator: the KV page free list. This machine uses several threads.

`PageAllocator` holds a lock (`mstar/engine/resources/kv/cache.py`). The lock
makes the size check and the gets after it atomic against a `free` on another
thread. This machine drives that path.

The schedule is part of the case. Each op names the worker to run next. One
worker runs at a time, and one driver makes every choice. Thus a case always
replays to the same interleaving. The preemption points are the accesses to
the free list and the lock operations.

Invariants
----------
alloc.no_double_issue_across_threads   no page is held by two threads at once
alloc.free_and_held_disjoint           a page is free or held, never both
alloc.conservation                     free + held is 0..max-1 at every point
alloc.no_deadlock                      unfinished threads are not all blocked
alloc.quiesce_accounts_for_every_page  every page is accounted for at the end
alloc.thread_completes                 no worker blocks forever or raises

Not covered:

* the full set of interleavings: the driver samples them
* a large configuration: the generator makes a small one
* a fault between two preemption points
* a caller that reads `num_free` and then allocates: `num_free` reads the
  queue without the lock"""

from __future__ import annotations

import queue as _queue
import random
import threading
import weakref
from collections.abc import Iterator

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401
from mstar.engine.resources.kv.cache import PageAllocator

AT_POINT = "point"
BLOCKED = "blocked"
DONE = "done"
RELEASE = "release"


class _AbandonedError(Exception):
    """Tells a worker to exit after the driver drops the case."""


class _WouldBlockError(Exception):
    """The real code would block here forever on an empty free list."""


class _Driver:
    """The only code that decides which worker runs next."""

    def __init__(self, tids: list[str]) -> None:
        self.state = {tid: AT_POINT for tid in tids}
        self.go = {tid: threading.Event() for tid in tids}
        self.reports: _queue.Queue = _queue.Queue()
        self.trace: list[str] = []
        self.dead = False
        self.primed = False

    # -- worker side ---------------------------------------------------------

    def _park(self, tid: str, kind: str) -> None:
        self.reports.put((tid, kind))
        self.go[tid].wait()
        self.go[tid].clear()
        if self.dead:
            raise _AbandonedError()

    def point(self, tid: str) -> None:
        self._park(tid, AT_POINT)

    def blocked(self, tid: str) -> None:
        self._park(tid, BLOCKED)

    def released(self, tid: str) -> None:
        self._park(tid, RELEASE)

    def done(self, tid: str) -> None:
        self.reports.put((tid, DONE))

    # -- driver side ---------------------------------------------------------

    def prime(self) -> None:
        """Wait for every worker to stop at its first preemption point.

        Without this step the workers run freely from the start barrier to
        that first point. The case then no longer decides the interleaving.
        """
        for _ in range(len(self.state)):
            tid, kind = self.reports.get()
            self.state[tid] = kind
        self.primed = True

    def ready(self) -> list[str]:
        return sorted(tid for tid, s in self.state.items() if s == AT_POINT)

    def step(self, choice: int) -> bool:
        """Let one worker run to its next preemption point.

        Returns False when no worker can run.
        """
        ready = self.ready()
        if not ready:
            return False
        tid = ready[choice % len(ready)]
        self.trace.append(tid)
        self.state[tid] = "running"
        self.go[tid].set()
        who, kind = self.reports.get()   # exactly one worker is running
        if kind == RELEASE:
            for other, s in self.state.items():
                if s == BLOCKED:
                    self.state[other] = AT_POINT
            self.state[who] = AT_POINT
        else:
            self.state[who] = kind
        return True

    def finished(self) -> bool:
        return all(s == DONE for s in self.state.values())

    def deadlocked(self) -> bool:
        return not self.ready() and not self.finished()

    def shutdown(self) -> None:
        """Release every parked worker so the thread can exit."""
        self.dead = True
        for event in self.go.values():
            event.set()


class _Lock:
    """The allocator's lock, made visible to the driver."""

    def __init__(self, driver: _Driver) -> None:
        self.driver = driver
        self.owner: str | None = None

    def __enter__(self):
        tid = threading.current_thread().name
        while True:
            self.driver.point(tid)
            if self.owner is None:
                self.owner = tid
                return self
            # Held by another worker; wait for the release.
            self.driver.blocked(tid)

    def __exit__(self, *exc):
        self.owner = None
        self.driver.released(threading.current_thread().name)
        return False


class _FreeList:
    """The free list of the allocator. Every access is a preemption point.

    This class also records which worker holds each page that left the list.
    A worker can stop between its ``get`` and the return of the pages to its
    caller. The conservation check stays correct across that gap.
    """

    def __init__(self, inner: _queue.Queue, driver: _Driver) -> None:
        self._inner = inner
        self._driver = driver
        self.outstanding: dict[str, list[int]] = {}

    def _point(self) -> str:
        tid = threading.current_thread().name
        self._driver.point(tid)
        return tid

    def qsize(self) -> int:
        self._point()
        return self._inner.qsize()

    def get(self, *args, **kwargs) -> int:
        tid = self._point()
        if self._inner.qsize() == 0:
            raise _WouldBlockError()
        page = self._inner.get(*args, **kwargs)
        self.outstanding.setdefault(tid, []).append(page)
        return page

    def put(self, page: int, *args, **kwargs) -> None:
        self._point()
        for pages in self.outstanding.values():
            if page in pages:
                pages.remove(page)
                break
        self._inner.put(page, *args, **kwargs)

    @property
    def queue(self):
        return self._inner.queue

    def held(self) -> list[int]:
        return [page for pages in self.outstanding.values() for page in pages]


class AllocConcurrentMachine(StateMachine):
    name = "alloc_concurrent"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        num_threads = rng.choice([2, 2, 3])
        return {
            "max_pages": rng.choice([1, 2, 3]),
            "programs": [
                [
                    [rng.choice(["alloc", "alloc", "free"]), rng.randint(1, 2)]
                    for _ in range(rng.randint(1, 2))
                ]
                for _ in range(num_threads)
            ],
        }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.max_pages = config["max_pages"]
        programs = [[tuple(step) for step in p] for p in config["programs"]]
        self.tids = [f"t{i}" for i in range(len(programs))]

        self.alloc = PageAllocator(self.max_pages)
        self.driver = _Driver(self.tids)
        self.free_list = _FreeList(self.alloc.free_pages, self.driver)
        self.alloc.free_pages = self.free_list
        self.alloc._lock = _Lock(self.driver)

        self.errors: list[str] = []
        gate = threading.Barrier(len(self.tids) + 1)

        def body(tid: str, program) -> None:
            gate.wait()
            mine: list[int] = []
            try:
                self.driver.point(tid)      # park; the driver starts the case
                for kind, count in program:
                    if kind == "alloc":
                        pages = self.alloc.try_allocate(count)
                        if pages:
                            mine += pages
                    elif mine:
                        give, mine = mine[:count], mine[count:]
                        self.alloc.free(give)
            except _AbandonedError:
                return
            except _WouldBlockError:
                self.errors.append(
                    f"{tid} would block forever in the free list: it passed a "
                    "size check that another thread then invalidated"
                )
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{tid} raised {type(exc).__name__}: {exc}")
            finally:
                if not self.driver.dead:
                    self.driver.done(tid)

        self.threads = [
            threading.Thread(target=body, args=(tid, program), name=tid, daemon=True)
            for tid, program in zip(self.tids, programs, strict=True)
        ]
        for thread in self.threads:
            thread.start()
        gate.wait()
        self.driver.prime()
        # Release parked workers when the shrinker drops a case.
        self._cleanup = weakref.finalize(self, self.driver.shutdown)

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        return Op("step", (rng.randrange(3),))

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        if config["max_pages"] > 1:
            yield {**config, "max_pages": config["max_pages"] - 1}
        for index, program in enumerate(config["programs"]):
            if len(program) > 1:
                shorter = list(config["programs"])
                shorter[index] = program[:-1]
                yield {**config, "programs": shorter}
        if len(config["programs"]) > 2:
            yield {**config, "programs": config["programs"][:-1]}

    # -- execution -----------------------------------------------------------

    def execute(self, op: Op) -> None:
        if op.kind != "step":
            raise AssertionError(f"unknown op {op.kind}")
        self.driver.step(op.args[0])
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self.errors:
            require("alloc.thread_completes", False, self.errors[0])

    # -- invariants ----------------------------------------------------------

    def check(self) -> None:
        """These hold at every preemption point, not only at the end."""
        free = list(self.free_list.queue)
        held = self.free_list.held()

        require(
            "alloc.no_double_issue_across_threads",
            len(set(held)) == len(held),
            f"page(s) {sorted({p for p in held if held.count(p) > 1})} are held "
            "by two threads at once; both would write the same KV block",
        )
        require(
            "alloc.free_and_held_disjoint",
            not (set(free) & set(held)),
            f"page(s) {sorted(set(free) & set(held))} are on the free list and "
            "held by a thread at the same time",
        )
        require(
            "alloc.conservation",
            sorted(free + held) == list(range(self.max_pages)),
            f"pages leaked or multiplied mid-interleaving: free={sorted(free)} "
            f"held={sorted(held)} expected exactly 0..{self.max_pages - 1}",
        )

    def final_check(self) -> None:
        """Run the case to the end, then hold the allocator to its contract."""
        for _ in range(200):
            if self.driver.finished() or not self.driver.step(0):
                break
        self._raise_worker_error()
        require(
            "alloc.no_deadlock",
            not self.driver.deadlocked(),
            "every thread that has not finished is blocked on the lock; "
            f"state={self.driver.state}",
        )
        if not self.driver.finished():
            return      # the schedule simply ran out of steps; not a finding
        self.check()
        require(
            "alloc.quiesce_accounts_for_every_page",
            len(list(self.free_list.queue)) + len(self.free_list.held())
            == self.max_pages,
            "after every thread finished, the free list and the threads "
            f"together hold {len(list(self.free_list.queue))} + "
            f"{len(self.free_list.held())} of {self.max_pages} pages",
        )
