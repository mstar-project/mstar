"""The KV resource under several threads, with the schedule in the case.

``kv_run`` drives the resource on one thread. This machine drives it on
several, because the worker does:

* the GPU thread runs the step cycle: ``admit``, ``plan``, the forward,
  ``commit``
* the main thread scans for ready work, which calls ``admit_retrieve``. It
  also ends requests, which calls ``reset_request`` and ``remove_request``.

The manager names where those threads meet. ``admit`` holds its lock for "one
critical section so the read-of-stored_len then alloc is atomic against a
concurrent reset/remove/commit on another thread". ``publish`` reads
``_streams`` outside the lock, and says why: "``remove_request`` can pop the
streams from another thread between the forward and finalize". This machine
drives those two pairs.

One thread runs at a time, and the case names which one. Thus an interleaving
replays, and a failure shrinks.

What this machine does not drive
--------------------------------

The pre-plan path is not a race on the CPU. ``_preplan_spec`` waits for the
``commit_done`` of batch N before it stages batch N+1 (``worker.py:1269``).
The GPU thread then waits for ``plan_future.result()`` before the forward
that reads the staged plan (``worker.py:1361``). Both ends hold a fence, so
no other resource call overlaps a staged step. ``kv_run`` drives the state
machine between the two fences, on one thread.

The fences leave one window open. ``engine.py:617`` releases ``commit_done``
before ``_collect_outputs``, so ``pre_plan(N+1)`` overlaps the per-request
tail of step N. Two attributes in that window carry no lock:
``_default_label`` and ``_default_layer_idx`` (``resources/base.py:246``).
``KVManager.plan`` resets them at its first line, and ``read_kv`` reads them
when its caller names no layer. No submodule reads the cache in that window
today. It is therefore a hazard with no reader, and not a fault. A lockset
pass reports such a hazard. A search over schedules does not, because no
thread takes the other side.

``admit_retrieve`` is the third pair that the manager names. ``commit`` holds
its lock to be "atomic against admit_retrieve reading stored_len on another
thread". This machine does not drive it. ``admit_retrieve`` does nothing
without a published info from another worker, and with one
``LocalOnlyKVTransferEngine`` refuses: "Cross-worker KV migration is
unavailable for this accelerator". That pair needs a transport that moves
pages between workers.

One thread plans. ``_current_plan_states`` holds the plan of a step, and the
GPU thread owns it. The plan thread writes ``_preplan_states`` instead. The
worker never makes two threads plan at the same time, so this machine does
not make that shape.

The GPU is not here. CUDA streams and events order the double-buffered slots,
``preplan_event``, and the copies inside ``offload``. A CPU sees none of that
order. Lane 1b covers it.

A measured limit
----------------

Take the mutual exclusion out of the manager lock, and keep every preemption
point. 400 cases then report nothing. The threads of this machine touch
separate streams, and the page arena holds a lock of its own that tier 0
covers. The pairs that would contend for one stream are the two above, and
neither one runs on a CPU.

Ops: run.

Invariants
----------
race.thread_completes      no thread raises and no thread blocks forever
race.no_deadlock           threads that have not finished are not all blocked
race.pages_are_not_shared  no page is held by two streams at any point
race.pages_are_conserved   the held pages and the free pages together are
                           every page, at every point
race.stream_holds_its_writes
                           a stream holds the tokens that the steps which
                           committed to it wrote, in the order they committed
race.quiesce_frees_every_page
                           once every thread is finished and every request is
                           removed, only the sink page is held

``tier0/alloc_concurrent`` holds the same driver, and drives the page
allocator with it. This machine drives the manager above that allocator. Its
oracle therefore reads what a page holds, and not only which thread holds it.
"""

from __future__ import annotations

import queue as _queue
import random
import threading
import weakref
from collections.abc import Iterator

import torch

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401  (quiets the per-admit warnings)
from fuzzer.tier1.resources import KV_KEY, WALK
from fuzzer.tier1.values import token_value
from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources import (
    KVConfig,
    KVSpec,
    KVStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.transfer import TransferEngineInfo

AT_POINT = "point"
BLOCKED = "blocked"
DONE = "done"
RELEASE = "release"
NODE = "n0"


class _AbandonedError(Exception):
    """Tells a worker to exit after the driver drops the case."""


class _Driver:
    """The only code that decides which worker runs next."""

    def __init__(self, tids: list[str]) -> None:
        self.tids = set(tids)
        self.state = {tid: AT_POINT for tid in tids}
        self.go = {tid: threading.Event() for tid in tids}
        self.reports: _queue.Queue = _queue.Queue()
        self.trace: list[str] = []
        self.dead = False

    # -- worker side ---------------------------------------------------------

    def owns(self, tid: str) -> bool:
        """Whether the driver schedules this thread.

        The machine reads the manager from the main thread, between the ops
        and inside the oracles. That thread is not part of the interleaving,
        so it must never park.
        """
        return tid in self.tids

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
        """Wait for every worker to stop at its first preemption point."""
        for _ in range(len(self.state)):
            tid, kind = self.reports.get()
            self.state[tid] = kind

    def ready(self) -> list[str]:
        return sorted(tid for tid, s in self.state.items() if s == AT_POINT)

    def step(self, choice: int) -> bool:
        """Let one worker run to its next preemption point."""
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
        self.dead = True
        for event in self.go.values():
            event.set()


class _Lock:
    """The manager's lock, made visible to the driver.

    Reentrant, because the manager's is: ``commit`` takes it and then calls
    ``_apply_fork``, which takes it again.
    """

    def __init__(self, driver: _Driver) -> None:
        self.driver = driver
        self.owner: str | None = None
        self.depth = 0

    def __enter__(self):
        tid = threading.current_thread().name
        if not self.driver.owns(tid):
            # A read of the machine itself. Take the lock, and make no
            # choice for the driver.
            self.owner = self.owner if self.depth else tid
            self.depth += 1
            return self
        while True:
            self.driver.point(tid)
            if self.owner in (None, tid):
                self.owner = tid
                self.depth += 1
                return self
            self.driver.blocked(tid)

    def __exit__(self, *exc):
        tid = threading.current_thread().name
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
        if not self.driver.owns(tid):
            return False
        if self.depth == 0:
            self.driver.released(tid)
        else:
            self.driver.point(tid)
        return False


def _instrument(manager, driver: _Driver) -> None:
    """Make the shared state of the manager visible to the driver.

    There are two kinds of preemption point:

    * the lock. Every thread that takes it, or gives it back, stops there.
    * the reads that ``plan`` makes outside the lock. ``_sequence_views``
      reads the ``stored_len`` of every stream of the step with no lock.
      ``commit`` on the other thread writes that same field.
    """
    manager._lock = _Lock(driver)

    def wrap(name: str):
        inner = getattr(manager, name)

        def wrapped(*args, **kwargs):
            tid = threading.current_thread().name
            if not driver.owns(tid):
                return inner(*args, **kwargs)
            driver.point(tid)
            out = inner(*args, **kwargs)
            driver.point(tid)
            return out

        setattr(manager, name, wrapped)

    for name in ("_sequence_views", "_setup_plan_states"):
        wrap(name)


class KvRaceMachine(StateMachine):
    name = "kv_race"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        num_requests = rng.randint(2, 3)
        # The GPU thread runs the step cycle. Every other thread is the main
        # thread, which scans for ready work and ends requests.
        #
        # No thread ends a request while a step of that request runs. The
        # drain protocol of the worker stops that shape: DRAIN_REQUEST, then
        # READS_DONE, then REMOVE_REQUEST. A machine that drove it would
        # report a shape that the worker does not make.
        #
        # ``publish`` does run against a request that is in a step. It reads
        # ``_streams`` outside the lock, and ``remove_request`` on another
        # thread can pop them.
        stepped = rng.randrange(num_requests)
        idle = [index for index in range(num_requests) if index != stepped]
        gpu = [["step", stepped] for _ in range(rng.randint(1, 3))]
        others = [
            [
                rng.choice([
                    # ``publish`` against the request that is in a step.
                    ["publish", stepped],
                    ["publish", rng.choice(idle)],
                    ["reset", rng.choice(idle)],
                    ["remove", rng.choice(idle)],
                ])
                for _ in range(rng.randint(1, 2))
            ]
            for _ in range(rng.randint(1, 2))
        ]
        return {
            "num_requests": num_requests,
            "max_pages": rng.choice([4, 6, 10]),
            "page_size": rng.choice([1, 2]),
            "num_layers": rng.randint(1, 2),
            "span": rng.randint(1, 2),
            "programs": [gpu, *others],
        }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.span = config["span"]
        self.num_layers = config["num_layers"]
        self.max_pages = config["max_pages"]
        self.request_ids = [f"r{i}" for i in range(config["num_requests"])]

        spec = KVSpec(
            resource_key=KV_KEY, nodes={NODE},
            config=KVConfig(
                num_layers=self.num_layers, num_kv_heads=1, head_dim=1,
                max_seq_len=256, max_num_pages=self.max_pages,
                page_size=config["page_size"],
            ),
        )
        self.kv = spec.resource_class.build(spec, EngineResourceInfo(
            device=torch.device("cpu"), kv_dtype=torch.int32,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id="fuzzer", my_session_id="fuzzer",
                transfer_engine=LocalTransferEngine("fuzzer"),
            ),
        ))
        self.runner = StepRunner({KV_KEY: self.kv}, node_resources={NODE: [KV_KEY]})
        for request_id in self.request_ids:
            self.runner.ingest_request(request_id)

        self.tids = [f"t{i}" for i in range(len(config["programs"]))]
        self.driver = _Driver(self.tids)
        _instrument(self.kv, self.driver)

        self.errors: list[str] = []
        # request -> the tokens that a committed step wrote, in commit order.
        self.written: dict[str, list[int]] = {rid: [] for rid in self.request_ids}
        # The requests that another thread reset or removed. What such a
        # request holds is no longer what its steps wrote.
        self.disturbed: set[str] = set()
        self.step_index = 0

        gate = threading.Barrier(len(self.tids) + 1)
        programs = [[tuple(op) for op in p] for p in config["programs"]]
        self.threads = [
            threading.Thread(
                target=self._body, args=(tid, program, gate),
                name=tid, daemon=True,
            )
            for tid, program in zip(self.tids, programs, strict=True)
        ]
        for thread in self.threads:
            thread.start()
        gate.wait()
        self.driver.prime()
        self._finished_the_case = False
        # A case that ends normally releases its threads in ``final_check``.
        # A case that ends on a failed invariant leaves them parked, and this
        # finalizer releases them when the machine goes.
        self._cleanup = weakref.finalize(self, self.driver.shutdown)

    # -- the threads ---------------------------------------------------------

    def _body(self, tid: str, program, gate: threading.Barrier) -> None:
        gate.wait()
        try:
            self.driver.point(tid)      # park; the driver starts the case
            for kind, index in program:
                request_id = self.request_ids[index % len(self.request_ids)]
                if kind == "step":
                    self._one_step(request_id)
                elif kind == "publish":
                    self.runner.publish([request_id], NODE)
                elif kind == "reset":
                    self.disturbed.add(request_id)
                    self.kv.reset_request(request_id, free=True)
                elif kind == "remove":
                    self.disturbed.add(request_id)
                    self.runner.remove_request(request_id)
        except _AbandonedError:
            return
        except Exception as exc:  # noqa: BLE001 - a raise is the finding
            self.errors.append(f"{tid} raised {type(exc).__name__}: {exc}")
        finally:
            if not self.driver.dead:
                self.driver.done(tid)

    def _one_step(self, request_id: str) -> None:
        """The cycle the GPU thread runs: admit, plan, the forward, commit."""
        context = StepContext(
            request_ids=(request_id,), graph_walk=WALK, slot=0, capture=False,
        )
        step = SubmoduleStep(steps={KV_KEY: KVStep(
            segments=(Segment(request_id, "main", self.span),),
        )})
        step.set_ctx(context)

        if not self.runner.admit(step).ok:
            return
        self.runner.plan(step)

        index = self.step_index
        self.step_index += 1
        values = [
            token_value(f"{request_id}#{index}", offset, 0)
            for offset in range(self.span)
        ]
        for layer in range(self.num_layers):
            keys = torch.tensor(values, dtype=torch.int32).view(-1, 1, 1)
            self.kv.write_kv(keys, keys + 1, layer_idx=layer, label="main")
        self.runner.commit(step)
        self.written[request_id].extend(values)

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        return Op("run", (rng.randrange(3),))

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        if config["max_pages"] > 2:
            yield {**config, "max_pages": config["max_pages"] - 1}
        if config["span"] > 1:
            yield {**config, "span": config["span"] - 1}
        if config["num_layers"] > 1:
            yield {**config, "num_layers": 1}
        for index, program in enumerate(config["programs"]):
            if len(program) > 1:
                shorter = list(config["programs"])
                shorter[index] = program[:-1]
                yield {**config, "programs": shorter}
        if len(config["programs"]) > 2:
            yield {**config, "programs": config["programs"][:-1]}

    # -- execution -----------------------------------------------------------

    def execute(self, op: Op) -> None:
        if op.kind != "run":
            raise AssertionError(f"unknown op {op.kind}")
        self.driver.step(op.args[0])
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self.errors:
            require("race.thread_completes", False, self.errors[0])

    # -- the oracles ---------------------------------------------------------

    def _pages(self) -> tuple[list[int], dict[tuple[str, str], list[int]]]:
        free = list(self.kv._arena.allocator.free_pages.queue)
        held = {
            (rid, label): list(stream.page_indices)
            for rid, streams in self.kv._streams.items()
            for label, stream in streams.items()
        }
        return free, held

    def check(self) -> None:
        """These hold at every preemption point, not only at the end."""
        free, held = self._pages()

        owner: dict[int, tuple[str, str]] = {}
        shared: list[str] = []
        for key, pages in held.items():
            for page in pages:
                if page in owner:
                    shared.append(f"page {page}: {owner[page]} and {key}")
                owner[page] = key
        require(
            "race.pages_are_not_shared",
            not shared,
            f"a page reached two streams mid-interleaving: {shared}; both "
            f"would write the same KV block",
        )
        both = sorted(set(free) & set(owner))
        require(
            "race.pages_are_conserved",
            not both and len(owner) + len(free) + 1 == self.max_pages,
            f"pages leaked or multiplied mid-interleaving: {len(owner)} held, "
            f"{len(free)} free, {len(both)} both, of {self.max_pages} with the "
            f"sink",
        )

    def final_check(self) -> None:
        """Run the case to the end, then hold the manager to its contract."""
        for _ in range(400):
            if self.driver.finished() or not self.driver.step(0):
                break
        self._raise_worker_error()
        self._finished_the_case = True
        require(
            "race.no_deadlock",
            not self.driver.deadlocked(),
            f"every thread that has not finished is blocked; "
            f"state={self.driver.state}",
        )
        if not self.driver.finished():
            return      # the schedule ran out of steps, which is not a fault
        self.check()

        # What a stream holds is what the steps that committed to it wrote.
        # A request that another thread reset or removed is left out. That
        # request lost its state on purpose.
        for request_id, values in self.written.items():
            if request_id in self.disturbed:
                continue
            stream = self.kv._streams.get(request_id, {}).get("main")
            got = [] if stream is None else self._read(stream, len(values))
            require(
                "race.stream_holds_its_writes",
                got == values,
                f"{request_id} holds {got}; the steps that committed to it "
                f"wrote {values}",
            )

        for request_id in self.request_ids:
            self.runner.remove_request(request_id)
        free, held = self._pages()
        try:
            require(
                "race.quiesce_frees_every_page",
                len(free) + 1 == self.max_pages and not any(held.values()),
                f"after every thread finished and every request went, "
                f"{len(free)} of {self.max_pages} pages are free and {held} is "
                f"still held",
            )
        finally:
            self._cleanup()

    def _read(self, stream, count: int) -> list[int]:
        """The numbers a stream holds, through the addressing of the cache."""
        if count <= 0 or not stream.page_indices:
            return []
        page_size = self.kv.config.page_size
        pages = torch.tensor(
            [stream.page_indices[offset // page_size] for offset in range(count)],
            dtype=torch.long,
        )
        slots = torch.tensor(
            [offset % page_size for offset in range(count)], dtype=torch.long,
        )
        got = self.kv.kv_cache.read_tokens(layer_idx=0, page_idx=pages, cache_idx=slots)
        return [int(value) for value in got[:, 0, 0, 0]]
