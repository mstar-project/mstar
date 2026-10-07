"""A free scan does not ask again what a pool has already answered "wait" to, until the pool moves.

The scheduler asks every waiting request on every free scan, and each ask goes through the
runner to the pool's gate, under its lock. With the park on (``MSTAR_SCHED_PARK``) a request
whose gate answered "wait" is skipped while ``admission_keys`` reads as it did when it was
asked: the same decisions, in the same order, at the same scans, with far fewer asks.

These count the asks a real ``MicroScheduler`` makes of a real ``KVManager`` behind a real
``StepRunner``, and compare a park on with a park off over random workloads.
"""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_admission_peak import (
    DECODE,
    NODE,
    PAGE_SIZE,
    PREFILL,
    ROOT,
    _ingest,
    _prefill,
    _request,
    _run,
    _StubTransfer,
)

from mstar.engine.engine import Engine
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import (
    ADMIT_OK,
    ADMIT_WAIT,
    ADMIT_WAIT_BEHIND,
    FULL_ADMIT_NOT_READY,
    FULL_ADMIT_OK,
    FULL_ADMIT_WAIT,
    FULL_ADMIT_WAIT_BEHIND,
    AdmitOutcome,
    AdmitRuntimeError,
    FullAdmitOutcome,
)
from mstar.graph.runtime.base import ColumnarEdgeSpecs, PopRidsOutput, ReadyNodeSpec
from mstar.utils.containers import ParallelList
from mstar.worker import micro_scheduler
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch

WINDOW = 3
# fit, order: the summed test in arrival order is the default; the rest plan
MODES = [("sum", "fifo"), ("sum", "backfill"), ("peak", "backfill"), ("peak", "fifo")]
SEEDS = [20261006, 7, 99]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    for name in (
        "MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_KV_BACKFILL_WINDOW",
        "MSTAR_STEP_TELEMETRY_DIR", "MSTAR_SCHED_PARK",
    ):
        monkeypatch.delenv(name, raising=False)


def _pool(fit, order, pages=32, window=WINDOW, rank=0, world_size=1) -> KVManager:
    group = SimpleNamespace(rank=rank, world_size=world_size) if world_size > 1 else None
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=pages, page_size=PAGE_SIZE,
            admission_fit=fit, admission_order=order, backfill_window=window,
        ),
        name="kv", joint_comm_group=group, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, {"main": (PREFILL, DECODE)})
    return kv


# ── the scheduler's side: a runtime that holds the ready rids, an engine over a real runner ──


class _NotReady(Resource):
    """A resource, not a pool, that is not ready for the rids it names: what a transfer still
    in flight is. Never an admission gate."""

    def __init__(self):
        self.rids: set = set()

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError("constructed directly")

    def admit_retrieve(self, rid, node_name, graph_walk, published):
        del node_name, graph_walk, published
        return AdmitOutcome(ok=True, ready=False) if rid in self.rids else ADMIT_OK


class _Engine:
    """``Engine.check_ready`` as the worker sees it, minus the offload branch, over a real
    ``StepRunner``; every ask is recorded."""

    def __init__(self, runner: StepRunner, keys: bool = True):
        self.runner = runner
        self.asks: list = []
        if keys:
            self.admission_keys = runner.admission_keys

    def get_max_batch_size(self, node_name, graph_walk):
        del node_name, graph_walk

    def capture_group(self, node_name, graph_walk, rid, fwd_info):
        del node_name, graph_walk, rid, fwd_info

    def check_ready(self, node_name, rid, fwd_info) -> FullAdmitOutcome:
        self.asks.append(rid)
        return self.runner.admit_retrieve(rid, node_name, fwd_info.graph_walk, None)


class _State:
    """The ``RequestStateManager`` and the graph runtime, for one node and one walk: the rids
    that are ready at graph level, in the order they became so."""

    def __init__(self):
        self.ready: dict = {}
        self.runtime = self

    def get_partition_for_node(self, node_name):
        del node_name
        return "default"

    def get_fwd_info(self, rid, partition):
        del rid, partition
        return SimpleNamespace(graph_walk=PREFILL)

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        del target
        if exclude_target == (NODE, PREFILL):
            return []
        rids = [rid for rid in self.ready if rid not in exclude_rids]
        return [ReadyNodeSpec(NODE, PREFILL, rids)] if rids else []

    def has_ready_excluding(self, exclude_rids, exclude_target=None):
        return bool(self.get_ready_nodes(exclude_rids, exclude_target=exclude_target))

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        del node_name, graph_walk
        return 0

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        del node_name, graph_walk
        if check_ready and any(rid not in self.ready for rid in request_ids):
            return None
        popped = [rid for rid in request_ids if self.ready.pop(rid, 0) is None]
        return PopRidsOutput(
            wg_ids=ParallelList(popped, [0] * len(popped)),
            input_edges=ColumnarEdgeSpecs.empty(),
        )


class _World:
    """A scheduler, scanning a real pool, with requests that arrive, run, grow and end."""

    def __init__(
        self, fit="sum", order="fifo", pages=32, park=True, window=WINDOW, extra=None, lazy=False,
    ):
        """``lazy``: a request the scan lets in holds its reservation and no page until the
        test starts it (``start``), as it does while its prefill waits to run."""
        self.kv = _pool(fit, order, pages, window)
        resources = {"kv": self.kv, **(extra or {})}
        self.runner = StepRunner(resources, node_resources={NODE: list(resources)})
        self.engine = _Engine(self.runner)
        self.sched = MicroScheduler(
            engine_manager=SimpleNamespace(get_engine=lambda name: self.engine),
            parallel_leader_nodes={NODE},
        )
        self.sched._park = park
        self.state = _State()
        self.sched.runtime = self.state
        self.running: dict[str, int] = {}
        self.prompts: dict[str, int] = {}
        self.lazy = lazy
        self.unstarted: list[str] = []

    # requests

    def arrive(self, rid, prompt_pages=2, decode_pages=18, tokens=None):
        if tokens is None:
            _ingest(self.kv, rid, prompt_pages * PAGE_SIZE, decode_pages * PAGE_SIZE)
            self.prompts[rid] = prompt_pages * PAGE_SIZE
        else:
            self.kv.ingest_request(rid, _request(tokens, decode_pages * PAGE_SIZE))
            self.prompts[rid] = len(tokens)
        self.state.ready[rid] = None

    def abort(self, rid):
        self.state.ready.pop(rid, None)
        self.running.pop(rid, None)
        if rid in self.unstarted:
            self.unstarted.remove(rid)
        self.kv.remove_request(rid)
        self.sched.clear_rid(rid, rid)

    def finish(self, rid):
        self.running.pop(rid)
        self.kv.remove_request(rid)
        self.sched.clear_rid(rid, rid)

    def grant(self, rid):
        """One more page, as decode asks for one every ``PAGE_SIZE`` tokens."""
        return _run(self.kv, {rid: PAGE_SIZE})

    def start(self, rid) -> bool:
        """Run its prefill. A lazy world's can be deferred by the pool, and is tried again."""
        if not _prefill(self.kv, rid, self.prompts[rid]).ok:
            assert self.lazy, f"{rid} was let in and its first pages were refused"
            return False
        if rid in self.unstarted:
            self.unstarted.remove(rid)
        self.running[rid] = self.prompts[rid]
        return True

    # scans

    def scan(self):
        """One free scan, its admitted requests run: who it let in, in order, and whom it asked."""
        before = len(self.engine.asks)
        batch = self.sched.get_next_batch(self.state)
        admitted = [] if batch is None else list(batch.request_to_worker_graph)
        for rid in admitted:
            if self.lazy:
                self.unstarted.append(rid)
            else:
                self.start(rid)
        return admitted, self.engine.asks[before:]

    def waiting(self):
        return list(self.state.ready)

    def front(self):
        """The waiting requests the pool answers on its own state: the head, and the window."""
        return self.waiting()[:WINDOW if self.kv._backfill else 1]


def _pressed(fit, order, n=8, park=True, **kwargs):
    """A pool of 31 pages with two requests running on 30 of them, and ``n`` waiting that
    need 20 each: none fits, by the sum or by the peak. Freeing the larger runner lets exactly
    one in; freeing the smaller one lets none."""
    world = _World(fit, order, park=park, **kwargs)
    world.arrive("run", 11, 11)
    world.arrive("small", 4, 4)
    admitted, _ = world.scan()
    assert admitted == ["run", "small"]
    for i in range(n):
        world.arrive(f"w{i}")
    admitted, asks = world.scan()
    assert admitted == [] and asks == [f"w{i}" for i in range(n)]
    return world


# ── what the sentinels are ──────────────────────────────────────────────


def test_a_gate_wait_is_equal_to_any_wait_and_is_not_it():
    """A caller that reads ``ok`` and ``ready`` sees no difference: only identity says it was a gate."""
    plain = AdmitOutcome(ok=True, ready=False)
    for gate in (ADMIT_WAIT, ADMIT_WAIT_BEHIND):
        assert gate == plain and gate is not plain
        assert gate.ok and not gate.ready and gate.reason is None
    assert ADMIT_WAIT is not ADMIT_WAIT_BEHIND
    for full in (FULL_ADMIT_WAIT, FULL_ADMIT_WAIT_BEHIND):
        assert full == FULL_ADMIT_NOT_READY and full is not FULL_ADMIT_NOT_READY
        assert full.ok and not full.ready and full.failed_resource is None


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_gate_says_which_wait_it_was(fit, order):
    world = _pressed(fit, order, n=6)
    kv = world.kv
    for rid in world.waiting():
        answer = kv.admit_retrieve(rid, NODE, PREFILL, None)
        if rid in world.front():
            assert answer is ADMIT_WAIT, f"{rid} is the head or in the window: it waits on the pool"
        else:
            assert answer is ADMIT_WAIT_BEHIND, f"{rid} is behind the front of the queue"


def test_the_runner_answers_for_a_step_held_back_only_by_gates():
    kv = _pool("sum", "fifo")
    other = _NotReady()
    runner = StepRunner({"kv": kv, "other": other}, node_resources={NODE: ["kv", "other"]})
    _ingest(kv, "run", 22 * PAGE_SIZE, 8 * PAGE_SIZE)
    assert runner.admit_retrieve("run", NODE, PREFILL, None) is FULL_ADMIT_OK
    for rid in ("a", "b"):
        _ingest(kv, rid, 2 * PAGE_SIZE, 18 * PAGE_SIZE)

    assert runner.admit_retrieve("a", NODE, PREFILL, None) is FULL_ADMIT_WAIT
    assert runner.admit_retrieve("b", NODE, PREFILL, None) is FULL_ADMIT_WAIT_BEHIND
    # anything else that is not ready makes the answer today's, which is never parked
    other.rids = {"a", "b"}
    assert runner.admit_retrieve("a", NODE, PREFILL, None) is FULL_ADMIT_NOT_READY
    assert runner.admit_retrieve("b", NODE, PREFILL, None) is FULL_ADMIT_NOT_READY
    # and once the gate lets a request in, it is that other resource that holds it
    kv.remove_request("run")
    assert runner.admit_retrieve("a", NODE, PREFILL, None) is FULL_ADMIT_NOT_READY
    other.rids = set()
    assert runner.admit_retrieve("b", NODE, PREFILL, None) is FULL_ADMIT_WAIT, "b heads the queue now"


def test_a_step_with_a_wait_and_a_failure_is_the_failure():
    class Doomed(Resource):
        @classmethod
        def build(cls, spec, info):
            raise NotImplementedError("constructed directly")

        def admit_retrieve(self, rid, node_name, graph_walk, published):
            del node_name, graph_walk, published
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(f"{rid} is doomed"))

    kv = _pool("sum", "fifo")
    runner = StepRunner({"kv": kv, "doomed": Doomed()}, node_resources={NODE: ["kv", "doomed"]})
    _ingest(kv, "run", 22 * PAGE_SIZE, 8 * PAGE_SIZE)
    assert runner.admit_retrieve("run", NODE, PREFILL, None).failed_resource == "doomed"
    _ingest(kv, "a", 2 * PAGE_SIZE, 18 * PAGE_SIZE)
    answer = runner.admit_retrieve("a", NODE, PREFILL, None)
    assert answer.failed_resource == "doomed" and isinstance(answer.reason, AdmitRuntimeError)


def test_the_runner_keys_come_from_the_pools_that_decide():
    kv = _pool("sum", "fifo")
    follower = _pool("sum", "fifo", rank=1, world_size=2)
    plain = _NotReady()
    only = StepRunner({"kv": kv}, node_resources={NODE: ["kv"]})
    assert only.admission_keys(NODE) == kv.admission_keys()
    assert only.admission_keys("elsewhere") == kv.admission_keys()
    nothing = StepRunner({"kv": follower, "plain": plain}, node_resources={NODE: ["kv", "plain"]})
    assert nothing.admission_keys(NODE) is None, "a pool that does not decide has no key to park on"
    both = StepRunner(
        {"kv": kv, "kv2": _pool("sum", "fifo"), "kv3": follower},
        node_resources={NODE: ["kv", "kv2", "kv3"]},
    )
    behind, front = both.admission_keys(NODE)
    assert len(behind) == len(front) == 2, "one key from each pool that decides"


def test_the_engine_hands_on_the_runners_keys():
    kv = _pool("sum", "fifo")
    runner = StepRunner({"kv": kv}, node_resources={NODE: ["kv"]})
    assert Engine.admission_keys(SimpleNamespace(_runner=runner), NODE) == kv.admission_keys()


# ── the key ─────────────────────────────────────────────────────────────


def test_a_pool_that_does_not_decide_has_no_keys():
    assert _pool("sum", "fifo", rank=1, world_size=2).admission_keys() is None
    assert _pool("sum", "fifo", rank=0, world_size=2).admission_keys() is not None


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_queue_epoch_ticks_as_a_request_leaves_the_queue_and_not_as_one_arrives(fit, order):
    world = _pressed(fit, order, n=0)
    kv = world.kv

    def epoch():
        return kv.admission_keys()[0]

    at = epoch()
    for rid in ("a", "b", "c"):
        world.arrive(rid, 2, 18)
        assert kv.admit_retrieve(rid, NODE, PREFILL, None).ready is False
    assert epoch() == at, "a request that arrives goes last, and moves no one"
    kv.admit_retrieve("a", NODE, PREFILL, None)
    assert epoch() == at, "asking again moves nothing"
    kv.remove_request("c")
    assert epoch() == at + 1, "a request behind the others left the queue"
    kv.remove_request("c")
    kv.remove_request("nobody")
    assert epoch() == at + 1, "one that was not waiting did not"
    world.finish("run")
    world.finish("small")
    assert kv.admit_retrieve("a", NODE, PREFILL, None) is ADMIT_OK
    assert epoch() == at + 2, "the head was admitted"


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_front_key_moves_with_the_pool_and_the_behind_key_does_not(fit, order):
    world = _pressed(fit, order, n=2)
    kv = world.kv
    behind, front = kv.admission_keys()

    assert world.grant("run").ok
    assert kv.admission_keys()[1] != front, "a page was granted"
    assert kv.admission_keys()[0] == behind, "which moves no request behind the front"
    behind, front = kv.admission_keys()
    world.finish("small")
    assert kv.admission_keys()[1] != front, "pages were released"
    assert kv.admission_keys()[0] == behind


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_reservation_given_back_before_it_took_a_page_moves_the_front_key(fit, order):
    """What it reserved is taken from no one now, though no page is granted and no one leaves
    the queue. (`remove_request` also releases an empty list of pages, which counts as an
    owner change in the arena, so the front moves there with or without the reserved epoch;
    the reservation made in the next test is the one that only the epoch carries.)"""
    kv = _pool(fit, order)
    _ingest(kv, "r", 2 * PAGE_SIZE, 2 * PAGE_SIZE)
    assert kv.admit_retrieve("r", NODE, PREFILL, None) is ADMIT_OK
    _, (reserved, free, _, queue) = kv.admission_keys()

    kv.remove_request("r")

    now_behind, (now_reserved, now_free, _, now_queue) = kv.admission_keys()
    assert (now_free, now_queue) == (free, queue)
    assert now_reserved != reserved


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_reservation_made_by_a_request_that_never_waited_moves_the_front_key(fit, order):
    """What admit reserves for a request that did not pass this pool's readiness (rank 0 at
    TP, sent a step) takes from what the others may reserve, and moves no page, no owner and
    no queue: the reserved epoch alone says so."""
    kv = _pool(fit, order, rank=0, world_size=2)
    _ingest(kv, "r", 2 * PAGE_SIZE, 2 * PAGE_SIZE)
    before = kv.admission_keys()

    assert kv._reserve_new(SimpleNamespace(request_ids=["r"], graph_walk=PREFILL)) is None

    behind, (reserved, free, owners, queue) = before
    now_behind, (now_reserved, now_free, now_owners, now_queue) = kv.admission_keys()
    assert (now_free, now_owners, now_queue, now_behind) == (free, owners, queue, behind)
    assert now_reserved != reserved


@pytest.mark.parametrize(("fit", "order"), [m for m in MODES if m[1] == "backfill"])
def test_a_request_refused_for_good_in_the_window_moves_the_front_key_by_the_queue_alone(fit, order):
    """The refused leaves the queue and is not yet removed (nothing is reserved or released),
    so the window slid and the epoch is all that says so."""
    world = _pressed(fit, order, n=1)
    kv = world.kv
    world.arrive("huge", 40, 40)
    before = kv.admission_keys()

    outcome = kv.admit_retrieve("huge", NODE, PREFILL, None)

    assert outcome.ok is False and "huge" not in kv._waiting
    behind, (reserved, free, owners, queue) = before
    now_behind, (now_reserved, now_free, now_owners, now_queue) = kv.admission_keys()
    assert (now_reserved, now_free, now_owners) == (reserved, free, owners)
    assert now_queue == queue + 1 and now_behind == behind + 1


# ── what a scan asks ────────────────────────────────────────────────────


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_parked_waiter_is_not_asked_again_until_the_key_moves(fit, order):
    world = _pressed(fit, order)

    for _ in range(3):
        admitted, asks = world.scan()
        assert admitted == [] and asks == []


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_park_off_asks_every_waiter_every_scan(fit, order):
    world = _pressed(fit, order, park=False)

    for _ in range(3):
        _, asks = world.scan()
        assert asks == world.waiting()


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_page_granted_asks_the_front_again_and_no_one_behind_it(fit, order):
    world = _pressed(fit, order)

    assert world.grant("run").ok
    admitted, asks = world.scan()

    assert admitted == [] and asks == world.front()
    assert world.scan() == ([], [])


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_release_asks_the_front_again(fit, order):
    world = _pressed(fit, order)

    world.finish("small")
    admitted, asks = world.scan()

    assert admitted == [] and asks == world.front()
    assert world.scan() == ([], [])


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_an_admission_asks_what_is_behind_it_again(fit, order):
    world = _pressed(fit, order)
    waiting = world.waiting()

    world.finish("run")
    world.finish("small")
    admitted, asks = world.scan()

    assert admitted == ["w0"], "room for the head and no more"
    assert asks == waiting, "the head was let in, and everything behind it moved up"
    # the pages w0 takes as it runs move the pool, which only the front of the queue asks of
    assert world.scan() == ([], world.front()), "each was asked after the admission, so is parked on it"
    assert world.scan() == ([], [])


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_waiter_that_is_aborted_asks_the_others_again(fit, order):
    world = _pressed(fit, order)

    world.abort("w5")
    admitted, asks = world.scan()

    assert admitted == [] and asks == world.waiting()
    assert world.scan() == ([], [])


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_reservation_given_back_before_it_took_a_page_lets_the_head_in(fit, order):
    world = _World(fit, order, lazy=True)
    world.arrive("hold", 11, 11)
    assert world.scan()[0] == ["hold"], "reserved, and prefill has not run: no page is held"
    world.arrive("w0")
    world.arrive("w1")
    admitted, asks = world.scan()
    assert admitted == [] and asks == ["w0", "w1"]
    assert world.scan() == ([], [])

    world.abort("hold")
    admitted, asks = world.scan()

    assert admitted == ["w0"], "the pool is as empty as it was, and w0 fits"
    assert asks[0] == "w0"


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_request_that_becomes_head_is_asked_and_let_in_when_it_fits(fit, order):
    world = _pressed(fit, order)
    world.finish("run")
    world.finish("small")
    world.abort("w0")

    admitted, asks = world.scan()

    assert admitted == ["w1"], "the next in line was not let in when the head left"
    assert asks[0] == "w1"


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_request_that_becomes_head_is_asked_when_the_head_is_let_in(fit, order):
    world = _pressed(fit, order, n=4)
    world.finish("run")
    world.finish("small")
    world.scan()
    # w1 heads the queue and does not fit; it is parked on the pool as the head
    assert "w1" in world.waiting()
    world.sched._parked.clear()
    world.scan()

    assert world.grant("w0").ok
    admitted, asks = world.scan()

    assert admitted == [] and asks == world.front()
    assert asks[0] == "w1"


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_first_ask_is_never_skipped(fit, order):
    world = _pressed(fit, order)

    world.arrive("late")
    admitted, asks = world.scan()

    assert admitted == [] and asks == ["late"], "an arrival moves no one, and is asked once"
    assert world.waiting()[-1] == "late"
    assert world.scan() == ([], [])


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_request_asked_again_after_it_left_and_came_back_is_asked(fit, order):
    world = _pressed(fit, order, n=2)
    world.abort("w1")
    world.arrive("w1")

    admitted, asks = world.scan()

    assert admitted == [] and "w1" in asks
    assert "w1" in world.sched._parked


def test_the_engine_that_has_no_keys_is_never_parked():
    world = _pressed("sum", "fifo")
    world.engine = _Engine(world.runner, keys=False)
    world.sched.engine_manager = SimpleNamespace(get_engine=lambda name: world.engine)
    world.sched._parked.clear()

    for _ in range(3):
        _, asks = world.scan()
        assert asks == world.waiting()
    assert not world.sched._parked


# ── what is not parked ──────────────────────────────────────────────────


def test_a_request_held_back_by_something_other_than_a_gate_is_never_parked():
    other = _NotReady()
    world = _World("sum", "fifo", extra={"other": other})
    world.arrive("run", 22, 8)
    assert world.scan()[0] == ["run"]
    world.arrive("a")
    world.arrive("b")
    other.rids = {"a", "b"}

    for _ in range(3):
        admitted, asks = world.scan()
        assert admitted == [] and asks == ["a", "b"]
    assert not world.sched._parked

    other.rids = set()
    world.finish("run")
    admitted, asks = world.scan()
    assert admitted == ["a"] and asks == ["a", "b"]


def test_a_request_the_gate_let_in_but_another_resource_holds_is_asked_every_scan():
    other = _NotReady()
    world = _World("sum", "fifo", extra={"other": other})
    world.arrive("a", 2, 2)
    other.rids = {"a"}

    for _ in range(3):
        admitted, asks = world.scan()
        assert admitted == [] and asks == ["a"]
    assert "a" in world.kv._reserved and not world.sched._parked


def test_a_request_that_became_ready_is_no_longer_parked():
    world = _pressed("sum", "fifo", n=2)
    assert set(world.sched._parked) == {"w0", "w1"}
    world.finish("run")
    world.finish("small")

    assert world.scan()[0] == ["w0"]
    assert "w0" not in world.sched._parked


def test_a_request_the_pool_cannot_ever_serve_is_failed_not_parked():
    world = _World("sum", "fifo")
    world.arrive("huge", 40, 40)

    admitted, asks = world.scan()

    assert admitted == [] and asks == ["huge"]
    assert "huge" in world.sched.take_admit_errors()
    assert "huge" in world.sched.failed_rids and "huge" not in world.sched._parked


# ── the other scans never skip ──────────────────────────────────────────


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_tp_follow_asks_every_time(fit, order):
    world = _pressed(fit, order, n=4)
    before = len(world.engine.asks)

    for _ in range(3):
        assert world.sched.pop_ready_rids(world.state, NODE, PREFILL, ["w0", "w3"]) is None

    assert world.engine.asks[before:] == ["w0"] * 3, "the first not-ready rid stops the set"
    assert world.scan() == ([], []), "and the free scan still has them parked"
    assert world.sched.pop_ready_rids(world.state, NODE, PREFILL, ["w3"]) is None
    assert world.engine.asks[-1] == "w3"


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_backlogged_chunk_is_asked_every_time(fit, order):
    world = _pressed(fit, order, n=4)
    for _ in range(3):
        world.sched.backlog[(NODE, PREFILL)] = ScheduledBatch(
            node_name=NODE, graph_walk=PREFILL,
            request_to_worker_graph={"w0": 0, "w2": 0}, input_edges=ColumnarEdgeSpecs.empty(),
        )
        before = len(world.engine.asks)
        batch = world.sched.get_next_batch(world.state)
        assert batch is None
        assert world.engine.asks[before:] == ["w0", "w2"], "the chunk is checked, parked or not"
        world.sched.backlog.clear()


@pytest.mark.parametrize(("fit", "order"), MODES)
def test_has_ready_excluding_parks_and_skips_like_the_free_scan(fit, order):
    world = _World(fit, order)
    world.arrive("run", 11, 11)
    world.arrive("small", 4, 4)
    assert world.scan()[0] == ["run", "small"]
    names = [f"w{i}" for i in range(6)]
    for rid in names:
        world.arrive(rid)

    before = len(world.engine.asks)
    assert world.sched.has_ready_excluding(world.state, None) is False
    assert world.engine.asks[before:] == names
    assert world.sched.has_ready_excluding(world.state, None) is False
    assert len(world.engine.asks) == before + len(names), "parked by the first peek"
    assert world.scan() == ([], []), "which the free scan reads as its own"

    world.finish("run")
    world.finish("small")
    assert world.sched.has_ready_excluding(world.state, None) is True
    assert world.engine.asks[-1] == "w0", "and a peek that finds one stops there"


def test_clearing_and_failing_a_rid_forget_it():
    world = _pressed("sum", "fifo", n=3)
    assert set(world.sched._parked) == {"w0", "w1", "w2"}

    world.sched.fail_rids({"w1"})
    assert set(world.sched._parked) == {"w0", "w2"}
    world.sched.clear_rid("w0", "w0")
    assert set(world.sched._parked) == {"w2"}


def test_the_park_is_read_from_the_environment_once(monkeypatch):
    monkeypatch.setattr(micro_scheduler, "_SCHED_PARK", True)
    on = MicroScheduler(engine_manager=None)
    monkeypatch.setattr(micro_scheduler, "_SCHED_PARK", False)
    off = MicroScheduler(engine_manager=None)

    assert on._park is True and off._park is False


# ── the same decisions ──────────────────────────────────────────────────


def _workload(world: _World, seed: int, steps: int, audit=None) -> list[tuple[int, tuple[str, ...]]]:
    """Requests arriving, some sharing a prefix; admitted ones growing a page at a time,
    ending, and being aborted while they wait (or, in a lazy world, while they hold a
    reservation and no page); one free scan per step. Returns who each scan
    let in. Every choice is drawn the same, so two worlds that decide alike get the same run.
    ``audit`` is called with the world just before each scan."""
    rng = random.Random(seed)
    shared = list(range(10**8, 10**8 + 2 * PAGE_SIZE))
    fresh = iter(range(0, 10**8, 10**4))
    life: dict[str, int] = {}
    made = 0
    log = []
    for step in range(steps):
        for _ in range(rng.choice([0, 0, 1, 1, 2, 4])):
            rid = f"q{made}"
            made += 1
            prompt_pages = rng.randint(1, 6)
            decode_pages = rng.randint(1, 10)
            tokens = None
            if rng.random() < 0.3:
                base = next(fresh)
                tokens = shared + list(range(base, base + max(0, prompt_pages - 2) * PAGE_SIZE))
            world.arrive(rid, prompt_pages, decode_pages, tokens)
            life[rid] = decode_pages
        for rid in list(world.unstarted):
            roll = rng.random()
            if roll < 0.5:
                world.start(rid)
            elif roll < 0.65:
                world.abort(rid)
        for rid in sorted(world.running):
            if life[rid] > 0 and rng.random() < 0.5 and world.grant(rid).ok:
                life[rid] -= 1
        for rid in sorted(world.running):
            if rng.random() < (0.5 if life[rid] == 0 else 0.03):
                world.finish(rid)
        waiting = world.waiting()
        if waiting and rng.random() < 0.08:
            world.abort(rng.choice(waiting))
        if audit is not None:
            audit(world)
        admitted, _ = world.scan()
        log.append((step, tuple(admitted)))
    return log


def _audit_parked(world: _World, audited: list) -> None:
    """Every request the next scan would skip is still a wait when the pool is asked afresh:
    what it keeps (refusals, the plan, the room) dropped, so nothing is answered from it."""
    kv = world.kv
    park = world.sched._park_scan(ReadyNodeSpec(NODE, PREFILL, world.waiting()))
    for rid in world.waiting():
        if not park.holds(rid):
            continue
        tier = world.sched._parked[rid][2]
        kv._refused.clear()
        kv._plan = kv._shadow = kv._room_at = kv._owed = None
        answer = kv.admit_retrieve(rid, NODE, PREFILL, None)
        assert answer is (ADMIT_WAIT if tier else ADMIT_WAIT_BEHIND), (
            f"{rid}, parked on tier {tier} at {world.sched._parked[rid][3]}, is "
            f"{'let in' if answer is ADMIT_OK else 'answered another wait'} asked afresh"
        )
        audited.append(rid)


@pytest.mark.parametrize("lazy", [False, True], ids=["started", "lazy"])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("fit", "order"), MODES)
def test_the_park_admits_the_same_requests_at_the_same_scans(fit, order, seed, lazy, monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    runs = {}
    for park in (False, True):
        world = _World(fit, order, pages=48, park=park, lazy=lazy)
        log = _workload(world, seed, steps=220)
        runs[park] = (log, len(world.engine.asks))
        world.kv.assert_pages_conserved()

    (off_log, off_asks), (on_log, on_asks) = runs[False], runs[True]
    assert on_log == off_log, "the park changed who was admitted, or when"
    assert sum(len(admitted) for _, admitted in off_log) > 20, "the workload admitted next to nothing"
    assert on_asks < off_asks, "the park asked no less"


@pytest.mark.parametrize("lazy", [False, True], ids=["started", "lazy"])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("fit", "order"), MODES)
def test_a_request_the_park_would_skip_is_still_a_wait_asked_afresh(
    fit, order, seed, lazy, monkeypatch,
):
    """The key says all that a wait depends on: no state the gate reads moves without it."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    world = _World(fit, order, pages=48, lazy=lazy)
    audited: list = []

    log = _workload(world, seed, steps=220, audit=lambda w: _audit_parked(w, audited))

    assert sum(len(admitted) for _, admitted in log) > 20
    assert len(audited) > 200, "the audit checked next to nothing"
