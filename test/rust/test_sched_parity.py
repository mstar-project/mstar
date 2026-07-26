"""Scheduler parity: the Rust MicroScheduler vs mstar's, driven with
identical ready-work and event sequences. mstar's scheduler reads live
manager/engine objects and pops queues itself; the Rust one takes a snapshot
and leaves pops to the caller — the harness bridges the two and asserts the
DECISIONS (node, walk, request set, order of batches) match at every call.
Skipped unless the ``mstar_rust`` extension is installed."""
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("torch")

from mstar_rust import MicroScheduler as RustScheduler

from mstar.engine.base import EngineType
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.micro_scheduler import MicroScheduler, SchedulingType

PRIORITY = {EngineType.KV_CACHE: 0, EngineType.STATELESS: 2}


class _Engine:
    def __init__(self, etype):
        self._etype = etype
        self.not_ready: set = set()  # (node, rid) pairs failing check_ready

    def engine_type(self):
        return self._etype

    def check_ready(self, node, rid, fwd_info):
        return (node, rid) not in self.not_ready


class _World:
    """One mutable ready-state driving BOTH schedulers."""

    def __init__(self, node_engines, leaders):
        self.engines = {n: _Engine(t) for n, t in node_engines.items()}
        self.leaders = set(leaders)
        # rid -> (node, walk) currently ready (one ready node per rid here)
        self.ready: dict[str, tuple[str, str]] = {}

        # ---- mstar-side stubs ----
        world = self

        class _Queue:
            def get_ready_node_names(self):
                return {rid: {nw[0]} for rid, nw in world.ready.items()}

            @property
            def per_request_queues(self):
                return {
                    rid: SimpleNamespace(ready_node_names={nw[0]})
                    for rid, nw in world.ready.items()
                }

            def pop_ready_nodes(self, rid, names):
                nw = world.ready.get(rid)
                if nw and nw[0] in names:
                    del world.ready[rid]
                    return [SimpleNamespace(name=nw[0])]
                return []

        queue = _Queue()
        self.wgm = SimpleNamespace(
            queues={"wg0": queue},
            per_request_info={},  # filled per rid below
            get_partition_for_node=lambda n: "p0",
            get_graph_walk=lambda rid, part: self.ready[rid][1],
            get_fwd_info=lambda rid, part: None,
            get_worker_graph_id_for_node=lambda rid, node, graph_walk=None: "wg0",
        )
        em = SimpleNamespace(
            node_to_engine=self.engines,
            get_engine=lambda name: self.engines[name],
        )
        self.py = MicroScheduler(
            em, sched_type=SchedulingType.ROUND_ROBIN,
            parallel_leader_nodes=self.leaders,
            max_consec_tp_follower_batches=1,
        )
        self.rs = RustScheduler("round_robin", 1)

    def use_priority(self):
        self.py.sched_type = SchedulingType.PRIORITY
        self.rs = RustScheduler("priority", 1)
        return self

    def add(self, rid, node, walk):
        self.ready[rid] = (node, walk)
        self.wgm.per_request_info[rid] = True

    def _snapshot(self):
        return [
            (
                node, walk, rid, "wg0",
                self.engines[node].check_ready(node, rid, None),
                PRIORITY.get(self.engines[node].engine_type(), 99),
                node in self.leaders,
            )
            for rid, (node, walk) in self.ready.items()
        ]

    def step(self, **kw):
        """One get_next_batch on both; assert identical decisions; apply the
        Rust decision's pops to the shared state (Python popped its own)."""
        now_ms = int(time.monotonic() * 1000)
        snapshot = self._snapshot()
        py_batch = self.py.get_next_batch(
            self.wgm,
            max_batch_size=kw.get("max_batch_size"),
            target_node_name=kw.get("target_node"),
            target_graph_walk=kw.get("target_walk"),
            exclude_target=kw.get("exclude_target"),
        )
        rs_batch = self.rs.get_next_batch(
            snapshot, now_ms,
            max_batch_size=kw.get("max_batch_size"),
            target_node=kw.get("target_node"),
            target_walk=kw.get("target_walk"),
            exclude_target=kw.get("exclude_target"),
        )
        if py_batch is None:
            assert rs_batch is None, f"rust scheduled {rs_batch}, python None"
            return None
        assert rs_batch is not None, f"python scheduled {py_batch}, rust None"
        node, walk, rids, _wgids, _tpf = rs_batch
        assert node == py_batch.node_name
        assert walk == py_batch.graph_walk
        assert sorted(rids) == sorted(py_batch.node_objects.keys())
        # Python's stub pop already removed its rids; mirror for Rust's view.
        for rid in rids:
            self.ready.pop(rid, None)
        return node, walk, sorted(rids)

    def hold(self, rids):
        self.py.hold_requests(rids)
        self.rs.hold_requests(rids, int(time.monotonic() * 1000))

    def tp_follow(self, node, walk, rids):
        self.py.register_tp_follow(
            ScheduleTPNode(node_name=node, graph_walk=walk, request_ids=rids))
        self.rs.register_tp_follow(node, walk, rids)


def test_round_robin_rotates():
    w = _World({"A": EngineType.KV_CACHE, "B": EngineType.KV_CACHE},
               leaders={"A", "B"})
    w.add("r1", "A", "w")
    w.add("r2", "B", "w")
    first = w.step()
    w.add(first[2][0], first[0], "w")  # re-ready the scheduled one
    second = w.step()
    assert first[0] != second[0], "round robin must rotate"


def test_priority_and_biggest_walk():
    w = _World({"KV": EngineType.KV_CACHE, "VOC": EngineType.STATELESS},
               leaders={"KV", "VOC"}).use_priority()
    w.add("r1", "KV", "decode")
    w.add("r2", "KV", "prefill")
    w.add("r3", "KV", "prefill")
    w.add("r4", "VOC", "decode")
    node, walk, rids = w.step()
    assert (node, walk) == ("KV", "prefill") and rids == ["r2", "r3"]


def test_engine_not_ready_is_skipped():
    w = _World({"A": EngineType.KV_CACHE}, leaders={"A"})
    w.add("r1", "A", "w")
    w.engines["A"].not_ready.add(("A", "r1"))
    assert w.step() is None
    w.engines["A"].not_ready.clear()
    assert w.step() is not None


def test_hold_backoff_expires():
    w = _World({"A": EngineType.KV_CACHE}, leaders={"A"})
    w.add("r1", "A", "w")
    w.hold(["r1"])
    assert w.step() is None
    time.sleep(0.06)  # HOLD_BACKOFF_SECONDS = 0.05
    assert w.step() is not None


def test_max_batch_and_exclude():
    w = _World({"A": EngineType.KV_CACHE}, leaders={"A"})
    for i in range(4):
        w.add(f"r{i}", "A", "w")
    node, walk, rids = w.step(max_batch_size=2)
    assert len(rids) == 2
    assert w.step(exclude_target=("A", "w")) is None


def test_tp_follow_order_and_fairness():
    w = _World({"T": EngineType.KV_CACHE, "B": EngineType.STATELESS},
               leaders={"B"})  # follower rank for T
    w.add("r1", "T", "w")
    w.add("r2", "B", "w")
    w.tp_follow("T", "w", ["r1"])
    node, _, _ = w.step()
    assert node == "T"  # the follow replays first
    w.add("r1", "T", "w")
    w.tp_follow("T", "w", ["r1"])
    node, _, _ = w.step()
    # consec cap hit and B is ready: fairness yields to B in both.
    assert node == "B"
    node, _, _ = w.step()
    assert node == "T"
