"""The Rust scheduler core must schedule exactly as the Python one.

Two identical worlds -- a Rust graph runtime with a Qwen3.5-shaped LLM node
(``prefill_text`` and ``decode`` under one combined walk, per-walk caps and a
row cap) beside a second node -- are driven by the same random script:
arrivals, scheduling passes of every kind (plain, a speculation's targeted
merge, a yield-away), completions, holds on an injected clock, deferred
removes, terminal check failures, and the fairness peek and backlog queries.
One world's ``MicroScheduler`` runs its Python core, the other's the
runtime's. After every operation everything observable is compared: the
batch, every engine check in order, the backlog in order, the round-robin
table, failures and admit errors, and the runtime's ready rows and in-flight
flags.
"""
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

pytest.importorskip("mstar_rust")

import torch  # noqa: E402

from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.base import ShardingConfig  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    FULL_ADMIT_NOT_READY,
    FULL_ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    FullAdmitOutcome,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop, TensorPointerInfo  # noqa: E402
from mstar.graph.runtime import rust as rust_runtime  # noqa: E402
from mstar.graph.runtime.base import ColumnarEdgeSpecs  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.worker import micro_scheduler as ms  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402
from mstar.worker.node_manager_utils import RequestStateManager  # noqa: E402
from mstar.worker.sched_core import RustSchedCore  # noqa: E402

ME, PART = "w0", "default"
LLM, ENC = "LLM", "enc"
PRE, DEC, ENCODE, MIXED = "prefill_text", "decode", "encode", "mixed"
COMBINED = {(LLM, PRE): MIXED, (LLM, DEC): MIXED}
SIGNAL = {PRE: "text_inputs", DEC: "text_inputs", ENCODE: "pixels"}
NODE_OF = {PRE: LLM, DEC: LLM, ENCODE: ENC}
WG_OF = {PRE: 0, DEC: 1, ENCODE: 2}


def _graphs():
    pre = GraphNode(
        name=LLM, input_names=["text_inputs"],
        outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text")],
    )
    dec = Loop(
        name="decode_loop",
        section=GraphNode(
            name=LLM, input_names=["text_inputs"],
            outputs=[
                GraphEdge(next_node=LLM, name="text_inputs"),
                GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
            ],
        ),
        max_iters=1 << 30, outputs=[],
    )
    enc = GraphNode(
        name=ENC, input_names=["pixels"],
        outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="emb", output_modality="text")],
    )
    return [
        WorkerGraph(section=pre, graph_walks={PRE}, ranks=[0], worker_graph_id=0),
        WorkerGraph(section=dec, graph_walks={DEC}, ranks=[0], worker_graph_id=1),
        WorkerGraph(section=enc, graph_walks={ENCODE}, ranks=[0], worker_graph_id=2),
    ]


class _Engine:
    """Caps per walk and check verdicts from a script both worlds share:
    a verdict is a function of the pass, the rid and how many times the rid
    was asked in that pass, so the same checks in the same order get the
    same answers."""

    def __init__(self, caps, row_caps, seed):
        self.caps, self.row_caps, self.seed = caps, row_caps, seed
        self.log: list[tuple[str, int]] = []
        self.pass_no = 0
        self._asked: dict[int, int] = {}

    def new_pass(self):
        self.pass_no += 1
        self._asked = {}

    def check_ready(self, node, rid, fwd_info, allow_reload=True):
        del fwd_info, allow_reload
        self.log.append((node, rid))
        nth = self._asked[rid] = self._asked.get(rid, 0) + 1
        roll = random.Random(hash((self.seed, self.pass_no, rid, nth))).random()
        if roll < 0.02:
            return FullAdmitOutcome(
                AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(f"{rid} is doomed")),
                "kv",
            )
        return FULL_ADMIT_NOT_READY if roll < 0.2 else FULL_ADMIT_OK

    def splits_by_capture_key(self, node, walk):
        return False

    def get_max_batch_size(self, node, walk):
        return self.caps.get((node, walk))

    def get_max_rows_in_combined(self, node, walk):
        return self.row_caps.get((node, walk))


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


class World:
    def __init__(self, rust_core: bool, caps, row_caps, seed, clock, leaders=(LLM, ENC)):
        book = RustTensorBookkeeping()
        self.store = TensorStore(bookkeeping=book)
        graphs = _graphs()
        self.rt = rust_runtime.RustGraphRuntime(
            my_worker_id=ME, my_worker_graphs=graphs,
            all_wg_ids_to_graph_walks={0: {PRE}, 1: {DEC}, 2: {ENCODE}},
            all_wg_ids_to_dyn_loops={0: set(), 1: {"decode_loop"}, 2: set()},
            all_wg_ids_to_nodes={0: {LLM}, 1: {LLM}, 2: {ENC}},
            node_to_partition={LLM: PART, ENC: PART},
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
            bookkeeping=book, combined_walk_of=dict(COMBINED),
        )
        self.engine = _Engine(caps, row_caps, seed)
        manager = SimpleNamespace(get_engine=lambda node: self.engine)
        self.sched = MicroScheduler(
            manager, parallel_leader_nodes=set(leaders), combined_walk_of=dict(COMBINED),
        )
        self.sched.runtime = self.rt
        if not rust_core:
            self.sched._core = self.sched._python_core()  # on the same runtime
        self.rs = RequestStateManager(node_to_partition={LLM: PART, ENC: PART})
        self.walk: dict[int, str] = {}
        self.uuid = 0
        self.clock = clock

    def admit(self, i: int, walk: str) -> int:
        rid = self.rt.add_request(
            request_id=f"r{i}", partition=PART, graph_walk=walk,
            partition_worker_graph_ids=[0, 1, 2],
            worker_graph_to_workers=ParallelList([0, 1, 2], [[ME]] * 3), loop_caps={},
        )
        self.rs.add_request(rid, CurrentForwardPassInfo(
            request_id=f"r{i}", graph_walk=walk, fwd_index=0, random_seed=0,
            max_tokens=1 << 30, rid_handle=rid, partition_name=PART,
        ))
        self.walk[rid] = walk
        return rid

    def make_ready(self, rids):
        block = ColumnarEdgeSpecs.empty()
        for rid in rids:
            self.uuid += 1
            info = TensorPointerInfo(
                dims=[4], dtype=torch.float16, nbytes=8, address=0, stride=(1,),
                uuid=self.uuid, source_session_id="h:1", source_entity=ME,
            )
            self.store.put_tensor(request_id=rid, uuid=self.uuid, tensor=torch.zeros(4), info=info)
            self.store.increment_ref(self.uuid, n=1)
            walk = self.walk[rid]
            block.add(rid, SIGNAL[walk], [self.uuid], False, next_node=NODE_OF[walk])
        self.rt.ingest_inputs_batch(block)

    def complete(self, batch, to_decode):
        """The step ran: its rows are idle again; some prefills move on to decode."""
        rids = list(batch.request_to_worker_graph)
        wgs = list(batch.request_to_worker_graph.values())
        self.rt.set_in_flight(batch.node_name, rids, wgs, False)
        freed = self.rt.cleanup_consumed_inputs(batch.node_name, rids, wgs)
        if freed.uuids:
            self.store.mark_forgotten(list(freed.uuids))
        for rid in rids:
            if rid in to_decode and self.walk[rid] == PRE:
                self.walk[rid] = DEC
                self.rt.set_walk(rid, PART, DEC)

    def remove(self, rid):
        """The worker's teardown of a request: the scheduler's state, then the
        runtime's (which holds the Rust core's backlog)."""
        self.sched.clear_rid(rid, f"r{rid}")
        self.rt.remove_request(rid)
        self.rs.remove_request(rid)
        del self.walk[rid]

    def fail_cleanup(self):
        """Failed rows leave the world, as the worker's teardown would."""
        for rid in sorted(self.sched.failed_rids):
            if rid in self.walk:
                self.remove(rid)
        self.sched.take_admit_errors()

    # -- observation ------------------------------------------------------
    def backlog(self):
        if isinstance(self.sched.core, RustSchedCore):
            return [tuple(b) for b in self.sched.core.state()[0]]
        return [
            (node, label, list(b.request_to_worker_graph),
             [b.walk_of(r) for r in b.request_to_worker_graph] if b.request_walks else [])
            for (node, label), b in self.sched.backlog.items()
        ]

    def rr(self):
        if isinstance(self.sched.core, RustSchedCore):
            _, rr, n = self.sched.core.state()
            return sorted(rr), n
        return sorted((n, w, v) for (n, w), v in self.sched.node_and_walk_to_last_batch_num.items()), \
            self.sched.batch_number

    def graph_state(self):
        ready = sorted((s.node_name, s.graph_walk, sorted(s.rids)) for s in self.rt.get_ready_nodes(set()))
        flags = sorted(
            (rid, walk) for rid, walk in self.walk.items()
            if self.rt.is_in_flight(NODE_OF[walk], WG_OF[walk], rid)
        )
        return ready, flags

    def observe(self):
        return dict(
            backlog=self.backlog(), rr=self.rr(), graph=self.graph_state(),
            failed=sorted(self.sched.failed_rids), held=sorted(self.sched.held_until),
            checks=list(self.engine.log),
        )


def _batch_view(b):
    if b is None:
        return None
    edges = {}
    for rid, sig, uuids, fin in b.input_edges.edge_tuples():
        edges.setdefault(rid, []).append((sig, list(uuids), fin))
    return (
        b.node_name, b.graph_walk, list(b.request_to_worker_graph.items()),
        dict(b.request_walks), list(b.output_signals),
        {w: list(s) for w, s in b.walk_output_signals.items()}, edges,
    )


CONFIGS = {
    # Qwen3.5-shaped: per-walk caps, the combined walk's own cap, a row cap on prefill
    "capped": ({(LLM, PRE): 4, (LLM, DEC): 8, (LLM, MIXED): 8, (ENC, ENCODE): 3},
               {(LLM, PRE): 2}),
    # uncapped decode, no combined cap: a mixed step takes the smallest present
    "loose": ({(LLM, PRE): 3, (LLM, DEC): None, (LLM, MIXED): None, (ENC, ENCODE): None}, {}),
    "tight": ({(LLM, PRE): 1, (LLM, DEC): 2, (LLM, MIXED): 2, (ENC, ENCODE): 1}, {(LLM, PRE): 1}),
}


# the nodes this worker initiates: all of them, or (a TP follower of LLM) the
# encoder only, with LLM's rows left to the leader
LEADERS = {"leader": (LLM, ENC), "follower": (ENC,)}


@pytest.mark.parametrize("leaders", sorted(LEADERS))
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("seed", range(12))
def test_both_cores_schedule_the_same(config, seed, leaders, monkeypatch):
    caps, row_caps = CONFIGS[config]
    clock = _Clock()
    monkeypatch.setattr(ms, "time", clock)
    worlds = [
        World(rust, caps, row_caps, seed, clock, leaders=LEADERS[leaders]) for rust in (False, True)
    ]
    rng = random.Random(seed)
    idle: set[int] = set()
    n_admitted = 0
    log: list[str] = []

    def both(name, fn):
        log.append(name)
        for w in worlds:
            w.engine.new_pass()
        outs = [fn(w) for w in worlds]
        views = [w.observe() for w in worlds]
        assert outs[0] == outs[1], f"{name}: {outs}\nops: {log}"
        for key in views[0]:
            assert views[0][key] == views[1][key], f"{name}: {key} differs\nops: {log}"
        return outs[0]

    for _step in range(60):
        op = rng.random()
        live = sorted(worlds[0].walk)
        if op < 0.2 or not live:
            walks = [rng.choice([PRE, PRE, DEC, ENCODE]) for _ in range(rng.randint(1, 12))]
            new = []
            for walk in walks:
                rids = {w.admit(n_admitted, walk) for w in worlds}
                assert len(rids) == 1
                n_admitted += 1
                new.append(rids.pop())
            both(f"arrive {new}", lambda w, new=new: w.make_ready(new))
        elif op < 0.3 and idle:
            again = sorted(rng.sample(sorted(idle), rng.randint(1, len(idle))))
            idle -= set(again)
            both(f"ready {again}", lambda w, again=again: w.make_ready(again))
        elif op < 0.75:
            kind = rng.random()
            if kind < 0.45:
                kwargs = {}
            elif kind < 0.8:
                walk = rng.choice([DEC, PRE])
                kwargs = dict(target=(LLM, walk), pre_existing_batch_size=rng.randint(0, 6))
            else:
                kwargs = dict(exclude_target=(LLM, rng.choice([DEC, PRE])))
            batch = both(
                f"get_next_batch {kwargs}",
                lambda w, kw=kwargs: _batch_view(w.sched.get_next_batch(w.rs, **kw)),
            )
            if batch is not None:
                rids = [r for r, _ in batch[2]]
                to_decode = {r for r in rids if rng.random() < 0.5}
                both("complete", lambda w, b=batch, td=to_decode: w.complete(
                    SimpleNamespace(node_name=b[0], request_to_worker_graph=dict(b[2])), td,
                ))
                idle |= set(rids)
            both("fail cleanup", lambda w: w.fail_cleanup())
            idle &= set(worlds[0].walk)
        elif op < 0.82 and live:
            rid = rng.choice(live)
            both(f"hold {rid}", lambda w, rid=rid: w.sched.hold_requests([rid]))
        elif op < 0.87 and live:
            rid = rng.choice(live)
            if rid in worlds[0].sched.pending_removes:
                both(f"undefer {rid}", lambda w, rid=rid: w.sched.pending_removes.pop(rid))
            else:
                both(f"defer {rid}", lambda w, rid=rid: w.sched.pending_removes.update({rid: None}))
        elif op < 0.90 and live:
            # a live request cancelled wherever it is: queued, backlogged or idle
            rid = rng.choice(live)
            idle.discard(rid)
            both(f"cancel {rid}", lambda w, rid=rid: w.remove(rid))
        elif op < 0.93:
            clock.t += 0.03
        else:
            both("peek", lambda w: w.sched.has_ready_excluding(w.rs, (LLM, DEC)))
            both("room", lambda w: (
                w.sched.room_for_continuing((LLM, DEC)), w.sched.room_for_continuing((LLM, PRE)),
                w.sched.room_for_continuing((ENC, ENCODE)),
            ))
            both("pending_rows", lambda w: dict(w.sched.pending_rows()))
            if live:
                rid = live[0]
                both("splits", lambda w, rid=rid: w.sched.backlog_splits_from(w.rs, (LLM, DEC), rid))
    assert isinstance(worlds[1].sched.core, RustSchedCore), "the Rust world must have run the Rust core"


def test_a_node_split_by_capture_key_keeps_the_python_core(monkeypatch):
    caps, row_caps = CONFIGS["capped"]
    clock = _Clock()
    monkeypatch.setattr(ms, "time", clock)
    w = World(True, caps, row_caps, 0, clock)
    w.engine.splits_by_capture_key = lambda node, walk: node == ENC
    assert w.sched._rust_core() is None
