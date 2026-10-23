"""CPU microbenchmark of one ``MicroScheduler.get_next_batch`` pass, Python
core against the Rust graph runtime's core.

No GPU, no server, no model. The scheduler, the Rust graph runtime and the
request state are real, in the Qwen3.5 shape (an LLM node whose
``prefill_text`` and ``decode`` walks share the combined walk ``mixed``); the
engine is a stub whose ``check_ready`` answers ready at once, so the check
itself costs only its call. Both cores answer the same checks through the same
Python loop (``MicroScheduler._run_checks``).

    python -m benchmark.scheduler.bench_sched_core --rows 128

Scenarios:

* ``steady``: a decode chain's targeted merge (``target=``, the chain's rows
  counted as pre-existing) with no fresh row ready -- every speculative step.
* ``burst``: ``--rows`` decode rows arrive at once, uncapped; one step takes
  them all (the yield-away or non-speculative schedule).
* ``burst_mixed``: half prefill, half decode rows, prefill capped at 8 with a
  row cap of 4 in a mixed step, the combined walk capped at ``--rows``.
* ``backlog``: a ready set over the cap (``--cap``): a first pass parks all
  but a step's worth, the rows it ran come back ready, and the measured pass
  drains a step from the backlog with the fresh rows beside it.
"""
from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.runtime import rust as rust_runtime
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.node_manager_utils import RequestStateManager

ME, PART, LLM = "w0", "default", "LLM"
PRE, DEC, MIXED = "prefill_text", "decode", "mixed"
COMBINED = {(LLM, PRE): MIXED, (LLM, DEC): MIXED}


class Engine:
    def __init__(self, caps, row_caps):
        self.caps, self.row_caps, self.checks = caps, row_caps, 0

    def check_ready(self, node, rid, info, allow_reload=True):
        self.checks += 1
        return FULL_ADMIT_OK

    def splits_by_capture_key(self, node, walk):
        return False

    def get_max_batch_size(self, node, walk):
        return self.caps.get(walk)

    def get_max_rows_in_combined(self, node, walk):
        return self.row_caps.get(walk)


def build(rust_core: bool, caps, row_caps, walks: list[str]):
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
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=ME,
        my_worker_graphs=[
            WorkerGraph(section=pre, graph_walks={PRE}, ranks=[0], worker_graph_id=0),
            WorkerGraph(section=dec, graph_walks={DEC}, ranks=[0], worker_graph_id=1),
        ],
        all_wg_ids_to_graph_walks={0: {PRE}, 1: {DEC}},
        all_wg_ids_to_dyn_loops={0: set(), 1: {"decode_loop"}},
        all_wg_ids_to_nodes={0: {LLM}, 1: {LLM}},
        node_to_partition={LLM: PART},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=RustTensorBookkeeping(), combined_walk_of=dict(COMBINED),
    )
    engine = Engine(caps, row_caps)
    sched = MicroScheduler(
        SimpleNamespace(get_engine=lambda node: engine),
        parallel_leader_nodes={LLM}, combined_walk_of=dict(COMBINED),
    )
    sched.runtime = rt
    if not rust_core:
        sched._core_chosen = True
    rs = RequestStateManager(node_to_partition={LLM: PART})
    rids = []
    for i, walk in enumerate(walks):
        rid = rt.add_request(
            request_id=f"r{i}", partition=PART, graph_walk=walk,
            partition_worker_graph_ids=[0, 1],
            worker_graph_to_workers=ParallelList([0, 1], [[ME], [ME]]), loop_caps={},
        )
        rs.add_request(rid, CurrentForwardPassInfo(
            request_id=f"r{i}", graph_walk=walk, fwd_index=0, random_seed=0,
            max_tokens=1 << 30, rid_handle=rid, partition_name=PART,
        ))
        rids.append(rid)
    # caps are asked once per scheduler lifetime, and the core configured, before any pass
    sched._rust_core()
    for key in ((LLM, MIXED),):
        sched._walk_caps(key)
        sched._walk_row_caps(key)
    return rt, rs, sched, engine, rids


def make_ready(rt, rids):
    block = ColumnarEdgeSpecs.empty()
    for rid in rids:
        block.add(rid, "text_inputs", [], False, next_node=LLM)
    rt.ingest_inputs_batch(block)


def setup(scenario, rust_core, rows, cap):
    """A world in the scenario's starting state, and the call to time."""
    if scenario == "steady":
        rt, rs, sched, engine, rids = build(rust_core, {DEC: None, MIXED: None}, {}, [DEC] * rows)
        return engine, lambda: sched.get_next_batch(
            rs, target=(LLM, DEC), pre_existing_batch_size=rows, capture_group_of=rids[0],
        )
    if scenario == "burst":
        rt, rs, sched, engine, rids = build(rust_core, {DEC: None, MIXED: None}, {}, [DEC] * rows)
        make_ready(rt, rids)
        return engine, lambda: sched.get_next_batch(rs)
    if scenario == "burst_mixed":
        walks = [PRE if i % 2 else DEC for i in range(rows)]
        rt, rs, sched, engine, rids = build(
            rust_core, {PRE: 8, DEC: None, MIXED: rows}, {PRE: 4}, walks,
        )
        make_ready(rt, rids)
        return engine, lambda: sched.get_next_batch(rs)
    assert scenario == "backlog"
    rt, rs, sched, engine, rids = build(rust_core, {PRE: cap, DEC: cap, MIXED: cap}, {}, [DEC] * rows)
    make_ready(rt, rids)
    first = sched.get_next_batch(rs)
    rt.set_in_flight(LLM, list(first.request_to_worker_graph),
                     list(first.request_to_worker_graph.values()), False)
    make_ready(rt, list(first.request_to_worker_graph))
    return engine, lambda: sched.get_next_batch(rs)


def bench(scenario, rust_core, rows, cap, reps):
    walls, checks, sizes = [], [], []
    for _ in range(reps):
        engine, call = setup(scenario, rust_core, rows, cap)
        engine.checks = 0
        t0 = time.perf_counter()
        batch = call()
        walls.append(time.perf_counter() - t0)
        checks.append(engine.checks)
        sizes.append(0 if batch is None else len(batch.request_to_worker_graph))
    return statistics.median(walls) * 1e6, statistics.mean(checks), statistics.mean(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    print("us per pass (median wall); checks and batch rows per pass")
    print(f"{'scenario':12} {'rows':>5} {'python':>9} {'rust':>9} {'checks':>7} {'batch':>6}")
    for scenario in ("steady", "burst", "burst_mixed", "backlog"):
        for rows in a.rows:
            py, py_checks, py_rows = bench(scenario, False, rows, a.cap, a.reps)
            rs, rs_checks, rs_rows = bench(scenario, True, rows, a.cap, a.reps)
            assert (py_checks, py_rows) == (rs_checks, rs_rows), (scenario, rows)
            print(f"{scenario:12} {rows:5d} {py:9.1f} {rs:9.1f} {py_checks:7.0f} {py_rows:6.0f}")


if __name__ == "__main__":
    main()
