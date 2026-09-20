"""Baseline: cost of the per-rid Python graph work, per forward pass.

Measures the three loops that scale with batch size:
  A. add_request   (deepcopy of the graph section, per rid)
  B. postprocess   (mark_node_complete + process_node_outputs, per rid)
  C. schedule scan (get_ready_node_names + per-(rid,node) walk/partition lookups)

Run: python docs/design/rust_graph_port/bench_python.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph_fixture import WALK, make_section, make_sharding_config, node_names, sampler_output_edges

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.worker.node_manager_utils import (
    PerPartitionInfo,
    PerRequestInfo,
    WorkerGraphQueues,
    WorkerGraphsManager,
)

WG_ID = "wg0"
PART = "p0"
ME = "w0"


class _NullTensorManager:
    def dereference(self, *a, **k): pass
    def increment_ref(self, *a, **k): pass


def build_manager(width: int, tp_size: int = 1) -> WorkerGraphsManager:
    from mstar.model.base import WorkerGraph
    names = node_names(width)
    wg = WorkerGraph(section=make_section(width), graph_walks={WALK})
    queues = WorkerGraphQueues(
        worker_graph_id=WG_ID, graph_walks={WALK}, worker_graph=wg,
        per_request_queues={}, tensor_manager=_NullTensorManager(),
    )
    return WorkerGraphsManager(
        queues={WG_ID: queues}, per_request_info={},
        base_sharding_config=make_sharding_config(width, ME, tp_size),
        worker_id=ME,
        all_worker_graph_ids_to_graph_walks={WG_ID: {WALK}},
        all_worker_graph_ids_to_nodes={WG_ID: set(names)},
        all_worker_graph_ids_to_dyn_loops={WG_ID: {"decode_loop"}},
        node_to_partition={n: PART for n in names},
    )


def add_requests(mgr: WorkerGraphsManager, rids, width, tp_size) -> float:
    """Returns seconds for the whole add (dominated by deepcopy)."""
    sc = make_sharding_config(width, ME, tp_size)
    t0 = time.perf_counter()
    for rid in rids:
        mgr.queues[WG_ID].add_request(rid)
        mgr.per_request_info[rid] = PerRequestInfo(
            node_to_workers={}, dyn_loop_to_workers={}, worker_graph_ids=[WG_ID],
            sharding_config=sc,
            per_partition_info={PART: PerPartitionInfo(
                current_fwd_info=CurrentForwardPassInfo(
                    request_id=rid, fwd_index=0, random_seed=0, max_tokens=128,
                    graph_walk=WALK, partition_name=PART),
                graph_walk_worker_graph_ids=[WG_ID],
            )},
        )
    return time.perf_counter() - t0


def seed(mgr, rids, width):
    """Drive each rid to the point where `sampler` has just run."""
    from graph_fixture import _tinfo

    from mstar.graph.base import GraphEdge
    q = mgr.queues[WG_ID]
    for rid in rids:
        for name in ("input_ids", "position"):
            q.process_new_inputs(rid, [GraphEdge(next_node="llm", name=name,
                                                 tensor_info=[_tinfo(f"{rid}-{name}")])])
        mgr.mark_node_complete(rid, WG_ID, "llm")
        q.process_new_inputs(rid, [GraphEdge(next_node="sampler", name="logits",
                                             tensor_info=[_tinfo(f"{rid}-lg")])])


def bench_postprocess(mgr, rids, width, iters) -> float:
    """ns per rid for: mark_node_complete + clone edges + process_node_outputs."""
    edges = {rid: sampler_output_edges(width, rid) for rid in rids}
    t0 = time.perf_counter()
    for _ in range(iters):
        for rid in rids:
            mgr.mark_node_complete(rid, WG_ID, "sampler")
            real = [e.clone() for e in edges[rid]]
            mgr.process_node_outputs(rid, node_name="sampler", outputs=real, graph_walk=WALK)
            # reset so the next iteration replays the same step
            mgr.queues[WG_ID].per_request_queues[rid].nodes["sampler"].ready_signals.clear()
    return (time.perf_counter() - t0) / (iters * len(rids)) * 1e9


def bench_sched_scan(mgr, rids, width, iters) -> float:
    """ns per rid for the ready-scan part of MicroScheduler.get_next_batch."""
    q = mgr.queues[WG_ID]
    t0 = time.perf_counter()
    for _ in range(iters):
        acc = {}
        for _wgid, queue in mgr.queues.items():
            for rid, names in queue.get_ready_node_names().items():
                if rid not in mgr.per_request_info:
                    continue
                for sname in names:
                    part = mgr.get_partition_for_node(sname)
                    gw = mgr.get_graph_walk(rid, part)
                    mgr.get_fwd_info(rid, part)  # engine check_ready() would go here
                    acc.setdefault(sname, []).append((rid, gw))
    return (time.perf_counter() - t0) / (iters * len(rids)) * 1e9


def main():
    print(f"{'B':>5} {'width':>6} {'add_req us/rid':>15} {'postproc ns/rid':>16} "
          f"{'sched ns/rid':>13} {'postproc+sched us/fwd':>22}")
    for width in (0, 4):
        for B in (1, 8, 32, 128):
            mgr = build_manager(width)
            rids = [f"r{i}" for i in range(B)]
            add_s = add_requests(mgr, rids, width, 1)
            seed(mgr, rids, width)
            iters = max(2, 4000 // B)
            post = bench_postprocess(mgr, rids, width, iters)
            sched = bench_sched_scan(mgr, rids, width, iters)
            print(f"{B:>5} {width:>6} {add_s / B * 1e6:>15.1f} {post:>16.0f} "
                  f"{sched:>13.0f} {(post + sched) * B / 1e3:>22.1f}")


if __name__ == "__main__":
    main()
