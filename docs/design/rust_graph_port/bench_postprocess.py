"""The postprocess routing path: per-request loop vs the batched call.

Measures exactly what `_postprocess_batch` does for the graph half —
mark-complete + route — under both paths, on the same manager state.

    python docs/design/rust_graph_port/bench_postprocess.py
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph_fixture import WALK, make_section, make_sharding_config, node_names

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.worker.node_manager_utils import (
    PerPartitionInfo,
    PerRequestInfo,
    WorkerGraphQueues,
    WorkerGraphsManager,
)

WG_ID, PART, ME = "wg0", "p0", "w0"


class _NullTensorManager:
    def dereference(self, *a, **k): pass
    def increment_ref(self, *a, **k): pass


def _tinfo(uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[8, 16], dtype="bfloat16", nbytes=256, address=0, stride=[16, 1],
        uuid=uuid, source_session_id="h:1", source_entity=ME,
    )


def build(width: int, rids: list[str], tp_size: int = 1) -> WorkerGraphsManager:
    from mstar.model.base import WorkerGraph

    names = node_names(width)
    wg = WorkerGraph(section=make_section(width, max_iters=10**9), graph_walks={WALK})
    mgr = WorkerGraphsManager(
        queues={WG_ID: WorkerGraphQueues(
            worker_graph_id=WG_ID, graph_walks={WALK}, worker_graph=wg,
            per_request_queues={}, tensor_manager=_NullTensorManager(), worker_id=ME,
        )},
        per_request_info={},
        base_sharding_config=make_sharding_config(width, ME, tp_size),
        worker_id=ME,
        all_worker_graph_ids_to_graph_walks={WG_ID: {WALK}},
        all_worker_graph_ids_to_nodes={WG_ID: set(names)},
        all_worker_graph_ids_to_dyn_loops={WG_ID: {"decode_loop"}},
        node_to_partition=dict.fromkeys(names, PART),
    )
    sc = make_sharding_config(width, ME, tp_size)
    from mstar.graph.base import NodeAndGraphWalk
    node_to_workers = {
        NodeAndGraphWalk(n, WALK): [f"w{i}" for i in range(tp_size)] for n in names
    }
    for rid in rids:
        mgr.queues[WG_ID].add_request(rid)
        mgr.per_request_info[rid] = PerRequestInfo(
            node_to_workers=node_to_workers, dyn_loop_to_workers={},
            worker_graph_ids=[WG_ID], sharding_config=sc,
            per_partition_info={PART: PerPartitionInfo(
                current_fwd_info=CurrentForwardPassInfo(
                    request_id=rid, fwd_index=0, random_seed=0, max_tokens=128,
                    graph_walk=WALK, partition_name=PART),
                graph_walk_worker_graph_ids=[WG_ID],
            )},
        )
        for name in ("input_ids", "position"):
            mgr.queues[WG_ID].process_new_inputs(rid, [GraphEdge(
                next_node="llm", name=name, tensor_info=[_tinfo(f"{rid}-{name}")])])
        mgr.mark_node_complete(rid, WG_ID, "llm")
        mgr.queues[WG_ID].process_new_inputs(rid, [GraphEdge(
            next_node="sampler", name="logits", tensor_info=[_tinfo(f"{rid}-lg")])])
    return mgr


def stamp(mgr, rids, name):
    """Put tensor_info on every request's output edges for `name`, as
    store_and_populate_graph_edges does."""
    for rid in rids:
        node = mgr.queues[WG_ID].per_request_queues[rid].nodes[name]
        for i, e in enumerate(node.outputs):
            e.tensor_info = [_tinfo(f"{rid}-{name}{i}")]


def _time(fn, width, b, tp_size, iters, order, repeats=5) -> float:
    """Min of `repeats` runs, GC off during timing.

    Both paths allocate heavily (an edge clone per output per request), so a
    collection landing inside one run and not the other swamps the difference.
    """
    rids = [f"r{i}" for i in range(b)]
    n = iters * b * len(order)
    best = float("inf")
    for _ in range(repeats):
        mgr = build(width, rids, tp_size)
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn(mgr, rids, order, iters)
            best = min(best, (time.perf_counter() - t0) / n * 1e9)
        finally:
            gc.enable()
    return best


def _per_request(mgr, rids, order, iters):
    for _ in range(iters):
        for name in order:
            stamp(mgr, rids, name)
            for rid in rids:
                out = mgr.mark_node_complete(rid, WG_ID, name)
                mgr.process_node_outputs(
                    rid, node_name=name,
                    outputs=[e.clone() for e in out.output_edges], graph_walk=WALK)


def _batched(mgr, rids, order, iters):
    rid_to_wg = dict.fromkeys(rids, WG_ID)
    for _ in range(iters):
        for name in order:
            stamp(mgr, rids, name)
            mgr.complete_and_route_batch(
                node_name=name, request_to_worker_graph=rid_to_wg, graph_walk=WALK)


def bench(width: int, b: int, iters: int, tp_size: int = 1) -> tuple[float, float]:
    """A full loop iteration per `iters`, so the loop actually advances and
    its output cache clears — otherwise _cached_outputs grows without bound
    and the measurement drifts."""
    order = ["llm", "sampler"] + [f"side{i}" for i in range(width)]
    return (
        _time(_per_request, width, b, tp_size, iters, order),
        _time(_batched, width, b, tp_size, iters, order),
    )


def main():
    print("mark_node_complete + route, ns per request per node completion:")
    print(f"{'width':>6} {'TP':>3} {'B':>5} {'per-request':>12} {'batched':>10} {'speedup':>9}")
    for width in (0, 4):
        for tp in (1, 2):
            for b in (8, 32, 128):
                a, c = bench(width, b, max(4, 2000 // b), tp)
                print(f"{width:>6} {tp:>3} {b:>5} {a:>12.0f} {c:>10.0f} {a / c:>8.2f}x")


if __name__ == "__main__":
    main()
