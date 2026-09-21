"""Per-request sharding, derived the same way by both runtimes.

A request's ``ShardingConfig`` is a pure function of the base config and the
``worker_graph_to_workers`` map the conductor sent with NEW_REQUEST. Rust
derives its own copy by the same rule (``ShardingTemplate::instantiate`` is
``clone_empty()`` + ``setup()`` verbatim), but that copy cannot come back out:
``TensorCommunicationManager.register_request`` wants the Python object, and
the TP fan-out paths reach into ``group._workers``.

So this stays in Python and both runtimes call it. Nothing is replicated --
``setup()`` IS the logic; this only assembles its argument.
"""
from __future__ import annotations

from mstar.distributed.base import NodeAndGraphWalk, ShardingConfig
from mstar.graph.runtime.base import ParallelList


def node_to_workers(
    worker_graph_to_workers: ParallelList[int, list[str]],
    all_wg_ids_to_graph_walks: dict[int, set[str]],
    all_wg_ids_to_nodes: dict[int, set[str]],
) -> dict[NodeAndGraphWalk, list[str]]:
    """Which workers own each (node, walk), from the per-worker-graph map.

    Worker graph ids this rank has never heard of are skipped: the conductor
    addresses the whole mesh, and a rank only knows the graphs it was built
    with.
    """
    out: dict[NodeAndGraphWalk, list[str]] = {}
    for wg_id, worker_ids in worker_graph_to_workers:
        if wg_id not in all_wg_ids_to_graph_walks:
            continue
        for walk in all_wg_ids_to_graph_walks[wg_id]:
            for name in all_wg_ids_to_nodes[wg_id]:
                out[NodeAndGraphWalk(node=name, graph_walk=walk)] = worker_ids
    return out


def for_request(
    base: ShardingConfig,
    worker_graph_to_workers: ParallelList[int, list[str]],
    all_wg_ids_to_graph_walks: dict[int, set[str]],
    all_wg_ids_to_nodes: dict[int, set[str]],
) -> ShardingConfig:
    """This request's config. ``clone_empty`` so the base is never mutated --
    it is shared by every request on this rank."""
    config = base.clone_empty()
    config.setup(node_to_workers(
        worker_graph_to_workers, all_wg_ids_to_graph_walks, all_wg_ids_to_nodes,
    ))
    return config
