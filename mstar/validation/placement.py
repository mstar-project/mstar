"""Placement checks: walks and partitions, ``node_groups``, the worker graphs
they divide into, the parallel groups each worker is handed, and the weights
that pick a request's replica among them."""

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from mstar.conductor.routing import RoutingWeights
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import GlobalParallelConfig
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.graph.base import GraphSection, self_triggered, top_level_entities
from mstar.model.base import Model, WorkerGraph

# MAX_INPUTS in rust/src/graph/spec.rs: the Rust runtime keeps a node's
# readiness in a u64 mask. Checked for both runtimes, since AUTO may pick either.
MAX_NODE_INPUTS = 64


def node_groups_problems(config: Mapping[str, Any]) -> list[str]:
    """A missing ``node_groups``, or an entry the placement checks cannot read."""
    node_groups = config.get("node_groups")
    if node_groups is None:
        return ["the config must define `node_groups`"]
    if not isinstance(node_groups, list):
        return [f"`node_groups` must be a list of entries; got {node_groups!r}"]
    return [
        problem for i, group in enumerate(node_groups)
        if (problem := _node_group_entry_problem(i, group))
    ]


def routing_weights_problems(config: Mapping[str, Any]) -> list[str]:
    """``routing_weights``, the data-parallel replica pick's: one per term of
    the cost, each a finite number >= 0, since the cost is a sum of weighted
    terms, and ``load`` above 0 when another term is, or nothing would spread
    a hot prefix."""
    weights = config.get("routing_weights")
    if weights is None:
        return []
    if not isinstance(weights, Mapping):
        return [f"routing_weights must be a mapping of {list(RoutingWeights._fields)}; got {weights!r}"]
    problems = []
    unknown = sorted(set(weights) - set(RoutingWeights._fields), key=str)
    if unknown:
        problems.append(f"routing_weights names {unknown}; it takes {list(RoutingWeights._fields)}")
    for name, weight in weights.items():
        if (
            isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0
        ):
            problems.append(f"routing_weights.{name} must be a finite number >= 0; got {weight!r}")
    if not problems and not weights.get("load", RoutingWeights._field_defaults["load"]) and any(
        weights.get(name, default) for name, default in RoutingWeights._field_defaults.items() if name != "load"
    ):
        problems.append("routing_weights.load must be above 0 while another weight is")
    return problems


def _is_count(value: Any, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _node_group_entry_problem(i: int, group: Any) -> str | None:
    if (
        not isinstance(group, dict)
        or not isinstance(group.get("node_names"), list)
        or not isinstance(group.get("ranks"), list)
        or not group["ranks"]
    ):
        return f"node_groups[{i}] needs a `node_names` list and a non-empty `ranks` list; got {group!r}"
    for key in ("node_names", "graph_walks"):
        names = group.get(key, [])
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            return f"node_groups[{i}]: {key} must be a list of names; got {names!r}"
    if not all(_is_count(rank, 0) for rank in group["ranks"]):
        return f"node_groups[{i}]: ranks must be integers >= 0; got {group['ranks']!r}"
    for key in ("tp_size", "sp_size"):
        if not _is_count(group.get(key, 1), 1):
            return f"node_groups[{i}]: {key} must be an integer >= 1; got {group[key]!r}"
    return None


def check_placement(model: Model, config: Mapping[str, Any]) -> list[str]:
    """Problems with ``node_groups`` and the walks and partitions it places.

    ``Model.get_worker_graphs`` runs this before it divides the walks, after
    any setup a model does from ``node_groups`` (BAGEL adds its CFG walk).
    """
    malformed = node_groups_problems(config)
    if malformed:
        return malformed
    node_groups = config["node_groups"]
    walk_graphs = model.get_graph_walk_graphs()
    return [
        *_node_group_shapes(model, node_groups),
        *_walk_partitions(model, walk_graphs),
        *_node_placement(walk_graphs, node_groups),
    ]


def check_worker_graphs(
    model: Model,
    worker_graphs: Sequence[WorkerGraph],
    sharding_config: ShardingConfig,
    specs: Sequence[NodeResourceSpec] | None,
) -> list[str]:
    """Problems with the worker graphs ``model.get_worker_graphs`` built, and
    with the parallel groups the conductor would hand each worker. With no
    ``specs`` (the model could not declare them) only the checks on resource
    placement are skipped."""
    walk_to_wgs: dict[str, list[WorkerGraph]] = {}
    for wg in worker_graphs:
        for walk in sorted(wg.graph_walks):
            walk_to_wgs.setdefault(walk, []).append(wg)
    return [
        *_input_limits(walk_to_wgs),
        *_stream_driven_walks(model, walk_to_wgs),
        *_loop_successors(walk_to_wgs),
        *_sharding_groups(worker_graphs, sharding_config),
        *_worker_parallelism(worker_graphs, specs or ()),
    ]


def _describe(wg: WorkerGraph) -> str:
    return (
        f"the worker graph of {sorted(wg.graph_walks)} with nodes "
        f"{sorted(wg.section.get_nodes())} on ranks {wg.ranks}"
    )


def _node_group_shapes(model: Model, node_groups: list[dict]) -> Iterator[str]:
    sharding = model.get_default_sharding_config()
    for i, group in enumerate(node_groups):
        names, ranks = group["node_names"], group["ranks"]
        tp, sp = group.get("tp_size", 1), group.get("sp_size", 1)
        if len(ranks) % (tp * sp):
            yield (
                f"node_groups[{i}] (nodes {names}) has {len(ranks)} ranks; not "
                f"divisible by tp_size*sp_size ({tp}*{sp}={tp * sp})."
            )
        for size, axis, enabled in (
            (tp, "tp", sharding.tp_enabled_nodes), (sp, "sp", sharding.sp_enabled_nodes),
        ):
            if size > 1 and any(node not in enabled for node in names):
                yield (
                    f"node_groups[{i}] has {axis}_size {size} and contains nodes not "
                    f"in {axis}_enabled_nodes {sorted(enabled)}: {names}"
                )


def _walk_partitions(model: Model, walk_graphs: Mapping[str, GraphSection]) -> Iterator[str]:
    """A walk in no partition or in two: its worker graphs are registered on no
    worker, or in both partitions."""
    partitions = model.get_partitions()
    for part in partitions:
        unknown = sorted(part.graph_walks - walk_graphs.keys())
        if unknown:
            yield (
                f"Partition {part.name!r} lists graph walks {unknown} that "
                f"get_graph_walk_graphs() does not define (walks: {sorted(walk_graphs)})."
            )
    for walk in walk_graphs:
        owners = [part.name for part in partitions if walk in part.graph_walks]
        if len(owners) != 1:
            yield (
                f"Graph walk {walk!r} is in {len(owners)} partitions {owners}; "
                "every walk must be listed in the graph_walks of exactly one "
                "PartitionDefinition from get_partitions()."
            )


def _node_placement(walk_graphs: Mapping[str, GraphSection], node_groups: list[dict]) -> Iterator[str]:
    """A node of a walk in no node_group or in two (the later group silently
    wins), and a node_group naming a walk the model does not define."""
    for i, group in enumerate(node_groups):
        unknown = sorted(set(group.get("graph_walks", [])) - walk_graphs.keys())
        if unknown:
            yield (
                f"node_groups[{i}] (nodes {group['node_names']}) lists "
                f"graph_walks {unknown}, which this model does not define "
                f"(walks: {sorted(walk_graphs)})."
            )
    for walk, graph in walk_graphs.items():
        for node in graph.get_nodes():
            groups = [
                i for i, group in enumerate(node_groups)
                if node in group["node_names"]
                and ("graph_walks" not in group or walk in group["graph_walks"])
            ]
            if len(groups) != 1:
                where = ", ".join(f"node_groups[{i}]" for i in groups) or "no node_groups entry"
                yield (
                    f"node {node!r} of graph walk {walk!r} is placed by "
                    f"{where}; place it in exactly one node_groups entry for that walk "
                    "(use graph_walks to give one node different ranks per walk)."
                )


def _input_limits(walk_to_wgs: Mapping[str, list[WorkerGraph]]) -> Iterator[str]:
    seen: set[str] = set()
    for walk, wgs in walk_to_wgs.items():
        for wg in wgs:
            for name, node in wg.section.get_nodes().items():
                if len(node.input_names) > MAX_NODE_INPUTS and name not in seen:
                    seen.add(name)
                    yield (
                        f"node {name!r} of graph walk {walk!r} has {len(node.input_names)} "
                        f"inputs; a node takes at most {MAX_NODE_INPUTS}, the width of the "
                        "Rust graph runtime's readiness mask."
                    )


def _on_same_workers(wgs: list[WorkerGraph]) -> bool:
    """Whether every request runs all of these worker graphs on the same
    workers: the conductor picks one replica for every node_group on the same
    lockstep instances (``ReplicaRouter``), so they share those instances."""
    return len({tuple(map(tuple, wg.replica_ranks)) for wg in wgs}) == 1


def _stream_driven_walks(model: Model, walk_to_wgs: Mapping[str, list[WorkerGraph]]) -> Iterator[str]:
    """Stream-driven walk shapes the runtime and conductor can't account for.

    A worker graph counts entity completions per pass, and a self-triggered
    consumer (every input streamed) is re-armed for the next chunk as soon as
    it completes, so a second top-level entity can be counted done before it
    runs, and the pass-end reset drops input it has buffered. A consumer
    gated on a non-streamed input waits for that input instead, which is
    safe when it arrives once per pass (from the conductor, say).
    Across worker graphs, self-triggered passes overlap with no pass
    identity, and the conductor reads the final-stream flag only off the
    report that closes a pass: the flag is per worker, so it is there only
    if every worker graph of the walk runs on the consumer's workers. A walk
    with no stream-consuming node is conductor driven and unaffected, and a
    TP worker graph is one worker graph on several ranks.
    """
    topology = model.get_partition_topology()
    consumer_partitions = {conn.to_partition for conn in topology.connections}
    walk_partition = {
        walk: part.name for part in model.get_partitions() for walk in part.graph_walks
    }
    for walk, wgs in walk_to_wgs.items():
        if walk_partition[walk] not in consumer_partitions:
            continue
        if not any(wg.consumes_stream for wg in wgs):
            continue
        where = f"graph walk {walk!r} of consumer partition {walk_partition[walk]!r}"
        entities = [top_level_entities(wg.section) for wg in wgs]
        triggered = sorted(
            name for ents in entities for name, e in ents.items() if self_triggered(e)
        )
        if len(wgs) > 1 and (triggered or not _on_same_workers(wgs)):
            yield (
                f"{where} consumes a stream and spans {len(wgs)} worker graphs "
                f"{[sorted(wg.section.get_nodes()) for wg in wgs]} on ranks "
                f"{[wg.ranks for wg in wgs]}. Across worker graphs a stream-driven walk "
                "is served only with every consumer gated on a non-streamed input that "
                "arrives once per pass and every worker graph on the same workers (node_groups "
                "on the same ranks, instance for instance); otherwise run it as "
                "one worker graph, or move the work after the consumer into a "
                "downstream partition fed by a StreamingGraphEdge."
            )
        if len(wgs) == 1 and len(entities[0]) > 1 and triggered:
            yield (
                f"{where} has self-triggered consumers {triggered} (every input "
                f"streamed) among {len(entities[0])} top-level entities {sorted(entities[0])} "
                "in one worker graph. Gate the consumer on a non-streamed input that "
                "arrives once per pass (from the conductor, say), or move the work "
                "after it into a downstream partition fed by a StreamingGraphEdge."
            )


def _loop_successors(walk_to_wgs: Mapping[str, list[WorkerGraph]]) -> Iterator[str]:
    """A Loop member with an edge to a node outside the Loop in the same
    worker graph.

    A worker graph counts top-level completions, and that successor
    completes once per iteration, so the pass ends before the Loop and the
    successor's later inputs are done, and the pass-end reset drops them.
    ``Loop.accumulated_outputs`` hands the successor every iteration's
    output once, when the Loop finishes.
    """
    for walk, wgs in walk_to_wgs.items():
        for wg in wgs:
            wg_nodes = wg.section.get_nodes()
            for loop in wg.section.get_loops().values():
                members = loop.get_nodes()
                for name, node in members.items():
                    for edge in node.outputs:
                        if edge.next_node in wg_nodes and edge.next_node not in members:
                            yield (
                                f"graph walk {walk!r}: node {name!r} of Loop "
                                f"{loop.name!r} sends {edge.name!r} to {edge.next_node!r}, "
                                "outside the Loop in the same worker graph. The worker graph "
                                "would report done before the Loop and its successor finish. "
                                "Send it through Loop.accumulated_outputs (every iteration's "
                                "output, once the Loop finishes) or Loop.outputs instead."
                            )


def _sharding_groups(
    worker_graphs: Sequence[WorkerGraph], sharding_config: ShardingConfig,
) -> Iterator[str]:
    """v1: one sharding group per worker graph, with the worker graph's
    tp_size, and no custom graph_walks on a group holding a streaming
    consumer."""
    streaming_consumers: set[str] = set()
    node_walk_to_wg: dict[tuple[str, str], WorkerGraph] = {}
    for wg in worker_graphs:
        for name, node in wg.section.get_nodes().items():
            for walk in wg.graph_walks:
                node_walk_to_wg[(name, walk)] = wg
            if node.consumes_stream:
                streaming_consumers.add(name)
    all_walks = {walk for wg in worker_graphs for walk in wg.graph_walks}

    owner: dict[int, str] = {}
    reported: set[tuple[int, str]] = set()
    for group in sharding_config.groups:
        if group.graph_walks is not None and group.nodes & streaming_consumers:
            yield (
                f"Sharding group with nodes {sorted(group.nodes)} includes a streaming "
                f"consumer but has custom graph walk configuration {sorted(group.graph_walks)}. "
                "It is currently disallowed to set custom graph walks for TP groups that "
                "include streaming consumer nodes."
            )
        key = group.key_str()
        for walk in sorted(group.graph_walks or all_walks):
            for node in sorted(group.nodes):
                wg = node_walk_to_wg.get((node, walk))
                if wg is None:
                    continue
                prior = owner.setdefault(id(wg), key)
                if prior != key and (id(wg), "owner") not in reported:
                    reported.add((id(wg), "owner"))
                    yield (
                        f"{_describe(wg)} is claimed by two sharding groups ({prior!r} "
                        f"and {key!r}). v1 requires one TP group per worker graph; split "
                        "the worker graph or merge the groups."
                    )
                if wg.tp_size != group.tp_size and (id(wg), "tp") not in reported:
                    reported.add((id(wg), "tp"))
                    yield (
                        f"{_describe(wg)} has tp_size {wg.tp_size}, but its sharding group "
                        f"has tp_size {group.tp_size}. node_groups and sharding_config disagree."
                    )


def _worker_parallelism(
    worker_graphs: Sequence[WorkerGraph], specs: Sequence[NodeResourceSpec],
) -> Iterator[str]:
    """What each worker would check of its own ``WorkerParallelGroups``, on the
    same groups the conductor hands it: at most one lockstep-parallel node per
    worker, and every resource spec's nodes on one TP x SP shape and, on each
    worker, in one parallel group. Over every node placed on the worker; the
    engine checked only nodes it built a submodule for, so a dummy run (no
    weights) skipped the resource checks."""
    ranks = sorted({rank for wg in worker_graphs for rank in wg.ranks})
    if not ranks:
        return
    # GlobalParallelConfig and init_dist number workers by their position
    # among the deployment's ranks, and parallel groups by rank
    misplaced = sorted({
        rank for wg in worker_graphs for group in wg._tp_ranks + wg._sp_ranks
        for rank in group if rank >= len(ranks) or ranks[rank] != rank
    })
    if misplaced:
        yield (
            f"the deployment uses ranks {ranks}; ranks {misplaced} are in tensor- or "
            "sequence-parallel groups, which need every lower rank in use too."
        )
        return
    worker_ids = [f"worker_{rank}" for rank in ranks]
    try:
        per_worker = GlobalParallelConfig(dict(enumerate(worker_graphs)), worker_ids).per_worker_config
    except RuntimeError as e:
        yield str(e)
        return

    deployed: dict[str, set[str]] = {
        wid: {
            node for wg in worker_graphs if rank in wg.ranks
            for node in wg.section.get_nodes()
        }
        for rank, wid in zip(ranks, worker_ids, strict=True)
    }
    for wid, nodes in deployed.items():
        parallel_nodes = sorted(
            node for node in nodes
            if per_worker[wid].get_instance_world_size_for_node(node) > 1
        )
        if len(parallel_nodes) > 1:
            yield (
                f"Multiple parallel nodes {parallel_nodes} found in worker {wid}; "
                "current implementation requires at most one lockstep-parallel node "
                "per worker."
            )

    all_nodes = set().union(*deployed.values())
    any_worker = per_worker[worker_ids[0]]
    for spec in specs:
        if not spec.nodes & all_nodes:
            continue
        if not any_worker.all_have_compatible_parallel_shape(spec.nodes):
            yield (
                f"Resource spec {spec.resource_key} nodes {sorted(spec.nodes)} "
                "must use the same TP x SP shape across replicas"
            )
        for wid, nodes in deployed.items():
            local = spec.nodes & nodes
            if local and not per_worker[wid].all_in_same_group(sorted(local)):
                yield (
                    f"Resource spec {spec.resource_key} local nodes {sorted(local)} "
                    f"on {wid} must all be in the same parallel (tp x sp) group"
                )
