"""Resource checks: the model's declarations and the deployment's ``resources:``
overrides, as every worker's engine would build them."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import PositionSpec, PosScheme
from mstar.engine.resources.recurrent.config import RecurrentStateSpec
from mstar.engine.resources.runner import topo_sort
from mstar.engine.resources.spec import (
    TOP_LEVEL_KV_CACHE,
    NodeResourceSpec,
    spec_dependency_problems,
    unknown_overrides_message,
)
from mstar.graph.special_destinations import SPECIAL_DESTINATIONS
from mstar.model.base import Model, keyed_labels
from mstar.streaming.topology import StreamingGraphEdge


def check_resources(
    model: Model, specs: Sequence[NodeResourceSpec], config: Mapping[str, Any],
) -> list[str]:
    """Apply ``config``'s overrides to ``specs`` and return every problem.

    A deployment this accepts passes the worker's ``apply_yaml_overrides``,
    ``resolve_spec_dependencies`` and ``topo_sort``, which still raise on the
    first problem for callers that skip validation.
    """
    problems = spec_dependency_problems(specs)
    if not problems:
        try:
            topo_sort({spec.resource_key: spec for spec in specs})
        except ValueError as e:
            problems.append(str(e))
    problems += _apply_overrides(specs, config)
    problems += _uncacheable_positions(specs, model)
    problems += _unskippable_resources(specs, model)
    problems += _unknown_prefix_walks(model)
    problems += _undeclared_pools(specs, model)
    problems += _unservable_walks(specs, model)
    return problems


def resources_shape_problem(config: Mapping[str, Any]) -> str | None:
    """A ``resources:`` block that is not a mapping keyed by resource name."""
    overrides = config.get("resources") or {}
    if not isinstance(overrides, Mapping) or not all(isinstance(key, str) for key in overrides):
        return f"`resources:` must map resource keys to their overrides; got {overrides!r}"
    return None


def _apply_overrides(
    specs: Sequence[NodeResourceSpec], config: Mapping[str, Any],
) -> Iterator[str]:
    """``apply_yaml_overrides``, one ``resources:`` block at a time, so a typo
    in one block does not hide another's."""
    if "kv_cache" in config:
        yield TOP_LEVEL_KV_CACHE
    if problem := resources_shape_problem(config):
        yield problem
        return
    overrides = config.get("resources") or {}
    by_key = {spec.resource_key: spec for spec in specs}
    unknown = sorted(overrides.keys() - by_key.keys())
    if unknown:
        yield unknown_overrides_message(unknown, by_key)
    for key, kwargs in overrides.items():
        if key not in by_key:
            continue
        try:
            by_key[key].apply_yaml_overrides(**kwargs)
        except (TypeError, ValueError) as e:
            yield f"resources.{key}: {e}"


def _uncacheable_positions(
    specs: Sequence[NodeResourceSpec], model: Model,
) -> Iterator[str]:
    """A keyed cache whose positions are not sequential.

    A cached page is reusable only at the positions it was written at, since its
    K is stored already rotated.
    """
    declared = set(keyed_labels(model))
    if not declared:
        return
    for spec in specs:
        if not isinstance(spec, PositionSpec):
            continue
        cached = declared & spec.depends_on()
        if cached and spec.config.scheme is not PosScheme.SEQUENTIAL:
            yield (
                f"resource {spec.resource_key!r} places positions by "
                f"{spec.config.scheme.value}, but it sits over "
                f"{sorted(cached)}, which the model keyed for prefix reuse; "
                "either drop the stream or give the resource a sequential "
                "scheme"
            )


def _unskippable_resources(
    specs: Sequence[NodeResourceSpec], model: Model,
) -> Iterator[str]:
    """A declared node carrying a resource that cannot skip a prefix.

    A hit takes the front of a request away from every resource on the node,
    not just the ones that answered the probe, so one that cannot account for
    tokens it never saw would plan its step against a sequence that never ran.
    A recurrent pool restores its state from snapshots instead, and a resource
    that depends only on pools plans against what they restore. The engine
    keeps the node's caches shut where a pool keeps no snapshots.
    """
    declared = set(keyed_labels(model))
    if not declared:
        return
    cached_nodes = {
        node for spec in specs if spec.resource_key in declared
        for node in spec.nodes
    }
    pools = {spec.resource_key for spec in specs if isinstance(spec, RecurrentStateSpec)}
    for spec in specs:
        shared = spec.nodes & cached_nodes
        if not shared or spec.resource_class.prefix_skip_safe or spec.resource_key in pools:
            continue
        if spec.depends_on() and spec.depends_on() <= pools:
            continue
        yield (
            f"resource {spec.resource_key!r} cannot serve a request whose "
            f"prefix came from the cache, but it sits on {sorted(shared)}, "
            "which the model keyed for prefix reuse; either drop the stream "
            "or give the resource the prefix hooks"
        )


def _unknown_prefix_walks(model: Model) -> Iterator[str]:
    """A declared label that names a walk the model never runs, or a growth
    loop its walk doesn't run.

    Only the named walks may file pages or be probed, so a name that matches
    nothing leaves that cache empty for good, and nothing says why.
    """
    graphs = model.get_graph_walk_graphs()
    walks = set(graphs)
    for key, by_label in model.kv_labels().items():
        for label, kv_label in by_label.items():
            named = {walk for walk, _ in kv_label.growth}
            if kv_label.keys is not None:
                named.update(kv_label.keys.walks)
            unknown = sorted(named - walks)
            if unknown:
                yield (
                    f"{type(model).__name__} declares {key!r}/{label!r} over "
                    f"{unknown}, which it never runs; its walks are "
                    f"{sorted(walks)}"
                )
            for walk, loop in kv_label.growth:
                if walk not in graphs:
                    continue
                loops = graphs[walk].get_loops()
                if loop not in loops:
                    yield (
                        f"{type(model).__name__} declares {key!r}/{label!r} growing on "
                        f"{loop!r}, which {walk!r} doesn't run; its loops are {sorted(loops)}"
                    )


def _undeclared_pools(specs: Sequence[NodeResourceSpec], model: Model) -> Iterator[str]:
    """A paged pool with no labels. A ring pool needs none, since its size is
    fixed per request."""
    declared = model.kv_labels()
    for spec in specs:
        if isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig) and not declared.get(spec.resource_key):
            yield (
                f"{type(model).__name__} declares no labels for its paged pool "
                f"{spec.resource_key!r}; add them to `kv_labels`, and return each "
                "request's layout for them from `process_prompt`"
            )


def _unservable_walks(specs: Sequence[NodeResourceSpec], model: Model) -> Iterator[str]:
    """A keyed label over several walks if the cache could not serve one of
    them whole: a walk where a keyed node feeds another node in the walk's
    partition, or where no keyed node runs.

    A walk the cache serves whole completes with no outputs, so a node that
    reads them would get nothing. A label keyed over one walk is not checked,
    because its one span always runs its last token.
    """
    graphs = model.get_graph_walk_graphs()
    partition_of = {walk: p.name for p in model.get_partitions() for walk in p.graph_walks}
    by_key = {spec.resource_key: spec for spec in specs}
    for key, by_label in keyed_labels(model).items():
        for label, kv_label in by_label.items():
            if len(kv_label.keys.walks) < 2:
                continue
            if key not in by_key:
                yield (
                    f"{type(model).__name__} lays {key!r}/{label!r} out, but "
                    f"declares no resource {key!r}; its resources are {sorted(by_key)}"
                )
                continue
            keyed_nodes = by_key[key].nodes
            # _unknown_prefix_walks reports a walk the model does not run
            for walk in kv_label.keys.walks:
                if walk not in graphs:
                    continue
                nodes = graphs[walk].get_nodes()
                ran = sorted(keyed_nodes & set(nodes))
                if not ran:
                    yield (
                        f"{type(model).__name__} lays {key!r}/{label!r} out over "
                        f"{walk!r}, where none of {sorted(keyed_nodes)} runs"
                    )
                for name in ran:
                    fed = sorted(
                        {
                            edge.next_node for edge in nodes[name].outputs
                            if not isinstance(edge, StreamingGraphEdge)
                            or edge.target_partition == partition_of[walk]
                        }
                        - SPECIAL_DESTINATIONS
                    )
                    if fed:
                        yield (
                            f"{type(model).__name__} lays {key!r}/{label!r} out over "
                            f"{walk!r}, where {name} feeds {fed}, which would get "
                            "nothing from a walk the cache serves whole"
                        )


def streamed_partitions(specs: Sequence[NodeResourceSpec], model: Model) -> frozenset[str]:
    """The partitions a node streams into when its label is keyed over several walks.

    A walk the cache serves streams nothing for the tokens it skips, so the
    conductor doesn't cache a request that runs one of these partitions.
    """
    graphs = model.get_graph_walk_graphs()
    partition_of = {walk: p.name for p in model.get_partitions() for walk in p.graph_walks}
    nodes_of = {spec.resource_key: spec.nodes for spec in specs}
    return frozenset(
        edge.target_partition
        for key, by_label in keyed_labels(model).items()
        for kv_label in by_label.values() if len(kv_label.keys.walks) > 1
        for walk in kv_label.keys.walks
        for name, node in graphs[walk].get_nodes().items() if name in nodes_of[key]
        for edge in node.outputs
        if isinstance(edge, StreamingGraphEdge) and edge.target_partition != partition_of[walk]
    )
