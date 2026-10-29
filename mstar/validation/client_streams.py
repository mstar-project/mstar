"""The streams a client feeds a live call: the model's declaration of them,
and the serving config's limits on them (``client_streams:``,
``max_call_input_mb``). The conductor and the API server read both as
checked here."""

import math
from collections.abc import Mapping
from typing import Any

from mstar.model.base import Model
from mstar.streaming.topology import ClientStreamLimits, Connection

# The limits a deployment may override, per client stream.
_YAML_LIMITS = ("max_items",)


def check_client_streams(model: Model, config: Mapping[str, Any]) -> list[str]:
    mb = config.get("max_call_input_mb", 512)
    problems = []
    if isinstance(mb, bool) or not isinstance(mb, (int, float)) or not 0 < mb < math.inf:
        problems.append(f"max_call_input_mb must be a positive number of MiB, not {mb!r}")
    connections = model.get_partition_topology().client_connections()
    return problems + _consumer_problems(model, connections) + _limit_problems(
        connections, config.get("client_streams") or {},
    )


def _consumer_problems(model: Model, connections: list[Connection]) -> list[str]:
    """What would otherwise fail every live call: each client stream needs
    the one node that consumes it, in the partition it flows into."""
    if not connections:
        return []
    problems = []
    if type(model).process_stream_chunk is Model.process_stream_chunk:
        problems.append(
            f"{type(model).__name__} declares client streams but does not implement process_stream_chunk"
        )
    walks = model.get_graph_walk_graphs()
    partitions = {p.name: p for p in model.get_partitions()}
    seen = set()
    for conn in connections:
        name, partition = conn.edge_name, partitions.get(conn.to_partition)
        if name in seen:
            problems.append(f"client stream {name!r} is declared twice; one consumer per stream")
            continue
        seen.add(name)
        if partition is None:
            problems.append(f"client stream {name!r} flows into {conn.to_partition!r}, which is not a partition")
            continue
        # Unique over every walk, as the worker resolves a streamed edge's
        # consumer (Worker._build_consumer_node_cache), so both agree.
        consumers = {
            node.name for walk in walks.values() for node in walk.get_nodes().values()
            if name in node.input_names
        }
        if len(consumers) != 1 or not any(consumers & walks[w].get_nodes().keys() for w in partition.graph_walks):
            problems.append(
                f"client stream {name!r} needs exactly one consuming node, in "
                f"partition {partition.name!r}; found {sorted(consumers) or 'none'}"
            )
    return problems


def _limit_problems(connections: list[Connection], overrides: Any) -> list[str]:
    """A deployment may set a bound the model leaves open, or lower one it
    declares, but not raise or lift it: the model sized its consumer for that
    much (nemotron_duplex's session buffers and KV pools for
    ``max_session_s``). An unknown stream, key or value would otherwise be a
    silently ineffective setting."""
    if not isinstance(overrides, Mapping):
        return ["client_streams must map client stream names to their limits"]
    declared = {conn.edge_name: conn.limits or ClientStreamLimits() for conn in connections}
    problems = []
    unknown = sorted(overrides.keys() - declared.keys())
    if unknown:
        problems.append(
            f"serving config limits unknown client stream(s) {unknown}; this model takes {sorted(declared)}"
        )
    for name, base in declared.items():
        block = overrides.get(name) or {}
        if not isinstance(block, Mapping) or set(block) - set(_YAML_LIMITS):
            problems.append(f"client_streams.{name} takes {list(_YAML_LIMITS)}; got {block!r}")
            continue
        bound = block.get("max_items", base.max_items)
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 1):
            problems.append(f"client_streams.{name}.max_items must be a positive integer or null, not {bound!r}")
        elif base.max_items is not None and (bound is None or bound > base.max_items):
            problems.append(
                f"client_streams.{name}.max_items may lower the model's bound of {base.max_items}, "
                f"not raise or lift it ({bound!r})"
            )
    return problems
