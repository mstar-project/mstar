"""The streams a client feeds a live call: the model's declaration of them,
and the serving config's limits on them (``client_streams:``,
``max_call_input_mb``). The conductor and the API server read both as
checked here."""

import math
from collections.abc import Mapping
from typing import Any

from mstar.distributed.base import ShardingConfig
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model
from mstar.streaming.topology import ClientStreamLimits, Connection, find_client_stream

# The limits a deployment may override, per client stream.
_YAML_LIMITS = ("max_items",)


def check_client_streams(
    model: Model, config: Mapping[str, Any], sharding_config: ShardingConfig | None = None,
) -> list[str]:
    """Without a ``sharding_config`` (the deployment could not be read) only
    the shard_dim check is skipped."""
    mb = config.get("max_call_input_mb", 512)
    problems = []
    if isinstance(mb, bool) or not isinstance(mb, (int, float)) or not 0 < mb < math.inf:
        problems.append(f"max_call_input_mb must be a positive number of MiB, not {mb!r}")
    connections = model.get_partition_topology().client_connections()
    # the data worker sends every rank of a client stream's consumer the whole item
    sharded = sorted(
        conn.edge_name for conn in connections
        if sharding_config is not None and sharding_config.shard_dim.get(conn.edge_name) is not None
    )
    if sharded:
        problems.append(
            f"client stream(s) {sharded} have a shard_dim; a client stream reaches "
            "every rank of its consumer whole: drop its sharding_config.shard_dim entry"
        )
    return problems + _consumer_problems(model, connections) + _limit_problems(
        connections, config.get("client_streams") or {},
    )


def _consumer_problems(model: Model, connections: list[Connection]) -> list[str]:
    """What would otherwise fail every live call: each client stream needs
    the one node that consumes it, in the partition it flows into, and a
    ``ClientAudio`` stream needs text and audio emitted from inside a loop."""
    if not connections:
        return []
    problems = []
    # the data worker turns appends into ClientAudio items itself
    if type(model).process_stream_chunk is Model.process_stream_chunk and any(
            conn.audio is None for conn in connections):
        problems.append(
            f"{type(model).__name__} declares client streams without ClientAudio but does not implement "
            "process_stream_chunk"
        )
    names = [conn.edge_name for conn in connections]
    problems += [
        f"client stream {name!r} is declared twice; one consumer per stream"
        for name in sorted({name for name in names if names.count(name) > 1})
    ]
    walks = model.get_graph_walk_graphs()
    if any(conn.audio is not None for conn in connections):
        # /v1/realtime pairs a declared-audio model's text with its audio by the emitting loop's step
        loops = [loop for walk in walks.values() for loop in walk.get_loops().values()]
        looped = {name for loop in loops for name in loop.get_nodes()}
        nested = {name for loop in loops for name in loop.section.get_loops()}
        # a top-level loop's own outputs are emitted at its exit, outside any loop
        emitters = [(node.name, node.outputs) for walk in walks.values() for node in walk.get_nodes().values()
                    if node.name not in looped]
        emitters += [(loop.name, loop.outputs + loop.accumulated_outputs) for loop in loops if loop.name not in nested]
        unlooped = sorted({
            name for name, edges in emitters if any(
                edge.next_node == EMIT_TO_CLIENT and edge.output_modality in ("text", "audio") for edge in edges)
        })
        if unlooped:
            problems.append(f"{type(model).__name__} declares a ClientAudio client stream, so its text and audio "
                            f"must come from inside a loop; {unlooped} emit outside one")
    partitions = {p.name: p for p in model.get_partitions()}
    for conn in connections:
        try:
            find_client_stream(conn, walks, partitions)
        except ValueError as e:
            problems.append(str(e))
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
