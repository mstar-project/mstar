from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import ChunkPolicy


@dataclass
class StreamingGraphEdge(GraphEdge):
    """A graph edge that carries streaming data between partitions.

    Routed like a normal GraphEdge (producer is unaware it's streaming).
    On the consumer worker, the arriving tensors are buffered in a
    StreamBuffer and gated by a ChunkPolicy before satisfying the
    consuming node's input.
    """
    target_partition: str = ""

    def __post_init__(self):
        self.is_streaming = True


# The producer of a stream the client feeds into a running request:
# ``Connection(from_partition=CLIENT, ...)``. It names no partition, so it has
# no walks, worker or partition state; the API server appends its chunks and
# ends it when the client's input ends.
CLIENT = "client"


class ClientStreamLimits(NamedTuple):
    """Bounds on what the client feeds one client stream; None is unbounded.
    A bound the model declares is the most its consumer can take; a
    deployment may lower it (``client_streams:`` in the serving config, see
    :func:`client_stream_limits`)."""
    # What the stream may carry over the call's life, each item counting
    # item_size(item): a call past it ends with a 400.
    max_items: int | None = None
    # How much of max_items one item uses: 1 when None. A stream of
    # variable-length chunks counts their frames or samples.
    item_size: Callable[[Any], int] | None = None
    # what item_size counts, for the error a call past max_items gets
    unit: str = "items"


# The limits a deployment may override, per client stream.
_YAML_LIMITS = ("max_items",)


@dataclass
class Connection:
    """Defines a streaming connection between two partitions, or from the
    client (``from_partition=CLIENT``) into one. Only a client stream takes
    ``limits``: a partition's output is bounded by the model itself."""
    from_partition: str
    to_partition: str
    edge_name: str
    chunk_policy_factory: Callable[[], ChunkPolicy]
    limits: ClientStreamLimits | None = None


@dataclass
class PartitionTopology:
    """Declares how a model's computation is split into async partitions.

    Each partition has its own set of graph walks. Connections define
    streaming data flow between partitions via StreamBuffers.
    """
    partitions: list[str]
    connections: list[Connection] = field(default_factory=list)

    def __post_init__(self):
        if CLIENT in self.partitions:
            raise ValueError(f"{CLIENT!r} is reserved for client streams; rename the partition")
        for conn in self.connections:
            if conn.to_partition == CLIENT:
                raise ValueError(f"the client only produces streams; {conn.edge_name!r} flows into it")
            if conn.limits is not None and conn.from_partition != CLIENT:
                raise ValueError(f"{conn.edge_name!r} has limits but is not a client stream")

    def client_connections(self) -> list[Connection]:
        """The streams the client feeds."""
        return [conn for conn in self.connections if conn.from_partition == CLIENT]


def client_stream_limits(
    topology: PartitionTopology, model_config: Mapping | None,
) -> dict[str, ClientStreamLimits]:
    """Each client stream's limits: the model's declaration, with the serving
    config's ``client_streams: {<edge name>: {max_items}}`` applied. A
    deployment may set a bound the model leaves open, or lower one it
    declares, but not raise or lift it: the model sized its consumer for
    that much (nemotron_duplex's session buffers and KV pools for
    ``max_session_s``). An unknown stream, key or value is an error: it
    would otherwise be a silently ineffective setting."""
    declared = {conn.edge_name: conn.limits or ClientStreamLimits() for conn in topology.client_connections()}
    overrides = (model_config or {}).get("client_streams") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("client_streams must map client stream names to their limits")
    unknown = sorted(overrides.keys() - declared.keys())
    if unknown:
        raise ValueError(
            f"serving config limits unknown client stream(s) {unknown}; this model takes {sorted(declared)}"
        )
    limits = {}
    for name, base in declared.items():
        block = overrides.get(name) or {}
        if not isinstance(block, Mapping) or set(block) - set(_YAML_LIMITS):
            raise ValueError(f"client_streams.{name} takes {list(_YAML_LIMITS)}; got {block!r}")
        limits[name] = base._replace(**block)
        bound = limits[name].max_items
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 1):
            raise ValueError(f"client_streams.{name}.max_items must be a positive integer or null, not {bound!r}")
        if base.max_items is not None and (bound is None or bound > base.max_items):
            raise ValueError(
                f"client_streams.{name}.max_items may lower the model's bound of {base.max_items}, "
                f"not raise or lift it ({bound!r})"
            )
    return limits
