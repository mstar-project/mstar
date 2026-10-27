from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import ChunkPolicy

if TYPE_CHECKING:
    from mstar.model.base import Model


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
# no walks, worker or partition state; the API server's data worker sends its
# items to the consumer's ranks and ends it when the client's input ends.
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
    config's ``client_streams: {<edge name>: {max_items}}`` applied, as
    ``validate_deployment`` accepted it (see
    ``mstar.validation.client_streams``)."""
    overrides = (model_config or {}).get("client_streams") or {}
    return {
        conn.edge_name: (conn.limits or ClientStreamLimits())._replace(**(overrides.get(conn.edge_name) or {}))
        for conn in topology.client_connections()
    }


@dataclass(frozen=True, slots=True)
class ClientStream:
    """Where a client stream's items go: the node that consumes the edge, a
    walk it runs in (for routing to its workers) and its partition. A
    dataclass, which the wire codec carries in the conductor's route and a
    NamedTuple it would refuse."""
    node: str
    graph_walk: str
    partition: str


def find_client_streams(model: Model) -> dict[str, ClientStream]:
    """Each client stream's consumer, in the partition it flows into, as
    validate_deployment checked. The conductor derives it, after the worker
    graphs are built, and sends it to the data worker with each call's route."""
    connections = model.get_partition_topology().client_connections()
    if not connections:
        return {}
    walks = model.get_graph_walk_graphs()
    partitions = {p.name: p for p in model.get_partitions()}
    streams: dict[str, ClientStream] = {}
    for conn in connections:
        name, partition = conn.edge_name, partitions[conn.to_partition]
        (consumer,) = {
            node.name for walk in walks.values() for node in walk.get_nodes().values()
            if name in node.input_names
        }
        walk = max(walk for walk in partition.graph_walks if consumer in walks[walk].get_nodes())
        streams[name] = ClientStream(consumer, walk, partition.name)
    return streams
