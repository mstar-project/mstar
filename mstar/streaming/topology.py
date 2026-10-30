from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import ChunkPolicy, FixedChunkPolicy


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


@dataclass(slots=True)
class ProducerWalkCtx:
    """The producer's state for one pass that emits on a walk-driving edge.

    The producer worker refills one instance for every call, since the hook
    runs per request on every emitting step: valid only during the call.
    """
    producer_walk: str = ""
    # The node whose completed pass emitted the items
    node: str = ""
    # Dynamic loop name -> the iteration this pass ran, for the loops it is in;
    # the graph's own count, so every node and worker of the producer agrees
    loop_iters: Mapping[str, int] = field(default_factory=dict)
    # The producer's forward-pass info for this pass (step_metadata and all);
    # the hook runs on the producer, so nothing here crosses the wire
    fwd_info: CurrentForwardPassInfo | None = None


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
    # Dim to join a chunked producer's pieces on, or None to keep them apart.
    # A node whose prefill is chunked emits one piece per chunk, but its
    # consumer's ChunkPolicy counts ITEMS -- so to a FixedChunkPolicy(1) the
    # pieces look like separate items and it reads only the first. Joining them
    # here, on the consumer, keeps the item the consumer already expects. Done
    # on arrival rather than at the producer deliberately: the consumer owns
    # these tensors outright, so there is no cross-stream ordering to get
    # wrong, which is where the producer-side join went wrong before.
    join_dim: int | None = None
    # Set on the connections into a producer-triggered partition. The
    # producer worker calls it for each pass that emits on the edge, and the
    # pass's items run under the consumer walk it returns: the producer, which
    # owns the consumer's walk, decides it, and the consumer only applies it.
    consumer_walk: Callable[[ProducerWalkCtx], str] | None = None


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
            if conn.consumer_walk is not None and conn.from_partition == CLIENT:
                # the hook runs on the producer worker, and the client has none
                raise ValueError(f"client stream {conn.edge_name!r} cannot drive its consumer's walk")

    def client_connections(self) -> list[Connection]:
        """The streams the client feeds."""
        return [conn for conn in self.connections if conn.from_partition == CLIENT]

    def producer_triggered_partitions(self) -> set[str]:
        """Partitions whose graph walk the stream drives, not the conductor."""
        return {
            conn.to_partition for conn in self.connections
            if conn.consumer_walk is not None
        }

    def walk_drivers(self) -> dict[str, Connection]:
        """Producer-triggered partition -> the connection whose consumer_walk
        decides its walk (any one of them; check_walk_driving_connections
        makes them agree)."""
        return {
            conn.to_partition: conn for conn in self.connections
            if conn.consumer_walk is not None
        }

    def check_walk_driving_connections(self) -> None:
        """A producer-triggered partition has one authority over its walk: all
        its incoming connections come from one producer and share one
        consumer_walk, so every edge's items agree on the walk. Its chunks
        must not overlap either, since a chunk runs under exactly one walk."""
        drivers = self.walk_drivers()
        for conn in self.connections:
            driver = drivers.get(conn.to_partition)
            if driver is None:
                continue
            if (conn.from_partition != driver.from_partition
                    or conn.consumer_walk is not driver.consumer_walk):
                raise ValueError(
                    f"Partition {conn.to_partition!r} is producer-triggered, "
                    f"so its incoming edge {conn.edge_name!r} must come from "
                    f"{driver.from_partition!r} and share edge "
                    f"{driver.edge_name!r}'s consumer_walk"
                )
            policy = conn.chunk_policy_factory()
            if not isinstance(policy, FixedChunkPolicy):
                raise ValueError(
                    f"Edge {conn.edge_name!r} drives its consumer's walk, so it "
                    f"needs non-overlapping chunks (FixedChunkPolicy), not "
                    f"{type(policy).__name__}"
                )


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
