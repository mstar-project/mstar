from dataclasses import dataclass, field
from typing import Callable

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


@dataclass(slots=True)
class ProducerWalkCtx:
    """The producer's state for one pass that emits on a walk-driving edge.

    The producer worker refills one instance for every call, since the hook
    runs per request on every emitting step: valid only during the call.
    """
    producer_walk: str = ""
    # This pass's index among the producer worker's consecutive passes in
    # producer_walk that emitted on the edge; re-entering a walk restarts at 0
    pass_in_walk: int = 0
    # The walk this producer worker last assigned the consumer; None before
    # its first emission
    consumer_walk: str | None = None
    # The producer's forward-pass info for this pass (step_metadata and all);
    # the hook runs on the producer, so nothing here crosses the wire
    fwd_info: CurrentForwardPassInfo | None = None


@dataclass
class Connection:
    """Defines a streaming connection between two partitions."""
    from_partition: str
    to_partition: str
    edge_name: str
    chunk_policy_factory: Callable[[], ChunkPolicy]
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
