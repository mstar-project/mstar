from dataclasses import dataclass, field
from typing import Callable, NamedTuple

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


class WalkTransitionCtx(NamedTuple):
    """What a chunk tells its consumer about the producer's progress."""
    # The producer's graph walk when it emitted the chunk's items
    producer_walk: str
    # Whether this is the first chunk of a run of that walk; a walk the
    # producer re-enters (e.g. a later decode turn) starts a new run
    starts_producer_walk: bool


@dataclass
class Connection:
    """Defines a streaming connection between two partitions."""
    from_partition: str
    to_partition: str
    edge_name: str
    chunk_policy_factory: Callable[[], ChunkPolicy]
    # Set on the connections into a producer-triggered partition: maps each
    # chunk to the consumer walk it runs under, so the consumer moves through
    # its walks in step with the producer instead of on conductor triggers.
    # It sees only the stream, so every connection into one partition maps
    # the same items to the same walk.
    consumer_walk: Callable[[WalkTransitionCtx], str] | None = None


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

    def check_walk_driving_connections(self) -> None:
        """Every connection into a producer-triggered partition must map its
        chunks to walks, or its chunks would land in whatever walk the others
        chose; and its chunks must not overlap, since a chunk spans exactly
        one producer walk."""
        driven = self.producer_triggered_partitions()
        for conn in self.connections:
            if conn.to_partition not in driven:
                continue
            if conn.consumer_walk is None:
                raise ValueError(
                    f"Partition {conn.to_partition!r} is producer-triggered, "
                    f"but its incoming edge {conn.edge_name!r} has no "
                    "consumer_walk"
                )
            policy = conn.chunk_policy_factory()
            if not isinstance(policy, FixedChunkPolicy):
                raise ValueError(
                    f"Edge {conn.edge_name!r} drives its consumer's walk, so it "
                    f"needs non-overlapping chunks (FixedChunkPolicy), not "
                    f"{type(policy).__name__}"
                )
