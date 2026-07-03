from dataclasses import dataclass, field
from typing import Callable

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


@dataclass
class Connection:
    """Defines a streaming connection between two partitions."""
    from_partition: str
    to_partition: str
    edge_name: str
    chunk_policy_factory: Callable[[], ChunkPolicy]


@dataclass
class PartitionTopology:
    """Declares how a model's computation is split into async partitions.

    Each partition has its own set of graph walks. Connections define
    streaming data flow between partitions via StreamBuffers.
    """
    partitions: list[str]
    connections: list[Connection] = field(default_factory=list)
