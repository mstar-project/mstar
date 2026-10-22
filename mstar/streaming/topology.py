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


# What a client stream past its max_pending_items does with the next item.
PENDING_OVERFLOW = ("refuse", "keep_newest")


class ClientStreamLimits(NamedTuple):
    """Bounds on what the client feeds one client stream; None is unbounded.
    A bound the model declares is the most its consumer can take; a
    deployment may lower it (``client_streams:`` in the serving config, see
    :func:`client_stream_limits`)."""
    # What the stream may carry over the call's life, each item counting
    # item_size(item): a call past it ends with a 400.
    max_items: int | None = None
    # Items (not item_size units) the consumer has been sent and not consumed
    # yet, at once.
    max_pending_items: int | None = None
    # "refuse": an item past max_pending_items ends the call with a 400.
    # "keep_newest": the items that fit are sent; of the rest the conductor
    # holds only the newest, in one slot, until the consumer has room or the
    # stream ends, and drops the others (and any item the slot held before).
    pending_overflow: str = "refuse"
    # How much of max_items one item uses: 1 when None. A stream of
    # variable-length chunks counts their frames or samples.
    item_size: Callable[[Any], int] | None = None
    # what item_size counts, for the error a call past max_items gets
    unit: str = "items"


# The limits a deployment may override, per client stream.
_YAML_LIMITS = ("max_items", "max_pending_items", "pending_overflow")


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
    config's ``client_streams: {<edge name>: {max_items, max_pending_items,
    pending_overflow}}`` applied. A deployment may set a bound the model
    leaves open, or lower one it declares, but not raise or lift it: the
    model sized its consumer for that much (nemotron_duplex's session
    buffers and KV pools for ``max_session_s``). An unknown stream, key or
    value is an error: it would otherwise be a silently ineffective
    setting. So is a ``max_pending_items`` below what the stream's chunk
    policy waits for: the consumer would never take a chunk, so it would
    never make room."""
    conns = {conn.edge_name: conn for conn in topology.client_connections()}
    declared = {name: conn.limits or ClientStreamLimits() for name, conn in conns.items()}
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
        limits[name] = _checked_limits(name, base._replace(**block))
        for key in ("max_items", "max_pending_items"):
            declared_bound, bound = getattr(base, key), getattr(limits[name], key)
            if declared_bound is not None and (bound is None or bound > declared_bound):
                raise ValueError(
                    f"client_streams.{name}.{key} may lower the model's bound of {declared_bound}, "
                    f"not raise or lift it ({bound!r})"
                )
        _check_pending_fits_a_chunk(name, limits[name], conns[name])
    return limits


def _check_pending_fits_a_chunk(name: str, limits: ClientStreamLimits, conn: Connection) -> None:
    pending = limits.max_pending_items
    if pending is None:
        return
    window = conn.chunk_policy_factory().max_window_size()
    if window is None:
        raise ValueError(
            f"the {name!r} stream's chunks grow without bound, so no max_pending_items ({pending}) "
            f"fits one (client_streams.{name}.max_pending_items)"
        )
    if pending < window:
        raise ValueError(
            f"the {name!r} stream's max_pending_items ({pending}) is below the {window} items its "
            f"chunk policy waits for before it releases a chunk, so its consumer would never take "
            f"one (client_streams.{name}.max_pending_items)"
        )


def _checked_limits(name: str, limits: ClientStreamLimits) -> ClientStreamLimits:
    for key in ("max_items", "max_pending_items"):
        value = getattr(limits, key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"client_streams.{name}.{key} must be a positive integer or null, not {value!r}")
    if limits.pending_overflow not in PENDING_OVERFLOW:
        raise ValueError(
            f"client_streams.{name}.pending_overflow must be one of {list(PENDING_OVERFLOW)}, "
            f"not {limits.pending_overflow!r}"
        )
    return limits
