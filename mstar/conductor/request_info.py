from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from mstar.engine.resources import PublishedInfo, ResourceReqConfig

DEFAULT_PARTITION = "default"

# Step metadata the conductor sets on the step that samples a request's first
# token: a plan step whose next step's walk grows a KV label (`KVLabel.growth`).
FINAL_PREFILL = "final_prefill"


@dataclass
class CurrentForwardPassInfo:
    """
    Information that is passed into the worker / engines about this request
    at the current forward pass
    """
    request_id: str
    graph_walk: str
    fwd_index: int
    random_seed: int
    max_tokens: int

    # The worker-local integer handle for request_id, stamped by the worker on
    # receipt (GraphRuntime.add_request / get_rid_handle). Worker-internal state
    # keys on this; ``request_id`` stays the wire identity, so this object can
    # be forwarded to a peer without a copy. -1 means not yet stamped -- the
    # conductor and api server never set it, and a peer overwrites whatever
    # arrives with its own.
    rid_handle: int = -1

    # resource label -> the config this request's resources were opened
    # with. Sampling params, whether it needs CFG, retention: a request's
    # knobs live here now, one entry per resource that has any.
    resource_configs: dict[str, ResourceReqConfig] = field(default_factory=dict)
    step_metadata: dict = field(default_factory=dict)

    # resource label -> PublishedInfo
    resource_publish_info: dict[str, PublishedInfo] = field(default_factory=dict)

    # per_label_seq_info DEPRECATED
    # per_label_seq_info: PerLabelSeqInfo = field(default_factory=PerLabelSeqInfo)
    partition_name: str = field(default=DEFAULT_PARTITION)

    # Per-loop stop indices are worker-only, so they live on the graph runtime
    # rather than riding on this object across the wire.
    dynamic_loop_iter_counts: dict[str, int] = field(default_factory=dict)

    # stream edge -> items it starts with that are context only (a voice
    # clone's reference frames); primes the consumer's StreamBuffer. Make this
    # a per-stream dataclass if more per-request stream settings appear.
    stream_lead_items: dict[str, int] = field(default_factory=dict)

    # producer-triggered consumer partition -> the walk this pass's streamed
    # items run under; set by the producer worker as the pass emits
    stream_consumer_walks: dict[str, str] = field(default_factory=dict)

    def update_publish_info(self, other: dict[str, PublishedInfo]):
        merge_publish_info(self.resource_publish_info, other)


def merge_publish_info(
    into: dict[str, PublishedInfo], other: dict[str, PublishedInfo],
) -> None:
    """Fold one resource's published state into what is already held.

    Merging is the resource's own business (a KV cache folds in another rank's
    shard rather than replacing it), so an existing entry gets ``update`` and
    only a new key is taken wholesale.
    """
    for key, val in other.items():
        if key not in into:
            into[key] = val
        else:
            into[key].update(val)


# ---------------------------------------------------------------------------
# Partition types for async graph partitions
# ---------------------------------------------------------------------------

@dataclass
class PartitionDefinition:
    """Defines a partition within a model's computation graph.

    Each partition has its own set of graph walks and its own walk plan
    (`Model.walk_plan`), and can run asynchronously relative to other
    partitions.
    """
    name: str                                                   # e.g., "LLM", "SNAC"
    graph_walks: set[str]                                       # walks this partition uses


@dataclass(frozen=True)
class WalkStep:
    """One step of a partition's plan (`Model.walk_plan`): a walk to run.

    The conductor sends each input of the walk, other than a streamed one, the
    persisted signal of the same name, or an empty edge when there is none.
    ``bind`` renames: ``{"text_inputs": "new_token"}`` feeds a decode loop's
    ``text_inputs`` the prefill's ``new_token``. ``step_metadata`` reaches
    every node of the walk.
    """
    walk: str
    bind: Mapping[str, str] = MappingProxyType({})
    step_metadata: Mapping[str, Any] = MappingProxyType({})


@dataclass
class StreamingConnectionState:
    """Per-connection streaming state tracked by the conductor, for a stream
    between two partitions. A client stream has none: the data worker sends
    and ends it itself."""
    from_partition: str
    to_partition: str
    edge_name: str
    producer_done: bool = False
    # destination worker -> stream items the producer sent there, which
    # producer_done carries so the consumer waits for the last of them
    items_sent: dict[str, int] = field(default_factory=dict)


@dataclass
class PartitionState:
    """Per-partition conductor-level state for a request."""
    partition_name: str
    # the walk running: the plan's current step, or the walk a stream moved it to
    graph_walk: str = ""
    fwd_pass_number: int = 0
    random_seed: int = 0
    is_done: bool = False
    completed_worker_graph_ids: set[int] = field(default_factory=set)
    current_worker_graph_ids: set[int] = field(default_factory=set)
    # wg_id -> count of distinct TP ranks that have reported completion
    wg_rank_completions: dict[int, int] = field(default_factory=dict)
    # a report since the last advance set `partition_done` (it can be the
    # next pass's; see `Conductor._process_done_forward`)
    worker_reported_done: bool = False
    # the walks this request runs here (`Model.walk_plan`), and the step running
    plan: list[WalkStep] = field(default_factory=list)
    cursor: int = 0
    num_output_tokens: int = 0
    # resource label -> PublishedInfo, accumulated across all TP ranks and
    # handed back out on the next forward
    resource_publish_info: dict[str, PublishedInfo] = field(default_factory=dict)
