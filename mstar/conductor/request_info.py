from dataclasses import dataclass, field

from mstar.engine.resources import PublishedInfo, ResourceReqConfig
from mstar.graph.loop_indices import NestedLoopIndices


@dataclass
class CurrentForwardConductorMetadata:
    """
    Full-model forward pass-level metadata for running the current
    forward pass. On the conductor/model level.
    """
    graph_walk: str
    is_prefill: bool
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


DEFAULT_PARTITION = "default"


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

    # Per-loop stop indices; stop decisions come from each submodule's check_stop.
    loop_stop_times: dict[str, NestedLoopIndices] = field(default_factory=dict)
    dynamic_loop_iter_counts: dict[str, int] = field(default_factory=dict)

    def clear_loop_stop_info(self):
        self.loop_stop_times.clear()
        self.dynamic_loop_iter_counts.clear()

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

    Each partition has its own set of graph walks and transition logic,
    and can run asynchronously relative to other partitions.
    """
    name: str                                                   # e.g., "LLM", "SNAC"
    graph_walks: set[str]                                       # walks this partition uses
    initial_walk: str | None = None                             # first walk, or None = triggered later
    producer_partitions: list[str] = field(default_factory=list)  # partitions feeding tokens to this one


@dataclass
class StreamingConnectionState:
    """Per-connection streaming state tracked by the conductor."""
    from_partition: str
    to_partition: str
    edge_name: str
    token_count: int = 0
    consumed_count: int = 0
    producer_done: bool = False


@dataclass
class PartitionState:
    """Per-partition conductor-level state for a request."""
    partition_name: str
    metadata: CurrentForwardConductorMetadata
    fwd_pass_number: int = 0
    random_seed: int = 0
    is_done: bool = False
    completed_worker_graph_ids: set[str] = field(default_factory=set)
    current_worker_graph_ids: set[str] = field(default_factory=set)
    # wg_id -> count of distinct TP ranks that have reported completion
    wg_rank_completions: dict[str, int] = field(default_factory=dict)
    num_output_tokens: int = 0
    curr_forward_outputs: list[str] = field(default_factory=list)
    # resource label -> PublishedInfo, accumulated from the rank-0 worker's
    # reports and handed back out on the next forward
    resource_publish_info: dict[str, PublishedInfo] = field(default_factory=dict)
