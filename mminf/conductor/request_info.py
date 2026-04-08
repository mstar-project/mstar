from dataclasses import dataclass, field


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
    requires_cfg: bool = field(default=False)
    kwargs: dict = field(default_factory=dict)


@dataclass
class SequenceInfo:
    seq_len: int
    pos_id: int

    # for tracking KV cache
    latest_entity_id: str = ""
    latest_session_id: str = ""
    kv_cache_addr: int = -1
    page_indices: list[int] = field(default_factory=list)


@dataclass
class PerLabelSeqInfo:
    # {label: SequenceInfo}
    info: dict[str, SequenceInfo] = field(default_factory=dict)

    def update(self, other: "PerLabelSeqInfo"):
        self.info= {
            **self.info,
            **other.info
        }
        print(self.info)

    def get_all(self) -> dict:
        return self.info


@dataclass
class CurrentForwardPassInfo:
    """
    Information that is passed into the worker / engines about this request
    at the current forward pass
    """
    graph_walk: str
    requires_cfg: bool
    fwd_index: int
    random_seed: int
    step_metadata: dict = field(default_factory=dict)
    per_label_seq_info: PerLabelSeqInfo= field(default_factory=PerLabelSeqInfo)
    partition_name: str = field(default="default")


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
    new_tokens: dict[str, list[int]] = field(default_factory=dict)
    completed_worker_graph_ids: set[str] = field(default_factory=set)
    current_worker_graph_ids: set[str] = field(default_factory=set)
    num_output_tokens: int = 0
    curr_forward_outputs: list[str] = field(default_factory=list)
    per_label_seq_info: PerLabelSeqInfo = field(default_factory=PerLabelSeqInfo)
