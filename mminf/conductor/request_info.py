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
    per_label_seq_info: dict[str, SequenceInfo]  = field(default_factory=dict)
    