from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import torch

from mminf.communication.tensors import NameToTensorList


class EngineType(Enum):
    AR = "ar"
    FLOW = "flow"
    ENC_DEC = "enc_dec"
    AUDIO_CODEC = "audio_codec"


@dataclass
class NodeBatch:
    """Input to an engine's execute_batch()."""
    node_name: str
    graph_walk: str
    request_ids: list[str]

    # {request_id: {input_name: list[tensor]}}
    per_request_input_tensors: dict[str, NameToTensorList]
    metadata: dict = field(default_factory=dict)
    # {request_id: {key: value}} — per-request metadata (e.g., cache_label)
    per_request_metadata: dict[str, dict] = field(default_factory=dict)


@dataclass
class NodeOutput:
    """Output from an engine's execute_batch()."""
    # {request_id: {output_name: [tensor]}}
    per_request_output_tensors: dict[str, NameToTensorList]
    # {request_id: engine-specific metadata (e.g., generated token id)}
    per_request_metadata: dict[str, dict] = field(default_factory=dict)


class BaseEngine(ABC):
    @abstractmethod
    def engine_type(self) -> EngineType:
        ...

    @abstractmethod
    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
    ) -> None:
        """
        Receive the submodules this engine is responsible for
        (keyed by node name) and perform engine-specific initialization
        (KV cache allocation, FlashInfer workspace, etc.).
        """
        ...

    @abstractmethod
    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        ...

    @abstractmethod
    def add_request(self, request_id: str) -> None:
        ...

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        ...

    def warmup(self) -> None:
        """Optional CUDA graph capture. Override in subclasses."""
        return

    def shutdown(self) -> None:
        return
