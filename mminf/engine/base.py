from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo, SequenceInfo


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
    per_request_info: dict[str, CurrentForwardPassInfo] = field(default_factory=dict)

    # unused for now
    metadata: dict = field(default_factory=dict)


@dataclass
class NodeOutput:
    """Output from an engine's execute_batch()."""
    # {request_id: {output_name: [tensor]}}
    per_request_output_tensors: dict[str, NameToTensorList]


class BaseEngine(ABC):
    def __init__(self, enable_nvtx: bool = False, **kwargs):
        self.enable_nvtx = enable_nvtx

    @abstractmethod
    def engine_type(self) -> EngineType:
        ...

    @abstractmethod
    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
        **kwargs
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
    def add_request(self, request_id: str, **kwargs) -> None:
        ...

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        ...
    
    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
    ):
        return True

    def warmup(self) -> None:
        """Optional CUDA graph capture. Override in subclasses."""
        return

    def shutdown(self) -> None:
        return
