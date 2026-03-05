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
class StageBatch:
    """Input to an engine's execute_batch()."""
    stage_name: str
    phase: str
    request_ids: list[str]

    # {request_id: {input_name: list[tensor]}}
    per_request_input_tensors: dict[str, NameToTensorList]
    metadata: dict = field(default_factory=dict)


@dataclass
class StageOutput:
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
        Receive the nn.Module submodules this engine is responsible for
        (keyed by stage name) and perform engine-specific initialization
        (KV cache allocation, FlashInfer workspace, etc.).
        """
        ...

    @abstractmethod
    def execute_batch(self, batch: StageBatch) -> StageOutput:
        ...

    def execute_single_request(
        self,
        submodule: torch.nn.Module | None,
        input_tensors: NameToTensorList,
        **kwargs,
    ) -> NameToTensorList:
        """
        Execute a single request through a submodule. Called by Model.step().
        Default: uses preprocess/forward pattern if submodule has preprocess(),
        otherwise falls back to direct call.
        Override for engine-specific behavior (e.g., KV cache management).
        """
        if submodule is None:
            return {}
        with torch.no_grad():
            if hasattr(submodule, 'preprocess'):
                preprocessed = submodule.preprocess(**input_tensors)
                return submodule(**preprocessed, **kwargs)
            return submodule(input_tensors, **kwargs)

    @abstractmethod
    def add_request(self, request_id: str) -> None:
        ...

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        ...

    def warmup(self) -> None:
        """Optional CUDA graph capture. Override in subclasses."""
        pass

    def shutdown(self) -> None:
        pass
