from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import torch


class EngineType(Enum):
    AR = "ar"
    FLOW = "flow"
    ENC_DEC = "enc_dec"


@dataclass
class StageBatch:
    """Input to an engine's execute_batch()."""
    stage_name: str
    phase: str
    request_ids: list[str]
    # {request_id: {input_name: tensor}}
    per_request_input_tensors: dict[str, dict[str, torch.Tensor]]
    metadata: dict = field(default_factory=dict)


@dataclass
class StageOutput:
    """Output from an engine's execute_batch()."""
    # {request_id: {output_name: tensor}}
    per_request_output_tensors: dict[str, dict[str, torch.Tensor]]
    # {request_id: engine-specific metadata (e.g., generated token id)}
    per_request_metadata: dict[str, dict] = field(default_factory=dict)


class BaseEngine(ABC):
    @abstractmethod
    def engine_type(self) -> EngineType:
        ...

    @abstractmethod
    def load_model(self, model_config: dict, device: torch.device) -> None:
        ...

    @abstractmethod
    def execute_batch(self, batch: StageBatch) -> StageOutput:
        ...

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
