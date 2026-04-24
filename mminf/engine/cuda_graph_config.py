
from abc import ABC, abstractmethod
from enum import Enum

import torch

from mminf.model.submodule_base import ARNodeInputs, ARNodeSubmodule


class CudaGraphConfigType(Enum):
    BASIC_BATCHED = "basic_batched"
    FLASH_INFER_PACKED = "flash_infer_packed"


class CudaGraphConfig(ABC):
    def __init__(
        self,
        capture_graph_walk: str,  # "decode"
        replay_graph_walks: list[str] | None = None, # set to None to be just capture_graph_walk
        requires_cfg: bool = False,
        labels: list[str]  = None,  # cache labels used: ["main"] or ["main", "cfg_img"]
        compile: bool = True, # whether to run torch.compile on the submodule before cuda graph capture
        # Per-config override for the set of batch sizes to capture. None → use the
        # runner's default (AR engine default: DEFAULT_AR_CAPTURE_BATCH_SIZES;
        # CodecCudaGraphRunner picks its own default). Useful for codec-style
        # submodules where memory cost per size is high, or for AR walks where a
        # small subset is enough.
        capture_batch_sizes: list[int] | None = None
    ):
        self.capture_graph_walk = capture_graph_walk
        self.replay_graph_walks = replay_graph_walks or [capture_graph_walk]
        self.requires_cfg = requires_cfg
        self.labels = labels or ["main"]
        self.compile = compile
        self.capture_batch_sizes = capture_batch_sizes
    
    @abstractmethod
    def get_config_type(self) -> CudaGraphConfigType:
        pass

    @abstractmethod
    def get_total_tokens(self, bs: int) -> list[int]:
        pass


class BasicBatchedCudaGraphConfig(CudaGraphConfig):
    def __init__(
        self,
        capture_graph_walk: str,
        single_request_inputs: ARNodeInputs,
        replay_graph_walks: list[str] | None = None,
        requires_cfg: bool = False,
        labels: list[str]  = None,
        compile: bool = True,
        capture_batch_sizes: list[int] | None = None
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            requires_cfg=requires_cfg,
            labels=labels,
            compile=compile,
            capture_batch_sizes=capture_batch_sizes
        )
        self.single_request_inputs = single_request_inputs
    
    def get_config_type(self) -> CudaGraphConfigType:
        return CudaGraphConfigType.BASIC_BATCHED
    
    def get_total_tokens(self, bs: int) -> list[int]:
        return [self.single_request_inputs.input_seq_len * bs]


class FlashInferPackedCudaGraphConfig(CudaGraphConfig):
    def __init__(
        self,
        capture_graph_walk: str,
        packed_seq_len_to_inputs: dict[str, dict[str, torch.Tensor]],
        replay_graph_walks: list[str] | None = None,
        requires_cfg: bool = False,
        labels: list[str]  = None,
        compile: bool = True,
        causal_attention: bool = True,
        capture_batch_sizes: list[int] | None = None
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            requires_cfg=requires_cfg,
            labels=labels,
            compile=compile,
            capture_batch_sizes=capture_batch_sizes
        )
        self.num_token_to_inputs = packed_seq_len_to_inputs
        self.causal_attention = causal_attention

    def get_config_type(self) -> CudaGraphConfigType:
        return CudaGraphConfigType.FLASH_INFER_PACKED

    def get_total_tokens(self, bs: int) -> list[int]:
        return list(self.num_token_to_inputs.keys())