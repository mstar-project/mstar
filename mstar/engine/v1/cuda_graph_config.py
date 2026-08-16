from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch

from mstar.model.submodule_base import ARNodeInputs, NodeInputs


class CudaGraphConfigType(Enum):
    BASIC_BATCHED = "basic_batched"
    FLASH_INFER_PACKED = "flash_infer_packed"


class CudaGraphConfig(ABC):
    def __init__(
        self,
        capture_graph_walk: str,  # "decode"
        replay_graph_walks: list[str] | None = None, # set to None to be just capture_graph_walk

        # Additional information added to the capture bucket key (e.g., requires_cfg).
        # Must be hashable.
        additional_key_info: Any | None = None,
        compile: bool = True, # whether to run torch.compile before cuda graph capture

        # Per-config override for the set of batch sizes to capture
        capture_batch_sizes: list[int] | None = None,
        # Method on the submodule to capture
        capture_forward_method: str = "forward_batched",
        # Whether this config's captured batch sizes also cap the engine's max
        # (eager) batch size for the walk. Default True keeps the conservative
        # behavior: never batch beyond a captured graph size.
        caps_eager_batch_size: bool = True,
    ):
        self.capture_graph_walk = capture_graph_walk
        self.replay_graph_walks = replay_graph_walks or [capture_graph_walk]
        self.additional_key_info = additional_key_info
        self.compile = compile
        self.capture_batch_sizes = capture_batch_sizes
        self.capture_forward_method = capture_forward_method
        self.caps_eager_batch_size = caps_eager_batch_size

    @abstractmethod
    def get_config_type(self) -> CudaGraphConfigType:
        pass

    @abstractmethod
    def get_total_tokens(self, bs: int) -> list[int]:
        pass

    @abstractmethod
    def get_node_inputs(self, bs: int, num_tokens: int) -> list[ARNodeInputs]:
        pass


class BatchedCudaGraphConfig(CudaGraphConfig):
    def __init__(
        self,
        capture_graph_walk: str,  # "decode"
        single_request_inputs: ARNodeInputs,
        replay_graph_walks: list[str] | None = None,
        additional_key_info: Any | None = None,
        compile: bool = True,
        capture_batch_sizes: list[int] | None = None,
        capture_forward_method: str = "forward_batched",
        caps_eager_batch_size: bool = True,
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            additional_key_info=additional_key_info,
            compile=compile,
            capture_batch_sizes=capture_batch_sizes,
            capture_forward_method=capture_forward_method,
            caps_eager_batch_size=caps_eager_batch_size,
        )
        self.single_request_inputs = single_request_inputs

    def get_config_type(self) -> CudaGraphConfigType:
        return CudaGraphConfigType.BASIC_BATCHED

    def get_total_tokens(self, bs: int) -> list[int]:
        return [self.single_request_inputs.input_seq_len * bs]

    def get_node_inputs(self, bs: int, num_tokens: int):
        del num_tokens
        return [
            self.single_request_inputs.clone() for _  in range(bs)
        ]


def distribute_tokens(total_tokens: int, bs: int) -> list[int]:
    """Split ``total_tokens`` across ``bs`` requests, remainder on the first.
    """
    seq_lens = [total_tokens // bs] * bs
    seq_lens[0] += total_tokens % bs
    return seq_lens


class PackedCudaGraphConfig(CudaGraphConfig):
    def __init__(
        self,
        capture_graph_walk: str,
        capture_token_lengths: list[int],
        # seq len -> ARNodeInputs
        make_node_input: Callable[[int], ARNodeInputs],
        replay_graph_walks: list[str] | None = None,
        additional_key_info: Any | None = None,
        compile: bool = True,
        capture_batch_sizes: list[int] | None = None,
        capture_forward_method: str = "forward_batched",
        caps_eager_batch_size: bool = True
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            additional_key_info=additional_key_info,
            compile=compile,
            capture_batch_sizes=capture_batch_sizes,
            capture_forward_method=capture_forward_method,
            caps_eager_batch_size=caps_eager_batch_size
        )
        self.make_node_input = make_node_input
        self.capture_token_lengths = capture_token_lengths

    def get_config_type(self) -> CudaGraphConfigType:
        return CudaGraphConfigType.FLASH_INFER_PACKED

    def get_total_tokens(self, bs: int) -> list[int]:
        return self.capture_token_lengths

    def get_node_inputs(self, bs: int, num_tokens: int):
        seq_lens = distribute_tokens(num_tokens, bs)
        return [
            self.make_node_input(n) for n in seq_lens
        ]