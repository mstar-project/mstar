
from abc import ABC, abstractmethod
from enum import Enum

import torch

from mstar.model.submodule_base import ARNodeInputs


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
        # StatelessCudaGraphRunner picks its own default). Useful for codec-style
        # submodules where memory cost per size is high, or for AR walks where a
        # small subset is enough.
        capture_batch_sizes: list[int] | None = None,
        # Method on the submodule to capture. Defaults to ``forward_batched`` (the
        # same method the eager batched path uses). Diffusion-style walks that must
        # keep a non-capturable tail (e.g. a multistep scheduler step) out of the
        # graph capture a velocity-only method here and run the tail in
        # ``postprocess_captured`` after replay.
        capture_forward_method: str = "forward_batched",
        # Whether the runner advances KV seq_lens after replay. True for
        # autoregressive walks (each step appends a token). False for frozen-prefix
        # denoise loops that re-read a fixed prefix and overwrite the same tail
        # pages every step (advancing would grow the prefix and corrupt attention).
        advance_seq_lens: bool = True,
        # Whether this config's captured batch sizes also cap the engine's max
        # (eager) batch size for the walk. Default True keeps the conservative
        # behavior: never batch beyond a captured graph size. Set False when the
        # captured sizes are only an acceleration subset and the submodule's eager
        # batched path can handle larger batches — the engine then honors the
        # submodule's max_batch_size and uses a graph only when the exact batch
        # size was captured (gated by runner.can_run), falling back to eager
        # batched execution otherwise. Needed so a denoise loop that captures a
        # graph only at batch size 1 (single-request latency) can still batch
        # concurrent requests instead of serializing them.
        caps_eager_batch_size: bool = True,
    ):
        self.capture_graph_walk = capture_graph_walk
        self.replay_graph_walks = replay_graph_walks or [capture_graph_walk]
        self.requires_cfg = requires_cfg
        self.labels = labels or ["main"]
        self.compile = compile
        self.capture_batch_sizes = capture_batch_sizes
        self.capture_forward_method = capture_forward_method
        self.advance_seq_lens = advance_seq_lens
        self.caps_eager_batch_size = caps_eager_batch_size

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
        capture_batch_sizes: list[int] | None = None,
        capture_forward_method: str = "forward_batched",
        advance_seq_lens: bool = True,
        caps_eager_batch_size: bool = True,
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            requires_cfg=requires_cfg,
            labels=labels,
            compile=compile,
            capture_batch_sizes=capture_batch_sizes,
            capture_forward_method=capture_forward_method,
            advance_seq_lens=advance_seq_lens,
            caps_eager_batch_size=caps_eager_batch_size,
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
        capture_batch_sizes: list[int] | None = None,
        zero_padding_input: ARNodeInputs | None = None,
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
        self.zero_padding_input = zero_padding_input

    def get_config_type(self) -> CudaGraphConfigType:
        return CudaGraphConfigType.FLASH_INFER_PACKED

    def get_total_tokens(self, bs: int) -> list[int]:
        return list(self.num_token_to_inputs.keys())
