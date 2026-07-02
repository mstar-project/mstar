from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import torch

from mstar.model.submodule_base import ARNodeInputs

if TYPE_CHECKING:
    from mstar.engine.cache_manager import BatchedCacheManager


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


# ---------------------------------------------------------------------------
# Piecewise CUDA graph configs
#
# These configure ``PiecewiseCudaGraphRunner``, which captures ONE inner
# callable of a submodule's forward (e.g. a transformer block loop) as a CUDA
# graph while the surrounding preamble/postamble stays eager. They intentionally
# mirror the ``BASIC_BATCHED`` / ``FLASH_INFER_PACKED`` split above so the two
# runners share vocabulary:
#   - ``BATCHED`` captures ``[bs, seq_len, D]`` inputs, all sequences equal length.
#   - ``PACKED`` captures ``[total_tokens, D]`` inputs, variable-length sequences
#     packed together (FlashInfer-packed style).
# ---------------------------------------------------------------------------


def distribute_tokens(total_tokens: int, bs: int) -> list[int]:
    """Split ``total_tokens`` across ``bs`` requests, remainder on the first.

    Used to synthesize per-request capture-time seq_lens for a PACKED bucket
    (the runner only needs *a* valid partition to plan the dummy attention; the
    real per-request seq_lens arrive through ``PiecewiseCudaGraphRunner.run``).
    Mirrors ``CudaGraphRunner._make_dummy_seq_lens`` so both paths partition the
    same way.
    """
    seq_lens = [total_tokens // bs] * bs
    seq_lens[0] += total_tokens % bs
    return seq_lens


class PiecewiseConfigType(Enum):
    BATCHED = "batched"          # [bs, seq_len, D], uniform seq_lens
    PACKED = "packed"            # [total_tokens, D], variable seq_lens via indptr


@dataclass
class PiecewiseCaptureShape:
    """One capture bucket.

    Handed to the static-input factory (``make_static_inputs``) and the plan
    hook (``plan_fn``) so both generalize across config types.
    """
    bs: int
    seq_lens: list[int]          # per-request lengths (uniform for BATCHED,
                                 # an arbitrary partition of total_tokens for PACKED)
    total_tokens: int            # sum(seq_lens); == bs * seq_len for BATCHED


@dataclass(kw_only=True)
class PiecewiseCudaGraphConfig(ABC):
    """Base config for a single piecewise-captured callable.

    ``kw_only`` so subclasses can add required fields (e.g. ``seq_len``) without
    colliding with the defaulted fields declared here.
    """
    # (1) function that gets captured. Contract:
    #     capture_fn(static_inputs: dict[str, Tensor],
    #                static_cm: BatchedCacheManager | None = None,
    #                **forward_kwargs) -> dict[str, Tensor]
    # It must READ tensors out of ``static_inputs`` (never reassign them) so the
    # runner-owned buffers stay the ones captured into the graph.
    capture_fn: Callable[..., dict[str, torch.Tensor]]
    # (2) factory: shape -> the static input buffers the runner will own + copy
    #     real inputs into before each replay.
    make_static_inputs: Callable[[PiecewiseCaptureShape], dict[str, torch.Tensor]]
    # (3) static kwargs threaded into capture_fn (e.g. cond_tokens, is_causal).
    forward_kwargs: dict[str, Any] = field(default_factory=dict)
    # (4) attention planning. When ``uses_kv_cache`` is True the runner plans a
    #     FlashInfer wrapper outside the graph before each replay: via
    #     ``plan_fn(static_cm, shape)`` if given, else a type-default (uniform
    #     seq_lens for BATCHED, packed indptr for PACKED).
    uses_kv_cache: bool = False
    plan_fn: Callable[["BatchedCacheManager", PiecewiseCaptureShape], None] | None = None
    cache_labels: list[str] = field(default_factory=lambda: ["main"])
    # None => defer to the runner's default batch-size buckets.
    capture_batch_sizes: list[int] | None = None
    # Whether to torch.compile capture_fn before capture. Default off; the block
    # loop already benefits from graph capture alone.
    compile: bool = False

    @abstractmethod
    def get_config_type(self) -> PiecewiseConfigType:
        ...

    @abstractmethod
    def get_capture_shapes(self, batch_sizes: list[int]) -> list[PiecewiseCaptureShape]:
        """Enumerate the (bs, seq_lens, total_tokens) buckets to capture.

        ``batch_sizes`` is the resolved list the runner will iterate
        (``capture_batch_sizes`` or the runner default).
        """
        ...


@dataclass(kw_only=True)
class PiecewiseBatchedConfig(PiecewiseCudaGraphConfig):
    """Equal-length batched capture: static input ``[bs, seq_len, D]``."""
    seq_len: int                 # tokens per request

    def get_config_type(self) -> PiecewiseConfigType:
        return PiecewiseConfigType.BATCHED

    def get_capture_shapes(self, batch_sizes: list[int]) -> list[PiecewiseCaptureShape]:
        return [
            PiecewiseCaptureShape(
                bs=bs,
                seq_lens=[self.seq_len] * bs,
                total_tokens=self.seq_len * bs,
            )
            for bs in batch_sizes
        ]


@dataclass(kw_only=True)
class PiecewisePackedConfig(PiecewiseCudaGraphConfig):
    """Packed variable-length capture: static input ``[total_tokens, D]``.

    Captures one graph per (bs, token-bucket). Each token bucket in
    ``total_tokens`` is partitioned across ``bs`` requests for the capture-time
    dummy plan; real per-request seq_lens arrive through ``run``.
    """
    total_tokens: list[int]      # token-count buckets to capture

    def get_config_type(self) -> PiecewiseConfigType:
        return PiecewiseConfigType.PACKED

    def get_capture_shapes(self, batch_sizes: list[int]) -> list[PiecewiseCaptureShape]:
        shapes: list[PiecewiseCaptureShape] = []
        for bs in batch_sizes:
            for tt in self.total_tokens:
                shapes.append(
                    PiecewiseCaptureShape(
                        bs=bs,
                        seq_lens=distribute_tokens(tt, bs),
                        total_tokens=tt,
                    )
                )
        return shapes
