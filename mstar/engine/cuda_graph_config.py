from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch

from mstar.engine.resources import SubmoduleStep
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs


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
    def get_node_inputs(self, bs: int, num_tokens: int) -> list[NodeInputs]:
        pass


class BatchedCudaGraphConfig(CudaGraphConfig):
    def __init__(
        self,
        capture_graph_walk: str,  # "decode"
        single_request_inputs: NodeInputs,
        replay_graph_walks: list[str] | None = None,
        additional_key_info: Any | None = None,
        compile: bool = True,
        capture_batch_sizes: list[int] | None = None,
        capture_forward_method: str = "forward_batched",
        caps_eager_batch_size: bool = True,
        total_tokens_multiplier: int = 1,
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
        # ``single_request_inputs.input_seq_len`` is also read per-label by the
        # submodule's own ``declare_step`` (e.g. replicated across cond/uncond
        # labels for a combined guidance step), so it must stay a per-label
        # span there. When one request's captured step actually commits KV
        # across more than one label combined into a single plan (batched
        # classifier-free guidance packing cond+uncond into one sequence),
        # the static buffer this bucket allocates has to hold all of them —
        # this multiplier scales the buffer size independently of the
        # per-label span. Default 1 preserves every existing caller.
        self.total_tokens_multiplier = total_tokens_multiplier

    def get_config_type(self) -> CudaGraphConfigType:
        return CudaGraphConfigType.BASIC_BATCHED

    def get_total_tokens(self, bs: int) -> list[int]:
        return [self.single_request_inputs.input_seq_len * bs * self.total_tokens_multiplier]

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
        # seq len -> NodeInputs
        make_node_input: Callable[[int], NodeInputs],
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

class PiecewiseConfigType(Enum):
    BATCHED = "batched"
    PACKED = "packed"


@dataclass
class PiecewiseCaptureShape:
    """One piecewise capture bucket.

    Handed to the static-input factory and the step declaration so both
    generalize across config types.
    """
    bs: int
    seq_lens: list[int]  # per-request lengths (uniform for BATCHED, an
                         # arbitrary partition of total_tokens for PACKED)
    total_tokens: int    # sum(seq_lens); == bs * seq_len for BATCHED


@dataclass
class PiecewiseCallInputs:
    """Everything a captured region is handed, at capture and at replay alike.

    Both calls go through here so a region can't accidentally read something
    that only exists on one of the two paths.
    """
    # The runner-owned buffers the graph was captured against. READ from them;
    # reassigning an entry detaches the region from the address the graph holds.
    static_inputs: dict[str, torch.Tensor]
    # The region's own view of the batch: request ids and per-request info
    # padded to the capture bucket, plus the node's resources. At capture the
    # rows are the runner's padding ids.
    engine_inputs: ModelInputsFromEngine
    # ``config.forward_kwargs``, unchanged
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def resources(self) -> dict[str, Any]:
        return self.engine_inputs.resources


@dataclass(kw_only=True)
class PiecewiseCudaGraphConfig(ABC):
    """One inner callable of a submodule's forward, captured on its own.

    Unlike ``CudaGraphConfig``, which describes a whole ``forward_batched``
    the engine drives, this describes a SUB-REGION the submodule invokes
    itself while the surrounding preamble stays eager.

    ``kw_only`` so subclasses can add required fields (e.g. ``seq_len``)
    without colliding with the defaulted ones here.
    """
    # The captured callable, taking one PiecewiseCallInputs and returning the
    # region's outputs by name (a bare tensor is accepted for a single output).
    capture_fn: Callable[[PiecewiseCallInputs], dict[str, torch.Tensor]]
    # shape -> the static input buffers the runner owns and copies real inputs
    # into before each replay
    make_static_inputs: Callable[[PiecewiseCaptureShape], dict[str, torch.Tensor]]
    # This region's own resource work, declared over the padded batch. Separate
    # from the submodule's `declare_step` because the region's shape is its
    # own: the runner admits, plans, and commits it per replay.
    declare_step: Callable[[list[str], list[int]], SubmoduleStep | None] | None = None
    # Set when the outer forward's step already planned this region's work over
    # this same padded shape and lease; the runner then declares and plans only
    # at capture, and each replay rides the live plan. Wrong here is silent: the
    # region would read whatever plan the resources currently hold.
    # TODO: the runner can't see the outer lease to check the two agree. Thread
    # it through `run` once the forward carries its step context.
    reuses_outer_plan: bool = False
    # static kwargs threaded into capture_fn (e.g. cond_tokens, is_causal)
    forward_kwargs: dict[str, Any] = field(default_factory=dict)
    # None => defer to the runner's default batch-size buckets
    capture_batch_sizes: list[int] | None = None
    # Whether to torch.compile capture_fn before capture. Default off; the
    # block loop already benefits from graph capture alone.
    compile: bool = False

    @abstractmethod
    def get_config_type(self) -> PiecewiseConfigType:
        ...

    @abstractmethod
    def get_capture_shapes(self, batch_sizes: list[int]) -> list[PiecewiseCaptureShape]:
        """The (bs, seq_lens, total_tokens) buckets to capture.

        ``batch_sizes`` is the resolved list the runner iterates
        (``capture_batch_sizes`` or the runner default).
        """
        ...

    @abstractmethod
    def replay_seq_lens(
        self, shape: PiecewiseCaptureShape, seq_lens: list[int] | None, real_bs: int,
    ) -> list[int]:
        """Per-request lengths to declare at replay, padded to ``shape.bs``."""
        ...


@dataclass(kw_only=True)
class PiecewiseBatchedConfig(PiecewiseCudaGraphConfig):
    """Equal-length batched capture: static input ``[bs, seq_len, D]``."""
    seq_len: int  # tokens per request

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

    def replay_seq_lens(
        self, shape: PiecewiseCaptureShape, seq_lens: list[int] | None, real_bs: int,
    ) -> list[int]:
        # every row is capture-length, padding included: the captured shape is
        # the batch axis, so the padding rows carry real spans
        del seq_lens, real_bs
        return list(shape.seq_lens)


@dataclass(kw_only=True)
class PiecewisePackedConfig(PiecewiseCudaGraphConfig):
    """Packed variable-length capture: static input ``[total_tokens, D]``.

    One graph per (bs, token bucket). Each bucket is partitioned across ``bs``
    requests for the capture-time plan; real per-request lengths arrive at
    replay.
    """
    total_tokens: list[int]  # token-count buckets to capture

    def get_config_type(self) -> PiecewiseConfigType:
        return PiecewiseConfigType.PACKED

    def get_capture_shapes(self, batch_sizes: list[int]) -> list[PiecewiseCaptureShape]:
        return [
            PiecewiseCaptureShape(
                bs=bs,
                seq_lens=distribute_tokens(total, bs),
                total_tokens=total,
            )
            for bs in batch_sizes
            for total in self.total_tokens
        ]

    def replay_seq_lens(
        self, shape: PiecewiseCaptureShape, seq_lens: list[int] | None, real_bs: int,
    ) -> list[int]:
        if seq_lens is None:
            raise ValueError("packed piecewise replay requires seq_lens")
        # zero-length padding rows keep the planned qo_indptr summing to the
        # real token count, so attention skips the padded tail
        return list(seq_lens) + [0] * (shape.bs - real_bs)
