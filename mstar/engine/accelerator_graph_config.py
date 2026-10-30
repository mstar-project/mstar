from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch

from mstar.engine.resources import AttentionStep, Segment, SubmoduleStep
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs

# A compile mode of the engine's own rather than torch.compile's: Inductor's
# defaults in its deterministic mode, which picks each kernel's config by rule
# instead of timing candidates on the device, so every process compiles the
# same kernels and one input gives one output from any server. See
# ``accelerator_graph_runner.compile_kwargs``.
DETERMINISTIC_COMPILE_MODE = "deterministic"


def check_compile_mode(mode: str | None) -> str | None:
    """A capture config's ``compile_mode``, refused at build rather than at capture,
    where the per-capture fallback would serve the bucket eager."""
    if mode is None or mode in ("default", DETERMINISTIC_COMPILE_MODE):
        return mode
    if not mode:
        # torch.compile takes a falsy mode for "default"
        raise ValueError("compile_mode is empty; name a mode or leave it None")
    try:
        from torch._inductor import list_mode_options

        options = list_mode_options(mode)
    except Exception as exc:
        raise ValueError(
            f"compile_mode={mode!r} is not a mode this torch.compile accepts: {exc}"
        ) from exc
    if options.get("triton.cudagraphs"):
        # Inductor's own graphs inside the runner's capture fail it
        raise ValueError(
            f"compile_mode={mode!r} turns on Inductor's CUDA graphs inside the runner's "
            "capture; use a -no-cudagraphs mode"
        )
    return mode


class AcceleratorGraphConfigType(Enum):
    BASIC_BATCHED = "basic_batched"
    FLASH_INFER_PACKED = "flash_infer_packed"


class AcceleratorGraphConfig(ABC):
    def __init__(
        self,
        capture_graph_walk: str,  # "decode"
        replay_graph_walks: list[str] | None = None, # set to None to be just capture_graph_walk

        # Additional information added to the capture bucket key (e.g., requires_cfg).
        # Must be hashable.
        additional_key_info: Any | None = None,
        compile: bool = True, # whether to run torch.compile before cuda graph capture
        # torch.compile mode for the captured forward; None takes the deployment's
        # compile level (mstar serve -O), a mode opts the config out of it
        compile_mode: str | None = None,

        # Per-config override for the set of batch sizes to capture
        capture_batch_sizes: list[int] | None = None,
        # Method on the submodule to capture
        capture_forward_method: str = "forward_batched",
        # Whether this config's captured batch sizes also cap the engine's max
        # (eager) batch size for the walk. Default True keeps the conservative
        # behavior: never batch beyond a captured graph size.
        caps_eager_batch_size: bool = True,
        # Maps a static-input key, as returned by the submodule's
        # ``preprocess``, to the dim that varies with the bucket. Overrides
        # the runner's size-matching guess (``CudaGraphRunner._seq_dim``),
        # which can pick the wrong dim when an unrelated axis happens to
        # match the bucket's token count.
        input_seq_dims: dict[str, int] | None = None,
    ):
        self.capture_graph_walk = capture_graph_walk
        self.replay_graph_walks = replay_graph_walks or [capture_graph_walk]
        self.additional_key_info = additional_key_info
        self.compile = compile
        self.compile_mode = check_compile_mode(compile_mode)
        self.capture_batch_sizes = capture_batch_sizes
        self.capture_forward_method = capture_forward_method
        self.caps_eager_batch_size = caps_eager_batch_size
        # How many plan tokens one input token turns into (a step that commits
        # KV under several labels combined into one plan, e.g. classifier-free
        # guidance packing cond + uncond, has 2). Buckets are keyed by plan
        # tokens; the engine asks for a bucket in input tokens and the runner
        # scales by this. Subclasses that support it set it.
        self.total_tokens_multiplier = 1
        self.input_seq_dims = input_seq_dims

    @abstractmethod
    def get_config_type(self) -> AcceleratorGraphConfigType:
        pass

    @abstractmethod
    def get_total_tokens(self, bs: int) -> list[int]:
        pass

    @abstractmethod
    def get_node_inputs(self, bs: int, num_tokens: int) -> list[NodeInputs]:
        pass


class BatchedAcceleratorGraphConfig(AcceleratorGraphConfig):
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
        compile_mode: str | None = None,
        input_seq_dims: dict[str, int] | None = None,
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            additional_key_info=additional_key_info,
            compile=compile,
            compile_mode=compile_mode,
            capture_batch_sizes=capture_batch_sizes,
            capture_forward_method=capture_forward_method,
            caps_eager_batch_size=caps_eager_batch_size,
            input_seq_dims=input_seq_dims,
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

    def get_config_type(self) -> AcceleratorGraphConfigType:
        return AcceleratorGraphConfigType.BASIC_BATCHED

    def get_total_tokens(self, bs: int) -> list[int]:
        return [self.single_request_inputs.input_seq_len * bs * self.total_tokens_multiplier]

    def get_node_inputs(self, bs: int, num_tokens: int):
        del num_tokens
        return [
            self.single_request_inputs.clone() for _  in range(bs)
        ]


def geometric_batch_sizes(max_batch_size: int) -> list[int]:
    """Powers of two below ``max_batch_size``, then ``max_batch_size`` itself.

    Startup grows with ``log B`` rather than ``B``; a batch between two sizes
    pads up to the next (or splits into these sizes), and a full batch fits one.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be at least 1, got {max_batch_size}")
    return [1 << i for i in range((max_batch_size - 1).bit_length())] + [max_batch_size]


def distribute_tokens(total_tokens: int, bs: int) -> list[int]:
    """Split ``total_tokens`` across ``bs`` requests, remainder on the first.
    """
    seq_lens = [total_tokens // bs] * bs
    seq_lens[0] += total_tokens % bs
    return seq_lens


def decode_rows_then_prompt(total_tokens: int, bs: int) -> list[int]:
    """``bs - 1`` one-token rows and a prompt row of the rest: a mixed step's
    layout, for kernels that split rows by span. `distribute_tokens` when
    there are fewer tokens than rows."""
    if total_tokens < bs:
        return distribute_tokens(total_tokens, bs)
    return [1] * (bs - 1) + [total_tokens - (bs - 1)]


class PackedAcceleratorGraphConfig(AcceleratorGraphConfig):
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
        caps_eager_batch_size: bool = True,
        # ``capture_token_lengths`` count input tokens; a step whose plan
        # carries every input token under several combined labels (batched
        # guidance: cond + uncond in one plan) multiplies them by this, so
        # the resources size their per-bucket buffers for the whole plan.
        total_tokens_multiplier: int = 1,
        compile_mode: str | None = None,
        input_seq_dims: dict[str, int] | None = None,
        # (input tokens, bs) -> each capture row's span; `distribute_tokens`
        # by default. A walk whose kernels split rows by span (a mixed step's
        # one-token rows beside a prompt) captures a layout it will replay.
        token_layout: Callable[[int, int], list[int]] | None = None,
    ):
        super().__init__(
            capture_graph_walk=capture_graph_walk,
            replay_graph_walks=replay_graph_walks,
            additional_key_info=additional_key_info,
            compile=compile,
            compile_mode=compile_mode,
            capture_batch_sizes=capture_batch_sizes,
            capture_forward_method=capture_forward_method,
            caps_eager_batch_size=caps_eager_batch_size,
            input_seq_dims=input_seq_dims,
        )
        if total_tokens_multiplier < 1:
            raise ValueError("total_tokens_multiplier must be at least 1")
        self.make_node_input = make_node_input
        self.capture_token_lengths = capture_token_lengths
        self.total_tokens_multiplier = total_tokens_multiplier
        self.token_layout = token_layout or distribute_tokens

    def get_config_type(self) -> AcceleratorGraphConfigType:
        return AcceleratorGraphConfigType.FLASH_INFER_PACKED

    def get_total_tokens(self, bs: int) -> list[int]:
        # every real row carries a token, so a bucket with fewer tokens than
        # rows could never serve a step; capturing it would give rows of none
        return [
            n * self.total_tokens_multiplier for n in self.capture_token_lengths
            if n >= bs
        ]

    def get_node_inputs(self, bs: int, num_tokens: int):
        # ``num_tokens`` is the bucket's plan tokens; the rows get input tokens
        seq_lens = self.token_layout(num_tokens // self.total_tokens_multiplier, bs)
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
class PiecewiseAcceleratorGraphConfig(ABC):
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
    # Take this region's slot before the outer `declare_step` runs, and report
    # it on the step context. The region still declares, plans and commits its
    # own work; the lease only tells the outer declaration which resources are
    # already spoken for, so it can leave them out.
    lease_before_step: bool = False
    # For a `lease_before_step` region: whether this batch may take a slot at
    # all, beyond fitting a captured bucket. For a limit the bucket key does
    # not express, such as how many attention segments a replay's plan holds.
    # A batch it rejects gets no lease, so its forward runs the region eagerly
    # (`ModelInputsFromEngine.piecewise_runners` leaves the region out).
    # None: every batch that fits a bucket.
    can_lease: Callable[[list[NodeInputs]], bool] | None = None
    # The region replays a whole step as ONE row, however many requests the
    # step carries (a vision tower packing every request's images as segments
    # of one ragged batch), so the runner leases, checks and replays it at batch
    # size 1, under the step's first request. Only for a packed
    # `lease_before_step` region captured at batch size 1 alone.
    one_row_per_step: bool = False
    # static kwargs threaded into capture_fn (e.g. cond_tokens, is_causal)
    forward_kwargs: dict[str, Any] = field(default_factory=dict)
    # None => defer to the runner's default batch-size buckets
    capture_batch_sizes: list[int] | None = None
    # Whether to torch.compile capture_fn before capture. Default off; the
    # block loop already benefits from graph capture alone.
    compile: bool = False
    # torch.compile mode for the region; None takes the deployment's compile
    # level (mstar serve -O), a mode opts the config out of it
    compile_mode: str | None = None

    def __post_init__(self):
        check_compile_mode(self.compile_mode)
        if self.one_row_per_step and (
            self.get_config_type() is not PiecewiseConfigType.PACKED
            or not self.lease_before_step
            or list(self.capture_batch_sizes or ()) != [1]
        ):
            raise ValueError(
                "one_row_per_step needs a packed lease_before_step region "
                "with capture_batch_sizes=[1]"
            )

    def region_step(self, request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep | None:
        """This region's step over the padded batch (``declare_step``); None
        for a region that declares none."""
        return None if self.declare_step is None else self.declare_step(request_ids, seq_lens)

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
class PiecewiseBatchedConfig(PiecewiseAcceleratorGraphConfig):
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
class PiecewisePackedConfig(PiecewiseAcceleratorGraphConfig):
    """Packed variable-length capture: static input ``[total_tokens, D]``.

    One graph per (bs, token bucket). Each bucket is partitioned across ``bs``
    requests for the capture-time plan; real per-request lengths arrive at
    replay.
    """
    total_tokens: list[int]  # token-count buckets to capture
    # The attention resource of a region whose rows each attend within
    # themselves. Set with `declare_step` left None, the region declares one
    # non-causal "main" segment per row on it; a `one_row_per_step` region's
    # lengths are all segments of its one row.
    attn_key: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.attn_key is not None and self.declare_step is not None:
            raise ValueError("attn_key declares the region's step; leave declare_step None")

    def region_step(self, request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep | None:
        if self.attn_key is None:
            return super().region_step(request_ids, seq_lens)
        if self.one_row_per_step:
            (rid,) = request_ids
            request_ids = [rid] * len(seq_lens)
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=n)
                for rid, n in zip(request_ids, seq_lens, strict=True)
            ],
            steps={self.attn_key: AttentionStep(causal=False)},
        )

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
