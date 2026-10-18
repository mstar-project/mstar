"""Per-(bucket, slot, label) state for the two GDN walks.

A wrapper owns the buffers its kernels launch against. A captured graph holds
their addresses, so they are sized to the bucket on first plan and never
reallocated: a `torch.zeros` firing on a later, larger layout means a replay
reads freed memory. The delta-rule kernel itself is the backend's
(``gdn_kernels``); the conv kernels are the same on every backend.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from mstar.engine.resources.linear_attn.gdn_kernels import (
    FlashInferGDNKernels,
    GDNDecodeKernel,
    GDNDecodePlan,
    GDNKernels,
    GDNPrefillKernel,
    GDNPrefillPlan,
)

# Must match `BLOCK_M` in mstar/utils/causal_conv1d/kernels.py, which passes it
# as an explicit constexpr rather than autotuning it.
_CONV_BLOCK_M = 8


class GDNWrapper(ABC):
    # this wrapper's delta-rule kernel, from the resource's `GDNKernels`
    _kernel: GDNPrefillKernel | GDNDecodeKernel

    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        num_tokens: int | None=None,
        bs: int | None=None,
        cuda_graph: bool=False,
        null_slot_id: int = 0,
    ):
        if cuda_graph and (num_tokens is None or bs is None):
            # Every capacity below is `max(this layout, the bucket)`; without
            # the bucket it degrades to the layout and reallocates.
            raise ValueError(
                "a capture-mode GDN wrapper needs its bucket's num_tokens and "
                f"bs to size static buffers; got {num_tokens} and {bs}"
            )
        self._pad_slot_id = pad_slot_id
        # The pool's slot for rows with no request (its sink, or -1 with the
        # sink off). The conv kernels skip rows with this slot; their default
        # of 0 would skip a real request in slot 0 when the sink is off.
        self._null_slot_id = null_slot_id
        self._max_num_tokens = num_tokens
        self._bs = bs
        self._cuda_graph = cuda_graph
        self._capacity = -1
        self._device = device

    @property
    def qk_l2norm_in_kernel(self) -> bool:
        """Whether this wrapper's kernel normalises q and k itself; see
        `GDNPrefillKernel.qk_l2norm_in_kernel`."""
        return self._kernel.qk_l2norm_in_kernel

    @property
    def takes_strided_inputs(self) -> bool:
        """Whether the kernel reads strided `v`/`a`/`b` (column views of the
        fused projection and conv output) as they are, sparing a copy each."""
        return self._kernel.takes_strided_inputs

    @abstractmethod
    def plan(self, *args, **kwargs):
        pass

    @abstractmethod
    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class ConvMetadata:
    """Launch geometry for the varlen conv kernel, precomputed at plan time.

    The kernel otherwise derives it in its grid lambda with a D2H and H2D,
    which capture forbids.
    """

    nums_dict: dict
    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor


class GDNPrefillWrapper(GDNWrapper):
    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        sm_scale: float | None=None,
        qk_l2norm: bool=True,
        prefill_dtype: torch.dtype = torch.float32,
        num_tokens: int | None=None,
        bs: int | None=None,
        has_sink_state: bool = True,
        cuda_graph: bool=False,
        null_slot_id: int = 0,
        kernels: GDNKernels | None = None,
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            num_tokens=num_tokens,
            bs=bs,
            cuda_graph=cuda_graph,
            null_slot_id=null_slot_id,
        )
        self._has_sink_state = has_sink_state
        if kernels is None:
            kernels = FlashInferGDNKernels(
                sm_scale=sm_scale, qk_l2norm=qk_l2norm,
                has_sink=has_sink_state, prefill_dtype=prefill_dtype,
            )
        self._kernel: GDNPrefillKernel = (
            kernels.captured_prefill(bs, num_tokens) if cuda_graph
            else kernels.chunked_prefill()
        )

        self._rows_capacity = -1
        self._mask_capacity = -1
        self._cu_buffer: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._plan_state: GDNPrefillPlan | None = None
        self._plan_borrowed = 0

        # Conv buffers
        self._batch_ptr: torch.Tensor | None = None
        self._offset_ptr: torch.Tensor | None  = None
        self._conv_metadata: ConvMetadata | None = None
        self._seqlens_cpu: torch.Tensor | None = None

    def _plan_conv(
        self, spans: list[int],
    ):
        # Build Launch geometry for this step's conv; see `ConvMetadata`
        block = _CONV_BLOCK_M
        # one Triton program per BLOCK_M chunk of each row
        rows, offsets = [], []
        for row, span in enumerate(spans):
            chunks = -(-span // block)  # ceil
            rows.extend([row] * chunks)
            offsets.extend(range(chunks))

        needed = len(rows)
        if self._cuda_graph:
            capacity = max(
                needed, self._max_num_tokens // block + self._bs, 1,
            )
        else:
            capacity = max(needed, 1)

        if capacity > self._capacity:
            i32 = dict(dtype=torch.int32, device=self._device)
            self._batch_ptr = torch.zeros(capacity, **i32)
            self._offset_ptr = torch.zeros(capacity, **i32)
            self._capacity = capacity

        pin = torch.cuda.is_available()
        self._batch_ptr.fill_(self._pad_slot_id)
        self._offset_ptr.fill_(self._pad_slot_id)

        if rows:
            self._batch_ptr[: len(rows)].copy_(
                torch.tensor(rows, dtype=torch.int32, pin_memory=pin),
                non_blocking=True,
            )
            self._offset_ptr[: len(offsets)].copy_(
                torch.tensor(offsets, dtype=torch.int32, pin_memory=pin),
                non_blocking=True,
            )

        self._seqlens_cpu = torch.tensor(spans, dtype=torch.int32)
        self._conv_metadata = ConvMetadata(
            nums_dict={
                block: {
                    # the whole buffer, so the grid does not move between
                    # replays of one bucket
                    "tot": self._capacity,
                    "mlist_len": len(rows),
                    # unread when batch_ptr is set, but the kernel looks them up
                    "mlist": None,
                    "offsetlist": None,
                    "batch_ptr": self._batch_ptr,
                    "token_chunk_offset_ptr": self._offset_ptr,
                }
            },
            batch_ptr=self._batch_ptr,
            token_chunk_offset_ptr=self._offset_ptr,
        )


    def _build_cu_buffer(self, spans: list[int]):
        num_rows = len(spans)
        # Sized to the bucket under capture, not to this layout: one bucket
        # replays at several row counts, and the first one planned is not
        # necessarily the widest.
        capacity = max(num_rows, self._bs) if self._cuda_graph else num_rows
        if capacity > self._rows_capacity:
            self._cu_buffer = torch.zeros(
                capacity + 1, dtype=torch.int64, device=self._device
            )
            self._rows_capacity = capacity

        cu = [0]
        for span in spans:
            cu.append(cu[-1] + span)

        # same dtype as the device buffer: a converting copy is staged through
        # a pageable temporary and stops being asynchronous
        self._cu_buffer[: len(cu)].copy_(
            torch.tensor(
                cu, dtype=torch.int64, pin_memory=torch.cuda.is_available()
            ),
            non_blocking=True,
        )

    def _build_mask(
        self, total_tokens: int
    ):
        if self._cuda_graph:
            # `+ _bs` covers the room a kernel adds for the padding rows' tokens
            capacity = max(self._max_num_tokens + self._bs, total_tokens)
        else:
            capacity = total_tokens
        if capacity > self._mask_capacity:
            self._mask = torch.zeros(
                capacity, dtype=torch.float32, device=self._device
            )
            self._mask_capacity = capacity
        self._mask[:total_tokens].fill_(1.0)
        self._mask[total_tokens:].zero_()

    def plan(
        self,
        spans: list[int],
        slots: torch.Tensor,
        has_state: torch.Tensor,
    ):
        real_spans = tuple(spans)
        # Before the fixup below: the tokens it hands the padding rows sit past
        # the real ones, where the mask zeroes them.
        self._plan_tokens = sum(real_spans)
        self._plan_borrowed = sum(1 for span in real_spans if span == 0)
        # A padding row covers no tokens of its own, and a zero-length segment
        # deadlocks a chunked kernel, so give each one a token. A kernel that
        # reads it pads its inputs by `bs`, so it is addressable however full
        # the bucket is.
        spans = [span or 1 for span in spans]

        self._plan_conv(spans)
        self._build_cu_buffer(spans)

        self._build_mask(self._plan_tokens)
        num_rows = len(spans)
        self._plan_state = GDNPrefillPlan(
            slots=slots,
            cu_seqlens=self._cu_buffer[: len(spans) + 1],
            has_state=has_state[:num_rows],
            num_rows=num_rows,
            token_mask=self._mask,
            real_spans=real_spans,
            spans=tuple(spans),
            real_tokens=self._plan_tokens,
            borrowed=self._plan_borrowed,
        )
        self._kernel.plan(self._plan_state)

    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ):
        from mstar.utils.causal_conv1d import causal_conv1d_fn
        # the varlen kernel is feature-major
        out = causal_conv1d_fn(
            x=x.transpose(0, 1),
            weight=weight,
            bias=bias,
            conv_states=conv_layer,
            query_start_loc=self._plan_state.cu_seqlens,
            cache_indices=self._plan_state.slots,
            has_initial_state=self._plan_state.has_state,
            activation=activation,
            null_block_id=self._null_slot_id,
            # precomputed off-stream: the kernel's own path syncs and copies
            # H2D inside its grid lambda, which capture forbids
            seqlens_cpu=self._seqlens_cpu,
            metadata=self._conv_metadata,
        )

        out = out.transpose(0, 1)

        # The kernel allocates its output with `empty_like` and writes only the
        # `cu_seqlens` range, so a capture bucket's padded tail keeps whatever
        # the previous replay left in that memory.
        # Select rather than multiply because the tail can hold inf.
        return torch.where(
            self._plan_state.token_mask[: out.shape[0], None].bool(),
            out,
            torch.zeros((), dtype=out.dtype, device=out.device),
        )

    def run(self, q, k, v, a, b, state, a_log, dt_bias):
        return self._kernel.run(self._plan_state, q, k, v, a, b, state, a_log, dt_bias)


class GDNDecodeWrapper(GDNWrapper):
    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        sm_scale: float | None=None,
        qk_l2norm: bool=True,
        bs: int | None=None,
        cuda_graph: bool=False,
        null_slot_id: int = 0,
        kernels: GDNKernels | None = None,
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            num_tokens=bs,
            bs=bs,
            cuda_graph=cuda_graph,
            null_slot_id=null_slot_id,
        )
        if kernels is None:
            kernels = FlashInferGDNKernels(
                sm_scale=sm_scale, qk_l2norm=qk_l2norm, has_sink=null_slot_id >= 0,
            )
        self._kernel: GDNDecodeKernel = kernels.decode(bs if cuda_graph else None)

        self._plan_state: GDNDecodePlan | None = None

    def plan(
        self, spans: list[int], slots: torch.Tensor
    ):
        # No conv metadata: `causal_conv1d_update` is one step per row and
        # takes the slots directly, so there is no varlen grid to precompute.
        self._plan_state = GDNDecodePlan(slots=slots, num_rows=len(spans))

    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ):
        from mstar.utils.causal_conv1d import causal_conv1d_update
        return causal_conv1d_update(
            x=x,
            conv_state=conv_layer,
            weight=weight,
            bias=bias,
            activation=activation,
            conv_state_indices=self._plan_state.slots,
            null_block_id=self._null_slot_id,
        )

    def run(self, q, k, v, a, b, state, a_log, dt_bias):
        return self._kernel.run(self._plan_state, q, k, v, a, b, state, a_log, dt_bias)
