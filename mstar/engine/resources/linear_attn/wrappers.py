"""Per-(bucket, slot, label) state for the two GDN walks.

A wrapper owns the buffers its kernels launch against and is reused across
steps. Under CUDA-graph capture that ownership is the point: the captured
graph holds the *addresses* of these buffers, so they are sized to the bucket
on first plan and never reallocated afterwards. Any `torch.zeros` that can
fire on a later, larger layout is a replay reading freed memory.

Both wrappers take the gates raw and expose the same `plan`/`run`/`run_conv`
surface, so `GDNManager` dispatches on neither.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

# Must match `BLOCK_M` in mstar/utils/causal_conv1d/kernels.py, which passes it
# as an explicit constexpr rather than autotuning it.
_CONV_BLOCK_M = 8


class GDNWrapper(ABC):
    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        sm_scale: float | None=None,
        qk_l2norm: bool=True,
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
        # The slot the pool hands to rows that stand for no request (its sink,
        # or -1 with the sink off). The conv kernels skip rows whose slot is
        # this id; left at their default of 0 they would skip whichever real
        # request holds slot 0 when the sink is off.
        self._null_slot_id = null_slot_id
        self._max_num_tokens = num_tokens
        self._bs = bs
        self._cuda_graph = cuda_graph
        self._capacity = -1
        self._sm_scale = sm_scale
        self._qk_l2norm = qk_l2norm
        self._device = device

    # Whether this wrapper's kernel honours its own q/k L2-norm flag. The SM90
    # chunked kernel takes one and ignores it, returning NaN once a sequence is
    # long enough to matter, so the prefill path is normalised by the caller.
    qk_l2norm_in_kernel: bool = False

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
class GDNPrefillPlan:
    """Rows of any span, including 1 and 0."""

    slots: torch.Tensor       # [n] int32
    cu_seqlens: torch.Tensor  # [n + 1] int64
    has_state: torch.Tensor   # [n] bool, False where the slot reads as zeros
    num_rows: int

    # 1 on a real token, 0 on a capture bucket's padded tail. The chunked
    # kernel works a whole 64-token chunk at a time and masks to cu_seqlens
    # afterwards, so a non-finite value in the tail poisons the sum
    token_mask: torch.Tensor  # [padded_tokens] float32


@dataclass(frozen=True)
class ConvMetadata:
    """Precomputed launch geometry for the varlen conv kernel.

    Left to itself, the kernel derives this inside its grid lambda, which
    requires a non-cuda-graph-compatible D2H and H2D. We derive the metadata
    at plan time instead.
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
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            sm_scale=sm_scale,
            qk_l2norm=qk_l2norm,
            num_tokens=num_tokens,
            bs=bs,
            cuda_graph=cuda_graph,
            null_slot_id=null_slot_id,
        )
        self._has_sink_state = has_sink_state
        # What this arch's chunked kernel reads as its packed state; the
        # manager picks it off the device capability.
        self._prefill_dtype = prefill_dtype

        self._rows_capacity = -1
        self._mask_capacity = -1
        self._cu_buffer: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._plan_state: GDNPrefillPlan | None = None

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
            capacity = max(self._max_num_tokens, total_tokens)
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
        self._plan_conv(spans)
        self._build_cu_buffer(spans)

        self._plan_tokens = sum(spans)
        self._build_mask(self._plan_tokens)
        num_rows = len(spans)
        self._plan_state = GDNPrefillPlan(
            slots=slots,
            cu_seqlens=self._cu_buffer[: len(spans) + 1],
            has_state=has_state[:num_rows],
            num_rows=num_rows,
            token_mask=self._mask,
        )

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
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        # The chunked kernel takes the decay and the learning rate already
        # formed, where the decode kernel takes `a`/`b` raw and forms them from
        # the same two weights. Keep the formula here so the paths agree.
        g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
            a.float() + dt_bias.float()
        )
        beta = torch.sigmoid(b.float())

        slots = self._plan_state.slots.to(torch.int64)

        # _has_sink_state is a static property of the corresponding recurrent
        # state resource, making this if statement cuda graph-compatible
        if not self._has_sink_state:
            # With the pool's sink slot off, a padding row carries -1, which
            # index_select and index_copy_ cannot take. Point those rows at a live
            # row's slot instead; the gather is masked off below.
            live = slots >= 0
            any_live = live.any()
            ref = torch.argmax(live.to(torch.uint8)).view(1)
            ref_slot = torch.index_select(slots, 0, ref).clamp_min(0)
            addr = torch.where(live, slots, ref_slot)
            carried = self._plan_state.has_state & live
        else:
            addr = slots
            carried = self._plan_state.has_state

        # SM90 takes packed, sequence-ordered state — `state_indices` is
        # SM100/SM103 only — so gather here and scatter back. Once per prefill
        # step rather than per token, and prefill stays eager-cheap.
        initial = torch.index_select(state, 0, addr).to(self._prefill_dtype)

        # zero the rows that start fresh, by select rather than boolean mask:
        # `initial[~mask] = 0` is a data-dependent shape and cannot be captured
        initial = torch.where(
            carried.view(-1, 1, 1, 1),
            initial,
            torch.zeros((), dtype=initial.dtype, device=initial.device),
        )

        # Neutralise the bucket's padded tail before it reaches the kernel,
        # so inf values in the tail can't poison the result
        keep = self._plan_state.token_mask[: q.shape[0], None, None].bool()
        def blank(t, shape):
            return torch.where(
                keep.view(shape), t, torch.zeros((), dtype=t.dtype, device=t.device),
            )

        v = blank(v, (-1, 1, 1))
        g = blank(g, (-1, 1))
        beta = blank(beta, (-1, 1))

        out, final = chunk_gated_delta_rule(
            q=q, k=k, v=v,
            # FlashInfer wants the decay exponentiated; FLA's takes log space
            g=torch.exp(g),
            beta=beta,
            scale=self._sm_scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=self._plan_state.cu_seqlens,
            # ignored by this kernel (see `qk_l2norm_in_kernel`); the caller
            # has already normalised q and k
            use_qk_l2norm_in_kernel=False,
        )

        if not self._has_sink_state:
            # What the padding rows write to `ref_slot`: the live row's own result,
            # or — with no live row in the batch — whatever is already there, so
            # the write cannot disturb slot 0.
            padded = torch.where(
                any_live,
                torch.index_select(final, 0, ref),
                torch.index_select(state, 0, ref_slot).to(final.dtype),
            )
            final = torch.where(live.view(-1, 1, 1, 1), final, padded)
        state.index_copy_(0, addr, final.to(state.dtype))
        return out


@dataclass(frozen=True)
class GDNDecodePlan:
    """Every row is one token; row i is token i."""

    slots: torch.Tensor  # [n] int32, straight off the pool's addressing
    num_rows: int


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
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            sm_scale=sm_scale,
            qk_l2norm=qk_l2norm,
            num_tokens=bs,
            bs=bs,
            cuda_graph=cuda_graph,
            null_slot_id=null_slot_id,
        )

        self._plan_state: GDNDecodePlan | None = None

    # The cutlass decode kernel really does normalise in-kernel, which saves
    # the caller a norm + divide + two casts per q and k, per layer, per step.
    qk_l2norm_in_kernel: bool = True

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

    def run(self, q, k, v, g, beta, state, a_log, dt_bias):
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        # row i is token i, so the token axis is just unsqueezed
        out, _ = gated_delta_rule_decode_pretranspose(
            q=q.unsqueeze(1), k=k.unsqueeze(1), v=v.unsqueeze(1),
            state=None,
            A_log=a_log,
            a=g.unsqueeze(1),
            dt_bias=dt_bias,
            b=beta.unsqueeze(1),
            scale=self._sm_scale,
            initial_state=state,
            initial_state_indices=self._plan_state.slots,
            use_qk_l2norm=self._qk_l2norm,
        )
        return out.squeeze(1)
