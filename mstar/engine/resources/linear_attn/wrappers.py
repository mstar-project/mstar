"""Per-(bucket, slot, label) state for the two GDN walks.

A wrapper owns the buffers its kernels launch against. A captured graph holds
their addresses, so they are sized to the bucket on first plan and never
reallocated: a `torch.zeros` firing on a later, larger layout means a replay
reads freed memory.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
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
        accelerator_graph: bool=False,
        null_slot_id: int = 0,
    ):
        if accelerator_graph and (num_tokens is None or bs is None):
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
        self._accelerator_graph = accelerator_graph
        self._capacity = -1
        self._sm_scale = sm_scale
        self._qk_l2norm = qk_l2norm
        self._device = device

    # Whether this wrapper's kernel honours its own q/k L2-norm flag. The SM90
    # chunked kernel takes one and ignores it, returning NaN once a sequence is
    # long enough to matter, so the prefill path is normalised by the caller.
    qk_l2norm_in_kernel: bool = False
    # Whether the kernel reads strided `v`/`a`/`b` (column views of the fused
    # projection and conv output) as they are, sparing the caller a copy each.
    takes_strided_inputs: bool = False

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
    """Launch geometry for the varlen conv kernel, precomputed at plan time.

    The kernel otherwise derives it in its grid lambda with a D2H and H2D,
    which capture forbids.
    """

    nums_dict: dict
    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor


def _with_room(t: torch.Tensor, rows: int) -> torch.Tensor:
    """``t`` with ``rows`` zeroed rows appended along dim 0."""
    return torch.cat((t, t.new_zeros((rows, *t.shape[1:]))), dim=0)


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
        accelerator_graph: bool=False,
        null_slot_id: int = 0,
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            sm_scale=sm_scale,
            qk_l2norm=qk_l2norm,
            num_tokens=num_tokens,
            bs=bs,
            accelerator_graph=accelerator_graph,
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
        if self._accelerator_graph:
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

        pin = torch.accelerator.is_available()
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
        capacity = max(num_rows, self._bs) if self._accelerator_graph else num_rows
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
                cu, dtype=torch.int64, pin_memory=torch.accelerator.is_available()
            ),
            non_blocking=True,
        )

    def _build_mask(
        self, total_tokens: int
    ):
        if self._accelerator_graph:
            # `+ _bs` covers the room `run` adds for the padding rows' tokens
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
        # Before the fixup below: the tokens it hands the padding rows sit past
        # the real ones, where the mask zeroes them.
        self._plan_tokens = sum(spans)
        self._plan_borrowed = sum(1 for span in spans if span == 0)
        # A padding row covers no tokens of its own, and a zero-length segment
        # deadlocks the chunked kernel, so give each one a token. `run` pads its
        # buffers by `bs` so these are addressable however full the bucket is.
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

        # static per pool, so safe to branch on under capture
        if not self._has_sink_state:
            # Without a sink a padding row carries -1, which index_select and
            # index_copy_ cannot take; point it at a live row's slot and mask.
            live = slots >= 0
            any_live = live.any()
            ref = torch.argmax(live.to(torch.uint8)).view(1)
            ref_slot = torch.index_select(slots, 0, ref).clamp_min(0)
            addr = torch.where(live, slots, ref_slot)
            carried = self._plan_state.has_state & live
        else:
            addr = slots
            carried = self._plan_state.has_state

        # SM90 takes packed, sequence-ordered state (`state_indices` is
        # SM100/SM103 only), so gather here and scatter back.
        initial = torch.index_select(state, 0, addr).to(self._prefill_dtype)

        # zero the rows that start fresh, by select rather than boolean mask:
        # `initial[~mask] = 0` is a data-dependent shape and cannot be captured
        initial = torch.where(
            carried.view(-1, 1, 1, 1),
            initial,
            torch.zeros((), dtype=initial.dtype, device=initial.device),
        )

        # One spare row per batch row, so the token `plan` hands each padding
        # row is addressable even when the real batch fills the bucket exactly.
        # Unconditional under capture: capture itself sees no padding rows, so
        # branching on their count would bake the wrong shape into every replay.
        real_len = q.shape[0]
        rows = (
            self._plan_state.num_rows if self._accelerator_graph else self._plan_borrowed
        )
        if rows:
            q, k, v, g, beta = (
                _with_room(t, rows) for t in (q, k, v, g, beta)
            )

        # zero the padded tail so inf there can't poison the result
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
            # Padding rows write to `ref_slot` the live row's own result, or with
            # no live row what is already there, so slot 0 is not disturbed.
            padded = torch.where(
                any_live,
                torch.index_select(final, 0, ref),
                torch.index_select(state, 0, ref_slot).to(final.dtype),
            )
            final = torch.where(live.view(-1, 1, 1, 1), final, padded)
        state.index_copy_(0, addr, final.to(state.dtype))
        return out[:real_len]


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
        accelerator_graph: bool=False,
        null_slot_id: int = 0,
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            sm_scale=sm_scale,
            qk_l2norm=qk_l2norm,
            num_tokens=bs,
            bs=bs,
            accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id,
        )

        self._plan_state: GDNDecodePlan | None = None

    # The cutlass decode kernel really does normalise in-kernel, which saves
    # the caller a norm + divide + two casts per q and k, per layer, per step.
    qk_l2norm_in_kernel: bool = True
    # it marks q/k/v/a/b layout-dynamic, so the views go in as they are
    takes_strided_inputs: bool = True

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

        # Uninitialized output: the kernel writes every row, except one whose
        # slot is -1 (only with the sink off), which it would leave as found.
        # Its own default is a fresh `torch.zeros`, a fill per layer per step.
        output = (
            torch.empty(
                (q.shape[0], 1, v.shape[1], v.shape[2]),
                dtype=torch.bfloat16, device=q.device,
            )
            if self._null_slot_id >= 0 and q.dtype == torch.bfloat16 else None
        )
        # row i is token i, so the token axis is just unsqueezed
        out, _ = gated_delta_rule_decode_pretranspose(
            q=q.unsqueeze(1), k=k.unsqueeze(1), v=v.unsqueeze(1),
            state=None,
            A_log=a_log,
            a=g.unsqueeze(1),
            dt_bias=dt_bias,
            b=beta.unsqueeze(1),
            scale=self._sm_scale,
            output=output,
            initial_state=state,
            initial_state_indices=self._plan_state.slots,
            use_qk_l2norm=self._qk_l2norm,
        )
        return out.squeeze(1)


class _PinnedIndexStage:
    """Host index arrays a plan uploads, through page-locked staging buffers
    the wrapper owns. A packed step plans inline on the gpu thread behind the
    step still in flight, where a copy out of pageable memory would wait for
    that step to finish. ``done`` is recorded once the copies are issued; the
    next stage waits on it before rewriting the buffers."""

    def __init__(self):
        self._bufs: dict[str, torch.Tensor] = {}
        self._done = None

    def begin(self) -> None:
        if self._done is None:
            self._done = torch.cuda.Event() if torch.cuda.is_available() else None
        elif self._done is not None:
            self._done.synchronize()

    def upload(self, name: str, values: np.ndarray, dst: torch.Tensor) -> None:
        n = len(values)
        buf = self._bufs.get(name)
        if buf is None or buf.shape[0] < n or buf.dtype != dst.dtype:
            buf = self._bufs[name] = torch.empty(
                max(n, 64), dtype=dst.dtype, pin_memory=torch.cuda.is_available(),
            )
        buf.numpy()[:n] = values
        dst[:n].copy_(buf[:n], non_blocking=True)

    def end(self) -> None:
        if self._done is not None:
            self._done.record()


class GDNMixedWrapper(GDNWrapper):
    """A step mixing one-token rows that carry state (decode) with prompt rows.

    The chunked path over every row gathers and scatters each row's whole
    recurrent state (~1 MB a row at 16 heads x 128 x 128 fp32) and runs the
    chunked kernel on rows that need one recurrent update. Here the decode
    kernel takes every row, updating the decode rows' state in place by slot
    (the others point at the null slot, which it skips), and the chunked path
    takes at most ``max_prefill_rows`` prompt rows, its tokens moved to the
    front by a planned permutation. Both run over static shapes: the bucket's
    rows and tokens, and ``max_prefill_rows`` sequences.

    A one-token row without state (a one-token prompt) is a prompt row: the
    decode kernel reads the slot as it is, and a fresh slot is not cleared.
    """

    # the chunked half needs q/k normalised by the caller; the decode half is
    # built not to normalise again
    qk_l2norm_in_kernel: bool = False
    takes_strided_inputs: bool = False

    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        sm_scale: float | None = None,
        qk_l2norm: bool = True,
        prefill_dtype: torch.dtype = torch.float32,
        num_tokens: int | None = None,
        bs: int | None = None,
        has_sink_state: bool = True,
        accelerator_graph: bool = False,
        null_slot_id: int = 0,
        # the chunked half's rows under capture; eagerly, the step's own
        max_prefill_rows: int | None = None,
    ):
        super().__init__(
            device=device, pad_slot_id=pad_slot_id, sm_scale=sm_scale,
            qk_l2norm=qk_l2norm, num_tokens=num_tokens, bs=bs,
            accelerator_graph=accelerator_graph, null_slot_id=null_slot_id,
        )
        if accelerator_graph and max_prefill_rows is None:
            raise ValueError("GDNMixedWrapper: a captured split needs max_prefill_rows")
        self._max_prefill_rows = max_prefill_rows
        self._prefill = GDNPrefillWrapper(
            device=device, pad_slot_id=pad_slot_id, sm_scale=sm_scale,
            qk_l2norm=qk_l2norm, prefill_dtype=prefill_dtype,
            num_tokens=num_tokens, bs=max_prefill_rows if accelerator_graph else None,
            has_sink_state=has_sink_state, accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id,
        )
        self._decode = GDNDecodeWrapper(
            device=device, pad_slot_id=pad_slot_id, sm_scale=sm_scale,
            qk_l2norm=False, bs=bs, accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id,
        )
        self._stage = _PinnedIndexStage()
        self._idx: dict[str, torch.Tensor] = {}
        self._num_tokens = 0
        self._num_rows = 0
        self._front_len: int | None = None

    def _buf(self, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
        """A device index buffer, sized once to the bucket under capture."""
        buf = self._idx.get(name)
        if buf is None or buf.shape[0] < n:
            if buf is not None and self._accelerator_graph:
                raise RuntimeError(
                    f"GDNMixedWrapper: {name} needs {n} rows past its captured "
                    f"{buf.shape[0]}; a captured graph holds the old buffer"
                )
            buf = self._idx[name] = torch.zeros(n, dtype=dtype, device=self._device)
        return buf

    def plan(
        self,
        spans: list[int],
        host_slots: tuple[int, ...],
        host_has_state: tuple[bool, ...],
    ):
        null = self._null_slot_id
        span = np.asarray(spans, dtype=np.int64)
        slot = np.asarray(host_slots, dtype=np.int64)
        has = np.asarray(host_has_state, dtype=bool)
        rows = len(span)
        tokens = max(self._max_num_tokens or 0, int(span.sum()))
        bs = max(self._bs or 0, rows)

        start = np.zeros(rows, dtype=np.int64)
        np.cumsum(span[:-1], out=start[1:])
        decode = (span == 1) & (has | (slot == null))
        prefill_rows = np.flatnonzero((span > 0) & ~decode)
        cap = self._max_prefill_rows if self._accelerator_graph else max(len(prefill_rows), 1)
        if len(prefill_rows) > cap:
            raise RuntimeError(
                f"GDNMixedWrapper: {len(prefill_rows)} prompt rows in a step "
                f"captured for {cap}; the scheduler caps a mixed step's prompt rows"
            )

        # prompt tokens first, in row order; every other position after them
        front = np.concatenate(
            [np.arange(start[i], start[i] + span[i]) for i in prefill_rows]
            or [np.zeros(0, dtype=np.int64)]
        )
        rest = np.ones(tokens, dtype=bool)
        rest[front] = False
        perm = np.concatenate([front, np.flatnonzero(rest)])
        dec_src = np.zeros(bs, dtype=np.int64)
        dec_dst = np.full(bs, tokens, dtype=np.int64)  # the scratch row past the step's tokens
        dec_slots = np.full(bs, null, dtype=np.int64)
        dec_src[:rows] = np.where(decode, start, 0)
        dec_dst[:rows] = np.where(decode, start, tokens)
        dec_slots[:rows] = np.where(decode, slot, null)
        pad = cap - len(prefill_rows)
        pre_spans = span[prefill_rows].tolist() + [0] * pad
        pre_slots = np.concatenate([slot[prefill_rows], np.full(pad, null)])
        pre_has = np.concatenate([has[prefill_rows], np.zeros(pad, dtype=bool)])

        perm_buf = self._buf("perm", tokens, torch.int64)
        src_buf = self._buf("dec_src", bs, torch.int64)
        dst_buf = self._buf("dec_dst", bs, torch.int64)
        dslot_buf = self._buf("dec_slots", bs, torch.int32)
        pslot_buf = self._buf("pre_slots", cap, torch.int32)
        phas_buf = self._buf("pre_has", cap, torch.bool)
        self._stage.begin()
        self._stage.upload("perm", perm, perm_buf)
        self._stage.upload("dec_src", dec_src, src_buf)
        self._stage.upload("dec_dst", dec_dst, dst_buf)
        self._stage.upload("dec_slots", dec_slots, dslot_buf)
        self._stage.upload("pre_slots", pre_slots, pslot_buf)
        self._stage.upload("pre_has", pre_has, phas_buf)
        self._stage.end()

        self._num_tokens = tokens
        self._num_rows = bs
        # Under capture the chunked half takes the whole static token range
        # (its tail masked off); an eager one is sized to its prompt tokens,
        # and every other token of an eager step is a decode row's.
        self._front_len = None if self._accelerator_graph else len(front)
        self._prefill.plan(pre_spans, pslot_buf[:cap], phas_buf[:cap])
        self._decode.plan([1] * bs, dslot_buf[:bs])

    def _split(self, x: torch.Tensor):
        """``x`` with its prompt tokens first, and each row's decode token."""
        n = x.shape[0] if self._front_len is None else self._front_len
        return (
            x.index_select(0, self._idx["perm"][:n]),
            x.index_select(0, self._idx["dec_src"][: self._num_rows]),
        )

    def _merge(self, front: torch.Tensor, decode: torch.Tensor, n: int) -> torch.Tensor:
        """Back to token order: the chunked half's outputs by the permutation,
        then the decode rows' over theirs (the others land on a scratch row)."""
        out = front.new_empty((n + 1, *front.shape[1:]))
        m = n if self._front_len is None else self._front_len
        out.index_copy_(0, self._idx["perm"][:m], front[:m])
        out.index_copy_(0, self._idx["dec_dst"][: self._num_rows], decode.to(front.dtype))
        return out[:n]

    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ):
        n = x.shape[0]
        front, rows = self._split(x)
        front_out = self._prefill.run_conv(front, conv_layer, weight, bias, activation)
        rows_out = self._decode.run_conv(rows, conv_layer, weight, bias, activation)
        return self._merge(front_out, rows_out, n)

    def run(self, q, k, v, a, b, state, a_log, dt_bias):
        n = q.shape[0]
        qf, qd = self._split(q)
        kf, kd = self._split(k)
        vf, vd = self._split(v)
        af, ad = self._split(a)
        bf, bd = self._split(b)
        front_out = self._prefill.run(qf, kf, vf, af, bf, state, a_log, dt_bias)
        rows_out = self._decode.run(qd, kd, vd, ad, bd, state, a_log, dt_bias)
        return self._merge(front_out, rows_out, n)


# a gate this low makes a token a no-op on the state: no decay (softplus(a + dt_bias) is 0, so
# exp(g) is 1) and no write (sigmoid(b) is 0), exactly, in fp32
_CLOSED_GATE = -1.0e4
# a null id no slot has: the conv update kernel then skips no row
NO_PAD_SKIP = -2


@dataclass(frozen=True)
class GDNVerifyPlan:
    """A draft-verify step: every row is a block of ``block`` tokens."""

    # [n] int32, straight off the pool's addressing (padding rows: the sink); a captured graph
    # reads it in place, so anything derived from it is derived in the forward
    slots: torch.Tensor
    num_rows: int
    block: int


class GDNVerifyWrapper(GDNWrapper):
    """The rows of a draft-verify step: each a block of the last emitted token and its drafts,
    of which the verdict keeps a prefix it decides only after the forward.

    The forward runs the conv over each slot's window and the block, keeping the slot's window
    as it found it beside the block's inputs, and the delta rule over the block from the slot's
    state without writing it back (`run`). ``settle`` then writes, per row, the conv window
    after its accepted prefix, and the delta rule's state by running the block again from the
    slot with every rejected token's gates closed, so those tokens leave the state as it was.
    That second pass reads and writes the state once a row, where keeping the state after
    every token of the block would write it ``block`` times. Each layer's inputs stay
    referenced from its forward call to the settle, in the same graph.

    The kernels need the pool's sink: a padding row reads and writes it.
    """

    qk_l2norm_in_kernel: bool = True
    # FlashInfer's multi-token decode kernel reads the projection's column views as they are
    takes_strided_inputs: bool = True

    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        sm_scale: float | None = None,
        qk_l2norm: bool = True,
        bs: int | None = None,
        accelerator_graph: bool = False,
        null_slot_id: int = 0,
    ):
        super().__init__(
            device=device, pad_slot_id=pad_slot_id, sm_scale=sm_scale, qk_l2norm=qk_l2norm,
            num_tokens=bs, bs=bs, accelerator_graph=accelerator_graph, null_slot_id=null_slot_id,
        )
        if null_slot_id < 0:
            raise ValueError("a GDN verify step needs the pool's sink slot; unset disable_sink_slot")
        self._plan_state: GDNVerifyPlan | None = None
        # the slots as int64, derived once per forward (a graph reads the addressing in place)
        self._index: torch.Tensor | None = None
        self._closed: dict[torch.dtype, torch.Tensor] = {}
        # per layer, by its block's address: what `settle` needs of the step's forward
        self._conv: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._delta: dict[int, tuple] = {}

    def plan(self, spans: list[int], slots: torch.Tensor):
        block = spans[0] if spans else 0
        if block < 2 or any(span != block for span in spans):
            raise ValueError(f"a GDN verify step runs blocks of one length, at least 2; got spans {spans}")
        self._plan_state = GDNVerifyPlan(slots=slots, num_rows=len(spans), block=block)
        self._index = None
        self._conv.clear()
        self._delta.clear()

    def _slot_index(self) -> torch.Tensor:
        if self._index is None:
            self._index = self._plan_state.slots.to(torch.int64)
        return self._index

    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ) -> torch.Tensor:
        plan = self._plan_state
        rows, block = plan.num_rows, plan.block
        dim = x.shape[1]
        # the slot's window (every tap but the current token's) as the step found it
        old = conv_layer.index_select(0, self._slot_index())
        self._conv[conv_layer.data_ptr()] = (conv_layer, old, x)
        if x.device.type != "cuda":
            window = torch.cat([old, x.view(rows, block, dim).transpose(1, 2).to(old.dtype)], dim=2)
            out = torch.nn.functional.conv1d(
                window.float(), weight.float().unsqueeze(1), None if bias is None else bias.float(), groups=dim,
            )
            if activation in ("silu", "swish"):
                out = torch.nn.functional.silu(out)
            return out.transpose(1, 2).reshape(rows * block, dim).to(x.dtype)
        from mstar.utils.causal_conv1d import causal_conv1d_update

        # Through the whole block, writing the window after it into the slot, which the settle
        # corrects; padding rows too (into the sink), so no row's output is left unwritten.
        out = torch.empty(rows, block, dim, dtype=x.dtype, device=x.device)
        causal_conv1d_update(
            x.view(rows, block, dim).transpose(1, 2), conv_layer, weight, bias, activation,
            conv_state_indices=plan.slots, null_block_id=NO_PAD_SKIP, out=out.transpose(1, 2),
        )
        return out.view(rows * block, dim)

    def run(self, q, k, v, a, b, state, a_log, dt_bias):
        plan = self._plan_state
        rows, block = plan.num_rows, plan.block
        q, k, v = (t.reshape(rows, block, *t.shape[1:]) for t in (q, k, v))
        a, b = (t.reshape(rows, block, -1) for t in (a, b))
        out = self._delta_rule(q, k, v, a, b, state, a_log, dt_bias, update=False)
        self._delta[state.data_ptr()] = (state, q, k, v, a, b, a_log, dt_bias)
        return out.reshape(rows * block, *out.shape[2:])

    def settle(self, accepted: torch.Tensor) -> None:
        """Each row's state after its accepted prefix: the input token and ``accepted[i]``
        drafts. Inside the forward, after the verify; capturable."""
        plan = self._plan_state
        if plan is None or not (self._conv or self._delta):
            return
        rows, block = plan.num_rows, plan.block
        accepted = accepted[:rows].to(torch.int64)
        index = self._slot_index()
        at = None
        for conv_layer, old, x in self._conv.values():
            taps = conv_layer.shape[2]
            if at is None:
                # the window after token `accepted`: positions accepted + 1 .. accepted + taps
                at = (accepted + 1)[:, None] + torch.arange(taps, device=accepted.device)
                at = at[:, None, :].expand(rows, conv_layer.shape[1], taps)
            window = torch.cat([old, x.view(rows, block, -1).transpose(1, 2).to(old.dtype)], dim=2)
            conv_layer.index_copy_(0, index, window.gather(2, at))
        keep = (torch.arange(block, device=accepted.device)[None, :] <= accepted[:, None])[..., None]
        for state, q, k, v, a, b, a_log, dt_bias in self._delta.values():
            closed = self._closed.get(a.dtype)
            if closed is None:
                closed = self._closed[a.dtype] = torch.full((), _CLOSED_GATE, dtype=a.dtype, device=a.device)
            self._delta_rule(
                q, k, v, torch.where(keep, a, closed), torch.where(keep, b, closed),
                state, a_log, dt_bias, update=True,
            )
        self._conv.clear()
        self._delta.clear()

    def _delta_rule(self, q, k, v, a, b, state, a_log, dt_bias, update: bool) -> torch.Tensor:
        slots = self._plan_state.slots
        if q.device.type != "cuda":
            return _gated_delta_rule_reference(
                q, k, v, a, b, state, slots, a_log, dt_bias,
                self._sm_scale, self._qk_l2norm, update,
            )
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        out, _ = gated_delta_rule_decode_pretranspose(
            q=q, k=k, v=v, state=None, A_log=a_log, a=a, dt_bias=dt_bias, b=b,
            scale=self._sm_scale, use_qk_l2norm=self._qk_l2norm,
            initial_state=state, initial_state_indices=slots,
            disable_state_update=not update,
        )
        return out


def _gated_delta_rule_reference(
    q, k, v, a, b, state, slots, a_log, dt_bias, scale, qk_l2norm, update,
) -> torch.Tensor:
    """The gated delta rule over ``[rows, T, ...]`` blocks from the pool's slots, token by
    token in fp32, as the FlashInfer kernels compute it: decay, then the delta write, then
    the read. Where there is no kernel (CPU)."""
    rows, block, heads, dim_k = q.shape
    v_heads = v.shape[2]
    group = v_heads // heads
    scale = dim_k ** -0.5 if scale is None else scale
    qf, kf = q.float(), k.float()
    if qk_l2norm:
        qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    qf = qf.repeat_interleave(group, dim=2) * scale
    kf = kf.repeat_interleave(group, dim=2)
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    index = slots.to(torch.int64)
    s = state.index_select(0, index).float()  # [rows, HV, V, K]
    out = torch.empty(rows, block, v_heads, v.shape[3], dtype=torch.float32, device=q.device)
    for t in range(block):
        s = s * torch.exp(g[:, t])[:, :, None, None]
        recalled = torch.einsum("nhvk,nhk->nhv", s, kf[:, t])
        delta = (v[:, t].float() - recalled) * beta[:, t, :, None]
        s = s + delta[..., None] * kf[:, t, :, None, :]
        out[:, t] = torch.einsum("nhvk,nhk->nhv", s, qf[:, t])
    if update:
        state.index_copy_(0, index, s.to(state.dtype))
    return out.to(q.dtype)
