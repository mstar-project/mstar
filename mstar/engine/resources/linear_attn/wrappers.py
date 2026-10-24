"""Per-(bucket, slot, label) state for the two GDN walks.

A wrapper owns the buffers its kernels launch against. A captured graph holds
their addresses, so they are sized to the bucket on first plan and never
reallocated: a `torch.zeros` firing on a later, larger layout means a replay
reads freed memory.
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
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

        # Snapshots: the kernel checkpoints every row's state each `_every`
        # tokens, and the ones a row keeps are scattered into snapshot slots.
        # `_snapshot_index` holds, per snapshot, the checkpoint, the token it
        # ends at, and the slot. Padding entries write the sink.
        self._every = 0
        self._checkpoint_cu: torch.Tensor | None = None
        self._snapshot_index: torch.Tensor | None = None
        self._num_snapshots = 0
        self._checkpoints: torch.Tensor | None = None
        self._num_checkpoints = 0
        self._conv_taps: torch.Tensor | None = None

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
        snapshots: Sequence[tuple[int, int, int]] = (),
        every: int = 0,
        checkpoints: torch.Tensor | None = None,
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
        # a captured graph checkpoints whether or not the step keeps any
        self._every = every if snapshots or self._accelerator_graph else 0
        if self._every:
            self._plan_snapshots(spans, snapshots, checkpoints)

        self._build_mask(self._plan_tokens)
        num_rows = len(spans)
        self._plan_state = GDNPrefillPlan(
            slots=slots,
            cu_seqlens=self._cu_buffer[: len(spans) + 1],
            has_state=has_state[:num_rows],
            num_rows=num_rows,
            token_mask=self._mask,
        )

    def _plan_snapshots(
        self, spans: list[int], snapshots: Sequence[tuple[int, int, int]],
        checkpoints: torch.Tensor | None,
    ):
        every = self._every
        rows = len(spans)
        starts, checkpoint_cu = [0], [0]
        for span in spans:
            starts.append(starts[-1] + span)
            checkpoint_cu.append(checkpoint_cu[-1] + span // every)
        # under capture `rows` is the bucket's row count, and a row keeps at most two snapshots
        capacity = (
            min(2 * rows, self._max_num_tokens // every) if self._accelerator_graph else len(snapshots)
        )
        assert len(snapshots) <= capacity, f"{len(snapshots)} snapshots in a step captured for {capacity}"
        pad = capacity - len(snapshots)
        index = [
            [checkpoint_cu[row] + offset // every - 1 for row, offset, _ in snapshots] + [0] * pad,
            [starts[row] + offset for row, offset, _ in snapshots] + [0] * pad,
            [slot for _, _, slot in snapshots] + [self._null_slot_id] * pad,
        ]

        if self._checkpoint_cu is None or self._checkpoint_cu.numel() < self._rows_capacity + 1:
            self._checkpoint_cu = torch.zeros(self._rows_capacity + 1, dtype=torch.int64, device=self._device)
        if self._snapshot_index is None or self._snapshot_index.shape[1] < capacity:
            self._snapshot_index = torch.zeros((3, capacity), dtype=torch.int64, device=self._device)
        pin = torch.accelerator.is_available()
        self._checkpoint_cu[: rows + 1].copy_(
            torch.tensor(checkpoint_cu, dtype=torch.int64, pin_memory=pin), non_blocking=True,
        )
        self._snapshot_index[:, :capacity].copy_(
            torch.tensor(index, dtype=torch.int64, pin_memory=pin), non_blocking=True,
        )
        self._num_snapshots = capacity
        self._num_checkpoints = checkpoint_cu[-1]
        self._checkpoints = checkpoints
        assert checkpoints is None or self._num_checkpoints <= checkpoints.shape[0], (
            f"{self._num_checkpoints} checkpoints in a step captured for {checkpoints.shape[0]}"
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
        if self._every:
            self._save_conv_snapshots(x, conv_layer)
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

    def _save_conv_snapshots(self, x: torch.Tensor, conv_layer: torch.Tensor):
        """A snapshot's conv state is the inputs before its boundary, the
        window a prefill ending there leaves in its slot."""
        if self._conv_taps is None:
            self._conv_taps = torch.arange(-conv_layer.shape[-1], 0, device=x.device)
        index = self._snapshot_index[:, : self._num_snapshots]
        window = (index[1, :, None] + self._conv_taps).clamp_min(0)
        conv_layer.index_copy_(0, index[2], x[window].transpose(1, 2).to(conv_layer.dtype))

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

        snapshot_args = {}
        if self._every:
            checkpoints = self._checkpoints
            if checkpoints is None:
                checkpoints = torch.empty(
                    (self._num_checkpoints, *initial.shape[1:]), dtype=torch.float32, device=q.device,
                )
            snapshot_args = dict(
                state_checkpoints=checkpoints,
                checkpoint_cu_starts=self._checkpoint_cu[: self._plan_state.num_rows + 1],
                checkpoint_every_n_tokens=self._every,
            )
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
            **snapshot_args,
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
        if self._every:
            index = self._snapshot_index[:, : self._num_snapshots]
            state.index_copy_(0, index[2], checkpoints.index_select(0, index[0]).to(state.dtype))
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
        snapshots: Sequence[tuple[int, int, int]] = (),
        every: int = 0,
        checkpoints: torch.Tensor | None = None,
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
        # each snapshot's row, renumbered by its place among the chunked half's prompt rows
        pre_row = {int(row): i for i, row in enumerate(prefill_rows)}
        pre_snapshots = [(pre_row[row], offset, slot) for row, offset, slot in snapshots]
        self._prefill.plan(
            pre_spans, pslot_buf[:cap], phas_buf[:cap], pre_snapshots, every, checkpoints,
        )
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
