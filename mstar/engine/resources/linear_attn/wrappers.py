"""Per-(bucket, slot, label) state for the two GDN walks.

A wrapper owns the buffers its kernels launch against. A captured graph holds
their addresses, so they are sized to the bucket on first plan and never
reallocated: a `torch.zeros` firing on a later, larger layout means a replay
reads freed memory. The delta-rule kernel itself is the backend's
(``gdn_kernels``); the conv kernels are the same on every backend.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from mstar.engine.resources.linear_attn.gdn_kernels import (
    FlashInferGDNKernels,
    GDNDecodeKernel,
    GDNDecodePlan,
    GDNKernels,
    GDNPrefillKernel,
    GDNPrefillPlan,
    GDNVerifyPlan,
)
from mstar.utils.h2d import H2DMirror, PinnedStager

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
        accelerator_graph: bool=False,
        null_slot_id: int = 0,
        kernels: GDNKernels | None = None,
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            num_tokens=num_tokens,
            bs=bs,
            accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id,
        )
        self._has_sink_state = has_sink_state
        if kernels is None:
            kernels = FlashInferGDNKernels(
                sm_scale=sm_scale, qk_l2norm=qk_l2norm,
                has_sink=has_sink_state, prefill_dtype=prefill_dtype,
            )
        self._kernel: GDNPrefillKernel = (
            kernels.captured_prefill(bs, num_tokens) if accelerator_graph
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
        accelerator_graph: bool=False,
        null_slot_id: int = 0,
        kernels: GDNKernels | None = None,
    ):
        super().__init__(
            device=device,
            pad_slot_id=pad_slot_id,
            num_tokens=bs,
            bs=bs,
            accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id,
        )
        if kernels is None:
            kernels = FlashInferGDNKernels(
                sm_scale=sm_scale, qk_l2norm=qk_l2norm, has_sink=null_slot_id >= 0,
            )
        self._kernel: GDNDecodeKernel = kernels.decode(bs if accelerator_graph else None)

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
    (the others point at the null slot, which it skips), and the prefill path
    takes at most ``max_prefill_rows`` prompt rows, its tokens moved to the
    front by a planned permutation. Both run over static shapes: the bucket's
    rows and tokens, and ``max_prefill_rows`` sequences. Both halves run the
    resource's ``GDNKernels``.

    A one-token row without state (a one-token prompt) is a prompt row: the
    decode kernel reads the slot as it is, and a fresh slot is not cleared.
    """

    # the caller normalises q/k for both halves, which are built not to
    # normalise again (`GDNKernels.without_qk_l2norm`)
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
        # the prefill half's rows under capture; eagerly, the step's own
        max_prefill_rows: int | None = None,
        kernels: GDNKernels | None = None,
    ):
        super().__init__(
            device=device, pad_slot_id=pad_slot_id, num_tokens=num_tokens, bs=bs,
            accelerator_graph=accelerator_graph, null_slot_id=null_slot_id,
        )
        if accelerator_graph and max_prefill_rows is None:
            raise ValueError("GDNMixedWrapper: a captured split needs max_prefill_rows")
        self._max_prefill_rows = max_prefill_rows
        if kernels is None:
            kernels = FlashInferGDNKernels(
                sm_scale=sm_scale, qk_l2norm=qk_l2norm,
                has_sink=has_sink_state, prefill_dtype=prefill_dtype,
            )
        kernels = kernels.without_qk_l2norm()
        self._prefill = GDNPrefillWrapper(
            device=device, pad_slot_id=pad_slot_id,
            num_tokens=num_tokens, bs=max_prefill_rows if accelerator_graph else None,
            has_sink_state=has_sink_state, accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id, kernels=kernels,
        )
        self._decode = GDNDecodeWrapper(
            device=device, pad_slot_id=pad_slot_id, bs=bs, accelerator_graph=accelerator_graph,
            null_slot_id=null_slot_id, kernels=kernels,
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
        # Under capture the prefill half takes the whole static token range
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
        """Back to token order: the prefill half's outputs by the permutation,
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


class GDNVerifyWrapper(GDNWrapper):
    """A speculative verify step: every row ``tokens_per_row`` tokens (a
    request's last emitted token and its k drafts).

    A request holds one slot per token (the pool's slot group). Its first token
    starts from the slot its previous step accepted up to, which only the
    device knows (``num_accepted``, written by the speculative resource), and
    the state after token ``t`` lands in slot ``t`` of the group, so whatever
    the step accepts, the next one finds that state without a copy. The conv
    window is one slot wide by k + width - 1 and slides by the accepted count
    the same way. Padding rows (a group on the pool's null slot) are skipped
    by both kernels.
    """

    def __init__(
        self,
        device: torch.device,
        pad_slot_id: int,
        tokens_per_row: int,
        bs: int | None = None,
        accelerator_graph: bool = False,
        null_slot_id: int = 0,
        kernels: GDNKernels | None = None,
    ):
        super().__init__(
            device=device, pad_slot_id=pad_slot_id,
            num_tokens=None if bs is None else bs * tokens_per_row, bs=bs,
            accelerator_graph=accelerator_graph, null_slot_id=null_slot_id,
        )
        if kernels is None:
            raise ValueError("GDNVerifyWrapper needs the resource's GDNKernels")
        self._kernel: GDNPrefillKernel = kernels.verify()
        self._tokens = tokens_per_row
        self._stager = PinnedStager(torch.int32)
        self._mirror = H2DMirror() if accelerator_graph else None
        # [slot_map | conv_slots | cu_seqlens], one H2D a step, sized once to
        # the bucket under capture
        self._image: torch.Tensor | None = None
        self._rows_capacity = 0
        self._plan_state: GDNVerifyPlan | None = None

    def _views(self, rows: int):
        t = self._tokens
        cap = self._rows_capacity
        image = self._image
        slot_map = image[: cap * t].view(cap, t)
        conv = image[cap * t: cap * t + cap]
        cu = image[cap * (t + 1): cap * (t + 1) + cap + 1]
        return slot_map[:rows], conv[:rows], cu[: rows + 1]

    def plan(self, groups, num_accepted: torch.Tensor):
        rows, t = len(groups), self._tokens
        need = max(rows, self._bs or 0) if self._accelerator_graph else rows
        if need > self._rows_capacity:
            if self._image is not None and self._accelerator_graph:
                raise RuntimeError(
                    f"GDNVerifyWrapper: {rows} rows past the {self._rows_capacity} "
                    "its captured graph holds"
                )
            self._rows_capacity = max(need, 1)
            self._image = torch.zeros(
                self._rows_capacity * (t + 2) + 1, dtype=torch.int32, device=self._device,
            )
            if self._mirror is not None:
                self._mirror.invalidate()
        cap = self._rows_capacity
        null = self._null_slot_id
        host = np.full(cap * (t + 2) + 1, -1, dtype=np.int32)
        slot_map = host[: cap * t].reshape(cap, t)
        conv = host[cap * t: cap * t + cap]
        conv[:] = null
        for i, group in enumerate(groups):
            if group[0] != null:
                slot_map[i] = group
                conv[i] = group[0]
        host[cap * (t + 1): cap * (t + 1) + cap + 1] = np.arange(cap + 1, dtype=np.int32) * t
        self._stager.copy_(self._image, host, mirror=self._mirror)
        slot_map_d, conv_d, cu_d = self._views(rows)
        self._plan_state = GDNVerifyPlan(
            slot_map=slot_map_d, conv_slots=conv_d, cu_seqlens=cu_d,
            num_accepted=num_accepted[:rows], num_rows=rows, tokens_per_row=t,
        )

    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ):
        from mstar.utils.causal_conv1d import causal_conv1d_update

        plan = self._plan_state
        return causal_conv1d_update(
            x=x,
            conv_state=conv_layer,
            weight=weight,
            bias=bias,
            activation=activation,
            conv_state_indices=plan.conv_slots,
            num_accepted_tokens=plan.num_accepted,
            query_start_loc=plan.cu_seqlens,
            max_query_len=plan.tokens_per_row,
            null_block_id=self._null_slot_id,
        )

    def run(self, q, k, v, a, b, state, a_log, dt_bias):
        return self._kernel.run(self._plan_state, q, k, v, a, b, state, a_log, dt_bias)
