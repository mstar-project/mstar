"""Per-(bucket, slot, label) state for the two GDN walks.

A wrapper owns the buffers its kernels launch against. A captured graph holds
their addresses, so they are sized to the bucket on first plan and never
reallocated: a `torch.zeros` firing on a later, larger layout means a replay
reads freed memory.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Must match `BLOCK_M` in mstar/utils/causal_conv1d/kernels.py, which passes it
# as an explicit constexpr rather than autotuning it.
_CONV_BLOCK_M = 8

# ROCm: the largest captured prefill bucket that walks its tokens with the
# recurrent kernel. Its cost grows with the longest row, ~1.3 us a token a
# layer on MI300X (one 2048-token row: 2.7 ms against the chunked kernel's
# 0.55), which a bucket bounds only by its token count; larger buckets stay
# eager on the chunked kernel.
_ROCM_RECURRENT_MAX_TOKENS = 512


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
        # The pool's slot for rows with no request (its sink, or -1 with the
        # sink off). The conv kernels skip rows with this slot; their default
        # of 0 would skip a real request in slot 0 when the sink is off.
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
        # AITER's chunked kernels build their schedule on the host, which
        # capture forbids; see `_run_recurrent`.
        self._recurrent = (
            torch.version.hip is not None and cuda_graph and has_sink_state
            and num_tokens <= _ROCM_RECURRENT_MAX_TOKENS
        )

        self._rows_capacity = -1
        self._mask_capacity = -1
        self._cu_buffer: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._plan_state: GDNPrefillPlan | None = None
        self._plan_borrowed = 0
        self._aiter_schedule = None

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
        # ROCm: AITER's VK kernel takes a chunk schedule built from the host
        # spans, once here for every layer of the step. It refuses to run
        # under capture, so it serves the eager walks only.
        self._aiter_schedule = None
        if (
            torch.version.hip is not None and self._has_sink_state
            and not self._recurrent
        ):
            from aiter.ops.triton.gated_delta_net import (
                build_gated_delta_rule_prefill_metadata,
            )
            self._aiter_schedule = build_gated_delta_rule_prefill_metadata(
                spans, cu_seqlens=self._plan_state.cu_seqlens, chunk_size=64,
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
        if self._recurrent:
            return self._run_recurrent(q, k, v, a, b, state, a_log, dt_bias)
        # The chunked kernel takes the decay and the learning rate already
        # formed, where the decode kernel takes `a`/`b` raw and forms them from
        # the same two weights. Keep the formula here so the paths agree.
        g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
            a.float() + dt_bias.float()
        )
        beta = torch.sigmoid(b.float())

        if self._aiter_schedule is not None:
            return self._run_aiter_vk(q, k, v, g, beta, state)

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
            self._plan_state.num_rows if self._cuda_graph else self._plan_borrowed
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

        if torch.version.hip is not None:
            out, final = _chunk_gated_delta_rule_aiter(
                q, k, v, g, beta, self._sm_scale, initial, self._plan_state.cu_seqlens,
            )
        else:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule
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

    def _run_aiter_vk(self, q, k, v, g, beta, state):
        """ROCm with a sink: AITER's VK kernel takes q/k at their own head
        count and reads and writes the K-last pool in place by slot, so the
        state is neither gathered, transposed nor scattered back."""
        from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
            chunk_gated_delta_rule_opt_vk,
        )

        plan = self._plan_state
        addr = plan.slots.to(torch.int64)
        # a fresh row starts from zeros, not from what its slot last held
        state.index_copy_(0, addr, torch.where(
            plan.has_state.view(-1, 1, 1, 1),
            torch.index_select(state, 0, addr),
            torch.zeros((), dtype=state.dtype, device=state.device),
        ))

        # The schedule covers exactly the real tokens plus the one `plan`
        # hands each padding row, and the kernel checks the token count.
        # The kernels address q/k as packed [T, H, D]; without the L2 norm they
        # arrive as column views of the fused projection.
        real_len, n = q.shape[0], self._plan_tokens
        q, k, v, g, beta = (
            _with_room(t[:n], self._plan_borrowed) if self._plan_borrowed
            else t[:n].contiguous()
            for t in (q, k, v, g, beta)
        )
        # The prebuilt HIP hidden-state kernel: per layer on MI300X it took
        # 768 us at 4x2048 tokens where the Triton one took 1050 (job 465388),
        # and the FlyDSL prepare stage takes 17-22% off at 2k-32k (465605).
        out, _ = chunk_gated_delta_rule_opt_vk(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            g=g.unsqueeze(0), beta=beta.unsqueeze(0), scale=self._sm_scale,
            initial_state=state, output_final_state=True,
            cu_seqlens=plan.cu_seqlens, state_dtype=state.dtype, use_exp2=False,
            use_chunk_hip=True, use_prepare_flydsl=True,
            prefill_metadata=self._aiter_schedule,
            initial_state_indices=plan.slots, inplace_final_state=True,
        )
        out = out.squeeze(0)[:n]
        return _with_room(out, real_len - n) if real_len > n else out

    def _run_recurrent(self, q, k, v, a, b, state, a_log, dt_bias):
        """ROCm, captured bucket: AITER's recurrent kernel, the one decode
        runs, walks each row's tokens in turn. One program per row and head,
        so no host schedule; correct for any span, slow only on long ones."""
        from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
            fused_rearrange_sigmoid_gated_delta_rule,
        )

        plan = self._plan_state
        rows = plan.num_rows
        addr = plan.slots[:rows].to(torch.int64)
        # the kernel reads every row's slot, so a fresh row's must read as zeros
        state.index_copy_(0, addr, torch.where(
            plan.has_state.view(-1, 1, 1, 1),
            torch.index_select(state, 0, addr),
            torch.zeros((), dtype=state.dtype, device=state.device),
        ))

        # room for the token `plan` hands each padding row, as in `run`
        real_len = q.shape[0]
        q, k, v, a, b = (_with_room(t, rows) for t in (q, k, v, a, b))
        n = q.shape[0]
        qkv = torch.cat(
            (q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)), dim=-1,
        )

        # In place, the kernel stores the state after token t to column t of a
        # [rows, tokens] index and skips -1. Column 0 is also where it reads the
        # initial state, so a row's slot goes there and at its last token.
        cu = plan.cu_seqlens
        slots = addr.to(torch.int32).view(-1, 1)
        index = torch.full((rows, n), -1, dtype=torch.int32, device=q.device)
        index[:, :1] = slots
        index.scatter_(1, (cu[1:] - cu[:-1] - 1).view(-1, 1), slots)
        out, _ = fused_rearrange_sigmoid_gated_delta_rule(
            A_log=a_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv,
            key_dim=q.shape[1] * q.shape[2], value_dim=v.shape[1] * v.shape[2],
            head_k_dim=q.shape[2], head_v_dim=v.shape[2],
            scale=self._sm_scale, initial_state=state, inplace_final_state=True,
            cu_seqlens=cu, ssm_state_indices=index,
            # the caller has normalised q and k (see `qk_l2norm_in_kernel`)
            use_qk_l2norm_in_kernel=False,
        )
        return out.reshape(n, v.shape[1], v.shape[2])[:real_len]


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
        # ROCm: AITER's FlyDSL kernel, until its first call fails; see
        # `_run_flydsl`. It skips no slot, so it needs the sink.
        self._flydsl = torch.version.hip is not None and null_slot_id >= 0
        self._dt_bias_cast: dict[int, torch.Tensor] = {}

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
        if self._flydsl:
            try:
                return self._run_flydsl(q, k, v, g, beta, state, a_log, dt_bias)
            except (ImportError, OSError) as e:
                # first call, in warmup: no FlyDSL, or a read-only JIT cache
                logger.warning("FlyDSL GDN decode unavailable, using Triton: %s", e)
                self._flydsl = False
        if torch.version.hip is not None:
            return self._run_aiter(q, k, v, g, beta, state, a_log, dt_bias)
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

    def _run_flydsl(self, q, k, v, a, b, state, a_log, dt_bias):
        """ROCm with a sink: AITER's FlyDSL decode, tuned for gfx942. It reads
        q/k/v as the conv output's column views, so nothing is concatenated,
        and updates the K-last pool in place by slot. Per layer on MI300X:
        75 us at 64 rows where the Triton path takes 93 (job 465605)."""
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

        head_k_dim = q.shape[-1]
        if self._sm_scale is not None and abs(self._sm_scale * head_k_dim ** 0.5 - 1) > 1e-6:
            # the kernel's scale is fixed at 1/sqrt(head_k_dim)
            self._flydsl = False
            return self._run_aiter(q, k, v, a, b, state, a_log, dt_bias)
        # It takes dt_bias in the activations' dtype. Qwen3.5's checkpoint
        # stores it bf16 and `GatedDeltaNet` widens it to fp32, so there the
        # cast back is exact.
        key = dt_bias.data_ptr()
        dt_bias_cast = self._dt_bias_cast.get(key)
        if dt_bias_cast is None:
            dt_bias_cast = self._dt_bias_cast[key] = dt_bias.to(q.dtype)
        n = q.shape[0]
        out = torch.empty(
            (n, 1, v.shape[1], v.shape[2]), dtype=q.dtype, device=q.device,
        )
        # one token per row: the sequence axis is unsqueezed
        flydsl_gdr_decode(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            a.unsqueeze(1), b.unsqueeze(1), dt_bias_cast, a_log,
            self._plan_state.slots.to(torch.int32), state, out,
            use_qk_l2norm=self._qk_l2norm, need_shuffle_state=False,
        )
        return out.view(n, v.shape[1], v.shape[2])

    def _run_aiter(self, q, k, v, a, b, state, a_log, dt_bias):
        """ROCm: AITER's fused decode reads q/k/v packed, forms the decay and
        the learning rate from `a`/`b` itself, and updates the K-last pool in
        place by slot, skipping a -1 slot as FlashInfer's does."""
        from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
            fused_rearrange_sigmoid_gated_delta_rule,
        )
        n = q.shape[0]
        # one token per row, so each row is its own sequence
        if getattr(self, "_rows_cu", None) is None or self._rows_cu.numel() < n + 1:
            self._rows_cu = torch.arange(max(n, self._bs or 0) + 1, dtype=torch.int32, device=q.device)
        qkv = torch.cat((q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)), dim=-1)
        out, _ = fused_rearrange_sigmoid_gated_delta_rule(
            A_log=a_log, a=a.contiguous(), b=b.contiguous(), dt_bias=dt_bias, qkv=qkv,
            key_dim=q.shape[1] * q.shape[2], value_dim=v.shape[1] * v.shape[2],
            head_k_dim=q.shape[2], head_v_dim=v.shape[2],
            scale=self._sm_scale, initial_state=state, inplace_final_state=True,
            cu_seqlens=self._rows_cu[: n + 1], ssm_state_indices=self._plan_state.slots,
            use_qk_l2norm_in_kernel=self._qk_l2norm,
        )
        return out.reshape(n, v.shape[1], v.shape[2])


def _chunk_gated_delta_rule_aiter(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    """ROCm: AITER's Triton port of FLA's chunked kernel. It wants a batch dim,
    one head count for q/k/v, the decay in log space and the state [N, H, K, V];
    the pool is K-last, so the state goes in and comes back transposed."""
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_gated_delta_rule

    rep = v.shape[1] // q.shape[1]
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)
    out, final = chunk_gated_delta_rule(
        q=q.unsqueeze(0).contiguous(), k=k.unsqueeze(0).contiguous(), v=v.unsqueeze(0).contiguous(),
        g=g.unsqueeze(0).contiguous(), beta=beta.unsqueeze(0).contiguous(), scale=scale,
        initial_state=initial_state.transpose(-1, -2).contiguous(), output_final_state=True,
        cu_seqlens=cu_seqlens.long(), use_qk_l2norm_in_kernel=False,
    )
    return out.squeeze(0), final.transpose(-1, -2)
