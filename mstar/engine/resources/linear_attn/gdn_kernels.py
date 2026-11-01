"""The kernels a GDN resource runs, chosen once when the resource is built.

The wrappers in ``wrappers.py`` own a step's layout: cu_seqlens, the conv
launch geometry, the mask over a capture bucket's padded tail. The kernel calls
are delegated to the objects here, one per wrapper: a chunked prefill for eager
steps, a prefill a captured bucket can hold, and decode.

``FlashInferGDNKernels`` runs FlashInfer's CUTLASS kernels (NVIDIA), whose
chunked prefill captures as is. ``AiterGDNKernels`` (``gdn_kernels_aiter.py``)
runs AITER's Triton and FlyDSL kernels (AMD), whose chunked prefill builds its
schedule on the host and so serves eager steps only.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from mstar.engine.resources.linear_attn.config import LinearAttnBackend, LinearAttnConfig
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.platforms import current_platform


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

    # The step's spans, 0 for a padding row or a request reading its state
    # without extending it. `cu_seqlens` counts `spans` instead, which gives
    # each such row one token past the real ones; `token_mask` leaves it off.
    real_spans: tuple[int, ...] = ()
    spans: tuple[int, ...] = ()
    real_tokens: int = 0  # sum(real_spans)
    borrowed: int = 0     # rows given a token they did not have

    # State checkpoints for the prefix cache (0: none). The kernel saves every
    # row's state each `every` tokens into `checkpoints`, row i's from
    # `checkpoint_cu[i]`, and `snapshot_index` (checkpoint, token, slot) picks
    # the ones the step keeps for the pool's snapshot slots.
    every: int = 0
    checkpoints: torch.Tensor | None = None
    checkpoint_cu: torch.Tensor | None = None
    snapshot_index: torch.Tensor | None = None


@dataclass(frozen=True)
class GDNDecodePlan:
    """Every row is one token; row i is token i."""

    slots: torch.Tensor  # [n] int32, straight off the pool's addressing
    num_rows: int


def _with_room(t: torch.Tensor, rows: int) -> torch.Tensor:
    """``t`` with ``rows`` zeroed rows appended along dim 0."""
    return torch.cat((t, t.new_zeros((rows, *t.shape[1:]))), dim=0)


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The log decay and the learning rate (fp32) from the raw projections.

    The decode kernels form these from ``a``/``b`` themselves; a prefill
    kernel that takes them formed gets them here, so the paths agree.
    """
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
        a.float() + dt_bias.float()
    )
    return g, torch.sigmoid(b.float())


class GDNPrefillKernel(ABC):
    """A prefill kernel as one wrapper runs it."""

    # Whether the kernel normalises q and k itself; otherwise
    # `GDNManager.run` does, before the call.
    qk_l2norm_in_kernel: bool = False
    # Whether the kernel reads strided `v`/`a`/`b` (column views of the fused
    # projection and conv output) as they are, sparing the caller a copy each.
    takes_strided_inputs: bool = False

    def plan(self, plan: GDNPrefillPlan) -> None:  # noqa: B027 - most kernels stage nothing
        """Stage what this step's layers share. Once per step, outside capture."""

    @abstractmethod
    def run(
        self, plan: GDNPrefillPlan, q, k, v, a, b,
        state: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        """``[total_tokens, HV, V]``; ``state`` is updated in place."""


class GDNDecodeKernel(ABC):
    """A decode kernel as one wrapper runs it."""

    qk_l2norm_in_kernel: bool = False
    takes_strided_inputs: bool = False

    @abstractmethod
    def run(
        self, plan: GDNDecodePlan, q, k, v, a, b,
        state: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        """``[rows, HV, V]``; ``state`` is updated in place."""


class GDNKernels(ABC):
    """The GDN kernels one resource runs: the per-deployment facts they share,
    and a kernel object per wrapper."""

    # whether a kernel that can L2-normalise q and k does it
    qk_l2norm: bool
    # Whether the prefill kernels save the state checkpoints a recurrent pool's
    # prefix snapshots are copied from. Without them the pool's prefix cache
    # stays shut (see `GDNManager.bind_dependencies`).
    writes_snapshots: bool = False

    def without_qk_l2norm(self) -> GDNKernels:
        """These kernels for a caller that normalises q and k itself, as a
        mixed step's split does for both of its halves."""
        twin = copy.copy(self)
        twin.qk_l2norm = False
        return twin

    def uncapturable_prefill(self, num_tokens: int) -> str | None:
        """Why a prefill bucket of ``num_tokens`` cannot be captured, or None
        if it can. The bucket's token count bounds its longest row."""
        return None

    @abstractmethod
    def chunked_prefill(self) -> GDNPrefillKernel:
        """For eager steps, of any length."""

    @abstractmethod
    def captured_prefill(self, bs: int, num_tokens: int) -> GDNPrefillKernel:
        """For one capture bucket; its buffers are sized to it."""

    @abstractmethod
    def decode(self, bs: int | None) -> GDNDecodeKernel:
        """For a decode wrapper: a capture bucket's ``bs``, None for eager."""


# What each NVIDIA arch's chunked prefill kernel will read as its packed state.
# SM90 is fp32 only; Blackwell takes the narrower ones, so a bf16 pool is not
# forced through an fp32 round trip there.
_PREFILL_STATE_DTYPES: dict[int, tuple[torch.dtype, ...]] = {
    9: (torch.float32,),
    10: (
        torch.float32, torch.bfloat16, torch.float16,
        torch.float8_e4m3fn, torch.float8_e5m2,
    ),
    12: (torch.float32,),
}


class FlashInferGDNKernels(GDNKernels):
    """FlashInfer's GDN kernels: chunked prefill (captured as is) and the
    pretransposed decode, both against the K-last pool."""

    writes_snapshots = True

    def __init__(
        self,
        *,
        sm_scale: float | None,
        qk_l2norm: bool,
        has_sink: bool,
        prefill_dtype: torch.dtype = torch.float32,
    ):
        self.sm_scale = sm_scale
        self.qk_l2norm = qk_l2norm
        self.has_sink = has_sink
        # what the chunked kernel reads as its packed state
        self.prefill_dtype = prefill_dtype

    @classmethod
    def build(
        cls,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        state_dtype: torch.dtype,
        has_sink: bool,
        device: torch.device,
    ) -> FlashInferGDNKernels:
        if device.type == "cuda" and not current_platform.is_cuda():
            raise ValueError(
                "FlashInfer's GDN kernels are CUTLASS kernels for NVIDIA GPUs; "
                f"this is {current_platform!r}. Leave the linear attention "
                "backend at 'auto', or name 'aiter' on AMD GPUs."
            )
        capability = (
            current_platform.cuda_capability(device) if device.type == "cuda" else None
        )
        allowed = (
            _PREFILL_STATE_DTYPES.get(capability[0], (torch.float32,))
            if capability is not None else (torch.float32,)
        )
        cls._check_state_dtype(state_dtype, geometry, has_sink)
        return cls(
            sm_scale=config.sm_scale,
            qk_l2norm=config.qk_l2norm,
            has_sink=has_sink,
            # Cast only where the arch cannot read the pool as is: a bf16 pool
            # on SM90 round-trips through fp32 for prefill.
            prefill_dtype=state_dtype if state_dtype in allowed else torch.float32,
        )

    @staticmethod
    def _check_state_dtype(
        state_dtype: torch.dtype, geometry: DeltaNetGeometry, has_sink: bool,
    ) -> None:
        if state_dtype is torch.float32:
            return
        if state_dtype is not torch.bfloat16:
            raise NotImplementedError(
                f"gdn state dtype {state_dtype} is not supported: FlashInfer's "
                "pool decode paths are fp32 and bf16."
            )
        # The bf16 decode kernel is pool-only and K=V=128 only.
        if geometry.head_k_dim != 128 or geometry.head_v_dim != 128:
            raise NotImplementedError(
                f"a bf16 gdn state needs K=V=128, got K={geometry.head_k_dim} "
                f"V={geometry.head_v_dim}; declare an fp32 pool instead."
            )
        from flashinfer import gdn_decode

        if not gdn_decode._GDN_DECODE_BF16_STATE_AVAILABLE:
            raise NotImplementedError(
                "a bf16 gdn state needs FlashInfer's bf16 decode backend, "
                "which this build does not have; declare an fp32 pool instead."
            )
        if not has_sink:
            # That kernel writes a -1 index to slot 0, which without a sink
            # belongs to a request.
            raise ValueError(
                "a bf16 gdn state needs the pool's sink slot: its decode "
                "kernel writes padding rows to slot 0 rather than skipping "
                "them. Unset RecurrentStateConfig.disable_sink_slot."
            )

    def chunked_prefill(self) -> GDNPrefillKernel:
        return FlashInferPrefill(self, captured=False)

    def captured_prefill(self, bs: int, num_tokens: int) -> GDNPrefillKernel:
        return FlashInferPrefill(self, captured=True)

    def decode(self, bs: int | None) -> GDNDecodeKernel:
        return FlashInferDecode(self)


class FlashInferPrefill(GDNPrefillKernel):
    """FlashInfer's chunked prefill, on packed state gathered from the pool.

    The same kernel eagerly and under capture: it reads no host schedule.
    """

    # The SM90 chunked kernel takes an L2-norm flag and ignores it, returning
    # NaN once a sequence is long enough to matter, so the caller normalises.
    qk_l2norm_in_kernel = False
    takes_strided_inputs = False

    def __init__(self, kernels: FlashInferGDNKernels, captured: bool):
        self._kernels = kernels
        self._captured = captured

    def run(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        # The chunked kernel takes the decay and the learning rate already
        # formed, where the decode kernel takes `a`/`b` raw and forms them from
        # the same two weights. Keep the formula here so the paths agree.
        g, beta = gdn_gates(a, b, a_log, dt_bias)

        slots = plan.slots.to(torch.int64)

        # static per pool, so safe to branch on under capture
        if not self._kernels.has_sink:
            # Without a sink a padding row carries -1, which index_select and
            # index_copy_ cannot take; point it at a live row's slot and mask.
            live = slots >= 0
            any_live = live.any()
            ref = torch.argmax(live.to(torch.uint8)).view(1)
            ref_slot = torch.index_select(slots, 0, ref).clamp_min(0)
            addr = torch.where(live, slots, ref_slot)
            carried = plan.has_state & live
        else:
            addr = slots
            carried = plan.has_state

        # SM90 takes packed, sequence-ordered state (`state_indices` is
        # SM100/SM103 only), so gather here and scatter back.
        initial = torch.index_select(state, 0, addr).to(self._kernels.prefill_dtype)

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
        rows = plan.num_rows if self._captured else plan.borrowed
        if rows:
            q, k, v, g, beta = (
                _with_room(t, rows) for t in (q, k, v, g, beta)
            )

        # zero the padded tail so inf there can't poison the result
        keep = plan.token_mask[: q.shape[0], None, None].bool()
        def blank(t, shape):
            return torch.where(
                keep.view(shape), t, torch.zeros((), dtype=t.dtype, device=t.device),
            )

        v = blank(v, (-1, 1, 1))
        g = blank(g, (-1, 1))
        beta = blank(beta, (-1, 1))

        snapshot_args = {}
        if plan.every:
            snapshot_args = dict(
                state_checkpoints=plan.checkpoints,
                checkpoint_cu_starts=plan.checkpoint_cu,
                checkpoint_every_n_tokens=plan.every,
            )
        out, final = chunk_gated_delta_rule(
            q=q, k=k, v=v,
            # FlashInfer wants the decay exponentiated; FLA's takes log space
            g=torch.exp(g),
            beta=beta,
            scale=self._kernels.sm_scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=plan.cu_seqlens,
            # ignored by this kernel (see `qk_l2norm_in_kernel`); the caller
            # has already normalised q and k
            use_qk_l2norm_in_kernel=False,
            **snapshot_args,
        )

        if not self._kernels.has_sink:
            # Padding rows write to `ref_slot` the live row's own result, or with
            # no live row what is already there, so slot 0 is not disturbed.
            padded = torch.where(
                any_live,
                torch.index_select(final, 0, ref),
                torch.index_select(state, 0, ref_slot).to(final.dtype),
            )
            final = torch.where(live.view(-1, 1, 1, 1), final, padded)
        state.index_copy_(0, addr, final.to(state.dtype))
        if plan.every:
            index = plan.snapshot_index
            state.index_copy_(0, index[2], plan.checkpoints.index_select(0, index[0]).to(state.dtype))
        return out[:real_len]


class FlashInferDecode(GDNDecodeKernel):
    """FlashInfer's pretransposed decode, in place on the pool by slot."""

    # it marks q/k/v/a/b layout-dynamic, so the views go in as they are
    takes_strided_inputs = True

    def __init__(self, kernels: FlashInferGDNKernels):
        self._kernels = kernels
        # The cutlass decode kernel really does normalise in-kernel, which saves
        # the caller a norm + divide + two casts per q and k, per layer, per step.
        self.qk_l2norm_in_kernel = kernels.qk_l2norm

    def run(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        # Uninitialized output: the kernel writes every row, except one whose
        # slot is -1 (only with the sink off), which it would leave as found.
        # Its own default is a fresh `torch.zeros`, a fill per layer per step.
        output = (
            torch.empty(
                (q.shape[0], 1, v.shape[1], v.shape[2]),
                dtype=torch.bfloat16, device=q.device,
            )
            if self._kernels.has_sink and q.dtype == torch.bfloat16 else None
        )
        # row i is token i, so the token axis is just unsqueezed
        out, _ = gated_delta_rule_decode_pretranspose(
            q=q.unsqueeze(1), k=k.unsqueeze(1), v=v.unsqueeze(1),
            state=None,
            A_log=a_log,
            a=a.unsqueeze(1),
            dt_bias=dt_bias,
            b=b.unsqueeze(1),
            scale=self._kernels.sm_scale,
            output=output,
            initial_state=state,
            initial_state_indices=plan.slots,
            use_qk_l2norm=self._kernels.qk_l2norm,
        )
        return out.squeeze(1)


def build_gdn_kernels(
    backend: LinearAttnBackend,
    *,
    config: LinearAttnConfig,
    geometry: DeltaNetGeometry,
    state_dtype: torch.dtype,
    has_sink: bool,
    device: torch.device,
) -> GDNKernels:
    """The kernels ``backend`` (already resolved from ``AUTO``) names."""
    if backend is LinearAttnBackend.FLASHINFER:
        return FlashInferGDNKernels.build(
            config, geometry, state_dtype, has_sink, device,
        )
    if backend is LinearAttnBackend.AITER:
        # imports no kernel library; the strategy does that on first use
        from mstar.engine.resources.linear_attn.gdn_kernels_aiter import AiterGDNKernels

        return AiterGDNKernels.build(
            config, geometry, state_dtype, has_sink, device,
        )
    raise ValueError(f"Unknown linear attention backend {backend!r}")
