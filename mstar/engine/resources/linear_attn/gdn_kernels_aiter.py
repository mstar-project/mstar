"""GDN on AMD GPUs through AITER's Triton and FlyDSL kernels.

Every kernel reads and writes the K-last pool in place by slot, so nothing is
gathered, transposed or scattered:

- eager prefill: ``chunk_gated_delta_rule_opt_vk``, its chunk schedule built
  once per step in ``plan`` and shared by every layer. It reads that schedule
  on the host, so it cannot be captured.
- captured prefill: ``fused_rearrange_sigmoid_gated_delta_rule``, which walks
  each row's tokens in turn. One program per (row, head, value block) and no
  host schedule, but its cost grows with the longest row, which a bucket bounds
  only by its token count; buckets past ``max_captured_prefill_tokens`` stay
  eager on the chunked kernel.
- decode: FlyDSL's ``flydsl_gdr_decode``, or the sequential kernel above where
  FlyDSL cannot serve the geometry or the scale.

AITER is imported on first use, never at module import.
"""

from __future__ import annotations

import importlib.util
import logging

import torch

from mstar.engine.resources.linear_attn.config import LinearAttnConfig
from mstar.engine.resources.linear_attn.gdn_kernels import (
    GDNDecodeKernel,
    GDNKernels,
    GDNPrefillKernel,
    GDNPrefillPlan,
    _with_room,
    gdn_gates,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import SINK_SLOT
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)

# The largest prefill capture bucket served by the sequential kernel. It costs
# about 1.8 us per token of the bucket's longest row per layer on MI300X, where
# an eager chunked call costs about 270 us of host time per layer; see
# `AiterGDNKernels.uncapturable_prefill`.
DEFAULT_MAX_CAPTURED_PREFILL_TOKENS = 512

# Chunk length of AITER's chunked kernels; their HIP and FlyDSL stages are
# built for this one.
_CHUNK = 64

# Architectures FlyDSL's decode kernel is built for.
_FLYDSL_ARCHS = ("gfx942", "gfx950")


def _installed(name: str) -> bool:
    """Whether a package can be imported, without importing it."""
    return importlib.util.find_spec(name) is not None


class _FreshSlots:
    """Zeroes the slot of every row that starts fresh, before each layer's
    kernel.

    These kernels take an initial state from every row's slot, where the
    reference reads zeros for a row with no carried state. The index is built
    once per step: a fresh row's own slot, and for a carried row the sink,
    whose contents nothing reads, so it stays one slot per row whatever the
    step's layout and a captured graph can hold it.
    """

    def __init__(self):
        self._index: torch.Tensor | None = None
        self._rows = 0

    def plan(self, plan: GDNPrefillPlan) -> None:
        rows = plan.num_rows
        if self._index is None or self._index.numel() < rows:
            # a capture wrapper's first plan is its widest
            self._index = torch.empty(rows, dtype=torch.int64, device=plan.slots.device)
        index = self._index[:rows]
        index.copy_(plan.slots[:rows])
        index.masked_fill_(plan.has_state[:rows], SINK_SLOT)
        self._rows = rows

    def zero(self, state: torch.Tensor) -> None:
        state.index_fill_(0, self._index[: self._rows], 0)


class AiterGDNKernels(GDNKernels):
    def __init__(
        self,
        *,
        geometry: DeltaNetGeometry,
        state_dtype: torch.dtype,
        sm_scale: float | None,
        qk_l2norm: bool,
        max_captured_prefill_tokens: int,
        flydsl_decode: bool,
    ):
        self.geometry = geometry
        self.state_dtype = state_dtype
        self.scale = (
            sm_scale if sm_scale is not None else geometry.head_k_dim ** -0.5
        )
        self.qk_l2norm = qk_l2norm
        self.max_captured_prefill_tokens = max_captured_prefill_tokens
        self.flydsl_decode = flydsl_decode
        # dt_bias in the activations' dtype, which FlyDSL's decode takes; see
        # `dt_bias_as`
        self._dt_bias: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
        # [0, 1, ..., n] for one-token rows, grown outside capture
        self._row_offsets: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        state_dtype: torch.dtype,
        has_sink: bool,
        device: torch.device,
    ) -> AiterGDNKernels:
        if not _installed("aiter"):
            raise ImportError(
                "the 'aiter' linear attention backend needs AMD's aiter "
                "package, which is not installed"
            )
        if not has_sink:
            # The chunked kernel addresses the pool with every row's slot
            # unguarded, so a padding row needs a real one to land on.
            raise ValueError(
                "the 'aiter' GDN kernels need the pool's sink slot: a -1 "
                "padding row would index the pool out of bounds. Unset "
                "RecurrentStateConfig.disable_sink_slot."
            )
        if state_dtype not in (torch.float32, torch.bfloat16):
            raise NotImplementedError(
                f"gdn state dtype {state_dtype} is not supported: AITER's GDN "
                "kernels keep the state in fp32 or bf16."
            )
        k, v = geometry.head_k_dim, geometry.head_v_dim
        if k > 256 or k & (k - 1) or v % 32:
            raise NotImplementedError(
                f"AITER's GDN kernels need a power-of-two K <= 256 and V a "
                f"multiple of 32, got K={k} V={v}"
            )

        bound = config.max_captured_prefill_tokens
        if bound is None:
            bound = DEFAULT_MAX_CAPTURED_PREFILL_TOKENS

        reason = cls._flydsl_decode_unsupported(config, geometry, state_dtype, device)
        if reason is not None:
            logger.info(
                "gdn decode: AITER's sequential kernel, not FlyDSL's (%s)", reason,
            )
        return cls(
            geometry=geometry,
            state_dtype=state_dtype,
            sm_scale=config.sm_scale,
            qk_l2norm=config.qk_l2norm,
            max_captured_prefill_tokens=bound,
            flydsl_decode=reason is None,
        )

    @staticmethod
    def _flydsl_decode_unsupported(
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        state_dtype: torch.dtype,
        device: torch.device,
    ) -> str | None:
        """Why FlyDSL's decode cannot serve this deployment, or None."""
        k, v = geometry.head_k_dim, geometry.head_v_dim
        if config.sm_scale is not None and abs(config.sm_scale * k ** 0.5 - 1) > 1e-6:
            return f"its scale is fixed at 1/sqrt(K), and sm_scale={config.sm_scale}"
        # 16-byte state vectors along K, across an 8-lane K group
        vector = 4 if state_dtype is torch.float32 else 8
        if k % (8 * vector) or v % 32:
            return f"K={k} V={v} does not tile its {state_dtype} state"
        if device.type != "cuda" or current_platform.arch(device) not in _FLYDSL_ARCHS:
            return "it is built for gfx942 and gfx950 only"
        if not _installed("flydsl"):
            return "flydsl is not installed"
        return None

    # Kernel objects

    def uncapturable_prefill(self, num_tokens: int) -> str | None:
        if num_tokens <= self.max_captured_prefill_tokens:
            return None
        return (
            f"AITER's chunked GDN prefill builds its schedule on the host, and "
            f"a {num_tokens}-token bucket is past the sequential kernel's "
            f"bound of {self.max_captured_prefill_tokens} "
            "(linear_attn.max_captured_prefill_tokens)"
        )

    def chunked_prefill(self) -> GDNPrefillKernel:
        return AiterChunkedPrefill(self)

    def captured_prefill(self, bs: int, num_tokens: int) -> GDNPrefillKernel:
        return AiterSequentialPrefill(self, bs=bs, num_tokens=num_tokens)

    def decode(self, bs: int | None) -> GDNDecodeKernel:
        if self.flydsl_decode:
            return FlyDSLDecode(self)
        return AiterSequentialDecode(self)

    # Shared state

    def dt_bias_as(self, dt_bias: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``dt_bias`` cast to ``dtype``, once per layer.

        Qwen3.5's checkpoint stores it in bf16 and the model widens it to
        fp32, so the cast back is exact there. Cached outside capture: a copy
        made while capturing would live in the graph's pool and be written only
        by replays.
        """
        hit = self._dt_bias.get(id(dt_bias))
        if (
            hit is None or hit[0] is not dt_bias or hit[1] != dt_bias._version
            or hit[2].dtype != dtype
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "gdn decode first saw this layer's dt_bias inside graph "
                    "capture; capture must follow an eager warmup"
                )
            hit = (dt_bias, dt_bias._version, dt_bias.detach().to(dtype))
            self._dt_bias[id(dt_bias)] = hit
        return hit[2]

    def row_offsets(self, rows: int, device: torch.device) -> torch.Tensor:
        """``cu_seqlens`` for ``rows`` one-token rows, from a buffer allocated
        outside capture."""
        if self._row_offsets is None or self._row_offsets.numel() <= rows:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "gdn decode needs more row offsets inside graph capture "
                    "than its eager warmup allocated"
                )
            self._row_offsets = torch.arange(
                max(rows, 128) + 1, dtype=torch.int32, device=device,
            )
        return self._row_offsets[: rows + 1]


class AiterChunkedPrefill(GDNPrefillKernel):
    """``chunk_gated_delta_rule_opt_vk`` with in-place pool slots, for eager
    steps."""

    # its own L2 norm reads q and k as contiguous rows, which the caller's
    # normalisation hands it anyway
    qk_l2norm_in_kernel = False
    takes_strided_inputs = False

    def __init__(self, kernels: AiterGDNKernels):
        self._kernels = kernels
        self._schedule = None
        self._fresh = _FreshSlots()

    def plan(self, plan: GDNPrefillPlan) -> None:
        from aiter.ops.triton.gated_delta_net import (
            build_gated_delta_rule_prefill_metadata,
        )

        # Bound to `plan.cu_seqlens` (the object and its version), which every
        # layer of this step passes back unchanged.
        self._schedule = build_gated_delta_rule_prefill_metadata(
            plan.spans, cu_seqlens=plan.cu_seqlens, chunk_size=_CHUNK,
        )
        self._fresh.plan(plan)

    def run(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

        # an eager step's buffers are reused and sized to the widest batch
        # seen; a mismatch here reads past the end asynchronously
        assert q.shape[0] == plan.real_tokens, (q.shape, plan.real_tokens)
        self._fresh.zero(state)
        g, beta = gdn_gates(a, b, a_log, dt_bias)
        # the token `plan` gave each zero-span row, past the real ones
        if plan.borrowed:
            q, k, v, g, beta = (
                _with_room(t, plan.borrowed) for t in (q, k, v, g, beta)
            )
        geometry = self._kernels.geometry
        out, _ = chunk_gated_delta_rule_opt_vk(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            g=g.unsqueeze(0), beta=beta.unsqueeze(0),
            scale=self._kernels.scale,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=plan.cu_seqlens,
            state_dtype=state.dtype,
            # the prebuilt HIP hidden-state stage takes bf16 K=V=128 only
            use_chunk_hip=(
                q.dtype is torch.bfloat16
                and geometry.head_k_dim == 128 and geometry.head_v_dim == 128
            ),
            # AITER falls back to Triton where FlyDSL's prepare cannot serve
            use_prepare_flydsl=True,
            prefill_metadata=self._schedule,
            initial_state_indices=plan.slots,
            inplace_final_state=True,
        )
        return out.squeeze(0)[: plan.real_tokens]


class AiterSequentialPrefill(GDNPrefillKernel):
    """``fused_rearrange_sigmoid_gated_delta_rule`` over a capture bucket.

    The kernel reads a row's initial state from column 0 of a ``[rows,
    tokens]`` slot map and stores the state after token t to column t where it
    is not -1, so each row gets its slot at column 0 and at its last token. A
    row the pool addresses to its sink (padding, or nothing to extend) gets -1
    throughout and is skipped.
    """

    # q/k/v are packed into one [tokens, 2K+V] operand here, and a/b copied
    # by AITER's wrapper
    takes_strided_inputs = True

    def __init__(self, kernels: AiterGDNKernels, bs: int, num_tokens: int):
        self._kernels = kernels
        self.qk_l2norm_in_kernel = kernels.qk_l2norm
        self._bs, self._num_tokens = bs, num_tokens
        self._fresh = _FreshSlots()
        # A captured graph holds the map's address, so it is sized once to the
        # bucket (no row of it is longer than its token count), on first plan.
        self._map: torch.Tensor | None = None
        self._rows = 0

    def _allocate(self, device: torch.device) -> None:
        i32 = dict(dtype=torch.int32, device=device)
        self._map = torch.full((self._bs, self._num_tokens), -1, **i32)
        self._last = torch.empty(self._bs, **i32)
        self._cols = torch.arange(self._num_tokens, **i32)
        self._first = self._cols == 0
        self._none = torch.full((), -1, **i32)

    def plan(self, plan: GDNPrefillPlan) -> None:
        rows = plan.num_rows
        if rows > self._bs or max(plan.spans, default=0) > self._num_tokens:
            raise ValueError(
                f"a {rows}-row step with spans up to {max(plan.spans, default=0)} "
                f"does not fit a bucket of {self._bs} rows and "
                f"{self._num_tokens} tokens"
            )
        device = plan.slots.device
        if self._map is None:
            self._allocate(device)
        # Each row's last token, from the host; -1 for a row the pool sends to
        # its sink, which then never matches. The slots are already on device.
        last = [span - 1 if span else -1 for span in plan.real_spans]
        self._last[:rows].copy_(
            torch.tensor(last, dtype=torch.int32, pin_memory=device.type == "cuda"),
            non_blocking=True,
        )
        ends = self._last[:rows, None]
        cells = (self._cols == ends) | (self._first & (ends >= 0))
        torch.where(
            cells, plan.slots[:rows, None], self._none, out=self._map[:rows],
        )
        self._fresh.plan(plan)
        self._rows = rows

    def run(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from aiter.ops.triton.gated_delta_net import (
            fused_rearrange_sigmoid_gated_delta_rule,
        )

        n = q.shape[0]
        self._fresh.zero(state)
        qkv = torch.cat((q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)), dim=-1)
        # Skipped rows and a bucket's tail past the last row are never written;
        # zeros there keep the tail finite for the layers after.
        out = torch.zeros((n, v.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
        fused_rearrange_sigmoid_gated_delta_rule(
            A_log=a_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv,
            key_dim=q.shape[1] * q.shape[2], value_dim=v.shape[1] * v.shape[2],
            head_k_dim=q.shape[2], head_v_dim=v.shape[2],
            scale=self._kernels.scale,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=plan.cu_seqlens,
            ssm_state_indices=self._map[: self._rows],
            use_qk_l2norm_in_kernel=self._kernels.qk_l2norm,
            core_attn_out=out,
        )
        return out


class FlyDSLDecode(GDNDecodeKernel):
    """FlyDSL's decode: q/k/v/a/b read as the projection's column views, the
    pool updated in place by slot. A -1 slot gets zeros and leaves the pool
    alone (only without a sink, which this backend refuses).

    Compiled on first call per launch shape (strides and a tiling chosen from
    rows x heads), so each capture bucket compiles in its eager warmup.
    """

    takes_strided_inputs = True

    def __init__(self, kernels: AiterGDNKernels):
        self._kernels = kernels
        self.qk_l2norm_in_kernel = kernels.qk_l2norm

    def run(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

        n = q.shape[0]
        out = torch.empty((n, 1, v.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
        # one token per row: the sequence axis is unsqueezed
        flydsl_gdr_decode(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            a.unsqueeze(1), b.unsqueeze(1),
            self._kernels.dt_bias_as(dt_bias, q.dtype), a_log,
            plan.slots, state, out,
            use_qk_l2norm=self._kernels.qk_l2norm,
            need_shuffle_state=False,
        )
        return out.view(n, v.shape[1], v.shape[2])


class AiterSequentialDecode(GDNDecodeKernel):
    """The sequential kernel at one token per row, for geometries or scales
    FlyDSL's decode does not take."""

    takes_strided_inputs = True

    def __init__(self, kernels: AiterGDNKernels):
        self._kernels = kernels
        self.qk_l2norm_in_kernel = kernels.qk_l2norm

    def run(self, plan, q, k, v, a, b, state, a_log, dt_bias):
        from aiter.ops.triton.gated_delta_net import (
            fused_rearrange_sigmoid_gated_delta_rule,
        )

        n = q.shape[0]
        qkv = torch.cat((q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)), dim=-1)
        out = torch.empty((n, v.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
        fused_rearrange_sigmoid_gated_delta_rule(
            A_log=a_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv,
            key_dim=q.shape[1] * q.shape[2], value_dim=v.shape[1] * v.shape[2],
            head_k_dim=q.shape[2], head_v_dim=v.shape[2],
            scale=self._kernels.scale,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=self._kernels.row_offsets(n, q.device),
            # one token per row, so a 1-D map is each row's read and write slot
            ssm_state_indices=plan.slots,
            use_qk_l2norm_in_kernel=self._kernels.qk_l2norm,
            core_attn_out=out,
        )
        return out
