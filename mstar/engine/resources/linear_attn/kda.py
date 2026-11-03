"""Kimi delta attention planned against the recurrent state pool.

As in ``gdn.py``, the pool's addressing arrives through ``ctx.plan_results`` and a
layer's blocks reach ``run`` as tensors, which ``convenience.KDACallable`` resolves.
The math is a kernel bundle's, which the resource picks from the pool and its layers' gates:
``kda_triton.TritonKDAKernels`` where they run it, ``kda_torch.TorchKDAKernels``
elsewhere. A bundle implements ``layout`` (the token layout its prefill reads, or
None), ``run_paged`` and, for a speculative step, ``run_verify``. The plan stages the
layout into buffers that stay put under capture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn import kda_torch, kda_triton
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
from mstar.engine.resources.step import Segment, StepContext
from mstar.platforms import current_platform
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)


@dataclass
class KDAParams:
    """What the kernels need besides activations: per-rank weights and constants."""

    conv_weight: torch.Tensor  # [3P, W]: q | k | v
    A_log: torch.Tensor  # [H] fp32
    dt_bias: torch.Tensor  # [P] fp32
    lower_bound: float | None
    num_heads: int
    head_dim: int
    scale: float
    # the forget gate's up-projection [P, D], when g is its low-rank factor [T, D]
    f_b: torch.Tensor | None = None
    # the output gate's up-projection [P, D] and the norm, when the kernels apply them
    g_b: torch.Tensor | None = None
    norm_weight: torch.Tensor | None = None
    norm_eps: float = 1e-5


class SpecBlocks(NamedTuple):
    """One layer's speculative blocks (``DeltaNetGeometry.to_blocks(speculative_tokens=k)``),
    slot-major: the last verify block's inputs on two sides, then the per-slot count and
    side, which every layer shares."""

    prefix: torch.Tensor  # [S, 2, k + 1, 3P] pre-conv q | k | v
    g: torch.Tensor  # [S, 2, k + 1, H, K] the forget gate before its activation
    beta: torch.Tensor  # [S, 2, k + 1, H] before its sigmoid
    conv: torch.Tensor  # [S, 2, 3P, W - 1] the conv window before the block
    prefix_len: torch.Tensor  # [S, 1] int32: the block's accepted tokens
    side: torch.Tensor  # [S, 1] int32: the side holding them

    @classmethod
    def of(cls, pool, layer: int) -> SpecBlocks:
        """``layer``'s blocks of a ``RecurrentStatePool``."""
        return cls(
            *(pool.block(f"spec_{name}", layer) for name in ("prefix", "g", "beta", "conv")),
            pool.block("spec_len"), pool.block("spec_side"),
        )


@dataclass
class KDAPlan:
    """One label's rows this step, as the kernels read them."""

    slot_ids: torch.Tensor  # [rows] int32, the pool's addressing: the sink for padding rows
    has_state: torch.Tensor  # [rows] bool: False where the slot reads as zeros
    spans: tuple[int, ...]  # tokens per row
    num_tokens: int  # tokens the forward runs over: a capture bucket's under a lease
    is_decode: bool  # every row appends exactly one token
    layout: torch.Tensor | None = None  # the bundle's token layout on device
    # every row is a verify block of `block` tokens (LinearAttnStep.speculative)
    is_verify: bool = False
    block: int = 1
    # the verify step's accepted count went to the pool (`KDAManager.settle_verify`)
    settled: bool = False
    # the bucket slot a captured verify step records, so its replays know it settles
    capture_key: CGSlotKey | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def num_rows(self) -> int:
        return len(self.spans)

    def cached(self, key, build: Callable):
        """``build()`` once per plan: what every layer of the step reads."""
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]


def _default_kernels(device: torch.device, geometry: DeltaNetGeometry, conv_dtype: torch.dtype):
    """The fused kernels where they run the pool's shape, else the torch reference. The layers'
    gates can still move it to the reference (`KDAManager.bind_layer`)."""
    # the Triton kernels inline NVIDIA PTX
    if device.type != "cuda" or current_platform.is_rocm() or not kda_triton._HAS_TRITON:
        return kda_torch.TorchKDAKernels()
    if not kda_triton.supported(geometry.head_k_dim):
        reason = f"head dim {geometry.head_k_dim}"
    elif geometry.conv_kernel_size != 4:
        reason = f"conv width {geometry.conv_kernel_size}"
    elif conv_dtype is not torch.bfloat16:
        reason = f"{conv_dtype} activations"
    else:
        return kda_triton.TritonKDAKernels()
    logger.warning("kda: the fused kernels don't run %s; the torch reference runs", reason)
    return kda_torch.TorchKDAKernels()


class KDAManager(LinearAttnManager):
    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        num_layers: int,
        state_dtype: torch.dtype,
        device: torch.device,
        speculative_tokens: int = 0,
        conv_dtype: torch.dtype = torch.bfloat16,
    ):
        if (geometry.num_k_heads != geometry.num_v_heads
                or geometry.head_k_dim != geometry.head_v_dim):
            raise ValueError(
                f"KDA runs one head count and width for q, k and v; the pool's "
                f"geometry is {geometry}"
            )
        if state_dtype is not torch.float32:
            raise NotImplementedError(
                f"KDA state dtype {state_dtype} is not supported: the kernels "
                "keep the recurrence in fp32"
            )
        self.config = config
        self.geometry = geometry
        self.num_layers = num_layers
        self._device = device
        self._pool_key = config.recurrent_state
        self.kernels = _default_kernels(device, geometry, conv_dtype)
        # k of the pool's speculative blocks: a verify block spans up to k + 1 tokens
        self.speculative_tokens = speculative_tokens

        # per (bucket, slot, label): a captured graph reads its layout here
        self._cg_layout: dict[CGSlotKey, torch.Tensor] = {}
        # the verify captures that settled their count, which their replays then repeat
        self._cg_settled: set[CGSlotKey] = set()
        # requests that have verified: their state lags their accepted drafts, which only a
        # verify step replays
        self._verifying: set = set()
        self._layout_stager = PinnedStager(torch.int32)
        self._current: dict[str, KDAPlan] = {}
        # Pre-planning: the plan mutates no live state, so staging is caching
        # it. Follows the pool, which it depends on.
        self._preplanned = False
        self._cached_plan_output: dict[str, KDAPlan] | None = None

    def bind_layer(self, *, lower_bound: float, full_rank_output_gate: bool) -> None:
        """A layer's gates, as it binds. The fused kernels apply only the low-rank output gate
        and a forget-gate bound their decay blocks hold, so a layer they can't run moves every
        layer to the torch reference."""
        if not isinstance(self.kernels, kda_triton.TritonKDAKernels):
            return
        cfg = self.kernels.config
        if full_rank_output_gate:
            reason = "a full-rank output gate"
        elif not kda_triton.bound_fits(lower_bound, min(cfg.row_block, cfg.chunk)):
            reason = f"forget-gate bound {lower_bound}"
        else:
            return
        if self._current or self._preplanned:
            raise RuntimeError("kda: a layer bound after the resource planned with its kernels")
        logger.warning("kda: the fused kernels don't run %s; the torch reference runs", reason)
        self.kernels = kda_torch.TorchKDAKernels()

    @property
    def prefill_capturable(self) -> bool:
        """Whether a captured prefill replays with each step's spans on these kernels."""
        return self.kernels.prefill_capturable

    def depends_on(self) -> set[str]:
        # the pool plans first; its addressing arrives via `ctx.plan_results`
        return {self._pool_key}

    # Step lifecycle

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: LinearAttnStep, ctx: StepContext):
        assert not (self._preplanned and ctx.is_preplan), (
            "kda preplan is already pending; clear_preplan before planning a "
            "different step ahead"
        )
        self.reset_default_cursors()
        if self._preplanned:
            self._current = self._cached_plan_output
            self.clear_preplan()
            return self._current

        if (not step.speculative and self._verifying
                and not self._verifying.isdisjoint(ctx.request_ids)):
            raise ValueError(
                "kda: a step that doesn't verify, for a request that verified; its state is "
                "behind the drafts it accepted, which only a verify step replays"
            )
        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]
        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            self._current[label] = self._build_plan(
                label, segments, addressing[label], ctx, step.speculative,
            )
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current
        return self._current

    def clear_preplan(self):
        self._preplanned = False
        self._cached_plan_output = None

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def _build_plan(
        self, label: str, segments: list[Segment],
        addressing: RecurrentAddressing, ctx: StepContext, speculative: bool = False,
    ) -> KDAPlan:
        spans = tuple(max(seg.span, 0) for seg in segments)
        rows = len(spans)
        lease = ctx.slot_lease
        num_tokens = lease.bucket.num_tokens if lease is not None else sum(spans)
        # as GDN: a packed capture replays uneven spans, so it always takes the prefill
        # kernels, even when it captured one token per row
        is_decode = (not speculative and not ctx.force_prefill and rows > 0
                     and all(s == 1 for s in spans))
        plan = KDAPlan(
            slot_ids=addressing.slot_indices[:rows],
            has_state=addressing.has_state[:rows],
            spans=spans, num_tokens=num_tokens, is_decode=is_decode,
        )
        if speculative:
            block = spans[0] if spans else 1
            if not 1 <= block <= self.speculative_tokens + 1 or any(s != block for s in spans):
                raise ValueError(
                    f"kda: a verify step runs one block of 1..{self.speculative_tokens + 1} "
                    f"tokens per row (the pool's speculative blocks); got spans {spans}"
                )
            assert num_tokens == rows * block, (num_tokens, rows, block)
            plan.is_verify, plan.block = True, block
            if lease is not None:
                key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
                if ctx.capture:
                    plan.capture_key = key
                elif key not in self._cg_settled:
                    raise RuntimeError(
                        f"kda: the capture of {key} ran no set_prefix_len, so its replays "
                        "would leave each verify step's count unsettled"
                    )
        elif not is_decode and self.kernels is not None:
            host = self.kernels.layout(spans, num_tokens, lease is not None)
            if host is not None:
                plan.layout = self._stage_layout(host, label, ctx)
        return plan

    def _stage_layout(self, host: list[int], label: str, ctx: StepContext) -> torch.Tensor:
        """A fresh device copy eager; under a lease, the static buffer of that
        (bucket, slot, label), which every replay of the bucket fills at the
        same size.

        No ``force_double_buffer``: the stager's ring never rewrites a host
        buffer whose copy is in flight; pre-plan gets a second slot from
        ``supports_preplan``, and without it the copy queues behind the
        previous replay on the stream that runs the step."""
        lease = ctx.slot_lease
        if lease is None:
            return self._layout_stager.to_device(host, self._device)
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        buf = self._cg_layout.get(key)
        if buf is None:
            buf = self._cg_layout[key] = torch.empty(
                len(host), dtype=torch.int32, device=self._device,
            )
        assert buf.numel() == len(host), (
            f"kda layout for {key} holds {buf.numel()} entries but this step "
            f"planned {len(host)}"
        )
        self._layout_stager.copy_(buf, host)
        return buf

    def current_plan(self, label: str | None = None) -> KDAPlan:
        if label is None:
            label = self._default_label
        found = self._current.get(label)
        if found is None:
            raise KeyError(
                f"kda has no plan for label {label!r}; this step planned "
                f"{sorted(self._current)}. Every label a forward runs must "
                "carry a segment in the step declaration."
            )
        return found

    def commit(self, step: LinearAttnStep, ctx: StepContext) -> None:
        if not step.speculative or ctx.capture:
            return
        if ctx.slot_lease is None:
            # an eager forward ran this step's Python, which had to settle it
            for label, plan in self._current.items():
                if not plan.settled:
                    raise RuntimeError(
                        f"kda: a verify step under {label!r} ended without set_prefix_len"
                    )
        self._verifying.update(ctx.request_ids)

    def remove_request(self, rid: str) -> None:
        self._verifying.discard(rid)

    # Submodule-level functionality

    @torch.compiler.disable
    def settle_verify(self, label: str | None = None) -> KDAPlan:
        """This verify step's plan, for ``KDACallable.set_prefix_len`` to write its accepted
        count; once per step."""
        plan = self.current_plan(label)
        if not plan.is_verify:
            raise RuntimeError("kda: set_prefix_len after a step that didn't verify")
        if plan.settled:
            raise RuntimeError("kda: set_prefix_len twice for one verify step")
        plan.settled = True
        if plan.capture_key is not None:
            self._cg_settled.add(plan.capture_key)
        return plan

    @torch.compiler.disable
    def run(
        self,
        qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        conv_layer: torch.Tensor,
        state_layer: torch.Tensor,
        params: KDAParams,
        gate: torch.Tensor | None = None,
        label: str | None = None,
        spec: SpecBlocks | None = None,
    ) -> torch.Tensor:
        """One layer's KDA over this step's packed tokens.

        ``qkv [T, 3P]`` is pre-conv, ``beta [T, H]`` pre-sigmoid, and ``g`` the
        forget gate pre-activation, ``[T, H, D]`` or its low-rank factor
        ``[T, D]`` with ``params.f_b``. ``conv_layer`` / ``state_layer`` are the
        layer's pool blocks, updated in place, and ``spec`` its speculative
        blocks for a verify step. Returns ``[T, H, D]``, or the gated-normed
        ``[T, P]`` when ``gate`` (the output gate's low-rank factor) comes with
        ``params.g_b``.
        """
        plan = self.current_plan(label)
        if plan.is_verify:
            if spec is None:
                raise ValueError("kda: a verify step needs the layer's SpecBlocks")
            return self.kernels.run_verify(
                qkv, g, beta, plan, conv_layer, state_layer, spec, params, gate=gate,
            )
        return self.kernels.run_paged(
            qkv, g, beta, plan, conv_layer, state_layer, params, gate=gate,
        )

    # Engine lifecycle

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        # The layout buffers are sized on the first plan of each bucket; a
        # bucket's layout has one size whatever the split of its tokens.
        del slots, max_bs, max_seq_len

    def cleanup(self):
        self._cg_layout.clear()
        self._cg_settled.clear()
        self._current.clear()
