"""Kimi delta attention planned against the recurrent state pool.

Same shape as ``gdn.py``: the pool's addressing arrives through
``ctx.plan_results`` and its per-layer blocks as plain tensors to ``run``. The
math is a kernel bundle's: ``kda_triton.TritonKDAKernels`` on CUDA, or one a
model installs with ``set_kernels`` (its torch reference off-GPU). A bundle
implements ``layout(spans, num_tokens, fixed)``, the token layout its prefill
reads on device (or None), ``run_paged(qkv, g, beta, plan, conv, state,
params, gate=None)``, and for a speculative step ``run_verify`` with the
layer's ``SpecBlocks`` after the state. The plan stages the layout, into
buffers that stay put under CUDA-graph capture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn import kda_triton
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
from mstar.engine.resources.step import Segment, StepContext
from mstar.utils.pinned_staging import pinned, to_device_async

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
    # The forget gate's up-projection [P, D]: set when the gate reaches the
    # kernels as its low-rank factor [T, D] rather than as [T, H, D].
    f_b: torch.Tensor | None = None
    # The output gate's up-projection [P, D] and the gated RMSNorm: set when
    # the kernels take the output gate too and return the normed [T, P].
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
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def num_rows(self) -> int:
        return len(self.spans)

    def cached(self, key, build: Callable):
        """``build()`` once per plan: what every layer of the step reads."""
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]


def _default_kernels(device: torch.device, head_dim: int):
    """The fused Triton bundle on CUDA at a head dim it is validated for;
    elsewhere None, for a model to fill."""
    if device.type != "cuda" or not kda_triton._HAS_TRITON or not kda_triton.supported(head_dim):
        return None
    return kda_triton.TritonKDAKernels()


class KDAManager(LinearAttnManager):
    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        num_layers: int,
        state_dtype: torch.dtype,
        device: torch.device,
        kernels=None,
        speculative_tokens: int = 0,
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
        self.kernels = kernels if kernels is not None else _default_kernels(device, geometry.head_k_dim)
        # k of the pool's speculative blocks: a verify block spans up to k + 1 tokens
        self.speculative_tokens = speculative_tokens

        # per (bucket, slot, label): a captured graph reads its layout here
        self._cg_layout: dict[CGSlotKey, torch.Tensor] = {}
        self._current: dict[str, KDAPlan] = {}
        # Pre-planning: the plan mutates no live state, so staging is caching
        # it. Follows the pool, which it depends on.
        self._preplanned = False
        self._cached_plan_output: dict[str, KDAPlan] | None = None

    def set_kernels(self, kernels) -> None:
        """Install the bundle every layer runs; see the module docstring."""
        self.kernels = kernels

    def depends_on(self) -> set[str]:
        # so the pool plans first and its addressing reaches us through
        # `ctx.plan_results`; see `StepRunner.topo_sort`
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
        if lease is not None:
            # A replay runs the kernels its bucket was captured with, so the walk decides,
            # not this step's spans: a prefill bucket of as many tokens as rows captures
            # one-token spans (decode kernels, before) and replays uneven ones.
            is_decode = not speculative and ctx.graph_walk == "decode"
        else:
            is_decode = (not speculative and rows > 0 and num_tokens == rows
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
        elif not is_decode and self.kernels is not None:
            host = self.kernels.layout(spans, num_tokens, lease is not None)
            if host is not None:
                plan.layout = self._stage_layout(host, label, ctx)
        return plan

    def _stage_layout(self, host: list[int], label: str, ctx: StepContext) -> torch.Tensor:
        """A fresh device copy eager; under a lease, the static buffer of that
        (bucket, slot, label), which every replay of the bucket fills at the
        same size."""
        lease = ctx.slot_lease
        if lease is None:
            return to_device_async(host, torch.int32, self._device)
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
        buf.copy_(pinned(host, torch.int32), non_blocking=True)
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

    @contextmanager
    def token_window(self, pieces: list[tuple[int, int]], label: str | None = None) -> Iterator[None]:
        """Run this step's KDA over a window of its prefill tokens: ``pieces`` are (row, tokens)
        in packed order, each the next run of that row's tokens. A row's state carries over in
        its slot, so windows run in order compute what the whole step would."""
        if label is None:
            label = self._default_label
        full = self.current_plan(label)
        assert not (full.is_decode or full.is_verify), "kda: a token window splits a prefill"
        rows = to_device_async([row for row, _ in pieces], torch.long, self._device)
        spans = tuple(n for _, n in pieces)
        plan = KDAPlan(
            slot_ids=full.slot_ids.index_select(0, rows),
            has_state=full.has_state.index_select(0, rows),
            spans=spans, num_tokens=sum(spans), is_decode=False,
        )
        if self.kernels is not None:
            host = self.kernels.layout(spans, plan.num_tokens, False)
            if host is not None:
                plan.layout = to_device_async(host, torch.int32, self._device)
        self._current[label] = plan
        try:
            yield
        finally:
            self._current[label] = full

    # Submodule-level functionality

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

        ``qkv`` is ``[T, 3P]`` before the conv, ``beta`` ``[T, H]`` before its
        sigmoid, and ``g`` the forget gate before its activation: ``[T, H, D]``,
        or its low-rank factor ``[T, D]`` with ``params.f_b``. ``conv_layer``
        and ``state_layer`` are this layer's ``pool.block`` views, updated in
        place, and ``spec`` its speculative blocks, which a verify step needs.
        Returns ``[T, H, D]``, or the gated-normed ``[T, P]`` when ``gate``,
        the output gate's low-rank factor, comes with ``params.g_b``.
        """
        assert self.kernels is not None, (
            "no KDA kernels: off-GPU the model installs its reference with set_kernels"
        )
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

    @torch.compiler.disable
    def set_prefix_len(
        self, spec: SpecBlocks, accepted: torch.Tensor, label: str | None = None,
    ) -> None:
        """After a verify step: each row's first ``accepted + 1`` tokens (the
        accepted drafts and the token after them) are the prefix the next step
        replays, from the side this step wrote. Device ops only, so a captured
        step records them; padding rows write the sink."""
        plan = self.current_plan(label)
        rows = plan.num_rows
        slots = plan.slot_ids[:rows].long()
        count = (accepted[:rows] + 1).to(torch.int32).unsqueeze(1)
        spec.prefix_len.index_copy_(0, slots, count)
        spec.side.index_copy_(0, slots, 1 - spec.side.index_select(0, slots))

    # Engine lifecycle

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        # The layout buffers are sized on the first plan of each bucket; a
        # bucket's layout has one size whatever the split of its tokens.
        del slots, max_bs, max_seq_len

    def cleanup(self):
        self._cg_layout.clear()
        self._current.clear()
