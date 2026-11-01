"""Gated delta rule, on the kernels a ``GDNKernels`` names for this platform.

One kernel per batch: all-one-token rows take the recurrent decode path,
anything else the prefill path, which also handles span-1 and zero-span padding.

The pool's plan result arrives through ``ctx.plan_results`` and its per-layer
block as a plain tensor argument to ``run``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, Resource
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.linear_attn.gdn_kernels import GDNKernels
from mstar.engine.resources.linear_attn.wrappers import (
    GDNDecodeWrapper,
    GDNMixedWrapper,
    GDNPrefillWrapper,
    GDNVerifyWrapper,
    GDNWrapper,
    PrefillStagers,
)
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import (
    NO_SLOT,
    SINK_SLOT,
    RecurrentAddressing,
    RecurrentStatePool,
    checkpoint_rows,
)
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import Segment, SlotLease, StepContext
from mstar.utils.causal_conv1d import PAD_SLOT_ID

logger = logging.getLogger(__name__)


class GDNManager(LinearAttnManager):
    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        num_layers: int,
        state_dtype: torch.dtype,
        has_sink: bool,
        device: torch.device,
        kernels: GDNKernels,
        max_rows_in_combined: Mapping[str, int | None] | None = None,
    ):
        self.config = config
        self._splits = {split.combined_walk: split for split in config.splits}
        self._max_rows_in_combined = max_rows_in_combined or {}
        self.geometry = geometry
        self.num_layers = num_layers
        self.state_dtype = state_dtype
        self._device = device
        self._pool_key = config.recurrent_state
        self._spec_key = config.spec_decode
        self._has_sink = has_sink
        # chosen once, by `LinearAttnManager.build`; every wrapper runs these
        self._kernels = kernels

        # See `build_cuda_graph_buffers`.
        self._cg_max_bs = 0
        self._cg_max_tokens = 0
        self._cg_wrappers: dict[CGSlotKey, GDNWrapper] = {}
        # The prefills' state checkpoints, sized once to the largest bucket and
        # shared: one forward runs at a time, and each graph holds the address.
        # An eager step fits it (`RecurrentStatePool._checkpoint_every`), and
        # without graphs it grows to the largest step.
        self._checkpoints: torch.Tensor | None = None
        self._pool: RecurrentStatePool | None = None

        # (label, is_decode) -> wrapper
        self._eager_wrappers: dict[tuple[str, bool], GDNWrapper] = {}
        self._prefill_stagers = PrefillStagers()

        self._current: dict[str, GDNWrapper] = {}

        # Pre-planning mutates no live state, so staging is just caching the
        # result.
        self._preplanned = False
        self._cached_plan_output: dict[str, GDNWrapper] | None = None

    def depends_on(self) -> set[str]:
        # the pool plans first; its addressing arrives via `ctx.plan_results`,
        # and a speculative step's start slots via the speculative resource's
        if self._spec_key is not None:
            return {self._pool_key, self._spec_key}
        return {self._pool_key}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._pool = resources.get(self._pool_key)
        if self._pool is not None and not self._kernels.writes_snapshots:
            self._pool.refuse_snapshots(
                f"{self._pool_key!r}'s GDN kernels ({type(self._kernels).__name__}) keep no state checkpoints"
            )

    # Step lifecycle

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: LinearAttnStep, ctx: StepContext):
        assert not (self._preplanned and ctx.is_preplan), (
            "linear-attn preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        self.reset_default_cursors()
        if self._preplanned:
            self._current = self._cached_plan_output
            self.clear_preplan()
            return self._current

        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]
        verify = self._verify_plan(ctx)

        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            if verify is not None:
                self._current[label] = self._build_verify(label, addressing[label], verify, ctx)
                continue
            self._current[label] = self._build_plan(
                label, segments, addressing[label], ctx
            )
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current
        return self._current

    def clear_preplan(self):
        # The wrappers' plan state is overwritten by whatever plans next, so
        # an abandoned stage leaves nothing to rewind.
        self._preplanned = False
        self._cached_plan_output = None

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def _verify_plan(self, ctx: StepContext):
        """The speculative resource's plan, if this step verifies drafts."""
        if self._spec_key is None:
            return None
        from mstar.engine.resources.spec_decode.config import SpecPhase

        plan = ctx.plan_results.get(self._spec_key)
        if plan is None or plan.phase is not SpecPhase.VERIFY:
            return None
        return plan

    def _build_verify(
        self, label: str, addressing: RecurrentAddressing, verify, ctx: StepContext,
    ) -> GDNWrapper:
        groups = addressing.host_slot_groups
        if groups is None:
            raise RuntimeError(
                "a speculative verify step needs the pool's slot groups; declare "
                "the pool with RecurrentStateConfig.spec_tokens"
            )
        lease = ctx.slot_lease
        if lease is None:
            key, store, bs = (label, "verify"), self._eager_wrappers, None
        else:
            key, store = CGSlotKey(lease.bucket, lease.slot, label), self._cg_wrappers
            bs = max(self._cg_max_bs, lease.bucket.bs)
        wrapper = store.get(key)
        if wrapper is None:
            wrapper = store[key] = GDNVerifyWrapper(
                device=self._device,
                pad_slot_id=PAD_SLOT_ID,
                tokens_per_row=len(groups[0]) if groups else 1,
                bs=bs,
                accelerator_graph=lease is not None,
                null_slot_id=SINK_SLOT if self._has_sink else NO_SLOT,
                kernels=self._kernels,
            )
        wrapper.plan(groups[: addressing.num_rows], verify.num_accepted)
        return wrapper

    def _get_wrapper(
        self, label: str, is_decode: bool, lease: SlotLease | None = None,
        split: CombinedSplit | None = None,
    ) -> GDNWrapper:
        """The wrapper this (label, walk) plans into, built once and reused.

        Under capture it is keyed per (bucket, slot, label) and sized to the
        bucket, since the graph holds its buffers' addresses.
        """
        if lease is None:
            # both walks can run eagerly under one label
            key, store = (label, "split" if split is not None else is_decode), self._eager_wrappers
            bs, tok = None, None
        else:
            key = CGSlotKey(lease.bucket, lease.slot, label)
            store = self._cg_wrappers
            bs = max(self._cg_max_bs, lease.bucket.bs)
            tok = lease.bucket.num_tokens

        if key not in store:
            null_slot_id = SINK_SLOT if self._has_sink else NO_SLOT
            if split is not None:
                store[key] = GDNMixedWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    bs=bs,
                    num_tokens=tok,
                    accelerator_graph=lease is not None,
                    has_sink_state=self._has_sink,
                    null_slot_id=null_slot_id,
                    max_prefill_rows=(
                        split.max_prefill_rows(self._max_rows_in_combined, lease.bucket.bs)
                        if lease is not None else None
                    ),
                    kernels=self._kernels,
                )
            elif is_decode:
                store[key] = GDNDecodeWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    bs=bs,
                    accelerator_graph=lease is not None,
                    null_slot_id=null_slot_id,
                    kernels=self._kernels,
                )
            else:
                store[key] = GDNPrefillWrapper(
                    device=self._device,
                    pad_slot_id=PAD_SLOT_ID,
                    bs=bs,
                    num_tokens=tok,
                    accelerator_graph=lease is not None,
                    has_sink_state=self._has_sink,
                    null_slot_id=null_slot_id,
                    kernels=self._kernels,
                    stagers=self._prefill_stagers,
                )
        return store[key]

    def _build_plan(
        self,
        label: str,
        segments: list[Segment],
        addressing: RecurrentAddressing,
        ctx: StepContext,
    ) -> GDNWrapper:
        spans = [seg.span for seg in segments]
        num_rows = len(spans)
        # a narrow, not a gather: padding rows keep pointing at the sink
        slots = addressing.slot_indices[:num_rows]

        # One kernel for the step: the chunked one unless every row is one
        # token, or always under a packed capture, which replays mixed steps.
        # A split walk's step (decode rows beside prompt rows) takes both.
        is_decode = not ctx.force_prefill and bool(spans) and all(s == 1 for s in spans)
        walk = ctx.slot_lease.bucket.graph_walk if ctx.slot_lease is not None else ctx.graph_walk
        split = self._splits.get(walk)
        every = addressing.snapshot_every
        walks = split.prefill_walks if split is not None else (walk,)
        if every and self._pool.keyed_walks(label).isdisjoint(walks):
            # a walk the KV cache doesn't key keeps no snapshots, so its graphs skip the checkpoints too
            every = 0
        assert every or not addressing.snapshots, (
            f"gdn: a {walk!r} step keeps snapshots for {label!r}, which its KV cache doesn't key there"
        )
        checkpoints = None
        if every and ctx.slot_lease is not None:
            checkpoints = self._checkpoint_buffer(checkpoint_rows(self._cg_max_tokens, self._cg_max_bs, every))
        elif every and addressing.snapshots:
            checkpoints = self._checkpoint_buffer(sum(max(span, 1) // every for span in spans))
        if split is not None and addressing.host_slot_indices is not None:
            wrapper = self._get_wrapper(label, False, ctx.slot_lease, split=split)
            wrapper.plan(
                spans, addressing.host_slot_indices[:num_rows],
                addressing.host_has_state[:num_rows],
                addressing.snapshots, every, checkpoints,
            )
            return wrapper
        wrapper = self._get_wrapper(label, is_decode, ctx.slot_lease)
        if is_decode:
            wrapper.plan(spans, slots)
        else:
            wrapper.plan(
                spans, slots, addressing.has_state[:num_rows],
                addressing.snapshots, every, checkpoints,
            )
        return wrapper

    def _checkpoint_buffer(self, rows: int) -> torch.Tensor:
        if self._checkpoints is None or self._checkpoints.shape[0] < rows:
            geometry = self.geometry
            self._checkpoints = torch.empty(
                (rows, geometry.num_v_heads, geometry.head_v_dim, geometry.head_k_dim),
                dtype=torch.float32, device=self._device,
            )
        return self._checkpoints

    def current_plan(self, label: str | None = None) -> GDNWrapper:
        if label is None:
            label = self._default_label
        found = self._current.get(label)
        if found is None:
            raise KeyError(
                f"gdn has no plan for label {label!r}; this step planned "
                f"{sorted(self._current)}. Every label a forward runs must "
                "carry a segment in the step declaration."
            )
        return found

    # Submodule-level functionality

    @torch.compiler.disable
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state_layer: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """One layer's gated delta rule over this step's packed tokens.

        ``q``/``k`` are ``[total_tokens, H, K]``, ``v`` ``[total_tokens, HV,
        V]``, and the return ``[total_tokens, HV, V]``. ``state_layer`` is this
        layer's ``[max_slots, HV, V, K]`` view of the pool, updated in place.

        ``a``/``b`` are the raw gate projections, not the decay and learning
        rate. q and k are L2-normalized here only where the kernel will not do
        it (see `GDNWrapper.qk_l2norm_in_kernel`).
        """
        plan = self.current_plan(label)
        if self.config.qk_l2norm and not plan.qk_l2norm_in_kernel:
            q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
            k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        # a split projection gives column views; copy only for a kernel that
        # needs them contiguous
        if not plan.takes_strided_inputs:
            v = v.contiguous()
            a = a.contiguous()
            b = b.contiguous()

        return plan.run(q, k, v, a, b, state_layer, a_log, dt_bias)

    @torch.compiler.disable
    def run_conv(
        self,
        x: torch.Tensor,
        conv_layer: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
        label: str | None = None,
    ) -> torch.Tensor:
        """The depthwise conv over ``[q|k|v]``, before the delta rule.

        ``x`` is ``[total_tokens, conv_dim]``, ``weight``
        ``[conv_dim, kernel_size]``, and ``conv_layer`` this layer's
        ``[max_slots, conv_dim, width]`` view of the pool's conv block, updated
        in place. Padding rows carry the pool's null slot, which both conv
        kernels skip.
        """
        return self.current_plan(label).run_conv(
            x=x, conv_layer=conv_layer, weight=weight, bias=bias,
            activation=activation,
        )

    # Engine lifecycle

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots
        # A floor under every capture wrapper's row count, so none reallocates
        # if a wider layout reaches it; announced before any bucket plans.
        self._cg_max_bs = max(self._cg_max_bs, max_bs)
        self._cg_max_tokens = max(self._cg_max_tokens, max_seq_len)

    def capture_refusal(self, spec: CGSlotSpec) -> str | None:
        # one-token rows take the decode kernel, which every backend captures;
        # a longer row takes the prefill kernel the backend can capture, whose
        # cost may grow with that row
        if spec.max_row_tokens <= 1:
            return None
        return self._kernels.uncapturable_prefill(spec.max_row_tokens)

    def cleanup(self):
        self._cg_wrappers.clear()
        self._eager_wrappers.clear()
        self._current.clear()
        self._checkpoints = None
