"""Cacheless varlen attention through FlashInfer's ragged prefill wrapper."""

from collections.abc import Iterable

import torch

from mstar.engine.resources.attn.base import EagerSlotKey, WorkspacePool
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.base import (
    RaggedAttnManager,
    RaggedBlockCausalAttnManager,
    RaggedCrossAttnManager,
)
from mstar.engine.resources.attn.ragged.block_causal import RaggedBlockCausalWrapper
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.attn.ragged.wrappers import (
    RaggedCrossPrefillWrapper,
    RaggedPrefillWrapper,
    _RaggedPrefillBase,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.step import Segment, SlotLease, StepContext


class _FlashInferRaggedBase:
    """What the self- and cross-attention managers share: persistent wrappers per
    (bucket, slot, label) and per (label, slot), the preplan protocol, and ``run``.
    A subclass names its wrapper class and lays its step out in ``_plan_layouts``."""

    wrapper_class: type[_RaggedPrefillBase]

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
    ):
        self._config = config
        self._device = device
        self._dtype = dtype

        # label -> the wrapper this step's `run` attends through
        self._current_plan_states: dict[str, _RaggedPrefillBase] = {}

        # Persistent, because constructing one allocates FlashInfer's own
        # buffers and is far from free per step.
        self._eager_plan_states: dict[EagerSlotKey, _RaggedPrefillBase] = {}
        self._cg_plan_states: dict[CGSlotKey, _RaggedPrefillBase] = {}

        self._preplan_states: dict[str, _RaggedPrefillBase] = {}
        self._preplanned = False

        self._workspaces = WorkspacePool(device)

        self._kwargs = dict(
            device=device,
            q_data_type=dtype,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            sm_scale=config.sm_scale,
            backend=config.flashinfer_backend,
        )

    @property
    def force_double_buffer(self):
        # FlashInfer's plan stages the schedule into a pinned buffer it holds
        # per wrapper and H2Ds it on the stream
        return True

    def _cg_wrapper(
        self, lease: SlotLease, label: str,
    ) -> _RaggedPrefillBase:
        """The captured-graph wrapper for one (bucket, slot, label).

        Built on the first plan for that key, like the paged backend's. Its
        static buffers are fixed for its lifetime, so they are sized off the
        config's per-request ceilings and the bucket — NOT off this first
        plan's ``num_rows``, which is only one of the layouts the bucket will
        replay.
        """
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is not None:
            return wrapper

        bucket = lease.bucket
        max_segments = self._config.max_segments_for(bucket.bs)
        # the bucket's own token count is the ceiling; the per-request override
        # is for a runner that buckets by batch size alone
        max_tokens = max(
            bucket.num_tokens, self._config.max_tokens_for(bucket.bs) or 0
        )
        wrapper = self.wrapper_class(
            workspace_buffer=self._workspaces.get(label, lease.slot),
            use_cuda_graph=True,
            max_num_segments=max_segments,
            max_total_tokens=max_tokens,
            **self._kwargs,
        )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str, slot: int) -> _RaggedPrefillBase:
        """The persistent eager wrapper for one (label, slot).

        Slotted for the same reason the captured ones are, and on the same
        workspace; see `FlashInferManager._eager_wrapper`.
        """
        key = EagerSlotKey(label=label, slot=slot)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            wrapper = self._eager_plan_states[key] = self.wrapper_class(
                workspace_buffer=self._workspaces.get(label, slot),
                **self._kwargs,
            )
        return wrapper

    @property
    def supports_preplan(self):
        return True

    @staticmethod
    def _group_segments_by_label(
        segments: Iterable[Segment],
    ) -> dict[str, list[Segment]]:
        res: dict[str, list[Segment]] = {}
        for seg in segments:
            res.setdefault(seg.label, []).append(seg)
        return res

    @staticmethod
    def _cu_seqlens(segments: list[Segment]) -> torch.Tensor:
        # On the CPU: FlashInfer's plan wants it there anyway, and building it
        # on device would sync (fatally so under preplan). Pinned and freshly
        # built per plan: the graph wrapper hands this straight to FlashInfer's
        # non-blocking H2D, so a reused buffer could be overwritten while its
        # DMA is still in flight. A fresh pinned allocation is held by the
        # caching host allocator until the copy retires, keeping the plan async
        # without a per-step sync.
        cu = [0]
        for seg in segments:
            cu.append(cu[-1] + seg.span)
        return torch.tensor(cu, dtype=torch.int32, pin_memory=torch.cuda.is_available())

    def _wrapper_for(self, lease: SlotLease | None, slot: int, label: str) -> _RaggedPrefillBase:
        if lease is not None:
            return self._cg_wrapper(lease, label)
        return self._eager_wrapper(label, slot)

    def plan(self, step, ctx: StepContext):
        self.reset_default_cursors()

        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: an eager wrapper shares its "
            "workspace with the captured one on the same slot"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "ragged attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )

        if self._preplanned:
            # the wrappers were planned a step early against this same step's
            # layout; nothing left to do but promote them
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan \
            else self._current_plan_states

        # A label's segments are its whole layout — there is no cache to read
        # them off, so an undeclared label is unattendable. Clear rather than
        # update, so last step's wrapper can't be attended through by mistake.
        plan_states.clear()
        self._plan_layouts(step, ctx, plan_states)

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality

    def num_segments(self, label: str | None = None) -> int:
        """Real (unpadded) segment count this step planned under ``label``."""
        return self._wrapper(label).num_segments

    def _plan_layouts(self, step, ctx: StepContext, plan_states: dict[str, _RaggedPrefillBase]) -> None:
        raise NotImplementedError

    def _wrapper(self, label: str | None) -> _RaggedPrefillBase:
        if label is None:
            label = self._default_label
        wrapper = self._current_plan_states.get(label)
        if wrapper is None:
            raise KeyError(
                f"ragged attention has no plan for label {label!r}; this step "
                f"planned {sorted(self._current_plan_states)}. Every label a "
                "forward attends must carry a segment in the step declaration."
            )
        return wrapper

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """One layer's attention through the plan for ``label``.

        Not behind a custom op, unlike the paged backend's ``run``: the ragged
        caller (an encoder tower) has no per-layer KV write to keep in the same
        graph, so the break this costs is one per layer of a region that is
        CUDA-graph captured rather than compiled.
        """
        return self._wrapper(label).run(q, k, v)


class FlashInferRaggedManager(_FlashInferRaggedBase, RaggedAttnManager):
    """Self-attention: every label in the step is its own layout, each segment
    attending within itself."""

    wrapper_class = RaggedPrefillWrapper

    def _plan_layouts(self, step: AttentionStep, ctx: StepContext, plan_states) -> None:
        for label, segments in self._group_segments_by_label(step.segments or ()).items():
            wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            # TODO: cache the latest plan state
            wrapper.plan(cu_seqlens=self._cu_seqlens(segments), causal=step.causal)
            plan_states[label] = wrapper


class FlashInferRaggedCrossManager(_FlashInferRaggedBase, RaggedCrossAttnManager):
    """Cross-attention: each ``(q_label, kv_label)`` pair of the step is one plan,
    run under ``cross_label(q_label, kv_label)``."""

    wrapper_class = RaggedCrossPrefillWrapper

    def _plan_layouts(self, step: RaggedCrossAttentionStep, ctx: StepContext, plan_states) -> None:
        by_label = self._group_segments_by_label(step.segments or ())
        for q_label, kv_label in step.pairs:
            q_segments, kv_segments = by_label[q_label], by_label[kv_label]
            if [s.request_id for s in q_segments] != [s.request_id for s in kv_segments]:
                raise ValueError(
                    f"ragged cross-attention {q_label!r} <- {kv_label!r}: the two labels "
                    "must be declared for the same requests in the same order"
                )
            label = cross_label(q_label, kv_label)
            wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            wrapper.plan(self._cu_seqlens(q_segments), self._cu_seqlens(kv_segments))
            plan_states[label] = wrapper


class FlashInferRaggedBlockCausalManager(_FlashInferRaggedBase, RaggedBlockCausalAttnManager):
    """Block-causal self-attention: every label in the step is its own layout,
    each segment attending block-causally within itself."""

    wrapper_class = RaggedBlockCausalWrapper

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: RaggedAttentionConfig,
        block_size: int,
    ):
        super().__init__(device=device, dtype=dtype, config=config)
        self._kwargs["block_size"] = block_size

    def _plan_layouts(self, step: AttentionStep, ctx: StepContext, plan_states) -> None:
        if step.causal:
            raise ValueError(
                "block-causal attention is its own mask; declare it with "
                "AttentionStep(causal=False)"
            )
        for label, segments in self._group_segments_by_label(step.segments or ()).items():
            wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
            wrapper.plan([seg.span for seg in segments])
            plan_states[label] = wrapper
