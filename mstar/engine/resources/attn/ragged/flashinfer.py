"""Cacheless varlen attention through FlashInfer's ragged prefill wrapper."""

from collections.abc import Iterable

import torch

from mstar.engine.resources.attn.base import WorkspacePool
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.base import RaggedAttnManager
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig
from mstar.engine.resources.attn.ragged.wrappers import RaggedPrefillWrapper
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.step import Segment, SlotLease, StepContext


class FlashInferRaggedManager(RaggedAttnManager):
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
        self._current_plan_states: dict[str, RaggedPrefillWrapper] = {}

        # Persistent, because constructing one allocates FlashInfer's own
        # buffers and is far from free per step.
        self._eager_plan_states: dict[str, RaggedPrefillWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, RaggedPrefillWrapper] = {}

        self._preplan_states: dict[str, RaggedPrefillWrapper] = {}
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

    def _cg_wrapper(
        self, lease: SlotLease, label: str, num_rows: int,
    ) -> RaggedPrefillWrapper:
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
        max_segments = max(self._config.max_segments_for(bucket.bs), num_rows)
        # the bucket's own token count is the ceiling; the per-request override
        # is for a runner that buckets by batch size alone
        max_tokens = max(
            bucket.num_tokens, self._config.max_tokens_for(bucket.bs) or 0
        )
        wrapper = RaggedPrefillWrapper(
            workspace_buffer=self._workspaces.get(label, lease.slot),
            use_cuda_graph=True,
            max_num_segments=max_segments,
            max_total_tokens=max_tokens,
            **self._kwargs,
        )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str) -> RaggedPrefillWrapper:
        """The persistent eager wrapper for one label."""
        wrapper = self._eager_plan_states.get(label)
        if wrapper is None:
            wrapper = self._eager_plan_states[label] = RaggedPrefillWrapper(
                workspace_buffer=self._workspaces.get(label),
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

    # NOTE: can reuse the same AttentionStep dataclass as KV-backed attention
    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()

        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: eager wrappers share one "
            "workspace per label with the forward still in flight"
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
        # cu_seqlens on the CPU: FlashInfer's plan wants it there anyway, and
        # building it on device would sync (fatally so under preplan)
        for label, segments in self._group_segments_by_label(
            step.segments or ()
        ).items():
            cu = [0]
            for seg in segments:
                cu.append(cu[-1] + seg.span)
            cu_seqlens = torch.tensor(cu, dtype=torch.int32)
            if lease is not None:
                wrapper = self._cg_wrapper(lease, label, len(segments))
            else:
                wrapper = self._eager_wrapper(label)

            # TODO: cache the latest plan state
            wrapper.plan(cu_seqlens=cu_seqlens, causal=step.causal)
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality

    def num_segments(self, label: str | None = None) -> int:
        """Real (unpadded) segment count this step planned under ``label``."""
        return self._wrapper(label).num_segments

    def _wrapper(self, label: str | None) -> RaggedPrefillWrapper:
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
        """One layer's varlen self-attention over this step's packed segments.

        Not behind a custom op, unlike the paged backend's ``run``: the ragged
        caller (an encoder tower) has no per-layer KV write to keep in the same
        graph, so the break this costs is one per layer of a region that is
        CUDA-graph captured rather than compiled.
        """
        return self._wrapper(label).run(q, k, v)
