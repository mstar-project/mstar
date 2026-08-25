"""does `plan_rope`/`apply_rope` of `cache_manager.py`

has ownership of per-(request,label) position counter"""

from dataclasses import dataclass, field

import torch

from mstar.engine.resources.attn.wrappers import (
    rope_apply_qk_inplace,  # noqa: F401  (registers mstar::rope_apply_qk_inplace)
)
from mstar.engine.resources.base import CGSlotKey, EngineResourceInfo, PublishedInfo, Resource
from mstar.engine.resources.kv.manager import KVPlanOutputs, SequenceView
from mstar.engine.resources.position.config import (
    PosBackend,
    PositionConfig,
    PositionSpec,
    PositionStep,
    PosScheme,
)
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, StepContext


@dataclass
class PublishedPositionInfo(PublishedInfo):
    """Where each of a request's streams has reached. label -> next position."""
    counters: dict[str, int] = field(default_factory=dict)

    def update(self, other: "PublishedPositionInfo") -> None:
        for label, position in other.counters.items():
            if position > self.counters.get(label, 0):
                self.counters[label] = position


class PositionManager(Resource):
    # NOTE: not an `AttentionResource`, so no label/layer cursors — this is
    # called once per step, not per layer. Layers reach `apply_qk` with the
    # label off `AttentionCallable.label`; make this one if that stops holding.

    @classmethod
    def build(cls, spec: PositionSpec, info: EngineResourceInfo):
        if spec.config.backend == PosBackend.ROPE:
            return RopeManager(config=spec.config, device=info.device)
        raise ValueError(f"Unknown position backend {spec.config.backend!r}")


class RopeManager(PositionManager):
    """rotary position on top flashinfer in-place kernels"""

    def __init__(self, config: PositionConfig, device: torch.device):
        self._config = config
        self._device = device
        self._kv_cache_name = config.kv_cache

        # rid -> label -> next pos of stream
        self._counters: dict[str, dict[str, int]] = {}

        self._static_pos_ids: dict[CGSlotKey, torch.Tensor] = {}
        # plan label -> ids of this step on device
        # to buffer when capture; to fresh tensor when reager
        self._current_pos_ids: dict[str, torch.Tensor] = {}

        self._preplan_pos_ids: dict[str, torch.Tensor] = {}
        self._preplanned = False

    def depends_on(self):
        return {self._kv_cache_name}

    def _static_buffer(self, key: CGSlotKey, num_tokens: int) -> torch.Tensor:
        """The captured-graph pos_ids buffer for one (bucket, slot, label).

        Built on the first plan for that key rather than up front: which labels
        a walk plans under is the step's to declare, and the first plan for a
        bucket runs during capture, before the graph is recorded, so the buffer
        is just as persistent either way.
        """
        buffer = self._static_pos_ids.get(key)
        if buffer is None:
            buffer = self._static_pos_ids[key] = torch.zeros(
                num_tokens, dtype=torch.long, device=self._device
            )
        return buffer

    def ingest_request(self, rid: str, overrides=None):
        del overrides
        self._counters[rid] = {}

    def remove_request(self, rid: str):
        self._counters.pop(rid, None)

    def publish(self, request_id: str) -> "PublishedPositionInfo | None":
        counters = self._counters.get(request_id)
        if not counters:
            return None
        return PublishedPositionInfo(counters=dict(counters))

    def admit_retrieve(
        self, rid: str, node_name: str, graph_walk: str,
        published: "PublishedPositionInfo | None",
    ) -> AdmitOutcome:
        """Take on the counters a stream was published with.

        A node that receives another node's KV (a CFG branch on its own GPU,
        a decode engine after disaggregated prefill) has never advanced these
        streams itself, so its counters start at zero while the KV it just
        pulled in is many tokens long. Monotonic, like the KV lengths beside
        them: a stale echo of this node's own publish never rewinds it.
        """
        del node_name, graph_walk
        if published is None:
            return ADMIT_OK
        counters = self._counters.setdefault(rid, {})
        for label, position in published.counters.items():
            if position > counters.get(label, 0):
                counters[label] = position
        return ADMIT_OK

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_pos_ids = {}

    def plan(self, step: PositionStep, ctx: StepContext) -> dict[str, torch.Tensor]:
        """submits one position id vector for each plan label

        plan labels and token ordering comes from KV plan output"""
        assert not (self._preplanned and ctx.is_preplan), (
            "position preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # staged a step early against this same step's KV plan
            self._current_pos_ids = self._preplan_pos_ids
            self._preplan_pos_ids = {}
            self._preplanned = False
            return self._current_pos_ids

        assert not ctx.is_preplan or ctx.slot_lease is not None, (
            "preplan requires a cuda graph step: the eager path hands back a "
            "fresh tensor rather than writing a slot's buffer"
        )
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(
            self._kv_cache_name
        )
        assert plan_outputs is not None, (
            f"position manager expected plan result from {self._kv_cache_name}"
        )

        # a preplan leases a different slot, so the buffers it writes are
        # disjoint from the ones the in-flight forward reads
        # a stream forked off another starts at the source's position, so
        # mirror the forks KV just applied before any counter is read below
        self._apply_forks(plan_outputs.pre_forks, ctx.request_ids)

        pos_ids_out = self._preplan_pos_ids if ctx.is_preplan else self._current_pos_ids
        pos_ids_out.clear()
        for plan_label, kv_out in plan_outputs.items():
            pos_ids = self._explicit_pos_ids(step, plan_label, len(plan_outputs))
            if pos_ids is None:
                pos_ids = self._build_pos_ids(kv_out.views)
            pos_ids_out[plan_label] = self._place(pos_ids, plan_label, ctx)
        self._preplanned = ctx.is_preplan
        return pos_ids_out

    def commit(self, step: PositionStep, ctx: StepContext):
        for segment, advance in zip(
            step.segments, self._advances(step), strict=True
        ):
            if advance:
                counters = self._counters.setdefault(segment.request_id, {})
                counters[segment.label] = counters.get(segment.label, 0) + advance

        # post-forks copy the source after this step advanced it, matching
        # where KV applies them
        plan_outputs = ctx.plan_results.get(self._kv_cache_name)
        if plan_outputs is not None:
            self._apply_forks(plan_outputs.post_forks, ctx.request_ids)

    def _apply_forks(
        self, forks: tuple[tuple[str, str], ...], request_ids: tuple[str, ...],
    ) -> None:
        for from_label, to_label in forks:
            for rid in request_ids:
                counters = self._counters.setdefault(rid, {})
                counters[to_label] = counters.get(from_label, 0)

    def _advances(self, step: PositionStep) -> tuple[int, ...]:
        """how far does each segment move the stream counter?

        `step.advance` does override"""
        if step.advance is not None:
            return step.advance
        if self._config.scheme == PosScheme.BLOCK:
            return tuple(
                self._config.block_step if segment.span else 0
                for segment in step.segments
            )
        return tuple(segment.span for segment in step.segments)

    def _explicit_pos_ids(
        self, step: PositionStep, plan_label: str, num_plans: int
    ) -> torch.Tensor | None:
        if step.pos_ids is None:
            return None
        if isinstance(step.pos_ids, torch.Tensor):
            assert num_plans == 1, (
                "PositionStep.pos_ids given as a bare tensor but the step "
                f"plans {num_plans} labels; key them by plan label instead"
            )
            return step.pos_ids
        return step.pos_ids.get(plan_label)

    def _build_pos_ids(self, views: list[SequenceView]) -> torch.Tensor:
        """step positions in KV plan order from stream counters"""
        block = self._config.scheme == PosScheme.BLOCK
        pos_ids: list[int] = []
        for view in views:
            start = self.position(view.request_id, view.label)
            if block:
                pos_ids.extend([start] * view.to_compute)
            else:
                pos_ids.extend(range(start, start + view.to_compute))
        return torch.tensor(pos_ids, dtype=torch.long)

    def _place(
        self, pos_ids: torch.Tensor, plan_label: str, ctx: StepContext
    ) -> torch.Tensor:
        if ctx.slot_lease is None:
            if pos_ids.device != self._device:
                pos_ids = pos_ids.to(self._device, non_blocking=True)
            return pos_ids

        key = CGSlotKey(
            bucket=ctx.slot_lease.bucket, slot=ctx.slot_lease.slot, label=plan_label
        )
        buffer = self._static_buffer(key, ctx.slot_lease.bucket.num_tokens)
        n = pos_ids.shape[0]
        assert n <= buffer.shape[0], (
            f"plan label {plan_label!r} carries {n} tokens but its captured "
            f"bucket holds {buffer.shape[0]}"
        )
        # tail beyond n keeps the previous step's values. padding.
        buffer[:n].copy_(pos_ids, non_blocking=True)
        return buffer



    def position(self, rid: str, label: str) -> int:
        return self._counters.get(rid, {}).get(label, 0)

    def pos_ids(self, label: str) -> torch.Tensor | None:
        return self._current_pos_ids.get(label)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        label: str,
        rotary_dim: int | None = None,
        interleave: bool | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        rope_dtype: torch.dtype | None = None,
        **kwargs,
    ):
        pos_ids = self._current_pos_ids[label]
        config = self._config

        orig_dtype = q.dtype
        rope_dtype = rope_dtype if rope_dtype is not None else config.rope_dtype
        if rope_dtype is not None:
            q, k = q.to(rope_dtype), k.to(rope_dtype)
        elif torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            q, k = q.to(dtype), k.to(dtype)
        elif q.dtype == torch.float32:
            q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)

        llama31_params = {
            key: value for key, value in kwargs.items()
            if key in ("low_freq_factor", "high_freq_factor", "old_context_len")
        } or config.llama31_params

        torch.ops.mstar.rope_apply_qk_inplace(
            q, k, pos_ids[:q.shape[0]],
            rotary_dim if rotary_dim is not None else config.rotary_dim,
            interleave if interleave is not None else config.interleave,
            rope_scale if rope_scale is not None else config.rope_scale,
            rope_theta if rope_theta is not None else config.rope_theta,
            **llama31_params,
        )
        return q.to(orig_dtype), k.to(orig_dtype)


# TODO: mrope / 3D positions (qwen omni, cosmos), learned position embeddings
