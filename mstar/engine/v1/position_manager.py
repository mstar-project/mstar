"""does `plan_rope`/`apply_rope` of `cache_manager.py`

has ownership of per-(request,label) position counter"""

from dataclasses import dataclass, field
from enum import Enum

import torch

from mstar.engine.resources.base import CGSlotKey, Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceType
from mstar.engine.resources.step import PositionStep, StepContext
from mstar.engine.v1.kv_manager import KVPlanOutput, SequenceView


class PosBackend(Enum):
    ROPE = "rope"


class PosScheme(Enum):
    SEQUENTIAL = "sequential"
    BLOCK = "block"


@dataclass
class PositionConfig:
    kv_cache: str  # name of upstream KV cache
    backend: PosBackend = PosBackend.ROPE
    scheme: PosScheme = PosScheme.SEQUENTIAL
    block_step: int = 1  # relevant to BLOCK `PosScheme` only

    # rope params; all overridable
    rotary_dim: int | None = None
    interleave: bool = False
    rope_scale: float = 1.0
    rope_theta: float = 10000.0
    rope_dtype: torch.dtype | None = None

    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    old_context_len: int | None = None

    @property
    def llama31_params(self) -> dict[str, float]:
        params = dict(
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )
        if any(value is None for value in params.values()):
            return {}
        return params


@dataclass
class PositionSpec(NodeResourceSpec):
    config: PositionConfig

    @property
    def resource_type(self):
        return ResourceType.POSITIONS


class PositionManager(Resource):
    @classmethod
    def build(
        cls, spec: PositionSpec,
        device: torch.device,
        **engine_kwargs
    ):
        if spec.config.backend == PosBackend.ROPE:
            return RopeManager(config=spec.config, device=device)


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

    def plan(self, step: PositionStep, ctx: StepContext) -> dict[str, torch.Tensor]:
        """submits one position id vector for each plan label

        plan labels and token ordering comes from KV plan output"""
        plan_outputs: dict[str, KVPlanOutput] = ctx.plan_results.get(
            self._kv_cache_name
        )
        assert plan_outputs is not None, (
            f"position manager expected plan result from {self._kv_cache_name}"
        )

        self._current_pos_ids.clear()
        for plan_label, kv_out in plan_outputs.items():
            pos_ids = self._explicit_pos_ids(step, plan_label, len(plan_outputs))
            if pos_ids is None:
                pos_ids = self._build_pos_ids(kv_out.views)
            self._current_pos_ids[plan_label] = self._place(pos_ids, plan_label, ctx)
        return self._current_pos_ids

    def commit(self, step: PositionStep, ctx: StepContext):
        del ctx
        for segment, advance in zip(
            step.segments, self._advances(step), strict=True
        ):
            if advance:
                counters = self._counters.setdefault(segment.request_id, {})
                counters[segment.label] = counters.get(segment.label, 0) + advance

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

        rope_kwargs = dict(
            rotary_dim=rotary_dim if rotary_dim is not None else config.rotary_dim,
            interleave=interleave if interleave is not None else config.interleave,
            rope_scale=rope_scale if rope_scale is not None else config.rope_scale,
            rope_theta=rope_theta if rope_theta is not None else config.rope_theta,
        )
        llama31_params = {
            key: value for key, value in kwargs.items()
            if key in ("low_freq_factor", "high_freq_factor", "old_context_len")
        } or config.llama31_params

        import flashinfer

        if not llama31_params:
            flashinfer.rope.apply_rope_pos_ids_inplace(
                q, k, pos_ids[:q.shape[0]], **rope_kwargs
            )
        else:
            flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
                q, k, pos_ids[:q.shape[0]], **rope_kwargs, **llama31_params
            )
        return q.to(orig_dtype), k.to(orig_dtype)


# TODO: mrope / 3D positions (qwen omni, cosmos), learned position embeddings
