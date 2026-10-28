"""does `plan_rope`/`apply_rope` of `cache_manager.py`

has ownership of per-(request,label) position counter"""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.base import (
    CachedPrefix,
    CGSlotKey,
    EngineResourceInfo,
    PublishedInfo,
    Resource,
    RewindKind,
)
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position.config import (
    PosBackend,
    PositionConfig,
    PositionSpec,
    PositionStep,
    PosScheme,
)
from mstar.engine.resources.position.rope import (
    rope_apply_qk_inplace,  # noqa: F401  (registers mstar::rope_apply_qk_inplace)
)
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, StepContext

if TYPE_CHECKING:
    from mstar.engine.resources.draft_verify.config import VerdictBatch


@dataclass
class PublishedPositionInfo(PublishedInfo):
    """Where each of a request's streams has reached. label -> next position."""
    counters: dict[str, int] = field(default_factory=dict)

    def update(self, other: "PublishedPositionInfo") -> None:
        for label, position in other.counters.items():
            if position > self.counters.get(label, 0):
                self.counters[label] = position

    def clone(self) -> "PublishedPositionInfo":
        return PublishedPositionInfo(counters=self.counters.copy())


class PositionManager(Resource):
    # only under a sequential scheme, which the load check enforces
    prefix_skip_safe = True

    # NOTE: not an `AttentionResource`, so no label/layer cursors — this is
    # called once per step, not per layer. Layers reach `apply_qk` with the
    # label off `AttentionCallable.label`; make this one if that stops holding.

    @classmethod
    def build(cls, spec: PositionSpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KVConfig
        kv_spec: KVSpec = info.dependency(spec.config.kv_cache)
        if spec.config.backend == PosBackend.ROPE:
            return RopeManager(
                config=spec.config,
                device=info.device,
                max_seq_len=kv_spec.config.max_seq_len,
                head_dim=kv_spec.config.head_dim,
                dtype=info.kv_dtype,
            )
        raise ValueError(f"Unknown position backend {spec.config.backend!r}")

    def fusable_rope_theta(self, head_dim: int) -> float | None:
        """The theta of the RoPE ``apply_qk`` applies, when that is plain
        rotate-half RoPE over the whole ``head_dim`` (not interleaved, no
        scaling, no RoPE dtype), which a layer may apply in a fused kernel of
        its own instead; None otherwise."""
        return None


class RopeManager(PositionManager):
    """rotary position on top flashinfer in-place kernels"""

    def __init__(
        self,
        config: PositionConfig,
        device: torch.device,
        max_seq_len: int = 32768,
        head_dim: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._config = config
        self._device = device
        self._kv_cache_name = config.kv_cache
        self._max_seq_len = max_seq_len
        self._rope_cache_key: tuple | None = None
        self._rope_cache: torch.Tensor | None = None
        if device.type == "xpu":
            if config.llama31_params:
                raise NotImplementedError(
                    "Llama 3.1 RoPE scaling is not implemented for XPU"
                )
            rotary_dim = config.rotary_dim or head_dim
            if rotary_dim is None:
                raise ValueError(
                    "XPU RoPE needs PositionConfig.rotary_dim or the "
                    "dependent KV cache's head_dim"
                )
            cache_dtype = config.rope_dtype or dtype
            # XPU always rotates from this cache, so it already rounds the
            # table to cache_dtype; another rope_table_dtype has no kernel here.
            if config.rope_table_dtype not in (None, cache_dtype):
                raise NotImplementedError(
                    f"rope_table_dtype={config.rope_table_dtype} is not implemented for XPU, "
                    f"which rotates from a {cache_dtype} cos/sin table (rope_dtype or the KV dtype)"
                )
            self._rope_cache_key = (
                rotary_dim,
                config.rope_scale,
                config.rope_theta,
                cache_dtype,
            )
            self._rope_cache = self._build_rope_cache(
                rotary_dim,
                config.rope_scale,
                config.rope_theta,
                cache_dtype,
            )
        # CUDA rotates from this table instead of in-kernel angles when set.
        self._rope_table: torch.Tensor | None = None
        if config.rope_table_dtype is not None and device.type != "xpu":
            if config.llama31_params:
                raise NotImplementedError("rope_table_dtype does not support Llama 3.1 RoPE scaling")
            rotary_dim = config.rotary_dim or head_dim
            if rotary_dim is None:
                raise ValueError("rope_table_dtype needs PositionConfig.rotary_dim or the KV cache's head_dim")
            if device.type == "cuda" and rotary_dim % 16:
                # FlashInfer's cos/sin-cache kernel moves 16 elements a thread: a narrower
                # rotation reads past each head and returns garbage, without an error
                raise ValueError(
                    f"rope_table_dtype needs a rotary_dim that is a multiple of 16 on CUDA, "
                    f"not {rotary_dim}")
            self._rope_table = self._build_rope_cache(
                rotary_dim,
                config.rope_scale,
                config.rope_theta,
                config.rope_table_dtype,
            ).float()
            self._rope_table_key = (rotary_dim, config.rope_scale, config.rope_theta)

        # rid -> label -> next pos of stream
        self._counters: dict[str, dict[str, int]] = {}

        self._static_pos_ids: dict[CGSlotKey, torch.Tensor] = {}
        # Host side of the same addressing, keyed the same way; see
        # `_pinned_pos_ids_buffer`.
        self._pinned_pos_ids: dict[CGSlotKey, torch.Tensor] = {}
        # plan label -> ids of this step on device
        # to buffer when capture; to fresh tensor when reager
        self._current_pos_ids: dict[str, torch.Tensor] = {}

        self._preplan_pos_ids: dict[str, torch.Tensor] = {}
        self._preplanned = False
        # (rid, to_label, counter) for each pre-fork applied, so
        # `clear_preplan` can put the targets back. Unlike KV's, these forks are
        # staged: a counter is host-only, so copying it early races nothing.
        self._preplan_fork_undo: list[tuple[str, str, int | None]] = []

    def fusable_rope_theta(self, head_dim: int) -> float | None:
        cfg = self._config
        plain = (
            cfg.rope_scale == 1.0 and not cfg.interleave and cfg.rope_dtype is None
            and not cfg.llama31_params and cfg.rotary_dim in (None, head_dim)
        )
        return float(cfg.rope_theta) if plain else None

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

    def fingerprint(self) -> bytes:
        # every field, so a rope parameter nobody thought to list still
        # invalidates what an older setting wrote
        return fingerprint(*(
            getattr(self._config, f.name) for f in fields(self._config)
        ))

    def ingest_request(self, rid: str, overrides=None):
        del overrides
        # setdefault: a worker serving two partitions ingests the rid twice
        self._counters.setdefault(rid, {})

    def remove_request(self, rid: str):
        self._counters.pop(rid, None)

    def reset_request(self, rid: str, free: bool=False):
        self._counters[rid].clear()

    @property
    def rewind_kind(self) -> RewindKind | None:
        # a block scheme moves by a fixed step per block, not per token
        return RewindKind.HOST if self._config.scheme != PosScheme.BLOCK else None

    def rewind(self, verdicts: "VerdictBatch") -> None:
        """Take each row's rejected drafts back off its stream's position, beside the KV
        length: a verify step advanced it by its whole block."""
        for rid, label, rejected in zip(*verdicts, strict=True):
            counters = self._counters.setdefault(rid, {})
            counters[label] = max(0, counters.get(label, 0) - rejected)

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs,
        prefix: CachedPrefix | None,
    ) -> None:
        """Move the stream's counter past what the cache holds of this walk.

        The counter moves here and not at admit. A walk the cache serves never
        admits, and the walk after it can place its positions in `prepare_inputs`,
        before its own admit, so a counter seeded at admit would put that walk on
        top of the served ones. The position is absolute, so a repeated probe
        moves nothing.
        """
        del node_name, graph_walk, inputs
        if prefix is None:
            return
        counters = self._counters.setdefault(rid, {})
        if prefix.position > counters.get(prefix.label, 0):
            counters[prefix.label] = prefix.position

    def publish(self, request_id: str) -> "PublishedPositionInfo | None":
        counters = self._counters.get(request_id)
        if not counters:
            return None
        return PublishedPositionInfo(counters=dict(counters))

    def publish_snapshot_for_step(
        self, request_id: str, node_name: str | None, graph_walk: str | None,
    ):
        del node_name, graph_walk
        counters = self._counters.get(request_id)
        return tuple(counters.items()) if counters else None

    def publish_from_snapshot(
        self, request_id: str, snapshot, node_name: str | None,
        graph_walk: str | None,
    ):
        del request_id, node_name, graph_walk
        return PublishedPositionInfo(counters=dict(snapshot)) if snapshot else None

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

    @property
    def force_double_buffer(self):
        # a slot's pinned host buffer must not be refilled while `_place`'s
        # non-blocking copy out of it may still be queued
        return True

    def clear_preplan(self):
        # `None` means the label had no counter before the staging, which is
        # not the same as having had 0: restoring 0 leaves behind a label the
        # abandoned step invented. KV draws the same line with
        # `_preplan_new_labels`
        for rid, label, counter in reversed(self._preplan_fork_undo):
            counters = self._counters.get(rid)
            if counters is None:
                continue
            if counter is None:
                counters.pop(label, None)
            else:
                counters[label] = counter
        self._preplan_fork_undo = []
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
            # Promotion, not abandonment: the staged forks are kept, so the
            # undo record is dropped rather than replayed.
            self._preplan_fork_undo = []
            return self._current_pos_ids

        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires an accelerator graph step: the eager path hands back a "
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
        self._apply_forks(
            plan_outputs.pre_forks, ctx.padded_request_ids,
            undo=self._preplan_fork_undo if ctx.is_preplan else None,
        )

        pos_ids_out = self._preplan_pos_ids if ctx.is_preplan else self._current_pos_ids
        pos_ids_out.clear()
        for plan_label, kv_out in plan_outputs.items():
            pos_ids = self._explicit_pos_ids(step, plan_label, len(plan_outputs))
            if pos_ids is None:
                rows = kv_out.rows
                if rows is None:
                    rows = [(v.request_id, v.label, v.to_compute) for v in kv_out.views]
                pos_ids = self._build_pos_ids(rows, plan_label, lease)
            pos_ids_out[plan_label] = self._place(pos_ids, plan_label, lease)
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
            self._apply_forks(plan_outputs.post_forks, ctx.padded_request_ids)

    def _apply_forks(
        self, forks: tuple[tuple[str, str], ...], request_ids: tuple[str, ...],
        undo: list | None = None,
    ) -> None:
        for from_label, to_label in forks:
            for rid in request_ids:
                counters = self._counters.setdefault(rid, {})
                if undo is not None:
                    undo.append((rid, to_label, counters.get(to_label)))
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

    def _pinned_pos_ids_buffer(
        self, key: CGSlotKey, num_tokens: int,
    ) -> torch.Tensor:
        """Pinned host staging for one (bucket, slot, label)'s positions.

        Reuse is safe because the runner does not re-lease a slot whose step is
        in flight, so a buffer is refilled only after its async copy ran.
        """
        buffer = self._pinned_pos_ids.get(key)
        if buffer is None:
            buffer = self._pinned_pos_ids[key] = torch.empty(
                num_tokens, dtype=torch.long, pin_memory=True,
            )
        return buffer

    def _build_pos_ids(
        self, rows, plan_label: str, lease,
    ) -> torch.Tensor:
        """step positions in KV plan order from stream counters; ``rows`` is
        ``(request_id, label, to_compute)`` per view in plan order

        Under a lease these go into pinned memory so `_place`'s `non_blocking`
        copy is really asynchronous; from pageable memory the host waits. The
        eager path has no slot to key a buffer by, so allocates plainly.
        """
        block = self._config.scheme == PosScheme.BLOCK
        pos_ids: list[int] = []
        for rid, label, to_compute in rows:
            start = self.position(rid, label)
            if block:
                pos_ids.extend([start] * to_compute)
            else:
                pos_ids.extend(range(start, start + to_compute))
        if lease is None:
            return torch.tensor(pos_ids, dtype=torch.long)
        key = CGSlotKey(
            bucket=lease.bucket, slot=lease.slot, label=plan_label
        )
        buffer = self._pinned_pos_ids_buffer(key, lease.bucket.num_tokens)
        num_tokens = len(pos_ids)
        assert num_tokens <= buffer.shape[0], (
            f"plan label {plan_label!r} carries {num_tokens} tokens but its "
            f"captured bucket holds {buffer.shape[0]}"
        )
        # through the numpy view, to avoid allocating a tensor
        buffer.numpy()[:num_tokens] = pos_ids
        return buffer[:num_tokens]

    def _place(
        self, pos_ids: torch.Tensor, plan_label: str, lease
    ) -> torch.Tensor:
        if lease is None:
            if pos_ids.device != self._device:
                pos_ids = pos_ids.to(self._device, non_blocking=True)
            return pos_ids

        key = CGSlotKey(
            bucket=lease.bucket, slot=lease.slot, label=plan_label
        )
        buffer = self._static_buffer(key, lease.bucket.num_tokens)
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

    def _check_rope_table_args(
        self,
        rotary_dim: int,
        rope_scale: float,
        rope_theta: float,
        llama31_params: dict,
    ) -> None:
        """Refuse per-call RoPE arguments the ``rope_table_dtype`` table
        does not encode: the CUDA kernel reads only the table."""
        runtime = (rotary_dim, rope_scale, rope_theta)
        if runtime != self._rope_table_key:
            raise RuntimeError(
                "RoPE runtime arguments do not match the rope_table_dtype "
                "table built from PositionConfig: "
                f"runtime (rotary_dim, rope_scale, rope_theta)={runtime}, "
                f"configured={self._rope_table_key}"
            )
        # Llama 3.1 scaling at rope_scale 1 leaves every frequency unchanged,
        # which is what Attention passes by default.
        llama31 = all(
            llama31_params.get(key) is not None
            for key in ("low_freq_factor", "high_freq_factor", "old_context_len")
        )
        if llama31 and rope_scale != 1.0:
            raise NotImplementedError(
                "rope_table_dtype does not support Llama 3.1 RoPE scaling"
            )

    def _build_rope_cache(
        self,
        rotary_dim: int,
        rope_scale: float,
        rope_theta: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = (
            torch.arange(
                self._max_seq_len,
                device=self._device,
                dtype=torch.float32,
            )
            / rope_scale
        )
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    device=self._device,
                    dtype=torch.float32,
                )
                / rotary_dim
            )
        )
        freqs = torch.outer(positions, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        label: str | None = None,
        rotary_dim: int | None = None,
        interleave: bool | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        rope_dtype: torch.dtype | None = None,
        pos_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Rotate ``q`` and ``k`` at ``label``'s planned positions, or at ``pos_ids``, one per
        row, for a model that threads positions itself (an offset, or a slice of the step).
        ``k=None`` rotates ``q`` alone and returns ``(q, None)``."""
        if pos_ids is None:
            pos_ids = self._current_pos_ids[label]
        config = self._config
        alone = k is None
        if alone:
            # the kernel rotates a (q, k) pair; a one-head scratch key stands in
            k = q.new_zeros(q.shape[0], 1, q.shape[-1])

        orig_dtype = q.dtype
        rope_dtype = rope_dtype if rope_dtype is not None else config.rope_dtype
        if rope_dtype is not None:
            q, k = q.to(rope_dtype), k.to(rope_dtype)
        elif torch.is_autocast_enabled(q.device.type):
            dtype = torch.get_autocast_dtype(q.device.type)
            q, k = q.to(dtype), k.to(dtype)
        elif q.dtype == torch.float32:
            q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)

        llama31_params = {
            key: value for key, value in kwargs.items()
            if key in ("low_freq_factor", "high_freq_factor", "old_context_len")
        } or config.llama31_params

        rotary_dim = rotary_dim if rotary_dim is not None else config.rotary_dim
        interleave = (
            interleave if interleave is not None else config.interleave
        )
        rope_scale = (
            rope_scale if rope_scale is not None else config.rope_scale
        )
        rope_theta = (
            rope_theta if rope_theta is not None else config.rope_theta
        )

        cos_sin_cache = self._rope_table
        if cos_sin_cache is not None:
            self._check_rope_table_args(
                rotary_dim or q.shape[-1], rope_scale, rope_theta, llama31_params,
            )
        kernel_pos_ids = pos_ids[:q.shape[0]]
        if q.device.type == "xpu":
            rotary_dim = rotary_dim or q.shape[-1]
            cache_key = (rotary_dim, rope_scale, rope_theta, q.dtype)
            if cache_key != self._rope_cache_key:
                raise RuntimeError(
                    "RoPE runtime arguments do not match the cache built "
                    f"from PositionConfig: runtime={cache_key}, "
                    f"configured={self._rope_cache_key}"
                )
            cos_sin_cache = self._rope_cache
            assert cos_sin_cache is not None
            # vllm-xpu-kernels requires int64 position IDs. Normalize only at
            # this backend boundary so position planning and CUDA stay
            # unchanged.
            kernel_pos_ids = kernel_pos_ids.to(
                device=q.device,
                dtype=torch.long,
            )
        torch.ops.mstar.rope_apply_qk_inplace(
            q,
            k,
            kernel_pos_ids,
            cos_sin_cache,
            rotary_dim,
            interleave,
            rope_scale,
            rope_theta,
            **llama31_params,
        )
        return q.to(orig_dtype), None if alone else k.to(orig_dtype)


# TODO: mrope / 3D positions (qwen omni, cosmos), learned position embeddings
