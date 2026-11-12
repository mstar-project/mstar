"""does `plan_rope`/`apply_rope` of `cache_manager.py`

has ownership of per-(request,label) position counter"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

from mstar.engine.node_dtype import warn_implicit_downcast
from mstar.engine.resources.base import (
    CachedPrefix,
    CGSlotKey,
    EngineResourceInfo,
    PublishedInfo,
    Resource,
)
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position.config import (
    LongRoPE,
    PosBackend,
    PositionConfig,
    PositionReqConfig,
    PositionSpec,
    PositionStep,
    PosScheme,
)
from mstar.engine.resources.position.rope import (
    rope_apply_qk_inplace,  # noqa: F401  (registers mstar::rope_apply_qk_inplace)
)
from mstar.engine.resources.resident import gather_column
from mstar.engine.resources.resident_hooks import ANY_REPEATS
from mstar.engine.resources.sequence_shard import check_shard_plan
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, StepContext

# a LongRoPE plan label's table ids, keyed beside its positions
_TABLE_IDS = "#longrope"

if TYPE_CHECKING:
    from mstar.engine.resources.resident import (
        CapturedResidentStep,
        ResidentPlanner,
        ResidentWrites,
    )


def check_table_rotary_dim(device: torch.device, rotary_dim: int, interleave: bool, what: str) -> None:
    """Widths FlashInfer's cos/sin-cache kernel rotates correctly. It moves 16 elements
    a thread, so a width that isn't a multiple of 16 reads past each head. With NeoX
    pairing it is also wrong, without an error, at widths of 16 more than a multiple of
    32 for some head sizes (measured on H200: 16 of 32, 16 and 48 of 48 and 80, ...);
    every multiple of 32 passed, and every multiple of 16 passed with GPT-J pairing."""
    if device.type != "cuda":
        return
    align = 16 if interleave else 32
    if rotary_dim % align:
        pairing = "GPT-J" if interleave else "NeoX"
        raise ValueError(f"{what} needs a rotary_dim that is a multiple of {align} on CUDA with "
                         f"{pairing} pairing, not {rotary_dim}")


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
    # a split label's ids are its range's (`apply_qk`)
    supports_split_steps = True

    # NOTE: not an `AttentionResource`, so no label/layer cursors — this is
    # called once per step, not per layer. Layers reach `apply_qk` with the
    # label off `AttentionCallable.label`; make this one if that stops holding.

    @classmethod
    def build(cls, spec: PositionSpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KVConfig
        kv_spec: KVSpec = info.dependency(spec.config.kv_cache)
        if spec.config.backend == PosBackend.ROPE:
            manager = (RopeManager if spec.config.longrope is None else LongRopeManager)(
                config=spec.config,
                device=info.device,
                max_seq_len=kv_spec.config.max_seq_len,
                head_dim=kv_spec.config.head_dim,
                dtype=info.kv_dtype,
            )
            manager._shard_key = spec.shard_key
            return manager
        raise ValueError(f"Unknown position backend {spec.config.backend!r}")

    def fusable_rope_theta(self, head_dim: int) -> float | None:
        """The theta of the RoPE ``apply_qk`` applies, when that is plain
        rotate-half RoPE over the whole ``head_dim`` (not interleaved, no
        scaling, no RoPE dtype), which a layer may apply in a fused kernel of
        its own instead; None otherwise."""
        return None


class RopeManager(PositionManager):
    """rotary position on top flashinfer in-place kernels"""

    # reads the KV plan's rows (request, label, tokens) and its own counters, never a page
    supports_context_parallel = True
    # the sequence shard over its node (`PositionSpec.derive`), set by `build`; and whether it can
    # split (`bind_dependencies`)
    _shard_key: str | None = None
    _can_split = False

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
            check_table_rotary_dim(device, rotary_dim, config.interleave, "rope_table_dtype")
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
        # plan label -> its ids' view over this rank's range, for a label the step splits over SP
        # ranks; swapped with the ids
        self._current_split_pos_ids: dict[str, torch.Tensor] = {}
        self._preplan_split_pos_ids: dict[str, torch.Tensor] = {}
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
        keys = {self._kv_cache_name}
        if self._shard_key is not None:
            keys.add(self._shard_key)
        return keys

    def bind_dependencies(self, resources) -> None:
        if self._shard_key is not None:
            self._can_split = resources[self._shard_key].can_split

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
        # a ceiling (`PositionReqConfig`) reaches only a LongRoPE resource: the
        # conductor refuses it for any other (`validate`)
        del overrides
        self._counters[rid] = {}

    def remove_request(self, rid: str):
        self._counters.pop(rid, None)

    def reset_request(self, rid: str, free: bool=False):
        self._counters[rid].clear()

    supports_drop = True

    def drop(self, rid: str, tokens: int) -> int:
        # Recomputing the KV starts positions from zero, and `apply_cached_prefix` never moves a counter back.
        self.reset_request(rid)
        return 0

    def correct_len(self, rid: str, label: str, delta: int) -> None:
        """Move a stream's position by ``delta``, as ``KVManager.correct_len`` moves its KV
        length after a speculative verify step."""
        counters = self._counters.setdefault(rid, {})
        counters[label] = max(0, counters.get(label, 0) + delta)

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
        self._preplan_split_pos_ids = {}

    def plan(self, step: PositionStep, ctx: StepContext) -> dict[str, torch.Tensor]:
        """submits one position id vector for each plan label

        plan labels and token ordering comes from KV plan output"""
        assert not (self._preplanned and ctx.is_preplan), (
            "position preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if ctx.resident is not None:
            # the replay fills the slot's buffer from the tables itself
            return self._current_pos_ids
        if self._preplanned:
            # staged a step early against this same step's KV plan
            self._promote_preplan()
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
            plan_outputs.pre_forks, self._forking_rids(plan_outputs, ctx),
            undo=self._preplan_fork_undo if ctx.is_preplan else None,
        )

        shard_plan = None
        if self._shard_key is not None:
            shard_plan = check_shard_plan(f"positions over KV {self._kv_cache_name!r}", self._shard_key, ctx)
            if not (self._can_split and shard_plan.split):
                shard_plan = None
        pos_ids_out = self._begin_staging(ctx.is_preplan)
        for plan_label, kv_out in plan_outputs.items():
            pos_ids = self._explicit_pos_ids(step, plan_label, len(plan_outputs))
            if pos_ids is not None:
                pos_ids_out[plan_label] = self._place_explicit(pos_ids, plan_label, lease)
                continue
            rows = kv_out.rows
            if rows is None:
                rows = [(v.request_id, v.label, v.to_compute) for v in kv_out.views]
            pos_ids_out[plan_label] = self._stage_rows(rows, plan_label, lease, ctx.is_preplan)
        if shard_plan is not None:
            split_out = self._preplan_split_pos_ids if ctx.is_preplan else self._current_split_pos_ids
            for plan_label in pos_ids_out:
                r = shard_plan.ranges.get(plan_label)
                if r is None:
                    continue
                ids = self._rotation_ids(plan_label, ctx.is_preplan)
                # a view of a contiguous range, so a captured graph reads the slot's buffer. A packed
                # replay's range covers its bucket's padding too, which the buffer holds
                split_out[plan_label] = (
                    r.local(ids) if r.rows is None else r.local(ids[: r.rows * sum(r.counts)].view(r.rows, -1), 1)
                )
        self._preplanned = ctx.is_preplan
        return pos_ids_out

    def _promote_preplan(self) -> None:
        self._current_pos_ids = self._preplan_pos_ids
        self._current_split_pos_ids = self._preplan_split_pos_ids
        self._preplan_pos_ids = {}
        self._preplan_split_pos_ids = {}
        self._preplanned = False
        # Promotion, not abandonment: the staged forks are kept, so the
        # undo record is dropped rather than replayed.
        self._preplan_fork_undo = []

    def _begin_staging(self, is_preplan: bool) -> dict[str, torch.Tensor]:
        """The (emptied) label -> positions dict this plan fills."""
        out = self._preplan_pos_ids if is_preplan else self._current_pos_ids
        out.clear()
        (self._preplan_split_pos_ids if is_preplan else self._current_split_pos_ids).clear()
        return out

    def _rotation_ids(self, label: str, is_preplan: bool = False) -> torch.Tensor:
        """The ids ``apply_qk`` rotates ``label``'s rows at: their positions."""
        return (self._preplan_pos_ids if is_preplan else self._current_pos_ids)[label]

    def _place_explicit(self, pos_ids: torch.Tensor, plan_label: str, lease) -> torch.Tensor:
        """A step's own positions for ``plan_label`` (``PositionStep.pos_ids``)."""
        return self._place(pos_ids, plan_label, lease)

    def _stage_rows(self, rows, plan_label: str, lease, is_preplan: bool) -> torch.Tensor:
        """``plan_label``'s positions on the device, from the stream counters."""
        del is_preplan
        return self._place(self._build_pos_ids(rows, plan_label, lease), plan_label, lease)

    # Resident planning: a stream's position runs ahead of its stored length by a
    # fixed amount between host-planned steps, so a row keeps that offset

    def max_repeats(self, step: PositionStep, ctx: StepContext) -> int:
        return ANY_REPEATS  # a resident row's position is its length plus a fixed offset

    def resident_refusal(self) -> str | None:
        if self._config.scheme != PosScheme.SEQUENTIAL:
            return f"position scheme {self._config.scheme} doesn't advance with the stream"
        return None

    def bind_resident(self, planner: "ResidentPlanner", key: str) -> None:
        self._resident_store = planner
        self._resident_key = key
        self._resident_delta = planner.table()

    def resident_row(self, rid: int, slot: int, row: list[int], writes: "ResidentWrites") -> None:
        # the KV pool binds ahead of this, its dependent, and sets its length first
        label = writes.streams[self._resident_key].label
        row[self._resident_delta.col] = self.position(rid, label) - writes.kv_len[self._kv_cache_name]

    def resident_gather(self, step: "CapturedResidentStep") -> None:
        key = CGSlotKey(bucket=step.bucket, slot=step.slot, label=self._resident_label(step))
        self._gather_offset(step, self._resident_delta, self._static_buffer(key, step.bucket.num_tokens))

    def _resident_label(self, step: "CapturedResidentStep") -> str:
        own, pool = step.streams[self._resident_key], step.streams[self._kv_cache_name]
        if own.label != pool.label or own.reads or pool.reads:
            # the offset holds only while both advance a token a step, as one stream
            raise ValueError(
                f"positions on stream {own.label!r} over {self._kv_cache_name!r}'s {pool.label!r}: "
                "a resident step's positions advance with the stream their KV cache appends to"
            )
        return own.label

    def _gather_offset(self, step: "CapturedResidentStep", table, out: torch.Tensor) -> None:
        """``out[i]``: row ``i``'s cell of ``table`` plus its stored length."""
        gather_column(
            self._resident_store.store, step.slots, table.stride, table.col,
            out, plus=step.buffers.kv_len[self._kv_cache_name],
        )

    def commit(self, step: PositionStep, ctx: StepContext):
        all_counters, repeats = self._counters, ctx.repeats
        for segment, advance in zip(
            step.segments, self._advances(step), strict=True
        ):
            if advance:
                counters = all_counters.get(segment.request_id)
                if counters is None:
                    counters = all_counters[segment.request_id] = {}
                counters[segment.label] = counters.get(segment.label, 0) + advance * repeats

        # post-forks copy the source after this step advanced it, matching
        # where KV applies them
        plan_outputs = ctx.plan_results.get(self._kv_cache_name)
        if plan_outputs is not None:
            self._apply_forks(plan_outputs.post_forks, self._forking_rids(plan_outputs, ctx))

    @staticmethod
    def _forking_rids(plan_outputs: KVPlanOutputs, ctx: StepContext) -> Sequence[int]:
        """The rows KV forked: a step's forks may cover only some of its rows.
        A chunked prompt forks on some of its chunks only (BAGEL's pre-forks on
        its first), and a later chunk re-forking the target's counter would
        restart its positions."""
        fork_rids = plan_outputs.fork_rids
        if fork_rids is None:
            return ctx.padded_request_ids
        return [rid for rid in ctx.padded_request_ids if rid in fork_rids]

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
        counters = self._counters
        pos_ids: list[int] = []
        append, extend = pos_ids.append, pos_ids.extend
        for rid, label, to_compute in rows:
            labels = counters.get(rid)
            start = labels.get(label, 0) if labels is not None else 0
            if to_compute == 1:
                append(start)  # a decode row
            elif block:
                extend([start] * to_compute)
            else:
                extend(range(start, start + to_compute))
        return self._host_ids(pos_ids, plan_label, lease)

    def _host_ids(self, pos_ids: list[int], plan_label: str, lease) -> torch.Tensor:
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
        # by the worker's handle (AGENTS.md invariant 14): the wire id raises
        return self._counters[rid].get(label, 0)

    def pos_ids(self, label: str) -> torch.Tensor | None:
        return self._current_pos_ids.get(label)

    @property
    def cos_sin_table(self) -> torch.Tensor | None:
        """The ``(max_seq_len, rotary_dim)`` table ``rope_table_dtype`` builds, each row
        ``rotary_dim / 2`` cosines then as many sines, for a kernel that rotates inside a
        larger fusion; None without ``rope_table_dtype``."""
        return self._rope_table

    @property
    def interleave(self) -> bool:
        """Whether RoPE turns adjacent pairs ``(2i, 2i + 1)`` (GPT-J), else the halves (NeoX)."""
        return self._config.interleave

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
        ext_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = (
            torch.arange(
                self._max_seq_len,
                device=self._device,
                dtype=torch.float32,
            )
            / rope_scale
        )
        wavelengths = rope_theta ** (
            torch.arange(0, rotary_dim, 2, device=self._device, dtype=torch.float32) / rotary_dim
        )
        if ext_factors is not None:
            # LongRoPE stretches each wavelength by its factor
            wavelengths = ext_factors * wavelengths
        inv_freq = 1.0 / wavelengths
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
        ``k=None`` rotates ``q`` alone and returns ``(q, None)``. On a label this step splits over
        SP ranks, ``q`` holds this rank's range of it, and so do the planned positions."""
        split = self._current_split_pos_ids.get(label) if self._can_split else None
        if pos_ids is None:
            pos_ids = self._rotation_ids(label) if split is None else split.flatten()
        elif split is not None:
            # refused on what every rank shares: the counts differ by rank, so a refusal on them could
            # stop some ranks before the exchange's collectives and hang the others
            if pos_ids.shape[0] == self._current_pos_ids[label].shape[0]:
                raise ValueError(
                    f"apply_qk on label {label!r}, which this step splits over SP ranks, got the whole label's "
                    f"{pos_ids.shape[0]} positions: pass this rank's range's positions"
                )
            assert pos_ids.shape[0] == q.shape[0], (
                f"apply_qk got {pos_ids.shape[0]} positions for {q.shape[0]} tokens this rank holds of label {label!r}"
            )
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
            warn_implicit_downcast("RoPE", q.dtype, torch.bfloat16)
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


class LongRopeManager(RopeManager):
    """RoPE from `LongRoPE`'s short and long tables (Phi-3, Phi-4-mini), one per request.

    One table on the device: the short table's ``max_seq_len`` rows, then the long
    table's. A request opened with a `PositionReqConfig` past the config's original
    context rotates with the long half for its whole life, so its rows' table ids sit
    ``max_seq_len`` past their positions. The ids are planned beside the positions,
    which `pos_ids` keeps publishing unchanged; the spec names the choice in the
    request's prefix keys (`PositionSpec.prefix_variant`). On resident tables a row
    keeps its table id's offset from the stored length beside its position's, so a
    replay builds both from the row.
    """

    def __init__(
        self,
        config: PositionConfig,
        device: torch.device,
        max_seq_len: int = 32768,
        head_dim: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if device.type == "xpu":
            raise NotImplementedError("LongRoPE is not implemented for XPU")
        if config.llama31_params or config.rope_scale != 1.0:
            raise NotImplementedError("LongRoPE does not combine with rope_scale or Llama 3.1 scaling")
        # the plain manager without a table of its own; this one builds its own below
        super().__init__(dataclasses.replace(config, rope_table_dtype=None), device, max_seq_len, head_dim, dtype)
        self._config = config  # what the prefix fingerprint covers
        self._longrope: LongRoPE = config.longrope
        rotary_dim = config.rotary_dim or head_dim
        if rotary_dim is None:
            raise ValueError("LongRoPE needs PositionConfig.rotary_dim or the KV cache's head_dim")
        check_table_rotary_dim(device, rotary_dim, config.interleave, "LongRoPE")
        if len(self._longrope.short_factor) != rotary_dim // 2:
            raise ValueError(
                f"LongRoPE needs {rotary_dim // 2} factors per set for rotary_dim {rotary_dim}, "
                f"got {len(self._longrope.short_factor)}"
            )
        dtype = config.rope_table_dtype or torch.float32
        tables = [
            self._build_rope_cache(
                rotary_dim, 1.0, config.rope_theta, torch.float32,
                ext_factors=torch.tensor(factors, dtype=torch.float32, device=device),
            ).mul_(self._longrope.attention_scale).to(dtype)
            for factors in (self._longrope.short_factor, self._longrope.long_factor)
        ]
        self._rope_table = torch.cat(tables).float()
        self._rope_table_key = (rotary_dim, config.rope_scale, config.rope_theta)
        # requests on the long half
        self._long_rids: set[str] = set()
        # plan label -> each row's table row, beside its positions
        self._current_table_ids: dict[str, torch.Tensor] = {}
        self._preplan_table_ids: dict[str, torch.Tensor] = {}

    def fusable_rope_theta(self, head_dim: int) -> float | None:
        return None

    @property
    def cos_sin_table(self) -> torch.Tensor | None:
        # indexed by table id, not by position: no fused kernel may read it by position
        return None

    # Resident planning: the table id runs ahead of the position by the request's
    # half of the table, fixed at ingest, so a row keeps that offset too

    def bind_resident(self, planner: "ResidentPlanner", key: str) -> None:
        super().bind_resident(planner, key)
        self._resident_table_delta = planner.table()

    def resident_row(self, rid: int, slot: int, row: list[int], writes: "ResidentWrites") -> None:
        super().resident_row(rid, slot, row, writes)
        half = self._max_seq_len if rid in self._long_rids else 0
        row[self._resident_table_delta.col] = row[self._resident_delta.col] + half

    def resident_gather(self, step: "CapturedResidentStep") -> None:
        # the two rows of the buffer `_stage_rows` plans into, which the capture reads
        width = step.bucket.num_tokens
        key = CGSlotKey(bucket=step.bucket, slot=step.slot, label=f"{self._resident_label(step)}{_TABLE_IDS}")
        positions, table_ids = self._static_buffer(key, 2 * width).view(2, width)
        self._gather_offset(step, self._resident_delta, positions)
        self._gather_offset(step, self._resident_table_delta, table_ids)

    def ingest_request(self, rid: str, overrides: PositionReqConfig | None = None):
        super().ingest_request(rid, overrides)
        self._long_rids.discard(rid)
        if overrides is not None and self._longrope.is_long(overrides.max_position):
            self._long_rids.add(rid)

    def remove_request(self, rid: str):
        super().remove_request(rid)
        self._long_rids.discard(rid)

    def clear_preplan(self):
        super().clear_preplan()
        self._preplan_table_ids = {}

    def _promote_preplan(self) -> None:
        super()._promote_preplan()
        self._current_table_ids = self._preplan_table_ids
        self._preplan_table_ids = {}

    def _rotation_ids(self, label: str, is_preplan: bool = False) -> torch.Tensor:
        # rows into the short or the long half of the table
        return (self._preplan_table_ids if is_preplan else self._current_table_ids)[label]

    def _begin_staging(self, is_preplan: bool) -> dict[str, torch.Tensor]:
        (self._preplan_table_ids if is_preplan else self._current_table_ids).clear()
        return super()._begin_staging(is_preplan)

    def _place_explicit(self, pos_ids: torch.Tensor, plan_label: str, lease) -> torch.Tensor:
        raise ValueError(
            "LongRoPE picks each row's table from its request; a step's explicit "
            "pos_ids don't say which rows are whose"
        )

    def _stage_rows(self, rows, plan_label: str, lease, is_preplan: bool) -> torch.Tensor:
        """Positions and table ids in one pass over the rows, staged as the two rows
        of one buffer per slot. Under a lease each comes back the bucket's width,
        as `RopeManager._place` hands its buffer over."""
        block = self._config.scheme == PosScheme.BLOCK
        counters, long_rids, offset = self._counters, self._long_rids, self._max_seq_len
        pos_ids: list[int] = []
        table_ids: list[int] = []
        pos_append, pos_extend = pos_ids.append, pos_ids.extend
        table_append, table_extend = table_ids.append, table_ids.extend
        for rid, label, to_compute in rows:
            labels = counters.get(rid)
            start = labels.get(label, 0) if labels is not None else 0
            row = start + offset if rid in long_rids else start
            if to_compute == 1:  # a decode row
                pos_append(start)
                table_append(row)
            elif block:
                pos_extend([start] * to_compute)
                table_extend([row] * to_compute)
            else:
                pos_extend(range(start, start + to_compute))
                table_extend(range(row, row + to_compute))
        n = len(pos_ids)
        if lease is None:
            both = torch.tensor([pos_ids, table_ids], dtype=torch.long).to(self._device, non_blocking=True)
        else:
            width = lease.bucket.num_tokens
            assert n <= width, (
                f"plan label {plan_label!r} carries {n} tokens but its captured bucket holds {width}"
            )
            key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=f"{plan_label}{_TABLE_IDS}")
            host = self._pinned_pos_ids_buffer(key, 2 * width).view(2, width)
            staged = host.numpy()
            staged[0, :n] = pos_ids
            staged[1, :n] = table_ids
            # the tail keeps the previous step's ids, in range of the table: padding.
            # A row at a time: a partly filled bucket's [:, :n] isn't contiguous, and
            # torch would stage it through a pageable copy and a device temporary.
            both = self._static_buffer(key, 2 * width).view(2, width)
            both[0, :n].copy_(host[0, :n], non_blocking=True)
            both[1, :n].copy_(host[1, :n], non_blocking=True)
        (self._preplan_table_ids if is_preplan else self._current_table_ids)[plan_label] = both[1]
        return both[0]

    def apply_qk(self, q, k, label=None, *args, pos_ids=None, **kwargs):
        if pos_ids is not None:
            raise ValueError(
                "LongRoPE picks each row's table from its request; explicit pos_ids don't say "
                "which rows are whose"
            )
        return super().apply_qk(q, k, label, *args, **kwargs)


# TODO: mrope / 3D positions (qwen omni, cosmos), learned position embeddings
