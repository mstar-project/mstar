"""A pool of fixed-size per-request KV slots with sink + window retention.

Unlike the paged ``KVManager`` there is nothing to reserve per step and
nothing to release: a request takes a slot when it first runs and keeps it,
the window is a ring overwritten in place, and a step's layout is a small
integer table computed on the host from positions the step declares.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.kv.bounded.config import BoundedKVConfig, BoundedKVSpec, BoundedKVStep
from mstar.engine.resources.kv.bounded.layout import (
    EMPTY_ROW,
    NUM_READS,
    READ_FROM_SOURCE,
    ROW_HEAD,
    ROW_INTS,
    RowLayout,
    flatten_row,
    row_layout,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, AdmitRuntimeError, AllocationFailed, StepContext
from mstar.utils.h2d import PinnedStager


class BoundedPlan(NamedTuple):
    table: torch.Tensor  # [>= rows, ROW_INTS] int32
    # (slot, layout, source) per row, padding included; what the table holds
    rows: tuple[tuple[int, RowLayout, int], ...]
    # host facts ``attend`` checks its arguments against
    max_retained: int = 0
    max_span: int = 0
    reads_source: bool = False
    reads_row_source: bool = False

    def retained(self) -> torch.Tensor:
        """Each row's retained keys before its step, ``[rows]`` int32 on the device."""
        return self.table[: len(self.rows), ROW_HEAD + 1:ROW_HEAD + 2 * NUM_READS:2].sum(1)



class _Request:
    __slots__ = ("slot", "written")

    def __init__(self, slot: int):
        self.slot = slot
        self.written = 0


class BoundedKVManager(Resource):
    @classmethod
    def build(cls, spec: BoundedKVSpec, info: EngineResourceInfo) -> "BoundedKVManager":
        if info.joint_comm_group is not None and info.joint_comm_group.head_shards > 1:
            raise NotImplementedError("bounded KV does not shard across ranks yet")
        # resolved here, not written back: the config is the model's, shared by its specs
        return cls(spec.config, info.device, info.kv_dtype)

    def __init__(self, config: BoundedKVConfig, device: torch.device, kv_dtype: torch.dtype | None = None):
        """``kv_dtype`` (the engine's) stands in for ``config.dtype=None``."""
        self.config = config
        self._device = torch.device(device)
        c = config
        dtype = c.dtype or kv_dtype or torch.get_default_dtype()
        # layer-major so one layer's slots are a contiguous [slots, rows, H, cap, 2D]
        self._cache = torch.zeros(
            c.num_layers, c.max_slots, c.rows_per_request, c.num_heads, max(c.slot_tokens, 1),
            2 * c.head_dim, dtype=dtype, device=self._device,
        )
        self._requests: dict[str, _Request] = {}
        self._free = list(reversed(range(c.max_slots)))
        self._lock = threading.RLock()
        # rid -> slot the live admit took fresh, for ``rollback_admit``
        self._admit_taken: dict[str, int] = {}

        self._stager = PinnedStager(torch.int32)
        self._cg_max_bs = 0
        self._cg_tables: dict[CGSlotKey, torch.Tensor] = {}
        self._eager_table: torch.Tensor | None = None
        self._current: BoundedPlan | None = None

    # Model-facing

    def layer(self, layer_idx: int) -> torch.Tensor:
        """One layer's slots, ``[max_slots, rows, H, cap, 2D]``."""
        return self._cache[layer_idx]

    @property
    def current(self) -> BoundedPlan:
        if self._current is None:
            raise RuntimeError("bounded KV has no plan for this step; declare a BoundedKVStep")
        return self._current

    def attend(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        source: torch.Tensor | None = None,
        plan: BoundedPlan | None = None,
        rel_bias: torch.Tensor | None = None,
        ieee: bool = False,
        row_source: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Attention for ``q, k, v [B * rows, H, T, D]`` over each row's retained
        keys (``source [sources, rows, H, L, 2D]`` is this layer's source prefixes,
        each row's picked by its ``StreamPosition.source``) and this
        step's own, then this step's keys into the slots. ``[B * rows, T, H, D]``.
        ``plan`` defaults to the step's. ``rel_bias`` is a per-distance score term
        (``kernels.bounded_attention``), for relative-position attention; with a
        per-query window (``BoundedKVConfig.query_window``) it is indexed by distance,
        ``[B * rows, H, T, window + 1]``. Rows whose
        source is -1 read theirs from ``row_source = (storage [slots, rows, H, L, 2D],
        slots [>= B])``, e.g. another resource's per-request entries."""
        plan = plan or self.current
        cache = self._cache[layer_idx]
        self._check_attend(q, k, v, cache, source, plan, rel_bias, row_source)
        if q.device.type == "cpu":
            return self._attend_torch(q, k, v, cache, source, plan, rel_bias, row_source)
        from mstar.engine.resources.kv.bounded.kernels import bounded_attention, bounded_store

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = bounded_attention(q, k, v, cache, source, plan.table, rel_bias, ieee, row_source,
                                self.config.query_window)
        bounded_store(k, v, cache, plan.table, self.config.reverse_step_order)
        return out

    def _check_attend(self, q, k, v, cache, source, plan: BoundedPlan, rel_bias, row_source) -> None:
        if not q.dtype == k.dtype == v.dtype == cache.dtype:
            raise ValueError(f"bounded KV holds {cache.dtype}; q, k, v are {q.dtype}, {k.dtype}, {v.dtype}")
        if plan.reads_source and source is None:
            raise ValueError("a row of this step reads its source prefix, and attend was given no source")
        if plan.reads_row_source and row_source is None:
            raise ValueError("a row of this step reads its row_source entry, and attend was given no row_source")
        t = q.shape[2]
        if plan.max_span > t:
            raise ValueError(f"a row of this step writes {plan.max_span} tokens; attend was given {t}")
        if rel_bias is not None:
            window = self.config.query_window
            need = plan.max_retained + 2 * t - 1 if window is None else window + 1
            if rel_bias.shape[-1] < need:
                raise ValueError(f"rel_bias holds {rel_bias.shape[-1]} distances; this step needs {need}")

    def _attend_torch(
        self, q, k, v, cache, source, plan: BoundedPlan, rel_bias=None, row_source=None,
    ) -> torch.Tensor:
        """The kernels' semantics row by row, from the host layouts (CPU)."""
        n, h, t, d = q.shape
        rows = self.config.rows_per_request
        window = self.config.query_window
        out = q.new_empty(n, t, h, d)
        own = row_source[1].tolist() if row_source is not None else None
        for i in range(n):
            b, c = divmod(i, rows)
            slot, layout, src = plan.rows[b]
            row_src = row_source[0][own[b], c] if src < 0 else (None if source is None else source[src, c])
            keys, values = [], []
            for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
                if count == 0 or (from_source and row_src is None):
                    continue
                kv = (row_src if from_source else cache[slot, c])[:, start:start + count]
                keys.append(kv[..., :d])
                values.append(kv[..., d:])
            total = sum(part.shape[1] for part in keys)
            own_len = t if window is not None or layout.span == 0 else min(layout.span, t)
            keys, values = torch.cat([*keys, k[i][:, :own_len]], 1), torch.cat([*values, v[i][:, :own_len]], 1)
            scores = q[i] @ keys.transpose(-1, -2)
            # query i sits at total + i, key j at j
            dist = (total + torch.arange(t)[:, None] - torch.arange(keys.shape[1])[None, :]).to(q.device)
            if window is None:
                if rel_bias is not None:
                    scores = scores + rel_bias[i].gather(-1, (dist + t - 1).expand(h, -1, -1))
            else:
                if rel_bias is not None:
                    scores = scores + rel_bias[i].gather(-1, dist.clamp(0, window).expand(h, -1, -1))
                scores = scores.masked_fill((dist < 0) | (dist > window), float("-inf"))
            att = torch.softmax(scores * d ** -0.5, dim=-1)
            out[i] = (att @ values).transpose(0, 1)
        for i in range(n):
            b, c = divmod(i, rows)
            slot, layout, _ = plan.rows[b]
            for offset, start, count in layout.writes:
                if count == 0:
                    continue
                if self.config.reverse_step_order:
                    fresh = torch.arange(offset, offset - count, -1, device=k.device)
                else:
                    fresh = torch.arange(offset, offset + count, device=k.device)
                cache[slot, c, :, start:start + count] = torch.cat([k[i][:, fresh], v[i][:, fresh]], dim=-1)
        return out

    # Request lifecycle

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None = None):
        del overrides

    def remove_request(self, rid: str):
        self.reset_request(rid, free=True)

    def reset_request(self, rid: str, free: bool = False):
        """Nothing to zero: a slot's contents are only read where a later
        position says they were written."""
        with self._lock:
            req = self._requests.get(rid)
            if req is None:
                return
            if free:
                del self._requests[rid]
                self._admit_taken.pop(rid, None)
                self._free.append(req.slot)
            else:
                req.written = 0

    # Step lifecycle

    def admit(self, step: BoundedKVStep, ctx: StepContext) -> AdmitOutcome:
        if ctx.capture:
            return ADMIT_OK
        with self._lock:
            taken = self._admit_taken
            for rid in [r for r in taken if ctx.is_real_row(r)]:
                del taken[rid]
            for seg in step.segments or ():
                if ctx.is_padding_row(seg.request_id):
                    continue
                pos = step.positions.get(seg.request_id)
                if pos is None:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"bounded KV step declares no position for {seg.request_id!r}",
                        request_ids=frozenset({seg.request_id})))
                if not self.config.min_source_len <= pos.source_len <= self.config.max_source_len:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"{seg.request_id!r} has a {pos.source_len}-token source; the slots "
                        f"take {self.config.min_source_len} to {self.config.max_source_len}",
                        request_ids=frozenset({seg.request_id})))
                req = self._requests.get(seg.request_id)
                if req is None:
                    if not self._free:
                        return AdmitOutcome(ok=False, reason=AllocationFailed(
                            message=f"bounded KV is full: {self.config.max_slots} slots",
                            pages_short=1, label=seg.label, request_id=seg.request_id))
                    req = self._requests[seg.request_id] = _Request(self._free.pop())
                    taken[seg.request_id] = req.slot
                if pos.written != req.written:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"{seg.request_id!r} declares {pos.written} written tokens; "
                        f"{req.written} were committed", request_ids=frozenset({seg.request_id})))
        return ADMIT_OK

    def rollback_admit(self, step: BoundedKVStep, ctx: StepContext) -> None:
        """Free the slots this step's admit took fresh; a slot a request already
        held stays with it."""
        del step
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            taken = self._admit_taken
            for rid in [r for r in taken if ctx.is_real_row(r)]:
                slot = taken.pop(rid)
                req = self._requests.get(rid)
                if req is not None and req.slot == slot:
                    del self._requests[rid]
                    self._free.append(slot)

    def plan(self, step: BoundedKVStep, ctx: StepContext) -> BoundedPlan:
        c = self.config
        values: list[int] = []
        rows = []
        # a capture's rows are padding, but its replays may retain up to the config's max
        max_retained = c.max_retained if ctx.capture else 0
        max_span, reads_source, reads_row_source = 0, False, False
        for seg in step.segments or ():
            req = None if ctx.capture or ctx.is_padding_row(seg.request_id) else self._requests.get(seg.request_id)
            if req is None:
                # padding rows read and write nothing
                rows.append((0, EMPTY_ROW, 0))
            else:
                pos = step.positions[seg.request_id]
                layout = row_layout(pos.source_len, pos.written, seg.span, c.retention_for(pos.source_len),
                                    c.sink_capacity, c.reverse_step_order)
                rows.append((req.slot, layout, pos.source))
                reads = layout.reads
                max_retained = max(max_retained, sum(count for _, count in reads))
                max_span = max(max_span, seg.span)
                if reads[0][1] or reads[2][1]:      # the source's ranges (READ_FROM_SOURCE)
                    if pos.source < 0:
                        reads_row_source = True
                    else:
                        reads_source = True
            values += flatten_row(*rows[-1])
        table = self._table(ctx, len(rows))
        self._stager.copy_(table, values, pad_value=0)
        self._current = BoundedPlan(table, tuple(rows), max_retained, max_span, reads_source, reads_row_source)
        return self._current

    def commit(self, step: BoundedKVStep, ctx: StepContext) -> None:
        if ctx.capture:
            return
        with self._lock:
            for seg in step.segments or ():
                if seg.span == 0:
                    # writes nothing; skipped so a step that only reads a row
                    # (an outer step whose region writes it) can't undo the write
                    continue
                req = self._requests.get(seg.request_id)
                pos = step.positions.get(seg.request_id)
                if req is not None and pos is not None and not ctx.is_padding_row(seg.request_id):
                    # from the declared position, so committing a step twice is harmless
                    req.written = pos.written + seg.span

    # No ``force_double_buffer``: neither half of its hazard exists here. The host
    # side is a PinnedStager ring, which never rewrites a buffer whose copy is in
    # flight. And with no pre-plan, ``plan`` runs only on the stream that then
    # runs the step, so plan(N+1)'s copy into a table queues behind step N's
    # reads of it. A second slot would only double the captured graphs.
    def _table(self, ctx: StepContext, rows: int) -> torch.Tensor:
        lease = ctx.slot_lease
        if lease is None:
            if self._eager_table is None or self._eager_table.shape[0] < rows:
                self._eager_table = torch.zeros(max(rows, 1), ROW_INTS, dtype=torch.int32, device=self._device)
            return self._eager_table[:max(rows, 1)]
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label="main")
        table = self._cg_tables.get(key)
        if table is None:
            size = max(self._cg_max_bs, lease.bucket.bs, rows)
            table = self._cg_tables[key] = torch.zeros(size, ROW_INTS, dtype=torch.int32, device=self._device)
        return table

    def build_accelerator_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def post_warmup_validate(self):
        if self._requests:
            raise RuntimeError(f"bounded KV slots still held after capture: {sorted(self._requests)}")

    def cleanup(self):
        self._cg_tables.clear()
        self._eager_table = None
        self._current = None
