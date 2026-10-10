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
    # (slot, layout) per row, padding included; what the table holds
    rows: tuple[tuple[int, RowLayout], ...]

    def retained(self) -> torch.Tensor:
        """Each row's retained keys before its step, ``[rows]`` int32 on the device."""
        return self.table[: len(self.rows), 2:2 + 2 * NUM_READS:2].sum(1)

    def select(self, rows: list[int]) -> "BoundedPlan":
        """These rows of the plan, for a forward that runs a subset of its batch."""
        index = torch.tensor(rows, dtype=torch.long, device=self.table.device)
        return BoundedPlan(self.table.index_select(0, index), tuple(self.rows[i] for i in rows))


class _Request:
    __slots__ = ("slot", "written")

    def __init__(self, slot: int):
        self.slot = slot
        self.written = 0


class BoundedKVManager(Resource):
    @classmethod
    def build(cls, spec: BoundedKVSpec, info: EngineResourceInfo) -> "BoundedKVManager":
        if info.joint_comm_group is not None and info.joint_comm_group.world_size > 1:
            raise NotImplementedError("bounded KV does not shard across ranks yet")
        return cls(spec.config, info.device)

    def __init__(self, config: BoundedKVConfig, device: torch.device):
        self.config = config
        self._device = torch.device(device)
        c = config
        # layer-major so one layer's slots are a contiguous [slots, rows, H, cap, 2D]
        self._cache = torch.zeros(
            c.num_layers, c.max_slots, c.rows_per_request, c.num_heads, max(c.slot_tokens, 1),
            2 * c.head_dim, dtype=c.dtype, device=self._device,
        )
        self._requests: dict[str, _Request] = {}
        self._free = list(reversed(range(c.max_slots)))
        self._lock = threading.RLock()

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
    ) -> torch.Tensor:
        """Attention for ``q, k, v [B * rows, H, T, D]`` over each row's retained
        keys (``source [rows, H, L, 2D]`` is this layer's source prefix) and this
        step's own, then this step's keys into the slots. ``[B * rows, T, H, D]``.
        ``plan`` defaults to the step's; a forward running a subset of its rows
        passes ``select`` of it. ``rel_bias`` is a per-distance score term
        (``kernels.bounded_attention``), for relative-position attention."""
        plan = plan or self.current
        cache = self._cache[layer_idx]
        if not q.is_cuda:
            return self._attend_torch(q, k, v, cache, source, plan, rel_bias)
        from mstar.engine.resources.kv.bounded.kernels import bounded_attention, bounded_store

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = bounded_attention(q, k, v, cache, source, plan.table, rel_bias, ieee)
        bounded_store(k, v, cache, plan.table, self.config.reverse_step_order)
        return out

    def _attend_torch(self, q, k, v, cache, source, plan: BoundedPlan, rel_bias=None) -> torch.Tensor:
        """The kernels' semantics row by row, from the host layouts (CPU)."""
        n, h, t, d = q.shape
        rows = self.config.rows_per_request
        out = q.new_empty(n, t, h, d)
        for i in range(n):
            b, c = divmod(i, rows)
            slot, layout = plan.rows[b]
            keys, values = [], []
            for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
                if count == 0 or (from_source and source is None):
                    continue
                kv = (source[c] if from_source else cache[slot, c])[:, start:start + count]
                keys.append(kv[..., :d])
                values.append(kv[..., d:])
            keys, values = torch.cat([*keys, k[i]], 1), torch.cat([*values, v[i]], 1)
            scores = q[i] @ keys.transpose(-1, -2)
            if rel_bias is not None:
                total = keys.shape[1] - t
                # query i sits at total + i, key j at j
                rel = total + torch.arange(t)[:, None] - torch.arange(keys.shape[1])[None, :] + t - 1
                scores = scores + rel_bias[i].gather(-1, rel.to(q.device).expand(h, -1, -1))
            att = torch.softmax(scores * d ** -0.5, dim=-1)
            out[i] = (att @ values).transpose(0, 1)
        for i in range(n):
            b, c = divmod(i, rows)
            slot, layout = plan.rows[b]
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
                self._free.append(req.slot)
            else:
                req.written = 0

    # Step lifecycle

    def admit(self, step: BoundedKVStep, ctx: StepContext) -> AdmitOutcome:
        if ctx.capture:
            return ADMIT_OK
        with self._lock:
            for seg in step.segments or ():
                if ctx.is_padding_row(seg.request_id):
                    continue
                pos = step.positions.get(seg.request_id)
                if pos is None:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"bounded KV step declares no position for {seg.request_id!r}"))
                if pos.source_len > self.config.max_source_len:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"{seg.request_id!r} has a {pos.source_len}-token source; the slots "
                        f"hold up to {self.config.max_source_len}"))
                req = self._requests.get(seg.request_id)
                if req is None:
                    if not self._free:
                        return AdmitOutcome(ok=False, reason=AllocationFailed(
                            message=f"bounded KV is full: {self.config.max_slots} slots",
                            pages_short=1, label=seg.label, request_id=seg.request_id))
                    req = self._requests[seg.request_id] = _Request(self._free.pop())
                if pos.written != req.written:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"{seg.request_id!r} declares {pos.written} written tokens; "
                        f"{req.written} were committed"))
        return ADMIT_OK

    def plan(self, step: BoundedKVStep, ctx: StepContext) -> BoundedPlan:
        c = self.config
        values: list[int] = []
        rows = []
        for seg in step.segments or ():
            req = None if ctx.capture or ctx.is_padding_row(seg.request_id) else self._requests.get(seg.request_id)
            if req is None:
                # padding rows read and write nothing
                rows.append((0, EMPTY_ROW))
            else:
                pos = step.positions[seg.request_id]
                rows.append((req.slot, row_layout(pos.source_len, pos.written, seg.span,
                                                  c.retention(pos.source_len), c.sink_capacity,
                                                  c.reverse_step_order)))
            values += flatten_row(*rows[-1])
        table = self._table(ctx, len(rows))
        self._stager.copy_(table, values, pad_value=0)
        self._current = BoundedPlan(table, tuple(rows))
        return self._current

    def commit(self, step: BoundedKVStep, ctx: StepContext) -> None:
        if ctx.capture:
            return
        with self._lock:
            for seg in step.segments or ():
                req = self._requests.get(seg.request_id)
                pos = step.positions.get(seg.request_id)
                if req is not None and pos is not None and not ctx.is_padding_row(seg.request_id):
                    # from the declared position, so committing a step twice is harmless
                    req.written = pos.written + seg.span

    @property
    def force_double_buffer(self):
        return True

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

    def build_cuda_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def post_warmup_validate(self):
        if self._requests:
            raise RuntimeError(f"bounded KV slots still held after capture: {sorted(self._requests)}")

    def cleanup(self):
        self._cg_tables.clear()
        self._eager_table = None
        self._current = None
