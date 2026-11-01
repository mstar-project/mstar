"""A pool of rolling sink + window K/V, one contiguous region per request.

The pool (``pool_tokens`` a layer, keys and values) is reserved at startup. A
request takes a region of exactly its declared sink plus window at its first
step, first fit, and gives it back when it is removed. When no free gap is
large enough but the free tokens are, the resident regions slide left to join
the gaps. A request that doesn't fit is refused with ``AllocationFailed``, which
the worker answers by holding it until a region frees; one larger than the
whole pool fails.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv.rolling.config import RollingKVConfig, RollingKVSpec, RollingKVStep, RollingPosition
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, AdmitRuntimeError, AllocationFailed, StepContext


class RollingRow(NamedTuple):
    rid: int
    # where this step's tokens land in the request's K/V
    write_start: int
    write_stop: int
    # the step's keys are ``[0, kept)``, its own tokens last
    kept: int


class RollingPlan(NamedTuple):
    # one per segment, in step order
    rows: tuple[RollingRow, ...]


class Compaction(NamedTuple):
    """Move ``length`` tokens from ``src`` to ``dst`` in a request's K/V (``src > dst``)."""

    src: int
    dst: int
    length: int


def row_layout(pos: RollingPosition, span: int) -> tuple[RollingRow, Compaction | None]:
    """A step's write range and kept length in K/V that hold ``pos.written``
    tokens of the stream compacted (see ``config``), and the shift that makes
    room for it. Request 0; the caller sets the request."""
    n, sink, window = pos.written, pos.sink, pos.window
    total = n + span
    if total <= sink + window:
        return RollingRow(0, n, total, total), None
    # The new window's history, stream[total - window, n), is where the store put it:
    # at its own index while nothing was evicted, else behind the sink, shifted by
    # what the previous window had moved past.
    first = total - window
    src = first if n <= sink + window else sink + first - (n - window)
    length = window - span
    shift = Compaction(src, sink, length) if src != sink and length > 0 else None
    return RollingRow(0, sink + length, sink + window, sink + window), shift


class _Request:
    __slots__ = ("offset", "capacity", "written", "compacted_for")

    def __init__(self, offset: int, capacity: int):
        # the request's region of the pool, ``[offset, offset + capacity)``
        self.offset = offset
        self.capacity = capacity
        self.written = 0
        # the ``written`` count the K/V were last compacted for, so a step planned
        # twice (or retried after a failed forward) shifts once
        self.compacted_for: int | None = None


class RollingKVManager(Resource):
    @classmethod
    def build(cls, spec: RollingKVSpec, info: EngineResourceInfo) -> RollingKVManager:
        if info.joint_comm_group is not None and info.joint_comm_group.world_size > 1:
            raise NotImplementedError("rolling KV does not shard across ranks yet")
        return cls(spec.config, info.device, info.kv_dtype, name=spec.resource_key)

    def __init__(self, config: RollingKVConfig, device: torch.device, kv_dtype: torch.dtype | None = None,
                 name: str = "rolling_kv"):
        self.config = config
        # the resource key, which a deployment's ``resources:`` block names
        self.name = name
        self._device = torch.device(device)
        c = config
        dtype = c.dtype or kv_dtype or torch.get_default_dtype()
        # ``[layers, 2, pool_tokens, H, D]``: per layer, keys and values token-major, so a
        # region is a plain ``[tokens, H, D]`` slice the attention kernels take as is.
        # Empty: a region is only read up to what its request wrote.
        self._pool = torch.empty((c.num_layers, 2, c.pool_tokens, c.num_heads, c.head_dim), dtype=dtype,
                                 device=self._device)
        # free gaps ``(start, length)``, by start, adjacent ones merged
        self._free: list[tuple[int, int]] = [(0, c.pool_tokens)]
        self._requests: dict[int, _Request] = {}
        self._lock = threading.RLock()
        # rids the live admit allocated for, for ``rollback_admit``
        self._admit_taken: set[int] = set()
        self._current: RollingPlan | None = None

    def can_capture(self) -> bool:
        # A replay's padding rows would need scratch K/V to write into, and regions
        # move when the pool is rearranged; a step that declares this store runs eager.
        return False

    # Model-facing

    @property
    def current(self) -> RollingPlan:
        if self._current is None:
            raise RuntimeError("rolling KV has no plan for this step; declare a RollingKVStep")
        return self._current

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """This step's ``k, v [rows, span, H, D]`` into each row's K/V. A step may
        write several times (a denoise loop rewriting its chunk); the last write
        is what later steps read."""
        rows = self.current.rows
        if k.shape[0] != len(rows):
            raise ValueError(f"rolling KV step has {len(rows)} rows; write was given {k.shape[0]}")
        kv = self._pool[layer_idx]
        for i, row in enumerate(rows):
            at = self._requests[row.rid].offset
            span = row.write_stop - row.write_start
            kv[0, at + row.write_start:at + row.write_stop] = k[i, :span]
            kv[1, at + row.write_start:at + row.write_stop] = v[i, :span]

    def window(self, layer_idx: int, row: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Row ``row``'s keys and values for this step, ``[1, kept, H, D]`` views
        in stream order: the sink, the retained window, then the step's own."""
        r = self.current.rows[row]
        at = self._requests[r.rid].offset
        kv = self._pool[layer_idx]
        return kv[0, at:at + r.kept][None], kv[1, at:at + r.kept][None]

    # Request lifecycle

    def ingest_request(self, rid, overrides: ResourceReqConfig | None = None):
        del overrides

    def remove_request(self, rid):
        self.reset_request(rid, free=True)

    def reset_request(self, rid, free: bool = False):
        """Nothing to zero: K/V are only read up to what their request wrote."""
        with self._lock:
            req = self._requests.get(rid)
            if req is None:
                return
            if free:
                self._release(rid)
            else:
                req.written, req.compacted_for = 0, None

    def num_free_tokens(self) -> int:
        with self._lock:
            return sum(n for _, n in self._free)

    # Pool regions

    def _take(self, capacity: int) -> int | None:
        """The start of a region of ``capacity`` tokens, first fit; None when no gap holds it."""
        for i, (start, length) in enumerate(self._free):
            if length >= capacity:
                if length == capacity:
                    del self._free[i]
                else:
                    self._free[i] = (start + capacity, length - capacity)
                return start
        return None

    def _release(self, rid) -> None:
        req = self._requests.pop(rid, None)
        self._admit_taken.discard(rid)
        if req is None:
            return
        gaps = sorted([*self._free, (req.offset, req.capacity)])
        merged = [gaps[0]]
        for start, length in gaps[1:]:
            last_start, last_length = merged[-1]
            if last_start + last_length == start:
                merged[-1] = (last_start, last_length + length)
            else:
                merged.append((start, length))
        self._free = merged

    def _rearrange(self) -> None:
        """Slide every resident region left, in pool order, so the free gaps join
        into one at the end. Runs inside ``admit``, on the stream the steps run on,
        so a region moves only after every step already issued has read it, and
        the next step's plan takes its new offset."""
        cursor = 0
        for req in sorted(self._requests.values(), key=lambda r: r.offset):
            if req.offset != cursor:
                # only what the request holds: its sink and window so far
                self._move_left(req.offset, cursor, min(req.written, req.capacity))
                req.offset = cursor
            cursor += req.capacity
        self._free = [(cursor, self.config.pool_tokens - cursor)] if cursor < self.config.pool_tokens else []

    # Step lifecycle

    def admit(self, step: RollingKVStep, ctx: StepContext) -> AdmitOutcome:
        if ctx.capture:
            return ADMIT_OK
        with self._lock:
            taken = self._admit_taken
            taken -= {r for r in taken if ctx.is_real_row(r)}
            for seg in step.segments or ():
                rid = seg.request_id
                if ctx.is_padding_row(rid):
                    continue
                pos = step.positions.get(rid)
                if pos is None:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        "rolling KV step declares no position for a request", request_ids=frozenset({rid})))
                capacity = pos.sink + pos.window
                if capacity > self.config.pool_tokens or pos.window < seg.span:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"the request keeps a {pos.sink}-token sink and a {pos.window}-token window, writing "
                        f"{seg.span} a step; the rolling KV pool holds {self.config.pool_tokens} tokens "
                        f"(resources: {self.name}: pool_tokens) and the window must hold the step",
                        request_ids=frozenset({rid})))
                req = self._requests.get(rid)
                if req is None:
                    start = self._take(capacity)
                    if start is None and self.num_free_tokens() >= capacity:
                        self._rearrange()
                        start = self._take(capacity)
                    if start is None:
                        return AdmitOutcome(ok=False, reason=AllocationFailed(
                            message=f"rolling KV pool is full: {capacity} tokens wanted, "
                                    f"{self.num_free_tokens()} of {self.config.pool_tokens} free",
                            pages_short=capacity - self.num_free_tokens(), label=seg.label, request_id=rid))
                    req = self._requests[rid] = _Request(start, capacity)
                    taken.add(rid)
                elif capacity != req.capacity:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"the request declares a {capacity}-token sink and window; its first step "
                        f"declared {req.capacity}", request_ids=frozenset({rid})))
                if pos.written != req.written:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"the request declares {pos.written} written tokens; {req.written} were committed",
                        request_ids=frozenset({rid})))
        return ADMIT_OK

    def rollback_admit(self, step: RollingKVStep, ctx: StepContext) -> None:
        """Free the regions this step's admit took; a region a request already held
        stays with it."""
        del step
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            for rid in [r for r in self._admit_taken if ctx.is_real_row(r)]:
                self._release(rid)

    def plan(self, step: RollingKVStep, ctx: StepContext) -> RollingPlan:
        """Each row's write range, and the shift that makes room for it.

        The shift runs here, on the stream the step then runs on (the store has
        no pre-plan), so it is ordered after the previous step's reads of the
        request's K/V and before this step's writes.
        """
        rows = []
        for seg in step.segments or ():
            req = self._requests.get(seg.request_id)
            if req is None:
                raise RuntimeError("rolling KV holds nothing for a row of this step; was the step admitted?")
            pos = step.positions[seg.request_id]
            row, shift = row_layout(pos, seg.span)
            if shift is not None and req.compacted_for != pos.written:
                self._move_left(req.offset + shift.src, req.offset + shift.dst, shift.length)
            req.compacted_for = pos.written
            rows.append(row._replace(rid=seg.request_id))
        self._current = RollingPlan(tuple(rows))
        return self._current

    def _move_left(self, src: int, dst: int, length: int) -> None:
        """Move pool tokens ``[src, src + length)`` to ``dst < src`` in every layer.
        The ranges may overlap: copied left to right in blocks no longer than the
        distance moved, no block reads what an earlier one wrote, so no temporary
        is needed (half the traffic of a clone)."""
        gap = src - dst
        if length <= 0 or gap <= 0:
            return
        blocks = [(i, min(gap, length - i)) for i in range(0, length, gap)]
        for layer in self._pool:
            for cache in layer:
                for i, n in blocks:
                    cache[dst + i:dst + i + n] = cache[src + i:src + i + n]

    def commit(self, step: RollingKVStep, ctx: StepContext) -> None:
        if ctx.capture:
            return
        with self._lock:
            for seg in step.segments or ():
                req = self._requests.get(seg.request_id)
                pos = step.positions.get(seg.request_id)
                if req is not None and pos is not None and not ctx.is_padding_row(seg.request_id):
                    # from the declared position, so committing a step twice is harmless
                    req.written = pos.written + seg.span

    def post_warmup_validate(self):
        if self._requests:
            raise RuntimeError(f"rolling KV still held for {len(self._requests)} requests after warmup")

    def cleanup(self):
        self._current = None
