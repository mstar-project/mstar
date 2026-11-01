"""A pool of rolling sink + window K/V, one contiguous region per request.

The pool (``pool_tokens`` a layer, keys and values) is reserved at startup. A
request takes a region of exactly its sink plus window at its first step, first
fit, and gives it back when it is removed. When no free gap is large enough but
the free tokens are, the resident regions slide left to join the gaps.

A request that declares its size when it opens (``RollingKVReqConfig``) waits in
the engine's admission queue, in arrival order, until the pool has room for it
(see ``Resource.ask``); the room is set aside then and the region placed at its
first step. One that declares nothing is refused with ``AllocationFailed`` while
it doesn't fit, which the worker answers by retrying it later.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv.rolling.config import (
    RollingKVConfig,
    RollingKVReqConfig,
    RollingKVSpec,
    RollingKVStep,
    RollingPosition,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    ASKS,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    StepContext,
)


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
        # The pool as the widest words a row divides into, for moving regions: a strided
        # copy of 2-byte elements runs at well below the bandwidth of 8-byte ones.
        row_bytes = c.head_dim * self._pool.element_size()
        word = next(t for t in (torch.int64, torch.int32, torch.int16, torch.uint8) if row_bytes % t.itemsize == 0)
        self._words = self._pool.view(word)
        # free gaps ``(start, length)``, by start, adjacent ones merged
        self._free: list[tuple[int, int]] = [(0, c.pool_tokens)]
        self._requests: dict[int, _Request] = {}
        self._lock = threading.RLock()
        # Admission. `_wants` holds the tokens each request declared when it opened,
        # `_reserved` the tokens set aside for an admitted request whose region isn't
        # placed yet, and `_kept` their sum, which no other request may take.
        self._wants: dict[int, int] = {}
        self._reserved: dict[int, int] = {}
        self._kept = 0
        # rid -> the reservation its region took (0 for none), for regions the
        # live admit placed, so ``rollback_admit`` can undo them
        self._admit_taken: dict[int, int] = {}
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
        if isinstance(overrides, RollingKVReqConfig):
            with self._lock:
                self._wants[rid] = overrides.tokens

    def remove_request(self, rid):
        with self._lock:
            self.reset_request(rid, free=True)
            self._wants.pop(rid, None)
            self._kept -= self._reserved.pop(rid, 0)

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

    def _room(self) -> int:
        """Free tokens not set aside for an admitted request."""
        return self.num_free_tokens() - self._kept

    # Admission: room for a request's declared tokens, before its first step

    def ask(self, rid, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        with self._lock:
            tokens = self._wants.get(rid)
            if tokens is None or rid in self._requests or rid in self._reserved:
                return None
        if tokens > self.config.pool_tokens:
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                f"the request keeps {tokens} tokens of rolling K/V a layer; the pool holds "
                f"{self.config.pool_tokens} (resources: {self.name}: pool_tokens)", request_ids=frozenset({rid})))
        return ASKS

    def reserve(self, rid, node_name: str, graph_walk: str, published: None) -> int | None:
        with self._lock:
            need, room = self._wants.get(rid, 0), self._room()
            if need > room:
                return room
            if not need:
                return None
            self._reserved[rid] = need
            self._kept += need
            return None

    def unreserve(self, rid) -> None:
        with self._lock:
            self._kept -= self._reserved.pop(rid, 0)

    def grew_since(self, mark: int) -> bool:
        with self._lock:
            return self._room() > mark

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
        self._admit_taken.pop(rid, None)
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
            for r in [r for r in taken if ctx.is_real_row(r)]:
                del taken[r]
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
                    # `_kept` already counts this request's own reservation
                    reserved = self._reserved.get(rid, 0)
                    if reserved and reserved != capacity:
                        return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                            f"the request declared {reserved} tokens of rolling K/V when it opened; its "
                            f"first step keeps {capacity}", request_ids=frozenset({rid})))
                    room = self._room() + reserved
                    if capacity > room:
                        return AdmitOutcome(ok=False, reason=AllocationFailed(
                            message=f"rolling KV pool is full: {capacity} tokens wanted, {room} of "
                                    f"{self.config.pool_tokens} free and not set aside",
                            pages_short=capacity - room, label=seg.label, request_id=rid))
                    start = self._take(capacity)
                    if start is None:
                        self._rearrange()
                        start = self._take(capacity)
                    req = self._requests[rid] = _Request(start, capacity)
                    if reserved:
                        del self._reserved[rid]
                        self._kept -= reserved
                    taken[rid] = reserved
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
        """Free the regions this step's admit took, giving back the reservation each
        took; a region a request already held stays with it."""
        del step
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            for rid in [r for r in self._admit_taken if ctx.is_real_row(r)]:
                reserved = self._admit_taken.pop(rid)
                self._release(rid)
                if reserved:
                    self._reserved[rid] = reserved
                    self._kept += reserved

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
        is needed (half the traffic of a clone). Each block is one strided copy
        across all layers, keys and values."""
        gap = src - dst
        if length <= 0 or gap <= 0:
            return
        pool = self._words
        for i in range(0, length, gap):
            n = min(gap, length - i)
            pool[:, :, dst + i:dst + i + n] = pool[:, :, src + i:src + i + n]

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
        held = len(self._requests) + len(self._wants) + len(self._reserved)
        if held or self._kept:
            raise RuntimeError(f"rolling KV still holds state for {held} requests after warmup")

    def cleanup(self):
        self._current = None
