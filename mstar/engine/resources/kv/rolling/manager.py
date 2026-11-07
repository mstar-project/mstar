"""Rolling sink + window rules over each request's K/V buffer in a request pool.

The K/V themselves live in the request's region of the pool the window depends
on (``RollingKVConfig.pool``), in the buffer named after this resource; the
pool admits the request, places and moves its region, and frees it. The window
checks each step's position against what was committed, shifts the retained
window left behind the sink when a step would overflow it (in ``plan``, on the
step's stream), and gives the model each layer's write range and the slice it
attends.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv.rolling.config import (
    RollingKVConfig,
    RollingKVSpec,
    RollingKVStep,
    RollingPosition,
)
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    StepContext,
)

# Tokens a layer of the staging block the window shifts through when the
# distance it moves is short next to what it keeps (a step that writes few
# tokens into a long window); a per-chunk shift moves by a whole chunk and
# copies in place.
STAGE_TOKENS = 512


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
    # each row's K/V buffer, ``[layers, 2, sink + window, H, D]``, for this step only
    kv: tuple[torch.Tensor, ...]


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
    __slots__ = ("capacity", "written", "compacted_for")

    def __init__(self, capacity: int):
        # tokens a layer the request keeps, its sink plus window
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
        return cls(spec.config, info.device, name=spec.resource_key)

    def __init__(self, config: RollingKVConfig, device: torch.device, name: str = "rolling_kv"):
        self.config = config
        # the resource key, which names the K/V buffer in each request's pool region
        self.name = name
        self._device = torch.device(device)
        c = config
        # K/V move as the widest words a row divides into: a strided copy of 2-byte
        # elements runs at well below the bandwidth of 8-byte ones.
        row_bytes = c.head_dim * c.dtype.itemsize
        words = (torch.int64, torch.int32, torch.int16, torch.uint8)
        self._word = next(t for t in words if row_bytes % t.itemsize == 0)
        # Staging for a shift across a short distance (see ``_move_left``), reserved at
        # build so a step allocates nothing.
        self._stage = torch.empty((c.num_layers, 2, STAGE_TOKENS, c.num_heads, c.head_dim), dtype=c.dtype,
                                  device=self._device).view(self._word)
        self._pool = None
        self._requests: dict[int, _Request] = {}
        # Guards the requests against removal on the main thread while admit and
        # rollback_admit run on the GPU thread; plan, write, window and commit touch only
        # their own step's requests, which the worker doesn't remove while a step holds them.
        self._lock = threading.RLock()
        # requests the live admit started, which ``rollback_admit`` forgets again
        self._admit_started: set[int] = set()
        self._current: RollingPlan | None = None

    def depends_on(self) -> set[str]:
        # the pool admits and plans first: the request's region is placed before the window reads it
        return {self.config.pool}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._pool = resources.get(self.config.pool)
        if self._pool is None:
            raise RuntimeError(f"rolling KV {self.name}: no request pool {self.config.pool!r} is bound")

    def can_capture(self) -> bool:
        # the pool's regions move when it is rearranged; a step that declares the window runs eager
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
        plan = self.current
        if k.shape[0] != len(plan.rows):
            raise ValueError(f"rolling KV step has {len(plan.rows)} rows; write was given {k.shape[0]}")
        for i, (row, kv) in enumerate(zip(plan.rows, plan.kv, strict=True)):
            span = row.write_stop - row.write_start
            kv[layer_idx, 0, row.write_start:row.write_stop] = k[i, :span]
            kv[layer_idx, 1, row.write_start:row.write_stop] = v[i, :span]

    def window(self, layer_idx: int, row: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Row ``row``'s keys and values for this step, ``[1, kept, H, D]`` views
        in stream order: the sink, the retained window, then the step's own."""
        r, kv = self.current.rows[row], self.current.kv[row]
        return kv[layer_idx, 0, :r.kept][None], kv[layer_idx, 1, :r.kept][None]

    # Request lifecycle

    def remove_request(self, rid):
        with self._lock:
            self._requests.pop(rid, None)
            self._admit_started.discard(rid)

    # Step lifecycle

    def admit(self, step: RollingKVStep, ctx: StepContext) -> AdmitOutcome:
        if ctx.capture:
            return ADMIT_OK
        with self._lock:
            started = self._admit_started
            started -= {r for r in started if ctx.is_real_row(r)}
            for seg in step.segments or ():
                rid = seg.request_id
                if ctx.is_padding_row(rid):
                    continue
                pos = step.positions.get(rid)
                if pos is None:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        "rolling KV step declares no position for a request", request_ids=frozenset({rid})))
                if pos.window < seg.span:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"the request keeps a {pos.window}-token window and writes {seg.span} tokens a "
                        "step; the window must hold the step", request_ids=frozenset({rid})))
                capacity = pos.sink + pos.window
                req = self._requests.get(rid)
                if req is None:
                    # the pool admitted the request first, so its region is in place
                    shape, dtype = self.config.buffer(capacity)
                    if not self._pool.has_buffer(rid, self.name, shape, dtype):
                        return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                            f"the request's region in {self.config.pool} has no {self.name} buffer of "
                            f"{capacity} tokens: a model using rolling KV declares it with the request's "
                            f"other buffers (RollingKVConfig.buffer)", request_ids=frozenset({rid})))
                    req = self._requests[rid] = _Request(capacity)
                    started.add(rid)
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
        """Forget the requests this step's admit started; the pool gives back their regions."""
        del step
        if ctx.capture or not self._admit_started:
            return
        with self._lock:
            for rid in [r for r in self._admit_started if ctx.is_real_row(r)]:
                self._requests.pop(rid, None)
                self._admit_started.discard(rid)

    def plan(self, step: RollingKVStep, ctx: StepContext) -> RollingPlan:
        """Each row's write range, and the shift that makes room for it.

        The shift runs here, on the stream the step then runs on (the window has
        no pre-plan), so it is ordered after the previous step's reads of the
        request's K/V and before this step's writes. The pool planned first, so
        each buffer is where this step's admit left the region.
        """
        rows, kvs = [], []
        for seg in step.segments or ():
            req = self._requests.get(seg.request_id)
            if req is None:
                raise RuntimeError("rolling KV holds nothing for a row of this step; was the step admitted?")
            kv = self._pool.buffer(seg.request_id, self.name)
            pos = step.positions[seg.request_id]
            row, shift = row_layout(pos, seg.span)
            if shift is not None and req.compacted_for != pos.written:
                self._move_left(kv, shift.src, shift.dst, shift.length)
            req.compacted_for = pos.written
            rows.append(row._replace(rid=seg.request_id))
            kvs.append(kv)
        self._current = RollingPlan(tuple(rows), tuple(kvs))
        return self._current

    def _move_left(self, kv: torch.Tensor, src: int, dst: int, length: int) -> None:
        """Move tokens ``[src, src + length)`` to ``dst < src`` in every layer of one
        request's K/V. The ranges may overlap: copied left to right in blocks no
        longer than the distance moved, no block reads what an earlier one wrote,
        so no temporary is needed. Across a distance under a quarter of the staging
        block those copies are small enough to be launch-bound, so the move goes
        through the staging block instead, two copies a block. Each copy is one
        strided copy across all layers, keys and values."""
        gap = src - dst
        if length <= 0 or gap <= 0:
            return
        words, stage = kv.view(self._word), self._stage
        block = stage.shape[2]
        if 4 * gap >= min(length, block):
            for i in range(0, length, gap):
                n = min(gap, length - i)
                words[:, :, dst + i:dst + i + n] = words[:, :, src + i:src + i + n]
            return
        for i in range(0, length, block):
            n = min(block, length - i)
            stage[:, :, :n] = words[:, :, src + i:src + i + n]
            words[:, :, dst + i:dst + i + n] = stage[:, :, :n]

    def commit(self, step: RollingKVStep, ctx: StepContext) -> None:
        if ctx.capture:
            return
        for seg in step.segments or ():
            req = self._requests.get(seg.request_id)
            pos = step.positions.get(seg.request_id)
            if req is not None and pos is not None and not ctx.is_padding_row(seg.request_id):
                # from the declared position, so committing a step twice is harmless
                req.written = pos.written + seg.span

    def post_warmup_validate(self):
        if self._requests:
            raise RuntimeError(f"rolling KV still holds state for {len(self._requests)} requests after warmup")

    def cleanup(self):
        self._current = None
