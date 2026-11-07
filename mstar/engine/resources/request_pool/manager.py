"""A request pool: one region per request in a block reserved at startup.

A request declares its buffers when it opens (``RequestPoolReqConfig``) and
waits in the engine's admission queue, in arrival order, until the pool has
room for all of them (see ``Resource.ask``). The room is set aside then; the
request's first step takes a region of exactly that size, first fit, and gives
it back when the request is removed. When no free gap is large enough but the
free bytes are, the resident regions slide left to join the gaps.

Storage only: the model, or a resource that depends on the pool, reaches a
request's buffer through the step's plan (``view`` for this step's rows,
``buffer`` for a request in the step a dependent plans). A region moves only
in ``admit``, so a view taken in a step is good for that step and no longer.
"""

from __future__ import annotations

import logging
import threading
from typing import NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.request_pool.config import (
    ALIGN,
    RequestPoolConfig,
    RequestPoolReqConfig,
    RequestPoolSpec,
    RequestPoolStep,
)
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    ASKS,
    AdmitOutcome,
    AdmitRuntimeError,
    StepContext,
)

logger = logging.getLogger(__name__)

# Bytes of the staging block a region moves through when the gap it slides
# across is short next to the region (a rearrange after a small request ends):
# copies no longer than the gap would each be small enough to be launch-bound.
STAGE_BYTES = 64 * 2**20
WORD = torch.int64.itemsize


class RequestPoolPlan(NamedTuple):
    # the step's requests, one per segment, in step order
    rows: tuple[int, ...]


class _Region:
    __slots__ = ("offset", "nbytes", "layout")

    def __init__(self, offset: int, nbytes: int, layout: dict):
        # ``[offset, offset + nbytes)`` of the pool
        self.offset = offset
        self.nbytes = nbytes
        # name -> (byte offset in the region, shape, dtype)
        self.layout = layout


class RequestPool(Resource):
    @classmethod
    def build(cls, spec: RequestPoolSpec, info: EngineResourceInfo) -> RequestPool:
        if info.joint_comm_group is not None and info.joint_comm_group.world_size > 1:
            raise NotImplementedError("the request pool does not shard across ranks yet")
        return cls(spec.config, info.device, name=spec.resource_key)

    def __init__(self, config: RequestPoolConfig, device: torch.device, name: str = "request_pool"):
        self.config = config
        # the resource key, which a deployment's ``resources:`` block names
        self.name = name
        self._device = torch.device(device)
        size = config.pool_bytes // ALIGN * ALIGN
        self._size = size
        # Empty: a buffer is only read once its request has written it. Started on an
        # ALIGN boundary, so every buffer is aligned whatever the allocator hands back.
        raw = torch.empty(size + ALIGN, dtype=torch.uint8, device=self._device)
        lead = -raw.data_ptr() % ALIGN
        self._bytes = raw[lead:lead + size]
        # moved as 8-byte words, which a copy streams at far more of the bandwidth than bytes
        self._words = self._bytes.view(torch.int64)
        self._stage = torch.empty(min(STAGE_BYTES, size) // WORD, dtype=torch.int64, device=self._device)
        # free gaps ``(start, length)`` in bytes, by start, adjacent ones merged
        self._free: list[tuple[int, int]] = [(0, size)] if size else []
        self._regions: dict[int, _Region] = {}
        # Guards the free gaps and the set-aside room, which request intake, removal and
        # the admission calls (ask / reserve / unreserve / grew_since) on the main thread
        # share with admit and rollback_admit on the GPU thread. plan and the views touch
        # only their own step's requests, which the worker doesn't remove while a step
        # holds them, and regions move only in admit, on that same thread.
        self._lock = threading.RLock()
        # `_wants` holds what each request declared when it opened, `_reserved` the bytes
        # set aside for an admitted request whose region isn't placed yet, and `_kept`
        # their sum, which no other request may take.
        self._wants: dict[int, RequestPoolReqConfig] = {}
        self._reserved: dict[int, int] = {}
        self._kept = 0
        # regions the live admit placed, which ``rollback_admit`` gives back
        self._admit_taken: set[int] = set()
        self._current: RequestPoolPlan | None = None

    def can_capture(self) -> bool:
        # regions move when the pool is rearranged, so a step that declares the pool runs eager
        return False

    # Model- and dependent-facing

    @property
    def current(self) -> RequestPoolPlan:
        if self._current is None:
            raise RuntimeError(f"request pool {self.name} has no plan for this step; declare a RequestPoolStep")
        return self._current

    def view(self, name: str, row: int = 0) -> torch.Tensor:
        """Buffer ``name`` of this step's row ``row``: a tensor into the pool, good for this step only."""
        return self.buffer(self.current.rows[row], name)

    def buffer(self, rid, name: str) -> torch.Tensor:
        """Buffer ``name`` of request ``rid``'s region, contiguous, its declared shape and dtype. For
        a resource that depends on the pool, in the step it plans; a region moves only in admit."""
        region = self._regions.get(rid)
        if region is None:
            raise RuntimeError(f"request pool {self.name} holds no region for the request; was its step admitted?")
        entry = region.layout.get(name)
        if entry is None:
            raise KeyError(f"request pool {self.name}: the request declared no buffer {name!r} "
                           f"(it has {sorted(region.layout)})")
        offset, shape, dtype = entry
        start = region.offset + offset
        n = 1
        for d in shape:
            n *= d
        return self._bytes[start:start + n * dtype.itemsize].view(dtype).view(shape)

    def has_buffer(self, rid, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> bool:
        region = self._regions.get(rid)
        entry = region.layout.get(name) if region is not None else None
        return entry is not None and entry[1] == tuple(shape) and entry[2] == dtype

    # Request lifecycle

    def ingest_request(self, rid, overrides: ResourceReqConfig | None = None):
        if isinstance(overrides, RequestPoolReqConfig):
            with self._lock:
                self._wants[rid] = overrides

    def remove_request(self, rid):
        with self._lock:
            self._release(rid)
            self._wants.pop(rid, None)
            self._kept -= self._reserved.pop(rid, 0)

    def num_free_bytes(self) -> int:
        with self._lock:
            return sum(n for _, n in self._free)

    def _room(self) -> int:
        """Free bytes not set aside for an admitted request."""
        return self.num_free_bytes() - self._kept

    # Admission: room for a request's declared buffers, before its first step

    def ask(self, rid, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        with self._lock:
            wants = self._wants.get(rid)
            if wants is None or rid in self._regions or rid in self._reserved:
                return None
        if wants.nbytes() > self._size:
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                f"the request keeps {wants.nbytes()} bytes in the request pool; it holds {self._size} "
                f"(resources: {self.name}: pool_bytes)", request_ids=frozenset({rid})))
        return ASKS

    def reserve(self, rid, node_name: str, graph_walk: str, published: None) -> int | None:
        with self._lock:
            wants, room = self._wants.get(rid), self._room()
            need = wants.nbytes() if wants is not None else 0
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

    # Regions

    def _take(self, nbytes: int) -> int | None:
        """The start of a region of ``nbytes``, first fit; None when no gap holds it."""
        for i, (start, length) in enumerate(self._free):
            if length >= nbytes:
                if length == nbytes:
                    del self._free[i]
                else:
                    self._free[i] = (start + nbytes, length - nbytes)
                return start
        return None

    def _release(self, rid) -> None:
        region = self._regions.pop(rid, None)
        self._admit_taken.discard(rid)
        if region is None:
            return
        gaps = sorted([*self._free, (region.offset, region.nbytes)])
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
        so a region moves only after every step already issued has used it, and
        the next step's plan finds its new offset."""
        cursor = moved = nbytes = 0
        for region in sorted(self._regions.values(), key=lambda r: r.offset):
            if region.offset != cursor:
                self._move_left(region.offset // WORD, cursor // WORD, region.nbytes // WORD)
                region.offset = cursor
                moved, nbytes = moved + 1, nbytes + region.nbytes
            cursor += region.nbytes
        self._free = [(cursor, self._size - cursor)] if cursor < self._size else []
        # rare (a request that fits no gap), and it stalls the step's stream for the copy
        logger.info("request pool %s: rearranged, moving %d regions (%.2f GiB)", self.name, moved, nbytes / 2**30)

    def _move_left(self, src: int, dst: int, length: int) -> None:
        """Move pool words ``[src, src + length)`` to ``dst < src``. The ranges may
        overlap: copied left to right in blocks no longer than the distance moved,
        no block reads what an earlier one wrote. Across a gap short next to the
        region, those blocks would be small enough to be launch-bound, so the move
        goes through the staging block instead, two copies a block."""
        gap = src - dst
        if length <= 0 or gap <= 0:
            return
        pool, stage = self._words, self._stage
        block = stage.numel()
        if 4 * gap >= min(length, block):
            for i in range(0, length, gap):
                n = min(gap, length - i)
                pool[dst + i:dst + i + n] = pool[src + i:src + i + n]
            return
        for i in range(0, length, block):
            n = min(block, length - i)
            stage[:n] = pool[src + i:src + i + n]
            pool[dst + i:dst + i + n] = stage[:n]

    # Step lifecycle

    def admit(self, step: RequestPoolStep, ctx: StepContext) -> AdmitOutcome:
        if ctx.capture:
            return ADMIT_OK
        with self._lock:
            taken = self._admit_taken
            taken -= {r for r in taken if ctx.is_real_row(r)}
            for seg in step.segments or ():
                rid = seg.request_id
                if ctx.is_padding_row(rid) or rid in self._regions:
                    continue
                reserved = self._reserved.get(rid)
                wants = self._wants.get(rid)
                if reserved is None or wants is None:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"no room was set aside in request pool {self.name} for the request: a model using "
                        "the pool declares each request's buffers (RequestPoolReqConfig, from "
                        "get_request_resource_configs)", request_ids=frozenset({rid})))
                # the free bytes always cover what is set aside, so rearranging makes room
                start = self._take(reserved)
                if start is None:
                    self._rearrange()
                    start = self._take(reserved)
                del self._reserved[rid]
                self._kept -= reserved
                self._regions[rid] = _Region(start, reserved, wants.layout())
                taken.add(rid)
        return ADMIT_OK

    def rollback_admit(self, step: RequestPoolStep, ctx: StepContext) -> None:
        """Free the regions this step's admit placed, setting their room aside again
        for the retry; a region a request already held stays with it."""
        del step
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            for rid in [r for r in self._admit_taken if ctx.is_real_row(r)]:
                nbytes = self._regions[rid].nbytes
                self._release(rid)
                self._reserved[rid] = nbytes
                self._kept += nbytes

    def plan(self, step: RequestPoolStep, ctx: StepContext) -> RequestPoolPlan:
        rows = []
        for seg in step.segments or ():
            if seg.request_id not in self._regions:
                raise RuntimeError(f"request pool {self.name} holds nothing for a row of this step; "
                                   "was the step admitted?")
            rows.append(seg.request_id)
        self._current = RequestPoolPlan(tuple(rows))
        return self._current

    def commit(self, step: RequestPoolStep, ctx: StepContext) -> None:
        return

    def post_warmup_validate(self):
        held = len(self._regions) + len(self._wants) + len(self._reserved)
        if held or self._kept:
            raise RuntimeError(f"request pool {self.name} still holds state for {held} requests after warmup")

    def cleanup(self):
        self._current = None
