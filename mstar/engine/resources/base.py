"""Boundary types for engine resources.

A resource (KV cache pool, attention manager, positional embedder) owns one
piece of per-step machinery. Everything passed between resources is one of
the immutable values below, valid for a single step.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.kv_pool import KVCachePool


@dataclass(frozen=True)
class Segment:
    """One step's addition to a request's cache stream.

    A request contributes one segment per label active for it in a step;
    the batch's ordered segment list defines the layout of per-token
    arrays. ``span`` may be 0: a zero-span segment reads its stream
    without extending it (admission reserves nothing, commit is a no-op).
    """
    request_id: str
    label: str
    span: int


@dataclass(frozen=True)
class Reservation:
    """A KV cache pool's answer to admitting one segment: how much of the
    span is already resident, how much must actually be computed, and
    whether residency is still being established asynchronously."""
    resident: int
    to_compute: int
    pending: bool = False


@dataclass(frozen=True)
class SequenceView:
    """What a pool holds for one segment's stream, after admission: the
    storage the page table indexes into (by pool reference), the page table
    itself, and the logical extent it covers. The extent is the positions
    ``[start, start + length)``, not necessarily a prefix from zero."""
    pool: "KVCachePool"
    page_indices: tuple[int, ...]
    start: int
    length: int


@dataclass(frozen=True)
class PositionPlan:
    """A positional embedder's output for one step: position identifiers in
    segment order (in whatever shape the scheme requires), and the amount by
    which each segment's position counter advances at commit."""
    pos_ids: torch.Tensor
    advance: tuple[int, ...]
