"""Coalesce per-request tensor operations that touch one underlying storage.

A batched forward hands the engine one tensor per request, usually a slice of one batch-wide
result (``new_tokens[i:i+1]`` for 64 requests). Cloning or copying them one by one costs one
kernel launch (and the Python around it) per request per step, which at 64 rows is most of the
worker's CPU tail. These helpers group contiguous tensors by the storage they share and run the
operation once per storage span, then hand back per-tensor views in the original order. Tensors
that share nothing (or are not contiguous) fall back to the per-tensor operation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class _Member:
    index: int
    offset: int  # element offset into the span
    numel: int
    shape: tuple[int, ...]


@dataclass
class StorageSpan:
    """A 1-D view over the part of one storage that a group of tensors covers."""
    view: torch.Tensor
    members: list[_Member]


def storage_spans(tensors: list[torch.Tensor]) -> list[StorageSpan]:
    """Group contiguous tensors by ``(storage, dtype)`` into spans. Each span's ``view`` is a
    1-D tensor over the storage from the first member's start to the last member's end (a
    tensor without companions gets a span of its own)."""
    groups: dict[tuple, list[_Member]] = {}
    for i, t in enumerate(tensors):
        if not t.is_contiguous():
            key = ("alone", i)
        else:
            key = (t.untyped_storage().data_ptr(), t.dtype, t.device)
        groups.setdefault(key, []).append(_Member(i, t.storage_offset(), t.numel(), tuple(t.shape)))
    spans = []
    for key, members in groups.items():
        if key[0] == "alone" or len(members) == 1:
            m = members[0]
            t = tensors[m.index]
            spans.append(StorageSpan(view=t.reshape(-1) if t.is_contiguous() else t.contiguous().reshape(-1),
                                     members=[_Member(m.index, 0, m.numel, m.shape)]))
            continue
        lo = min(m.offset for m in members)
        hi = max(m.offset + m.numel for m in members)
        first = tensors[members[0].index]
        view = torch.empty(0, dtype=first.dtype, device=first.device).set_(first.untyped_storage(), lo, (hi - lo,))
        span_members = [_Member(m.index, m.offset - lo, m.numel, m.shape) for m in members]
        spans.append(StorageSpan(view=view, members=span_members))
    return spans


def apply_coalesced(
    tensors: list[torch.Tensor], span_op: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """``span_op`` maps a span view to a same-sized 1-D result (a clone, a copy to pinned host
    memory, ...); returns per-tensor views of the results, in the input order."""
    out: list[torch.Tensor | None] = [None] * len(tensors)
    for span in storage_spans(tensors):
        res = span_op(span.view)
        members = span.members
        if len(members) > 1 and _tiles(members, res.numel()):
            # the usual case (one row per request, in order): every view from one split call
            for m, piece in zip(members, res.split([m.numel for m in members]), strict=True):
                out[m.index] = piece if len(m.shape) == 1 else piece.view(m.shape)
            continue
        for m in members:
            out[m.index] = res[m.offset:m.offset + m.numel].view(m.shape)
    return out  # type: ignore[return-value]


def _tiles(members: list[_Member], total: int) -> bool:
    """The members cover ``[0, total)`` back to back in list order."""
    end = 0
    for m in members:
        if m.offset != end:
            return False
        end += m.numel
    return end == total


def clone_coalesced(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Clones, one kernel per shared storage instead of one per tensor."""
    return apply_coalesced(tensors, lambda v: v.clone())


__all__ = ["StorageSpan", "apply_coalesced", "clone_coalesced", "storage_spans"]
