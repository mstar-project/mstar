"""Positional embedding resources.

A positional embedder owns position semantics: how position identifiers
are built for a step, how they are applied to queries and keys, and how
far each stream's position counter advances when the step commits (the
plan's ``advance`` field, consumed by the pool's commit). Positions are
read from the KV cache pool and never mutated outside commit.
"""

import torch

from mstar.engine.resources.base import PositionPlan, Segment
from mstar.engine.resources.kv_pool import KVCachePool


class RopeEmbedder:
    """Default 1D rotary scheme: one integer position per token, and the
    position counter advances by each segment's span."""

    def plan(self, segments: list[Segment], pool: KVCachePool) -> PositionPlan:
        """Position identifiers for one step, in segment order, read from
        the pool's counters. The tensor is built on CPU; the caller decides
        placement (a captured path copies it into a static device buffer,
        the eager path moves it to the device)."""
        pos_ids: list[int] = []
        advance: list[int] = []
        for segment in segments:
            start = pool.positions(segment.request_id, segment.label)
            pos_ids.extend(range(start, start + segment.span))
            advance.append(segment.span)
        return PositionPlan(
            pos_ids=torch.tensor(pos_ids, dtype=torch.long),
            advance=tuple(advance),
        )

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_ids: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
        rope_dtype=None,
        **kwargs,
    ):
        """Rotate q and k by ``pos_ids``. Runs on device inside the step's
        execution, so it allocates nothing beyond the dtype casts."""
        orig_dtype = q.dtype

        if rope_dtype is not None:
            q, k = q.to(rope_dtype), k.to(rope_dtype)
        elif torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            q, k = q.to(dtype), k.to(dtype)
        elif q.dtype == torch.float32:
            dtype = torch.bfloat16
            q, k = q.to(dtype), k.to(dtype)

        llama31_params = {
            key: value for key, value in kwargs.items()
            if key in ("low_freq_factor", "high_freq_factor", "old_context_len")
        }

        import flashinfer

        if not llama31_params:
            flashinfer.rope.apply_rope_pos_ids_inplace(
                q, k, pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
            )
        else:
            flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
                q, k, pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
                **llama31_params,
            )
        return q.to(orig_dtype), k.to(orig_dtype)


class BlockRopeEmbedder(RopeEmbedder):
    """Block positional scheme: every token of a segment shares the
    stream's current position, and the counter advances by ``step`` per
    non-empty segment regardless of span. This is the image-block shape,
    where a block of N tokens occupies one position: the positions a step
    consumes differ from the tokens it carries. ``apply`` is inherited
    unchanged; only the identifiers and the advance rule differ."""

    def __init__(self, step: int = 1):
        self.step = step

    def plan(self, segments: list[Segment], pool: KVCachePool) -> PositionPlan:
        pos_ids: list[int] = []
        advance: list[int] = []
        for segment in segments:
            start = pool.positions(segment.request_id, segment.label)
            pos_ids.extend([start] * segment.span)
            advance.append(self.step if segment.span else 0)
        return PositionPlan(
            pos_ids=torch.tensor(pos_ids, dtype=torch.long),
            advance=tuple(advance),
        )
