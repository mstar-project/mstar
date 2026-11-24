"""CPU oracle for the proposed Praxis expert within-step masks.

These layouts describe the mask for a planned request segment. They do not
bind an attention kernel yet. Keeping this oracle separate from the model
lets a resource implementation test its BlockMask against the exact policy.
"""

from __future__ import annotations

import torch

from mstar.engine.state.contract import ExpertCross as ExpertCrossMask
from mstar.engine.state.contract import ExpertCrossLayout, ExpertSelfLayout
from mstar.engine.state.contract import ExpertSelf as ExpertSelfMask


def noised_frame_offset(noised_frames: int, clean_frames: int, chunk_len: int) -> int:
    """The reference's distinction between a warm served chunk and training."""
    if clean_frames == 0:
        return 0
    if noised_frames == 1:
        return clean_frames
    if chunk_len > 1 and noised_frames == chunk_len:
        return clean_frames
    if noised_frames == clean_frames + 1:
        return 0
    if chunk_len > 1 and noised_frames == clean_frames + chunk_len:
        return 0
    raise ValueError(
        f"unsupported expert frame layout: noised={noised_frames}, "
        f"clean={clean_frames}, chunk={chunk_len}"
    )


def query_chunks(num_chunks: int, chunk_len: int, future_per_chunk: int) -> torch.Tensor:
    chunks = torch.arange(num_chunks)
    return torch.cat((
        chunks,
        chunks.repeat_interleave(future_per_chunk),
        chunks.repeat_interleave(chunk_len),
    ))


def cross_visibility(mask: ExpertCrossMask, layout: ExpertCrossLayout) -> torch.Tensor:
    """Dense Q-by-KV mask, for testing the future resource's planned mask."""
    if layout.noised_frames != layout.num_chunks * mask.chunk_len:
        raise ValueError("noised_frames must equal num_chunks * chunk_len")
    offset = noised_frame_offset(layout.noised_frames, layout.clean_frames, mask.chunk_len)
    q_chunk = query_chunks(layout.num_chunks, mask.chunk_len, mask.future_per_chunk)
    q_start = q_chunk * mask.chunk_len + offset
    g, c, n, t = (
        layout.global_len, layout.clean_frames, layout.noised_frames, mask.tokens_per_frame
    )
    kv_len = g + (c + n) * t
    key = torch.arange(kv_len)
    global_key = key < g
    frame = (key - g).clamp(min=0) // t
    clean_key = (key >= g) & (frame < c)
    noised_key = (key >= g + c * t) & (frame < c + n)
    clean = clean_key[None] & (frame[None] < q_start[:, None]) & (
        frame[None] >= q_start[:, None] - mask.window_frames
    )
    visible = global_key[None].expand(q_chunk.numel(), -1) | clean
    if mask.include_noised:
        noised_chunk = (frame - c).clamp(min=0) // mask.chunk_len
        visible |= noised_key[None] & (noised_chunk[None] == q_chunk[:, None])
    return visible


def self_visibility(mask: ExpertSelfMask, layout: ExpertSelfLayout) -> torch.Tensor:
    """Queries in the same chunk see each other, regardless of token kind."""
    q_chunk = query_chunks(layout.num_chunks, mask.chunk_len, mask.future_per_chunk)
    return q_chunk[:, None] == q_chunk[None, :]
