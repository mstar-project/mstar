"""Qwen's interleaved 3D MRoPE (Qwen3-VL, Qwen3.5, Qwen3-Omni's Thinker): the 3D
positions of a prompt with images, and the rotary tables they give.

Text advances all three grids together; an image holds T and sweeps H and W over its
merged patch grid, and moves the cursor by ``max(h', w')``. The frequency dims are
interleaved ``[T, H, W, T, H, W, ...]`` (``mrope_section`` dims each, T taking the tail).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.model.submodule_base import ARNodeInputs


def mrope_position_ids(pos_ids: torch.Tensor | None, inputs: list[ARNodeInputs]) -> torch.Tensor:
    """``[3, total_tokens]`` for the step, in packed request order.

    A pure-text step is ``pos_ids``, the position resource's planned 1D
    positions, broadcast across the three grids (a view, no copy). An image's
    grids do not advance together, so a request with `custom_pos_ids` supplies
    its own and the step concatenates, using the slice of ``pos_ids`` for text.
    """
    text_rows = [inp.custom_pos_ids is None for inp in inputs]
    assert pos_ids is not None or not any(text_rows), (
        "position resource has no plan for this step; `plan` must run before `preprocess`"
    )
    if all(text_rows):
        return pos_ids[:sum(inp.input_seq_len for inp in inputs)].unsqueeze(0).expand(3, -1)

    parts: list[torch.Tensor] = []
    offset = 0
    for inp in inputs:
        span = inp.input_seq_len
        if inp.custom_pos_ids is not None:
            parts.append(inp.custom_pos_ids.float())
        else:
            parts.append(pos_ids[offset:offset + span].float().unsqueeze(0).expand(3, -1))
        offset += span
    return torch.cat(parts, dim=1)


def compute_inv_freq(
    rotary_dim: int, rope_theta: float, device: torch.device | None = None,
) -> torch.Tensor:
    """``1 / theta^(2i/d)`` over the rotated dims only; ``[rotary_dim // 2]``."""
    return 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float()
            / rotary_dim
        )
    )


def recompose_frequencies(
    freqs: torch.Tensor, mrope_section: list[int],
) -> torch.Tensor:
    """Weave the three grids together, T as the base and H/W overwriting.

    ``freqs`` is ``[3, seq_len, rotary_dim // 2]``; T keeps dims 0,3,6..., H
    takes 1,4,7..., W 2,5,8... (the ``mrope_interleaved`` layout).
    """
    out = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        idx = slice(offset, mrope_section[dim] * 3, 3)
        out[..., idx] = freqs[dim, ..., idx]
    return out


def compute_3d_cos_sin(
    position_ids_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: list[int],
    target_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape ``[seq_len, rotary_dim]`` from ``[3, seq_len]`` positions."""
    pos = position_ids_3d[:, None, :].float()        # [3, 1, seq_len]
    ifreq = inv_freq[None, :, None].float()          # [1, rotary_dim//2, 1]
    freqs = (ifreq * pos).transpose(1, 2)            # [3, seq_len, rotary_dim//2]
    freqs = recompose_frequencies(freqs, mrope_section)
    # doubled for the rotate-half convention
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if target_dtype is not None:
        cos, sin = cos.to(target_dtype), sin.to(target_dtype)
    return cos, sin


def text_position_ids(
    seq_len: int, start_pos: float = 0, device: torch.device | None = None,
) -> torch.Tensor:
    """``[3, seq_len]`` for a pure-text span: all three grids advance together."""
    pos = torch.arange(seq_len, dtype=torch.float, device=device) + float(start_pos)
    return pos.unsqueeze(0).expand(3, -1).contiguous()


def image_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    start_pos: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``[3, t * h' * w']`` for one image, where ``h' = h // merge``.

    T is constant across a frame while H and W sweep the merged patch grid,
    all based at ``start_pos``; row-major over ``(t, h', w')``, the merger's
    token order.
    """
    t, h, w = (int(v) for v in grid_thw.tolist())
    h //= spatial_merge_size
    w //= spatial_merge_size

    def axis(n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.float, device=device) + start_pos

    grids = torch.meshgrid(axis(t), axis(h), axis(w), indexing="ij")
    return torch.stack(grids, dim=0).reshape(3, -1)


def image_position_advance(
    grid_thw: torch.Tensor, spatial_merge_size: int,
) -> int:
    """How far one image moves the position cursor.

    ``max(h', w')`` positions, not its ``t * h' * w'`` tokens: the grids run
    in parallel.
    """
    _, h, w = (int(v) for v in grid_thw.tolist())
    return max(h, w) // spatial_merge_size
