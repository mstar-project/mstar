"""Multi-axis rotary position embeddings for FLUX-family DiTs.

A token's position is a small integer vector (FLUX.2: ``(t, h, w, l)``,
Z-Image: ``(t, h, w)``); each axis owns a slice of the head dimension
(``axes_dims``) and gets its own 1D rotary table. The tables are derived state
computed from the position ids, never checkpoint state.

Numerics match diffusers' ``get_1d_rotary_pos_embed(..., use_real=True,
repeat_interleave_real=True, freqs_dtype=float64)`` and ``apply_rotary_emb(...,
use_real_unbind_dim=-1)``: float64 frequencies, cos/sin stored fp32 with each
value repeated twice along the head dim, and the interleaved-pair rotation
computed in fp32 and cast back to the activation dtype.
"""

from __future__ import annotations

import torch


class MultiAxisRoPE:
    """``ids [S, A] -> (cos, sin)``, each fp32 ``[S, sum(axes_dims)]``.

    Deliberately not an ``nn.Module``: there is nothing to load and nothing
    that should ride through ``to_empty`` / ``state_dict``.
    """

    def __init__(self, theta: float, axes_dims: tuple[int, ...]):
        if any(d % 2 for d in axes_dims):
            raise ValueError(f"every rotary axis dim must be even, got {axes_dims}")
        self.theta = float(theta)
        self.axes_dims = tuple(int(d) for d in axes_dims)
        self.head_dim = sum(self.axes_dims)

    def __call__(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2 or ids.shape[1] != len(self.axes_dims):
            raise ValueError(f"expected ids of shape [S, {len(self.axes_dims)}], got {tuple(ids.shape)}")
        # float64 on devices that have it (CUDA/CPU); the reference downgrades
        # only on MPS/NPU, which mstar does not serve.
        freqs_dtype = torch.float64
        pos = ids.to(torch.float32)
        cos_out, sin_out = [], []
        for axis, dim in enumerate(self.axes_dims):
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=ids.device) / dim))
            freqs = torch.outer(pos[:, axis], freqs)  # [S, dim/2] float64
            cos_out.append(freqs.cos().repeat_interleave(2, dim=1, output_size=dim).float())
            sin_out.append(freqs.sin().repeat_interleave(2, dim=1, output_size=dim).float())
        return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def apply_rotary_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [B, S, H, D]`` by the fp32 tables ``cos, sin [S, D]`` in
    interleaved-pair form (diffusers ``apply_rotary_emb`` with
    ``use_real_unbind_dim=-1`` and ``sequence_dim=1``)."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
