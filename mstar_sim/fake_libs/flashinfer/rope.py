"""Stand-ins for FlashInfer's in-place RoPE kernels."""

from __future__ import annotations

import torch

from mstar_sim import clock


def _cost(q: torch.Tensor, k: torch.Tensor):
    nbytes = 2.0 * (q.numel() * q.element_size() + k.numel() * k.element_size())
    return 3.0 * (q.numel() + k.numel()), nbytes


@clock.kernel_fn("flashinfer.apply_rope_pos_ids_inplace", cost=lambda q, k, *a, **kw: _cost(q, k))
def apply_rope_pos_ids_inplace(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, rotary_dim: int | None = None,
    interleave: bool = False, rope_scale: float = 1, rope_theta: float = 1e4,
) -> None:
    return None


@clock.kernel_fn("flashinfer.apply_rope_pos_ids", cost=lambda q, k, *a, **kw: _cost(q, k))
def apply_rope_pos_ids(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, rotary_dim: int | None = None,
    interleave: bool = False, rope_scale: float = 1, rope_theta: float = 1e4,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


@clock.kernel_fn("flashinfer.apply_llama31_rope_pos_ids_inplace",
                 cost=lambda q, k, *a, **kw: _cost(q, k))
def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, rotary_dim: int | None = None,
    interleave: bool = False, rope_scale: float = 8, rope_theta: float = 5e5,
    low_freq_factor: float = 1, high_freq_factor: float = 4, old_context_len: int = 8192,
) -> None:
    return None


@clock.kernel_fn("flashinfer.apply_rope_with_cos_sin_cache",
                 cost=lambda positions, query, key, *a, **kw: _cost(query, key))
def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int,
    cos_sin_cache: torch.Tensor, is_neox: bool = True,
) -> None:
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")
