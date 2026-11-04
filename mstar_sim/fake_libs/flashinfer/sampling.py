"""Stand-ins for FlashInfer's sampling kernels: one token per row (token 0,
the probabilities having no values), priced as a few passes over the rows."""

from __future__ import annotations

import torch

from mstar_sim import clock


def _cost(probs, *args, **kwargs) -> tuple[float, float]:
    # a few passes over the rows (FlashInfer's are sort-free)
    return 10.0 * probs.numel(), 2.0 * probs.numel() * probs.element_size()


def _drawn(probs, indices, return_valid):
    rows = indices.numel() if indices is not None else probs.shape[0]
    dtype = indices.dtype if indices is not None else torch.int32
    samples = torch.zeros(rows, dtype=dtype, device=probs.device)
    return (samples, torch.ones(rows, dtype=torch.bool, device=probs.device)) if return_valid else samples


@clock.kernel_fn("flashinfer.sampling_from_probs", cost=_cost, static=True)
def sampling_from_probs(probs, indices=None, deterministic=True, generator=None, check_nan=False,
                        seed=None, offset=None, return_valid=False):
    return _drawn(probs, indices, return_valid)


@clock.kernel_fn("flashinfer.top_p_sampling_from_probs", cost=_cost, static=True)
def top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    return _drawn(probs, indices, return_valid)


@clock.kernel_fn("flashinfer.top_k_top_p_sampling_from_probs", cost=_cost, static=True)
def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None,
                                    filter_apply_order="top_k_first", deterministic=True,
                                    generator=None, check_nan=False, seed=None, offset=None,
                                    return_valid=False):
    if filter_apply_order not in ("top_k_first", "joint"):
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")
    return _drawn(probs, indices, return_valid)


@clock.kernel_fn("flashinfer.top_k_renorm_probs", cost=_cost, static=True)
def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    return torch.empty_like(probs)


@clock.kernel_fn("flashinfer.top_p_renorm_probs", cost=_cost, static=True)
def top_p_renorm_probs(probs: torch.Tensor, top_p, is_deterministic: bool = False) -> torch.Tensor:
    return torch.empty_like(probs)
