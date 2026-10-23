"""Scale rules: how a format's scales are chosen, within the layout it fixes.

A rule takes ``blocks`` [G, B] (fp32, one row per scale) and the value dtype,
and returns the G scales. Adding one is safe: it cannot change the layout.
"""
from collections.abc import Callable

import torch

from mstar.engine.quant.ops import MIN_SCALE

ScaleRule = Callable[[torch.Tensor, torch.dtype], torch.Tensor]

SCALE_RULES: dict[str, ScaleRule] = {}


def register_scale_rule(name: str):
    def wrap(fn: ScaleRule) -> ScaleRule:
        if name in SCALE_RULES:
            raise ValueError(f"scale rule {name!r} is already registered")
        SCALE_RULES[name] = fn
        return fn
    return wrap


def get_scale_rule(name: str) -> ScaleRule:
    if name not in SCALE_RULES:
        raise ValueError(f"unknown scale rule {name!r}; registered: {sorted(SCALE_RULES)}")
    return SCALE_RULES[name]


def qmax(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max)


def fake_quant(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``blocks`` rounded through ``dtype`` at ``scales`` and back, in fp32."""
    bound = qmax(dtype)
    s = scales.unsqueeze(-1)
    q = (blocks / s).clamp(-bound, bound)
    q = q.to(dtype).float() if dtype.is_floating_point else q.round()
    return q * s


@register_scale_rule("amax")
def amax_rule(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Largest magnitude maps to the largest value: nothing clips."""
    return (blocks.abs().amax(dim=-1) / qmax(dtype)).clamp_min(MIN_SCALE)


# clip ratios searched by `mse`, from no clipping down to half the range
_MSE_RATIOS = torch.linspace(1.0, 0.5, 26).tolist()


@register_scale_rule("mse")
def mse_rule(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per block, the clip ratio of amax whose round trip has the least squared error."""
    base = amax_rule(blocks, dtype)
    best_scale = base.clone()
    best_err = torch.full_like(base, float("inf"))
    for ratio in _MSE_RATIOS:
        scale = (base * ratio).clamp_min(MIN_SCALE)
        err = (fake_quant(blocks, scale, dtype) - blocks).square().sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
    return best_scale
