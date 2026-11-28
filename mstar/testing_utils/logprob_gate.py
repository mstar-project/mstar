"""The log-prob parity gate of a served model against transformers (test/integration/test_qwen3_real_weights.py).

A prompt set passes two checks:

- **Trimmed R.** ``R = trim(e_srv).mean() / trim(e_bf).mean() <= R_MAX``, each error's positions under its own
  ``TRIM`` quantile. It measures drift spread over the replies, such as a rope base read wrong. The trim keeps one
  greedy flip from carrying the mean.
- **The flip cap.** The served greedy flips (served token != fp32's argmax) are at most the bf16 flips plus
  ``FLIP_SLACK``. Errors confined to a few positions, such as a page-boundary off-by-one, sit past the quantile, so
  the trim drops them; where they flip tokens, they show up here.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

R_MAX = 1.5
# each error's positions past this quantile are left out of its mean
TRIM = 0.98
FLIP_SLACK = 3


class PromptErrors(NamedTuple):
    """One prompt's per-position errors against fp32, at the served tokens."""
    served: torch.Tensor  # e_srv
    bf16: torch.Tensor  # e_bf
    served_flips: int  # served token != fp32's argmax
    bf16_flips: int  # bf16's argmax != fp32's
    deficit: float  # fp32's margin given up at a served flip, the largest


def trimmed_mean(x: torch.Tensor) -> float:
    return float(x[x <= x.quantile(TRIM)].mean())


class Ratios(NamedTuple):
    trimmed: float  # R, the gate's
    mean: float
    p90: float


def ratios(errors: list[PromptErrors]) -> Ratios:
    srv, bf = torch.cat([e.served for e in errors]), torch.cat([e.bf16 for e in errors])
    assert bf.numel() and float(bf.mean()) > 0, "no positions to compare"
    return Ratios(
        trimmed_mean(srv) / trimmed_mean(bf), float(srv.mean() / bf.mean()),
        float(srv.quantile(0.9) / bf.quantile(0.9)),
    )


class Verdict(NamedTuple):
    """One prompt set's two checks."""
    r: float
    served_flips: int
    bf16_flips: int

    @property
    def r_ok(self) -> bool:
        return self.r <= R_MAX

    @property
    def flips_ok(self) -> bool:
        return self.served_flips <= self.bf16_flips + FLIP_SLACK

    @property
    def ok(self) -> bool:
        return self.r_ok and self.flips_ok


def verdict(errors: list[PromptErrors]) -> Verdict:
    """The gate over one prompt set's prompts, pooled."""
    return Verdict(
        ratios(errors).trimmed, sum(e.served_flips for e in errors), sum(e.bf16_flips for e in errors),
    )
