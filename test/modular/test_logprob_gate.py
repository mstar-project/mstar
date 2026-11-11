"""The log-prob parity gate's two checks (`mstar.testing_utils.logprob_gate`), on synthetic per-position errors:
trimmed R catches drift spread over the replies, and the flip cap catches errors confined to a few positions,
which the trim drops."""
from __future__ import annotations

import pytest
import torch

from mstar.testing_utils.logprob_gate import FLIP_SLACK, R_MAX, PromptErrors, ratios, verdict

# three long replies of 128 tokens, as the gate's long set
PROMPTS, TOKENS = 3, 128


def _set(seed: int, spike: float = 0.0, spikes: int = 0, scale: float = 1.0, bf16_flips: int = 0) -> list[PromptErrors]:
    """Per-position errors drawn |N(0, 0.03)|, the served ones ``scale`` times larger, plus ``spike`` nats at
    ``spikes`` positions of the pooled set, each a served greedy flip."""
    gen = torch.Generator().manual_seed(seed)
    srv = torch.randn(PROMPTS * TOKENS, generator=gen).abs() * 0.03 * scale
    bf = torch.randn(PROMPTS * TOKENS, generator=gen).abs() * 0.03
    at = torch.randperm(PROMPTS * TOKENS, generator=gen)[:spikes]
    srv[at] += spike
    flips = torch.zeros(PROMPTS * TOKENS, dtype=torch.long)
    flips[at] = 1
    out = []
    for p in range(PROMPTS):
        rows = slice(p * TOKENS, (p + 1) * TOKENS)
        out.append(PromptErrors(
            served=srv[rows], bf16=bf[rows], served_flips=int(flips[rows].sum()),
            bf16_flips=bf16_flips if p == 0 else 0, deficit=0.0,
        ))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_few_large_localized_errors_pass_trimmed_r_and_fail_the_gate(seed):
    """+2 nats at 8 of 384 positions, each a greedy flip: the trim drops most of them, so R stays near the real
    model's, and the flip cap fails the set."""
    v = verdict(_set(seed, spike=2.0, spikes=8))
    assert v.r_ok and v.r < 1.3, v
    assert ratios(_set(seed, spike=2.0, spikes=8)).mean > R_MAX, "a plain mean would have failed the spikes"
    assert not v.flips_ok and not v.ok, v


@pytest.mark.parametrize("served, bf16", [(0, 2), (5, 7)], ids=["short", "long"])
def test_the_real_models_flip_counts_pass(served, bf16):
    """The H100 run's flip counts on the committed prompts, served against bf16."""
    v = verdict(_set(0, spike=0.0, spikes=served, bf16_flips=bf16))
    assert (v.served_flips, v.bf16_flips) == (served, bf16)
    assert v.ok, v


def test_spread_drift_fails_on_trimmed_r_alone():
    """Errors 2.5 times bf16's at every position, with no flips: R fails, the flip cap doesn't."""
    v = verdict(_set(0, scale=2.5))
    assert not v.r_ok and v.flips_ok and not v.ok, v


def test_the_flip_cap_is_bf16s_count_plus_the_slack():
    assert verdict(_set(0, spikes=3 + FLIP_SLACK, bf16_flips=3)).flips_ok
    assert not verdict(_set(0, spikes=4 + FLIP_SLACK, bf16_flips=3)).flips_ok
