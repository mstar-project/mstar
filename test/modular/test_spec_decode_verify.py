"""CPU tests for benchmark/spec_decode: the verify primitives and the speculative loop, on bigram table models.

The fakes assert the loop's bookkeeping as it runs: every draft position the loop extends must pair a target
position with the token that followed it, chain positions must be rolled back before the next extend, and the
target must be fed exactly the token the draft was told would come next.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from benchmark.spec_decode.spec_loop import run_speculative
from benchmark.spec_decode.verify import SamplingParams, accept_draft, sampling_probs

V = 5
H = 3


class TableTarget:
    """Bigram model: next-token logits depend on the last token only; hidden states are a fixed embedding."""

    def __init__(self, logits: torch.Tensor, hidden: torch.Tensor):
        self.logits = logits
        self.hidden = hidden
        self.fed: list[int] = []
        self.expect_next: int | None = None

    def prefill(self, tokens):
        self.fed = list(tokens)
        return self.logits[tokens[-1]].clone(), self.hidden[tokens]

    def step(self, token):
        if self.expect_next is not None:
            assert token == self.expect_next, "the target was fed a token other than the one the draft paired"
            self.expect_next = None
        self.fed.append(token)
        return self.logits[token].clone(), self.hidden[token]


class TableDraft:
    """Bigram drafter keyed by the token paired with each position; checks the cache bookkeeping of the loop."""

    def __init__(self, logits: torch.Tensor, target: TableTarget):
        self.logits = logits
        self.target = target
        self.length = 0
        self.chained = 0
        self.extends = 0

    def reset(self):
        self.length = 0
        self.chained = 0

    def extend(self, hiddens, tokens):
        assert self.chained == 0, "chain positions must be rolled back before extending"
        n = tokens.shape[0]
        assert hiddens.shape == (n, H)
        fed = self.target.fed
        # Position p pairs the target's hidden state at p with token p+1; the last token is still pending.
        assert tokens[:-1].tolist() == fed[self.length + 1:], "tokens do not follow the positions they pair with"
        assert self.length + n == len(fed), "extend must cover exactly the positions the target processed"
        self.target.expect_next = int(tokens[-1])
        self.length += n
        self.extends += 1
        return self.logits[int(tokens[-1])].clone()

    def chain(self, token):
        self.length += 1
        self.chained += 1
        return self.logits[token].clone()

    def rollback(self, n):
        assert n == self.chained, "rollback must drop exactly the chain positions"
        self.length -= n
        self.chained = 0


def make_tables(seed: int, scale: float = 2.0, noise: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(V, V, generator=g) * scale
    draft = target + torch.randn(V, V, generator=g) * noise
    hidden = torch.randn(V, H, generator=g)
    return target, draft, hidden


def greedy_reference(logits: torch.Tensor, prompt: list[int], max_new: int, stop: set[int]) -> list[int]:
    out: list[int] = []
    last = prompt[-1]
    while len(out) < max_new:
        last = int(logits[last].argmax())
        out.append(last)
        if last in stop:
            break
    return out


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_speculation_matches_plain_greedy(k, seed):
    target_logits, draft_logits, hidden = make_tables(seed)
    target = TableTarget(target_logits, hidden)
    draft = TableDraft(draft_logits, target)
    prompt = [1, 2, 3]
    stop = {4}
    generated, stats = run_speculative(target, draft, prompt, k, SamplingParams(), 40, frozenset(stop), None)
    assert generated == greedy_reference(target_logits, prompt, 40, stop)
    assert stats.tokens == len(generated)
    # Each step emits accepted + 1 tokens, except a final step cut short by the stop token or the length cap.
    assert 0 <= stats.steps + stats.accepted_total - (len(generated) - 1) <= 1
    # Greedy decisions have accept_prob 0 or 1, so both estimators agree where a position was reached at all.
    for rate, expected in zip(stats.position_rates(), stats.position_expected_rates(), strict=True):
        assert rate == expected or (math.isnan(rate) and math.isnan(expected))
    assert all(stats.evaluated[j] >= stats.accepted[j] for j in range(k))
    assert all(stats.evaluated[j + 1] <= stats.accepted[j] for j in range(k - 1))
    assert sum(stats.accepted) == stats.accepted_total


def test_perfect_draft_accepts_everything():
    target_logits, _, hidden = make_tables(0)
    target = TableTarget(target_logits, hidden)
    draft = TableDraft(target_logits.clone(), target)
    generated, stats = run_speculative(target, draft, [0, 1], 3, SamplingParams(), 31, frozenset(), None)
    assert generated == greedy_reference(target_logits, [0, 1], 31, set())
    assert stats.accepted == stats.evaluated
    assert stats.position_rates() == [1.0, 1.0, 1.0]
    assert stats.tokens_per_step == pytest.approx(1 + stats.accepted_total / stats.steps)


def test_hopeless_draft_accepts_nothing():
    target_logits, _, hidden = make_tables(1)
    # The draft's argmax is always one token off the target's.
    draft_logits = torch.zeros(V, V)
    for t in range(V):
        draft_logits[t, (int(target_logits[t].argmax()) + 1) % V] = 1.0
    target = TableTarget(target_logits, hidden)
    draft = TableDraft(draft_logits, target)
    generated, stats = run_speculative(target, draft, [2], 4, SamplingParams(), 20, frozenset(), None)
    assert generated == greedy_reference(target_logits, [2], 20, set())
    assert stats.accepted_total == 0
    assert stats.steps == len(generated) - 1
    assert stats.evaluated == [stats.steps, 0, 0, 0]


def test_sampling_probs():
    logits = torch.tensor([3.0, 1.0, 0.5, -2.0, 2.0])
    greedy = sampling_probs(logits, SamplingParams())
    assert greedy.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    plain = sampling_probs(logits, SamplingParams(1.0, 1.0))
    assert torch.allclose(plain, torch.softmax(logits, -1))
    scaled = sampling_probs(logits, SamplingParams(0.7, 1.0))
    assert torch.allclose(scaled, torch.softmax(logits / 0.7, -1))
    # top_p keeps the smallest prefix reaching the mass, the crossing token included: 0.754 then 0.181.
    nucleus = sampling_probs(logits, SamplingParams(0.7, 0.9))
    assert (nucleus > 0).tolist() == [True, False, False, False, True]
    assert nucleus.sum().item() == pytest.approx(1.0)
    assert (nucleus[0] / nucleus[4]).item() == pytest.approx((scaled[0] / scaled[4]).item())
    assert (sampling_probs(logits, SamplingParams(0.7, 0.5)) > 0).tolist() == [True, False, False, False, False]


def test_accept_draft_is_deterministic_for_one_hot():
    p = torch.tensor([0.0, 1.0, 0.0])
    q_same = torch.tensor([0.0, 1.0, 0.0])
    q_other = torch.tensor([0.0, 0.0, 1.0])
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        assert accept_draft(1, q_same, p, g).accepted
        decision = accept_draft(2, q_other, p, g)
        assert not decision.accepted and decision.replacement == 1 and decision.accept_prob == 0.0


def test_accept_draft_preserves_target_distribution():
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.2, 0.2, 0.1])
    q = torch.tensor([0.1, 0.4, 0.3, 0.2])
    n = 20000
    counts = torch.zeros(4)
    accepted = 0
    for _ in range(n):
        d = int(torch.multinomial(q, 1, generator=g))
        decision = accept_draft(d, q, p, g)
        counts[d if decision.accepted else decision.replacement] += 1
        accepted += decision.accepted
    assert (counts / n - p).abs().max() < 0.015
    assert abs(accepted / n - torch.minimum(p, q).sum().item()) < 0.015


def exact_two_token_distribution(logits: torch.Tensor, first: int, params: SamplingParams) -> dict:
    p1 = sampling_probs(logits[first], params)
    return {
        (x1, x2): (p1[x1] * sampling_probs(logits[x1], params)[x2]).item()
        for x1, x2 in itertools.product(range(V), repeat=2)
    }


def total_variation(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in set(a) | set(b))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_speculative_sampling_preserves_target_distribution(k):
    target_logits, draft_logits, hidden = make_tables(3, scale=1.5, noise=1.5)
    params = SamplingParams(temperature=0.8, top_p=0.9)
    prompt = [2]
    exact = exact_two_token_distribution(target_logits, prompt[-1], params)
    # The draft must disagree with the target enough that accepting its tokens blindly would be visible.
    assert total_variation(exact, exact_two_token_distribution(draft_logits, prompt[-1], params)) > 0.15
    runs = 3000
    counts: dict = {}
    for r in range(runs):
        target = TableTarget(target_logits, hidden)
        draft = TableDraft(draft_logits, target)
        g = torch.Generator().manual_seed(r)
        generated, _ = run_speculative(target, draft, prompt, k, params, 2, frozenset(), g)
        assert len(generated) == 2
        counts[tuple(generated)] = counts.get(tuple(generated), 0) + 1
    empirical = {key: n / runs for key, n in counts.items()}
    assert total_variation(empirical, exact) < 0.08


@pytest.mark.parametrize("k", [1, 2, 4])
def test_sampling_run_bookkeeping_with_stop_token(k):
    target_logits, draft_logits, hidden = make_tables(5)
    target = TableTarget(target_logits, hidden)
    draft = TableDraft(draft_logits, target)
    g = torch.Generator().manual_seed(1)
    generated, stats = run_speculative(
        target, draft, [0, 3], k, SamplingParams(0.7, 0.95), 25, frozenset({1}), g
    )
    assert stats.prompts == 1 and stats.tokens == len(generated) <= 25
    if 1 in generated:
        assert generated.index(1) == len(generated) - 1
    assert 0 <= stats.steps + stats.accepted_total - (len(generated) - 1) <= 1
    assert draft.extends >= 1
