"""Rejection-sampling verification of a draft block (``SamplerResource.verify`` on a node with
``enable_sampled_verify``), on its own.

What makes it sound: whatever the drafter, the tokens a verify row emits are distributed as
sampling the target one token at a time. A row's outcome is how many drafts it kept, ``a``,
and the token it emitted after them, ``y``. With the target's filtered distribution ``p_j`` and
the drafter's ``q_j`` at each position (a point mass for a drafter that gives tokens only),

    P(a, y) = prod_{j < a} sum_t min(p_j(t), q_j(t)) * r_a(y),

``r_a = max(0, p_a - q_a)`` for ``a < k`` and ``r_k = p_k``: exactly what sampling the target
one token at a time gives up to the first token that leaves the drafts. Many rows, each a
request with its own seed, are checked against it with a chi-square test, for several ``k``
and drafters that are always wrong, partly right, always right, or sample their drafts. The
torch version runs on CPU; FlashInfer's kernel on a GPU.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.verify import verify_stride

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
VOCAB = 8
TEMPERATURE, TOP_K, TOP_P = 0.8, 6, 0.95
# a chi-square p-value under this fails the test; the seeds are fixed, so a pass is stable
P_MIN = 1e-4


def _sampler(device, **spec) -> SamplerResource:
    if device == "cuda":
        pytest.importorskip("flashinfer")
    return SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=torch.device(device),
        draft_verify=True, enable_sampled_verify=True, **spec,
    )


def _requests(sampler, n, first_seed=1000, **config) -> list[int]:
    config = {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P, **config}
    for rid in range(n):
        sampler.ingest_request(rid, SamplingReqConfig(**config, _seed=first_seed + rid))
    return list(range(n))


def _filtered(logits, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P) -> torch.Tensor:
    """The target as the sampler draws from it, written out on its own: softmax at the
    temperature, the top k, then the smallest top set whose mass above each kept token is under
    top_p, renormalised (FlashInfer's rule, which main's sampler draws by)."""
    probs = torch.softmax(logits.double() / temperature, -1)
    out = torch.zeros_like(probs)
    for row in range(probs.shape[0]):
        ranked, order = probs[row].sort(descending=True)
        ranked = ranked[:top_k] / ranked[:top_k].sum()
        keep = (ranked.cumsum(0) - ranked) < top_p
        out[row, order[:top_k][keep]] = ranked[keep]
    return out / out.sum(-1, keepdim=True)


def _expected(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``[k + 1, V]`` outcome probabilities P(a, y) of one row, from the target ``p``
    ``[k + 1, V]`` and the drafter ``q`` ``[k, V]``."""
    k = q.shape[0]
    reach = torch.cat([torch.ones(1, dtype=p.dtype), torch.minimum(p[:k], q).sum(-1).cumprod(0)])
    residual = torch.cat([(p[:k] - q).clamp_min(0), p[k:]])
    return reach[:, None] * residual


def _chi2(observed: torch.Tensor, expected: torch.Tensor) -> float:
    """The p-value of ``observed`` counts against ``expected`` ones (any shape, same total),
    cells expecting fewer than 5 pooled. An observation in a cell that can't happen fails."""
    observed, expected = observed.double().flatten(), expected.double().flatten()
    impossible = expected < 1e-9
    assert observed[impossible].sum() == 0, "a verify row emitted an outcome the target can't produce"
    small = (expected < 5) & ~impossible
    obs = torch.cat([observed[~small & ~impossible], observed[small].sum()[None]])
    exp = torch.cat([expected[~small & ~impossible], expected[small].sum()[None]])
    if exp[-1] < 1e-9:
        obs, exp = obs[:-1], exp[:-1]
    stat = ((obs - exp) ** 2 / exp).sum()
    df = len(obs) - 1
    if df == 0:
        return 1.0  # one possible outcome, and every row had it
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), stat / 2))


def _outcomes(tokens, accepted, k) -> torch.Tensor:
    """``[k + 1, V]`` counts of (drafts kept, the token after them)."""
    y = tokens.gather(1, accepted[:, None]).squeeze(1)
    return torch.bincount(accepted * VOCAB + y, minlength=(k + 1) * VOCAB).view(k + 1, VOCAB)


def _target_logits(k: int, gen: torch.Generator, peaked: bool = False) -> torch.Tensor:
    """A different distribution at each of the ``k + 1`` positions. ``peaked``: every
    position before the last puts all of the filtered mass on token 1 + its index."""
    logits = torch.randn(k + 1, VOCAB, generator=gen) * 1.5
    if peaked:
        for j in range(k):
            logits[j, (1 + j) % VOCAB] += 60.0
    return logits


DRAFTERS = ["wrong", "likely", "right", "random", "sampled", "sampled_exact"]


class _Case(NamedTuple):
    """One drafter's verify over many rows: the chi-square p-values of the outcomes (drafts
    kept, the token after them), of the first token alone, and of each sampled draft against
    its distribution; and the mean drafts kept per row."""
    outcomes: float
    first: float
    drafts: list[float]
    accepted: float


def _run_case(device: str, k: int, drafter: str, n: int, first_seed: int = 1000) -> _Case:
    gen = torch.Generator().manual_seed(17 * k + DRAFTERS.index(drafter))
    logits = _target_logits(k, gen, peaked=drafter == "right")
    p = _filtered(logits)
    sampler = _sampler(device)
    rids = _requests(sampler, n, first_seed)
    draft_probs, draft_p = None, []
    if drafter in ("sampled", "sampled_exact"):
        draft_logits = logits[:k] if drafter == "sampled_exact" else torch.randn(k, VOCAB, generator=gen) * 1.5
        drafts, dists = [], []
        for j in range(k):
            token, dist = sampler.draft_sample(rids, draft_logits[j].expand(n, VOCAB).to(device))
            drafts.append(token.cpu())
            dists.append(dist)
        drafts, draft_probs = torch.stack(drafts, 1), torch.stack(dists, 1)
        q = _filtered(draft_logits)
        draft_p = [_chi2(torch.bincount(drafts[:, j], minlength=VOCAB), q[j] * n) for j in range(k)]
        expected = _expected(p, q) * n
    elif drafter == "random":
        drafts = torch.randint(VOCAB, (n, k), generator=gen)
        # each row has its own drafts: sum the rows' outcome probabilities
        expected = torch.zeros(k + 1, VOCAB, dtype=torch.float64)
        for combo, count in zip(*drafts.unique(dim=0, return_counts=True), strict=True):
            expected += _expected(p, F.one_hot(combo, VOCAB).double()) * count
    else:
        chosen = p[:k].argmin(-1) if drafter == "wrong" else p[:k].argmax(-1)
        if drafter == "wrong":
            assert bool((p[:k].gather(1, chosen[:, None]) == 0).all())
        drafts = chosen.expand(n, k)
        expected = _expected(p, F.one_hot(chosen, VOCAB).double()) * n
    block = torch.cat([torch.zeros(n, 1, dtype=torch.long), drafts], 1).to(device)
    tokens, accepted = sampler.verify(rids, block, logits.repeat(n, 1).to(device), draft_probs)
    tokens, accepted = tokens.cpu(), accepted.cpu()
    # a row emits its kept drafts, then one token
    kept = torch.arange(k + 1)[None, :] < accepted[:, None]
    assert torch.equal(tokens[kept], F.pad(drafts, (0, 1))[kept])
    return _Case(
        outcomes=_chi2(_outcomes(tokens, accepted, k), expected),
        first=_chi2(torch.bincount(tokens[:, 0], minlength=VOCAB), p[0] * n),
        drafts=draft_p,
        accepted=float(accepted.double().mean()),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("drafter", DRAFTERS)
def test_the_emitted_tokens_follow_the_target_whatever_the_drafter(device, k, drafter):
    """Point-mass drafters: one the filtered target never holds ("wrong", nothing is kept),
    its likeliest token ("likely"), the only token it holds at each position but the last
    ("right", everything is kept), a uniformly random token per row ("random"). Sampling
    drafters: from a different distribution ("sampled"), and from the target's own
    ("sampled_exact", everything is kept). The first token alone, kept draft or not, must be
    the target's; a sampling drafter's draws must be its own distribution's."""
    case = _run_case(device, k, drafter, 20000 if device == "cpu" else 200000)
    assert case.outcomes > P_MIN and case.first > P_MIN, case
    assert all(p > P_MIN for p in case.drafts), case
    if drafter == "wrong":
        assert case.accepted == 0
    elif drafter in ("right", "sampled_exact"):
        assert case.accepted > k - 0.01
    else:
        assert 0 < case.accepted < k


def test_the_check_tells_a_residual_that_keeps_the_refused_draft(monkeypatch):
    """The test above is not vacuous: replace the refused position's residual by the target
    itself, so a refused draft can come straight back, and the outcomes leave the target's."""
    from mstar.engine.resources.sampler import verify as verify_mod

    def wrong(probs, drafts, draft_probs, uniforms):
        n, block, _ = probs.shape
        accepted, _ = real(probs, drafts, draft_probs, uniforms)
        p_at = probs[torch.arange(n), accepted]
        return accepted, verify_mod._inverse_cdf(p_at, uniforms[:, block - 1])

    real = verify_mod._chain_verify_torch
    monkeypatch.setattr(verify_mod, "_chain_verify_torch", wrong)
    k, n = 2, 20000
    logits = _target_logits(k, torch.Generator().manual_seed(3))
    p = _filtered(logits)
    chosen = p[:k].argmax(-1)
    sampler = _sampler("cpu")
    rids = _requests(sampler, n)
    block = torch.cat([torch.zeros(n, 1, dtype=torch.long), chosen.expand(n, k)], 1)
    tokens, accepted = sampler.verify(rids, block, logits.repeat(n, 1))
    expected = _expected(p, F.one_hot(chosen, VOCAB).double()) * n
    with pytest.raises(AssertionError, match="can't produce"):
        _chi2(_outcomes(tokens, accepted, k), expected)
    assert _chi2(torch.bincount(tokens[:, 0], minlength=VOCAB), p[0] * n) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_a_greedy_row_beside_sampled_ones_is_verified_greedily(device):
    """Greedy rows go through the same kernel, as targets one-hot at their argmax: they keep
    exactly the drafts a greedy verify keeps and emit the argmax after them."""
    n, k, vocab = 8, 3, 50
    gen = torch.Generator().manual_seed(2)
    logits = torch.randn(n, k + 1, vocab, generator=gen)
    target = logits.argmax(-1)
    drafts = target[:, :k].clone()
    for r in range(n):
        if r % (k + 1) < k:
            drafts[r, r % (k + 1)] = (target[r, r % (k + 1)] + 1) % vocab
    sampler = _sampler(device)
    rids = _requests(sampler, n, temperature=0.0)
    sampler.remove_request(0)
    sampler.ingest_request(0, SamplingReqConfig(temperature=0.9, _seed=7))
    block = torch.cat([torch.zeros(n, 1, dtype=torch.long), drafts], 1)
    tokens, accepted = sampler.verify(rids, block.to(device), logits.view(n * (k + 1), vocab).to(device))
    want = (drafts == target[:, :k]).long().cumprod(1).sum(1)
    for r in range(1, n):
        assert int(accepted[r]) == int(want[r]), r
        upto = int(want[r]) + 1
        assert torch.equal(tokens[r, :upto].cpu(), target[r, :upto]), r


def _one_step(device, seeds, k=3, logits=None):
    sampler = _sampler(device)
    n = len(seeds)
    for rid, seed in enumerate(seeds):
        sampler.ingest_request(rid, SamplingReqConfig(temperature=1.0, _seed=seed))
    gen = torch.Generator().manual_seed(5)
    if logits is None:
        logits = torch.randn(n, k + 1, VOCAB, generator=gen)
    drafts = torch.randint(VOCAB, (n, k), generator=gen)
    block = torch.cat([torch.zeros(n, 1, dtype=torch.long), drafts], 1)
    out = []
    for _ in range(3):
        tokens, accepted = sampler.verify(list(range(n)), block.to(device), logits.view(-1, VOCAB).to(device))
        out.append((tokens.cpu(), accepted.cpu()))
    return out, sampler


@pytest.mark.parametrize("device", DEVICES)
def test_a_request_seed_reproduces_its_draws(device):
    """The same seeds draw the same verdicts step for step; other seeds draw others; and each
    step draws past the last one, so a request's verdicts change from step to step."""
    seeds = list(range(100, 164))
    first, sampler = _one_step(device, seeds)
    again, _ = _one_step(device, seeds)
    for (t1, a1), (t2, a2) in zip(first, again, strict=True):
        assert torch.equal(t1, t2) and torch.equal(a1, a2)
    other, _ = _one_step(device, [s + 1000 for s in seeds])
    assert not torch.equal(first[0][0], other[0][0])
    assert not torch.equal(first[0][0], first[1][0]), "a step reused the last step's numbers"
    # three verify steps of k = 3 moved each request's offset by 3 (k + 2)
    assert all(sampler._sampler._step_offset[rid] == 3 * verify_stride(3) for rid in range(len(seeds)))


def test_on_cpu_a_request_draws_the_same_in_any_batch():
    """The torch version keys each row's numbers off its own seed and offset. (FlashInfer's
    kernel reads the first row's, as main's sampler does on a GPU: a GPU run reproduces for
    the same seeds in the same batch layout.)"""
    seeds = [11, 12, 13, 14]
    gen = torch.Generator().manual_seed(9)
    logits = torch.randn(4, 4, VOCAB, generator=gen)
    block = torch.cat([torch.zeros(4, 1, dtype=torch.long), torch.randint(VOCAB, (4, 3), generator=gen)], 1)
    batch, single = _sampler("cpu"), _sampler("cpu")
    for rid, seed in enumerate(seeds):
        batch.ingest_request(rid, SamplingReqConfig(temperature=1.0, _seed=seed))
    single.ingest_request(7, SamplingReqConfig(temperature=1.0, _seed=13))
    for _ in range(4):
        t_batch, a_batch = batch.verify([0, 1, 2, 3], block, logits.view(-1, VOCAB))
        t_one, a_one = single.verify([7], block[2:3], logits[2].view(-1, VOCAB))
        assert torch.equal(t_batch[2], t_one[0]) and int(a_batch[2]) == int(a_one[0])


def test_a_draft_sample_moves_the_offset_by_one_draw():
    sampler = _sampler("cpu")
    rids = _requests(sampler, 2)
    sampler.draft_sample(rids, torch.randn(2, VOCAB))
    sampler.draft_sample(rids, torch.randn(2, VOCAB))
    sampler.verify(rids, torch.zeros(2, 3, dtype=torch.long), torch.randn(6, VOCAB))
    assert sampler._sampler._step_offset == {0: 2 + verify_stride(2), 1: 2 + verify_stride(2)}


def test_a_greedy_verify_drafts_the_argmax_with_no_distribution():
    greedy = SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=torch.device("cpu"), draft_verify=True,
    )
    logits = torch.randn(3, VOCAB)
    tokens, probs = greedy.draft_sample([0, 1, 2], logits)
    assert torch.equal(tokens, logits.argmax(-1)) and probs is None


def test_what_a_sampled_verify_serves_and_refuses():
    spec_kwargs = dict(vocab_size=None, enable_repetion_penalty=False, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="needs draft_verify=True"):
        SamplerResource(**spec_kwargs, enable_sampled_verify=True)
    sampler = _sampler("cpu", enable_min_p=True)
    sampler.ingest_request(1, SamplingReqConfig(temperature=0.7, top_k=20, top_p=0.9, min_p=0.05))
    for bad in ({"repetition_penalty": 1.2}, {"min_tokens": 2}):
        with pytest.raises(ValueError, match="can't apply"):
            sampler.ingest_request(2, SamplingReqConfig(temperature=0.7, **bad))
        assert 2 not in sampler._sampler._sampling_config
    greedy = SamplerResource(**spec_kwargs, draft_verify=True)
    with pytest.raises(ValueError, match="greedy-only on this deployment"):
        greedy.ingest_request(3, SamplingReqConfig(temperature=0.7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_draft_refused_at_u_one_is_emitted(monkeypatch):
    """curand's uniforms are in (0, 1]: at u = 1 the kernel refuses a draft the target holds
    all of q's mass on, finds an empty residual, and returns an id it never set. The verify
    emits the draft there instead."""
    flashinfer = pytest.importorskip("flashinfer")
    sampler = _sampler("cuda")
    rids = _requests(sampler, 2, temperature=0.0)
    k = 2
    logits = torch.zeros(2 * (k + 1), VOCAB, device="cuda")
    logits[:, 3] = 5.0  # greedy target: one-hot at 3 everywhere
    block = torch.tensor([[0, 3, 3], [0, 3, 3]], device="cuda")

    def kernel(draft_probs, drafts, target, *args, **kwargs):
        n = drafts.shape[0]
        out = torch.full((n, k + 1), -1, dtype=torch.int32, device="cuda")
        out[0, 0] = 123456  # row 0: refused at 0, garbage id
        out[1, :2] = 3  # row 1: kept one, refused at 1, an id the target doesn't hold
        out[1, 1] = 5
        return out, None, torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    monkeypatch.setattr(flashinfer.sampling, "chain_speculative_sampling", kernel)
    tokens, accepted = sampler.verify(rids, block, logits)
    assert accepted.tolist() == [0, 1]
    assert tokens[0, 0].item() == 3 and tokens[1, :2].tolist() == [3, 3]
