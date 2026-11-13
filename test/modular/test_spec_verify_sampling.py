"""Speculative decoding: the verify math (`sampler.verify`) on the CPU.

Greedy rows accept drafts while they equal the argmax; sampled rows emit a run
distributed as sampling the target one token at a time, whatever the drafts.
"""
from __future__ import annotations

import itertools
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.sampler.verify import _uniforms, chain_verify, filtered_probs  # noqa: E402


def _greedy_probs(logits: torch.Tensor) -> torch.Tensor:
    n, block, vocab = logits.shape
    rows = n * block
    return filtered_probs(
        logits.reshape(rows, vocab), torch.zeros(rows), torch.zeros(rows, dtype=torch.int32), torch.ones(rows),
    ).view(n, block, vocab)


def test_greedy_rows_accept_drafts_while_they_equal_the_argmax():
    torch.manual_seed(0)
    n, k, vocab = 6, 4, 11
    logits = torch.randn(n, k + 1, vocab)
    top = logits.argmax(-1)
    drafts = top[:, :k].clone()
    # row i's first wrong draft at position i (row k and beyond: none)
    for i in range(min(n, k)):
        drafts[i, i] = (drafts[i, i] + 1) % vocab
    tokens, accepted = chain_verify(_greedy_probs(logits), drafts, torch.arange(n), torch.zeros(n, dtype=torch.long))
    for i in range(n):
        want = i if i < k else k
        assert accepted[i] == want
        # the accepted drafts, then the target's own token at the first wrong one (or the bonus)
        assert tokens[i, : want + 1].tolist() == top[i, : want + 1].tolist()


def test_an_always_wrong_drafter_emits_exactly_plain_greedy():
    torch.manual_seed(1)
    n, k, vocab = 3, 5, 7
    logits = torch.randn(n, k + 1, vocab)
    top = logits.argmax(-1)
    drafts = (top[:, :k] + 1) % vocab
    tokens, accepted = chain_verify(_greedy_probs(logits), drafts, torch.arange(n), torch.zeros(n, dtype=torch.long))
    assert accepted.tolist() == [0] * n
    assert tokens[:, 0].tolist() == top[:, 0].tolist()


def _sampled_run_distribution(p: torch.Tensor, drafts: torch.Tensor, rows: int, seed: int = 7):
    """Empirical distribution of each sampled row's emitted run, as tuples."""
    k = drafts.shape[0]
    probs = p.expand(rows, k + 1, p.shape[-1]).contiguous()
    tokens, accepted = chain_verify(
        probs, drafts.expand(rows, k), torch.full((rows,), seed), torch.arange(rows) * (k + 1),
    )
    runs: dict[tuple[int, ...], int] = {}
    for row, a in zip(tokens.tolist(), accepted.tolist(), strict=True):
        run = tuple(row[: a + 1])
        runs[run] = runs.get(run, 0) + 1
    return {run: count / rows for run, count in runs.items()}


@pytest.mark.parametrize("drafts", [[0, 0], [2, 1], [1, 2]])
def test_a_sampled_run_is_distributed_as_the_target_token_by_token(drafts):
    # independent positions: the target's distribution at each of k + 1 rows
    p = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5]])
    k = len(drafts)
    observed = _sampled_run_distribution(p, torch.tensor(drafts), rows=60000)
    # every run the verify can emit: accepted drafts d_0..d_{j-1}, then one token
    expected: dict[tuple[int, ...], float] = {}
    for j in range(k + 1):
        prefix = tuple(drafts[:j])
        mass = 1.0
        for i in range(j):
            mass *= float(p[i, drafts[i]])
        for token in range(3):
            if j < k and token == drafts[j]:
                continue  # that would have been an acceptance
            expected[prefix + (token,)] = expected.get(prefix + (token,), 0.0) + mass * float(p[j, token])
    assert set(observed) <= set(expected)
    for run, mass in expected.items():
        # about 4.5 standard errors at 60000 rows
        assert abs(observed.get(run, 0.0) - mass) < 0.009, (run, observed.get(run), mass)


def test_top_k_filters_what_a_sampled_row_can_emit():
    vocab = 6
    logits = torch.tensor([[[3.0, 2.9, 2.8, 0.0, 0.0, 0.0]] * 2] * 4000)
    rows = logits.shape[0] * 2
    probs = filtered_probs(
        logits.reshape(rows, vocab), torch.ones(rows), torch.full((rows,), 2, dtype=torch.int32), torch.ones(rows),
    ).view(-1, 2, vocab)
    drafts = torch.full((logits.shape[0], 1), 4)  # outside the top 2: never accepted
    n = logits.shape[0]
    tokens, accepted = chain_verify(probs, drafts, torch.arange(n), torch.zeros(n, dtype=torch.long))
    assert accepted.sum() == 0
    assert set(tokens[:, 0].tolist()) == {0, 1}


def test_draws_depend_on_seed_and_offset_alone():
    seed, offset = torch.tensor([5, 5, 9]), torch.tensor([0, 3, 0])
    a = _uniforms(seed, offset, 4, torch.float32)
    # the same request alone in a batch, and at the offset it would reach later
    b = _uniforms(seed[:1], torch.tensor([3]), 4, torch.float32)
    assert torch.equal(a[1], b[0])
    assert torch.equal(a[0, 3], b[0, 0])
    assert not torch.equal(a[0], a[2])
    assert ((a >= 0) & (a < 1)).all()


def test_consecutive_steps_never_share_a_number():
    k = 3
    seed = torch.tensor([11])
    steps = [_uniforms(seed, torch.tensor([s * (k + 1)]), k + 1, torch.float64)[0] for s in range(50)]
    values = torch.cat(steps).tolist()
    # a counter hash: each step's numbers are the next k + 1 of one stream
    assert len(values) == len(set(values))
    assert all(x != y for x, y in itertools.pairwise(values))


def test_an_eager_verify_moves_only_the_kept_rows_offsets():
    """As `sample` does: a dropped row (``SamplerStep.kept_rids``) still draws, but its offset stays."""
    from types import SimpleNamespace

    from mstar.engine.resources.sampler.resource import SamplerResource
    from mstar.engine.resources.sampler.utils import Sampler

    sampler = SamplerResource.__new__(SamplerResource)
    configs = {rid: SimpleNamespace(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, seed=rid) for rid in (1, 2)}
    eager = sampler._sampler = SimpleNamespace(
        _sampling_config=configs, _step_offset={}, _gather_filter=lambda configs: None,
    )
    # the eager sampler's own settings gather, which sample reads too
    eager.row_settings = lambda rids, device, filters=True: Sampler.row_settings(eager, rids, device, filters)
    eager._row_settings = lambda cfgs, device, apply_filters: Sampler._row_settings(eager, cfgs, device, apply_filters)
    sampler._cg_sampler = sampler._cg_buffers = sampler._comm_group = None
    sampler._apply_filters_this_step = True
    sampler._greedy_this_step = False
    sampler._keep_last_token = False
    sampler._kept_rids = frozenset({1})
    k, vocab = 3, 8
    result = sampler.verify([1, 2], torch.randn(2, k + 1, vocab), torch.zeros(2, k, dtype=torch.long))
    assert result.tokens.shape == (2, k + 1)
    assert sampler._sampler._step_offset == {1: k + 1}


# ---- the CUDA branch (GPU) ----

def _filter_case(rows: int = 12, vocab: int = 1000, seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=gen) * 3
    temperature = torch.tensor([0.0, 0.6, 1.0, 0.3] * (rows // 4))
    top_k = torch.tensor([0, 20, 0, 50] * (rows // 4), dtype=torch.int32)
    top_p = torch.tensor([1.0, 0.95, 0.8, 1.0] * (rows // 4))
    min_p = torch.tensor([0.0, 0.0, 0.05, 0.1] * (rows // 4))
    return logits, temperature, top_k, top_p, min_p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("with_min_p", [False, True])
def test_the_cuda_filters_match_the_reference(with_min_p):
    """Qwen3-8B's defaults (T 0.6, top_k 20, top_p 0.95) verify through
    the CUDA branch of filtered_probs; it matches the torch reference."""
    pytest.importorskip("flashinfer")
    logits, temperature, top_k, top_p, min_p = _filter_case()
    min_p = min_p if with_min_p else None
    cpu = filtered_probs(logits, temperature, top_k, top_p, min_p)
    gpu = filtered_probs(*(t.cuda() if t is not None else None for t in (logits, temperature, top_k, top_p, min_p)))
    torch.testing.assert_close(gpu.cpu(), cpu, atol=2e-4, rtol=1e-3)
    # the same support: no token one keeps and the other filters
    assert torch.equal(gpu.cpu() > 1e-6, cpu > 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chain_verify_on_cuda_matches_the_cpu_for_the_same_numbers():
    """The same probabilities, drafts, seeds and offsets: the same tokens and
    accepted counts on either device (the uniforms are a counter hash)."""
    torch.manual_seed(0)
    n, k, vocab = 64, 7, 1000
    probs = torch.softmax(torch.randn(n, k + 1, vocab) * 2, -1)
    # drafts the target likes, so both branches run
    drafts = torch.where(torch.rand(n, k) < 0.6, probs[:, :k].argmax(-1), torch.randint(vocab, (n, k)))
    seed, offset = torch.randint(0, 2**31, (n,)), torch.randint(0, 2**20, (n,))
    cpu = chain_verify(probs, drafts, seed, offset)
    gpu = chain_verify(probs.cuda(), drafts.cuda(), seed.cuda(), offset.cuda())
    assert torch.equal(gpu[1].cpu(), cpu[1])
    assert torch.equal(gpu[0].cpu(), cpu[0])
    assert 0 < cpu[1].float().mean() < k


# ---- the two-renorm branch (vocabularies past the fused filter) ----

def _wide_case(vocab: int, rows: int = 24, seed: int = 5):
    """Greedy, top-p 1 and top-p < 1, top-k 0 and > 0, and Qwen3-8B's defaults (T 0.6, top_k 20, top_p 0.95)."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=gen) * 4
    temperature = torch.tensor([0.0, 1.0, 0.6, 0.8, 1.0, 0.6] * (rows // 6))
    top_k = torch.tensor([0, 0, 20, 0, 100, 0] * (rows // 6), dtype=torch.int32)
    top_p = torch.tensor([1.0, 1.0, 0.95, 0.7, 1.0, 0.5] * (rows // 6))
    return logits, temperature, top_k, top_p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("vocab", [32768, 151936, 248320])
def test_the_two_renorm_verify_filter_matches_the_counting_path_and_the_reference(vocab):
    """`verify_probs` past ``FUSED_FILTER_MAX_VOCAB``: the chosen renorm, row by
    row, is close to the counting path (`filtered_probs` on CUDA) and to the
    torch reference, with the same support."""
    pytest.importorskip("flashinfer")
    from mstar.engine.resources.sampler.verify import _filtered_probs_torch, _TwoFilters, verify_probs

    logits, temperature, top_k, top_p = _wide_case(vocab)
    cuda = [t.cuda() for t in (logits, temperature, top_k, top_p)]
    dist = verify_probs(*cuda)
    assert isinstance(dist, _TwoFilters)
    chosen = torch.where(dist.use_then[:, None], dist.then, dist.first)
    counting = filtered_probs(*cuda)
    reference = _filtered_probs_torch(logits, temperature, top_k, top_p, None)
    torch.testing.assert_close(chosen, counting, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(chosen.cpu(), reference, atol=2e-4, rtol=1e-3)
    assert torch.equal(chosen > 0, counting > 0)
    assert torch.equal(chosen.cpu() > 1e-6, reference > 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_two_renorm_verify_draws_the_target_run_by_run():
    """At a vocabulary past the fused filter, sampled rows with top-k and top-p:
    the emitted runs are distributed as the filtered target token by token (the
    torch reference's distribution), whatever the drafts."""
    pytest.importorskip("flashinfer")
    from mstar.engine.resources.sampler.verify import _filtered_probs_torch, chain_verify_rows, verify_probs

    vocab, k, rows = 32768, 2, 12000
    block = k + 1
    # three tokens hold the mass at each position, as in the small-vocabulary test
    base = torch.full((block, vocab), -30.0)
    base[:, :4] = torch.tensor([[2.0, 1.5, 1.0, 0.2], [0.2, 2.0, 1.2, 0.0], [1.0, 1.0, 1.6, 0.4]])
    # top-p cuts a token past top-k's three at positions 0 and 1 (no ties there) and not at 2, so the
    # verify mixes rows of the two filters (`_TwoFilters.use_then`)
    temperature, top_k = torch.ones(block), torch.full((block,), 3, dtype=torch.int32)
    top_p = torch.tensor([0.7, 0.7, 1.0])
    p = _filtered_probs_torch(base, temperature, top_k, top_p, None)[:, :4].double()
    uncut = _filtered_probs_torch(base, temperature, top_k, torch.ones(block), None)[:, :4].double()
    assert not torch.allclose(p[:2], uncut[:2]) and torch.allclose(p[2], uncut[2])
    drafts = [1, 2]
    expect: dict[tuple[int, ...], float] = {}
    for j in range(k + 1):
        prefix, mass = tuple(drafts[:j]), 1.0
        for i in range(j):
            mass *= float(p[i, drafts[i]])
        for token in range(4):
            if j < k and token == drafts[j]:
                continue
            expect[prefix + (token,)] = expect.get(prefix + (token,), 0.0) + mass * float(p[j, token])
    logits = base.cuda().repeat(rows, 1)
    dist = verify_probs(logits, temperature.cuda().repeat(rows), top_k.cuda().repeat(rows), top_p.cuda().repeat(rows))
    tokens, accepted = chain_verify_rows(
        dist, rows, torch.tensor(drafts, device="cuda").expand(rows, k), torch.full((rows,), 17, device="cuda"),
        torch.arange(rows, device="cuda") * block,
    )
    runs: dict[tuple[int, ...], int] = {}
    for row, a in zip(tokens.tolist(), accepted.tolist(), strict=True):
        runs[tuple(row[: a + 1])] = runs.get(tuple(row[: a + 1]), 0) + 1
    assert set(runs) <= {r for r, m in expect.items() if m > 0}
    for run, mass in expect.items():
        assert abs(runs.get(run, 0) / rows - mass) < 0.02, (run, runs.get(run, 0) / rows, mass)


# ---- the top-p renorm's unread rows (p = 1 reads ``first``) ----

def test_the_renorm_runs_each_read_row_at_its_own_top_p_and_unread_ones_at_the_sentinel():
    from mstar.engine.resources.sampler.verify import _UNREAD_TOP_P, renorm_top_p

    assert 0.0 < _UNREAD_TOP_P < 1.0
    top_p = torch.tensor([1.0, 0.95, 1.0, 0.5, 0.999999, 1.0])
    expect = torch.tensor([_UNREAD_TOP_P, 0.95, _UNREAD_TOP_P, 0.5, 0.999999, _UNREAD_TOP_P])
    assert torch.equal(renorm_top_p(top_p), expect)


def _mixed_verify_case(vocab: int, temperature: float, n: int = 8, k: int = 3, seed: int = 11):
    """``n`` requests of ``k + 1`` rows, alternating top-p 1 and top-p < 1, drafts half right (their tokens
    peaked, so a sampled row accepts them too) and half random."""
    gen = torch.Generator().manual_seed(seed)
    block = k + 1
    logits = torch.randn(n * block, vocab, generator=gen) * 4
    top_p = torch.tensor([1.0, 0.9, 1.0, 0.5] * (n // 4)).repeat_interleave(block)
    drafts = logits.view(n, block, vocab)[:, :k].argmax(-1)
    right = drafts[: n // 2, :, None]
    logits.view(n, block, vocab)[: n // 2, :k].scatter_add_(-1, right, torch.full(right.shape, 12.0))
    drafts[n // 2:] = torch.randint(0, vocab, (n - n // 2, k), generator=gen)
    rows = n * block
    return (logits, torch.full((rows,), temperature), torch.zeros(rows, dtype=torch.int32), top_p, drafts,
            torch.arange(n) + 3, torch.arange(n) * block)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_a_mixed_top_p_batch_verifies_each_request_as_it_would_alone(temperature):
    """Requests at top-p 1 and top-p < 1 in one verify: each one's tokens and accepted count are what it gets
    verified alone, at the same seed and offset."""
    from mstar.engine.resources.sampler.verify import chain_verify_rows, verify_probs

    logits, temp, top_k, top_p, drafts, seed, offset = _mixed_verify_case(1000, temperature)
    n, block = drafts.shape[0], drafts.shape[1] + 1
    tokens, accepted = chain_verify_rows(verify_probs(logits, temp, top_k, top_p), n, drafts, seed, offset)
    for i in range(n):
        rows = slice(i * block, (i + 1) * block)
        alone = chain_verify_rows(
            verify_probs(logits[rows], temp[rows], top_k[rows], top_p[rows]), 1, drafts[i:i + 1],
            seed[i:i + 1], offset[i:i + 1],
        )
        assert torch.equal(tokens[i], alone[0][0]) and accepted[i] == alone[1][0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_the_unread_rows_sentinel_leaves_every_verify_result_bit_identical(temperature):
    """At Qwen3.5's vocabulary, a mixed top-p batch: running the top-p renorm at `renorm_top_p` instead of each
    row's own p changes no token and no accepted count, the read rows' renorm is bit-equal, and nothing is
    non-finite."""
    flashinfer = pytest.importorskip("flashinfer")
    from mstar.engine.resources.sampler.verify import _TwoFilters, chain_verify_rows, verify_probs

    logits, temp, top_k, top_p, drafts, seed, offset = (
        t.cuda() for t in _mixed_verify_case(248320, temperature)
    )
    n = drafts.shape[0]
    dist = verify_probs(logits.bfloat16(), temp, top_k, top_p)
    assert isinstance(dist, _TwoFilters)
    with torch.cuda.device(logits.device):
        own = flashinfer.sampling.top_p_renorm_probs(dist.first, top_p)
    read = dist.use_then
    assert torch.equal(dist.then[read], own[read])
    assert torch.isfinite(dist.then).all() and torch.isfinite(dist.first).all()
    tokens, accepted = chain_verify_rows(dist, n, drafts, seed, offset)
    before = chain_verify_rows(_TwoFilters(dist.first, own, dist.use_then), n, drafts, seed, offset)
    assert torch.equal(tokens, before[0]) and torch.equal(accepted, before[1])
    # the batch exercises both a rejected first draft and an accepted one
    assert (accepted == 0).any() and (accepted > 0).any()
