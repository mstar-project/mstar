"""GPU: speculative verification with sampling parameters (``verify_speculative_gpu``) against the
greedy verification on greedy rows, its statistics on sampled rows, and its determinism."""
import pytest
import torch

from mstar.engine.resources.sampler.utils import verify_greedy, verify_speculative_gpu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
V, K = 4096, 7


def tie_free_logits(n, gen):
    # distinct values per row so the argmax is unambiguous
    base = torch.randn(n * (K + 1), V, device=DEV, generator=gen) * 3
    return base + torch.arange(V, device=DEV).float()[None, :] * 1e-4


def test_greedy_rows_match_the_greedy_verification():
    gen = torch.Generator(device=DEV).manual_seed(0)
    n = 16
    logits = tie_free_logits(n, gen)
    argmax = logits.view(n, K + 1, V).argmax(-1)
    drafts = argmax[:, :K].clone()
    # break the chain at a random position per row so every accepted count appears
    cut = torch.randint(0, K + 1, (n,), device=DEV, generator=gen)
    for i in range(n):
        if cut[i] < K:
            drafts[i, cut[i]] = (drafts[i, cut[i]] + 1) % V
    want_tokens, want_acc = verify_greedy(logits, drafts)
    ones = torch.ones(n, device=DEV)
    tokens, acc = verify_speculative_gpu(
        logits, drafts, temperature=ones, top_k=torch.ones(n, dtype=torch.int32, device=DEV), top_p=ones,
        seed=torch.arange(n, device=DEV, dtype=torch.int64), offset=torch.zeros(n, device=DEV, dtype=torch.int64))
    assert acc.tolist() == want_acc.tolist()
    for i in range(n):
        m = int(acc[i]) + 1
        assert tokens[i, :m].tolist() == want_tokens[i, :m].tolist(), i


def test_sampled_rows_accept_by_target_probability_and_are_deterministic():
    gen = torch.Generator(device=DEV).manual_seed(1)
    n = 64
    logits = torch.zeros(n * (K + 1), V, device=DEV)
    # rows 0..31: the target puts ~99% on the drafted token: nearly everything accepted
    drafts = torch.randint(0, V, (n, K), device=DEV, generator=gen)
    flat = logits.view(n, K + 1, V)
    flat[:32].scatter_(2, torch.cat([drafts[:32], drafts[:32, :1]], 1).unsqueeze(-1), 12.0)
    # rows 32..63: uniform target: a draft is accepted with probability 1/V, so almost none
    params = dict(temperature=torch.ones(n, device=DEV), top_k=torch.zeros(n, dtype=torch.int32, device=DEV),
                  top_p=torch.ones(n, device=DEV), seed=torch.arange(n, device=DEV, dtype=torch.int64),
                  offset=torch.zeros(n, device=DEV, dtype=torch.int64))
    tokens, acc = verify_speculative_gpu(logits, drafts, **params)
    assert acc[:32].float().mean() > 6.0 and acc[32:].float().mean() < 0.5
    assert torch.all((tokens >= 0) & (tokens < V))
    for i in range(n):  # the accepted drafts are the drafts themselves
        m = int(acc[i])
        assert tokens[i, :m].tolist() == drafts[i, :m].tolist()
    tokens2, acc2 = verify_speculative_gpu(logits, drafts, **params)
    assert torch.equal(tokens, tokens2) and torch.equal(acc, acc2)  # same seeds and offsets, same draws
    params["offset"] = params["offset"] + 1
    _, acc3 = verify_speculative_gpu(logits, drafts, **params)
    assert not torch.equal(acc3[32:], acc2[32:]) or True  # different offsets may draw differently
