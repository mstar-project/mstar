"""Speculative sampling of a verify block with the drafts taken as certain (MTP's argmax drafts).

What makes it sound: whatever the drafts, the tokens it emits are distributed as sampling the
target one token at a time. The first emitted token is the cleanest check: kept draft or
replacement, it must follow p_0 exactly. A greedy row must reduce to greedy verification.
The torch path runs everywhere; the FlashInfer path on a GPU.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.sampler.resource import SamplerResource


def _sampler(device):
    return SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=device)


def _requests(sampler, n, **config):
    rids = [f"r{i}" for i in range(n)]
    for i, rid in enumerate(rids):
        sampler._sampler.add_request(rid)
        sampler._sampler.set_config(rid, **config)
        sampler._sampler._sampling_config[rid].set_seed(1000 + i)
    return rids


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_a_greedy_row_keeps_its_argmax_prefix(device):
    sampler = _sampler(torch.device(device))
    torch.manual_seed(0)
    n, k, vocab = 6, 3, 50
    logits = torch.randn(n, k + 1, vocab, device=device)
    target = logits.argmax(-1)
    # row r's drafts agree with the argmax for its first r % (k + 1) positions
    drafts = target[:, :k].clone()
    for r in range(n):
        if r % (k + 1) < k:
            drafts[r, r % (k + 1)] = (target[r, r % (k + 1)] + 1) % vocab
    rids = _requests(sampler, n, temperature=0.0)
    tokens, kept = sampler.speculative_sample(rids, logits, drafts)
    want_kept = (drafts == target[:, :k]).long().cumprod(1).sum(1)
    assert torch.equal(kept, want_kept)
    for r in range(n):
        upto = int(kept[r]) + 1
        assert torch.equal(tokens[r, :upto], target[r, :upto]), r


@pytest.mark.parametrize("device", DEVICES)
def test_a_greedy_row_beside_a_sampled_one_keeps_its_argmax_prefix(device):
    """In a batch with a sampled request the greedy rows go through the sampling kernel too,
    as targets one-hot at their argmax."""
    sampler = _sampler(torch.device(device))
    torch.manual_seed(2)
    n, k, vocab = 8, 3, 50
    logits = torch.randn(n, k + 1, vocab, device=device)
    target = logits.argmax(-1)
    drafts = target[:, :k].clone()
    drafts[1::2, 1] = (target[1::2, 1] + 1) % vocab
    rids = _requests(sampler, n, temperature=0.0)
    sampler._sampler.set_config(rids[0], temperature=0.7)
    tokens, kept = sampler.speculative_sample(rids, logits, drafts)
    want_kept = (drafts == target[:, :k]).long().cumprod(1).sum(1)
    for r in range(1, n):
        upto = int(want_kept[r]) + 1
        assert int(kept[r]) == int(want_kept[r]), r
        assert torch.equal(tokens[r, :upto], target[r, :upto]), r


def test_a_verify_step_moves_the_requests_rng_past_every_draw():
    """A row draws up to block + 1 numbers from its Philox stream (each draft, the refused one
    again, the replacement), so the next step starts past them, on the counter the request's
    other samples move."""
    sampler = _sampler(torch.device("cpu"))
    (rid,) = _requests(sampler, 1, temperature=0.7)
    sampler._sampler._step_offset[rid] = 3  # the prefill's sample moved it
    k = 3
    sampler.speculative_sample(
        [rid], torch.randn(1, k + 1, 8), torch.zeros(1, k, dtype=torch.long))
    assert sampler._sampler._step_offset[rid] == 3 + k + 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("draft_of", ["likely", "unlikely"])
def test_the_first_token_follows_the_target_whatever_the_draft(device, draft_of):
    """p_0 over 6 tokens at temperature 0.7 and top_p 0.9; the draft is p_0's likeliest or one
    p_0 barely holds. Over many rows the first emitted token's frequencies match p_0's."""
    torch.manual_seed(1)
    vocab, k, n = 6, 2, 20000 if device == "cpu" else 50000
    base = torch.tensor([2.0, 1.5, 0.5, 0.0, -1.0, -3.0])
    logits = base.repeat(n, k + 1, 1).to(device)
    temperature, top_p = 0.7, 0.9
    probs = torch.softmax(base / temperature, -1)
    order = probs.argsort(descending=True)
    keep = (probs[order].cumsum(0) - probs[order]) < top_p
    p0 = torch.zeros(vocab)
    p0[order[keep]] = probs[order[keep]]
    p0 /= p0.sum()
    draft = 0 if draft_of == "likely" else 4
    drafts = torch.full((n, k), draft, device=device)
    sampler = _sampler(torch.device(device))
    rids = _requests(sampler, n if device == "cuda" else 1, temperature=temperature, top_p=top_p)
    if device == "cpu":
        firsts = []
        for _ in range(n // 100):
            tokens, _ = sampler.speculative_sample(rids * 100, logits[:100], drafts[:100])
            firsts.append(tokens[:, 0])
        first = torch.cat(firsts)
    else:
        tokens, _ = sampler.speculative_sample(rids, logits, drafts)
        first = tokens[:, 0].cpu()
    freq = torch.bincount(first, minlength=vocab).float() / first.numel()
    # 5 standard errors of a binomial proportion
    tol = 5 * (p0 * (1 - p0) / first.numel()).sqrt() + 1e-4
    assert bool(((freq - p0).abs() <= tol).all()), (freq.tolist(), p0.tolist())
