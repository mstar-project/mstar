"""The sampled leg of a verify step, on CPU tensors: the math of speculative
sampling on filtered distributions, exactly on small vocabularies, and the
bookkeeping around it in the resource.

The torch paths of `sampled` stand in for the kernels, and the resource runs as a
forward drives it. Whatever the draft, every emitted token must follow the
filtered target; how much of the time a row keeps its draft is `sum min(p, q)`.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.spec_decode import sampled  # noqa: E402
from mstar.engine.resources.spec_decode.config import SpecDecodeConfig, SpecPhase  # noqa: E402
from mstar.engine.resources.spec_decode.resource import (  # noqa: E402
    RNG_START,
    SpecDecodePlan,
    SpecDecodeResource,
)

CPU = torch.device("cpu")
PAGE = 4
VOCAB = 12


def _filter(logits, temperature, top_k=0, top_p=1.0):
    rows = logits.shape[0]
    return sampled.filtered_probs(
        logits, torch.full((rows,), float(temperature)),
        torch.full((rows,), top_k or logits.shape[1], dtype=torch.int32), torch.full((rows,), float(top_p)),
    )


def _spec(rows: int, k: int = 1, vocab: int = VOCAB, temperature=0.7, top_p=1.0, top_k=0) -> SpecDecodeResource:
    """The resource with a verify plan over `rows` requests, all at the same settings."""
    spec = SpecDecodeResource(device=CPU, num_draft_tokens=k, max_requests=rows, page_size=PAGE, kv_key="kv",
                              attn_key="attn", sampling=True, vocab_size=vocab)
    for r in range(rows):
        spec.ingest_request(r)
    image = spec._new_image(rows)
    image.rows.copy_(torch.tensor([spec._row_of[r] for r in range(rows)], dtype=torch.int32))
    image.real.fill_(1)
    table = torch.arange(1, rows * 8 + 1, dtype=torch.int32).view(rows, 8)
    state = torch.zeros(rows * (k + 1), dtype=torch.long)
    spec._current = SpecDecodePlan(phase=SpecPhase.VERIFY, num_rows=rows, image=image, block_table=table,
                                   kv_lens=torch.zeros(rows, dtype=torch.int32), write_pages=state,
                                   write_offsets=state.clone())
    regs = slice(1, rows + 1)
    spec._temperature[regs], spec._top_p[regs], spec._top_k[regs] = temperature, top_p, top_k
    spec._seed[regs] = torch.arange(rows)
    spec._rng_offset[regs] = RNG_START
    spec._budget[regs] = 1 << 20
    spec._seq_len[regs] = 10
    return spec


def _set_drafts(spec, tokens, probs):
    """Registers as a step that drew these drafts leaves them. `probs`: [k, vocab]."""
    regs = slice(1, spec._tokens.shape[0])
    spec._tokens[regs] = tokens
    spec._draft_probs[regs] = probs * sampled.DRAFT_MARGIN
    spec._num_acc[regs] = 1
    spec._seq_len[regs] = 10


# --- the filter --------------------------------------------------------------


def test_the_filter_is_top_k_then_top_p_of_the_renormalised_rest():
    probs = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    logits = probs.log()[None]
    # top-k 3 leaves 0.5/0.85, 0.2/0.85, 0.15/0.85. Top-p 0.8 keeps a token while the mass above it is under 0.8
    # of that, so the third (0.824 above it) goes
    got = _filter(logits, 1.0, top_k=3, top_p=0.8)[0]
    want = torch.tensor([0.5, 0.2, 0, 0, 0]) / 0.7
    assert torch.allclose(got, want, atol=1e-5), got


def test_the_filter_keeps_the_token_that_crosses_top_p_and_all_its_ties():
    probs = torch.tensor([0.4, 0.2, 0.2, 0.1, 0.1])
    got = _filter(probs.log()[None], 1.0, top_p=0.5)[0]
    want = torch.tensor([0.4, 0.2, 0.2, 0, 0]) / 0.8
    assert torch.allclose(got, want, atol=1e-5), got


def test_temperature_zero_is_a_point_mass_and_settings_at_their_limits_are_off():
    logits = torch.tensor([[0.1, 2.0, -1.0]])
    assert _filter(logits, 0.0)[0].tolist() == [0.0, 1.0, 0.0]
    assert torch.allclose(_filter(logits, 1.0, top_k=3, top_p=1.0), torch.softmax(logits, -1))


# --- the accept chain, in closed form and against the code --------------------


def _emission(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """What rejection sampling emits at one position, by its definition."""
    stay = torch.minimum(p, q)
    residual = (q - p).clamp_min(0)
    return stay + (1 - stay.sum()) * residual / residual.sum().clamp_min(1e-30)


@pytest.mark.parametrize("settings", [(1.0, 0, 1.0), (0.7, 0, 0.9), (0.7, 5, 1.0), (0.8, 4, 0.6), (1.3, 3, 0.9)])
def test_rejection_sampling_on_filtered_distributions_emits_the_target_exactly(settings):
    """The identity the step rests on, with supports that differ, as a filter makes them."""
    temperature, top_k, top_p = settings
    gen = torch.Generator().manual_seed(1)
    for _ in range(200):
        raw = torch.randn(2, VOCAB, generator=gen) * 2
        q, p = _filter(raw, temperature, top_k, top_p).double()
        assert torch.allclose(_emission(p, q), q, atol=1e-6)


@pytest.mark.parametrize("settings", [(0.7, 0, 0.9), (0.8, 4, 0.6), (1.3, 3, 0.9), (1.0, 0, 1.0)])
def test_a_verify_step_emits_the_filtered_target_and_accepts_sum_min(settings):
    """k = 2 through the resource, drafts drawn from a different filtered
    distribution: the first token follows q0, the token after a kept draft q1,
    and the keep rates are the closed form's."""
    temperature, top_k, top_p = settings
    rows, draws = 500, 40
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(3, VOCAB, generator=gen) * 2
    raw_draft = logits[:2] + torch.randn(2, VOCAB, generator=gen)
    q = _filter(logits, temperature, top_k, top_p)
    p = _filter(raw_draft, temperature, top_k, top_p)
    spec = _spec(rows, k=2, temperature=temperature, top_p=top_p, top_k=top_k)
    first, second = torch.zeros(VOCAB), torch.zeros(VOCAB)
    n = kept_first = kept_second = 0
    for _ in range(draws):
        d1 = torch.multinomial(p[0], rows, replacement=True, generator=gen)
        d2 = torch.multinomial(p[1], rows, replacement=True, generator=gen)
        tokens = torch.stack((torch.full((rows,), 3), d1, d2), dim=1)
        _set_drafts(spec, tokens, p)
        spec.verify_inputs()
        out = spec.accept(logits.repeat(rows, 1), sample=True)
        n += rows
        first += torch.bincount(out.tokens[:, 0], minlength=VOCAB)
        past = out.accepted >= 2
        kept_first += int(past.sum())
        second += torch.bincount(out.tokens[past, 1], minlength=VOCAB)
        kept_second += int((out.accepted >= 3).sum())
    assert 0.5 * (first / n - q[0]).abs().sum() < 0.015
    assert 0.5 * (second / second.sum() - q[1]).abs().sum() < 0.03
    # a token the filter removed from the target is never emitted, whatever the draft holds
    assert not first[q[0] == 0].any() and not second[q[1] == 0].any()
    # the draft's d2 is drawn independently of d1 here, as the toy has no chain: p does not move with d1
    assert abs(kept_first / n - float(torch.minimum(p[0], q[0]).sum())) < 0.015
    assert abs(kept_second / max(kept_first, 1) - float(torch.minimum(p[1], q[1]).sum())) < 0.03


def test_a_point_mass_draft_accepts_as_often_as_the_target_holds_it():
    """A request's first drafts are the head's argmax. Rejection sampling needs no
    special case: it accepts q(d), and otherwise draws from q without d."""
    rows, draws = 500, 40
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.0, -1.0, -3.0]])
    vocab = logits.shape[1]
    q = _filter(logits, 1.0)[0]
    point = torch.zeros(1, vocab)
    point[0, 1] = 1.0
    spec = _spec(rows, k=1, vocab=vocab, temperature=1.0)
    first = torch.zeros(vocab)
    kept = 0
    for _ in range(draws):
        _set_drafts(spec, torch.tensor([3, 1]), point)
        spec.verify_inputs()
        out = spec.accept(torch.cat((logits, logits)).repeat(rows, 1), sample=True)
        first += torch.bincount(out.tokens[:, 0], minlength=vocab)
        kept += int((out.accepted == 2).sum())
    assert abs(kept / (rows * draws) - float(q[1])) < 0.015
    assert 0.5 * (first / (rows * draws) - q).abs().sum() < 0.015


def test_a_seed_stores_its_drafts_as_point_masses_and_restarts_the_stream():
    spec = _spec(2, k=2, temperature=0.7)
    image = spec._new_image(2)
    image.rows[:2] = torch.tensor([1, 2], dtype=torch.int32)
    image.real[:2] = 1
    spec._current = SpecDecodePlan(phase=SpecPhase.SEED, num_rows=2, image=image)
    spec._rng_offset[1:3] = 999
    spec.seed(torch.tensor([4, 5]), torch.tensor([6, 7]))
    assert spec._rng_offset[1:3].tolist() == [RNG_START, RNG_START]
    assert spec._tokens[1].tolist() == [4, 6, 6]
    assert spec._draft_probs[1, 0, 6] == sampled.DRAFT_MARGIN and spec._draft_probs[2, 1, 7] == sampled.DRAFT_MARGIN
    assert spec._draft_probs[1].sum() == pytest.approx(2.0)


def test_a_draft_that_is_all_the_target_holds_is_always_kept():
    """u reaches 1, and with p = q a bare test would reject a draft nothing can
    replace; the stored margin keeps it. The worst u is forced here."""
    q = torch.zeros(1, 2, 4)
    q[0, :, 2] = 1.0
    p = q[:, :1] * sampled.DRAFT_MARGIN
    assert (1.0 * p[0, 0, 2]) < q[0, 0, 2]
    # and a rejection always has somewhere to go, even from a stale distribution on the argmax
    assert (q[0, 0] - p[0, 0]).clamp_min(0).sum() > 0


# --- greedy rows in a sampled batch -------------------------------------------


@pytest.mark.parametrize("stale", ["argmax", "other", "empty"])
def test_a_greedy_row_keeps_exact_greedy_acceptance_whatever_the_registers_hold(stale):
    """A greedy request drafts in the greedy capture, which keeps no draft
    distribution, and may then land in a sampled batch."""
    spec = _spec(3, k=2, temperature=0.0)
    logits = torch.zeros(9, VOCAB)
    # row 0's drafts are right, row 1 goes wrong at the second, row 2 at the first
    for r, argmax in enumerate(((5, 6, 7), (5, 6, 7), (5, 6, 7))):
        for slot, token in enumerate(argmax):
            logits[r * 3 + slot, token] = 9.0
    held = torch.zeros(2, VOCAB)
    if stale == "argmax":
        held[:, 5] = 1.0
    elif stale == "other":
        held[:, 9] = 1.0
    spec._tokens[1:4] = torch.tensor([[3, 5, 6], [3, 5, 1], [3, 2, 1]])
    spec._draft_probs[1:4] = held * sampled.DRAFT_MARGIN
    spec._seq_len[1:4] = 10
    spec.verify_inputs()
    out = spec.accept(logits, sample=True)
    assert out.accepted.tolist() == [3, 2, 1]
    assert out.tokens.tolist() == [[5, 6, 7], [5, 6, -1], [5, -1, -1]]


# --- seeds and offsets --------------------------------------------------------


def _step(spec, logits, k, seen=None):
    spec.verify_inputs()
    out = spec.accept(logits, sample=True)
    drafted = out.target.reshape(-1)
    for j in range(k):
        spec.begin_draft_pass(j)
        drafted = spec.record_draft(j, spec.draw_draft(j, torch.randn(spec._current.num_rows, VOCAB) * 0 + logits[j]))
    return out


def test_a_step_takes_its_words_of_the_stream_without_overlap(monkeypatch):
    """The accept chain and each draft draw read disjoint words of the request's
    stream, and consecutive steps do too, for every request."""
    k, rows, steps = 3, 2, 4
    words: dict[int, list[tuple[int, int]]] = {r: [] for r in range(rows)}
    chain, draw = sampled.accept_chain, sampled.draw

    def spy_chain(p, d, q, seed, offset):
        for r in range(rows):
            words[r].append((int(offset[r]), k + 1))
        return chain(p, d, q, seed, offset)

    def spy_draw(probs, seed, offset):
        for r in range(rows):
            words[r].append((int(offset[r]), 1))
        return draw(probs, seed, offset)

    monkeypatch.setattr(sampled, "accept_chain", spy_chain)
    monkeypatch.setattr(sampled, "draw", spy_draw)
    spec = _spec(rows, k=k)
    logits = torch.arange(VOCAB, dtype=torch.float32).repeat(rows * (k + 1), 1)
    spec._draft_probs[1:] = 1.0 / VOCAB
    for _ in range(steps):
        spec._seq_len[1:] = 10
        _step(spec, logits, k)
    for r in range(rows):
        taken = sorted(words[r])
        assert len(taken) == steps * (1 + k)
        assert taken[0][0] == RNG_START
        for (start, n), (following, _) in zip(taken, taken[1:], strict=False):
            assert start + n <= following, taken
        assert taken[-1][0] + taken[-1][1] == RNG_START + steps * sampled.words_per_step(k)


def test_the_stream_starts_clear_of_the_sampler_resource():
    # the sampler counts from 0 under the same seed; a request would need 2**31 steps to get there
    assert RNG_START == 1 << 31
    assert sampled.words_per_step(2) == 5


def test_a_request_repeats_its_tokens_from_the_same_seed_and_layout():
    k, rows = 2, 3
    logits = torch.randn(rows * (k + 1), VOCAB, generator=torch.Generator().manual_seed(9))

    def run(seed_shift):
        spec = _spec(rows, k=k)
        spec._seed[1:] += seed_shift
        spec._draft_probs[1:] = 1.0 / VOCAB
        spec._tokens[1:] = torch.tensor([3, 4, 5])
        out = []
        for _ in range(5):
            spec._seq_len[1:] = 10
            out.append(_step(spec, logits, k).tokens.clone())
        return torch.cat(out)

    assert torch.equal(run(0), run(0))
    assert not torch.equal(run(0), run(1000))


def test_a_new_request_on_a_used_register_starts_from_its_own_seed():
    spec = _spec(1, k=1)
    spec._rng_offset[1] = RNG_START + 500
    spec._draft_probs[1] = 1.0 / VOCAB
    image = spec._current.image
    image.seeds[:1] = 77
    image.params[0], image.params[1] = 0.5, 0.25
    image.top_k[:1] = 9
    spec._current = SpecDecodePlan(phase=SpecPhase.SEED, num_rows=1, image=image)
    spec.seed(torch.tensor([1]), torch.tensor([2]))
    assert (int(spec._seed[1]), int(spec._rng_offset[1]), int(spec._top_k[1])) == (77, RNG_START, 9)
    assert spec._draft_probs[1].sum() == pytest.approx(sampled.DRAFT_MARGIN)
    assert spec._draft_probs[1, 0, 2] == sampled.DRAFT_MARGIN


def test_a_sampling_node_declares_its_vocabulary():
    with pytest.raises(ValueError, match="vocab_size"):
        SpecDecodeConfig(num_draft_tokens=2, sampling=True)
    SpecDecodeConfig(num_draft_tokens=2, sampling=True, vocab_size=8)
    SpecDecodeConfig(num_draft_tokens=2)
