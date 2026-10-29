"""Draft-verify at temperature > 0 through the engine's step path, on CPU: the toy of
``draft_verify_toy.py`` on a sampler with ``enable_sampled_verify``.

A sampled verify's stream must be distributed as sampling the target one token at a time: the
first tokens of many requests, each with its own seed, against the float64 trunk's filtered
distributions, for drafters that are partly right, nearly always wrong, or sample. The rewinds
are what keep the later tokens' distribution right (skip one and the check fails). A request's
seed reproduces its stream in any batch, and the streams stay inside their KV footprint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402

from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402

CPU = torch.device("cpu")
PROMPT = [3, 9, 12, 40]
TEMPERATURE, TOP_K = 1.0, 4
# a chi-square p-value under this fails the test; the seeds are fixed, so a pass is stable
P_MIN = 1e-4


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


def _target(ids: list[int], seed: int = toy.SEED) -> torch.Tensor:
    """The float64 trunk's distribution of the token after ``ids`` as the request samples it:
    softmax at the temperature, the top k renormalised."""
    probs = torch.softmax(toy.next_logits(ids, seed=seed) / TEMPERATURE, -1)
    top = probs.topk(TOP_K)
    out = torch.zeros_like(probs)
    out[top.indices] = top.values / top.values.sum()
    return out


def _chi2(observed: torch.Tensor, expected: torch.Tensor) -> float:
    observed, expected = observed.double().flatten(), expected.double().flatten()
    impossible = expected < 1e-9
    assert observed[impossible].sum() == 0, "a stream emitted a token the target can't produce"
    small = (expected < 5) & ~impossible
    obs = torch.cat([observed[~small & ~impossible], observed[small].sum()[None]])
    exp = torch.cat([expected[~small & ~impossible], expected[small].sum()[None]])
    if exp[-1] < 1e-9:
        obs, exp = obs[:-1], exp[:-1]
    stat = ((obs - exp) ** 2 / exp).sum()
    return float(torch.special.gammaincc(torch.tensor((len(obs) - 1) / 2, dtype=torch.float64), stat / 2))


def _streams(k: int, drafter: str, n: int, chunk: int = 256, first_seed: int = 1, max_tokens: int = 3,
             toy_seed: int = toy.SEED, **sampling) -> tuple[dict[int, list[int]], list]:
    """``n`` requests of ``PROMPT``, each its own seed, ``chunk`` at a time on a fresh engine."""
    out, engines = {}, []
    sampling = {"temperature": TEMPERATURE, "top_k": TOP_K, **sampling}
    for start in range(0, n, chunk):
        sub = toy.ToySubmodule(
            k=k, corrupt=1 if drafter == "wrong" else 0, sampled_drafts=drafter == "sampled", seed=toy_seed,
        )
        eng = toy.engine(sub, CPU, sampled_verify=True, max_num_pages=2048)
        drv = toy.Driver(eng, sub)
        rids = list(range(start, min(start + chunk, n)))
        drv.add_batch({rid: PROMPT for rid in rids}, max_tokens, ignore_eos=True,
                      seeds={rid: first_seed + rid for rid in rids}, **sampling)
        out.update(drv.run(rids))
        engines.append(eng)
    return out, engines


def _first_two(streams: dict[int, list[int]]) -> tuple[int, torch.Tensor]:
    """The prefill's token (the argmax on CPU, so the same for every request) and the counts of
    the two tokens after it, which verify steps emitted."""
    firsts = {tokens[0] for tokens in streams.values()}
    assert len(firsts) == 1
    counts = torch.zeros(toy.VOCAB, toy.VOCAB, dtype=torch.long)
    for tokens in streams.values():
        counts[tokens[1], tokens[2]] += 1
    return firsts.pop(), counts


def _expected(t1: int, n: int, seed: int = toy.SEED) -> torch.Tensor:
    p2 = _target(PROMPT + [t1], seed)
    expected = torch.zeros(toy.VOCAB, toy.VOCAB, dtype=torch.float64)
    for t2 in p2.nonzero().flatten().tolist():
        expected[t2] = p2[t2] * _target(PROMPT + [t1, t2], seed)
    return expected * n


@pytest.mark.parametrize(
    ("k", "drafter", "toy_seed"),
    [(1, "argmax", toy.SEED), (3, "argmax", 0), (3, "wrong", toy.SEED), (2, "sampled", toy.SEED)],
)
def test_the_stream_follows_the_target(k, drafter, toy_seed):
    """The joint distribution of the two tokens after the prefill's, over requests with their
    own seeds. ``argmax`` drafts the toy head's likeliest tokens, ``wrong`` its second-likeliest
    (often outside the target's top k), ``sampled`` draws them from the head (``draft_sample``).
    The toy's weight seed 0 makes the trunk lean on its attention (see the KV check below)."""
    n = 1536
    streams, engines = _streams(k, drafter, n, toy_seed=toy_seed)
    t1, counts = _first_two(streams)
    assert t1 == int(toy.next_logits(PROMPT, seed=toy_seed).argmax())
    assert _chi2(counts, _expected(t1, n, toy_seed)) > P_MIN, counts
    rows = sum(eng._resources[toy.DV].rows_verified for eng in engines)
    accepted = sum(eng._resources[toy.DV].drafts_accepted for eng in engines)
    # the drafters are partly right, so the rewinds ran on both sides of a verdict
    assert 0 < accepted < k * rows


def test_the_check_needs_the_kv_rewind(monkeypatch):
    """Not vacuous: skip the KV rewind and a token after a rejection reads the rejected drafts'
    rows, and the pair's distribution leaves the target's. On weight seed 0, whose trunk leans
    on its attention (on the default seed, a stale row barely moves the distribution)."""
    monkeypatch.setattr(kv_manager.KVManager, "rewind", lambda self, verdicts: None)
    n = 1536
    streams, _ = _streams(3, "argmax", n, toy_seed=0)
    t1, counts = _first_two(streams)
    try:
        p = _chi2(counts, _expected(t1, n, seed=0))
    except AssertionError:  # a token the target can't produce
        p = 0.0
    assert p < 1e-6


def test_greedy_requests_on_a_sampled_node_emit_the_greedy_stream():
    """A temperature-0 request goes through the sampling kernel as a one-hot target and keeps
    exactly what a greedy verify keeps."""
    prompts = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25], [60, 2, 33]]
    sub = toy.ToySubmodule(k=3, corrupt=2)
    eng = toy.engine(sub, CPU, sampled_verify=True)
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, 16, seed=rid)
    out = drv.run(list(range(len(prompts))))
    assert out == {rid: toy.reference(prompt, 16) for rid, prompt in enumerate(prompts)}


def _seeded(seeds: list[int], batched: bool, k: int = 3, drafter: str = "argmax"):
    sub = toy.ToySubmodule(k=k, corrupt=2, sampled_drafts=drafter == "sampled")
    eng = toy.engine(sub, CPU, sampled_verify=True)
    drv = toy.Driver(eng, sub)
    prompts = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25], [60, 2, 33]]
    for rid, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
        drv.add(rid, prompt, 16, seed=seed, temperature=0.9, top_k=8)
    return drv.run(list(range(len(prompts))), batched=batched), eng


@pytest.mark.parametrize("drafter", ["argmax", "sampled"])
def test_a_request_seed_reproduces_its_stream_in_any_batch(drafter):
    """The same seeds give the same streams, batched or one request a step (each row draws from
    its own seed and offset on CPU); other seeds give other streams. Every stream ends at the
    stop token or at max_tokens, whichever row of a block it lands on."""
    seeds = [101, 202, 303, 404]
    batched, eng = _seeded(seeds, True, drafter=drafter)
    alone, _ = _seeded(seeds, False, drafter=drafter)
    assert batched == alone
    other, _ = _seeded([s + 1 for s in seeds], True, drafter=drafter)
    assert other != batched
    for tokens in list(batched.values()) + list(other.values()):
        assert len(tokens) <= 16
        assert toy.STOP not in tokens[:-1]
        assert tokens[-1] == toy.STOP or len(tokens) == 16
    dv = eng._resources[toy.DV]
    assert 0 < dv.drafts_accepted < 3 * dv.rows_verified


@pytest.mark.parametrize("drafter", ["argmax", "sampled"])
def test_sampled_streams_stay_inside_their_footprint(drafter, monkeypatch):
    """To the cap, then the async worker's one more step: the KV holds no more pages than its
    footprint, every commit stays under the layout's bound, and positions follow lengths."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    sub = toy.ToySubmodule(k=3, sampled_drafts=drafter == "sampled")
    eng = toy.engine(sub, CPU, sampled_verify=True)
    drv = toy.Driver(eng, sub)
    rids = list(range(8))
    drv.add_batch({rid: PROMPT for rid in rids}, 20, ignore_eos=True,
                  seeds={rid: 50 + rid for rid in rids}, temperature=0.9, top_k=8)
    out = drv.run(rids)
    drv.decode(rids)
    assert all(len(tokens) == 20 for tokens in out.values())
    kv, pos = eng._resources[toy.KV], eng._resources[toy.POS]
    for rid in rids:
        stream = kv._streams[rid]["main"]
        assert len(stream.page_indices) <= stream.footprint
        assert stream.stored_len == pos.position(rid, "main")
