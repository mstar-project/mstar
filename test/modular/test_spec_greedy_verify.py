"""Speculative decoding: a verify step whose rows all sample greedily reads only the target's argmax
(`greedy_verify`), in a lean twin of each verify bucket (`SamplerResource.lean_captures`), and gives
exactly the tokens and acceptance the full path (`verify_probs`, `chain_verify_rows`) gives."""
from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.sampler.verify import chain_verify_rows, greedy_verify, verify_probs  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    from mstar.engine.resources.kv import manager as manager_mod

    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _full(logits: torch.Tensor, drafts: torch.Tensor):
    """The full path for greedy rows: temperature 0, no filters, at any seed and offset."""
    n, block, vocab = logits.shape
    rows = n * block
    dist = verify_probs(
        logits.reshape(rows, vocab), torch.zeros(rows, device=logits.device),
        torch.zeros(rows, dtype=torch.int32, device=logits.device), torch.ones(rows, device=logits.device),
    )
    return chain_verify_rows(dist, n, drafts, torch.arange(n, device=logits.device),
                             torch.zeros(n, dtype=torch.long, device=logits.device))


def _case(n: int, k: int, vocab: int, dtype, accepting: str, ties: bool, seed: int, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, k + 1, vocab, generator=g).to(dtype)
    if ties:
        # every row's maximum held by two to four tokens, the lowest of them anywhere
        for i in range(n):
            for j in range(k + 1):
                at = torch.randperm(vocab, generator=g)[: int(torch.randint(2, 5, (), generator=g))]
                logits[i, j, at] = logits[i, j].max() + 1
    top = logits.float().argmax(-1)
    drafts = top[:, :k].clone()
    if accepting == "none":
        drafts[:, 0] = (drafts[:, 0] + 1) % vocab
    elif accepting == "mixed":
        # row i refuses at a random position, or accepts all
        for i in range(n):
            at = int(torch.randint(0, k + 1, (), generator=g))
            if at < k:
                drafts[i, at] = (drafts[i, at] + 1 + int(torch.randint(0, vocab - 1, (), generator=g))) % vocab
                drafts[i, at + 1:] = torch.randint(0, vocab, (k - at - 1,), generator=g)
    if ties:
        # some drafts the other tokens tied at the top: refused, since the lowest index wins
        for i in range(0, n, 3):
            tied = (logits[i, 0] == logits[i, 0].max()).nonzero().flatten()
            drafts[i, 0] = tied[-1]
    return logits.to(device), drafts.to(device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [3, 7])
@pytest.mark.parametrize("n", [1, 2, 7, 64])
@pytest.mark.parametrize("accepting", ["all", "none", "mixed"])
@pytest.mark.parametrize("ties", [False, True])
def test_greedy_verify_is_the_full_path_exactly(dtype, k, n, accepting, ties):
    logits, drafts = _case(n, k, 97, dtype, accepting, ties, seed=n * 31 + k)
    tokens, accepted = greedy_verify(logits.reshape(-1, logits.shape[-1]), drafts)
    want_tokens, want_accepted = _full(logits, drafts)
    assert torch.equal(accepted, want_accepted), (accepted, want_accepted)
    assert torch.equal(tokens, want_tokens), (tokens, want_tokens)
    assert tokens.dtype == want_tokens.dtype and accepted.dtype == want_accepted.dtype
    if accepting == "all" and not ties:
        assert accepted.tolist() == [k] * n
    if accepting == "none":
        assert accepted.tolist() == [0] * n


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer's filters and the fused softmax run on CUDA",
)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [97, 151936])
@pytest.mark.parametrize("ties", [False, True])
def test_greedy_verify_is_the_full_path_exactly_on_cuda(vocab, ties, dtype):
    """Past `FUSED_FILTER_MAX_VOCAB` the full path is the fused softmax's one-hot and FlashInfer's renorms."""
    for n, k in ((1, 3), (16, 3), (64, 7)):
        logits, drafts = _case(n, k, vocab, dtype, "mixed", ties, seed=vocab + n, device="cuda")
        tokens, accepted = greedy_verify(logits.reshape(-1, vocab), drafts)
        want_tokens, want_accepted = _full(logits, drafts)
        assert torch.equal(accepted, want_accepted) and torch.equal(tokens, want_tokens), (n, k)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [97, 4096, 151936])
def test_the_fused_softmaxs_greedy_one_hot_sits_where_greedy_tokens_does(vocab, dtype):
    """Rows whose maximum two to four tokens share: the fused kernel's one-hot at temperature 0 lands on
    the lowest of them, the token `greedy_tokens` (and so `greedy_verify`) reads."""
    from mstar.engine.resources.sampler.utils import fused_temperature_softmax, greedy_tokens

    logits, _ = _case(32, 3, vocab, dtype, "all", True, seed=vocab, device="cuda")
    logits = logits.reshape(-1, vocab)
    probs = fused_temperature_softmax(logits, torch.zeros(logits.shape[0], device="cuda"), include_greedy=True)
    hot = probs.argmax(-1)
    assert torch.equal(probs.max(-1).values, torch.ones_like(probs[:, 0])), "not one-hot"
    assert torch.equal(hot, greedy_tokens(logits)), (hot, greedy_tokens(logits))
    tied = logits == logits.max(-1, keepdim=True).values
    assert torch.equal(hot, tied.int().argmax(-1)), "not the lowest of the tied maxima"


# ── which path a step takes ─────────────────────────────────────────────


def test_a_verifying_sampler_captures_lean_twins_of_batched_buckets_only():
    from mstar.engine.accelerator_graph_config import AcceleratorGraphConfigType
    from mstar.engine.resources.sampler.resource import SamplerResource
    from mstar.engine.resources.step import BucketKey

    sampler = SamplerResource(vocab_size=None, enable_repetition_penalty=False, device=torch.device("cpu"),
                              greedy_capture=True)
    sampler.bind_verify(3)
    assert sampler.lean_capture
    assert sampler.lean_captures(AcceleratorGraphConfigType.BASIC_BATCHED)
    assert not sampler.lean_captures(AcceleratorGraphConfigType.FLASH_INFER_PACKED)
    plain = BucketKey("decode", 4, 16, None)
    assert BucketKey("decode", 4, 16, None, frozenset({"sampler"})) != plain


@pytest.mark.parametrize("config, greedy", [
    (dict(temperature=0.0), True),
    (dict(temperature=0.0, top_p=0.5, top_k=4, min_p=0.2), True),
    (dict(temperature=0.7), False),
    (dict(temperature=0.0, repetition_penalty=1.2), False),
    (dict(temperature=0.0, return_logprobs=True), False),
], ids=["greedy", "greedy-filters-off", "sampled", "penalty", "logprobs"])
def test_a_step_runs_lean_only_when_every_row_samples_greedily(config, greedy):
    """From the configs set at ingest, which every rank of a lockstep instance holds alike: two
    samplers given the same requests in any order agree, row for row and step for step."""
    from mstar.engine.resources.sampler.config import SamplingReqConfig
    from mstar.engine.resources.sampler.resource import SamplerResource

    def sampler(order):
        s = SamplerResource(vocab_size=None, enable_repetition_penalty=False, device=torch.device("cpu"),
                            greedy_capture=True, enable_min_p=True, enable_logprobs=True)
        for rid in order:
            s.ingest_request(rid, SamplingReqConfig(**(config if rid == "b" else dict(temperature=0.0))))
        return s

    ranks = [sampler(["a", "b", "c"]), sampler(["c", "b", "a"])]
    for s in ranks:
        assert s.lean_for(["a", "c"])
        assert s.lean_for(["a", "b"]) is greedy
    assert len({s.lean_for(rows) for s in ranks for rows in (["a"], ["b"], ["a", "b", "c"])}) == (1 if greedy else 2)


def test_a_min_tokens_floor_is_never_greedy():
    """It moves the token off the argmax (a CUDA-only control, so read off the config here)."""
    from types import SimpleNamespace

    from mstar.engine.resources.sampler.utils import samples_greedy

    base = dict(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, return_logprobs=False)
    assert samples_greedy(SimpleNamespace(**base, min_tokens=0))
    assert not samples_greedy(SimpleNamespace(**base, min_tokens=4))


def _counting(monkeypatch):
    import mstar.engine.resources.sampler.verify as verify_mod

    calls = {"greedy": 0, "probs": 0}
    greedy, probs = verify_mod.greedy_verify, verify_mod.verify_probs

    def counted_greedy(*args, **kwargs):
        calls["greedy"] += 1
        return greedy(*args, **kwargs)

    def counted_probs(*args, **kwargs):
        calls["probs"] += 1
        return probs(*args, **kwargs)

    monkeypatch.setattr(verify_mod, "greedy_verify", counted_greedy)
    monkeypatch.setattr(verify_mod, "verify_probs", counted_probs)
    return calls


def _offsets(eng, rids):
    """Each request's sampler counter, which its next draw reads."""
    sampler = eng._resources["sampler"]
    if sampler._cg_buffers is not None:
        return sampler._cg_buffers.eager_offsets(list(rids))[1].tolist()
    return [sampler._sampler._step_offset.get(rid, 0) for rid in rids]


def test_a_lean_step_moves_every_rows_offset_as_a_full_one_does(monkeypatch):
    """The counters a row's draws read move by the same stride on the argmax path as on the full one, step
    for step, so a request later drawn at its counter, or a peer rank's copy of it, draws the same uniforms."""
    from spec_toy import Driver, engine, oracle

    calls = _counting(monkeypatch)
    prompts = {1: [5, 9, 2], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    seen = {}
    for lean in (False, True):
        driver = Driver(engine(3, oracle, greedy_capture=lean))
        for rid, prompt in prompts.items():
            driver.add(rid, prompt, 12)
        steps = [_offsets(driver.eng, prompts)]
        live = list(prompts)
        while live:
            stopped = driver.step(live)
            steps.append(_offsets(driver.eng, prompts))
            live = [rid for rid in live if not stopped[rid]]
        seen[lean] = steps
    assert calls["greedy"] > 0 and calls["probs"] > 0
    assert seen[True] == seen[False]
    # and they did move, every step
    assert all(b != a for a, b in zip(seen[False], seen[False][1:], strict=False))


def test_a_step_with_one_sampled_row_verifies_the_full_way_and_keys_the_plain_bucket(monkeypatch):
    """A sampled row beside greedy ones: its step leaves no resource lean (`lean_of`), so it verifies by the
    probs eagerly, and a captured step tries the plain bucket alone, keyed apart from the twin; once only
    greedy rows run, the step is lean again."""
    from spec_toy import NODE, Driver, engine, oracle

    from mstar.engine.resources.step import BucketKey

    calls = _counting(monkeypatch)
    driver = Driver(engine(3, oracle, greedy_capture=True))
    mgmt = driver.eng._submodules[NODE]
    driver.add(1, [5, 9, 2], 30)
    driver.add(2, [17, 4], 6, temperature=0.8, seed=3)
    assert mgmt.lean_of([1]) == frozenset({"sampler"}) and mgmt.lean_of([1, 2]) == frozenset()
    # no twin to try: a captured step leases the plain bucket alone
    assert mgmt.lean_candidates(frozenset()) == ()
    assert mgmt.lean_candidates(frozenset({"sampler"})) == (frozenset({"sampler"}),)
    assert BucketKey("decode", 2, 8, None, frozenset()) != BucketKey("decode", 2, 8, None, frozenset({"sampler"}))
    paths = []
    live = [1, 2]
    while live:
        before = dict(calls)
        stopped = driver.step(live)
        paths.append((tuple(live), "greedy" if calls["greedy"] > before["greedy"] else "probs"))
        assert calls["greedy"] + calls["probs"] == before["greedy"] + before["probs"] + 1
        live = [rid for rid in live if not stopped[rid]]
    assert all(path == "probs" for rows, path in paths if 2 in rows), paths
    assert any(path == "greedy" for rows, path in paths if rows == (1,)), paths


def test_a_lean_verify_step_reads_the_argmax_and_a_full_one_the_probs(monkeypatch):
    """Through the real engine's verify steps on the toy: a lean sampler takes `greedy_verify` and
    never builds the vocabulary's probs; a plain one never takes it. The tokens, acceptance and
    offsets agree (`test_spec_engine`'s lean test checks the whole run)."""
    from spec_toy import Driver, engine, oracle

    calls = _counting(monkeypatch)
    outs = {}
    for lean in (False, True):
        calls.update(greedy=0, probs=0)
        driver = Driver(engine(3, oracle, greedy_capture=lean))
        prompts = {1: [5, 9, 2], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        for rid, prompt in prompts.items():
            driver.add(rid, prompt, 12)
        driver.run(list(prompts))
        outs[lean] = driver.out
        assert (calls["greedy"] > 0, calls["probs"] > 0) == (lean, not lean), calls
    assert outs[True] == outs[False]
