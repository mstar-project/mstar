"""A draft model inside the verifying node, through the engine's own step path on CPU: the
toy trunk verifies the drafts of a separate tiny LLM (``draft_model_toy``) that keeps its own
KV, positions and attention, one token behind the target's.

Greedy verify must emit what plain greedy decode emits; the drafter must draft at every step
what a drafter recomputed from scratch drafts, which only holds if the verdict rewound its KV
and positions; a drafter with the target's weights drafts every token; at temperature > 0
the stream must follow the target's distribution with the drafter's sampled drafts and their
distributions; both KVs stay inside their footprints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_model_toy as dm  # noqa: E402
import draft_verify_toy as toy  # noqa: E402
import test_draft_verify_sampled as sampled  # noqa: E402

from mstar.engine.resources import SamplingReqConfig  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager  # noqa: E402
from mstar.engine.resources.position.manager import RopeManager  # noqa: E402
from mstar.model.components.draft_model import DraftModel  # noqa: E402

CPU = torch.device("cpu")
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25], [60, 2, 33]]
MAX_TOKENS = 16
# a drafter whose drafts the toy target accepts at every count from 0 to 3 (at k = 3)
DRAFTER = 3


@pytest.fixture(autouse=True)
def _threads():
    # the comparisons are exact; keep the CPU reductions' order fixed
    torch.set_num_threads(2)


def _run(k: int, drafter_seed: int = DRAFTER, batched: bool = True, max_tokens: int = MAX_TOKENS,
         prompts=PROMPTS, ignore_eos: bool = False, corrupt: int = 0, check_lag: bool = True,
         extra_step: bool = False):
    sub = dm.ToyDraftModelSubmodule(k=k, drafter_seed=drafter_seed, corrupt=corrupt)
    eng = dm.engine(sub, CPU)
    drv = dm.Driver(eng, sub)
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, max_tokens, ignore_eos=ignore_eos)
    kv, dkv = eng._resources[toy.KV], eng._resources[dm.DKV]
    pos, dpos = eng._resources[toy.POS], eng._resources[dm.DPOS]

    def lags():
        # the drafter holds every kept token but the last, before and after each rewind
        for rid in range(len(prompts)):
            stored = kv._streams[rid]["main"].stored_len
            assert dkv._streams[rid]["main"].stored_len == stored - 1
            assert dpos.position(rid, "main") == pos.position(rid, "main") - 1 == stored - 1

    if check_lag:
        real = drv.decode

        def decode(rids):
            lags()
            real(rids)
            lags()

        drv.decode = decode
    out = drv.run(list(range(len(prompts))), batched=batched)
    if extra_step:
        before = {rid: list(tokens) for rid, tokens in out.items()}
        drv.decode(list(out))
        assert all(drv.emitted[rid] == before[rid] for rid in out), "a step after the stop emitted"
    return out, eng, drv


def _reference(prompts=PROMPTS, max_tokens=MAX_TOKENS, ignore_eos=False):
    return {rid: toy.reference(p, max_tokens, ignore_eos=ignore_eos) for rid, p in enumerate(prompts)}


@pytest.mark.parametrize("batched", [True, False], ids=["batched", "one_per_step"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_greedy_verify_of_a_draft_model_emits_what_plain_decode_emits(k, batched):
    out, eng, _ = _run(k, batched=batched)
    assert out == _reference()
    dv = eng._resources[toy.DV]
    # partly right, so both sides of every rewind ran
    assert 0 < dv.drafts_accepted < k * dv.rows_verified, dv.stats()


@pytest.mark.parametrize("k", [1, 3, 5])
def test_the_drafter_drafts_what_a_drafter_without_a_cache_drafts(k):
    """Each verify step accepts as many drafts as a float64 drafter recomputed from scratch
    over the kept tokens gets right: a stale or misplaced row in the drafter's KV, after a
    rejection or after a full acceptance left its last row unwritten, changes its drafts."""
    prompts = PROMPTS[:2]
    _, eng, _ = _run(k, prompts=prompts, ignore_eos=True)
    want = [dm.drafter_reference(p, k, DRAFTER, MAX_TOKENS) for p in prompts]
    hist = torch.bincount(torch.tensor([a for steps in want for a in steps]), minlength=k + 1)
    dv = eng._resources[toy.DV]
    assert dv.accepted_hist.tolist() == hist.tolist(), (dv.accepted_hist, hist)
    assert hist[0] > 0 and hist[1:].sum() > 0


def test_without_the_drafters_kv_rewind_its_drafts_go_wrong(monkeypatch):
    """The check above is not vacuous: skip the drafter's KV rewind (the target's still runs)
    and its drafts leave the from-scratch drafter's, while the stream stays the target's."""
    k, prompts = 3, PROMPTS[:2]
    real = kv_manager.KVManager.rewind

    def rewind(self, verdicts):
        if self.name != dm.DKV:
            real(self, verdicts)

    monkeypatch.setattr(kv_manager.KVManager, "rewind", rewind)
    out, eng, _ = _run(k, prompts=prompts, ignore_eos=True, check_lag=False)
    assert out == _reference(prompts, ignore_eos=True)
    want = [dm.drafter_reference(p, k, DRAFTER, MAX_TOKENS) for p in prompts]
    hist = torch.bincount(torch.tensor([a for steps in want for a in steps]), minlength=k + 1)
    assert eng._resources[toy.DV].accepted_hist.tolist() != hist.tolist()


def test_without_the_drafters_position_rewind_its_positions_run_ahead(monkeypatch):
    """The lag checks in ``_run`` are not vacuous: skip the drafter's position rewind and its
    positions leave its KV's length. (The toy's drafts barely depend on positions at this
    seed, so the drafts are no witness here.)"""
    real = RopeManager.rewind

    def rewind(self, verdicts):
        if self._config.kv_cache != dm.DKV:
            real(self, verdicts)

    monkeypatch.setattr(RopeManager, "rewind", rewind)
    with pytest.raises(AssertionError):
        _run(3, prompts=PROMPTS[:2], ignore_eos=True)
    _, eng, _ = _run(3, prompts=PROMPTS[:2], ignore_eos=True, check_lag=False)
    dkv, dpos = eng._resources[dm.DKV], eng._resources[dm.DPOS]
    assert any(dpos.position(rid, "main") != dkv._streams[rid]["main"].stored_len for rid in range(2))


def test_a_drafter_with_the_targets_weights_drafts_every_token():
    """The target drafting for itself: every verify step accepts all k drafts, which holds
    only if the drafter's block, positions and catch-up row line up with the target's."""
    k = 3
    out, eng, _ = _run(k, drafter_seed=None, ignore_eos=True)
    assert out == _reference(ignore_eos=True)
    dv = eng._resources[toy.DV]
    assert dv.drafts_accepted == k * dv.rows_verified, dv.stats()


def test_after_a_rejected_draft_the_drafter_drafts_right_again():
    """The target drafting for itself with a draft swapped at every position divisible by 4:
    a step accepts every draft before the first swapped one and none after, so the drafter
    is right again from the token after each rejection."""
    k, corrupt = 3, 4
    prompts = PROMPTS
    out, eng, _ = _run(k, drafter_seed=None, ignore_eos=True, corrupt=corrupt, prompts=prompts)
    assert out == _reference(prompts, ignore_eos=True)
    want = []
    for prompt in prompts:
        kept = len(prompt) + 1
        while kept - len(prompt) < MAX_TOKENS:
            good = 0
            while good < k and (kept + good) % corrupt != 0:
                good += 1
            want.append(good)
            kept += good + 1
    hist = torch.bincount(torch.tensor(want), minlength=k + 1)
    assert eng._resources[toy.DV].accepted_hist.tolist() == hist.tolist()


@pytest.mark.parametrize("max_tokens", [1, 2, 5])
def test_max_tokens_cuts_inside_a_block(max_tokens):
    out, _, _ = _run(3, drafter_seed=None, max_tokens=max_tokens)
    assert out == _reference(max_tokens=max_tokens)


def test_both_kvs_stay_inside_their_footprints(monkeypatch):
    """To the cap, then the async worker's one more step: neither KV holds more pages than
    its layout's footprint, and every commit stays under the layout's bound (the debug
    assert); the drafter's is the target's less the prompt's last token."""
    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    seen = []
    real = kv_manager.KVManager.commit

    def commit(self, step, ctx):
        real(self, step, ctx)
        for rid in ctx.request_ids:
            stream = self._streams[rid]["main"]
            seen.append((self.name, len(stream.page_indices), stream.footprint))

    monkeypatch.setattr(kv_manager.KVManager, "commit", commit)
    out, eng, _ = _run(3, drafter_seed=None, ignore_eos=True, extra_step=True, check_lag=False)
    assert out == _reference(ignore_eos=True)
    assert {name for name, _, _ in seen} == {toy.KV, dm.DKV}
    assert all(held <= footprint for _, held, footprint in seen), max(seen, key=lambda s: s[1] - s[2])


def test_a_drafter_layout_without_the_lookahead_is_refused():
    sub = dm.ToyDraftModelSubmodule(k=3)
    eng = dm.engine(sub, CPU)
    drv = dm.Driver(eng, sub, draft_lookahead=3)
    with pytest.raises(ValueError, match="lookahead of 3, not 7"):
        drv.add(0, [3, 9, 12], 8)


def test_the_declaration():
    decl = DraftModel(max_drafts=3, kv=dm.draft_model(3).kv)
    with pytest.raises(ValueError, match="at least 2"):
        decl.check_prompt(1)
    decl.check_prompt(2)
    with pytest.raises(ValueError, match="at least 1 token"):
        DraftModel(max_drafts=0, kv=decl.kv)
    layout = dm.draft_layout(decl, 10, 8, 7).kv_layout["main"]
    assert [span.length for span in layout.spans] == [9]
    assert layout.lookahead == 7 and layout.growth == (("decode_loop", 1),)
    steps = decl.prefill_steps([4, 5], [10, 2])
    assert [seg.span for seg in steps[dm.DKV].segments] == [9, 1]
    assert {key for key in decl.verify_steps()} == {dm.DKV, dm.DATTN, dm.DPOS}
    assert {spec.resource_key for spec in decl.node_resources({"llm"})} == {dm.DKV, dm.DATTN, dm.DPOS}


def test_a_request_a_greedy_verify_cannot_serve_is_refused():
    sub = dm.ToyDraftModelSubmodule(k=2)
    eng = dm.engine(sub, CPU)
    with pytest.raises(ValueError, match="greedy-only"):
        eng.add_request(9, {toy.SAMPLER: SamplingReqConfig(temperature=0.7)})


# ── temperature > 0: the sampled verify (DV-2) with the drafter's distributions ──


def _sampled_streams(k: int, sample: bool, n: int, chunk: int = 256, drafter_seed: int = DRAFTER,
                     max_tokens: int = 3):
    out, engines = {}, []
    for start in range(0, n, chunk):
        sub = dm.ToyDraftModelSubmodule(k=k, drafter_seed=drafter_seed, sample=sample, max_num_pages=2048)
        eng = dm.engine(sub, CPU, sampled_verify=True, max_num_pages=2048)
        drv = dm.Driver(eng, sub)
        rids = list(range(start, min(start + chunk, n)))
        drv.add_batch({rid: sampled.PROMPT for rid in rids}, max_tokens, ignore_eos=True,
                      seeds={rid: 1 + rid for rid in rids},
                      temperature=sampled.TEMPERATURE, top_k=sampled.TOP_K)
        out.update(drv.run(rids, max_steps=400))
        engines.append(eng)
    return out, engines


@pytest.mark.parametrize(("k", "sample"), [(2, True), (3, False)], ids=["sampled_drafts", "argmax_drafts"])
def test_the_sampled_stream_follows_the_target(k, sample):
    """The joint law of the two tokens after the prefill's, over requests with their own
    seeds, against the float64 target: with the drafter's drafts sampled as the request
    samples and verified against their distributions, and with its argmax drafts."""
    n = 1536
    streams, engines = _sampled_streams(k, sample, n)
    t1, counts = sampled._first_two(streams)
    assert sampled._chi2(counts, sampled._expected(t1, n)) > sampled.P_MIN, counts
    rows = sum(eng._resources[toy.DV].rows_verified for eng in engines)
    accepted = sum(eng._resources[toy.DV].drafts_accepted for eng in engines)
    assert 0 < accepted < k * rows


def test_wrong_draft_distributions_are_caught(monkeypatch):
    """Not vacuous: hand verify a distribution other than the one the drafts came from (the
    uniform one) and the stream leaves the target's law."""
    from mstar.model.components import draft_model as module

    real = module.DraftLLMProposer.propose

    def propose(self, inputs, dv, request_ids):
        drafts = real(self, inputs, dv, request_ids)
        return drafts._replace(probs=torch.full_like(drafts.probs, 1.0 / drafts.probs.shape[-1]))

    monkeypatch.setattr(module.DraftLLMProposer, "propose", propose)
    n = 1536
    streams, _ = _sampled_streams(2, True, n)
    t1, counts = sampled._first_two(streams)
    try:
        p = sampled._chi2(counts, sampled._expected(t1, n))
    except AssertionError:  # a token the target can't produce
        p = 0.0
    assert p < 1e-6
