"""Prompt-lookup drafting (``NgramProposer``) on the draft-verify engine core, on CPU.

The index on its own: the longest suffix that ended earlier wins, its last earlier end wins,
and the drafts repeat a stretch that just repeated; tokens appended a few at a time index the
same as all at once; a row's next owner never sees the last one's entries.

Through the engine (the toy of ``draft_verify_toy.py``, drafting with the n-gram proposer
instead of its head): greedy verify emits exactly what plain greedy decode emits, the proposer's
history is the prompt and the emitted tokens, and it is purged with the request. At
temperature > 0 the stream follows the target (the check of ``test_draft_verify_sampled.py``
with n-gram drafts).

Adaptive (``DraftPolicy.adaptive``): a step drafts only when the proposer expects its rows to
keep enough drafts, else runs one token a row; the streams are plain decode's whichever steps
draft, the choice moves only at a commit, and a step of no drafts leaves its verdict unread
until the stop check (its tokens) and the row's next commit (the proposer's history).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402
import test_draft_verify_sampled as sampled  # noqa: E402

from mstar.engine.resources import NGRAM, DraftConfig, DraftPolicy, ProposeStep  # noqa: E402
from mstar.engine.resources.draft_verify.ngram import History, NgramIndex, NgramProposer  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402

CPU = torch.device("cpu")
MAX_TOKENS = 24


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


# ── the index ──


def _index(k=3, min_n=2, max_n=4) -> NgramIndex:
    return NgramIndex(k, min_n, max_n)


def _fill(index: NgramIndex, tokens: list[int], fresh: bool = False) -> History:
    history = index.open()
    index.extend(history, list(tokens))
    history.fresh = fresh
    return history


def _drafts(index: NgramIndex, history: History, follow: bool = True) -> list[int]:
    """The row's drafts, filler where nothing matched; ``follow=False`` looks them up afresh,
    as a row that follows no stretch would."""
    if not follow:
        history.follow = -1
    return index.propose(history) or [0] * index.k


def test_the_longest_suffix_that_ended_before_wins():
    # suffix [8, 2, 3]: the 3-gram ended at 3 (then 9 ...); the 2-gram [2, 3] last ended at 8
    index = _index(k=2)
    row = _fill(index, [8, 2, 3, 9, 9, 7, 2, 3, 6, 6, 8, 2, 3])
    assert _drafts(index, row) == [9, 9]


def test_the_last_earlier_end_wins_and_short_suffixes_match_down_to_min():
    index = _index(k=2, min_n=1, max_n=3)
    # [5] ended at 1 (then 6) and at 3 (then 7); no 2-gram ending in 5 occurred before
    row = _fill(index, [5, 6, 5, 7, 1, 5])
    assert _drafts(index, row) == [7, 1]
    index2 = _index(k=2, min_n=2, max_n=3)
    assert index2.propose(_fill(index2, [5, 6, 5, 7, 1, 5])) is None


def test_a_stretch_that_repeated_repeats_past_the_end():
    index = _index(k=5, min_n=2)
    row = _fill(index, [1, 4, 5, 6, 4, 5, 6, 4, 5])
    # [4, 5] last ended at 6 (then 6, 4, 5, 6 ...): the period 4, 5, 6 continues
    assert _drafts(index, row) == [6, 4, 5, 6, 4]


def test_a_fresh_row_drafts_past_the_input_it_does_not_hold_yet():
    # the prefill's token is the step's input, not yet in the history: assume the match's next
    index = _index(k=3)
    row = _fill(index, [7, 1, 2, 3, 4, 5, 7, 1], fresh=True)
    assert _drafts(index, row) == [3, 4, 5]


def _brute(tokens: list[int], k: int, min_n: int = 2, max_n: int = 4) -> list[int]:
    """What the index answers, by search: the last earlier end of the longest suffix, then the
    stretch from there repeated."""
    length = len(tokens)
    for n in range(max_n, min_n - 1, -1):
        suffix = tokens[length - n:] if length >= n else None
        ends = [j for j in range(n, length) if tokens[j - n:j] == suffix]
        if suffix is not None and ends:
            best = ends[-1]
            return [tokens[best + i % (length - best)] for i in range(k)]
    return [0] * k


def test_appending_a_few_at_a_time_indexes_as_appending_at_once():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 6, size=300).tolist()
    index = _index(k=4)
    steps = _fill(index, [])
    at = 0
    while at < len(tokens):
        chunk = tokens[at:at + int(rng.integers(1, 6))]
        index.extend(steps, chunk)
        at += len(chunk)
        assert _drafts(index, steps, follow=False) == _brute(tokens[:at], 4)
    whole = _fill(index, tokens)
    assert _drafts(index, whole, follow=False) == _drafts(index, steps, follow=False)
    assert steps.tokens == whole.tokens == tokens
    assert steps.ends == whole.ends


def test_a_row_keeps_following_the_stretch_it_copies():
    # [7, 7] occurs twice in the source; the reply is copying the first
    source = [10, 11, 7, 7, 12, 13, 7, 7, 14, 15]
    index = _index(k=4, min_n=2, max_n=2)
    row = _fill(index, source + [10, 11])
    assert _drafts(index, row) == [7, 7, 12, 13]
    index.extend(row, [7, 7])
    # still on the first stretch, though [7, 7] last ended before the 14
    assert _drafts(index, row) == [12, 13, 7, 7]
    assert _drafts(index, row, follow=False) == [14, 15, 10, 11]
    # a token off the stretch: looked up afresh
    row.follow = 4
    index.extend(row, [99])
    assert row.follow == -1
    assert index.propose(row) is None


# ── the resource ──


def _ctx(rids, preplan=False, slot=0) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="decode", slot=slot, capture=False, is_preplan=preplan)


def test_the_resource_takes_prompt_then_verdicts_and_purges():
    prop = NgramProposer(CPU, max_drafts=2, min_ngram=2, max_ngram=3)
    prop.ingest_request(7)
    prompt = torch.tensor([4, 1, 2, 9, 9, 4, 1])
    prop.commit(ProposeStep(history={7: prompt}), _ctx([7]))
    # the first verdict: input 2 (the prefill's), drafts [9, 9] both kept, bonus 4
    prop.observe(7, np.array([2, 2, 9, 9, 4]))
    prop.plan(ProposeStep(drafts=2), _ctx([7]))
    history = prop._histories[7]
    assert history.tokens == [4, 1, 2, 9, 9, 4, 1, 2, 9, 9, 4]
    # [9, 4] (then 1, 2) ended last at 6
    assert prop.propose([7], 1)[0].tolist() == [[1, 2]]
    # a later verdict: one draft kept, then the target's token
    prop.observe(7, np.array([1, 4, 1, 5, 0]))
    prop.plan(ProposeStep(drafts=2), _ctx([7]))
    assert history.tokens[-2:] == [1, 5]
    prop.remove_request(7)
    assert not prop._histories and not prop._prompts and not prop._verdicts
    # a verdict read after the removal is dropped, not kept for the handle's next owner
    prop.observe(7, np.array([0, 1, 2, 3, 4]))
    prop.ingest_request(7)
    assert not prop._verdicts and prop._histories[7].tokens == []


def test_a_preplan_is_promoted_only_for_its_own_step():
    prop = NgramProposer(CPU, max_drafts=2, min_ngram=2, max_ngram=2)
    for rid, prompt in ((1, [5, 6, 7, 5, 6]), (2, [8, 9, 3, 8, 9])):
        prop.ingest_request(rid)
        prop.commit(ProposeStep(history={rid: torch.tensor(prompt)}), _ctx([rid]))
    step = ProposeStep(drafts=2, verify=True)
    ahead = _ctx([1, 2], preplan=True, slot=1)
    prop.plan(step, ahead)
    # planned ahead of the rows' verdicts, it stages nothing until they are read
    assert prop._staged[1] is None
    assert prop.correct_preplan(None, step, ahead)
    staged = prop._staged[1]
    assert staged.tolist() == [[5, 6], [8, 9]]
    prop.plan(step, _ctx([1, 2], slot=1))
    assert prop._current is staged
    # staged for [1], run as [2]: planned again
    prop.plan(step, _ctx([1], preplan=True))
    prop.correct_preplan(None, step, _ctx([1], preplan=True))
    prop.plan(step, _ctx([2]))
    # fresh: [8, 9] last ended before the 3, taken as the prefill's token, so 8, 9 follow
    assert prop.propose([2], 1)[0].tolist() == [[8, 9]]


def test_the_drafts_planned_ahead_are_the_ones_planned_after_the_verdict():
    """A verify step planned ahead stages its drafts at ``correct_preplan``, after the runner
    read the step in flight's verdict: the same drafts an inline plan stages then."""
    rows = {}
    for ahead in (True, False):
        prop = NgramProposer(CPU, max_drafts=2, min_ngram=2, max_ngram=3)
        prop.ingest_request(7)
        prop.commit(ProposeStep(history={7: torch.tensor([4, 1, 2, 9, 9, 4, 1])}), _ctx([7]))
        step = ProposeStep(drafts=2, verify=True)
        if ahead:
            prop.plan(step, _ctx([7], preplan=True))
        # the verdict lands: input 2 (the prefill's), drafts [9, 9] kept, then 4
        prop.observe(7, np.array([2, 2, 9, 9, 4]))
        if ahead:
            prop.correct_preplan(None, step, _ctx([7], preplan=True))
        prop.plan(step, _ctx([7]))
        rows[ahead] = prop.propose([7], 1)[0].tolist()
    assert rows[True] == rows[False] == [[1, 2]]


def test_a_row_that_keeps_missing_looks_up_less_often(monkeypatch):
    """On steps of no drafts, a row whose lookups find nothing backs off (1, 3, then 7 steps
    between lookups), its verdicts waiting for the next one; a match resets it. Deterministic:
    it moves only at the commits that call ``predict``."""
    prop = NgramProposer(CPU, max_drafts=3, min_ngram=2, max_ngram=3)
    prop.ingest_request(0)
    prop.commit(ProposeStep(history={0: torch.tensor([90, 91, 92, 93])}), _ctx([0]))
    looked = []
    real = NgramIndex.locate

    def counting(self, history):
        looked.append(len(history.tokens))
        return real(self, history)

    monkeypatch.setattr(NgramIndex, "locate", counting)
    for step in range(20):
        prop.observe(0, np.array([0, 5, 100 + step]))
        prop.predict([0], 0)
    # the step each lookup ran at, from its history's length (the prompt and the prefill's
    # token, then a token a step)
    steps = [n - 6 for n in looked]
    assert steps == [0, 1, 3, 7, 15], steps
    assert prop._histories[0].predicted == 0.0
    # the prompt's start comes back: the row's next lookup (step 23) finds it, and the backoff
    # resets
    for step in range(20, 24):
        prop.observe(0, np.array([0, 5, (77, 90, 91, 92)[step - 20]]))
        prop.predict([0], 0)
    assert [n - 6 for n in looked][5:] == [23]
    history = prop._histories[0]
    assert history.follow >= 0 and history.misses == 0 and history.predicted > 0


def test_draft_config_from_yaml_kwargs():
    assert DraftConfig.from_kwargs(None) is None
    assert DraftConfig.from_kwargs({}) is None
    cfg = DraftConfig.from_kwargs({"method": "ngram", "max_drafts": 3, "max_ngram": 5})
    policy = cfg.policy(stop_ids={2})
    # prompt lookup decides each step by default, against a gain that grows with the batch
    assert policy == DraftPolicy(
        max_drafts=3, stop_ids=frozenset({2}), proposer=NGRAM, adaptive=True, min_gain=((1, 0.5), (32, 1.0)),
    )
    assert policy.draft_options == (0, 3)
    assert (policy.gain_needed(1), policy.gain_needed(32), policy.gain_needed(64)) == (0.5, 32.0, 64.0)
    assert policy.gain_needed(9) == pytest.approx(9 * (0.5 + 0.5 * 8 / 31))
    fixed = DraftConfig.from_kwargs({"method": "ngram", "adaptive": False, "min_gain": 2}).policy()
    assert not fixed.adaptive and fixed.draft_options == (4,) and fixed.gain_needed(3) == 6.0
    # a YAML list of [rows, gain] points
    listed = DraftConfig.from_kwargs({"method": "ngram", "min_gain": [[16, 1.0], [1, 0.25]]})
    assert listed.min_gain == ((1, 0.25), (16, 1.0)) and listed.policy().gain_needed(100) == 100.0
    specs = cfg.resources({"LLM"})
    assert [type(s).__name__ for s in specs] == ["DraftVerifySpec", "NgramSpec"]
    with pytest.raises(ValueError, match="unknown draft settings"):
        DraftConfig.from_kwargs({"method": "ngram", "k": 3})
    with pytest.raises(ValueError, match="draft.method"):
        DraftConfig.from_kwargs({"method": "eagle"})
    with pytest.raises(ValueError, match="min_ngram"):
        DraftConfig.from_kwargs({"method": "ngram", "min_ngram": 4, "max_ngram": 3})
    # a draft model drafts in the forward: no engine-side proposer, no NgramSpec
    cfg = DraftConfig.from_kwargs({"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "max_drafts": 4})
    assert cfg.policy().proposer is None and cfg.policy().lookahead == 9
    # a draft model drafts every step: it predicts nothing to decide one by
    assert not cfg.adaptive
    assert [type(s).__name__ for s in cfg.resources({"LLM"})] == ["DraftVerifySpec"]
    bad_configs = ({"method": "draft_model"}, {"method": "ngram", "model": "x"},
                   {"method": "ngram", "draft_sample": True},
                   {"method": "draft_model", "model": "x", "adaptive": True},
                   {"method": "ngram", "min_gain": -1}, {"method": "ngram", "min_gain": [[0, 1.0]]},
                   {"method": "ngram", "min_gain": "high"})
    for bad in bad_configs:
        with pytest.raises(ValueError, match="(?i)draft"):
            DraftConfig.from_kwargs(bad)


# ── through the engine ──

# prompts that repeat themselves, so the lookup finds drafts from the first verify step
PROMPTS = [
    [3, 9, 12, 40, 7, 3, 9, 12, 40, 7, 3, 9],
    [5, 11, 5, 11, 5, 11],
    [20, 21, 22, 23, 24, 25, 20, 21, 22],
    [60, 2, 33, 60, 2],
]


# prompts whose suffix never occurred earlier: nothing to draft from at first
FRESH = [
    [44, 1, 30, 2, 61, 17, 8, 50],
    [12, 55, 7, 33, 21],
]


def _run(k: int, batched: bool = True, prompts=PROMPTS, max_tokens: int = MAX_TOKENS,
         ignore_eos: bool = False, seed: int = toy.SEED, adaptive: bool = False, min_gain: float = 0.5, **ngram):
    sub = toy.ToySubmodule(k=k, seed=seed, proposer=NGRAM, adaptive=adaptive, min_gain=min_gain)
    eng = toy.engine(sub, CPU, ngram={"max_drafts": k, **ngram})
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, max_tokens, ignore_eos=ignore_eos)
    return drv.run(list(range(len(prompts))), batched=batched), eng, drv


def _reference(prompts=PROMPTS, max_tokens=MAX_TOKENS, ignore_eos=False, seed=toy.SEED):
    return {
        rid: toy.reference(prompt, max_tokens, ignore_eos=ignore_eos, seed=seed)
        for rid, prompt in enumerate(prompts)
    }


@pytest.mark.parametrize("batched", [True, False], ids=["batched", "one_per_step"])
@pytest.mark.parametrize("k", [1, 2, 4, 5])
def test_greedy_ngram_emits_what_plain_decode_emits(k, batched):
    out, eng, _ = _run(k, batched=batched)
    assert out == _reference()
    dv, prop = eng._resources[toy.DV], eng._resources[NGRAM]
    assert 0 < dv.drafts_accepted < k * dv.rows_verified, dv.stats()
    assert 0 < prop.rows_matched <= prop.rows_proposed


@pytest.mark.parametrize("batched", [True, False], ids=["batched", "one_per_step"])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("min_gain", [0.5, 0.0, 1e9], ids=["adaptive", "always_drafts", "never_drafts"])
def test_adaptive_steps_emit_what_plain_decode_emits(k, batched, min_gain):
    """Whichever steps draft, and in whatever mix, the streams are plain greedy decode's: a step
    of no drafts emits its one token a row and leaves nothing to rewind, and a verify step after
    it drafts from a history that holds that token."""
    prompts = PROMPTS + FRESH
    out, eng, drv = _run(k, batched=batched, prompts=prompts, adaptive=True, min_gain=min_gain)
    assert out == _reference(prompts)
    dv = eng._resources[toy.DV]
    kinds = {drafts for drafts in drv.step_drafts if drafts is not None}
    if min_gain == 0.0:
        assert kinds == {k} and dv.steps_plain == 0
    elif min_gain == 1e9:
        assert kinds == {0} and dv.rows_verified == 0
    else:
        # a request's first step has no prediction yet, and the repeating prompts keep drafts
        assert kinds == {0, k} and dv.drafts_accepted > 0, dv.stats()
    assert dv.steps_plain + dv.steps_verified == len(drv.step_drafts)


def test_the_choice_moves_only_at_a_commit(monkeypatch):
    """The plan thread chooses a step's drafts while the gpu thread finishes the step before it
    and the main loop runs its stop check: reading that step's verdicts early, or emitting its
    tokens, must not change what the predictions say, or two paths (or two ranks) could choose
    apart."""
    sub = toy.ToySubmodule(k=3, proposer=NGRAM, adaptive=True)
    eng = toy.engine(sub, CPU)
    drv = toy.Driver(eng, sub)
    prompts = PROMPTS + FRESH
    for rid, prompt in enumerate(prompts):
        drv.add(rid, prompt, MAX_TOKENS)
    dv, prop, policy = eng._resources[toy.DV], eng._resources[NGRAM], sub.get_draft_policy("decode")
    real = eng.check_stop_for_batch
    seen = []

    def checking(batch, outputs, host_rows=None):
        rids = list(batch.request_ids)
        before = (dv.choose_drafts(rids, policy), prop.expected_gain(rids))
        # a pre-plan's read of the verdicts, ahead of the next commit
        dv.harvest(rids)
        stops = real(batch, outputs, host_rows)
        seen.append((before, (dv.choose_drafts(rids, policy), prop.expected_gain(rids))))
        return stops

    monkeypatch.setattr(eng, "check_stop_for_batch", checking)
    out = drv.run(list(range(len(prompts))))
    assert out == _reference(prompts)
    assert len(seen) > 5 and all(before == after for before, after in seen), seen
    assert {choice for (choice, _), _ in seen} == {0, 3}


def test_a_step_of_no_drafts_is_emitted_at_its_stop_check():
    """The gpu thread doesn't wait for a step of no drafts: it counts the step's tokens and
    leaves its verdict pending; the stop check (after the worker waited for the step) emits
    them, and the rows' next commit hands the verdict to the proposer."""
    sub = toy.ToySubmodule(k=3, proposer=NGRAM, adaptive=True, min_gain=1e9)
    eng = toy.engine(sub, CPU)
    drv = toy.Driver(eng, sub)
    drv.add(0, FRESH[0], MAX_TOKENS)
    dv, prop = eng._resources[toy.DV], eng._resources[NGRAM]
    real = eng.check_stop_for_batch
    states = []

    def checking(batch, outputs, host_rows=None):
        states.append((0 in dv._pending, "new_token" in outputs.get(0, {}), batch.verify_plain is not None))
        return real(batch, outputs, host_rows)

    eng.check_stop_for_batch = checking
    drv.decode([0])
    assert states == [(True, False, True)]
    # its token rides inline to the client: host rows the step's forward filled
    from mstar.engine.resources.draft_verify.resource import PLAIN_TOKENS

    assert eng.inline_client_signals(toy.NODE, "decode") == {"new_token": PLAIN_TOKENS}
    assert eng.inline_client_signals(toy.NODE, "prefill") == {}
    assert len(drv.emitted[0]) == 2 and not prop._verdicts
    history = list(prop._histories[0].tokens)
    drv.decode([0])
    # the first step's verdict, read at the second's commit: the prefill's token and its own
    assert prop._histories[0].tokens == FRESH[0] + drv.emitted[0][:2]
    assert history == FRESH[0]


def test_a_step_with_a_row_past_its_cap_sends_its_tokens_as_tensors():
    """A step of no drafts whose every row emits its token hands the worker host rows to send
    them inline; one with a row that emits nothing (the async worker's step after a
    ``max_tokens`` stop) doesn't, and the rows go as tensors, cut."""
    from mstar.engine.resources.draft_verify.resource import PLAIN_TOKENS

    sub = toy.ToySubmodule(k=3, proposer=NGRAM, adaptive=True, min_gain=1e9)
    eng = toy.engine(sub, CPU)
    drv = toy.Driver(eng, sub)
    drv.add(0, FRESH[0], 3)
    inline = []
    real = eng.check_stop_for_batch

    def checking(batch, outputs, host_rows=None):
        stops = real(batch, outputs, host_rows)
        inline.append(PLAIN_TOKENS in batch.outputs.host_rows.buffers)
        return stops

    eng.check_stop_for_batch = checking
    drv.run([0])
    drv.decode([0])
    assert inline == [True, True, False]
    assert drv.emitted[0] == toy.reference(FRESH[0], 3)


def test_a_runtime_that_inlines_gets_no_tensor_of_a_plain_steps_tokens():
    """Under a graph runtime that sends a client signal inline, a step of no drafts emits no
    tensor of its tokens (the worker sends them off the host rows); a verify step still does.
    The two together are the stream plain decode emits."""
    from types import SimpleNamespace

    from mstar.engine.resources.draft_verify.resource import PLAIN_TOKENS

    sub = toy.ToySubmodule(k=3, proposer=NGRAM, adaptive=True)
    eng = toy.engine(sub, CPU)
    eng._graph_runtime = SimpleNamespace(supports_inline_emit=True)
    drv = toy.Driver(eng, sub)
    prompt = PROMPTS[0]
    drv.add(0, prompt, MAX_TOKENS)
    emitted, kinds = list(drv.emitted[0]), []
    real = eng.check_stop_for_batch

    def checking(batch, outputs, host_rows=None):
        stops = real(batch, outputs, host_rows)
        host = batch.outputs.host_rows
        inline = host is not None and PLAIN_TOKENS in host.buffers
        kinds.append((batch.verify_drafts, inline, "new_token" in outputs[0]))
        emitted.extend(host.buffers[PLAIN_TOKENS].tolist() if inline else outputs[0]["new_token"][0].tolist())
        return stops

    eng.check_stop_for_batch = checking
    for _ in range(MAX_TOKENS):
        out = drv._step("decode", {0: drv.next[0]})
        drv.next[0] = out[0]["text_inputs"][0]
        if drv.stops:
            break
    assert emitted == toy.reference(prompt, MAX_TOKENS)
    # a plain step's tokens ride inline, with no tensor; a verify step's are tensors
    assert {kind for kind in kinds} == {(0, True, False), (3, False, True)}


@pytest.mark.parametrize("max_ngram", [1, 3, 6])
def test_ngram_sizes_and_ignore_eos(max_ngram):
    out, _, _ = _run(3, ignore_eos=True, min_ngram=1, max_ngram=max_ngram)
    assert out == _reference(ignore_eos=True)


@pytest.mark.parametrize("adaptive", [False, True])
def test_the_history_is_the_prompt_and_what_was_emitted(adaptive):
    out, eng, _ = _run(3, adaptive=adaptive)
    prop = eng._resources[NGRAM]
    for rid, prompt in enumerate(PROMPTS):
        history = prop._histories[rid].tokens
        # every verdict but the last one read (the request stopped before another step)
        assert history == (prompt + out[rid])[: len(history)]
        assert len(history) > len(prompt)
    for rid in range(len(PROMPTS)):
        eng.remove_request(rid)
    assert not prop._histories and not prop._verdicts and not prop._prompts


def test_an_unknown_proposer_or_a_mismatched_one_refuses_to_load():
    sub = toy.ToySubmodule(k=3, proposer="nope")
    with pytest.raises(ValueError, match="proposer 'nope'"):
        toy.engine(sub, CPU)
    sub = toy.ToySubmodule(k=3, proposer=NGRAM)
    with pytest.raises(ValueError, match="its proposer 'ngram' drafts 2"):
        toy.engine(sub, CPU, ngram={"max_drafts": 2})


@pytest.mark.parametrize("mode", ["every_step", "adaptive", "no_drafts"])
def test_sampled_ngram_streams_follow_the_target(monkeypatch, mode):
    """``test_draft_verify_sampled``'s joint check of the two tokens after the prefill's, with
    the n-gram proposer drafting (point-mass drafts) from a prompt that ends in the cycle the
    toy's greedy streams settle into, so its guesses are often in the target's top k. Adaptive,
    the first of the two comes from a step of no drafts (a request's first step has no
    prediction) and the second from a verify step; ``no_drafts`` draws both one at a time."""
    prompt = [3, 9, 12, 40, 17, 37, 40, 17, 37, 40]
    monkeypatch.setattr(sampled, "PROMPT", prompt)
    real = toy.ToySubmodule
    adaptive = {"every_step": {}, "adaptive": {"adaptive": True}, "no_drafts": {"adaptive": True, "min_gain": 1e9}}

    def ngram_sub(**kwargs):
        kwargs.pop("corrupt", None)
        kwargs.pop("sampled_drafts", None)
        return real(proposer=NGRAM, **adaptive[mode], **kwargs)

    monkeypatch.setattr(toy, "ToySubmodule", ngram_sub)
    n = 1536
    streams, engines = sampled._streams(3, "argmax", n)
    t1, counts = sampled._first_two(streams)
    assert sampled._chi2(counts, sampled._expected(t1, n)) > sampled.P_MIN, counts
    rows = sum(eng._resources[toy.DV].rows_verified for eng in engines)
    accepted = sum(eng._resources[toy.DV].drafts_accepted for eng in engines)
    plain = sum(eng._resources[toy.DV].rows_plain for eng in engines)
    if mode == "no_drafts":
        assert rows == 0 and plain == 2 * n
    else:
        assert 0 < accepted < 3 * rows
        assert plain == (n if mode == "adaptive" else 0)
