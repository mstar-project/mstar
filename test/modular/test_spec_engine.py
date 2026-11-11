"""Speculative decoding end to end on the real `Engine`, synchronous mode, on
the CPU: a toy target whose tokens depend on its KV cache and positions, and
scripted drafters (`spec_toy.py`). Greedy output equals plain greedy decoding
whatever the drafts, and each mechanism has a test that fails without it.
"""
from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace

import pytest
import torch
from spec_toy import NODE, STOP, Driver, always_wrong, engine, every, oracle, reference

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.manager import RopeManager


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


PROMPTS = {1: [5, 9, 2], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9]}


def _decode(k, drafter, max_tokens=24, prompts=PROMPTS, batched=True):
    eng = engine(k, drafter)
    driver = Driver(eng)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, max_tokens)
    if batched:
        driver.run(list(prompts))
    else:
        for rid in prompts:
            driver.run([rid])
    return driver.out, eng


@pytest.mark.parametrize("k,drafter", [
    (1, oracle), (3, oracle), (3, always_wrong), (3, every(2)), (5, every(3)), (4, every(4)),
])
@pytest.mark.parametrize("batched", [True, False])
def test_greedy_output_equals_plain_greedy_whatever_the_drafts(k, drafter, batched):
    out, eng = _decode(k, drafter, batched=batched)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference(prompt, 24), rid


def test_an_oracle_drafter_has_every_draft_accepted():
    k = 3
    out, eng = _decode(k, oracle)
    draft = eng._drafts[NODE]
    assert draft.accepted == draft.drafted
    # the cut trims the last block to the budget or a stop id, never more
    assert all(out[rid] == reference(prompt, 24) for rid, prompt in PROMPTS.items())


def test_an_always_wrong_drafter_accepts_nothing():
    out, eng = _decode(3, always_wrong)
    assert eng._drafts[NODE].accepted == 0


@pytest.mark.parametrize("drafter", [oracle, always_wrong], ids=["oracle", "always_wrong"])
def test_a_verifying_node_whose_sampler_runs_lean_decodes_as_one_that_doesnt(monkeypatch, drafter):
    """The toy's narrow vocabulary keeps its sampler off the lean path; turned on, every step of
    these all-greedy requests runs it lean (an eager step does whenever its rows allow), so a
    greedy prefill hands the verifying node its first token, and each verify step has the
    sampler in ``ctx.lean``. Tokens, acceptance and each row's RNG offset come out as without."""
    from mstar.engine.resources.sampler.resource import SamplerResource

    lean_steps = []
    real_flag = SamplerResource._set_greedy_flag

    def flag(self, ctx):
        real_flag(self, ctx)
        lean_steps.append((ctx.graph_walk, self._greedy_this_step))

    monkeypatch.setattr(SamplerResource, "_set_greedy_flag", flag)
    runs = {}
    for lean in (False, True):
        lean_steps.clear()
        eng = engine(3, drafter, greedy_capture=lean)
        sampler = eng._resources["sampler"]
        assert sampler.lean_capture is lean
        offsets = {}
        remove = eng.remove_request

        def removed(rid, remove=remove, sampler=sampler, offsets=offsets):
            # every row here has a capture slot, whose master counter the draws advance
            assert sampler._cg_buffers.has_slot(rid)
            offsets[rid] = int(sampler._cg_buffers.eager_offsets([rid])[1].item())
            return remove(rid)

        eng.remove_request = removed
        driver = Driver(eng)
        for rid, prompt in PROMPTS.items():
            driver.add(rid, prompt, 24)
        driver.run(list(PROMPTS))
        draft = eng._drafts[NODE]
        runs[lean] = (driver.out, draft.accepted, draft.drafted, offsets)
        walks = {walk for walk, greedy in lean_steps if greedy}
        assert walks == ({"prefill", "decode"} if lean else set()), lean_steps
    assert runs[True] == runs[False]
    # each row drew at least once, so a path that skips the advance shows up as a lower count
    assert all(offset > 0 for offset in runs[False][3].values()), runs[False][3]
    for rid, prompt in PROMPTS.items():
        assert runs[True][0][rid] == reference(prompt, 24), rid


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 4, 5, 9])
def test_the_budget_ends_a_reply_exactly_inside_a_block(max_tokens):
    out, _ = _decode(3, oracle, max_tokens=max_tokens)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference(prompt, max_tokens)
        assert len(out[rid]) == min(max_tokens, len(reference(prompt, max_tokens)))


def test_lengths_end_exact_after_the_last_take_back():
    k = 3
    eng = engine(k, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 40)
    for _ in range(5):
        driver.step([1])
    # one more admit applies the last result: the host then holds the true length
    driver.step([1])
    kv = eng._resources["kv"]
    stream = kv._streams[1]["main"]
    emitted_before_last = len(PROMPTS[1]) + len(driver.out[1])
    # the last step's block is still counted whole (reserved), its result unapplied
    assert stream.stored_len - (k + 1) <= emitted_before_last - 1 <= stream.stored_len


def test_without_the_kv_take_back_the_output_goes_wrong(monkeypatch):
    """Negative control: rejected drafts stay in the KV the target reads."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    monkeypatch.setattr(KVManager, "take_back", lambda self, rids, counts, label: None)
    out, _ = _decode(3, every(2))
    assert any(out[rid] != reference(prompt, 24) for rid, prompt in PROMPTS.items())


def test_without_the_position_take_back_the_output_goes_wrong(monkeypatch):
    """Negative control: positions run ahead of the KV by the rejected count."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    monkeypatch.setattr(RopeManager, "take_back", lambda self, rids, counts, label: None)
    out, _ = _decode(3, every(2))
    assert any(out[rid] != reference(prompt, 24) for rid, prompt in PROMPTS.items())


def test_a_stop_token_inside_a_block_ends_the_reply_there():
    # find a prompt whose greedy reply hits the stop id
    prompt = next(
        [a, b] for a in range(64) for b in range(64)
        if STOP in reference([a, b], 40) and reference([a, b], 40).index(STOP) >= 3
    )
    out, _ = _decode(5, oracle, max_tokens=40, prompts={1: prompt})
    assert out[1] == reference(prompt, 40)
    assert out[1][-1] == STOP


def test_a_row_offloaded_mid_verify_step_is_recorded_with_nothing_to_take_back(monkeypatch):
    """The engine path from KV's commit to the ledger: a stream offloaded while its verify step runs has its
    span dropped by the commit, which records it on the step (`KVStep.dropped`); `NodeDraft.committed` reads it
    there, so the ledger records that row with a counted span of 0 and the others with the whole block."""
    k = 3
    eng = engine(k, oracle)
    driver = Driver(eng)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, 24)
    ledger = eng._drafts[NODE].ledger
    recorded = []
    record = ledger.committed

    def committed(rids, labels=None, counted=None):
        if rids is not None:
            recorded.append((list(rids), list(counted)))
        return record(rids, labels, counted)

    monkeypatch.setattr(ledger, "committed", committed)
    victim = 2
    real_commit = KVManager.commit

    def commit(self, step, ctx):
        stream = self._streams[victim]["main"]
        if ctx.graph_walk != "decode" or recorded:
            return real_commit(self, step, ctx)
        # what an offload's claim leaves on the stream while its copy runs with the lock down
        stream.offloaded = True
        try:
            return real_commit(self, step, ctx)
        finally:
            stream.offloaded = False

    monkeypatch.setattr(KVManager, "commit", commit)
    driver.step(list(PROMPTS))
    (rids, counted), = recorded
    assert dict(zip(rids, counted, strict=True)) == {rid: 0 if rid == victim else k + 1 for rid in PROMPTS}


def test_a_removed_request_leaves_no_results_for_its_handle():
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 40)
    driver.step([1])
    draft = eng._drafts[NODE]
    assert draft.ledger.pending(1)
    eng.remove_request(1)
    assert not draft.ledger.pending(1)
    # the handle comes back as another request, which decodes from scratch
    driver.out.pop(1)
    driver.add(1, PROMPTS[2], 12)
    driver.run([1])
    assert driver.out[1] == reference(PROMPTS[2], 12)


def test_tokens_are_ints_on_the_host():
    out, _ = _decode(2, oracle, max_tokens=5, prompts={1: [3]})
    assert all(isinstance(t, int) for t in out[1])
    assert torch.tensor(out[1]).dtype == torch.long


def test_an_asynchronous_verify_step_without_a_resident_capture_raises():
    """A host plan reads the reserved length as the attention length and write
    slot, so an asynchronous verify step never runs host-planned."""
    eng = engine(3, oracle, mode="asynchronous")
    assert eng._drafts[NODE].asynchronous
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 10)
    with pytest.raises(RuntimeError, match="found no resident capture"):
        driver.step([1])


def test_a_synchronous_node_is_reported_to_the_worker():
    assert engine(3, oracle).synchronous_verify_nodes() == {NODE}
    assert engine(3, oracle, mode="asynchronous").synchronous_verify_nodes() == frozenset()


def test_auto_stays_synchronous_when_the_verify_walk_has_no_resident_capture(monkeypatch):
    from spec_toy import ToyLM

    monkeypatch.setattr(ToyLM, "resident_walks", lambda self: frozenset())
    eng = engine(3, oracle, mode="auto")
    assert not eng._drafts[NODE].asynchronous
    assert "resident_walks" in eng._drafts[NODE].reason


def test_a_verify_step_run_row_by_row_is_refused(caplog):
    from spec_toy import ToyLM

    eng = engine(3, oracle)
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 10)
    driver.add(2, PROMPTS[2], 10)
    from spec_toy import _batch

    ToyLM.can_batch = lambda self, batch, model_inputs: False
    try:
        batch = _batch("decode", {rid: torch.tensor([driver.out[rid][-1]]) for rid in (1, 2)}, driver._max)
        eng.prepare_inputs(batch)
        eng.exec_and_postprocess(batch)
    finally:
        ToyLM.can_batch = lambda self, batch, model_inputs: True
    # the second row's commit would be a second result for one step: that row fails, loudly
    assert list(batch.failed_requests) == [2]
    assert "runs batched" in caplog.text


def _stub_offload(monkeypatch, snapshots):
    """KV's host copy needs CUDA: record the length an offload would snapshot instead."""
    def offload(self, rid):
        snapshots[rid] = self._streams[rid]["main"].stored_len
        return 1

    monkeypatch.setattr(KVManager, "supports_eviction", property(lambda self: True))
    monkeypatch.setattr(KVManager, "offload", offload)
    monkeypatch.setattr(KVManager, "reload", lambda self, rid: True)
    monkeypatch.setattr(KVManager, "is_offloaded", lambda self, rid: False)


def test_an_offload_marks_its_victim_and_its_results_come_due_at_its_first_admit_back(monkeypatch):
    snapshots: dict[int, int] = {}
    _stub_offload(monkeypatch, snapshots)
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.add(2, PROMPTS[2], 30)
    driver.step([1, 2])
    driver.step([1])
    ledger = eng._drafts[NODE].ledger
    # row 2's result is recorded but not applied: a victim
    assert ledger.pending(2) and not eng.verify_unrecorded(NODE, 2)
    assert eng.offload_request(NODE, 2) > 0
    # nothing is applied at the offload: the host copy holds the reserved length, and the
    # results stay until the row's first admit back, which has them all due
    assert snapshots[2] > len(PROMPTS[2]) + len(driver.out[2]) - 1
    assert ledger.pending(2) and 2 in ledger._returning
    # the runner's pass at the next admit takes them back
    driver.step([1, 2])
    assert 2 not in ledger._returning
    driver.run([1, 2])
    for rid in (1, 2):
        assert driver.out[rid] == reference(PROMPTS[rid], 30)


def test_an_offload_that_frees_nothing_marks_nothing():
    """The row keeps its slot, so the two-step rule stays right for it."""
    eng = engine(3, always_wrong)
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.add(2, PROMPTS[2], 30)
    driver.step([1, 2])
    driver.step([1])
    ledger = eng._drafts[NODE].ledger
    assert eng.offload_request(NODE, 2) == 0
    assert ledger.pending(2) and not ledger._returning


def test_a_returning_row_with_an_unrecorded_result_trips_its_first_admit():
    """Rank 0 offloads only a row whose steps are all recorded; a returning row that breaks
    that is refused at its first admit back, on every rank."""
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    ledger = eng._drafts[NODE].ledger
    ledger.committed([1], {"kv": "main", "rope": "main"}, [4])
    assert eng.verify_unrecorded(NODE, 1)
    ledger.mark_offloaded(1)
    with pytest.raises(RuntimeError, match="never recorded"):
        ledger.take_due(ledger.next_step, {1})


def test_the_acceptance_log_opens_its_window_at_the_first_verify_step(monkeypatch, caplog):
    """Loading can take longer than a window: the first verify step opens it, and each line
    counts its own window, not the totals since load."""
    import logging
    import time

    import numpy as np

    from mstar.engine.resources.draft import node as node_mod

    draft = engine(3, every(2))._drafts[NODE]
    # a load that took longer than a window
    clock = [time.monotonic() + 1000.0]
    monkeypatch.setattr(node_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    caplog.set_level(logging.INFO, logger=node_mod.logger.name)

    def lines():
        return [r.getMessage() for r in caplog.records if "tau" in r.getMessage()]

    draft._count(np.array([3, 3]))
    clock[0] += 30.0
    draft._count(np.array([0]))
    assert lines() == []
    clock[0] += 31.0
    draft._count(np.array([1]))
    (first,) = lines()
    assert "the last 61 s: tau 2.75 over 4 verify rows" in first
    clock[0] += 61.0
    draft._count(np.array([0, 0]))
    assert "the last 61 s: tau 1.00 over 2 verify rows" in lines()[1]
    # the totals are the node's lifetime ones
    assert draft.verify_rows == 6 and draft.acceptance_length() == (7 + 6) / 6


def test_a_removed_request_takes_its_offload_mark_with_it():
    """Handle-keyed (invariant 14): a recycled handle must not inherit it."""
    eng = engine(3, every(2))
    ledger = eng._drafts[NODE].ledger
    ledger.mark_offloaded(7)
    ledger.remove_request(7)
    assert not ledger._returning


def test_debug_asserts_catch_positions_drifting_from_kv(monkeypatch):
    """``MSTAR_KV_DEBUG_ASSERTS``: a row's position counter minus its KV length is fixed."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    out, _ = _decode(3, every(2))
    assert out[1] == reference(PROMPTS[1], 24)
    monkeypatch.setattr(RopeManager, "take_back", lambda self, rids, counts, label: None)
    with pytest.raises(AssertionError, match="position offset"):
        _decode(3, every(2))


def test_the_offset_check_skips_an_offloaded_row_and_checks_only_the_step(monkeypatch):
    """An offload resets a stream's KV length but not its position counter: the
    debug check of another row's step must not trip on it."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, [1, 2, 3, 4, 5], 20)
    driver.add(2, [6, 7, 8], 20)
    for _ in range(2):
        driver.step([1, 2])
    stream = eng._resources["kv"]._streams[2]["main"]
    # what an offload commit leaves on row 2's stream
    stream.offloaded = True
    stream.reset(freed=False, content_reset=False)
    driver.step([1])
    draft = eng._drafts[NODE]
    draft.check_offsets(eng._submodules[NODE].resources, [1, 2])


def test_the_offset_check_reads_the_rows_position_label():
    """A model whose position label differs from its KV label: the check pairs
    them by the step's segments rather than looking the KV label up in positions."""
    from types import SimpleNamespace

    from mstar.engine.resources.draft.node import NodeDraft
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.engine.resources.step import Segment, SubmoduleStep

    kv, rope = KVManager.__new__(KVManager), RopeManager.__new__(RopeManager)
    stream = SimpleNamespace(offloaded=False, page_indices=[0], stored_len=10)
    kv._streams = {1: {"main": stream}}
    rope._counters = {1: {"text": 13}}
    step = SubmoduleStep(steps={}, segments=None)
    step.steps["kv"] = SimpleNamespace(segments=[Segment(1, "main", 4)])
    step.steps["rope"] = SimpleNamespace(segments=[Segment(1, "text", 4)])
    draft = NodeDraft.__new__(NodeDraft)
    draft.node, draft._offsets, draft._rope_labels = NODE, {}, {}
    draft.check_offsets({"kv": kv, "rope": rope}, [1], step)
    assert draft._offsets == {(1, "main"): 3}
    stream.stored_len, rope._counters[1]["text"] = 14, 17
    draft.check_offsets({"kv": kv, "rope": rope}, [1], step)
    rope._counters[1]["text"] = 18
    with pytest.raises(AssertionError, match="moved from 3 to 4"):
        draft.check_offsets({"kv": kv, "rope": rope}, [1], step)


def _keyed(eng, rid, tokens, max_tokens, scratch):
    """A request whose prompt pages are keyed, as a model's ``process_prompt`` declares them."""
    from spec_toy import PAGE, _take_sampler_row

    from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
    from mstar.engine.resources.kv.keys import chain
    from mstar.engine.resources.sampler.config import SamplingReqConfig

    whole = len(tokens) // PAGE
    kv = KVReqConfig(
        prefix_keys={"main": chain([tokens[at: at + PAGE] for at in range(0, whole * PAGE, PAGE)])},
        prefix_tail={"main": tokens[whole * PAGE:]},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(tokens), len(tokens), "prefill"),), growth=(("decode_loop", 1),),
            scratch=scratch, keyed_spans=1,
        )},
        loop_caps={"decode_loop": max_tokens},
    )
    sampling = SamplingReqConfig(temperature=0)
    sampling.apply_conductor_config(seed=0)
    eng.add_request(rid, {"kv": kv, "sampler": sampling})
    _take_sampler_row(eng, rid)


def _prefill(eng, driver, rid, tokens, max_tokens):
    from spec_toy import _batch

    driver.out[rid], driver._max = [], {**getattr(driver, "_max", {}), rid: max_tokens}
    batch = _batch("prefill", {rid: torch.tensor(tokens)}, driver._max)
    eng.prepare_inputs(batch)
    out = eng.exec_and_postprocess(batch)
    driver.out[rid].append(int(out.per_rid_outputs[rid]["new_token"][0][0]))


def test_a_prompt_prefix_hits_on_a_verifying_node_and_greedy_parity_holds():
    from mstar.model.submodule_base import ARNodeInputs

    k = 3
    eng = engine(k, every(2))
    kv = eng._resources["kv"]
    assert kv._index is not None
    # the decode walk verifies, so only the prompt walk keys pages
    assert kv._keyed_walks["main"] == frozenset({"prefill"})
    prompt = list(range(1, 14))
    driver = Driver(eng)
    _keyed(eng, 1, prompt, 12, scratch=2 * (k + 1))
    _prefill(eng, driver, 1, prompt, 12)
    driver.run([1])
    assert driver.out[1] == reference(prompt, 12)
    _keyed(eng, 2, prompt, 12, scratch=2 * (k + 1))
    inputs = ARNodeInputs(input_seq_len=len(prompt))
    prefix = eng._runner.resolve_cached_prefix(2, NODE, "prefill", inputs)
    assert prefix is not None and prefix.tokens == 8
    eng._runner.apply_cached_prefix(2, NODE, "prefill", inputs, prefix)
    # the engine cuts a keyed walk's inputs by the prefix applied, as it does for the worker
    _prefill(eng, driver, 2, prompt, 12)
    driver.run([2])
    assert driver.out[2] == reference(prompt, 12)


def test_a_layout_without_the_slack_is_refused_on_a_verifying_node():
    eng = engine(3, oracle)
    with pytest.raises(ValueError, match="scratch"):
        _keyed(eng, 1, list(range(1, 14)), 12, scratch=0)


def test_a_verifying_node_whose_kv_another_instance_reads_is_refused_at_load(monkeypatch):
    from mstar.engine.resources.kv.manager import KVManager

    real = KVManager.__init__

    def published(self, *args, **kwargs):
        real(self, *args, **kwargs)
        self._needs_remote_transfer = True

    monkeypatch.setattr(KVManager, "__init__", published)
    refusal = f"speculative decoding on {NODE}: resource 'kv': KV .* is read by another instance"
    with pytest.raises(RuntimeError, match=refusal):
        engine(3, oracle)


def test_a_step_that_fails_before_its_record_is_voided_and_blocks_nothing():
    """A verify step commits, then postprocess never records it. The
    worker voids it with its failed rows; later admits, of other requests too, go on."""
    from spec_toy import _batch

    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.add(2, PROMPTS[2], 30)
    batch = _batch("decode", {1: torch.tensor([driver.out[1][-1]])}, driver._max)
    eng.prepare_inputs(batch)
    eng.exec_and_postprocess(batch)  # committed, never cut or recorded
    assert eng._drafts[NODE].ledger.unrecorded(1)
    eng.void_verify(batch)
    eng.remove_request(1)
    driver.step([2])
    driver.add(3, PROMPTS[3], 30)
    driver.run([2, 3])
    assert driver.out[2] == reference(PROMPTS[2], 30)
    assert driver.out[3] == reference(PROMPTS[3], 30)


def test_teardown_purges_the_ledger_before_the_streams():
    """No take-back may find a row whose streams are gone."""
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.step([1])
    order = []
    ledger, kv = eng._drafts[NODE].ledger, eng._resources["kv"]
    real_ledger, real_kv = ledger.remove_request, kv.remove_request
    ledger.remove_request = lambda rid: (order.append("ledger"), real_ledger(rid))
    kv.remove_request = lambda rid: (order.append("kv"), real_kv(rid))
    eng.remove_request(1)
    assert order.index("ledger") < order.index("kv")


def test_a_request_decodes_to_exactly_the_client_limit():
    """Synchronous mode on the engine: 256 positions, every draft accepted."""
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    budget = 256 - len(prompt)
    out, eng = _decode(3, oracle, max_tokens=budget, prompts={1: prompt})
    assert out[1] == reference(prompt, budget)


def _toy_specs():
    """The toy node's specs by key, as `Engine.load_model` hands them to `_agreed_mode`."""
    from spec_toy import specs

    return {spec.resource_key: spec for spec in specs(3)}


class _Joint:
    """A lockstep instance of two ranks whose other rank can run asynchronous
    mode (``other``) and can verify at all (``verifies``)."""

    def __init__(self, other: int, rank: int = 0, verifies: int = 1):
        self.world_size, self.rank, self.answers, self.asked = 2, rank, (other, verifies), []

    def agree(self, values, op=None):
        self.asked.append(values)
        return [min(v, a) for v, a in zip(values, self.answers, strict=True)]


@pytest.mark.parametrize("other,mode,expect", [
    (1, "auto", "asynchronous"), (0, "auto", "synchronous"), (1, "synchronous", "synchronous"),
])
def test_the_mode_is_agreed_across_the_lockstep_instance(monkeypatch, other, mode, expect):
    """One MIN reduction; a rank that can't run asynchronous
    mode makes every rank synchronous under auto."""
    from spec_toy import ToyModel

    from mstar.engine.resources.draft.config import DraftConfig

    eng = engine(3, oracle, mode=mode)
    mgmt = eng._submodules[NODE]
    joint = mgmt.joint_comm_group = _Joint(other)
    decl = ToyModel(3, mode).verify_nodes()[0]
    assert isinstance(decl.config, DraftConfig)
    got, reason = eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
    assert got.value == expect and joint.asked == [[int(mode == "auto"), 1]]


def test_explicit_async_refuses_on_every_rank_when_one_cant():
    from spec_toy import ToyModel

    eng = engine(3, oracle, mode="asynchronous")
    mgmt = eng._submodules[NODE]
    mgmt.joint_comm_group = _Joint(0)
    with pytest.raises(RuntimeError, match="another rank"):
        eng._agreed_mode(ToyModel(3, "asynchronous").verify_nodes()[0], mgmt, _toy_specs(), (True, ""))


def test_a_rank_that_cant_verify_joins_the_agreement_before_it_raises(monkeypatch):
    """A per-rank refusal (an env knob, a GPU without the
    explicit backend) raising before the reduction would leave the other ranks
    waiting in it. It agrees first, then every rank raises."""
    from spec_toy import ToyModel

    from mstar.utils import knobs

    eng = engine(3, oracle)
    mgmt = eng._submodules[NODE]
    joint = mgmt.joint_comm_group = _Joint(1)
    monkeypatch.setattr(knobs, "device_loopback_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="MSTAR_DEVICE_LOOPBACK"):
        eng._agreed_mode(ToyModel(3).verify_nodes()[0], mgmt, _toy_specs(), (True, ""))
    assert joint.asked == [[0, 0]]


def test_every_rank_raises_when_another_cant_verify():
    from spec_toy import ToyModel

    eng = engine(3, oracle)
    mgmt = eng._submodules[NODE]
    mgmt.joint_comm_group = _Joint(1, verifies=0)
    with pytest.raises(RuntimeError, match="another rank of the lockstep instance can't verify"):
        eng._agreed_mode(ToyModel(3).verify_nodes()[0], mgmt, _toy_specs(), (True, ""))


def _captured(*buckets):
    """A real runner whose capture left ``buckets``, each with a slot to replay."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._buckets = {bucket: SimpleNamespace(slots=[object()]) for bucket in buckets}
    runner.dropped_buckets = []
    return runner


def test_an_async_verify_walk_capture_left_empty_fails_startup():
    """A batch cap of 0 would hang every request after its prompt."""
    from mstar.engine.resources.step import BucketKey

    eng = engine(3, oracle, mode="asynchronous")
    eng._submodules[NODE].resident = SimpleNamespace(host_planned_by={})
    empty = SimpleNamespace(max_captured_bs=lambda walk: 0, dropped_buckets=["decode bs=1"])
    with pytest.raises(RuntimeError, match="mode: synchronous"):
        eng._refuse_uncaptured_verify({NODE: empty})
    eng._refuse_uncaptured_verify({NODE: _captured(BucketKey(eng._drafts[NODE].walk, 4, 16))})
    engine(3, oracle)._refuse_uncaptured_verify({})  # a synchronous node needs no capture


def test_an_async_verify_bucket_a_flashinfer_attention_planned_on_the_host_fails_startup():
    """A FlashInfer attention whose verify bucket got a wrapper other than FA3's
    (no verify=True on its step) host-plans it, over the reserved lengths. The
    check reads what capture decided; a host-planned bucket of another walk
    doesn't count."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.base import CGSlotKey
    from mstar.engine.resources.resident import ResidentPlanner
    from mstar.engine.resources.step import BucketKey, Segment, SubmoduleStep

    eng = engine(3, oracle, mode="asynchronous")
    walk, span, bs = eng._drafts[NODE].walk, 4, 2
    bucket = BucketKey(graph_walk=walk, bs=bs, num_tokens=bs * span)
    manager = FlashInferManager.__new__(FlashInferManager)
    manager._kv_cache_name = "attn"
    # a prefill wrapper: plans on the host, reads no tables
    plain = SimpleNamespace(plans_in_graph=False, supports_gather_resident=False)
    manager._cg_plan_states = {CGSlotKey(bucket=bucket, slot=0, label="main"): plain}
    # later in the stack the manager also looks up a lagged binding first; none here
    manager._lagged_cg = {}
    planner = ResidentPlanner(NODE, {walk}, {"attn": manager}, torch.device("cpu"), max_bs=bs, max_span=span)
    step = SubmoduleStep(steps={}, segments=[Segment(r, "main", span) for r in (1, 2)])
    planner.gather(planner.capture_step(bucket, 0, step))
    assert planner.host_planned_by == {bucket: {"attn"}}
    eng._submodules[NODE].resident = planner

    with pytest.raises(RuntimeError, match=r"capture planned .*\['attn'\].* on the host.*verify=True"):
        eng._refuse_uncaptured_verify({NODE: _captured(bucket)})
    # a host-planned bucket of another walk, beside a verify bucket that read the tables alone
    other = BucketKey(graph_walk=f"{walk}_other", bs=bs, num_tokens=bs)
    planner.host_planned_by[other] = frozenset({"attn"})
    exact = BucketKey(graph_walk=walk, bs=1, num_tokens=span)
    eng._refuse_uncaptured_verify({NODE: _captured(exact, other)})
    # synchronous mode host-plans its verify steps from exact host lengths
    sync = engine(3, oracle)
    sync._submodules[NODE].resident = planner
    sync._refuse_uncaptured_verify({NODE: _captured(bucket)})


def test_a_verify_walk_reads_its_block_off_the_loop_back_slot():
    """A looped-back verify row runs k + 1 tokens, and the step still
    builds per-request outputs for the cut and the client."""
    eng = engine(3, oracle)
    sub = eng._submodules[NODE].submodule
    assert sub.uniform_row_inputs("decode", has_loopback=True).input_seq_len == 4
    assert sub.device_loopback_signals("decode") == frozenset()
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 20)
    driver.step([1])
    # the first verify step read its token routed; the next reads the slot
    assert eng._loopback_rows(SimpleNamespace(
        node_name=NODE, request_ids=[1], step_context=SimpleNamespace(graph_walk="decode", request_walks={}),
    )) == [True]


def test_a_verify_step_whose_stream_rows_span_less_than_the_block_is_refused():
    """Whatever path prepared the rows, each counts k + 1 on the streams that take back."""
    from mstar.engine.resources.kv.config import KVStep
    from mstar.engine.resources.position.config import PositionStep
    from mstar.engine.resources.step import Segment, SubmoduleStep

    eng = engine(3, oracle)
    draft = eng._drafts[NODE]
    good = [Segment(1, "main", 4), Segment(2, "main", 4)]
    draft.refuse_step(SubmoduleStep(steps={"kv": KVStep(), "rope": PositionStep()}, segments=good))
    short = [Segment(1, "main", 4), Segment(2, "main", 1)]
    with pytest.raises(RuntimeError, match="k \\+ 1 = 4"):
        draft.refuse_step(SubmoduleStep(steps={"kv": KVStep(), "rope": PositionStep()}, segments=short))
def test_the_engine_adds_a_drafters_own_steps_to_its_prompt_and_verify_steps():
    """`drafter_step`, from `engine_step`: a drafter's resource steps (`VerifyNode.prefill_steps` / `verify_steps`)
    join the node's prompt and verify steps, fresh each step, and a target that declares one itself is refused."""
    from mstar.engine.resources.taps.config import TapStep
    from mstar.model.submodule_base import ARNodeInputs, engine_step

    sub = engine(3, oracle)._submodules[NODE].submodule
    sub.drafter_share = sub.drafter_share._replace(prompt_steps={"p": TapStep()}, verify_steps={"v": TapStep()})
    for walk, span, own in (("prefill", 2, "p"), ("decode", 4, "v")):
        kwargs = dict(graph_walk=walk, request_ids=[1], inputs=[ARNodeInputs(input_seq_len=span)])
        assert own not in sub.declare_step(**kwargs).steps
        first, second = engine_step(sub, **kwargs), engine_step(sub, **kwargs)
        assert own in first.steps and first.steps[own] is not second.steps[own]
    real = sub.declare_step

    def doubled(*args, **kw):
        step = real(*args, **kw)
        return dataclasses.replace(step, steps={**step.steps, "v": TapStep()})

    sub.declare_step = doubled
    with pytest.raises(RuntimeError, match="its drafter's steps, which the engine adds"):
        engine_step(sub, **kwargs)


def test_a_drafters_layer_set_bound_by_other_than_its_layers_is_refused():
    """The pool places a stack that joined it as a layer set; its attention
    layers name the set (kv_layer_set), one a layer. A stack that forgot the set
    would write the target's first layers."""
    from types import SimpleNamespace

    from torch import nn

    from mstar.engine.engine import _layer_set_problem
    from mstar.engine.resources.convenience import AttentionCallable
    from mstar.engine.resources.kv.config import KVLayerSet

    joined = KVLayerSet("kv", "draft", 2, 1, 8)
    pool = SimpleNamespace(layer_set=lambda name: SimpleNamespace(first=4, num_layers=2))

    def stack(*layer_sets):
        layers = nn.ModuleList()
        for name in layer_sets:
            layer = nn.Module()
            layer.attend = AttentionCallable(kv=pool, layer_set=name)
            layers.append(layer)
        return layers

    assert _layer_set_problem(stack("draft", "draft"), {"kv": pool}, joined) is None
    # what a set's layer writes and attends: its own index from where the pool placed the set
    seen = []

    class _Cursor:
        def set_default_layer_idx(self, idx):
            seen.append(idx)

    kv, attn = _Cursor(), _Cursor()
    kv.layer_set = pool.layer_set
    AttentionCallable(kv=kv, attn=attn, layer_set="draft").set_layer_idx(1)
    assert seen == [5, 5]
    seen.clear()
    AttentionCallable(kv=kv, attn=attn).set_layer_idx(1)
    assert seen == [1, 1]
    for wrong in (stack("draft"), stack(None, None), stack("draft", "draft", "draft")):
        assert "attention layers declare it" in _layer_set_problem(wrong, {"kv": pool}, joined)
    assert _layer_set_problem(stack(None), {"kv": pool}, None) is None


def test_a_verifying_node_that_sets_no_drafter_is_refused(monkeypatch):
    from mstar.engine.engine import Engine

    real_load = Engine.load_model

    def load_with_no_drafter(self, submodules, *args, **kwargs):
        submodules[NODE].drafter = None
        return real_load(self, submodules, *args, **kwargs)

    monkeypatch.setattr(Engine, "load_model", load_with_no_drafter)
    with pytest.raises(RuntimeError, match="must set `drafter`"):
        engine(3)


def test_a_verify_row_declaring_another_span_fails_before_admit(monkeypatch, caplog):
    from spec_toy import _batch

    from mstar.model.submodule_base import ARNodeInputs

    eng = engine(3)
    driver = Driver(eng)
    driver.add(1, [5, 9], 30)
    sub = eng._submodules[NODE].submodule
    monkeypatch.setattr(sub, "prepare_inputs", lambda graph_walk, fwd_info, inputs, **kwargs: ARNodeInputs(
        input_ids=inputs["ids"][0].reshape(-1)[-1:], input_seq_len=3,
    ))
    batch = _batch("decode", {1: torch.tensor([driver.out[1][-1]])}, driver._max)
    eng.prepare_inputs(batch)
    assert 1 in batch.failed_requests and "prepare_inputs declared 3" in caplog.text
    assert batch.verify_seq is None




def test_a_verify_step_times_its_cut_take_back_and_commit_under_fixed_names(monkeypatch):
    """Three nested timers (benchmark/worker_phases/README.md), one sample of
    the cut and of the commit a verify step, and the take-back pass on every
    admit of the node."""
    from mstar.utils import profiler

    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    buf = profiler._PhaseBuffer(list)
    monkeypatch.setattr(profiler, "_PHASE_BUF", buf)
    driver = Driver(engine(3, oracle))
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, 24)
    steps = 4
    for _ in range(steps):
        driver.step(list(PROMPTS))
    assert len(buf["engine.verify.cut"]) == len(buf["engine.verify.cut.cpu"]) == steps
    # the ledger sees every commit and admit of the node, its prompts' included
    assert len(buf["engine.verify.commit"]) == len(PROMPTS) + steps
    assert len(buf["res.verify.take_back"]) == len(PROMPTS) + steps


def test_the_engine_turns_rebuild_off_for_a_verifying_node(caplog, monkeypatch):
    """`rebuild_policies_of` leaves the verifying node out (the engine logs it
    once), so neither the engine's check nor the worker's rebuilds ask its
    resources to drop; another node keeps its policy, and the worker refuses a
    policy left for a node the engine verifies on."""
    import logging

    from spec_toy import ToyModel

    from mstar.model.rebuild import RebuildPolicy
    from mstar.testing_utils.resource import StubResource
    from mstar.worker.worker import _make_rebuilds

    policy = RebuildPolicy(walks=frozenset({"prefill"}), ids_input="ids", sample_output="new_token", loop="decode_loop")
    model = ToyModel(3)
    model.rebuild_policies = lambda: {NODE: policy, "other": policy}
    eng = engine(3, oracle)
    # a resource that can't drop a request's state, as a drafter's taps can't
    eng._submodules[NODE].resources["taps"] = StubResource()
    # negative control: the same check over the model's own policies
    unfiltered = SimpleNamespace(rebuild_policies=lambda m: m.rebuild_policies(), _submodules=eng._submodules)
    with pytest.raises(ValueError, match="can't drop"):
        type(eng)._check_rebuild_policies(unfiltered, model)

    with caplog.at_level(logging.INFO, logger="mstar.engine.engine"):
        assert eng.rebuild_policies(model) == {"other": policy}
        eng._check_rebuild_policies(model)
    assert sum("rebuild is off" in r.getMessage() for r in caplog.records) == 1

    # the worker, on the verifying node alone; the toy's graphs are stand-ins, so the policy's graph checks pass
    monkeypatch.setattr(RebuildPolicy, "check", lambda self, node, graphs: dict.fromkeys(self.walks, frozenset()))
    monkeypatch.setattr(RebuildPolicy, "loop_walks", lambda self, node, graphs: frozenset())
    model.rebuild_policies = lambda: {NODE: policy}
    cpu = torch.device("cpu")
    # every walk runs here, the decode loop's included
    walks = {(node, walk) for node in (NODE, "other") for walk in model.get_graph_walk_graphs()}
    # a policy left for a node the engine verifies on, from a model that doesn't declare it, is refused
    with monkeypatch.context() as m:
        m.setattr(model, "verify_nodes", lambda: [])
        with pytest.raises(RuntimeError, match="verifying nodes"):
            _make_rebuilds(model, {}, walks, set(), cpu, eng, verifying_nodes={NODE})
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        rebuilds = _make_rebuilds(model, {"rebuild_host_mb": 1}, walks, set(), cpu, eng, verifying_nodes={NODE})
    assert NODE not in rebuilds.nodes and rebuilds.pool is None
    assert any("rebuild_host_mb is set" in r.getMessage() for r in caplog.records)
    # no warning while another node of the worker keeps a policy, which the pool serves
    import mstar.worker.worker as worker_mod

    caplog.clear()
    built = []
    monkeypatch.setattr(worker_mod, "Rebuilds", lambda policies, graphs, pool, *rest: built.append((policies, pool)))
    model.rebuild_policies = lambda: {NODE: policy, "other": policy}
    # "other" is a node this toy engine doesn't build, so it has no dtype to report
    filtered = SimpleNamespace(host_allowance=None, node_dtype=lambda node: None)
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        _make_rebuilds(model, {"rebuild_host_mb": 1}, walks, set(), cpu, filtered, verifying_nodes={NODE})
    assert built[0][0] == {"other": policy} and built[0][1] is not None
    assert not any("rebuild_host_mb is set" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("backend,reason,verify,host", [
    ("auto", None, None, None),
    ("fa3", None, None, None),
    ("flashinfer", "no FA3 here", None, None),
    ("auto", "no FA3 here", None, "verify attention plans on the host (no FA3 here)"),
    ("fa3", "no FA3 here", "verify_backend fa3: no FA3 here", None),
])
def test_flashinfers_capabilities_say_what_this_gpu_cant_and_bind_picks_the_kernel(backend, reason, verify, host):
    """A worker's own refusal, by severity: a pinned fa3 it can't run can't verify, auto
    without FA3 can't be asynchronous. The kernel is chosen at bind, after the mode is agreed."""
    from mstar.engine.resources.attn.config import VerifyBackend
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    manager = FlashInferManager.__new__(FlashInferManager)
    manager._verify_backend = VerifyBackend.parse(backend)
    manager.__dict__["_fa3_reason"] = reason
    manager._verify_fa3 = False
    caps = manager.capabilities()
    assert (caps.verify_refusal, caps.async_verify_refusal) == (verify, host)
    assert manager._verify_fa3 is False  # asking changes nothing
    manager.bind_verify_walk("decode", 4, 8)
    assert manager._verify_fa3 is (backend != "flashinfer" and reason is None)


@pytest.mark.parametrize("refusal", ["verify", "async", None])
def test_a_workers_own_capability_refusal_enters_the_one_agreement(monkeypatch, refusal):
    """Invariant 11: a rank's own resource refusal is folded into the lockstep instance's one
    reduction, never raised before it (its peers would wait in it) nor applied after it (the ranks
    would run different modes). With none here, a peer's refusal makes this rank synchronous too."""
    from spec_toy import ToyModel

    from mstar.engine.resources.spec import ResourceCapabilities

    eng = engine(3, oracle, mode="auto")
    mgmt = eng._submodules[NODE]
    joint = mgmt.joint_comm_group = _Joint(0 if refusal is None else 1)
    caps = ResourceCapabilities(
        verify_refusal="cannot here" if refusal == "verify" else None,
        async_verify_refusal="cannot here" if refusal == "async" else None,
    )
    monkeypatch.setattr(mgmt.resources["kv"], "capabilities", lambda: caps)
    decl = ToyModel(3, "auto").verify_nodes()[0]
    if refusal == "verify":
        with pytest.raises(RuntimeError, match="resource 'kv': cannot here"):
            eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
        # asked, can't verify: every rank raises after the one reduction
        assert joint.asked == [[1, 0]]
        return
    got, reason = eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
    assert got.value == "synchronous"
    if refusal == "async":
        assert joint.asked == [[0, 1]] and "resource 'kv': cannot here" in reason
    else:
        assert joint.asked == [[1, 1]] and "another rank" in reason


@pytest.mark.parametrize("mode", ["auto", "asynchronous"])
def test_a_verifying_node_that_shares_a_resource_never_runs_asynchronous(mode):
    """Another node's offload releases this node's slots on the resources they share, and marks no
    verify ledger but its own; so such a node runs synchronous, whose rule needs no mark."""
    from spec_toy import ToyModel

    eng = engine(3, oracle, mode=mode)
    mgmt = eng._submodules[NODE]
    eng._resource_nodes["rope"] = frozenset({NODE, "other"})
    decl = ToyModel(3, mode).verify_nodes()[0]
    if mode == "asynchronous":
        with pytest.raises(RuntimeError, match="resource 'rope' is shared by nodes"):
            eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
        return
    got, reason = eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
    assert got.value == "synchronous" and "resource 'rope' is shared by nodes ['LLM', 'other']" in reason


def test_the_built_resources_must_be_the_declared_ones():
    from spec_toy import ToyModel

    eng = engine(3, oracle)
    mgmt = eng._submodules[NODE]
    specs = dict(_toy_specs())
    specs.pop(next(iter(specs)))
    with pytest.raises(RuntimeError, match="specs .* and resources .* differ"):
        eng._agreed_mode(ToyModel(3).verify_nodes()[0], mgmt, specs, (True, ""))


@pytest.mark.parametrize("refusal,mode,raises", [
    ("verify", "auto", "speculative decoding on LLM: resource 'kv': cannot here"),
    ("async", "auto", None),
    ("async", "asynchronous", "asynchronous verify mode was asked for, but resource 'kv': cannot here"),
])
def test_a_workers_own_refusal_is_fatal_or_drops_the_node_to_synchronous(monkeypatch, refusal, mode, raises):
    from spec_toy import ToyModel

    from mstar.engine.resources.spec import ResourceCapabilities

    eng = engine(3, oracle, mode=mode)
    mgmt = eng._submodules[NODE]
    caps = ResourceCapabilities(
        verify_refusal="cannot here" if refusal == "verify" else None,
        async_verify_refusal="cannot here" if refusal == "async" else None,
    )
    monkeypatch.setattr(mgmt.resources["kv"], "capabilities", lambda: caps)
    decl = ToyModel(3, mode).verify_nodes()[0]
    if raises is not None:
        with pytest.raises(RuntimeError, match=re.escape(raises)):
            eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
        return
    got, reason = eng._agreed_mode(decl, mgmt, _toy_specs(), (True, ""))
    assert got.value == "synchronous" and "resource 'kv': cannot here" in reason


def test_a_verifying_node_on_a_pool_with_no_host_pool_is_warned_about(caplog):
    """With no host pool every request on a verifying node is guaranteed, so the leader says so once at load.
    Negative control: a pool with host pages gets no warning."""
    import logging

    from mstar.engine import engine as engine_mod
    from mstar.engine.resources.kv.manager import KVManager

    with caplog.at_level(logging.WARNING, logger="mstar.engine.engine"):
        engine(3, oracle)
    warned = [r.getMessage() for r in caplog.records if "no host pool" in r.getMessage()]
    assert len(warned) == 1 and "KV 'kv'" in warned[0], warned
    caplog.clear()
    hosted = KVManager.__new__(KVManager)
    hosted.config = SimpleNamespace(cpu_offload_pages=4)
    with caplog.at_level(logging.WARNING, logger="mstar.engine.engine"):
        engine_mod._warn_hostless(NODE, {"kv": hosted})
    assert not any("no host pool" in r.getMessage() for r in caplog.records)
