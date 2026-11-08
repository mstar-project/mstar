"""Speculative decoding end to end on the real `Engine`, synchronous mode, on
the CPU: a toy target whose tokens depend on its KV cache and positions, and
scripted drafters (`spec_toy.py`). Greedy output equals plain greedy decoding
whatever the drafts, and each mechanism has a test that fails without it.
"""
from __future__ import annotations

import dataclasses
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


def test_an_offload_settles_its_victims_recorded_results_first(monkeypatch):
    snapshots: dict[int, int] = {}
    _stub_offload(monkeypatch, snapshots)
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.add(2, PROMPTS[2], 30)
    driver.step([1, 2])
    driver.step([1])
    ledger = eng._drafts[NODE].ledger
    # row 2's result is recorded but not applied: a victim, settled at the offload
    assert ledger.pending(2) and not eng.verify_unrecorded(2)
    assert eng.offload_request(NODE, 2) > 0
    assert snapshots[2] == len(PROMPTS[2]) + len(driver.out[2]) - 1
    assert not ledger.pending(2)
    driver.run([1, 2])
    for rid in (1, 2):
        assert driver.out[rid] == reference(PROMPTS[rid], 30)


def test_without_the_settle_an_offload_snapshots_the_reserved_length(monkeypatch):
    """Negative control: the victim's host length still counts its rejected drafts."""
    from mstar.engine.engine import Engine

    snapshots: dict[int, int] = {}
    _stub_offload(monkeypatch, snapshots)
    monkeypatch.setattr(Engine, "_settle_verify", lambda self, node, rid: None)
    eng = engine(3, always_wrong)
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.add(2, PROMPTS[2], 30)
    driver.step([1, 2])
    driver.step([1])
    eng.offload_request(NODE, 2)
    assert snapshots[2] > len(PROMPTS[2]) + len(driver.out[2]) - 1


def test_a_row_with_an_unrecorded_result_is_no_victim():
    eng = engine(3, every(2))
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    ledger = eng._drafts[NODE].ledger
    ledger.committed([1], {"kv": "main", "rope": "main"}, [4])
    assert eng.verify_unrecorded(1)
    with pytest.raises(RuntimeError, match="unrecorded"):
        ledger.settle(1)


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
    with pytest.raises(RuntimeError, match=f"speculative decoding on {NODE}: KV .* is read by another instance"):
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
    got, reason = eng._agreed_mode(decl, mgmt, {}, (True, ""))
    assert got.value == expect and joint.asked == [[int(mode == "auto"), 1]]


def test_explicit_async_refuses_on_every_rank_when_one_cant():
    from spec_toy import ToyModel

    eng = engine(3, oracle, mode="asynchronous")
    mgmt = eng._submodules[NODE]
    mgmt.joint_comm_group = _Joint(0)
    with pytest.raises(RuntimeError, match="another rank"):
        eng._agreed_mode(ToyModel(3, "asynchronous").verify_nodes()[0], mgmt, {}, (True, ""))


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
        eng._agreed_mode(ToyModel(3).verify_nodes()[0], mgmt, {}, (True, ""))
    assert joint.asked == [[0, 0]]


def test_every_rank_raises_when_another_cant_verify():
    from spec_toy import ToyModel

    eng = engine(3, oracle)
    mgmt = eng._submodules[NODE]
    mgmt.joint_comm_group = _Joint(1, verifies=0)
    with pytest.raises(RuntimeError, match="another rank of the lockstep instance can't verify"):
        eng._agreed_mode(ToyModel(3).verify_nodes()[0], mgmt, {}, (True, ""))


def test_an_async_verify_walk_capture_left_empty_fails_startup():
    """A batch cap of 0 would hang every request after its prompt."""
    from types import SimpleNamespace

    eng = engine(3, oracle, mode="asynchronous")
    empty = SimpleNamespace(max_captured_bs=lambda walk: 0, dropped_buckets=["decode bs=1"])
    with pytest.raises(RuntimeError, match="mode: synchronous"):
        eng._refuse_uncaptured_verify({NODE: empty})
    eng._refuse_uncaptured_verify({NODE: SimpleNamespace(max_captured_bs=lambda walk: 4, dropped_buckets=[])})
    engine(3, oracle)._refuse_uncaptured_verify({})  # a synchronous node needs no capture


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
    from mstar.engine.resources.draft.taps import TapStep
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


def test_a_drafter_drafts_from_its_carry_which_the_prompt_and_each_observe_store():
    """A drafter whose state crosses steps declares a carry; the engine keeps it
    in the sampler's loop-back rows (`DRAFT_CARRY`): the prompt's last prefill
    stores it, each verify step reads it before ``propose`` and stores what
    ``observe`` returns. Drafted from a correct carry, every draft is accepted."""
    from spec_toy import carry_oracle

    k = 3
    eng = engine(k, carry_oracle)
    out, accepted = _decode_with_accepted(eng, k)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference(prompt, 24)
    assert set(accepted) == {k}, accepted


def test_without_the_prompts_carry_the_first_step_drafts_wrong_and_output_holds():
    from spec_toy import carry_oracle

    k = 3
    eng = engine(k, carry_oracle)
    eng._submodules[NODE].submodule.skip_prefill_carry = True
    out, accepted = _decode_with_accepted(eng, k)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference(prompt, 24)
    # every step drafts from a sum short of the prompt's (a running sum carries the error): nothing is
    # accepted, and the verify alone keeps the output exact
    assert set(accepted) == {0}, accepted


def test_the_engine_stores_the_carry_through_prompt_and_verify_steps_that_declare_none():
    """The carry is the drafter's: the toy's steps list no loop-back writes, and the engine adds
    `DRAFT_CARRY` to the sampler step of its prompt and verify steps (`drafter_step`)."""
    from spec_toy import carry_oracle

    from mstar.engine.resources.draft.drafter import DRAFT_CARRY
    from mstar.model.submodule_base import ARNodeInputs, engine_step

    sub = engine(3, carry_oracle)._submodules[NODE].submodule
    for walk, span in (("prefill", 2), ("decode", 4)):
        kwargs = dict(graph_walk=walk, request_ids=[1], inputs=[ARNodeInputs(input_seq_len=span)])
        assert sub.declare_step(**kwargs).steps["sampler"].loopback_writes == frozenset()
        assert engine_step(sub, **kwargs).steps["sampler"].loopback_writes == {DRAFT_CARRY}


def test_a_drafter_that_reads_the_targets_hidden_states_observes_what_the_target_returned():
    """``DrafterNeeds.target_hidden``: a verify step's target returns ``(logits, hidden)``, and ``observe`` gets that
    hidden."""
    from spec_toy import _CarryDrafter

    class ReadsHidden(_CarryDrafter):
        def __init__(self, k):
            super().__init__(k)
            self.needs = self.needs._replace(target_hidden=True)
            self.seen = []

        def observe(self, block, result, hidden):
            self.seen.append(hidden)
            return super().observe(block, result)

    k = 3
    eng = engine(k, ReadsHidden)
    sub = eng._submodules[NODE].submodule
    target, returned = sub._target, []

    def with_hidden(engine_inputs, ids):
        logits = target(engine_inputs, ids)
        if engine_inputs.step.ctx.graph_walk != "decode":
            return logits
        returned.append(logits.clone())
        return logits, returned[-1]

    sub._target = with_hidden
    out, _ = _decode_with_accepted(eng, k)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference(prompt, 24)
    assert returned and all(a is b for a, b in zip(sub.drafter.seen, returned, strict=True))


def _decode_with_accepted(eng, k):
    driver = Driver(eng)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, 24)
    accepted = []
    real = eng.cut_verify_step

    def cut(batch, outputs, host):
        accepted.extend(host.buffers["verify_accepted"].tolist()[: len(host.request_ids)])
        return real(batch, outputs, host)

    eng.cut_verify_step = cut
    driver.run(list(PROMPTS))
    return driver.out, accepted


@pytest.mark.parametrize("ingraph", [True, False])
def test_the_loopback_masters_a_captured_carry_reads_are_never_replaced(ingraph):
    """With the in-graph scatter, a loop-back state's master is allocated once
    at the in-graph rows, as offset and last_token are: registrations past the
    capture's batch size leave the tensor a captured graph holds in place.
    Without it the master grows (a new tensor), which is why a carrying drafter
    is refused there."""
    from mstar.engine.resources.sampler.config import LoopbackState
    from mstar.engine.resources.sampler.utils import SamplerBuffers

    bufs = SamplerBuffers.allocate(
        max_batch_size=4, device=torch.device("cpu"), ingraph_scatter=ingraph, slots=64,
        loopback_state={"draft_carry": LoopbackState(2, torch.float32, (0.0, 0.0))},
    )
    master = bufs.loopback["draft_carry"].master
    for rid in range(10):
        bufs.register_request(rid)
    bufs.loopback_for("draft_carry", list(range(10)))
    same = bufs.loopback["draft_carry"].master is master
    assert same is ingraph
    if ingraph:
        assert master.shape[0] == 65


def test_a_drafter_still_unbound_at_load_is_refused():
    from spec_toy import _Drafter

    from mstar.engine.resources.draft.drafter import UNBOUND_KV_LAYER

    class Unbound(_Drafter):
        kv_layer = UNBOUND_KV_LAYER

        @property
        def kv_layers_unbound(self) -> bool:
            return self.kv_layer == UNBOUND_KV_LAYER

    with pytest.raises(RuntimeError, match="its drafter was never bound"):
        engine(3, Unbound)


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


def test_a_carrying_drafter_is_refused_where_the_loopback_masters_grow(monkeypatch):
    from spec_toy import carry_oracle

    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "0")
    with pytest.raises(RuntimeError, match="loop-back rows, which grow"):
        engine(3, carry_oracle)


def _two_rank_node(monkeypatch, rank):
    """The toy's node in a fake lockstep instance of two ranks, its pool with a KV head a rank; the load's one vote
    passes through."""
    import spec_toy

    from mstar.distributed.communication import CommGroup, JointGroups, WorkerParallelGroups

    real_specs = spec_toy.specs

    def a_head_a_rank(*args, **kwargs):
        kv, *rest = real_specs(*args, **kwargs)
        return [dataclasses.replace(kv, config=dataclasses.replace(kv.config, num_kv_heads=2, num_qo_heads=2)), *rest]

    monkeypatch.setattr(spec_toy, "specs", a_head_a_rank)
    group = CommGroup(rank, rank, [0, 1])
    joint = JointGroups(tp_group=group, sp_group=CommGroup.trivial(), instance=group)
    monkeypatch.setattr(JointGroups, "agree", lambda self, values, op=None: values)
    monkeypatch.setattr(WorkerParallelGroups, "get_joint_group_for_node", lambda self, node: joint)


@pytest.mark.parametrize("rank", [0, 1])
def test_a_carrying_drafter_is_refused_above_one_rank(monkeypatch, rank):
    """On every rank of the instance: above one rank the masters grow, whatever a rank's environment says."""
    from spec_toy import carry_oracle

    _two_rank_node(monkeypatch, rank)
    with pytest.raises(RuntimeError, match="not yet under TP"):
        engine(3, carry_oracle)


def test_a_carry_survives_many_requests_a_recycled_handle_and_an_offload(monkeypatch):
    """The carry oracle drafts the target's tokens only from a right carry, so
    every draft is accepted exactly when each row's carry is its own: ten
    requests (past the sampler buffers' eight), then the same handles again
    for new prompts, one of them offloaded mid-decode (the KV offload stubbed:
    its length is recorded, its pages stay)."""
    from spec_toy import carry_oracle

    snapshots: dict[int, int] = {}
    _stub_offload(monkeypatch, snapshots)
    k = 3
    eng = engine(k, carry_oracle)
    accepted: dict[int, list[int]] = {}
    real = eng.cut_verify_step

    def cut(batch, outputs, host):
        counts = host.buffers["verify_accepted"].tolist()
        for i, rid in enumerate(host.request_ids):
            accepted.setdefault(rid, []).append(counts[i])
        return real(batch, outputs, host)

    eng.cut_verify_step = cut
    first = {rid: [rid % 7 + 1, rid % 5 + 2, rid % 3 + 3] for rid in range(1, 11)}
    driver = Driver(eng)
    for rid, prompt in first.items():
        driver.add(rid, prompt, 20)
    driver.run(list(first))
    assert all(driver.out[rid] == reference(prompt, 20) for rid, prompt in first.items())
    # every request's verify steps but its last (cut by the budget) accept every draft
    assert all(set(counts[:-1]) <= {k} for counts in accepted.values()), accepted
    accepted.clear()
    again = {rid: [rid % 4 + 9, 11, rid % 6 + 1, 2] for rid in range(1, 5)}
    for rid, prompt in again.items():
        driver.add(rid, prompt, 20)
    driver.step(list(again))
    assert eng.offload_request(NODE, 2) > 0
    driver.run(list(again))
    for rid, prompt in again.items():
        assert driver.out[rid] == reference(prompt, 20)
    # the recycled handles too, before and after the offload (its KV is stubbed: lengths, not pages)
    assert all(set(counts[:-1]) <= {k} for counts in accepted.values()), accepted


def test_a_prompt_step_with_no_sampler_step_to_store_the_carry_through_is_refused():
    """The prompt step stores a request's first carry: without one, every first verify step
    would draft from a default row."""
    from spec_toy import carry_oracle

    from mstar.model.submodule_base import ARNodeInputs, engine_step

    sub = engine(3, carry_oracle)._submodules[NODE].submodule
    real = sub.declare_step

    def no_sampler(*args, **kwargs):
        step = real(*args, **kwargs)
        return dataclasses.replace(step, steps={k: v for k, v in step.steps.items() if k != "sampler"})

    sub.declare_step = no_sampler
    with pytest.raises(RuntimeError, match="'prefill' step declares no 'sampler' step"):
        engine_step(sub, graph_walk="prefill", request_ids=[1], inputs=[ARNodeInputs(input_seq_len=2)])
