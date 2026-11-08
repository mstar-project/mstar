"""Speculative decoding: when a verify result is applied, exactly once, and how
a verify step's tokens are cut at stops and the budget."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.draft.config import DraftConfig, DraftMode  # noqa: E402
from mstar.engine.resources.draft.drafter import (  # noqa: E402
    VERIFY_ACCEPTED,
    VERIFY_STOPPED,
    VERIFY_TOKENS,
    DrafterNeeds,
    VerifyNode,
)
from mstar.engine.resources.draft.ledger import VerifyLedger  # noqa: E402
from mstar.engine.resources.draft.node import NodeDraft  # noqa: E402
from mstar.model.submodule_base import HostRows, TokenStop  # noqa: E402

K = 3


def _commit(ledger: VerifyLedger, rids, span=K + 1, dropped=()):
    counted = [0 if rid in dropped else span for rid in rids]
    return ledger.committed(rids, {"kv": "main", "rope": "main"}, counted)


def _due(ledger, rids=()):
    takes = ledger.take_due(ledger.next_step, frozenset(rids))
    return {key: dict(zip(t.rids, t.counts, strict=True)) for key, t in takes.items()}


def test_asynchronous_results_come_due_two_steps_on_exactly_once():
    ledger = VerifyLedger("LLM", synchronous=False)
    s1 = _commit(ledger, [1, 2])
    ledger.record(s1, (1, 2), np.array([3, 0]))
    assert ledger.pending(1) and ledger.pending(2)
    # admit of step 2: step 1 is still in flight on the host's view
    assert _due(ledger, [1, 2]) == {}
    s2 = _commit(ledger, [1, 2])
    # admit of step 3: step 1 is due
    assert _due(ledger, [1, 2]) == {"kv": {2: 3}, "rope": {2: 3}}
    # the same admit reached twice (pre-admit, then admit) applies nothing more
    assert _due(ledger, [1, 2]) == {}
    ledger.record(s2, (1, 2), np.array([1, 2]))
    assert ledger.pending(1)
    _commit(ledger, [1])
    # only the step's own rows: row 2 isn't running, so its result waits for its next step
    assert _due(ledger, [1]) == {"kv": {1: 2}, "rope": {1: 2}}
    assert ledger.pending(2) and not ledger.unrecorded(2)
    assert _due(ledger, [2]) == {"kv": {2: 1}, "rope": {2: 1}}
    assert not ledger.pending(2)


def test_synchronous_results_of_the_steps_own_rows_come_due_a_step_on():
    ledger = VerifyLedger("LLM", synchronous=True)
    s1 = _commit(ledger, [1, 2])
    ledger.record(s1, (1, 2), np.array([0, 1]))
    # a step of row 1 alone: its own result is due, row 2's waits
    assert _due(ledger, [1]) == {"kv": {1: 3}, "rope": {1: 3}}
    assert ledger.pending(2) and not ledger.pending(1)
    _commit(ledger, [1])
    # never another row's: row 2's waits for a step of its own
    assert _due(ledger, [5]) == {}
    assert _due(ledger, [2]) == {"kv": {2: 2}, "rope": {2: 2}}


def test_a_due_result_that_was_never_recorded_raises():
    ledger = VerifyLedger("LLM", synchronous=False)
    _commit(ledger, [1])
    _commit(ledger, [1])
    # an unrelated step never absorbs it
    assert _due(ledger, [7]) == {}
    with pytest.raises(RuntimeError, match="never recorded"):
        _due(ledger, [1])


def test_a_voided_step_is_never_due_and_frees_its_rows():
    """A step that committed and failed before postprocess."""
    ledger = VerifyLedger("LLM", synchronous=False)
    s1 = _commit(ledger, [1, 2])
    _commit(ledger, [1])
    ledger.void(s1)
    assert not ledger.unrecorded(2) and not ledger.pending(2)
    assert _due(ledger, [1, 2]) == {}
    # a result for a voided step finds nothing
    ledger.record(s1, (1, 2), np.array([0, 0]))


def test_a_span_commit_dropped_gives_nothing_back():
    ledger = VerifyLedger("LLM", synchronous=True)
    s1 = _commit(ledger, [1, 2], dropped={2})
    ledger.record(s1, (1, 2), np.array([0, 0]))
    assert _due(ledger, [1, 2]) == {"kv": {1: 3}, "rope": {1: 3}}


def test_a_removed_request_takes_its_results_with_it_and_a_recycled_handle_starts_clean():
    ledger = VerifyLedger("LLM", synchronous=False)
    s1 = _commit(ledger, [1, 2])
    ledger.remove_request(1)
    assert not ledger.pending(1)
    # a late result for the removed row is dropped with it
    ledger.record(s1, (1, 2), np.array([2, 2]))
    s2 = _commit(ledger, [2])
    assert _due(ledger, [2]) == {"kv": {2: 1}, "rope": {2: 1}}
    ledger.record(s2, (2,), np.array([K]))
    # handle 1 now names a new request: nothing of the old one reaches it
    s3 = _commit(ledger, [1])
    ledger.record(s3, (1,), np.array([K]))
    # two steps of another walk of the node: counted, nothing to apply
    ledger.committed(None)
    ledger.committed(None)
    assert all(not rows for rows in _due(ledger, [1]).values())


def test_a_result_past_the_counted_span_raises():
    ledger = VerifyLedger("LLM", synchronous=True)
    s1 = _commit(ledger, [1], span=2)
    ledger.record(s1, (1,), np.array([K]))
    with pytest.raises(RuntimeError, match="counted 2 and kept 4"):
        _due(ledger, [1])


def test_recording_a_step_twice_raises():
    ledger = VerifyLedger("LLM", synchronous=False)
    s1 = _commit(ledger, [1])
    ledger.record(s1, (1,), np.array([0]))
    with pytest.raises(RuntimeError, match="twice"):
        ledger.record(s1, (1,), np.array([0]))


# ---- the token cut ----

STOP = 9


def _node(prefill=True, take_back_keys=("kv",)) -> NodeDraft:
    stop = TokenStop(
        loop="decode_loop", buffer="new_token", stop_ids=frozenset({STOP}), sampler="sampler", walk="decode",
        prefill_walks=frozenset({"prefill"}) if prefill else frozenset(),
    )
    decl = VerifyNode("LLM", "decode", DraftConfig("dspark", K, model="d"), DrafterNeeds(), token_stop=stop)
    return NodeDraft(decl, DraftMode.SYNCHRONOUS, "test", "sampler", stop, take_back_keys)


def _info(max_tokens, ignore_eos=False):
    return SimpleNamespace(max_tokens=max_tokens, resource_configs={"sampler": SimpleNamespace(ignore_eos=ignore_eos)})


def _rows(rids, tokens, accepted):
    return HostRows(tuple(rids), {
        VERIFY_TOKENS: torch.tensor(tokens), VERIFY_ACCEPTED: torch.tensor(accepted),
    })


# a verify step's declaration and its KV pool, as `NodeDraft.committed` reads them
_STEP = SimpleNamespace(get=lambda key: None, segments=[SimpleNamespace(label="main", span=K + 1)])
_KV = {"kv": SimpleNamespace(committed_spans=lambda rids, label, span: [span] * len(rids))}


def test_the_cut_keeps_accepted_plus_one_up_to_a_stop_id():
    node = _node()
    seq = node.committed("decode", [1, 2, 3], _STEP, _KV)
    rows = _rows([1, 2, 3], [[4, 5, STOP, 6], [4, 5, 6, 7], [STOP, 1, 2, 3]], [3, 1, 0])
    cut = node.cut(seq, rows, {1: _info(100), 2: _info(100), 3: _info(100)})
    assert cut.counts.tolist() == [3, 2, 1]
    assert cut.stopped.tolist() == [True, False, True]
    assert rows.buffers[VERIFY_STOPPED].tolist() == [True, False, True]


def test_the_budget_cuts_inside_a_block_and_ignore_eos_runs_past_a_stop():
    node = _node()
    step, kv = _STEP, _KV
    seq = node.committed("decode", [1, 2], step, kv)
    # 3 tokens of budget (one went to the prefill), 3 accepted: 3 emitted, stopped
    rows = _rows([1, 2], [[4, 5, 6, 7], [STOP, 5, 6, 7]], [3, 3])
    cut = node.cut(seq, rows, {1: _info(4), 2: _info(100, ignore_eos=True)})
    assert cut.counts.tolist() == [3, 4]
    assert cut.stopped.tolist() == [True, False]
    # the next step: request 2 keeps counting from what it emitted
    seq = node.committed("decode", [2], step, kv)
    cut = node.cut(seq, _rows([2], [[1, 2, 3, 4]], [K]), {2: _info(6, ignore_eos=True)})
    assert cut.counts.tolist() == [1] and cut.stopped.tolist() == [True]
    # the take-back still reads accepted: every slot was kept on the device
    assert node.ledger.take_due(node.ledger.next_step + 1, frozenset({1, 2}))["kv"].counts == []


def test_a_node_with_no_take_back_resource_cuts_its_verify_steps_and_records_no_rows():
    node = _node(take_back_keys=())
    seq = node.committed("decode", [1, 2], _STEP, {})
    cut = node.cut(seq, _rows([1, 2], [[4, 5, 6, 7], [4, 5, 6, 7]], [1, 3]), {1: _info(100), 2: _info(100)})
    assert cut.counts.tolist() == [2, 4]
    assert node.ledger.take_due(node.ledger.next_step + 1, frozenset({1, 2})) == {}
    node.ledger.remove_request(1)


@pytest.mark.parametrize("field", ["post_forks", "retention", "sink"])
def test_a_verify_step_that_forks_retains_or_sinks_is_refused(field):
    from mstar.engine.resources.kv.config import KVStep
    from mstar.engine.resources.step import Segment, SubmoduleStep

    value = {"post_forks": (("main", "copy"),), "retention": {(1, "main"): object()}, "sink": frozenset({(1, "main")})}
    rows = [Segment(1, "main", K + 1)]
    step = SubmoduleStep(steps={"kv": KVStep(**{field: value[field]})}, segments=rows)
    node = _node()
    with pytest.raises(RuntimeError, match=field):
        node.refuse_step(step)
    node.refuse_step(SubmoduleStep(steps={"kv": KVStep()}, segments=rows))


def test_a_span_kv_dropped_is_taken_back_by_no_resource():
    """Positions follow the KV length: a row whose KV span commit dropped gives
    nothing back on any resource."""
    node = _node()
    node.take_back_keys = ("kv", "rope")
    resources = {
        "kv": SimpleNamespace(committed_spans=lambda rids, label, span: [0 if r == 2 else span for r in rids]),
        "rope": SimpleNamespace(committed_spans=lambda rids, label, span: [span] * len(rids)),
    }
    seq = node.committed("decode", [1, 2], _STEP, resources)
    node.cut(seq, _rows([1, 2], [[1, 2, 3, 4], [1, 2, 3, 4]], [0, 0]), {1: _info(100), 2: _info(100)})
    takes = node.ledger.take_due(node.ledger.next_step + 1, frozenset({1, 2}))
    assert {key: dict(zip(t.rids, t.counts, strict=True)) for key, t in takes.items()} == {
        "kv": {1: K}, "rope": {1: K},
    }


def test_resources_counting_different_spans_for_a_row_are_refused():
    """The ledger keeps one counted span a row for every resource it corrects, so two that counted a row
    differently (neither dropped it) can't both be right."""
    node = _node()
    node.take_back_keys = ("kv", "rope")
    resources = {
        "kv": SimpleNamespace(committed_spans=lambda rids, label, span: [span] * len(rids)),
        "rope": SimpleNamespace(committed_spans=lambda rids, label, span: [span - 1] * len(rids)),
    }
    with pytest.raises(RuntimeError, match="another resource"):
        node.committed("decode", [1], _STEP, resources)


def test_acceptance_reaches_the_worker_metric_families():
    from mstar.engine.engine import Engine
    from mstar.observability import catalog
    from mstar.observability.recorders import EdgeTransfers
    from mstar.worker.rebuild import Rebuilds
    from mstar.worker.worker_metrics import WorkerMetrics

    node = _node()
    eng = Engine.__new__(Engine)
    eng._resources, eng._drafts = {}, {"LLM": node}
    metrics = WorkerMetrics(
        "w0", 3600.0, frozenset({"LLM"}), eng, SimpleNamespace(pending_rows=dict), Rebuilds({}, {}),
        send=lambda status: None,
        enable_transfers=lambda: EdgeTransfers("SHM"),
    )
    node.cut(node.committed("decode", [1], _STEP, _KV), _rows([1], [[1, 2, 3, 4]], [3]), {1: _info(100)})
    metrics.start_counting()  # what ran before the counting starts is left out
    seq = node.committed("decode", [1, 2], _STEP, _KV)
    node.cut(seq, _rows([1, 2], [[1, 2, 3, 4], [5, 6, 7, 8]], [3, 1]), {1: _info(100), 2: _info(100)})
    snaps = {s.name: s for s in metrics.snapshot()}
    assert [s.name for s in metrics.snapshot()] == list(catalog.WORKER_FAMILIES)
    assert snaps["mstar_spec_draft_tokens_total"].values == [2 * K]
    assert snaps["mstar_spec_accepted_tokens_total"].values == [4]
    assert snaps["mstar_spec_verify_rows_total"].values == [2]
    hist = snaps["mstar_spec_accepted_per_row"]
    assert hist.sums == [4.0] and hist.bucket_counts[0][-1] == 2


def test_a_verify_step_that_moves_positions_its_own_way_is_refused():
    """The position take-back assumes the counter moved by the span."""
    from mstar.engine.resources.position.config import PositionStep
    from mstar.engine.resources.step import Segment, SubmoduleStep

    step = SubmoduleStep(steps={"rope": PositionStep(advance=(1,))}, segments=[Segment(1, "main", K + 1)])
    with pytest.raises(RuntimeError, match="advance"):
        _node().refuse_step(step)


def test_a_verify_rows_per_request_stop_check_never_passes_silently():
    """The fallback from check_stop_batched raises rather than never stopping."""
    from mstar.model.submodule_base import ARNodeSubmodule

    sub = SimpleNamespace(token_stop=_node().token_stop)
    sub.token_stop = sub.token_stop.__class__(**{**sub.token_stop.__dict__, "verify_walks": frozenset({"decode"})})
    info = SimpleNamespace(graph_walk="decode")
    with pytest.raises(RuntimeError, match="token cut"):
        ARNodeSubmodule.check_stop(sub, 1, info, {VERIFY_TOKENS: [torch.zeros(4)]})


class _Taking:
    """A resource that records its take-backs."""

    def __init__(self):
        self.taken = []

    def take_back(self, rids, counts, label):
        self.taken.append((tuple(rids), tuple(counts)))


def _runner_with(resource):
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.runner import StepRunner
    from mstar.engine.resources.step import ADMIT_OK

    class _Pool(Resource):
        supports_preplan = True

        @classmethod
        def build(cls, spec, info):
            raise NotImplementedError

        def admit(self, step, ctx):
            return ADMIT_OK

        def take_back(self, rids, counts, label):
            resource.take_back(rids, counts, label)

    return StepRunner({"kv": _Pool()}, node_resources={"LLM": ["kv"]})


def _step(rids):
    from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep

    step = SubmoduleStep(steps={"kv": SimpleNamespace(segments=tuple(Segment(r, "main", K + 1) for r in rids))})
    step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False))
    return step


def test_pre_admit_then_admit_applies_a_result_exactly_once():
    resource = _Taking()
    runner = _runner_with(resource)
    ledger = VerifyLedger("LLM", synchronous=False)
    s1 = ledger.committed([1], {"kv": "main"}, [K + 1])
    ledger.record(s1, (1,), np.array([0]))
    ledger.committed([1], {"kv": "main"}, [K + 1])
    step = _step([1])
    runner.pre_admit(step, ledger)
    runner.admit(step, ledger)
    assert resource.taken == [((1,), (K,))]


def test_a_result_due_with_a_pre_plan_staged_raises():
    resource = _Taking()
    runner = _runner_with(resource)
    ledger = VerifyLedger("LLM", synchronous=False)
    s1 = ledger.committed([1], {"kv": "main"}, [K + 1])
    ledger.record(s1, (1,), np.array([0]))
    ledger.committed([1], {"kv": "main"}, [K + 1])
    runner._staged = ("decode", (2,), 0, None, ())
    with pytest.raises(RuntimeError, match="pre-plan staged"):
        runner._take_back(ledger, _step([1]))
