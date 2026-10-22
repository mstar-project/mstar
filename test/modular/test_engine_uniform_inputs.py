"""Engine.prepare_inputs: a walk whose rows all get the same inputs is
prepared with one call, the object shared by the rows; any other walk still
prepares row by row."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.engine import Engine  # noqa: E402


class _Batch:
    def __init__(self, rids, walk="decode"):
        self.node_name = "llm"
        self.request_ids = list(rids)
        self.step_context = SimpleNamespace(
            graph_walk=walk, request_walks={}, walk_of=lambda _r: walk,
        )
        self.per_request_info_wrapped = {r: SimpleNamespace() for r in rids}
        self.per_request_input_tensors = {}
        self.per_request_input_metadata = {}
        self.final_stream_rids = set()
        self.skipped_rids = set()
        self.rebuild_rids = frozenset()
        self.failed_requests = {}
        self.inputs = None
        self.running_batched = None
        self.rows_only = False
        self.loopback_rids = frozenset()

    def register_prepare_batch(self, inputs):
        self.inputs = inputs

    def drop_rids(self, rids):
        self.request_ids = [r for r in self.request_ids if r not in rids]

    def register_failure(self, rid, error):
        self.failed_requests[rid] = str(error)


class _Sub:
    def __init__(self, uniform=None):
        self.uniform = uniform
        self.calls = []
        self.row_loopback = []
        self.uniform_loopback = None

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        self.uniform_loopback = has_loopback
        return self.uniform

    def prepare_inputs(self, **kw):
        self.calls.append(kw["graph_walk"])
        self.row_loopback.append(kw["has_loopback"])
        return object()

    def can_batch(self, batch, model_inputs):
        return len(model_inputs) > 0


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {"llm": SimpleNamespace(submodule=sub, resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    return eng


def test_a_uniform_walk_is_prepared_with_one_object():
    row = object()
    sub = _Sub(uniform=row)
    batch = _Batch([1, 2, 3])
    _engine(sub).prepare_inputs(batch)
    assert batch.inputs == [row, row, row] and all(x is row for x in batch.inputs)
    assert sub.calls == [] and batch.running_batched is True


def test_other_walks_still_prepare_row_by_row():
    sub = _Sub(uniform=None)
    batch = _Batch([1, 2])
    _engine(sub).prepare_inputs(batch)
    assert sub.calls == ["decode", "decode"] and len(batch.inputs) == 2


def test_loopback_is_per_row_and_tracks_each_request_s_last_step():
    sub = _Sub(uniform=None)
    eng = _engine(sub)
    ran = _Batch([1, 2])
    ran.rows_only = True
    eng._record_loopback(ran)
    eng.prepare_inputs(_Batch([1, 2, 3]))
    # 3 never ran rows-only here (e.g. its prefill ran on another rank)
    assert sub.uniform_loopback is False and sub.row_loopback == [True, True, False]
    eng.prepare_inputs(_Batch([1, 2]))
    assert sub.uniform_loopback is True
    # a step that is not rows-only clears the record; so does another walk's
    other = _Batch([2])
    other.rows_only = False
    eng._record_loopback(other)
    sub.row_loopback.clear()
    eng.prepare_inputs(_Batch([1, 2]))
    assert sub.row_loopback == [True, False]
    sub.row_loopback.clear()
    eng.prepare_inputs(_Batch([1], walk="prefill"))
    assert sub.row_loopback == [False]


def test_a_mixed_step_records_loopback_under_each_row_s_own_walk():
    sub = _Sub(uniform=None)
    eng = _engine(sub)
    walks = {1: "decode", 2: "decode", 3: "prefill"}
    mixed = _Batch([1, 2, 3], walk="mixed")
    mixed.step_context.request_walks = walks
    mixed.step_context.walk_of = walks.__getitem__
    # rows 1 and 2 kept their token on the device; 3's prompt row routed a tensor
    mixed.loopback_rids = frozenset({1, 2})
    eng._record_loopback(mixed)
    eng.prepare_inputs(_Batch([1, 2, 3]))
    assert sub.row_loopback == [True, True, False]
