"""Engine.prepare_inputs: a walk whose rows all get the same inputs is
prepared with one call, the object shared by the rows; any other walk still
prepares row by row. A replay pads such a step with that object too."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig  # noqa: E402
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner  # noqa: E402
from mstar.engine.engine import Engine  # noqa: E402
from mstar.engine.resources.step import BucketKey, SlotLease  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402


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
        self.failed_requests = {}
        self.inputs = None
        self.uniform_inputs = None
        self.running_batched = None
        self.rows_only = False
        self.loopback_rids = frozenset()

    def register_prepare_batch(self, inputs, uniform=None):
        self.inputs = inputs
        self.uniform_inputs = uniform

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
    assert batch.uniform_inputs is row
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


def _runner(config, bs=4):
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    bucket = BucketKey(graph_walk="decode", bs=bs, num_tokens=bs)
    runner._buckets = {bucket: SimpleNamespace(config=config)}
    return runner, SlotLease(slot=0, bucket=bucket)


def _decode_config():
    return BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode",
        single_request_inputs=ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=1),
    )


def test_a_uniform_step_pads_with_its_shared_row():
    """The row any request of the walk gets is as good a padding row as a
    clone of the template, and costs nothing per row."""
    runner, lease = _runner(_decode_config())
    row = ARNodeInputs(input_seq_len=1)
    padded = runner.pad_inputs(lease, [row, row], uniform=row)
    assert len(padded) == 4 and all(x is row for x in padded)


def test_other_steps_pad_with_template_clones():
    runner, lease = _runner(_decode_config())
    row = ARNodeInputs(input_seq_len=1)
    other = ARNodeInputs(input_ids=torch.ones(1, dtype=torch.long), input_seq_len=1)
    for inputs, uniform in (
        ([row, row], None),
        # a row that is not the shared one: the step is not uniform
        ([row, other], row),
        # a shared row of another span would change what the padding declares
        ([ARNodeInputs(input_seq_len=2)] * 2, ARNodeInputs(input_seq_len=2)),
    ):
        if uniform is not None and inputs[0].input_seq_len == 2:
            inputs = [uniform] * 2
        padded = runner.pad_inputs(lease, inputs, uniform=uniform)
        assert padded[:2] == inputs
        assert all(p.input_ids is not None and p.input_seq_len == 1 for p in padded[2:])
        assert padded[2] is not padded[3]
