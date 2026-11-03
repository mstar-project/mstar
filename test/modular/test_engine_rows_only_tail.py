"""A rows-only step runs only a submodule's batch tail, and only when it has
one; a step on the unbatched forward (per request or a single row) is not
rows-only."""
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.engine import Engine  # noqa: E402
from mstar.model.submodule_base import BatchedModelOutput, NodeSubmodule  # noqa: E402


class _Plain(NodeSubmodule):
    def prepare_inputs(self, *a, **kw):
        return None

    def forward(self, *a, **kw):
        return None


class _WithTail(_Plain):
    def batch_postprocess(self, outputs, request_infos, inputs, piecewise_runners=None):
        self.seen = (outputs, request_infos, inputs, piecewise_runners)


RUNNERS = {"region": object()}


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {"llm": SimpleNamespace(submodule=sub, piecewise_runners=RUNNERS)}
    eng._loopback_steps = {}
    eng.per_request_tails = []
    eng.postprocess_batch = lambda batch, outputs: eng.per_request_tails.append(batch)
    return eng


def _batch(rows_only):
    return SimpleNamespace(
        node_name="llm", admit_error=None, rows_only=rows_only, cached_rids=set(),
        per_request_info_wrapped={1: object()}, inputs=[object()],
    )


@pytest.mark.parametrize("sub_cls", [_Plain, _WithTail])
def test_rows_only_runs_the_batch_tail_only_when_overridden(sub_cls):
    sub = sub_cls.__new__(sub_cls)
    eng = _engine(sub)
    out = BatchedModelOutput()
    eng.exec = lambda batch: out
    batch = _batch(rows_only=True)
    assert eng.exec_and_postprocess(batch) is out
    assert eng.per_request_tails == []
    if sub_cls is _WithTail:
        # the node's captured regions come along, as the forward gets them
        assert sub.seen == (out, batch.per_request_info_wrapped, batch.inputs, RUNNERS)
    eng.exec_and_postprocess(_batch(rows_only=False))
    assert len(eng.per_request_tails) == 1


@pytest.mark.parametrize("rids", [[1, 2], [1]])
def test_an_unbatched_step_clears_rows_only(rids):
    eng = _engine(_Plain.__new__(_Plain))
    eng._enable_nvtx = False
    eng._enable_profile = False
    eng._device = SimpleNamespace(type="cpu")
    eng._autocast_dtype_for = lambda sub: None
    eng.reserve_replay_slot = lambda batch: None
    seen = {}

    def _per_request(batch):
        seen["rows_only"] = batch.rows_only
        return BatchedModelOutput()

    eng._exec_per_request = _per_request
    eng._exec_single = _per_request
    batch = SimpleNamespace(
        node_name="llm", request_ids=rids, rows_only=True, running_batched=False,
        step_context=SimpleNamespace(slot_lease=None, graph_walk="decode"),
        exec_timings=SimpleNamespace(start=None), admit_error=None,
        outputs_ready=threading.Event(), release_waiters=lambda: None,
        preplanned_rids=None, preplan_event=None,
    )
    eng.exec(batch)
    assert seen["rows_only"] is False and batch.rows_only is False
    assert eng._loopback_steps == {}
