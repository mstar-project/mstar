"""A request whose ingest raises on one resource is left on none of them, and
the worker fails only that request instead of the batch in flight."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.testing_utils.worker import make_worker


class _Res:
    def __init__(self, name, log, fail=False):
        self.name, self.log, self.fail = name, log, fail

    def ingest_request(self, rid, overrides=None):
        self.log.append(("ingest", self.name, rid))
        if self.fail:
            raise RuntimeError("over capacity")

    def remove_request(self, rid):
        self.log.append(("remove", self.name, rid))


def _runner(resources):
    runner = StepRunner.__new__(StepRunner)
    runner._resources = {r.name: r for r in resources}
    runner._order = [r.name for r in resources]
    return runner


def test_a_raising_ingest_rolls_back_every_resource_it_reached():
    log = []
    runner = _runner([_Res("kv", log), _Res("sampler", log, fail=True), _Res("pos", log)])
    with pytest.raises(RuntimeError, match="over capacity"):
        runner.ingest_request(7)
    assert log == [
        ("ingest", "kv", 7), ("ingest", "sampler", 7),
        ("remove", "kv", 7), ("remove", "sampler", 7),
    ]


def _raise(err):
    raise err


@pytest.mark.parametrize("raises_in", ["runtime_add", "stop_limits", "engine_add", "input_read"])
def test_the_worker_fails_only_the_request_whose_ingest_raised(raises_in):
    """Registration that raises (before or after the request has a handle), an
    engine that refuses the request, or a read that raises as it starts (a CUDA
    OOM in the H2D copy, say): the request fails, alone."""
    registered, processed, reads, sent = [], [], [], []
    parked = SimpleNamespace(body=SimpleNamespace(request_id="wire"))
    handles = {}

    def add_request(**kw):
        if raises_in == "runtime_add":
            raise RuntimeError("registration failed")
        handles[kw["request_id"]] = 5
        return 5

    w = make_worker(
        communicator=SimpleNamespace(send=lambda to, msg: sent.append(msg)),
        graph_runtime=SimpleNamespace(
            add_request=add_request, get_rid_handle=handles.get, get_sharding_config=lambda rid: None,
            register_stop_limits=lambda rid, info: (
                _raise(RuntimeError("registration failed")) if raises_in == "stop_limits" else None
            ),
        ),
        engine_manager=SimpleNamespace(
            engine=SimpleNamespace(publisher_nodes={}, skip_publish=lambda rid, keys: None),
            add_request=lambda rid, cfg, info: (
                _raise(RuntimeError("over capacity")) if raises_in == "engine_add" else None
            ),
        ),
        request_state=SimpleNamespace(per_request_info={}, add_request=lambda rid, info: None),
        streams=SimpleNamespace(add_request=lambda rid, info: None),
        wakeup_event=SimpleNamespace(register_futures=lambda futures: None),
        tensor_manager=SimpleNamespace(
            get_tensor=None,
            register_request=lambda rid, sharding: registered.append(rid),
            start_read_tensors=lambda *a, **kw: (
                reads.append(a[0]) or _raise(RuntimeError("CUDA out of memory")) if raises_in == "input_read"
                else pytest.fail("no reads for a failed request")
            ),
        ),
    )
    assert w.lifecycle.park(parked)
    w.ingest._dispatch = processed.extend
    body = SimpleNamespace(
        request_id="wire", partition_worker_graph_ids=None, worker_graph_to_workers={}, loop_caps={},
        initial_inputs=[],
        request_info=SimpleNamespace(
            partition_name="p", graph_walk="g", resource_configs={}, rid_handle=None,
        ),
    )
    w.ingest._add_new_request(body)
    failed = w.lifecycle.take_failures()
    if raises_in == "runtime_add":
        assert not failed and [msg.body.errors.keys() for msg in sent] == [{"wire"}]
        error = sent[0].body.errors["wire"]
    else:
        assert not sent and list(failed) == [5]
        error = failed[5]
    assert registered == ([5] if raises_in in ("engine_add", "input_read") else [])
    assert reads == ([5] if raises_in == "input_read" else [])  # the read itself raised
    assert error.startswith("Error in worker: internal error (error id ")
    # messages parked before the NEW still go through (a parked REMOVE among them)
    assert processed == [parked] and "wire" not in w.lifecycle._parked
