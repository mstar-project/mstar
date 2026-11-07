"""A request whose ingest raises on one resource is left on none of them, and
the worker fails only that request instead of the batch in flight."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402


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


@pytest.mark.parametrize("raises_in", [
    "runtime_add", "worker_state", "stop_limits", "tensor_register", "engine_add", "input_read",
])
def test_the_worker_fails_only_the_request_whose_ingest_raised(raises_in):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w._draining_rids = set()
    w._reads_done_sent = set()
    w._unprocessed_messages = {"wire": ["parked"]}
    processed, failed, registered = [], [], []
    w._graph_runtime = SimpleNamespace(
        add_request=lambda **kw: (_raise(RuntimeError("registration failed")) if raises_in == "runtime_add" else 5),
        get_sharding_config=lambda rid: None,
        register_stop_limits=lambda rid, info: (
            _raise(RuntimeError("registration failed")) if raises_in == "stop_limits" else None
        ),
    )
    w.engine_manager = SimpleNamespace(
        engine=SimpleNamespace(publisher_nodes={}, skip_publish=lambda rid, keys: None),
        add_request=lambda rid, cfg, info: (
            _raise(RuntimeError("over capacity")) if raises_in == "engine_add" else None
        ),
    )
    w.request_state = SimpleNamespace(per_request_info={})
    def add_info(rid, info):
        if raises_in == "worker_state":
            raise RuntimeError("registration failed")
        w.request_state.per_request_info[rid] = info
    w.request_state.add_request = add_info
    w._rid = lambda rid: None if raises_in == "runtime_add" else 5
    sent = []
    w.communicator = SimpleNamespace(send=lambda destination, msg: sent.append(msg))
    w.scheduler = SimpleNamespace(
        note_arrival=lambda rid: None, read_failed_rids=set(),
        failed_rids=set(), fail_rids=lambda rids: None,
    )
    w.streams = SimpleNamespace(add_request=lambda rid, info: None)
    w.tensor_manager = SimpleNamespace(
        register_request=lambda rid, sharding: (
            _raise(RuntimeError("registration failed")) if raises_in == "tensor_register"
            else registered.append(rid)
        ),
        start_read_tensors=lambda *a, **kw: (
            _raise(RuntimeError("CUDA out of memory")) if raises_in == "input_read"
            else pytest.fail("no reads for a failed request")
        ),
    )
    w._fail_requests = failed.append
    w._process_message_list = processed.extend
    body = SimpleNamespace(
        request_id="wire", partition_worker_graph_ids=None, worker_graph_to_workers={}, loop_caps={},
        initial_inputs=[],
        request_info=SimpleNamespace(
            partition_name="p", graph_walk="g", resource_configs={}, rid_handle=None,
        ),
    )
    Worker._add_new_request(w, body)
    if raises_in in ("runtime_add", "worker_state"):
        assert not failed and len(sent) == 1
        assert list(sent[0].body.errors) == ["wire"]
        failure = sent[0].body.errors["wire"]
    else:
        assert len(failed) == 1 and list(failed[0]) == [5]
        failure = failed[0][5]
    assert registered == ([5] if raises_in in ("engine_add", "input_read") else [])
    assert failure.message.startswith("internal error (error id ")
    # messages parked before the NEW still go through (a parked REMOVE among them)
    assert processed == ["parked"] and "wire" not in w._unprocessed_messages
