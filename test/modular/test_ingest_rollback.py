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


def test_the_worker_fails_only_the_request_whose_ingest_raised():
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w._draining_rids = set()
    w._reads_done_sent = set()
    w._unprocessed_messages = {"wire": ["parked"]}
    w._last_active = {}
    processed, failed, registered = [], [], []
    w._graph_runtime = SimpleNamespace(
        add_request=lambda **kw: 5, get_sharding_config=lambda rid: None,
    )
    w.engine_manager = SimpleNamespace(
        evictable_nodes=lambda: [],
        add_request=lambda rid, cfg, info: (_ for _ in ()).throw(RuntimeError("over capacity")),
    )
    w.request_state = SimpleNamespace(add_request=lambda rid, info: None)
    w.tensor_manager = SimpleNamespace(
        register_request=lambda rid, sharding: registered.append(rid),
        start_read_tensors=lambda *a, **kw: pytest.fail("no reads for a failed request"),
    )
    w._fail_requests = failed.append
    w._process_message_list = processed.extend
    body = SimpleNamespace(
        request_id="wire", partition_worker_graph_ids=None, worker_graph_to_workers={}, loop_caps={},
        request_info=SimpleNamespace(
            partition_name="p", graph_walk="g", resource_configs=None, rid_handle=None,
        ),
    )
    Worker._add_new_request(w, body)
    assert registered == [5] and len(failed) == 1 and list(failed[0]) == [5]
    assert failed[0][5].startswith("Error in worker: internal error (error id ")
    # messages parked before the NEW still go through (a parked REMOVE among them)
    assert processed == ["parked"] and "wire" not in w._unprocessed_messages
