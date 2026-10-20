"""The conductor's metrics: its periodic status message and the admission wait
it stamps on ``RequestComplete``.

Drives the real ``Conductor.__init__`` and ``Conductor.run`` with worker launch
and ZMQ stubbed (the harness in ``test_ingest_unpersist``): every scripted
batch is one run-loop iteration, and what the conductor sends the API server
is read back off the stub socket.
"""
import msgpack
import pytest
import yaml

torch = pytest.importorskip("torch")

from test_ingest_unpersist import (  # noqa: E402  (sibling helper, by module name)
    PREPROCESS,
    _edge,
    _fwd,
    _info,
    _msg,
    _new_request,
    _run,
    _StubComm,
    _ToyModel,
)

import mstar.communication.wire_types  # noqa: E402,F401  (registers the wire tags)
import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.api_server.request_types import (  # noqa: E402
    APIServerMessage,
    ConductorStatus,
    RequestComplete,
    RequestFailed,
)
from mstar.communication import wire  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.graph.base import GraphEdge, GraphNode, Sequential  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.observability import catalog  # noqa: E402
from mstar.observability.recorders import MetricSnapshot  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    AbortRequest,
    ConductorMessageType,
    ReadsDone,
    WorkerGraphsDone,
)

NODE_GROUPS = [{"node_names": ["enc"], "ranks": [0]}, {"node_names": ["llm"], "ranks": [1]}]


def _walks():
    """One walk over two node groups: enc on rank 0 feeds llm on rank 1."""
    return {"prefill": Sequential([
        GraphNode(name="enc", input_names=["x"], outputs=[GraphEdge(next_node="llm", name="h")]),
        GraphNode(name="llm", input_names=["h"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="tok", output_modality="text"),
        ]),
    ])}


def _conductor(tmp_path, monkeypatch, status_interval_s, max_concurrent_requests=1):
    def initial(partition, signals):
        return _fwd("prefill", [_edge("enc", "x", signals["x"])])

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": 64, "node_groups": NODE_GROUPS,
        "max_concurrent_requests": max_concurrent_requests,
    }))
    comm = _StubComm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    c = Conductor(_ToyModel(_walks(), initial, lambda metadata, persist: _fwd("prefill", done=True)),
                  str(path), socket_path_prefix=str(tmp_path / "s"), status_interval_s=status_interval_s)
    c.comm = comm
    return c


def _request(rid):
    return _new_request(rid, {"x": [_info(int.from_bytes(rid.encode(), "little"))]})


def _done(c, rid, node):
    wg = c.node_walk_to_wg[(node, "prefill")].worker_graph_id
    return _msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
        request_id=rid, worker_graph_ids=[wg], is_first_tp_rank=True))


def _reads_done(rid):
    return _msg(ConductorMessageType.READS_DONE, ReadsDone(request_id=rid, entity_id=PREPROCESS))


def _to_api_server(c, message_type):
    return [m.body for e, m in c.comm.sent
            if e == "api_server" and getattr(m, "message_type", None) == message_type]


def _gauges(status: ConductorStatus) -> dict:
    """{family: {label values: value}} of one status message."""
    return {s.name: {tuple(row): v for row, v in zip(s.label_values, s.values, strict=True)}
            for s in status.metrics}


def _view(status):
    g = _gauges(status)
    states = {k[0]: v for k, v in g["mstar_conductor_requests"].items()}
    in_flight = {k[0]: v for k, v in g["mstar_component_in_flight"].items()}
    return states, in_flight


def test_each_status_reports_the_request_states_and_the_components_still_running(tmp_path, monkeypatch):
    # an interval of 0 sends one status per run-loop iteration, after its batch
    c = _conductor(tmp_path, monkeypatch, status_interval_s=0.0)
    _run(c, [
        [_request("r1"), _request("r2"), _request("r3")],  # cap 1: r2 and r3 queue
        [_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("r3"))],
        [_done(c, "r1", "enc")],
        [_done(c, "r1", "llm")],  # r1 completes and drains; r2 still waits for its slot
        [_reads_done("r1")],  # r1 freed, r2 admitted
    ])

    views = [_view(s) for s in _to_api_server(c, "conductor_status")]
    assert views == [
        ({"waiting": 2, "running": 1, "draining": 0}, {"enc": 1, "llm": 1}),
        ({"waiting": 1, "running": 1, "draining": 0}, {"enc": 1, "llm": 1}),
        ({"waiting": 1, "running": 1, "draining": 0}, {"enc": 0, "llm": 1}),
        ({"waiting": 1, "running": 0, "draining": 1}, {"enc": 0, "llm": 0}),
        ({"waiting": 0, "running": 1, "draining": 0}, {"enc": 1, "llm": 1}),
    ]
    assert c._queued_since == {}, "an aborted or admitted request left its queue stamp"


def test_admission_wait_is_zero_on_arrival_and_the_time_queued_otherwise(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, status_interval_s=0.0)
    _run(c, [
        [_request("r1"), _request("r2")],
        [_done(c, "r1", "enc"), _done(c, "r1", "llm")],
        [_reads_done("r1")],
        [_done(c, "r2", "enc"), _done(c, "r2", "llm")],
    ])

    done = {b.request_id: b for b in _to_api_server(c, "request_complete")}
    assert sorted(done) == ["r1", "r2"]
    assert done["r1"].conductor_arrival_time == done["r1"].conductor_ingest_time
    assert done["r2"].conductor_arrival_time < done["r2"].conductor_ingest_time
    # r2 was taken in with r1, and admitted only once r1's slot was freed
    assert done["r2"].conductor_arrival_time < done["r1"].conductor_finish_time < done["r2"].conductor_ingest_time


def test_with_metrics_off_nothing_is_sent_or_stamped(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, status_interval_s=None)
    _run(c, [
        [_request("r1"), _request("r2")],
        [_done(c, "r1", "enc"), _done(c, "r1", "llm")],
        [_reads_done("r1")],
        [_done(c, "r2", "enc"), _done(c, "r2", "llm")],
    ])

    assert _to_api_server(c, "conductor_status") == []
    done = _to_api_server(c, "request_complete")
    assert len(done) == 2 and all(b.conductor_arrival_time is None for b in done)
    assert c._queued_since == {}


def test_one_status_per_interval(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, status_interval_s=3600.0)
    _run(c, [[_request("r1")], [], [_done(c, "r1", "enc")], []])
    statuses = _to_api_server(c, "conductor_status")
    assert len(statuses) == 1
    assert _view(statuses[0])[0] == {"waiting": 0, "running": 1, "draining": 0}


def test_the_status_message_crosses_the_wire_typed():
    """Once per interval, but still on the typed path: no dynamic fallback,
    and it decodes as itself rather than as an earlier union branch."""
    status = ConductorStatus(metrics=[
        MetricSnapshot("mstar_conductor_requests", [["waiting"], ["running"]], [2.0, 1.0]),
        MetricSnapshot("h", [["a"]], bucket_counts=[[0, 1, 1]], sums=[0.5]),
    ])
    frame = wire.encode(APIServerMessage("conductor_status", status))
    tag, payload = msgpack.unpackb(frame, raw=False, strict_map_key=False)
    assert tag == "api_msg"
    # the dynamic path wraps a dataclass in an extension; the typed one is a plain map
    assert payload["body"] == {"metrics": [
        {"name": "mstar_conductor_requests", "label_values": [["waiting"], ["running"]],
         "values": [2.0, 1.0], "bucket_counts": [], "sums": []},
        {"name": "h", "label_values": [["a"]], "values": [], "bucket_counts": [[0, 1, 1]], "sums": [0.5]},
    ]}
    assert wire.decode(frame) == APIServerMessage("conductor_status", status)

    # the bodies before it in the union still decode as themselves
    for body in (
        RequestFailed(request_id="r", error_message="boom", status=503),
        RequestComplete(request_id="r", output_chunk_counts={"text": 1}, conductor_ingest_time=2.0,
                        conductor_finish_time=3.0, conductor_arrival_time=1.0),
        RequestComplete(request_id="r", output_chunk_counts={}, conductor_ingest_time=2.0,
                        conductor_finish_time=3.0),
    ):
        assert wire.decode(wire.encode(APIServerMessage("request_complete", body))).body == body


def test_the_status_families_are_the_ones_the_conductor_sends(tmp_path, monkeypatch):
    c = _conductor(tmp_path, monkeypatch, status_interval_s=0.0)
    _run(c, [[]])
    (status,) = _to_api_server(c, "conductor_status")
    assert [s.name for s in status.metrics] == list(catalog.STATUS_FAMILIES)
    assert sorted(_view(status)[0]) == sorted(catalog.CONDUCTOR_REQUEST_STATES)
