"""A request cancelled after its batch was scheduled has no tensors any more;
the worker must leave it out of the executing batch instead of failing every
other request in the batch with a KeyError (seen at 64 duplex sessions when
clients closed together)."""
from types import SimpleNamespace

import pytest

from mstar.worker.worker import Worker


class _Tensors:
    def __init__(self, live: dict[str, dict[str, object]]):
        self.live = live

    def get_tensor(self, request_id: str, uuid: str):
        return self.live[request_id][uuid]


def _node(uuids: list[str], final: bool = False):
    edge = SimpleNamespace(tensor_info=[SimpleNamespace(uuid=u) for u in uuids], _final_stream_chunk=final)
    return SimpleNamespace(ready_signals=SimpleNamespace(ready_inputs={"x": edge}))


def _worker(live, fwd_ok):
    graphs = SimpleNamespace(
        get_partition_for_node=lambda node: "P",
        get_fwd_info=lambda rid, partition: {"rid": rid} if rid in fwd_ok else (_ for _ in ()).throw(KeyError(rid)),
    )
    return SimpleNamespace(
        tensor_manager=_Tensors(live), worker_graphs_manager=graphs, worker_id="w0",
        _make_executing_batch=lambda **kw: kw,
    )


def _batch(nodes):
    return SimpleNamespace(node_name="talker", graph_walk="decode", node_objects=nodes)


def test_dropped_request_is_left_out_and_the_others_run():
    live = {"a": {"u1": "ta"}, "c": {"u3": "tc"}}                      # "b" was cancelled: no tensors
    nodes = {"a": _node(["u1"]), "b": _node(["u2"]), "c": _node(["u3"], final=True)}
    out = Worker._build_executing_batch(_worker(live, {"a", "b", "c"}), _batch(nodes))
    assert out["request_ids"] == ["a", "c"]
    assert set(out["per_request_input_tensors"]) == {"a", "c"}
    assert out["per_request_input_tensors"]["c"] == {"x": ["tc"]}
    assert out["per_request_info"] == {"a": {"rid": "a"}, "c": {"rid": "c"}}
    assert out["final_stream_rids"] == {"c"}


def test_dropped_graph_state_is_treated_the_same():
    live = {"a": {"u1": "ta"}, "b": {"u2": "tb"}}
    out = Worker._build_executing_batch(_worker(live, {"a"}), _batch({"a": _node(["u1"]), "b": _node(["u2"])}))
    assert out["request_ids"] == ["a"]


def test_a_batch_of_only_dropped_requests_is_empty_not_an_error():
    out = Worker._build_executing_batch(_worker({}, set()), _batch({"b": _node(["u2"])}))
    assert out["request_ids"] == [] and out["per_request_input_tensors"] == {}


def test_other_errors_still_propagate():
    class Boom(_Tensors):
        def get_tensor(self, request_id, uuid):
            raise RuntimeError("transport down")

    w = _worker({}, {"a"})
    w.tensor_manager = Boom({})
    with pytest.raises(RuntimeError):
        Worker._build_executing_batch(w, _batch({"a": _node(["u1"])}))
