"""The prompt tensors a model declares released by its first walk are released
at ingest, the same as every later walk's declaration. The preprocess worker
persists every input signal until the conductor unpersists it, so without this
pixel values, audio features and prompt tokens stay held for the whole request.

Drives the real ``Conductor.__init__`` and ``Conductor.run`` with worker launch
and ZMQ stubbed: the stub communicator records every send and replays one
scripted batch of inbound messages per run-loop iteration.
"""
from collections import deque

import pytest
import yaml

torch = pytest.importorskip("torch")

import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.request_info import (  # noqa: E402
    CurrentForwardConductorMetadata,
    PartitionDefinition,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.base import ForwardPassArgs, Model  # noqa: E402
from mstar.streaming.topology import PartitionTopology  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    SetupDone,
    WorkerGraphsDone,
)

PREPROCESS = "api_server_preprocess_worker"


class _StubComm:
    def __init__(self):
        self.sent = []
        self.batches = deque()

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return self.batches.popleft() if self.batches else []

    def poll_for_messages(self, timeout_ms=1):
        return False


class _ScriptDoneError(Exception):
    pass


class _ToyModel(Model):
    """Graph, partitions and walk transitions come from the test."""

    def __init__(self, walks, initial_args, next_args=None, partitions=None):
        self._walks = walks
        self._initial_args = initial_args
        self._next_args = next_args
        self._partitions = partitions

    def get_node_resources(self):
        return []

    def get_graph_walk_graphs(self):
        return self._walks

    def get_partitions(self):
        return self._partitions or super().get_partitions()

    def get_partition_topology(self):
        if not self._partitions:
            return super().get_partition_topology()
        return PartitionTopology(partitions=[p.name for p in self._partitions], connections=[])

    def get_initial_forward_pass_args(self, partition_name, input_modalities, output_modalities,
                                      input_signals, model_kwargs=None):
        return self._initial_args(partition_name, input_signals)

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals,
                                        incoming_connections=None):
        return self._next_args(partition_metadata, persist_signals)

    def process_prompt(self, *a, **k):
        return {}

    def postprocess(self, *a, **k):
        return b""

    def get_submodule(self, *a, **k):
        return None


def _conductor(model, node_groups, tmp_path, monkeypatch) -> Conductor:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_seq_len": 1024, "node_groups": node_groups}))
    comm = _StubComm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    c = Conductor(model, str(path), socket_path_prefix=str(tmp_path / "s"))
    c.comm = comm
    return c


def _run(c: Conductor, batches):
    """Run the real ``Conductor.run`` over ``batches``, one per iteration."""
    setup = [_msg(ConductorMessageType.SETUP_DONE, SetupDone(w)) for w in c.worker_ids]
    c.comm.batches.extend([setup, *batches])

    def poll():
        if not c.comm.batches:
            raise _ScriptDoneError

    c._poll_worker_liveness = poll
    with pytest.raises(_ScriptDoneError):
        c.run()


def _msg(kind, body):
    return ConductorMessage(message_type=kind, body=body)


def _new_request(rid, signals):
    return _msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id=rid, initial_signals=signals, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
    ))


def _info(uuid, entity=PREPROCESS, node=None, walk=None):
    return TensorPointerInfo(
        dims=[4], dtype=torch.int64, nbytes=32, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity=entity, _source_node_name=node, _source_graph_walk=walk,
    )


def _edge(node, name, infos):
    edge = GraphEdge(next_node=node, name=name)
    edge.tensor_info = list(infos)
    return edge


def _fwd(walk, inputs=(), unpersist=(), done=False):
    return ForwardPassArgs(
        full_metadata=CurrentForwardConductorMetadata(graph_walk=walk, is_prefill=False),
        inputs=list(inputs), unpersist_tensors=list(unpersist), request_done=done,
    )


def _sends(c, kinds=("NEW_REQUEST", "UNPERSIST_TENSORS")):
    """(kind, entity, body) for each send of one of ``kinds``, in send order."""
    out = []
    for entity, msg in c.comm.sent:
        kind = getattr(msg.message_type, "name", msg.message_type)
        if kind in kinds:
            out.append((kind, entity, msg.body))
    return out


def _releases(c):
    return [(entity, dict(body.uuid_to_ref_count))
            for kind, entity, body in _sends(c) if kind == "UNPERSIST_TENSORS"]


PROMPT, TOKEN = 101, 202


def test_first_walk_release_reaches_the_preprocess_worker_after_dispatch(tmp_path, monkeypatch):
    walks = {
        "prefill": GraphNode(name="llm", input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
        ]),
        "decode": Loop(name="decode_loop", max_iters=8, outputs=[], section=GraphNode(
            name="llm", input_names=["text_inputs"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                GraphEdge(next_node="llm", name="text_inputs"),
            ])),
    }

    def initial(partition, signals):
        edge = _edge("llm", "text_inputs", signals["text_inputs"])
        return _fwd("prefill", [edge], unpersist=edge.tensor_info)

    def next_args(metadata, persist_signals):
        if metadata.graph_walk == "prefill":
            edge = _edge("llm", "text_inputs", persist_signals.get("new_token", []))
            return _fwd("decode", [edge], unpersist=edge.tensor_info)
        return _fwd("decode", done=True)

    c = _conductor(_ToyModel(walks, initial, next_args),
                   [{"node_names": ["llm"], "ranks": [0]}], tmp_path, monkeypatch)
    prefill_wg = c.node_walk_to_wg[("llm", "prefill")].worker_graph_id
    token = _info(TOKEN, entity="worker_0", node="llm", walk="prefill")
    _run(c, [
        [_new_request("r", {"text_inputs": [_info(PROMPT)]})],
        [_msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id="r", worker_graph_ids=[prefill_wg], is_first_tp_rank=True,
            partition_name="default", persist_signals={"new_token": [token]},
        ))],
    ])

    assert [(kind, entity) for kind, entity, _ in _sends(c)] == [
        ("NEW_REQUEST", "worker_0"),
        ("UNPERSIST_TENSORS", PREPROCESS),  # at ingest, after the NewRequest
        ("UNPERSIST_TENSORS", "worker_0"),  # the prefill -> decode transition
    ]
    assert _releases(c) == [(PREPROCESS, {PROMPT: 1}), ("worker_0", {TOKEN: 1})]
    request = c.requests["r"]
    assert request.persist_signal_ref_cnt == {}
    assert all(not infos for infos in request.persist_signals.values())


def test_ingest_releases_the_union_of_every_partitions_declaration(tmp_path, monkeypatch):
    shared, only_a, done_partition, never_sent = 1, 2, 3, 4
    walks = {
        "a": GraphNode(name="na", input_names=["x", "y", "z"], outputs=[]),
        "b": GraphNode(name="nb", input_names=["x"], outputs=[]),
        "c": GraphNode(name="nc", input_names=["z"], outputs=[]),
    }
    partitions = [PartitionDefinition(name=p.upper(), graph_walks={p}, initial_walk=p) for p in "abc"]

    def initial(partition, signals):
        if partition == "A":
            inputs = [_edge("na", name, signals[name]) for name in ("x", "y", "z")]
            return _fwd("a", inputs, unpersist=signals["x"] + signals["w"])
        if partition == "B":
            # reads x too, and declares y, which only A sent
            return _fwd("b", [_edge("nb", "x", signals["x"])], unpersist=signals["y"])
        # inactive for this request, but declares z, which A sent
        return _fwd("c", unpersist=signals["z"], done=True)

    c = _conductor(
        _ToyModel(walks, initial, partitions=partitions),
        [{"node_names": ["na", "nc"], "ranks": [0]}, {"node_names": ["nb"], "ranks": [1]}],
        tmp_path, monkeypatch,
    )
    signals = {
        "x": [_info(shared)], "y": [_info(only_a)], "z": [_info(done_partition)], "w": [_info(never_sent)],
    }
    _run(c, [[_new_request("r", signals)]])

    sends = _sends(c)
    assert sorted((entity, body.request_info.partition_name)
                  for kind, entity, body in sends if kind == "NEW_REQUEST") == [
        ("worker_0", "A"), ("worker_0", "C"), ("worker_1", "B"),
    ]
    # one release, after every NewRequest, with x counted once per worker that reads it
    assert _releases(c) == [(PREPROCESS, {shared: 2, only_a: 1, done_partition: 1})]
    assert [kind for kind, _, _ in sends] == ["NEW_REQUEST"] * 3 + ["UNPERSIST_TENSORS"]
    # Never sent to a worker, so no reader count: it stays persisted (and visible to
    # later walks) until the request's teardown, like an untracked uuid on any pass.
    request = c.requests["r"]
    assert [i.uuid for i in request.persist_signals["w"]] == [never_sent]
    assert not any(request.persist_signals[name] for name in ("x", "y", "z"))
