"""Episode views and row ownership through real conductor intake and teardown."""

import torch
import yaml

import mstar.conductor.conductor as conductor_mod
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import WalkStep
from mstar.engine.state import Array, Entry, Read, TensorType, Window
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.testing_utils.model import StubModel
from mstar.utils.ipc_format import (
    EndEpisodeConductor,
    NewRequestConductor,
    ReadsDone,
    WorkerGraphsDone,
    WorkerMessageType,
)


class _EpisodeModel(StubModel):
    def __init__(self):
        self.views = []

    def declare_state(self):
        return [Array(
            "frames", nodes={"llm"}, layers={0},
            entry=Entry({"frame": TensorType((1,), dtype=torch.uint8)}),
            max_len=None, reads={"context": Read(mask=Window(4))}, lifetime="episode",
        )]

    def get_graph_walk_graphs(self):
        return {"run": GraphNode(
            name="llm", input_names=[], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="text", output_modality="text"),
            ],
        )}

    def walk_plan(self, partition_name, input_modalities, output_modalities,
                  input_signals, model_kwargs, *, episode):
        self.views.append(episode)
        return [WalkStep("run")]


class _Comm:
    def __init__(self):
        self.sent = []

    def send(self, worker, message):
        self.sent.append((worker, message))


def _conductor(tmp_path, monkeypatch, *, replicas=1, capacity=1):
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({
        "max_seq_len": 32,
        "node_groups": [{"node_names": ["llm"], "ranks": list(range(replicas))}],
        "episodes": {"max_episodes": capacity},
        "resources": {"frames": {"max_slots": capacity + 1}},
    }))
    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    model = _EpisodeModel()
    conductor = Conductor(model, str(config), socket_path_prefix=str(tmp_path / "s"))
    return conductor, model, comm


def _request(rid, episode=None, *, reset=False, keys=None):
    return NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
        episode=episode, episode_keys=keys, reset_episode=reset,
    )


def _new_requests(comm, rid):
    return [(worker, msg.body) for worker, msg in comm.sent
            if msg.message_type == WorkerMessageType.NEW_REQUEST and msg.body.request_id == rid]


def _complete(conductor, rid, held=None):
    if held is not None:
        conductor._record_episode_held(
            WorkerGraphsDone(rid, [0], True, episode_held=held), conductor.requests[rid],
        )
    conductor._process_request_done(rid)
    conductor._handle_reads_done(ReadsDone(rid, conductor.PREPROCESS_WORKER))


def test_named_calls_wait_for_teardown_then_plan_from_worker_counts(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a"))
    first = _new_requests(comm, "first")
    assert len(first) == 1
    assert (first[0][1].episode_id, first[0][1].episode_slot) == ("robot-a", 0)
    assert model.views[-1].calls == 0 and dict(model.views[-1].held) == {}

    conductor._ingest_request(_request("second", "robot-a"))
    assert _new_requests(comm, "second") == []
    _complete(conductor, "first", {"frames": 4})
    second = _new_requests(comm, "second")
    assert len(second) == 1 and second[0][0] == first[0][0]
    assert second[0][1].episode_slot == 0
    assert (model.views[-1].calls, model.views[-1].held["frames"]) == (1, 4)


def test_idless_calls_are_cold_and_share_the_same_admission_pool(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first"))
    assert _new_requests(comm, "first")[0][1].episode_slot == 0
    conductor._ingest_request(_request("second"))
    assert _new_requests(comm, "second") == []
    _complete(conductor, "first", {"frames": 4})
    assert _new_requests(comm, "second")[0][1].episode_slot == 0
    assert [(view.id, view.calls, dict(view.held)) for view in model.views] == [
        (None, 0, {}), (None, 0, {}),
    ]


def test_reset_and_close_clear_owner_after_teardown(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})
    conductor._ingest_request(_request("second", "robot-a", reset=True))
    assert model.views[-1].cold == "reset"
    assert dict(model.views[-1].held) == {}
    assert _new_requests(comm, "second")[0][1].reset_episode is True
    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert not any(msg.message_type == WorkerMessageType.END_EPISODE for _, msg in comm.sent)
    _complete(conductor, "second", {"frames": 2})
    assert any(msg.message_type == WorkerMessageType.END_EPISODE for _, msg in comm.sent)
    conductor._ingest_request(_request("third", "robot-a"))
    assert _new_requests(comm, "third") == []


def test_warm_episode_stays_on_first_replica_when_router_would_rotate(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch, replicas=2)
    conductor._ingest_request(_request("first", "robot-a"))
    first_worker = _new_requests(comm, "first")[0][0]
    _complete(conductor, "first", {"frames": 4})

    # An unrelated request exercises the router's next replica choice.
    conductor._ingest_request(_request("other"))
    other_worker = _new_requests(comm, "other")[0][0]
    assert other_worker != first_worker
    _complete(conductor, "other", {"frames": 2})

    conductor._ingest_request(_request("second", "robot-a"))
    assert _new_requests(comm, "second")[0][0] == first_worker
    assert model.views[-1].held["frames"] == 4


def test_changed_content_key_clears_just_its_array_before_planning(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a", keys={"frames": "a"}))
    _complete(conductor, "first", {"frames": 4})

    conductor._ingest_request(_request("second", "robot-a", keys={"frames": "b"}))
    assert model.views[-1].calls == 1
    assert dict(model.views[-1].held) == {}
    assert _new_requests(comm, "second")[0][1].episode_clear == ["frames"]
