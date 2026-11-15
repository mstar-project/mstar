"""Episode views and row ownership through real conductor intake and teardown."""

import pytest
import torch
import yaml

import mstar.conductor.conductor as conductor_mod
from mstar.conductor.conductor import Conductor
from mstar.conductor.episodes import EpisodeCapacityError
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


class _TwoLabelEpisodeModel(_EpisodeModel):
    def declare_state(self):
        return [Array(
            "frames", nodes={"llm"}, layers={0}, labels={"cond", "blank"},
            entry=Entry({"frame": TensorType((1,), dtype=torch.uint8)}),
            max_len=None, reads={"context": Read(mask=Window(4))}, lifetime="episode",
        )]


class _Comm:
    def __init__(self):
        self.sent = []

    def send(self, worker, message):
        self.sent.append((worker, message))


def _conductor(tmp_path, monkeypatch, *, replicas=1, capacity=1, pool_slots=None,
               model_type=_EpisodeModel, **episode_config):
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({
        "max_seq_len": 32,
        "node_groups": [{"node_names": ["llm"], "ranks": list(range(replicas))}],
        "episodes": {"max_episodes": capacity, **episode_config},
        "resources": {"frames": {"max_slots": pool_slots or capacity + 1}},
    }))
    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    model = model_type()
    conductor = Conductor(model, str(config), socket_path_prefix=str(tmp_path / "s"))
    return conductor, model, comm


def _request(rid, episode=None, *, reset=False, keys=None, input_open=False):
    return NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
        episode=episode, episode_keys=keys, reset_episode=reset, input_open=input_open,
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


def test_idle_expiry_admits_an_older_waiter_before_a_new_call(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, idle_timeout_s=10)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})

    conductor._ingest_request(_request("waiting"))
    assert _new_requests(comm, "waiting") == []
    conductor._episode_records._records["robot-a"].last_end -= 11

    conductor._ingest_request(_request("new", "robot-b"))
    assert len(_new_requests(comm, "waiting")) == 1
    assert _new_requests(comm, "new") == []
    assert [body.request_id for body in conductor.waiting_queue] == ["new"]


def test_close_admits_a_waiter_after_releasing_its_slot(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})

    conductor._ingest_request(_request("waiting", "robot-b"))
    assert _new_requests(comm, "waiting") == []
    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert len(_new_requests(comm, "waiting")) == 1
    assert conductor.waiting_queue == []


def test_a_free_replica_admits_the_oldest_waiter(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, replicas=2, capacity=1)
    conductor.max_concurrent_requests = 1

    conductor._ingest_request(_request("robot", "robot-a"))
    _complete(conductor, "robot", {"frames": 4})
    conductor._ingest_request(_request("runner"))
    for rid in ("older", "middle", "later"):
        conductor._ingest_request(_request(rid))
    assert [body.request_id for body in conductor.waiting_queue] == ["older", "middle", "later"]

    _complete(conductor, "runner", {"frames": 4})
    assert len(_new_requests(comm, "older")) == 1
    assert [body.request_id for body in conductor.waiting_queue] == ["middle", "later"]

    _complete(conductor, "older", {"frames": 4})
    assert len(_new_requests(comm, "middle")) == 1
    assert _new_requests(comm, "later") == []


def test_capacity_retry_keeps_older_waiters_ahead_of_later_calls(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, replicas=2, capacity=1)
    conductor.max_concurrent_requests = 1
    conductor._ingest_request(_request("robot", "robot-a"))
    _complete(conductor, "robot", {"frames": 4})
    conductor._ingest_request(_request("runner"))
    for rid in ("older", "middle", "later"):
        conductor._ingest_request(_request(rid))

    reserve = conductor._reserve_episode_slot
    failed = False

    def fail_first_reservation(owner, workers):
        nonlocal failed
        if not failed:
            failed = True
            raise EpisodeCapacityError
        return reserve(owner, workers)

    monkeypatch.setattr(conductor, "_reserve_episode_slot", fail_first_reservation)
    _complete(conductor, "runner", {"frames": 4})
    assert failed and len(_new_requests(comm, "middle")) == 1
    assert [body.request_id for body in conductor.waiting_queue] == ["older", "later"]

    _complete(conductor, "middle", {"frames": 4})
    assert len(_new_requests(comm, "older")) == 1
    assert _new_requests(comm, "later") == []


def test_open_input_is_refused_when_episode_slots_are_full(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("robot", "robot-a"))
    _complete(conductor, "robot", {"frames": 4})

    conductor._ingest_request(_request("live", input_open=True))
    assert "live" not in conductor.requests
    assert conductor.waiting_queue == []
    failures = [msg.body for _, msg in comm.sent
                if msg.message_type == "request_failed" and msg.body.request_id == "live"]
    assert len(failures) == 1 and failures[0].status == 503


def test_capacity_retry_keeps_its_original_queue_time(tmp_path, monkeypatch):
    conductor, _, _ = _conductor(tmp_path, monkeypatch)
    conductor._status_interval_s = 1.0
    conductor._ingest_request(_request("robot", "robot-a"))
    _complete(conductor, "robot", {"frames": 4})

    conductor._ingest_request(_request("waiting"))
    queued_at = conductor._queued_since["waiting"]
    conductor._try_admit_waiting()
    assert conductor._queued_since["waiting"] == queued_at

    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert conductor.requests["waiting"].conductor_arrival_time == queued_at


def test_named_calls_wait_for_teardown_then_plan_from_worker_counts(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a"))
    status = {metric.name: metric for metric in conductor._status().metrics}
    assert status["mstar_episodes"].label_values == [["running"], ["idle"], ["cold"]]
    assert status["mstar_episodes"].values == [1.0, 0.0, 0.0]
    assert status["mstar_episode_slots_used"].values == [1.0]
    first = _new_requests(comm, "first")
    assert len(first) == 1
    assert (first[0][1].episode_id, first[0][1].episode_slot) == ("robot-a", 0)
    assert model.views[-1].calls == 0 and dict(model.views[-1].held) == {"frames": 0}

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
    assert "second" not in conductor.requests
    _complete(conductor, "first", {"frames": 4})
    assert _new_requests(comm, "second")[0][1].episode_slot == 0
    assert [(view.id, view.calls, dict(view.held)) for view in model.views] == [
        (None, 0, {"frames": 0}), (None, 0, {"frames": 0}),
    ]


def test_reset_and_close_clear_owner_after_teardown(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})
    conductor._ingest_request(_request("second", "robot-a", reset=True))
    assert model.views[-1].cold == "reset"
    assert dict(model.views[-1].held) == {"frames": 0}
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
    assert dict(model.views[-1].held) == {"frames": 0}
    assert _new_requests(comm, "second")[0][1].episode_clear == ["frames"]


def test_weight_epoch_change_resets_worker_rows_before_a_warm_plan(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    conductor._ingest_request(_request("first", "robot-a", keys={"frames": "prompt"}))
    _complete(conductor, "first", {"frames": 4})
    conductor._episode_weight_epoch = b"new-weight-epoch"

    conductor._ingest_request(_request("second", "robot-a", keys={"frames": "prompt"}))
    assert (model.views[-1].calls, model.views[-1].cold, model.views[-1].held["frames"]) == (
        1, "identity", 0,
    )
    assert _new_requests(comm, "second")[0][1].reset_episode is True
    assert conductor._episode_records._records["robot-a"].held == {}


def test_same_adapter_name_with_new_digest_resets_unkeyed_rows(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch)
    first = _request("first", "robot-a")
    first.model_kwargs = {"lora_adapter": "same", "lora_digest": b"a" * 32}
    first.episode_adapter_digest = b"a" * 32
    conductor._ingest_request(first)
    _complete(conductor, "first", {"frames": 4})

    second = _request("second", "robot-a")
    second.model_kwargs = {"lora_adapter": "same", "lora_digest": b"b" * 32}
    second.episode_adapter_digest = b"b" * 32
    conductor._ingest_request(second)
    assert (model.views[-1].calls, model.views[-1].held["frames"], model.views[-1].cold) == (
        1, 0, "identity",
    )
    assert _new_requests(comm, "second")[0][1].reset_episode is True
    _complete(conductor, "second", {"frames": 7})

    third = _request("third", "robot-a")
    third.model_kwargs = {"lora_adapter": "same", "lora_digest": b"b" * 32}
    third.episode_adapter_digest = b"b" * 32
    conductor._ingest_request(third)
    assert (model.views[-1].calls, model.views[-1].held["frames"], model.views[-1].cold) == (
        2, 7, None,
    )
    assert _new_requests(comm, "third")[0][1].reset_episode is False


def test_episode_adapter_without_selected_digest_is_refused(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch)
    request = _request("first", "robot-a")
    request.model_kwargs = {"lora_adapter": "same", "lora_digest": b"a" * 32}
    conductor._ingest_request(request)
    failures = [msg.body for _, msg in comm.sent if msg.message_type == "request_failed"]
    assert [(failure.status, failure.request_id) for failure in failures] == [(400, "first")]
    assert not conductor._episode_records.has("robot-a")


def test_startup_refuses_more_episode_slots_than_the_dense_pool_has(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="has 1 usable slots; episodes.max_episodes needs 2"):
        _conductor(tmp_path, monkeypatch, capacity=2, pool_slots=2)


def test_startup_counts_each_label_of_an_episode_array(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="8 slots"):
        _conductor(tmp_path, monkeypatch, capacity=4, pool_slots=5,
                   model_type=_TwoLabelEpisodeModel)
    conductor, _, _ = _conductor(tmp_path, monkeypatch, capacity=4, pool_slots=9,
                                model_type=_TwoLabelEpisodeModel)
    assert conductor._episode_slots.capacity == 4


def test_call_budget_refuses_after_the_first_admitted_call(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, max_calls=1)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})
    conductor._ingest_request(_request("second", "robot-a"))
    assert _new_requests(comm, "second") == []
    failures = [msg.body for _, msg in comm.sent if msg.message_type == "request_failed"]
    assert any("max_calls=1" in failure.error_message for failure in failures)
    assert any(msg.message_type == WorkerMessageType.END_EPISODE for _, msg in comm.sent)


def test_call_budget_closes_later_calls_in_the_waiting_batch(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, max_calls=1, max_queued_calls=2)
    conductor._ingest_request(_request("first", "robot-a"))
    conductor._status_interval_s = 1.0
    conductor._ingest_request(_request("second", "robot-a"))
    conductor._ingest_request(_request("third", "robot-a"))
    assert set(conductor._queued_since) == {"second", "third"}

    _complete(conductor, "first", {"frames": 4})
    failures = {
        msg.body.request_id: msg.body for _, msg in comm.sent
        if msg.message_type == "request_failed"
    }
    assert failures["second"].status == 400
    assert failures["third"].status == 410
    assert conductor.waiting_queue == []
    assert conductor._queued_since == {}


def test_only_one_overlapping_call_is_queued_for_an_episode(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, max_queued_calls=1)
    conductor._ingest_request(_request("first", "robot-a"))
    conductor._ingest_request(_request("second", "robot-a"))
    conductor._ingest_request(_request("third", "robot-a"))
    assert [request.request_id for request in conductor.waiting_queue] == ["second"]
    assert _new_requests(comm, "third") == []
    failures = [msg.body for _, msg in comm.sent if msg.message_type == "request_failed"]
    assert any(failure.request_id == "third" and failure.status == 429 for failure in failures)


def test_slot_pressure_after_planning_drops_registered_request(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch, replicas=2)
    conductor._ingest_request(_request("busy"))
    # Simulate the slot becoming unavailable after the replica pick.
    def refuse_slot(*_):
        raise EpisodeCapacityError

    monkeypatch.setattr(conductor, "_reserve_episode_slot", refuse_slot)
    conductor._ingest_request(_request("waiting"))
    assert len(model.views) == 2
    assert [request.request_id for request in conductor.waiting_queue] == ["waiting"]
    assert "waiting" not in conductor.requests
    assert _new_requests(comm, "waiting") == []


def test_full_episode_pool_queues_until_idle_expiry_then_retries(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, idle_timeout_s=1)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})
    conductor._ingest_request(_request("second", "robot-b"))
    assert [request.request_id for request in conductor.waiting_queue] == ["second"]
    assert _new_requests(comm, "second") == []
    conductor._episode_records._records["robot-a"].last_end -= 2
    conductor._sweep_expiry()
    assert [request.request_id for request in conductor.waiting_queue] == []
    assert _new_requests(comm, "second")[0][1].episode_id == "robot-b"
