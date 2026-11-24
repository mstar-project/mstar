"""Episode views and row ownership through real conductor intake and teardown."""

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

import mstar.conductor.conductor as conductor_mod
import mstar.conductor.episodes as episodes_mod
from mstar.api_server import entrypoint, openpi
from mstar.api_server.positronic import PositronicPolicy
from mstar.api_server.request_types import ResultChunk
from mstar.conductor.conductor import Conductor
from mstar.conductor.episodes import EpisodeCapacityError
from mstar.conductor.request_info import EPISODE_CALLS, WalkStep
from mstar.engine.state import Array, Attend, ChunkCausal, Entry, KVEntry, Read, TensorType, Window
from mstar.engine.state.adapters import CHUNK_CAUSAL_LAYOUT_KWARG
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
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


class _AttentionEpisodeModel(_EpisodeModel):
    def declare_state(self):
        return [*super().declare_state(), Array(
            "within_step", nodes={"llm"}, layers={0}, entry=KVEntry(heads=2, head_dim=32),
            max_len=0, reads={"attn": Read(mask=ChunkCausal(8), op=Attend(q_heads=2))},
        )]

    def walk_plan(self, partition_name, input_modalities, output_modalities,
                  input_signals, model_kwargs, *, episode):
        self.views.append(episode)
        lengths = [4, 16, 8] if episode.held.get("frames", 0) else [4, 8, 0]
        return [WalkStep("run", step_metadata={
            CHUNK_CAUSAL_LAYOUT_KWARG: {"within_step": {"main": {"attn": lengths}}},
        })]


class _SeedOnlyModel(_EpisodeModel):
    def declare_state(self):
        return []


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
        "resources": {} if model_type is _SeedOnlyModel else {
            "frames": {"max_slots": pool_slots or capacity + 1},
        },
    }))
    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    model = model_type()
    conductor = Conductor(model, str(config), socket_path_prefix=str(tmp_path / "s"))
    return conductor, model, comm


def _request(rid, episode=None, *, reset=False, keys=None, input_open=False, deadline=None):
    return NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
        episode=episode, episode_keys=keys, reset_episode=reset, input_open=input_open,
        deadline=deadline,
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


def test_episode_held_lengths_reach_new_request_after_data_worker_defaults(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch, model_type=_AttentionEpisodeModel)
    cold = {"within_step": {"main": {"attn": [4, 8, 0]}}}
    first = _request("first", "robot-a")
    first.model_kwargs[CHUNK_CAUSAL_LAYOUT_KWARG] = cold
    conductor._ingest_request(first)
    assert _new_requests(comm, "first")[0][1].request_info.step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG] == cold
    _complete(conductor, "first", {"frames": 45})
    second = _request("second", "robot-a")
    second.model_kwargs[CHUNK_CAUSAL_LAYOUT_KWARG] = cold
    conductor._ingest_request(second)
    sent = _new_requests(comm, "second")
    assert len(sent) == 1
    assert model.views[-1].held["frames"] == 45
    assert sent[0][1].request_info.step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG] == {
        "within_step": {"main": {"attn": [4, 16, 8]}},
    }


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


def test_session_base_seed_reaches_worker_and_advances_across_reset(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch)
    first = _request("first", "robot-a")
    first.model_kwargs["episode_seed"] = 17
    conductor._ingest_request(first)
    assert _new_requests(comm, "first")[0][1].request_info.random_seed == 17
    _complete(conductor, "first", {"frames": 4})

    second = _request("second", "robot-a", reset=True)
    second.model_kwargs["episode_seed"] = 17
    conductor._ingest_request(second)
    assert _new_requests(comm, "second")[0][1].request_info.random_seed == 18
    assert conductor._episode_records.view("robot-a").calls == 2


def test_seeded_cold_session_uses_episode_record_without_arrays(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, model_type=_SeedOnlyModel)
    first = _request("first", "robot-a")
    first.model_kwargs["episode_seed"] = 17
    conductor._ingest_request(first)
    assert _new_requests(comm, "first")[0][1].request_info.random_seed == 17

    second = _request("second", "robot-a", reset=True)
    second.model_kwargs["episode_seed"] = 17
    conductor._ingest_request(second)
    assert _new_requests(comm, "second") == []
    _complete(conductor, "first")
    sent = _new_requests(comm, "second")
    assert len(sent) == 1 and sent[0][1].request_info.random_seed == 18
    assert sent[0][1].reset_episode is True
    _complete(conductor, "second")

    missing_seed = _request("missing-seed", "robot-a")
    conductor._ingest_request(missing_seed)
    assert _new_requests(comm, "missing-seed") == []

    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert not conductor._episode_records.has("robot-a")
    assert any(msg.message_type == WorkerMessageType.END_EPISODE for _, msg in comm.sent)


def test_cold_positronic_session_reaches_conductor_with_advanced_seed(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, model_type=_SeedOnlyModel)

    class IntakeAPI:
        positronic_policy = PositronicPolicy({"robot": "stub"}, observation_format="praxis1")

        def __init__(self):
            self.calls = 0

        def submit_request(self, *, episode, reset_episode, model_kwargs, **kwargs):
            self.calls += 1
            rid = f"call-{self.calls}"
            request = _request(rid, episode, reset=reset_episode)
            request.model_kwargs = model_kwargs
            conductor._ingest_request(request)
            return rid

        async def iter_result_chunks(self, request_id):
            yield ResultChunk(request_id, "action", openpi.pack([{}] * 21))
            _complete(conductor, request_id)

        def end_episode(self, episode_id):
            conductor._end_episode(EndEpisodeConductor(episode_id))

    monkeypatch.setattr(entrypoint, "api_server", IntakeAPI())
    with TestClient(entrypoint.app).websocket_connect("/api/v1/session?sample_seed=17") as ws:
        assert openpi.unpack(ws.receive_bytes())["status"] == "loading"
        ready = openpi.unpack(ws.receive_bytes())
        assert ready["meta"]["sample_seed_base"] == 17
        for call in range(2):
            ws.send_bytes(openpi.pack({
                "session_id": ready["session_id"], "observation": {"task": "go"},
            }))
            assert len(openpi.unpack(ws.receive_bytes())["result"]) == 21
            assert _new_requests(comm, f"call-{call + 1}")[0][1].request_info.random_seed == 17 + call
        ws.send_bytes(openpi.pack({"session_id": ready["session_id"], "end_session": True}))
        assert openpi.unpack(ws.receive_bytes())["end_session"] is True
    assert not conductor._episode_records.has(ready["session_id"])


def test_seed_only_session_keeps_call_index_after_idle_expiry(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, model_type=_SeedOnlyModel, capacity=1)

    def send(rid, episode, base):
        request = _request(rid, episode)
        request.model_kwargs["episode_seed"] = base
        conductor._ingest_request(request)

    def expire(episode):
        last_end = conductor._episode_records._records[episode].last_end
        assert conductor._expire_episode_slots(last_end + conductor._episode_idle_timeout_s + 1)
        conductor._try_admit_waiting()

    send("first", "robot-a", 17)
    assert _new_requests(comm, "first")[0][1].request_info.random_seed == 17
    _complete(conductor, "first")

    send("other", "robot-b", 99)
    assert _new_requests(comm, "other") == []
    expire("robot-a")
    assert _new_requests(comm, "other")[0][1].request_info.random_seed == 99
    _complete(conductor, "other")

    send("again", "robot-a", 17)
    assert _new_requests(comm, "again") == []
    expire("robot-b")
    assert _new_requests(comm, "again")[0][1].request_info.random_seed == 18
    _complete(conductor, "again")


def test_seed_only_call_limit_closes_later_queued_calls(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, model_type=_SeedOnlyModel, max_calls=1, max_queued_calls=2,
    )
    for rid in ("first", "second", "third"):
        request = _request(rid, "robot-a")
        request.model_kwargs["episode_seed"] = 17
        conductor._ingest_request(request)

    _complete(conductor, "first")
    failures = {
        msg.body.request_id: msg.body for _, msg in comm.sent
        if msg.message_type == "request_failed"
    }
    assert failures["second"].status == 400
    assert failures["third"].status == 410


def test_seeded_cold_session_serializes_multiple_queued_calls_without_arrays(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, model_type=_SeedOnlyModel, capacity=2, max_queued_calls=2,
    )
    for rid in ("c1", "c2", "c3"):
        request = _request(rid, "robot-a")
        request.model_kwargs["episode_seed"] = 17
        conductor._ingest_request(request)

    [(_, nr1)] = _new_requests(comm, "c1")
    assert (nr1.request_info.random_seed, nr1.request_info.step_metadata.get(EPISODE_CALLS)) == (17, 0)
    assert [body.request_id for body in conductor.waiting_queue] == ["c2", "c3"]
    assert _cold_starts(conductor) == {"new": 1.0}

    _complete(conductor, "c1")
    [(_, nr2)] = _new_requests(comm, "c2")
    assert (nr2.request_info.random_seed, nr2.request_info.step_metadata.get(EPISODE_CALLS)) == (18, 1)
    assert [body.request_id for body in conductor.waiting_queue] == ["c3"]
    assert _new_requests(comm, "c3") == []

    _complete(conductor, "c2")
    [(_, nr3)] = _new_requests(comm, "c3")
    assert (nr3.request_info.random_seed, nr3.request_info.step_metadata.get(EPISODE_CALLS)) == (19, 2)
    assert conductor.waiting_queue == []
    assert _cold_starts(conductor) == {"new": 1.0}
    _complete(conductor, "c3")


def test_seeded_cold_session_capacity_retry_and_sweep_expiry_without_arrays(tmp_path, monkeypatch):
    now = 100.0
    monkeypatch.setattr(conductor_mod.time, "perf_counter", lambda: now)
    monkeypatch.setattr(episodes_mod.time, "perf_counter", lambda: now)
    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, model_type=_SeedOnlyModel, capacity=1, idle_timeout_s=10,
    )

    def send(rid, episode, base):
        request = _request(rid, episode)
        request.model_kwargs["episode_seed"] = base
        conductor._ingest_request(request)

    send("a1", "robot-a", 17)
    _complete(conductor, "a1")

    # Full-slot EpisodeCapacityError queues b1 and must NOT infinite-loop in _try_admit_waiting().
    send("b1", "robot-b", 99)
    assert _new_requests(comm, "b1") == []
    assert [body.request_id for body in conductor.waiting_queue] == ["b1"]
    conductor._try_admit_waiting()
    assert [body.request_id for body in conductor.waiting_queue] == ["b1"]

    # Sweeping idle expiry works for arrayless seeded sessions and admits b1.
    now = 112.0
    conductor._sweep_expiry()
    [(_, nr_b1)] = _new_requests(comm, "b1")
    assert (nr_b1.request_info.random_seed, nr_b1.request_info.step_metadata.get(EPISODE_CALLS)) == (99, 0)
    assert conductor.waiting_queue == []
    assert conductor._episode_evictions.snapshot().values == [1.0]
    _complete(conductor, "b1")

    now = 125.0
    conductor._sweep_expiry()
    send("a2", "robot-a", 17)
    [(_, nr_a2)] = _new_requests(comm, "a2")
    assert (nr_a2.request_info.random_seed, nr_a2.request_info.step_metadata.get(EPISODE_CALLS)) == (18, 1)
    assert _cold_starts(conductor) == {"new": 2.0, "idle": 1.0}
    _complete(conductor, "a2")


def test_seeded_cold_session_rejects_closed_episode_waiter_without_arrays(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, model_type=_SeedOnlyModel, capacity=2,
    )
    conductor.max_concurrent_requests = 1
    blocker = _request("blocker", "robot-b")
    blocker.model_kwargs["episode_seed"] = 99
    conductor._ingest_request(blocker)
    c1 = _request("c1", "robot-a")
    c1.model_kwargs["episode_seed"] = 17
    conductor._ingest_request(c1)
    assert [body.request_id for body in conductor.waiting_queue] == ["c1"]

    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert conductor.waiting_queue == []
    failures = {
        msg.body.request_id: msg.body for _, msg in comm.sent
        if msg.message_type == "request_failed"
    }
    assert failures["c1"].status == 410
    _complete(conductor, "blocker")

    c2 = _request("c2", "robot-a")
    c2.model_kwargs["episode_seed"] = 17
    conductor._ingest_request(c2)
    failures = {
        msg.body.request_id: msg.body for _, msg in comm.sent
        if msg.message_type == "request_failed"
    }
    assert failures["c2"].status == 410


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


class _MultiStepEpisodeModel(_EpisodeModel):
    def get_graph_walk_graphs(self):
        return {
            "encode": GraphNode(
                name="llm", input_names=[], outputs=[
                    GraphEdge(next_node="llm", name="hid", persist=True),
                ],
            ),
            "decode": GraphNode(
                name="llm", input_names=["hid"], outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text", output_modality="text"),
                ],
            ),
        }

    def walk_plan(self, partition_name, input_modalities, output_modalities,
                  input_signals, model_kwargs, *, episode):
        self.views.append(episode)
        return [
            WalkStep("encode", step_metadata={"stage": "enc"}),
            WalkStep("decode", step_metadata={"stage": "dec"}),
        ]


class _NonEpisodeModel(StubModel):
    def get_graph_walk_graphs(self):
        return {"run": GraphNode(
            name="llm", input_names=[], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="text", output_modality="text"),
            ],
        )}

    def walk_plan(self, partition_name, input_modalities, output_modalities,
                  input_signals, model_kwargs):
        return [WalkStep("run", step_metadata={"stage": "plain"})]


def _input_signals(comm, rid):
    return [(worker, msg.body) for worker, msg in comm.sent
            if msg.message_type == WorkerMessageType.INPUT_SIGNALS and msg.body.request_id == rid]


def _cold_starts(conductor) -> dict[str, float]:
    snap = conductor._episode_cold_starts.snapshot()
    return {labels[0]: val for labels, val in zip(snap.label_values, snap.values, strict=True)}


def test_step_metadata_episode_calls_advances_across_calls_steps_and_resets(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, model_type=_MultiStepEpisodeModel, idle_timeout_s=10,
    )

    def run_two_steps(rid, *, reset=False):
        conductor._ingest_request(_request(rid, "robot-a", reset=reset))
        new_msg = _new_requests(comm, rid)[0][1]
        ptr = TensorPointerInfo(
            dims=[1], dtype=torch.uint8, nbytes=1, address=0, stride=(1,),
            uuid=f"u-{rid}", source_session_id="s", source_entity="worker_0", offset=0,
            source_tp_size=1, source_tp_rank=0,
        )
        done_parts = conductor._process_worker_graphs_done(WorkerGraphsDone(
            rid, [0], True, partition_name="default", persist_signals={"hid": [ptr]},
        ))
        for pname in done_parts:
            conductor._process_done_forward(rid, pname)
        sig_msg = _input_signals(comm, rid)[0][1]
        done_parts = conductor._process_worker_graphs_done(WorkerGraphsDone(
            rid, [0], True, partition_name="default", partition_done=True,
        ))
        for pname in done_parts:
            conductor._process_done_forward(rid, pname)
        _complete(conductor, rid, {"frames": 4})
        return new_msg.request_info.step_metadata, sig_msg.request_info.step_metadata

    assert run_two_steps("c1") == (
        {"stage": "enc", EPISODE_CALLS: 0}, {"stage": "dec", EPISODE_CALLS: 0},
    )
    assert run_two_steps("c2") == (
        {"stage": "enc", EPISODE_CALLS: 1}, {"stage": "dec", EPISODE_CALLS: 1},
    )
    assert run_two_steps("c3", reset=True) == (
        {"stage": "enc", EPISODE_CALLS: 2}, {"stage": "dec", EPISODE_CALLS: 2},
    )
    conductor._episode_records._records["robot-a"].last_end -= 11
    conductor._sweep_expiry()
    assert run_two_steps("c4") == (
        {"stage": "enc", EPISODE_CALLS: 3}, {"stage": "dec", EPISODE_CALLS: 3},
    )


def test_step_metadata_episode_calls_capacity_retry_idless_and_non_episode(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, capacity=1, idle_timeout_s=10)
    conductor._ingest_request(_request("c1", "robot-a"))
    _complete(conductor, "c1", {"frames": 4})
    conductor._ingest_request(_request("c2", "robot-a"))
    assert _new_requests(comm, "c2")[0][1].request_info.step_metadata == {EPISODE_CALLS: 1}
    _complete(conductor, "c2", {"frames": 4})

    # Idle-expire robot-a, fill the single slot with robot-b, then queue robot-a on EpisodeCapacityError.
    conductor._episode_records._records["robot-a"].last_end -= 11
    conductor._sweep_expiry()
    conductor._ingest_request(_request("blocker", "robot-b"))
    _complete(conductor, "blocker", {"frames": 2})

    conductor._ingest_request(_request("retry", "robot-a"))
    assert _new_requests(comm, "retry") == []
    assert [body.request_id for body in conductor.waiting_queue] == ["retry"]
    conductor._end_episode(EndEpisodeConductor("robot-b"))
    assert _new_requests(comm, "retry")[0][1].request_info.step_metadata == {EPISODE_CALLS: 2}
    _complete(conductor, "retry", {"frames": 4})
    conductor._end_episode(EndEpisodeConductor("robot-a"))

    # Idless calls on an episode model always receive 0.
    for rid in ("idless-1", "idless-2"):
        conductor._ingest_request(_request(rid))
        assert _new_requests(comm, rid)[0][1].request_info.step_metadata == {EPISODE_CALLS: 0}
        _complete(conductor, rid, {"frames": 1})

    # Non-episode models are untouched.
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    plain_cfg = plain_dir / "cfg.yaml"
    plain_cfg.write_text(yaml.safe_dump({
        "max_seq_len": 32,
        "node_groups": [{"node_names": ["llm"], "ranks": [0]}],
    }))
    plain_comm = _Comm()
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: plain_comm)
    plain_conductor = Conductor(_NonEpisodeModel(), str(plain_cfg), socket_path_prefix=str(plain_dir / "s"))
    plain_conductor._ingest_request(_request("plain-1"))
    assert plain_conductor._episode_cold_starts.snapshot().values == []
    assert _new_requests(plain_comm, "plain-1")[0][1].request_info.step_metadata == {"stage": "plain"}


def test_cold_starts_metric_counts_each_reason_and_ignores_warm_or_refused_calls(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, capacity=1, idle_timeout_s=10, max_calls=6,
    )
    conductor._ingest_request(_request("c1", "robot-a", keys={"frames": "k1"}))
    assert _cold_starts(conductor) == {"new": 1.0}
    _complete(conductor, "c1", {"frames": 4})

    # Warm continuation and per-array key change do NOT count as whole-episode cold starts.
    conductor._ingest_request(_request("c2", "robot-a", keys={"frames": "k1"}))
    _complete(conductor, "c2", {"frames": 4})
    conductor._ingest_request(_request("c3", "robot-a", keys={"frames": "k2"}))
    _complete(conductor, "c3", {"frames": 4})
    assert _cold_starts(conductor) == {"new": 1.0}

    # Explicit reset counts "reset".
    conductor._ingest_request(_request("c4", "robot-a", keys={"frames": "k2"}, reset=True))
    _complete(conductor, "c4", {"frames": 4})
    assert _cold_starts(conductor) == {"new": 1.0, "reset": 1.0}

    # Weight epoch change counts "identity".
    conductor._episode_weight_epoch = b"new-weights"
    conductor._ingest_request(_request("c5", "robot-a", keys={"frames": "k2"}))
    _complete(conductor, "c5", {"frames": 4})
    assert _cold_starts(conductor) == {"new": 1.0, "reset": 1.0, "identity": 1.0}

    # Idle expiry counts "idle" on next admitted call; capacity retry must NOT double-count.
    conductor._episode_records._records["robot-a"].last_end -= 11
    conductor._sweep_expiry()
    conductor._ingest_request(_request("temp"))
    assert _cold_starts(conductor) == {"new": 1.0, "reset": 1.0, "identity": 1.0, "temporary": 1.0}
    conductor._ingest_request(_request("c6", "robot-a", keys={"frames": "k2"}))
    assert _cold_starts(conductor) == {"new": 1.0, "reset": 1.0, "identity": 1.0, "temporary": 1.0}
    _complete(conductor, "temp", {"frames": 1})
    assert _cold_starts(conductor) == {
        "new": 1.0, "reset": 1.0, "identity": 1.0, "temporary": 1.0, "idle": 1.0,
    }
    _complete(conductor, "c6", {"frames": 4})

    # max_calls refusal does NOT increment.
    conductor._ingest_request(_request("c7", "robot-a"))
    assert _new_requests(comm, "c7") == []
    assert _cold_starts(conductor) == {
        "new": 1.0, "reset": 1.0, "identity": 1.0, "temporary": 1.0, "idle": 1.0,
    }


def test_abort_while_waiting_for_slot_preserves_order_and_episodes(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, capacity=1)
    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})

    conductor._ingest_request(_request("w1"))
    conductor._ingest_request(_request("w2", "robot-b"))
    conductor._ingest_request(_request("w3", "robot-c"))
    assert [r.request_id for r in conductor.waiting_queue] == ["w1", "w2", "w3"]

    conductor._abort_request("w2")
    assert [r.request_id for r in conductor.waiting_queue] == ["w1", "w3"]
    assert any(
        worker == conductor.PREPROCESS_WORKER
        and msg.message_type == WorkerMessageType.REMOVE_REQUEST
        and msg.body.request_id == "w2"
        for worker, msg in comm.sent
    )
    assert conductor._episode_records.view(
        "robot-a", identity=(conductor._episode_weight_epoch, None),
    ).held["frames"] == 4

    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert len(_new_requests(comm, "w1")) == 1
    _complete(conductor, "w1", {"frames": 1})
    assert len(_new_requests(comm, "w3")) == 1
    assert conductor.waiting_queue == []


def test_abort_while_waiting_behind_busy_episode_keeps_in_flight_episode_alive(tmp_path, monkeypatch):
    conductor, _, comm = _conductor(tmp_path, monkeypatch, max_queued_calls=1)
    conductor._ingest_request(_request("call-1", "robot-a"))
    conductor._ingest_request(_request("call-2", "robot-a"))
    assert [r.request_id for r in conductor.waiting_queue] == ["call-2"]

    conductor._abort_request("call-2")
    assert conductor.waiting_queue == []
    _complete(conductor, "call-1", {"frames": 6})
    assert conductor._episode_records.view(
        "robot-a", identity=(conductor._episode_weight_epoch, None),
    ).held["frames"] == 6

    conductor._ingest_request(_request("call-3", "robot-a"))
    assert len(_new_requests(comm, "call-3")) == 1


def test_waiting_queue_request_deadline_expires_without_admitting_and_unblocks_idle_expiry(
    tmp_path, monkeypatch,
):
    now = 100.0
    monkeypatch.setattr(conductor_mod.time, "perf_counter", lambda: now)
    import mstar.conductor.episodes as episodes_mod
    monkeypatch.setattr(episodes_mod.time, "perf_counter", lambda: now)

    conductor, _, comm = _conductor(
        tmp_path, monkeypatch, capacity=2, idle_timeout_s=10, max_queued_calls=2,
    )
    conductor.max_concurrent_requests = 1

    # Already-expired request arriving at _ingest_request is rejected immediately.
    conductor._ingest_request(_request("already-late", deadline=now))
    assert conductor.waiting_queue == []
    assert _new_requests(comm, "already-late") == []

    conductor._ingest_request(_request("first", "robot-a"))
    _complete(conductor, "first", {"frames": 4})

    # Hold the single concurrency slot with robot-b so robot-a-expired and waiter-expired queue.
    conductor._ingest_request(_request("blocker", "robot-b"))
    conductor._ingest_request(_request("waiter-expired", deadline=now + 5))
    conductor._ingest_request(_request("robot-a-expired", "robot-a", deadline=now + 5))
    conductor._ingest_request(_request("robot-c-survivor", "robot-c", deadline=now + 30))
    assert [r.request_id for r in conductor.waiting_queue] == [
        "waiter-expired", "robot-a-expired", "robot-c-survivor",
    ]

    # Advance past robot-a idle_timeout_s (110) and wait deadlines (105):
    # _sweep_expiry expires both waiters AND unblocks robot-a idle expiry in the same sweep!
    now = 112.0
    conductor._sweep_expiry()
    failures = {
        msg.body.request_id: msg.body
        for _, msg in comm.sent
        if msg.message_type == "request_failed"
    }
    assert failures["already-late"].status == 500
    assert failures["waiter-expired"].status == 500
    assert failures["robot-a-expired"].status == 500
    assert "robot-c-survivor" not in failures
    assert [r.request_id for r in conductor.waiting_queue] == ["robot-c-survivor"]
    assert conductor._episode_records.view(
        "robot-a", identity=(conductor._episode_weight_epoch, None),
    ).cold == "idle"

    _complete(conductor, "blocker", {"frames": 2})
    assert len(_new_requests(comm, "robot-c-survivor")) == 1
    _complete(conductor, "robot-c-survivor", {"frames": 2})
    conductor._end_episode(EndEpisodeConductor("robot-b"))
    conductor._end_episode(EndEpisodeConductor("robot-c"))

    # Also verify _try_admit_waiting skips an expired waiter when capacity opens.
    conductor._ingest_request(_request("holder", "robot-d"))
    conductor._ingest_request(_request("queued-to-expire", deadline=now + 1))
    conductor._ingest_request(_request("queued-fresh", deadline=now + 20))
    now += 2
    _complete(conductor, "holder", {"frames": 2})
    assert _new_requests(comm, "queued-to-expire") == []
    assert len(_new_requests(comm, "queued-fresh")) == 1


def test_worker_loss_cleans_up_slots_and_marks_next_call_cold_lost(tmp_path, monkeypatch):
    conductor, model, comm = _conductor(tmp_path, monkeypatch, replicas=2, capacity=1)
    # 1. Idle named episode on replica 0 (worker_0) and another on replica 1 (worker_1).
    conductor._ingest_request(_request("a1", "robot-a"))
    first_worker = _new_requests(comm, "a1")[0][0]
    _complete(conductor, "a1", {"frames": 4})

    conductor._ingest_request(_request("b1", "robot-b"))
    second_worker = _new_requests(comm, "b1")[0][0]
    assert first_worker != second_worker
    _complete(conductor, "b1", {"frames": 5})

    dead_worker = first_worker
    surviving_worker = second_worker
    conductor._handle_worker_loss({dead_worker})

    assert conductor._episode_slots.used_by_worker() == {surviving_worker: 1}
    identity = (conductor._episode_weight_epoch, None)
    assert conductor._episode_records.view("robot-a", identity=identity).cold == "lost"
    assert conductor._episode_records.view("robot-b", identity=identity).cold is None
    assert conductor._episode_records.view("robot-b", identity=identity).held["frames"] == 5

    # Next call for robot-a fails over to surviving_worker, sees cold="lost", and increments metric.
    conductor._end_episode(EndEpisodeConductor("robot-b"))
    conductor._ingest_request(_request("a2", "robot-a"))
    assert _new_requests(comm, "a2")[0][0] == surviving_worker
    assert (model.views[-1].calls, dict(model.views[-1].held), model.views[-1].cold) == (
        1, {"frames": 0}, "lost",
    )
    assert _cold_starts(conductor)["lost"] == 1.0
    _complete(conductor, "a2", {"frames": 8})
    assert conductor._episode_records.view("robot-a", identity=identity).cold is None

    # 2. In-flight named and temporary requests on a multi-rank/lost worker:
    # reset _dead_worker_ids to test in-flight loss on surviving_worker -> dead_worker failover.
    conductor._dead_worker_ids.clear()
    conductor._end_episode(EndEpisodeConductor("robot-a"))
    conductor._ingest_request(_request("c1", "robot-c"))
    c1_worker = _new_requests(comm, "c1")[0][0]
    other_worker = surviving_worker if c1_worker == dead_worker else dead_worker

    conductor._handle_worker_loss({c1_worker})
    failures = {
        msg.body.request_id: msg.body
        for _, msg in comm.sent
        if msg.message_type == "request_failed"
    }
    assert failures["c1"].status == 503
    assert not conductor._episode_records.closed_or_ending("robot-c")
    assert conductor._episode_slots.used_by_worker() == {}

    # Temporary idless in-flight request also fails 503 and releases its slot.
    conductor._ingest_request(_request("temp-live"))
    assert _new_requests(comm, "temp-live")[0][0] == other_worker
    conductor._handle_worker_loss({other_worker})
    assert "temp-live" not in conductor.requests
    assert conductor._episode_slots.used_by_worker() == {}


def test_startup_capacity_check_ignores_attention_read_key_matching_episode_array_name(
    tmp_path, monkeypatch,
):
    from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec

    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({
        "max_seq_len": 32,
        "node_groups": [{"node_names": ["llm"], "ranks": [0]}],
        "episodes": {"max_episodes": 1},
        "resources": {"frames": {"max_slots": 2}},
    }))
    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    model = _EpisodeModel()
    orig_node_resources = Conductor._node_resources
    attn_collision = AttentionSpec(
        resource_key="frames",
        nodes={"llm"},
        config=AttentionConfig(kv_cache="frames"),
    )
    monkeypatch.setattr(
        Conductor, "_node_resources",
        lambda self: [*orig_node_resources(self), attn_collision],
    )
    conductor = Conductor(model, str(config), socket_path_prefix=str(tmp_path / "s"))
    assert conductor._episode_slots.capacity == 1


def test_api_server_and_data_worker_propagate_request_deadline(monkeypatch):
    import queue
    import threading
    from types import SimpleNamespace

    from api_server_helpers import make_test_api_server

    import mstar.api_server.entrypoint as entrypoint_mod
    from mstar.api_server.data_worker import PreprocessWorkerThread

    now = 500.0
    monkeypatch.setattr(entrypoint_mod.time, "perf_counter", lambda: now)

    captured = []
    server = make_test_api_server(
        timeout_seconds=12.5,
        preprocess_worker=SimpleNamespace(new_request=captured.append),
    )
    server.submit_request(input_modalities=["text"], output_modalities=["text"], request_id="req-dl")
    assert len(captured) == 1
    assert captured[0].deadline == 512.5

    class _PreprocessModel(_NonEpisodeModel):
        def process_prompt(self, *args, **kwargs):
            return {}

    sent = []
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), model=_PreprocessModel(),
        model_config={"node_groups": [{"ranks": [0], "node_names": ["llm"]}]},
        communicator=SimpleNamespace(
            sent=sent, get_all_new_messages=lambda: [], send=lambda _, m: sent.append(m),
        ),
        tensor_manager=SimpleNamespace(
            store_and_return_tensor_info=lambda *a, **k: {},
            register_for_send=lambda *a, **k: None,
            set_persist=lambda *a, **k: None,
            force_cleanup_request=lambda *a, **k: None,
        ),
    )
    worker._process_input(captured[0])
    assert len(sent) == 1
    assert sent[0].body.deadline == 512.5
