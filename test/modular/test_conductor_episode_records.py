"""Episode placement and planning views use one conductor-owned record."""

from types import SimpleNamespace

import pytest

from mstar.conductor.conductor import Conductor
from mstar.conductor.episodes import EpisodeCapacityError, EpisodeRecords, EpisodeSlots
from mstar.conductor.routing import Placement
from mstar.observability import catalog
from mstar.observability.recorders import Counter as MetricCounter
from mstar.utils.ipc_format import EndEpisodeConductor, NewRequestConductor, WorkerGraphsDone


def _placement(replica: int) -> Placement:
    return Placement({0: replica}, {})


def test_a_warm_call_keeps_its_replica_and_sees_held_rows():
    episodes = EpisodeRecords()
    picks = []

    def pick():
        picks.append(len(picks))
        return _placement(len(picks) - 1)

    first = episodes.placement_for("robot-a", pick)
    assert first.replicas == {0: 0}
    cold = episodes.admit("robot-a", "call-1", 3, first)
    assert (cold.id, cold.calls, dict(cold.held), cold.cold) == ("robot-a", 0, {}, None)
    counts = {"frames": 21}
    episodes.finish("robot-a", "call-1", counts)
    counts["frames"] = 999  # a caller cannot mutate the saved view

    warm = episodes.view("robot-a")
    assert (warm.calls, dict(warm.held)) == (1, {"frames": 21})
    again = episodes.placement_for("robot-a", pick)
    assert again.replicas == {0: 0} and not again.facts
    assert picks == [0]  # the second call never asks the router to pick
    assert dict(episodes.admit("robot-a", "call-2", 3, again).held) == {"frames": 21}
    episodes.finish("robot-a", "call-2", {"frames": 45})
    assert episodes.view("robot-a").calls == 2


def test_idless_calls_stay_cold_and_never_get_an_episode_record():
    episodes = EpisodeRecords()
    first = episodes.admit(None, "call-1", 3, episodes.placement_for(None, lambda: _placement(0)))
    second = episodes.admit(None, "call-2", 4, episodes.placement_for(None, lambda: _placement(1)))
    assert first == second == episodes.view(None)
    assert episodes._records == {}


def test_reset_keeps_call_count_and_end_releases_the_logical_owner():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 3, _placement(0))
    episodes.finish("robot-a", "call-1", {"frames": 21})
    episodes.reset("robot-a")
    view = episodes.view("robot-a")
    assert (view.calls, dict(view.held), view.cold) == (1, {}, "reset")
    assert episodes.end("robot-a") == (3, set())
    assert episodes.view("robot-a").calls == 0


def test_live_and_wrong_owner_calls_are_refused_before_record_mutation():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 3, _placement(0))
    with pytest.raises(ValueError, match="already runs"):
        episodes.admit("robot-a", "call-2", 3, _placement(0))
    with pytest.raises(ValueError, match="in flight"):
        episodes.end("robot-a")
    episodes.finish("robot-a", "call-1", {"frames": 21})
    with pytest.raises(ValueError, match="must keep its slot and replica"):
        episodes.admit("robot-a", "call-2", 4, _placement(0))
    with pytest.raises(ValueError, match="must keep its slot and replica"):
        episodes.admit("robot-a", "call-2", 3, _placement(1))
    assert (episodes.view("robot-a").calls, dict(episodes.view("robot-a").held)) == (1, {"frames": 21})


def test_a_bad_worker_count_cannot_become_a_warm_planning_view():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 3, _placement(0))
    with pytest.raises(ValueError, match="held counts"):
        episodes.finish("robot-a", "call-1", {"frames": -1})
    assert episodes.view("robot-a").calls == 1


def test_changed_key_only_invalidates_its_own_array():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 3, _placement(0), keys={"text": "a"})
    episodes.finish("robot-a", "call-1", {"text": 8, "frames": 21})
    assert episodes.clear_arrays("robot-a", {"text": "b"}) == ["text"]
    assert dict(episodes.view("robot-a", {"text": "b"}).held) == {"frames": 21}
    episodes.admit("robot-a", "call-2", 3, _placement(0), keys={"text": "b"})
    assert dict(episodes.view("robot-a").held) == {"frames": 21}


def test_max_calls_refuses_before_changing_record():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 3, _placement(0), max_calls=1)
    episodes.finish("robot-a", "call-1", {"frames": 21})
    with pytest.raises(ValueError, match="episode reached max_calls=1"):
        episodes.admit("robot-a", "call-2", 3, _placement(0), max_calls=1)
    assert episodes.view("robot-a").calls == 1


def test_eviction_keeps_call_budget_and_marks_the_next_call_cold():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 3, _placement(0))
    episodes.finish("robot-a", "call-1", {"frames": 21})
    assert episodes.evict("robot-a") == (3, set())
    view = episodes.view("robot-a")
    assert (view.calls, dict(view.held), view.cold) == (1, {}, "evicted")
    assert episodes.placement_for("robot-a", lambda: _placement(1)).replicas == {0: 1}
    episodes.admit("robot-a", "call-2", 4, _placement(1), max_calls=2)
    episodes.finish("robot-a", "call-2", {"frames": 21})
    with pytest.raises(ValueError, match="max_calls=2"):
        episodes.can_admit("robot-a", max_calls=2)


def test_slots_are_shared_by_rank_and_temporary_calls_compete_with_named_ones():
    slots = EpisodeSlots(1)
    assert slots.reserve("episode:a", {"w0", "w1"}) == 0
    assert slots.reserve("request:r", {"w1"}) is None
    assert slots.reserve("request:r", {"w2"}) == 0
    slots.release("episode:a", {"w0", "w1"}, 0)
    assert slots.reserve("episode:b", {"w1"}) == 0


def test_worker_counts_become_the_next_planning_view():
    conductor = object.__new__(Conductor)
    conductor._episode_array_names = frozenset({"frames"})
    request = SimpleNamespace(episode_admitted=True, episode_held={})
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 0, _placement(0))
    report = WorkerGraphsDone("call-1", [0], True, episode_held={"frames": 21})
    conductor._record_episode_held(report, request)
    episodes.finish("robot-a", "call-1", request.episode_held)
    assert episodes.view("robot-a").held["frames"] == 21
    with pytest.raises(ValueError, match="undeclared"):
        conductor._record_episode_held(
            WorkerGraphsDone("call-1", [0], True, episode_held={"other": 2}), request,
        )


def test_global_waiting_queue_skips_a_busy_episode_until_its_teardown():
    conductor = object.__new__(Conductor)
    conductor._episode_array_names = frozenset({"frames"})
    conductor._episode_records = EpisodeRecords()
    conductor._episode_records.admit("robot-a", "call-1", 0, _placement(0))
    conductor.max_concurrent_requests = None
    conductor.requests = {}

    def body(rid, episode):
        return NewRequestConductor(rid, {}, [], [], {}, {}, episode=episode)

    conductor.waiting_queue = [body("call-2", "robot-a"), body("other", "robot-b")]
    admitted = []
    conductor._admit_request = lambda request: admitted.append(request.request_id) or True
    conductor._try_admit_waiting()
    assert admitted == ["other"]
    assert [request.request_id for request in conductor.waiting_queue] == ["call-2"]
    conductor._episode_records.finish("robot-a", "call-1", {"frames": 21})
    conductor._try_admit_waiting()
    assert admitted == ["other", "call-2"]


def test_idle_timeout_sends_end_before_releasing_slot():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor._episode_evictions = MetricCounter(catalog.EPISODE_EVICTIONS)
    conductor._episode_idle_timeout_s = 10
    conductor.waiting_queue = []
    workers = {"w0"}
    assert conductor._episode_slots.reserve("episode:robot-a", workers) == 0
    conductor._episode_records.admit("robot-a", "call-1", 0, _placement(0), workers=workers)
    conductor._episode_records.finish("robot-a", "call-1", {"frames": 21})
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))
    last_end = conductor._episode_records._records["robot-a"].last_end

    assert conductor._expire_episode_slots(last_end + 11)
    assert [(worker, msg.body.episode_id, msg.body.episode_slot) for worker, msg in sent] == [
        ("w0", "robot-a", 0),
    ]
    assert conductor._episode_records.view("robot-a").cold == "idle"
    assert conductor._episode_slots.reserve("request:r", workers) == 0
    assert conductor._episode_evictions.snapshot().label_values == [["idle"]]
    assert conductor._episode_evictions.snapshot().values == [1.0]


def test_slot_pressure_requeues_without_leaking_a_registered_request():
    conductor = object.__new__(Conductor)
    conductor.requests = {}
    conductor.waiting_queue = []
    conductor._status_interval_s = None
    body = NewRequestConductor("call-1", {}, [], [], {}, {}, episode="robot-a")

    def plan(_body):
        conductor.requests[_body.request_id] = SimpleNamespace(routing=None, episode_admitted=False)
        raise EpisodeCapacityError

    conductor._do_ingest_request = plan
    assert conductor._admit_request(body) is False
    assert conductor.requests == {}
    assert conductor.waiting_queue == [body]


def test_close_waits_for_in_flight_call_then_ends_its_slot():
    conductor = object.__new__(Conductor)
    conductor._episode_array_names = frozenset({"frames"})
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor.waiting_queue = []
    workers = {"w0"}
    assert conductor._episode_slots.reserve("episode:robot-a", workers) == 0
    conductor._episode_records.admit("robot-a", "call-1", 0, _placement(0), workers=workers)
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))

    conductor._end_episode(EndEpisodeConductor("robot-a"))
    assert sent == []
    with pytest.raises(ValueError, match="is ending"):
        conductor._episode_records.can_admit("robot-a")
    conductor.requests = {"call-1": SimpleNamespace(
        routing=None, episode_admitted=True, episode_slot=0, episode_workers=workers,
        episode_id="robot-a", completion_sent=True, episode_held={"frames": 21},
    )}
    conductor._drop_request("call-1")
    assert len(sent) == 1
    assert sent[0][1].body.episode_id == "robot-a"
    assert sent[0][1].body.episode_slot == 0
    assert conductor._episode_slots.reserve("request:r", workers) == 0


def test_close_before_preprocessing_finishes_rejects_late_admission():
    episodes = EpisodeRecords()
    assert episodes.mark_ending("robot-a") is False
    with pytest.raises(ValueError, match="has ended"):
        episodes.can_admit("robot-a")


def test_eviction_sends_end_before_reusing_a_workers_slot():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor._episode_evictions = MetricCounter(catalog.EPISODE_EVICTIONS)
    conductor.waiting_queue = []
    workers = {"w0"}
    conductor._episode_slots.reserve("episode:old", workers)
    conductor._episode_records.admit("old", "call-1", 0, _placement(0), workers=workers)
    conductor._episode_records.finish("old", "call-1", {"frames": 21})
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))

    assert conductor._reserve_episode_slot("episode:new", workers) == 0
    sent.append(("w0", "next NEW_REQUEST"))
    assert sent[0][1].body.episode_id == "old"
    assert sent[0][1].body.episode_slot == 0
    assert sent[1][1] == "next NEW_REQUEST"
    assert conductor._episode_records.view("old").cold == "evicted"
    assert conductor._episode_evictions.snapshot().label_values == [["pressure"]]
    assert conductor._episode_evictions.snapshot().values == [1.0]


def test_failed_end_send_does_not_reassign_a_live_row():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor.waiting_queue = []
    workers = {"w0"}
    conductor._episode_slots.reserve("episode:old", workers)
    conductor._episode_records.admit("old", "call-1", 0, _placement(0), workers=workers)
    conductor._episode_records.finish("old", "call-1", {"frames": 21})

    def fail_send(_worker, _msg):
        raise OSError("outbox closed")

    conductor.communicator = SimpleNamespace(send=fail_send)
    with pytest.raises(OSError, match="outbox closed"):
        conductor._reserve_episode_slot("episode:new", workers)
    assert conductor._episode_slots.reserve("episode:new", workers) is None
    assert conductor._episode_records.view("old").held["frames"] == 21


def test_slot_pressure_does_not_evict_an_unrelated_replica():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor.waiting_queue = []
    conductor._episode_slots.reserve("episode:idle", {"w0"})
    conductor._episode_slots.reserve("request:busy", {"w1"})
    conductor._episode_records.admit("idle", "call-1", 0, _placement(0), workers={"w0"})
    conductor._episode_records.finish("idle", "call-1", {"frames": 21})
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))

    with pytest.raises(EpisodeCapacityError):
        conductor._reserve_episode_slot("episode:new", {"w1"})
    assert sent == []
    assert conductor._episode_records.view("idle").held["frames"] == 21


def test_cold_record_keeps_its_call_budget_until_connection_ends():
    episodes = EpisodeRecords()
    episodes.admit("robot-a", "call-1", 0, _placement(0), workers={"w0"}, max_calls=1)
    episodes.finish("robot-a", "call-1", {"frames": 21})
    episodes.evict("robot-a", "idle")
    last_end = episodes._records["robot-a"].last_end
    episodes.forget_expired_closed(last_end + 1000, 10)
    view = episodes.view("robot-a")
    assert (view.calls, dict(view.held), view.cold) == (1, {}, "idle")
    with pytest.raises(ValueError, match="max_calls=1"):
        episodes.can_admit("robot-a", max_calls=1)


def test_queued_episode_keeps_its_slot_under_pressure_and_idle_timeout():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor._episode_evictions = MetricCounter(catalog.EPISODE_EVICTIONS)
    conductor._episode_idle_timeout_s = 10
    conductor.waiting_queue = [NewRequestConductor("call-2", {}, [], [], {}, {}, episode="robot-a")]
    workers = {"w0"}
    conductor._episode_slots.reserve("episode:robot-a", workers)
    conductor._episode_records.admit("robot-a", "call-1", 0, _placement(0), workers=workers)
    conductor._episode_records.finish("robot-a", "call-1", {"frames": 21})
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))

    with pytest.raises(EpisodeCapacityError):
        conductor._reserve_episode_slot("episode:robot-b", workers)
    last_end = conductor._episode_records._records["robot-a"].last_end
    assert not conductor._expire_episode_slots(last_end + 11)
    assert sent == []
    assert conductor._episode_records.view("robot-a").held["frames"] == 21


def test_multireplica_pressure_has_no_partial_eviction_without_a_common_slot():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor._episode_evictions = MetricCounter(catalog.EPISODE_EVICTIONS)
    conductor.waiting_queue = []
    conductor._episode_slots.reserve("episode:idle", {"w0"})
    conductor._episode_slots.reserve("request:busy", {"w1"})
    conductor._episode_records.admit("idle", "call-1", 0, _placement(0), workers={"w0"})
    conductor._episode_records.finish("idle", "call-1", {"frames": 21})
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))

    with pytest.raises(EpisodeCapacityError):
        conductor._reserve_episode_slot("episode:new", {"w0", "w1"})
    assert sent == []
    assert conductor._episode_records.view("idle").held["frames"] == 21


def test_multireplica_pressure_evicts_all_blockers_of_a_common_slot():
    conductor = object.__new__(Conductor)
    conductor._episode_records = EpisodeRecords()
    conductor._episode_slots = EpisodeSlots(1)
    conductor._episode_evictions = MetricCounter(catalog.EPISODE_EVICTIONS)
    conductor.waiting_queue = []
    for episode_id, worker in (("old-a", "w0"), ("old-b", "w1")):
        conductor._episode_slots.reserve(f"episode:{episode_id}", {worker})
        conductor._episode_records.admit(episode_id, f"call-{episode_id}", 0, _placement(0), workers={worker})
        conductor._episode_records.finish(episode_id, f"call-{episode_id}", {"frames": 21})
    sent = []
    conductor.communicator = SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg)))

    assert conductor._reserve_episode_slot("episode:new", {"w0", "w1"}) == 0
    assert {(worker, msg.body.episode_id) for worker, msg in sent} == {
        ("w0", "old-a"), ("w1", "old-b"),
    }
    assert conductor._episode_slots.used_by_worker() == {"w0": 1, "w1": 1}
    assert conductor._episode_evictions.snapshot().values == [2.0]
