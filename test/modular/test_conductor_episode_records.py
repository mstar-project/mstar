"""Episode placement and planning views use one conductor-owned record."""

from types import SimpleNamespace

import pytest

from mstar.conductor.conductor import Conductor
from mstar.conductor.episodes import EpisodeRecords, EpisodeSlots
from mstar.conductor.routing import Placement
from mstar.utils.ipc_format import NewRequestConductor, WorkerGraphsDone


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
