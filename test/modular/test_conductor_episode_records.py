"""Episode placement and planning views use one conductor-owned record."""

import pytest

from mstar.conductor.episodes import EpisodeRecords
from mstar.conductor.routing import Placement


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
    assert episodes.end("robot-a") == 3
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
