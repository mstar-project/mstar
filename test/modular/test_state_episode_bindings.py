"""Episode ownership is separate from the resource that stores an array's rows."""

import pytest

from mstar.engine.state import Array, KVEntry
from mstar.engine.state.adapters import PAGED, build_specs, storage_of
from mstar.engine.state.episodes import EpisodeRowBindings


def _kv(**overrides):
    return Array(
        "kv", nodes={"llm"}, layers={0}, entry=KVEntry(heads=2, head_dim=8),
        max_len=16, **overrides,
    )


def test_lifetime_is_declared_without_selecting_a_second_kv_storage():
    request = _kv()
    episode = _kv(lifetime="episode")
    assert request.lifetime == "request"
    assert storage_of(request) is storage_of(episode) is PAGED
    assert request != episode and hash(episode) == hash(_kv(lifetime="episode"))
    with pytest.raises(ValueError, match="episode lifetime needs row binding and storage teardown"):
        build_specs([episode])


@pytest.mark.parametrize("bad", [None, "session", "", 1, True])
def test_lifetime_rejects_values_outside_the_contract(bad):
    with pytest.raises(ValueError, match="Array 'kv'.lifetime"):
        _kv(lifetime=bad)


def test_named_episode_keeps_its_owner_across_requests_then_releases_it():
    rows = EpisodeRowBindings()
    first = rows.bind(11, 3, "robot-a")
    assert (first.slot, first.cold, first.temporary) == (3, True, False)
    assert rows.owner_of(11) == 3
    assert rows.unbind(11) == first

    second = rows.bind(12, 3, "robot-a")
    assert second.slot == 3 and not second.cold
    rows.unbind(12)
    assert rows.end("robot-a") == 3

    # The conductor may now assign the same slot to a different episode.
    assert rows.bind(13, 3, "robot-b").cold


def test_temporary_slot_is_cold_again_after_request_teardown():
    rows = EpisodeRowBindings()
    first = rows.bind(21, 4, None)
    assert first.cold and first.temporary
    assert rows.unbind(21) == first
    assert rows.bind(22, 4, None).cold


def test_a_live_request_or_another_episode_cannot_take_an_owned_slot():
    rows = EpisodeRowBindings()
    rows.bind(31, 5, "robot-a")
    with pytest.raises(ValueError, match="already bound"):
        rows.bind(31, 6, "robot-b")
    with pytest.raises(ValueError, match="already runs request"):
        rows.bind(32, 5, "robot-a")
    with pytest.raises(ValueError, match="still runs request"):
        rows.end("robot-a")
    rows.unbind(31)
    with pytest.raises(ValueError, match="belongs to 'robot-a'"):
        rows.bind(32, 5, "robot-b")
    with pytest.raises(ValueError, match="owns slot 5, not 6"):
        rows.bind(32, 6, "robot-a")


@pytest.mark.parametrize("handle,slot,episode", [
    (-1, 0, "a"), (True, 0, "a"), (0, -1, "a"), (0, False, "a"), (0, 0, ""),
])
def test_a_binding_refuses_invalid_assigned_ids(handle, slot, episode):
    with pytest.raises(ValueError):
        EpisodeRowBindings().bind(handle, slot, episode)
