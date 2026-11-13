"""Episode ownership is separate from the resource that stores an array's rows."""

import pytest
import torch

from mstar.engine.state import Array, Block, Entry, KVEntry, Read, TensorType, Window
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


def _frames(**overrides):
    return Array("frames", **{
        "nodes": {"decoder"}, "layers": {0},
        "entry": Entry({"rgb": TensorType((3, 8, 8), dtype=torch.uint8)}),
        "max_len": None, "reads": {"context": Read(mask=Window(4))}, **overrides,
    })


def test_unbounded_dense_log_has_finite_read_derived_retention():
    frames = _frames(reads={"recent": Read(mask=Window(3)), "long": Read(mask=Window(7))})
    assert frames.max_len is None
    assert max(read.mask.size for read in frames.reads.values()) == 7
    with pytest.raises(ValueError, match="dense window storage binding"):
        build_specs([frames])


@pytest.mark.parametrize("override,match", [
    ({"reads": {}}, "at least one dense Window read"),
    ({"reads": {"context": Read(mask=Block())}}, "only a no-op Window read"),
    ({"reads": {"context": Read(mask=Window(3), queries="other")}}, "queries"),
    ({"positions": object()}, "append-only tensor log"),
    ({"rewritable": True}, "append-only tensor log"),
    ({"entry": Entry({})}, "append-only tensor log"),
])
def test_unbounded_log_refuses_declarations_without_provable_dense_retention(override, match):
    with pytest.raises(ValueError, match=match):
        _frames(**override)
