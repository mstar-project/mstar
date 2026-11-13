"""Episode ownership and dense whole-read retention in the state contract."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.dense_window import DenseWindowPool, DenseWindowSpec
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Block, Entry, KVEntry, Read, TensorType, Window
from mstar.engine.state.adapters import DENSE_WINDOW, PAGED, build_specs, node_handles, state_steps, storage_of
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
    assert storage_of(frames) is DENSE_WINDOW
    with pytest.raises(ValueError, match="exactly one read"):
        build_specs([frames])


def test_dense_window_keeps_only_latest_executed_frames_across_calls():
    frames = _frames(reads={"context": Read(mask=Window(48))})
    (spec,) = build_specs([frames])
    assert isinstance(spec, DenseWindowSpec)
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    pool.ingest_request("a")
    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()
    executed = torch.empty((0, 3, 8, 8), dtype=torch.uint8)
    for span in (21, 24, 24, 24):
        values = torch.arange(executed.shape[0], executed.shape[0] + span, dtype=torch.uint8)
        chunk = values[:, None, None, None].expand(-1, 3, 8, 8).clone()
        step = state_steps(SubmoduleStep(
            segments=[Segment("a", "main", span)], state={"frames": ArrayStep()},
        ), {"frames": handle}, graph_walk="decode")
        step.set_ctx(StepContext(request_ids=["a"], graph_walk="decode", slot=0, capture=False))
        assert pool.admit(step.steps["frames"], step.ctx).ok
        pool.plan(step.steps["frames"], step.ctx)
        step.begin_calls()
        handle.add(0, rgb=chunk)
        reached = handle.read(0)
        executed = torch.cat((executed, chunk))
        assert torch.equal(reached.fields["rgb"], executed[-48:])
        assert reached.row_offsets.tolist() == [0, min(48, executed.shape[0])]
        step.check_calls()
        pool.commit(step.steps["frames"], step.ctx)
        step.end_calls()
        assert pool.logical_length("a", "main") == executed.shape[0]

    # A later zero-span read still addresses the resident row, then teardown
    # clears its logical count before the same request id is reused.
    step = state_steps(SubmoduleStep(
        segments=[Segment("a", "main", 0)], state={"frames": ArrayStep()},
    ), {"frames": handle}, graph_walk="decode")
    step.set_ctx(StepContext(request_ids=["a"], graph_walk="decode", slot=0, capture=False))
    assert pool.admit(step.steps["frames"], step.ctx).ok
    pool.plan(step.steps["frames"], step.ctx)
    step.begin_calls()
    assert torch.equal(handle.read(0).fields["rgb"], executed[-48:])
    step.check_calls()
    pool.commit(step.steps["frames"], step.ctx)
    step.end_calls()
    pool.remove_request("a")
    pool.ingest_request("a")
    assert pool.logical_length("a", "main") == 0
    step = state_steps(SubmoduleStep(
        segments=[Segment("a", "main", 0)], state={"frames": ArrayStep()},
    ), {"frames": handle}, graph_walk="decode")
    step.set_ctx(StepContext(request_ids=["a"], graph_walk="decode", slot=0, capture=False))
    assert pool.admit(step.steps["frames"], step.ctx).ok
    pool.plan(step.steps["frames"], step.ctx)
    step.begin_calls()
    assert handle.read(0).fields["rgb"].shape == (0, 3, 8, 8)
    step.check_calls()
    pool.commit(step.steps["frames"], step.ctx)
    step.end_calls()


def test_dense_window_packs_different_row_lengths_and_refuses_capture():
    frames = _frames(reads={"context": Read(mask=Window(4))})
    (spec,) = build_specs([frames], kernel_settings={"frames": {"max_slots": 3}})
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    assert not pool.can_capture()
    for rid in ("a", "b"):
        pool.ingest_request(rid)
    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()
    step = state_steps(SubmoduleStep(
        segments=[Segment("a", "main", 2), Segment("b", "main", 5)], state={"frames": ArrayStep()},
    ), {"frames": handle}, graph_walk="decode")
    step.set_ctx(StepContext(request_ids=["a", "b"], graph_walk="decode", slot=0, capture=False))
    assert pool.admit(step.steps["frames"], step.ctx).ok
    pool.plan(step.steps["frames"], step.ctx)
    step.begin_calls()
    values = torch.arange(7, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 8, 8).clone()
    handle.add(0, rgb=values)
    reached = handle.read(0)
    assert reached.row_offsets.tolist() == [0, 2, 6]
    assert torch.equal(reached.fields["rgb"], torch.cat((values[:2], values[-4:])))
    step.check_calls()
    pool.commit(step.steps["frames"], step.ctx)
    step.end_calls()


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
