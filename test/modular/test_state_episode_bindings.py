"""Episode ownership and dense whole-read retention in the state contract."""

from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.episodes import EpisodeRecords
from mstar.conductor.routing import Placement
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.dense_window import DenseWindowPool, DenseWindowSpec
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Block, Entry, KVEntry, Read, TensorType, Window
from mstar.engine.state.adapters import DENSE_WINDOW, PAGED, build_specs, node_handles, state_steps, storage_of
from mstar.engine.state.episodes import EpisodeRowBindings
from mstar.utils.ipc_format import EndEpisode, RemoveRequest, WorkerMessage, WorkerMessageType
from mstar.worker.input_ingest import InputIngest
from mstar.worker.step_pipeline import StepPipeline


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


def test_idle_slot_is_available_only_to_its_current_episode():
    rows = EpisodeRowBindings()
    assert rows.slot_available(2, "a")
    rows.bind(11, 2, "a")
    assert not rows.slot_available(2, "a")
    rows.unbind(11)
    assert rows.slot_available(2, "a")
    assert not rows.slot_available(2, "b")
    assert not rows.slot_available(2, None)
    rows.end("a")
    assert rows.slot_available(2, "b")
    assert rows.slot_available(2, None)


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
        step.commit_calls()
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
    step.commit_calls()
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
    step.commit_calls()
    step.end_calls()


def test_failed_warm_step_keeps_the_previous_episode_window():
    frames = _frames(reads={"context": Read(mask=Window(4))}, lifetime="episode")
    (spec,) = build_specs([frames])
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    owners = EpisodeRowBindings()
    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()

    def open_request(rid):
        pool.ingest_request(rid)
        binding = owners.bind(rid, 0, "episode-a")
        pool.bind_episode(rid, 0, "episode-a", cold=binding.cold, reset=False)

    def start_step(rid, span):
        step = state_steps(SubmoduleStep(
            segments=[Segment(rid, "main", span)], state={"frames": ArrayStep()},
        ), {"frames": handle}, graph_walk="decode")
        step.set_ctx(StepContext(request_ids=[rid], graph_walk="decode", slot=0, capture=False))
        assert pool.admit(step.steps["frames"], step.ctx).ok
        pool.plan(step.steps["frames"], step.ctx)
        step.begin_calls()
        return step

    open_request(1)
    first = start_step(1, 4)
    original = torch.arange(1, 5, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 8, 8).clone()
    handle.add(0, rgb=original)
    first.check_calls()
    pool.commit(first.steps["frames"], first.ctx)
    first.commit_calls()
    first.end_calls()
    pool.remove_request(1)
    owners.unbind(1)

    open_request(2)
    failed = start_step(2, 1)
    handle.add(0, rgb=torch.full((1, 3, 8, 8), 99, dtype=torch.uint8))
    assert handle.read(0).fields["rgb"][-1, 0, 0, 0].item() == 99
    pool.rollback_admit(failed.steps["frames"], failed.ctx)
    failed.end_calls()
    pool.remove_request(2)
    owners.unbind(2)

    open_request(3)
    resumed = start_step(3, 0)
    assert torch.equal(handle.read(0).fields["rgb"], original)
    resumed.check_calls()
    pool.commit(resumed.steps["frames"], resumed.ctx)
    resumed.commit_calls()
    resumed.end_calls()


def _text_embeds():
    return Array(
        "text_embeds", nodes={"backbone"}, layers={0}, labels={"cond", "blank"},
        entry=Entry({"h": TensorType((2,), dtype=torch.float32)}),
        max_len=None, reads={"history": Read(mask=Window(3))}, lifetime="episode",
    )


def test_dense_episode_step_keeps_two_labels_separate_across_calls_and_rollback():
    embeds = _text_embeds()
    (spec,) = build_specs([embeds])
    assert spec.config.usable_slots == 2
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    owners = EpisodeRowBindings()
    handle = node_handles(SimpleNamespace(declare_state=lambda: [embeds]), "backbone",
                          {"text_embeds": pool})["text_embeds"]

    def open_request(rid):
        pool.ingest_request(rid)
        binding = owners.bind(rid, 0, "episode-a")
        pool.bind_episode(rid, 0, "episode-a", cold=binding.cold, reset=False)

    def start(rid, cond_span, blank_span):
        step = state_steps(SubmoduleStep(
            segments=[Segment(rid, "cond", cond_span), Segment(rid, "blank", blank_span)],
            state={"text_embeds": ArrayStep()},
        ), {"text_embeds": handle}, graph_walk="backbone")
        step.set_ctx(StepContext(request_ids=[rid], graph_walk="backbone", slot=0, capture=False))
        assert pool.admit(step.steps["text_embeds"], step.ctx).ok
        pool.plan(step.steps["text_embeds"], step.ctx)
        step.begin_calls()
        return step

    def commit(step):
        step.check_calls()
        pool.commit(step.steps["text_embeds"], step.ctx)
        step.commit_calls()
        step.end_calls()

    cond = torch.tensor([[1., 1.], [2., 2.]])
    blank = torch.tensor([[8., 8.]])
    open_request(1)
    first = start(1, 2, 1)
    handle.add(0, h=torch.cat((cond, blank)))
    reached = handle.read(0)
    assert reached.row_offsets.tolist() == [0, 2, 3]
    assert torch.equal(reached.fields["h"], torch.cat((cond, blank)))
    commit(first)
    assert (pool.logical_length(1, "cond"), pool.logical_length(1, "blank"), pool.held(1)) == (2, 1, 3)
    pool.remove_request(1)
    owners.unbind(1)

    open_request(2)
    warm = start(2, 0, 0)
    reached = handle.read(0)
    assert reached.row_offsets.tolist() == [0, 2, 3]
    assert torch.equal(reached.fields["h"], torch.cat((cond, blank)))
    commit(warm)

    failed = start(2, 2, 1)
    changed = torch.tensor([[9., 9.], [10., 10.], [11., 11.]])
    handle.add(0, h=changed)
    assert handle.read(0).row_offsets.tolist() == [0, 3, 5]
    pool.rollback_admit(failed.steps["text_embeds"], failed.ctx)
    failed.end_calls()
    pool.remove_request(2)
    owners.unbind(2)

    open_request(3)
    resumed = start(3, 0, 0)
    reached = handle.read(0)
    assert reached.row_offsets.tolist() == [0, 2, 3]
    assert torch.equal(reached.fields["h"], torch.cat((cond, blank)))
    commit(resumed)

    newest = torch.tensor([[3., 3.], [4., 4.], [5., 5.]])
    wrapped = start(3, 3, 0)
    handle.add(0, h=newest)
    reached = handle.read(0)
    assert reached.row_offsets.tolist() == [0, 3, 4]
    assert torch.equal(reached.fields["h"], torch.cat((newest, blank)))
    commit(wrapped)
    assert (pool.logical_length(3, "cond"), pool.logical_length(3, "blank"), pool.held(3)) == (5, 1, 4)


def test_dense_episode_multilabel_packs_mixed_requests_and_enforces_slot_pressure():
    embeds = _text_embeds()
    (spec,) = build_specs([embeds], kernel_settings={"text_embeds": {"max_slots": 5}})
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    for rid in (1, 2, 3):
        pool.ingest_request(rid)
    handle = node_handles(SimpleNamespace(declare_state=lambda: [embeds]), "backbone",
                          {"text_embeds": pool})["text_embeds"]
    segments = [Segment(1, "cond", 2), Segment(2, "cond", 1),
                Segment(1, "blank", 1), Segment(2, "blank", 2)]
    step = state_steps(SubmoduleStep(segments=segments, state={"text_embeds": ArrayStep()}),
                       {"text_embeds": handle}, graph_walk="backbone")
    step.set_ctx(StepContext(request_ids=[1, 2], graph_walk="backbone", slot=0, capture=False))
    assert pool.admit(step.steps["text_embeds"], step.ctx).ok
    pool.plan(step.steps["text_embeds"], step.ctx)
    step.begin_calls()
    values = torch.arange(6, dtype=torch.float32)[:, None].expand(-1, 2).clone()
    handle.add(0, h=values)
    reached = handle.read(0)
    assert reached.row_offsets.tolist() == [0, 2, 3, 4, 6]
    assert torch.equal(reached.fields["h"], values)
    step.check_calls()
    pool.commit(step.steps["text_embeds"], step.ctx)
    step.commit_calls()
    step.end_calls()
    assert pool.held(1) == pool.held(2) == 3

    pressure = state_steps(SubmoduleStep(
        segments=[Segment(3, "cond", 1), Segment(3, "blank", 1)],
        state={"text_embeds": ArrayStep()},
    ), {"text_embeds": handle}, graph_walk="backbone")
    pressure.set_ctx(StepContext(request_ids=[3], graph_walk="backbone", slot=0, capture=False))
    assert not pool.admit(pressure.steps["text_embeds"], pressure.ctx).ok
    pool.rollback_admit(pressure.steps["text_embeds"], pressure.ctx)

    warm = state_steps(SubmoduleStep(
        segments=[Segment(1, "cond", 0), Segment(2, "cond", 0),
                  Segment(1, "blank", 0), Segment(2, "blank", 0)],
        state={"text_embeds": ArrayStep()},
    ), {"text_embeds": handle}, graph_walk="backbone")
    warm.set_ctx(StepContext(request_ids=[1, 2], graph_walk="backbone", slot=0, capture=False))
    assert pool.admit(warm.steps["text_embeds"], warm.ctx).ok
    pool.plan(warm.steps["text_embeds"], warm.ctx)
    warm.begin_calls()
    assert torch.equal(handle.read(0).fields["h"], values)
    warm.check_calls()
    pool.commit(warm.steps["text_embeds"], warm.ctx)
    warm.commit_calls()
    warm.end_calls()

    captured = state_steps(SubmoduleStep(
        segments=[Segment(1, "cond", 0), Segment(1, "blank", 0)],
        state={"text_embeds": ArrayStep()},
    ), {"text_embeds": handle}, graph_walk="backbone")
    captured.set_ctx(StepContext(request_ids=[1], graph_walk="backbone", slot=0, capture=True))
    with pytest.raises(ValueError, match="cannot replay a capture"):
        captured.begin_calls()


def test_dense_window_refuses_rebuild_drop_without_mutating_history():
    frames = _frames(reads={"context": Read(mask=Window(4))})
    (spec,) = build_specs([frames])
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    assert not pool.supports_drop
    pool.ingest_request(1)
    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()
    step = state_steps(SubmoduleStep(
        segments=[Segment(1, "main", 4)], state={"frames": ArrayStep()},
    ), {"frames": handle}, graph_walk="decode")
    step.set_ctx(StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False))
    assert pool.admit(step.steps["frames"], step.ctx).ok
    pool.plan(step.steps["frames"], step.ctx)
    step.begin_calls()
    data = torch.arange(1, 5, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 8, 8).clone()
    handle.add(0, rgb=data)
    step.check_calls()
    pool.commit(step.steps["frames"], step.ctx)
    step.commit_calls()
    step.end_calls()
    assert pool.check_drop(1, 4) is not None
    with pytest.raises(RuntimeError, match="cannot be rebuilt"):
        pool.drop(1, 4)
    assert pool.logical_length(1, "main") == 4
    resumed = state_steps(SubmoduleStep(
        segments=[Segment(1, "main", 0)], state={"frames": ArrayStep()},
    ), {"frames": handle}, graph_walk="decode")
    resumed.set_ctx(StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False))
    assert pool.admit(resumed.steps["frames"], resumed.ctx).ok
    pool.plan(resumed.steps["frames"], resumed.ctx)
    resumed.begin_calls()
    assert torch.equal(handle.read(0).fields["rgb"], data)
    resumed.end_calls()


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
    step.commit_calls()
    step.end_calls()


def test_episode_owner_survives_request_teardown_then_reset_and_end_clear_it():
    frames = _frames(reads={"context": Read(mask=Window(4))}, lifetime="episode")
    (spec,) = build_specs([frames])
    assert spec.episode_lifetime
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    owners = EpisodeRowBindings()

    def open_request(rid, episode, *, reset=False):
        pool.ingest_request(rid)
        bound = owners.bind(rid, 0, episode)
        pool.bind_episode(rid, 0, episode, cold=bound.cold, reset=reset)

    def call(rid, chunk=None):
        span = 0 if chunk is None else len(chunk)
        step = state_steps(SubmoduleStep(
            segments=[Segment(rid, "main", span)], state={"frames": ArrayStep()},
        ), {"frames": handle}, graph_walk="decode")
        step.set_ctx(StepContext(request_ids=[rid], graph_walk="decode", slot=0, capture=False))
        assert pool.admit(step.steps["frames"], step.ctx).ok
        pool.plan(step.steps["frames"], step.ctx)
        step.begin_calls()
        if chunk is not None:
            handle.add(0, rgb=chunk)
        reached = handle.read(0).fields["rgb"].clone()
        step.check_calls()
        pool.commit(step.steps["frames"], step.ctx)
        step.commit_calls()
        step.end_calls()
        return reached

    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()
    chunk = torch.arange(3, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 8, 8).clone()
    open_request(11, "episode-a")
    assert torch.equal(call(11, chunk), chunk)
    pool.remove_request(11)
    owners.unbind(11)

    open_request(12, "episode-a")
    assert torch.equal(call(12), chunk)
    pool.remove_request(12)
    owners.unbind(12)

    open_request(13, "episode-a", reset=True)
    assert call(13).shape[0] == 0
    pool.remove_request(13)
    owners.unbind(13)
    pool.end_episode(owners.idle_slot("episode-a"))
    owners.end("episode-a")

    open_request(14, "episode-b")
    assert call(14).shape[0] == 0
    pool.remove_request(14)
    owners.unbind(14)
    pool.end_episode(owners.idle_slot("episode-b"))
    owners.end("episode-b")

    # Idless calls are temporary even when the conductor uses the same logical
    # slot. Their resource rows leave on ordinary request teardown.
    open_request(15, None)
    assert torch.equal(call(15, chunk), chunk)
    pool.remove_request(15)
    owners.unbind(15)
    open_request(16, None)
    assert call(16).shape[0] == 0


def test_dense_window_pool_refuses_a_second_identity_without_capacity():
    frames = _frames(reads={"context": Read(mask=Window(4))}, lifetime="episode")
    (spec,) = build_specs([frames])
    assert spec.config.usable_slots == 1
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    for rid in (1, 2):
        pool.ingest_request(rid)
    ctx = StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False)
    first = state_steps(SubmoduleStep(
        segments=[Segment(1, "main", 1)], state={"frames": ArrayStep()},
    ), {"frames": node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder",
                                {"frames": pool})["frames"]}, graph_walk="decode")
    assert pool.admit(first.steps["frames"], ctx).ok
    pool.plan(first.steps["frames"], ctx)
    pool.commit(first.steps["frames"], ctx)
    ctx.request_ids = [2]
    second = state_steps(SubmoduleStep(
        segments=[Segment(2, "main", 1)], state={"frames": ArrayStep()},
    ), {"frames": node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder",
                                {"frames": pool})["frames"]}, graph_walk="decode")
    assert not pool.admit(second.steps["frames"], ctx).ok


def test_dense_window_requires_every_layer_before_committing_one_logical_length():
    frames = _frames(layers={0, 1}, reads={"context": Read(mask=Window(4))})
    (spec,) = build_specs([frames])
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    pool.ingest_request(1)
    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()
    step = state_steps(SubmoduleStep(
        segments=[Segment(1, "main", 3)], state={"frames": ArrayStep()},
    ), {"frames": handle}, graph_walk="decode")
    step.set_ctx(StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False))
    assert pool.admit(step.steps["frames"], step.ctx).ok
    pool.plan(step.steps["frames"], step.ctx)
    step.begin_calls()
    chunk = torch.full((3, 3, 8, 8), 5, dtype=torch.uint8)
    handle.add(0, rgb=chunk)
    with pytest.raises(ValueError, match="layer.*no add"):
        step.check_calls()
    handle.add(1, rgb=chunk + 1)
    assert torch.equal(handle.read(0).fields["rgb"], chunk)
    assert torch.equal(handle.read(1).fields["rgb"], chunk + 1)
    step.check_calls()
    pool.commit(step.steps["frames"], step.ctx)
    step.commit_calls()
    step.end_calls()
    assert pool.logical_length(1, "main") == 3


def test_engine_binds_warm_episode_and_releases_it_only_on_explicit_end():
    frames = _frames(reads={"context": Read(mask=Window(4))}, lifetime="episode")
    (spec,) = build_specs([frames])
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    engine = Engine(None)
    engine._resources = {"frames": pool}
    engine._episode_pools = {"frames": pool}
    engine._resource_nodes = {"frames": frozenset({"decoder"})}
    engine._loopback_steps = {}

    class Runner:
        def ingest_request(self, rid, overrides, arrival):
            pool.ingest_request(rid)

        def remove_request(self, rid):
            pool.remove_request(rid)

        def publish_snapshot(self, *args, **kwargs):
            return {}

        def publish(self, *args, **kwargs):
            return {}

    engine._runner = Runner()
    (handle,) = node_handles(SimpleNamespace(declare_state=lambda: [frames]), "decoder", {"frames": pool}).values()

    def read(rid, span=0):
        step = state_steps(SubmoduleStep(
            segments=[Segment(rid, "main", span)], state={"frames": ArrayStep()},
        ), {"frames": handle}, graph_walk="decode")
        step.set_ctx(StepContext(request_ids=[rid], graph_walk="decode", slot=0, capture=False))
        assert pool.admit(step.steps["frames"], step.ctx).ok
        pool.plan(step.steps["frames"], step.ctx)
        step.begin_calls()
        if span:
            data = torch.full((span, 3, 8, 8), 7, dtype=torch.uint8)
            handle.add(0, rgb=data)
        reached = handle.read(0).fields["rgb"].clone()
        step.check_calls()
        pool.commit(step.steps["frames"], step.ctx)
        step.commit_calls()
        step.end_calls()
        return reached

    with pytest.raises(ValueError, match="conductor-assigned episode_slot"):
        engine.add_request(10, episode_id="a")
    engine.add_request(11, episode_id="a", episode_slot=0)
    assert read(11, 3).shape[0] == 3
    first_batch = ExecutingBatch("decoder", {}, StepContext([11], "decode", 0, False))
    engine.finalize_batch(first_batch)
    assert first_batch.episode_held[11] == {"frames": 3}
    terminal_batch = ExecutingBatch("action_head", {}, StepContext([11], "decode", 0, False))
    engine.finalize_batch(terminal_batch)
    assert terminal_batch.episode_held[11] == {"frames": 3}
    records = EpisodeRecords()
    records.admit("a", "request", 0, Placement({0: 0}, {}))
    records.finish("a", "request", terminal_batch.episode_held[11])
    assert records.view("a").held["frames"] == 3
    assert read(11, 2).shape[0] == 4
    second_batch = ExecutingBatch("decoder", {}, StepContext([11], "decode", 0, False))
    engine.finalize_batch(second_batch)
    assert second_batch.episode_held[11] == {"frames": 4}
    assert first_batch.episode_held[11] == {"frames": 3}, "postprocess must not see the next commit"
    assert not engine.end_episode("a", 0)
    engine.remove_request(11)
    engine.add_request(12, episode_id="a", episode_slot=0)
    assert torch.all(read(12) == 7)
    engine.remove_request(12)
    engine.add_request(14, episode_id="a", episode_slot=0, episode_clear=["frames"])
    assert read(14).shape[0] == 0
    engine.remove_request(14)
    assert engine.end_episode("a", 0)
    engine.add_request(13, episode_id="b", episode_slot=0)
    assert read(13).shape[0] == 0
    engine.remove_request(13)
    assert engine.end_episode("b", 0)


def test_worker_defers_idle_episode_close_until_request_teardown():
    end = EndEpisode("a", 2)
    pending = [end]
    active = True

    class Manager:
        def end_episode(self, episode_id, episode_slot):
            assert (episode_id, episode_slot) == ("a", 2)
            return not active

    pipeline = object.__new__(StepPipeline)
    pipeline.lifecycle = SimpleNamespace(pending_episode_ends=pending)
    pipeline.engine_manager = Manager()
    pipeline.ingest = SimpleNamespace(has_pending_episode=lambda episode_id, slot: False)
    pipeline._apply_episode_ends()
    assert pending == [end]
    active = False
    pipeline._apply_episode_ends()
    assert pending == []


def test_end_waits_for_an_unadmitted_episode_queued_behind_a_busy_slot():
    end = EndEpisode("b", 0)
    queued = SimpleNamespace(request_id="call-b", episode_id="b", episode_slot=0)
    ingest = object.__new__(InputIngest)
    ingest._pending_episode_requests = {0: [queued]}
    busy = True
    ingest._engines = SimpleNamespace(episode_slot_available=lambda slot, episode: not busy)
    admitted = []
    ingest._add_new_request = admitted.append

    closed = False

    def end_episode(episode_id, slot):
        nonlocal closed
        assert (episode_id, slot) == ("b", 0)
        if not admitted:
            closed = True  # This would lose END before B owns its row.
            return True
        if busy:
            return False
        closed = True
        return True

    pipeline = object.__new__(StepPipeline)
    pipeline.ingest = ingest
    pipeline.lifecycle = SimpleNamespace(pending_episode_ends=[end])
    pipeline.engine_manager = SimpleNamespace(end_episode=end_episode)
    pipeline._apply_episode_ends()
    assert not closed and pipeline.lifecycle.pending_episode_ends == [end]

    busy = False  # The previous owner has drained; B can be admitted.
    ingest.retry_episode_requests()
    assert admitted == [queued]
    busy = True  # B now runs on this slot.
    pipeline._apply_episode_ends()
    assert not closed and pipeline.lifecycle.pending_episode_ends == [end]
    busy = False  # B has drained after its REMOVE.
    pipeline._apply_episode_ends()
    assert closed and pipeline.lifecycle.pending_episode_ends == []


def test_deferred_episode_new_request_replays_all_partitions_then_next_call():
    ingest = object.__new__(InputIngest)
    first = SimpleNamespace(request_id="call-a")
    second_partition = SimpleNamespace(request_id="call-a")
    next_call = SimpleNamespace(request_id="call-b")
    ingest._pending_episode_requests = {2: [first, second_partition, next_call]}
    ingest._engines = SimpleNamespace(episode_slot_available=lambda slot, episode: True)
    replayed = []
    ingest._add_new_request = replayed.append
    ingest.retry_episode_requests()
    assert replayed == [first, second_partition]
    assert ingest._pending_episode_requests == {2: [next_call]}
    ingest._drop_pending_episode_request("call-b")
    assert ingest._pending_episode_requests == {}


def test_remove_then_new_in_one_worker_batch_waits_for_drained_release():
    ingest = object.__new__(InputIngest)
    ingest._pending_episode_requests = {}
    ingest._runtime = SimpleNamespace(get_rid_handle=lambda wire: 11 if wire == "old" else None)
    busy = True
    ingest._engines = SimpleNamespace(episode_slot_available=lambda slot, episode: not busy)
    removed = []
    ingest._lifecycle = SimpleNamespace(remove=lambda rid, body, seq: removed.append((rid, seq)))
    ingest._lockstep = SimpleNamespace(last_consumed_seq=3)
    ingest._remove_request(RemoveRequest("old"))
    assert removed == [(11, 3)]
    next_call = SimpleNamespace(request_id="new", episode_slot=2)
    ingest._add_new_request(next_call)
    assert ingest._pending_episode_requests == {2: [next_call]}
    busy = False  # `_release_request` has finished on this worker.
    replayed = []
    ingest._add_new_request = replayed.append
    ingest.retry_episode_requests()
    assert replayed == [next_call]


def test_idle_end_then_new_in_one_worker_batch_waits_for_slot_release():
    rows = EpisodeRowBindings()
    rows.bind(11, 2, "old")
    rows.unbind(11)
    ingest = object.__new__(InputIngest)
    ingest._pending_episode_requests = {}
    ingest._runtime = SimpleNamespace(get_rid_handle=lambda wire: None)

    ingest._engines = SimpleNamespace(
        episode_slot_busy=rows.slot_busy,
        episode_slot_available=lambda slot, episode_id: rows.idle_slot("old") is None,
    )
    ends = []
    ingest._lifecycle = SimpleNamespace(
        end_episode=ends.append,
        takes_input=lambda wire: pytest.fail("new call admitted before old episode ended"),
    )
    ingest._handlers = {
        WorkerMessageType.END_EPISODE: ingest._end_episode,
        WorkerMessageType.NEW_REQUEST: ingest._add_new_request,
    }
    old_end = EndEpisode("old", 2)
    next_call = SimpleNamespace(request_id="new", episode_id="new", episode_slot=2)
    ingest._dispatch([
        WorkerMessage(WorkerMessageType.END_EPISODE, old_end),
        WorkerMessage(WorkerMessageType.NEW_REQUEST, next_call),
    ])
    assert ends == [old_end]
    assert ingest._pending_episode_requests == {2: [next_call]}

    rows.end("old")
    replayed = []
    ingest._add_new_request = replayed.append
    ingest.retry_episode_requests()
    assert replayed == [next_call]


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
