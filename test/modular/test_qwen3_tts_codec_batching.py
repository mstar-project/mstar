"""How the scheduler batches Qwen3-TTS codec windows under ``codec_split_windows``:
the real ``CodecSubmodule`` and ``Engine`` grouping, with readiness faked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_micro_scheduler import _batch, _Engine, _Manager, _next_batch
from test_qwen3_tts_model import _FakeCodecDecoder, _tiny_model_config

from mstar.engine.engine import Engine
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.submodules import SNACDecoderSubmodule
from mstar.model.qwen3_tts.submodules import CodecSubmodule
from mstar.streaming.stream_buffer import StreamChunkInfo
from mstar.worker.micro_scheduler import MicroScheduler

NODE = "Codec"
WALK = "codec_chunk"
# The tiny config's ramp: a first window of 1 frame, then 1 + 3, then 2 + 3.
FIRST, SECOND, WIDE = 1, 4, 5


def _engine(submodule, node=NODE, walk=WALK, max_bs=16) -> _Engine:
    runner = SimpleNamespace(captures_walk=lambda name: name == walk)
    engine = _Engine(max_bs=max_bs)
    engine._submodules = {node: SimpleNamespace(submodule=submodule, accelerator_graph_runner=runner)}
    engine.splits_by_capture_key = lambda n, w: Engine.splits_by_capture_key(engine, n, w)
    engine.capture_group_order = lambda n, w: Engine.capture_group_order(engine, n, w)
    engine.capture_group = lambda *a, **kw: Engine.capture_group(engine, *a, **kw)
    return engine


def _scheduler(engine: _Engine) -> MicroScheduler:
    return MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine), parallel_leader_nodes={NODE},
    )


def _codec_engine(split=True, max_bs=16) -> _Engine:
    return _engine(CodecSubmodule(_FakeCodecDecoder(4), _tiny_model_config(), split_windows=split), max_bs=max_bs)


def _manager(windows: dict[str, int], ready=None) -> _Manager:
    """Rows with their window widths; ``ready`` names those on the ready queue, default all."""
    manager = _Manager(list(windows) if ready is None else ready, node=NODE, walk=WALK)
    manager.stream_chunks = {
        rid: {"codec_tokens": StreamChunkInfo(start_offset=0, context_items=0, num_items=n, is_final=False)}
        for rid, n in windows.items()
    }
    return manager


def _backlog(rids):
    return _batch(rids, node=NODE, walk=WALK)


def _rows(batch) -> list:
    return list(batch.request_to_worker_graph)


def test_a_first_window_alone_runs_alone():
    batch = _next_batch(_scheduler(_codec_engine()), _manager({"a": FIRST}))
    assert _rows(batch) == ["a"]


def test_several_first_windows_batch_together():
    sched = _scheduler(_codec_engine())
    batch = _next_batch(sched, _manager({"a": FIRST, "b": FIRST, "c": FIRST}))
    assert _rows(batch) == ["a", "b", "c"]
    assert sched.backlog == {}


def test_a_first_window_never_rides_with_a_wider_one_and_runs_first():
    sched = _scheduler(_codec_engine())
    manager = _manager({"w0": WIDE, "a": FIRST, "s0": SECOND, "b": FIRST, "w1": WIDE})

    assert _rows(_next_batch(sched, manager)) == ["a", "b"]
    assert _rows(sched.backlog[(NODE, WALK)]) == ["w0", "s0", "w1"]


def test_wider_windows_batch_by_width_narrowest_first():
    sched = _scheduler(_codec_engine())
    manager = _manager({"w0": WIDE, "a": FIRST, "s0": SECOND, "w1": WIDE})
    _next_batch(sched, manager)

    assert _rows(_next_batch(sched, manager)) == ["s0"]
    assert _rows(_next_batch(sched, manager)) == ["w0", "w1"]
    assert sched.backlog == {}


def test_a_first_window_goes_ahead_of_a_backlog_that_is_older():
    sched = _scheduler(_codec_engine())
    sched.backlog[(NODE, WALK)] = _backlog(["w0", "w1"])
    manager = _manager({"w0": WIDE, "w1": WIDE, "a": FIRST}, ready=["a"])

    assert _rows(_next_batch(sched, manager)) == ["a"]
    assert _rows(sched.backlog[(NODE, WALK)]) == ["w0", "w1"]


def test_a_first_window_parked_behind_older_rows_still_goes_first():
    sched = _scheduler(_codec_engine())
    sched.backlog[(NODE, WALK)] = _backlog(["w0", "a", "w1"])
    manager = _manager({"w0": WIDE, "a": FIRST, "w1": WIDE}, ready=[])

    assert _rows(_next_batch(sched, manager)) == ["a"]
    assert _rows(sched.backlog[(NODE, WALK)]) == ["w0", "w1"]


def test_a_full_backlog_is_not_passed_by_a_first_window():
    """The scheduler doesn't scan for fresh rows when the backlog fills a step,
    so wide windows at the cap can't be starved by arrivals."""
    sched = _scheduler(_codec_engine(max_bs=2))
    sched.backlog[(NODE, WALK)] = _backlog(["w0", "w1"])
    manager = _manager({"w0": WIDE, "w1": WIDE, "a": FIRST}, ready=["a"])

    assert _rows(_next_batch(sched, manager)) == ["w0", "w1"]


def test_a_first_window_that_cannot_run_does_not_set_the_group():
    engine = _codec_engine()
    engine.not_ready = {"a"}
    sched = _scheduler(engine)
    sched.backlog[(NODE, WALK)] = _backlog(["a", "w0"])
    manager = _manager({"a": FIRST, "w0": WIDE}, ready=[])

    assert _rows(_next_batch(sched, manager)) == ["w0"]


def test_a_speculation_merge_keeps_its_own_rows_group():
    """A merge has to share the capture of the rows already in its step."""
    sched = _scheduler(_codec_engine())
    manager = _manager({"w0": WIDE, "a": FIRST, "w1": WIDE})

    batch = _next_batch(sched, manager, pre_existing_batch_size=1, capture_group_of="w0")

    assert _rows(batch) == ["w0", "w1"]
    assert _rows(sched.backlog[(NODE, WALK)]) == ["a"]


def test_without_the_knob_any_mix_of_windows_batches():
    """The default on every platform: the rows run as one padded replay."""
    sched = _scheduler(_codec_engine(split=False))
    batch = _next_batch(sched, _manager({"w0": WIDE, "a": FIRST, "s0": SECOND}))
    assert _rows(batch) == ["w0", "a", "s0"]
    assert sched.backlog == {}


def test_the_knob_only_names_the_codec_walk():
    codec = CodecSubmodule(_FakeCodecDecoder(4), _tiny_model_config(), split_windows=True)
    assert codec.split_batches_by_capture_key(WALK)
    assert not codec.split_batches_by_capture_key("talker_decode")
    assert codec.capture_group_order(WALK) == (FIRST, SECOND, WIDE)
    assert codec.capture_group_order("talker_decode") == ()


def test_the_group_order_is_asked_for_once():
    engine = _codec_engine()
    asked = []
    real = engine.capture_group_order
    engine.capture_group_order = lambda n, w: (asked.append((n, w)), real(n, w))[1]
    sched = _scheduler(engine)
    for _ in range(3):
        _next_batch(sched, _manager({"a": FIRST}))
    assert asked == [(NODE, WALK)]


@pytest.fixture
def snac():
    return SNACDecoderSubmodule(_FakeCodecDecoder(4), OrpheusModelConfig())


def test_orpheus_keeps_its_windows_apart_oldest_first(snac):
    """Orpheus splits by the same hook and declares no order: its groups stay
    in arrival order, narrow windows included."""
    assert snac.capture_group_order("snac_chunk") == ()
    frames = snac.config.tokens_per_frame
    engine = _engine(snac, walk="snac_chunk")
    sched = _scheduler(engine)
    manager = _Manager(["wide", "narrow", "wide2"], node=NODE, walk="snac_chunk")
    manager.stream_chunks = {
        rid: {"new_token": StreamChunkInfo(start_offset=0, context_items=0, num_items=n * frames, is_final=False)}
        for rid, n in (("wide", 8), ("narrow", 2), ("wide2", 8))
    }

    assert _rows(_next_batch(sched, manager)) == ["wide", "wide2"]
    assert _rows(sched.backlog[(NODE, "snac_chunk")]) == ["narrow"]

