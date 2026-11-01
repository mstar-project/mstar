"""Rolling KV store: retention, compaction and the slot lifecycle, on CPU.

Every token carries its stream index as its key and value, so what a step
attends can be checked against a brute-force reading of the retention rule:
the stream's first ``sink`` tokens and its last ``window``, the step's included.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.rolling import RollingKVConfig, RollingKVSpec, RollingKVStep, RollingPosition
from mstar.engine.resources.kv.rolling.manager import RollingKVManager, row_layout
from mstar.engine.resources.step import AdmitRuntimeError, AllocationFailed, Segment, StepContext

LAYERS, HEADS, DIM = 2, 1, 1


def _store(max_request_tokens: int, max_slots: int = 1) -> RollingKVManager:
    config = RollingKVConfig(num_layers=LAYERS, num_heads=HEADS, head_dim=DIM, max_request_tokens=max_request_tokens,
                             max_slots=max_slots, dtype=torch.float32)
    return RollingKVManager(config, torch.device("cpu"))


def _step(rid: int, written: int, span: int, sink: int, window: int):
    step = RollingKVStep(segments=(Segment(rid, "main", span),),
                         positions={rid: RollingPosition(written, sink, window)})
    return step, StepContext(request_ids=(rid,), graph_walk="rollout", slot=0, capture=False)


def _expected(total: int, sink: int, window: int) -> list[int]:
    if total <= sink + window:
        return list(range(total))
    return list(range(sink)) + list(range(total - window, total))


def _run(store, rid, spans, sink, window, passes=1, check=None):
    """Drive one stream through admit -> plan -> write (``passes`` times) -> commit,
    tagging every token with its stream index; yields what each step attended."""
    written = 0
    for span in spans:
        step, ctx = _step(rid, written, span, sink, window)
        assert store.admit(step, ctx).ok
        store.plan(step, ctx)
        tags = torch.arange(written, written + span, dtype=torch.float32).view(1, span, 1, 1)
        for p in range(passes):
            # a denoise pass rewrites the step's tokens; only the last write must stay
            for layer in range(LAYERS):
                store.write(layer, tags + (passes - 1 - p) * 1000, tags + (passes - 1 - p) * 1000)
        store.commit(step, ctx)
        k, v = store.window(LAYERS - 1)
        written += span
        yield written, k.flatten().long().tolist(), v.flatten().long().tolist()


@pytest.mark.parametrize("sink,window,span", [(6, 12, 4), (18, 36, 12), (0, 8, 4), (3, 5, 5), (2, 7, 3), (5, 40, 4)])
def test_a_step_attends_the_sink_and_the_latest_window_in_stream_order(sink, window, span):
    store = _store(sink + window)
    for total, keys, values in _run(store, 1, [span] * 12, sink, window, passes=5):
        assert keys == values == _expected(total, sink, window)


def test_lingbot_frames_keep_six_sink_frames_and_the_latest_twelve():
    """LingBot's retention in frames of 3 tokens: 6 sink frames, 12 more, 4 per chunk."""
    fs = 3
    store = _store(18 * fs)
    seen = [len(keys) // fs for _, keys, _ in _run(store, 1, [4 * fs] * 8, 6 * fs, 12 * fs)]
    assert seen == [4, 8, 12, 16, 18, 18, 18, 18]
    *_, (total, keys, _) = _run(_store(18 * fs), 1, [4 * fs] * 8, 6 * fs, 12 * fs)
    frames = [k // fs for k in keys[::fs]]
    assert frames == list(range(6)) + list(range(20, 32))


def test_varying_spans_stay_contiguous_and_in_order():
    store = _store(30)
    spans = [1, 7, 3, 9, 2, 8, 8, 1, 5]
    for total, keys, _ in _run(store, 1, spans, sink=4, window=26):
        assert keys == _expected(total, 4, 26)


def test_planning_a_step_twice_compacts_once():
    store = _store(18)
    list(_run(store, 1, [4] * 4, sink=6, window=12))  # 16 written: the next step evicts
    step, ctx = _step(1, 16, 4, 6, 12)
    assert store.admit(step, ctx).ok
    first = store.plan(step, ctx)
    second = store.plan(step, ctx)  # a region planning the declared step again, or a retry
    assert first == second
    store.write(0, torch.arange(16, 20.0).view(1, 4, 1, 1), torch.arange(16, 20.0).view(1, 4, 1, 1))
    assert store.window(0)[0].flatten().long().tolist() == _expected(20, 6, 12)


def test_row_layout_matches_the_shift_left_eviction():
    """The reference's rule (Self-Forcing's rolling cache): evict ``span + held - capacity``
    tokens behind the sink, roll the rest left, append the chunk."""
    sink, window, span, cap = 6, 12, 4, 18
    held = 0
    for n in range(0, 40, span):
        row, shift = row_layout(RollingPosition(n, sink, window), span)
        if n + span > cap and span + held > cap:
            evicted = span + held - cap
            assert shift is not None and (shift.src, shift.dst) == (sink + evicted, sink)
            assert shift.length == held - evicted - sink
        else:
            assert shift is None
        held = min(n + span, cap)
        assert (row.write_stop, row.kept) == (held, held) and row.write_stop - row.write_start == span


def test_a_request_holds_its_own_window_and_nothing_more_until_it_is_removed():
    """No memory up front: each request's K/V are its sink plus window, allocated
    at its first step and dropped with it, as a model-owned cache was."""
    store = _store(64, max_slots=2)
    assert store._requests == {}
    list(_run(store, 1, [4], sink=6, window=12))
    list(_run(store, 2, [3], sink=2, window=5))
    assert {rid: req.kv.shape for rid, req in store._requests.items()} == {
        1: (LAYERS, 2, 18, HEADS, DIM), 2: (LAYERS, 2, 7, HEADS, DIM)}
    assert store.config.request_bytes(18, torch.float32) == store._requests[1].kv.nbytes
    store.remove_request(1)
    assert list(store._requests) == [2] and store.num_free_slots() == 1


def test_a_request_must_keep_the_window_it_started_with():
    store = _store(64)
    list(_run(store, 1, [4], sink=6, window=12))
    step, ctx = _step(1, 4, 4, 6, 16)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)


def test_out_of_device_memory_waits_beside_others_and_fails_alone(monkeypatch):
    store = _store(64, max_slots=4)

    def oom(capacity):
        raise torch.OutOfMemoryError("CUDA out of memory")

    real = store._allocate
    monkeypatch.setattr(store, "_allocate", oom)
    step, ctx = _step(1, 0, 4, 6, 12)
    alone = store.admit(step, ctx)
    assert not alone.ok and isinstance(alone.reason, AdmitRuntimeError) and alone.reason.terminal
    monkeypatch.setattr(store, "_allocate", real)
    list(_run(store, 2, [4], 6, 12))  # another request now holds its K/V
    monkeypatch.setattr(store, "_allocate", oom)
    beside = store.admit(step, ctx)
    assert not beside.ok and isinstance(beside.reason, AllocationFailed) and not beside.reason.terminal
    assert list(store._requests) == [2]


def test_a_full_pool_refuses_with_an_allocation_failure_until_a_slot_frees():
    store = _store(18, max_slots=1)
    first, ctx1 = _step(1, 0, 4, 6, 12)
    assert store.admit(first, ctx1).ok
    second, ctx2 = _step(2, 0, 4, 6, 12)
    outcome = store.admit(second, ctx2)
    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed) and not outcome.reason.terminal
    store.remove_request(1)
    assert store.admit(second, ctx2).ok and store.num_free_slots() == 0


def test_a_refused_step_gives_back_the_slot_its_admit_took():
    store = _store(18, max_slots=2)
    step, ctx = _step(1, 0, 4, 6, 12)
    assert store.admit(step, ctx).ok and store.num_free_slots() == 1
    store.rollback_admit(step, ctx)
    assert store.num_free_slots() == 2
    # a slot held from an earlier step stays held
    list(_run(store, 3, [4], 6, 12))
    held, ctx3 = _step(3, 4, 4, 6, 12)
    assert store.admit(held, ctx3).ok
    store.rollback_admit(held, ctx3)
    assert store.num_free_slots() == 1


@pytest.mark.parametrize("sink,window,span", [(6, 13, 4), (0, 3, 4)])
def test_a_window_the_slot_cannot_hold_fails_the_request(sink, window, span):
    store = _store(18)
    step, ctx = _step(1, 0, span, sink, window)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal


def test_a_position_that_disagrees_with_the_commits_fails_the_request():
    store = _store(18)
    list(_run(store, 1, [4, 4], 6, 12))
    step, ctx = _step(1, 4, 4, 6, 12)  # 8 were committed
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)


def test_interleaved_requests_keep_their_own_streams():
    store = _store(18, max_slots=2)
    streams = {1: _run(store, 1, [4] * 7, 6, 12), 2: _run(store, 2, [3] * 7, 6, 12)}
    for _ in range(7):
        for rid, stream in streams.items():
            total, keys, _ = next(stream)
            assert keys == _expected(total, 6, 12), rid


def test_a_removed_request_starts_over_in_its_next_slot():
    store = _store(18)
    list(_run(store, 1, [4] * 6, 6, 12))
    store.remove_request(1)
    total, keys, _ = next(_run(store, 1, [4], 6, 12))
    assert (total, keys) == (4, [0, 1, 2, 3])


def test_writing_more_rows_than_the_step_has_is_refused():
    store = _store(18)
    step, ctx = _step(1, 0, 4, 6, 12)
    store.admit(step, ctx)
    store.plan(step, ctx)
    with pytest.raises(ValueError, match="rows"):
        store.write(0, torch.zeros(2, 4, 1, 1), torch.zeros(2, 4, 1, 1))


def test_the_store_runs_eager_and_does_not_shard_yet():
    spec = RollingKVSpec(resource_key="kv", nodes={"dit"},
                         config=RollingKVConfig(num_layers=1, num_heads=1, head_dim=1, max_request_tokens=4))
    assert not RollingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu"))).can_capture()
    tp2 = SimpleNamespace(world_size=2)
    with pytest.raises(NotImplementedError, match="shard"):
        RollingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=tp2))


def test_yaml_overrides_size_the_pool_and_refuse_nonsense():
    spec = RollingKVSpec(resource_key="kv", nodes={"dit"},
                         config=RollingKVConfig(num_layers=1, num_heads=1, head_dim=1, max_request_tokens=4))
    spec.apply_yaml_overrides(max_slots=3, max_request_tokens=64)
    assert (spec.config.max_slots, spec.config.max_request_tokens) == (3, 64)
    with pytest.raises(ValueError, match="max_slots"):
        spec.apply_yaml_overrides(max_slots=0)
