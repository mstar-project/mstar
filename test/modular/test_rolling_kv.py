"""Rolling KV store: retention, compaction and the pool's regions, on CPU.

Every token carries its stream index (plus a per-request base) as its key and
value, so what a step attends can be checked against a brute-force reading of
the retention rule: the stream's first ``sink`` tokens and its last ``window``,
the step's included.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.admission import Admission
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.rolling import (
    RollingKVConfig,
    RollingKVReqConfig,
    RollingKVSpec,
    RollingKVStep,
    RollingPosition,
)
from mstar.engine.resources.kv.rolling.manager import RollingKVManager, row_layout
from mstar.engine.resources.step import (
    ASKS,
    FULL_ADMIT_QUEUED,
    AdmitRuntimeError,
    AllocationFailed,
    Segment,
    StepContext,
)

LAYERS, HEADS, DIM = 2, 1, 1


def _store(pool_tokens: int) -> RollingKVManager:
    config = RollingKVConfig(num_layers=LAYERS, num_heads=HEADS, head_dim=DIM, pool_tokens=pool_tokens,
                             dtype=torch.float32)
    return RollingKVManager(config, torch.device("cpu"))


def _step(rid: int, written: int, span: int, sink: int, window: int):
    step = RollingKVStep(segments=(Segment(rid, "main", span),),
                         positions={rid: RollingPosition(written, sink, window)})
    return step, StepContext(request_ids=(rid,), graph_walk="rollout", slot=0, capture=False)


def _expected(total: int, sink: int, window: int, base: int = 0) -> list[int]:
    keep = range(total) if total <= sink + window else [*range(sink), *range(total - window, total)]
    return [base + i for i in keep]


def _run(store, rid, spans, sink, window, passes=1, base=0):
    """Drive one stream through admit -> plan -> write (``passes`` times) -> commit,
    tagging every token with ``base`` plus its stream index; yields what each step
    attended."""
    written = 0
    for span in spans:
        step, ctx = _step(rid, written, span, sink, window)
        assert store.admit(step, ctx).ok
        store.plan(step, ctx)
        tags = base + torch.arange(written, written + span, dtype=torch.float32).view(1, span, 1, 1)
        for p in range(passes):
            # a denoise pass rewrites the step's tokens; only the last write must stay
            junk = (passes - 1 - p) * 0.5
            for layer in range(LAYERS):
                store.write(layer, tags + junk, tags + junk)
        store.commit(step, ctx)
        k, v = store.window(LAYERS - 1)
        written += span
        yield written, k.flatten().tolist(), v.flatten().tolist()


@pytest.mark.parametrize("sink,window,span", [(6, 12, 4), (18, 36, 12), (0, 8, 4), (3, 5, 5), (2, 7, 3), (5, 40, 4)])
def test_a_step_attends_the_sink_and_the_latest_window_in_stream_order(sink, window, span):
    store = _store(sink + window)
    for total, keys, values in _run(store, 1, [span] * 12, sink, window, passes=5):
        assert keys == values == _expected(total, sink, window)


def test_lingbot_frames_keep_six_sink_frames_and_the_latest_twelve():
    """LingBot's retention in frames of 3 tokens: 6 sink frames, 12 more, 4 per chunk."""
    fs = 3
    seen = [len(keys) // fs for _, keys, _ in _run(_store(18 * fs), 1, [4 * fs] * 8, 6 * fs, 12 * fs)]
    assert seen == [4, 8, 12, 16, 18, 18, 18, 18]
    *_, (total, keys, _) = _run(_store(18 * fs), 1, [4 * fs] * 8, 6 * fs, 12 * fs)
    assert [int(k) // fs for k in keys[::fs]] == list(range(6)) + list(range(20, 32))


def test_varying_spans_stay_contiguous_and_in_order():
    for total, keys, _ in _run(_store(30), 1, [1, 7, 3, 9, 2, 8, 8, 1, 5], sink=4, window=26):
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
    assert store.window(0)[0].flatten().tolist() == _expected(20, 6, 12)


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


def test_the_pool_is_reserved_once_and_requests_take_exact_regions_first_fit():
    store = _store(40)
    assert store._pool.shape == (LAYERS, 2, 40, HEADS, DIM)
    assert store.config.token_bytes(40, torch.float32) == store._pool.nbytes
    list(_run(store, 1, [4], sink=6, window=12))  # 18 tokens
    list(_run(store, 2, [3], sink=2, window=5))   # 7 tokens
    assert {rid: (r.offset, r.capacity) for rid, r in store._requests.items()} == {1: (0, 18), 2: (18, 7)}
    assert store.num_free_tokens() == 15
    store.remove_request(1)
    assert store._free == [(0, 18), (25, 15)]
    list(_run(store, 3, [2], sink=2, window=4))   # 6 tokens: first fit, the front gap
    assert store._requests[3].offset == 0 and store._free == [(6, 12), (25, 15)]
    store.remove_request(2)
    assert store._free == [(6, 34)]  # merged with both neighbours


def test_a_full_pool_refuses_with_an_allocation_failure_until_a_region_frees():
    store = _store(18)
    first, ctx1 = _step(1, 0, 4, 6, 12)
    assert store.admit(first, ctx1).ok
    second, ctx2 = _step(2, 0, 4, 6, 12)
    outcome = store.admit(second, ctx2)
    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed) and not outcome.reason.terminal
    store.remove_request(1)
    assert store.admit(second, ctx2).ok and store.num_free_tokens() == 0


def test_rearranging_joins_the_gaps_and_keeps_every_requests_tokens():
    """Free tokens split across two gaps: the regions slide left so a request that
    needs both fits, and the requests already there attend what they did."""
    store = _store(30)
    streams = {rid: _run(store, rid, [4] * 6, 2, 8, base=1000 * rid) for rid in (1, 2, 3)}
    for _ in range(3):  # 12 tokens each: every region holds its sink and a full window
        for stream in streams.values():
            next(stream)
    store.remove_request(1)
    store.remove_request(3)
    assert store._free == [(0, 10), (20, 10)]
    newcomer = _run(store, 4, [4] * 4, 2, 18, base=4000)  # 20 tokens: neither gap holds it
    total, keys, _ = next(newcomer)
    assert store._requests[2].offset == 0 and store._requests[4].offset == 10 and store._free == []
    assert keys == _expected(total, 2, 18, base=4000)
    for total, keys, _ in streams[2]:  # request 2 carries on from its moved region
        assert keys == _expected(total, 2, 8, base=2000)
    for total, keys, _ in newcomer:
        assert keys == _expected(total, 2, 18, base=4000)


def test_a_refused_step_gives_back_the_region_its_admit_took():
    store = _store(36)
    step, ctx = _step(1, 0, 4, 6, 12)
    assert store.admit(step, ctx).ok and store.num_free_tokens() == 18
    store.rollback_admit(step, ctx)
    assert store.num_free_tokens() == 36 and store._free == [(0, 36)]
    # a region held from an earlier step stays held
    list(_run(store, 3, [4], 6, 12))
    held, ctx3 = _step(3, 4, 4, 6, 12)
    assert store.admit(held, ctx3).ok
    store.rollback_admit(held, ctx3)
    assert store.num_free_tokens() == 18


@pytest.mark.parametrize("sink,window,span", [(6, 13, 4), (0, 3, 4)])
def test_a_window_the_pool_cannot_hold_fails_the_request(sink, window, span):
    store = _store(18)
    step, ctx = _step(1, 0, span, sink, window)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal
    assert "pool_tokens" in outcome.reason.message


def test_a_request_must_keep_the_window_it_started_with():
    store = _store(64)
    list(_run(store, 1, [4], sink=6, window=12))
    step, ctx = _step(1, 4, 4, 6, 16)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)


def test_a_position_that_disagrees_with_the_commits_fails_the_request():
    store = _store(18)
    list(_run(store, 1, [4, 4], 6, 12))
    step, ctx = _step(1, 4, 4, 6, 12)  # 8 were committed
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)


def test_interleaved_requests_keep_their_own_streams():
    store = _store(36)
    streams = {1: _run(store, 1, [4] * 7, 6, 12, base=1000), 2: _run(store, 2, [3] * 7, 6, 12, base=2000)}
    for _ in range(7):
        for rid, stream in streams.items():
            total, keys, _ = next(stream)
            assert keys == _expected(total, 6, 12, base=1000 * rid), rid


def test_a_removed_request_starts_over():
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
                         config=RollingKVConfig(num_layers=1, num_heads=1, head_dim=1, pool_tokens=4))
    assert not RollingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu"))).can_capture()
    tp2 = SimpleNamespace(world_size=2)
    with pytest.raises(NotImplementedError, match="shard"):
        RollingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=tp2))


def test_yaml_overrides_size_the_pool_and_refuse_nonsense():
    spec = RollingKVSpec(resource_key="kv", nodes={"dit"},
                         config=RollingKVConfig(num_layers=1, num_heads=1, head_dim=1, pool_tokens=4))
    spec.apply_yaml_overrides(pool_tokens=64)
    assert spec.config.pool_tokens == 64
    with pytest.raises(ValueError, match="pool_tokens"):
        spec.apply_yaml_overrides(pool_tokens=0)
    with pytest.raises(TypeError):
        spec.apply_yaml_overrides(max_slots=2)  # the store has no slots


# Admission: a request that declares its tokens when it opens waits in the engine's queue


def _declared(store: RollingKVManager, rid: int, tokens: int) -> None:
    store.ingest_request(rid, RollingKVReqConfig(tokens=tokens))


def test_ask_queues_only_a_declared_request_that_holds_nothing_yet():
    store = _store(30)
    assert store.ask(1, "dit", "rollout") is None  # declared nothing: placed at its first step
    _declared(store, 2, 18)
    assert store.ask(2, "dit", "rollout") is ASKS
    assert store.reserve(2, "dit", "rollout", None) is None and store._kept == 18
    assert store.ask(2, "dit", "rollout") is None  # reserved
    list(_run(store, 2, [4], 6, 12))
    assert store._kept == 0 and store.ask(2, "dit", "rollout") is None  # placed
    _declared(store, 3, 31)
    outcome = store.ask(3, "dit", "rollout")
    assert not outcome.ok and outcome.reason.terminal and "pool_tokens" in outcome.reason.message


def test_requests_wait_in_arrival_order_so_a_large_one_is_not_overtaken():
    """A large request that doesn't fit holds back a later small one that would,
    and both start once the pool frees, the large one first."""
    store = _store(30)
    admission = Admission({"kv": store}, ["kv"])
    for rid, tokens in [(1, 18), (2, 18), (3, 6)]:
        _declared(store, rid, tokens)
    assert admission.admit(1, "dit", "rollout", None, ["kv"]) is None
    list(_run(store, 1, [4], 6, 12))  # 12 of 30 free
    assert admission.admit(2, "dit", "rollout", None, ["kv"]) is FULL_ADMIT_QUEUED  # 18 doesn't fit
    assert admission.admit(3, "dit", "rollout", None, ["kv"]) is FULL_ADMIT_QUEUED  # 6 would, but 2 is first
    assert admission.admit_waiting(lambda rid, node: None) == []
    store.remove_request(1)
    admission.remove_request(1)
    assert admission.admit_waiting(lambda rid, node: None) == [("dit", 2), ("dit", 3)]
    assert store._reserved == {2: 18, 3: 6} and store._kept == 24


def test_room_set_aside_for_a_waiting_request_is_not_taken_by_another():
    store = _store(30)
    _declared(store, 1, 18)
    assert store.reserve(1, "dit", "rollout", None) is None
    step, ctx = _step(2, 0, 4, 6, 12)  # declared nothing, needs 18 of the 12 not set aside
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed) and not outcome.reason.terminal
    assert store.reserve(3, "dit", "rollout", None) is None  # declared nothing either: takes nothing
    _declared(store, 4, 13)
    assert store.reserve(4, "dit", "rollout", None) == 12  # the room it didn't fit against
    assert not store.grew_since(12)
    first, ctx1 = _step(1, 0, 4, 6, 12)
    assert store.admit(first, ctx1).ok and store._kept == 0 and store.num_free_tokens() == 12


def test_a_reserved_request_is_placed_even_when_the_free_room_is_split():
    store = _store(30)
    streams = {rid: _run(store, rid, [4] * 3, 2, 8, base=1000 * rid) for rid in (1, 2, 3)}
    for stream in streams.values():
        next(stream)
    store.remove_request(1)
    store.remove_request(3)
    assert store._free == [(0, 10), (20, 10)]
    _declared(store, 4, 20)
    assert store.reserve(4, "dit", "rollout", None) is None
    total, keys, _ = next(_run(store, 4, [4], 2, 18, base=4000))
    assert keys == _expected(total, 2, 18, base=4000) and store._free == []
    total, keys, _ = next(streams[2])
    assert keys == _expected(total, 2, 8, base=2000)


def test_a_refused_step_gives_back_its_region_and_keeps_its_reservation():
    store = _store(30)
    _declared(store, 1, 18)
    assert store.reserve(1, "dit", "rollout", None) is None
    step, ctx = _step(1, 0, 4, 6, 12)
    assert store.admit(step, ctx).ok and store._kept == 0
    store.rollback_admit(step, ctx)
    assert store._reserved == {1: 18} and store._kept == 18 and store.num_free_tokens() == 30
    assert store.admit(step, ctx).ok and store._kept == 0


def test_removing_a_request_drops_what_it_declared_and_reserved():
    store = _store(30)
    _declared(store, 1, 18)
    assert store.reserve(1, "dit", "rollout", None) is None
    assert not store.grew_since(12)
    store.remove_request(1)
    assert store._wants == {} and store._reserved == {} and store._kept == 0
    assert store.grew_since(12)


def test_a_first_step_that_keeps_other_than_it_declared_fails_the_request():
    store = _store(30)
    _declared(store, 1, 18)
    assert store.reserve(1, "dit", "rollout", None) is None
    step, ctx = _step(1, 0, 4, 6, 14)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError) and "declared" in outcome.reason.message


def test_a_request_larger_than_the_pool_is_refused_before_it_reaches_a_worker():
    spec = RollingKVSpec(resource_key="kv", nodes={"dit"},
                         config=RollingKVConfig(num_layers=1, num_heads=1, head_dim=1, pool_tokens=30))
    RollingKVReqConfig(tokens=30).validate(spec)
    with pytest.raises(ValueError, match="resources: kv: pool_tokens"):
        RollingKVReqConfig(tokens=31).validate(spec)
    with pytest.raises(ValueError, match="at least one"):
        RollingKVReqConfig(tokens=0).validate(spec)
