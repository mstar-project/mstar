"""Rolling KV store: retention, compaction and the pool's regions, on CPU.

Every token carries its stream index (plus a per-request base) as its key and
value, so what a step attends can be checked against a brute-force reading of
the retention rule: the stream's first ``sink`` tokens and its last ``window``,
the step's included.
"""

import logging
import random
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
from mstar.engine.resources.kv.rolling import manager as manager_module
from mstar.engine.resources.kv.rolling.manager import RollingKVManager, row_layout
from mstar.engine.resources.step import (
    ASKS,
    FULL_ADMIT_QUEUED,
    AdmitRuntimeError,
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


def _open(store: RollingKVManager, rid: int, tokens: int) -> None:
    """Open a request as the engine does: it declares its tokens, and the admission
    queue sets that room aside once it fits."""
    store.ingest_request(rid, RollingKVReqConfig(tokens=tokens))
    assert store.reserve(rid, "dit", "rollout", None) is None


def _run(store, rid, spans, sink, window, passes=1, base=0):
    """Open ``rid`` unless the test already has, then drive one stream through
    admit -> plan -> write (``passes`` times) -> commit, tagging every token with
    ``base`` plus its stream index; yields what each step attended."""
    if rid not in store._requests and rid not in store._reserved:
        _open(store, rid, sink + window)
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


def test_a_request_gets_no_room_until_a_region_frees():
    store = _store(18)
    list(_run(store, 1, [4], 6, 12))
    store.ingest_request(2, RollingKVReqConfig(tokens=18))
    assert store.reserve(2, "dit", "rollout", None) == 0  # the room it didn't fit against
    assert not store.grew_since(0)
    store.remove_request(1)
    assert store.grew_since(0) and store.reserve(2, "dit", "rollout", None) is None
    second, ctx2 = _step(2, 0, 4, 6, 12)
    assert store.admit(second, ctx2).ok and store.num_free_tokens() == 0


def test_a_request_that_declared_nothing_fails_at_its_first_step():
    store = _store(30)
    assert store.ask(1, "dit", "rollout") is None  # nothing to wait for
    step, ctx = _step(1, 0, 4, 6, 12)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal
    assert "RollingKVReqConfig" in outcome.reason.message and store.num_free_tokens() == 30


def test_rearranging_joins_the_gaps_and_keeps_every_requests_tokens(caplog):
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
    with caplog.at_level(logging.INFO, logger=manager_module.__name__):
        total, keys, _ = next(newcomer)
    assert store._requests[2].offset == 0 and store._requests[4].offset == 10 and store._free == []
    # serve_concurrent.py counts this line in the server log
    assert [r.getMessage() for r in caplog.records] == [
        "rolling KV rolling_kv: rearranged the pool, moving 1 regions (10 tokens a layer)"]
    assert keys == _expected(total, 2, 18, base=4000)
    for total, keys, _ in streams[2]:  # request 2 carries on from its moved region
        assert keys == _expected(total, 2, 8, base=2000)
    for total, keys, _ in newcomer:
        assert keys == _expected(total, 2, 18, base=4000)


@pytest.mark.parametrize("gap", [1, 2, 3, 4, 7, 20])
def test_a_region_moved_left_keeps_every_token_however_short_the_gap(monkeypatch, gap):
    """A gap under a quarter of the staging block is crossed through it, block by
    block; a longer one in place. Either way every token lands where it should."""
    monkeypatch.setattr(manager_module, "STAGE_TOKENS", 16)
    store = _store(48)
    pool = store._pool
    pool.copy_(torch.arange(pool.numel(), dtype=pool.dtype).view_as(pool))
    store._stage.fill_(-7)
    expect = pool[:, :, gap:gap + 23].clone()
    store._move_left(gap, 0, 23)
    assert torch.equal(pool[:, :, :23], expect)
    staged = bool((store._stage != -7).any())
    assert staged == (4 * gap < 16)


def test_regions_that_slide_across_a_one_token_gap_keep_their_streams(monkeypatch):
    """A one-token request leaves a one-token gap; the regions after it cross it
    through the staging block when a newcomer needs that gap and the tail together."""
    monkeypatch.setattr(manager_module, "STAGE_TOKENS", 8)
    store = _store(30)
    store._stage.fill_(-7)
    tiny = _run(store, 1, [1] * 3, 0, 1, base=1000)
    next(tiny)
    streams = {rid: _run(store, rid, [4] * 6, 2, 8, base=1000 * rid) for rid in (2, 3)}
    for _ in range(3):
        for stream in streams.values():
            next(stream)
    store.remove_request(1)
    assert store._free == [(0, 1), (21, 9)]
    newcomer = _run(store, 4, [4] * 3, 2, 8, base=4000)  # 10 tokens: neither gap holds it
    total, keys, _ = next(newcomer)
    assert [store._requests[r].offset for r in (2, 3, 4)] == [0, 10, 20] and store._free == []
    assert (store._stage != -7).any()  # the 10-token regions crossed the 1-token gap staged
    assert keys == _expected(total, 2, 8, base=4000)
    for rid, stream in streams.items():
        for total, keys, _ in stream:
            assert keys == _expected(total, 2, 8, base=1000 * rid)


@pytest.mark.parametrize("stage", [2, 8])
def test_two_regions_with_a_gap_between_them_both_slide_and_keep_their_streams(monkeypatch, stage):
    """[gap 1][A][gap 3][B][tail]: a newcomer that fits no gap makes A slide by 1 and
    B by 4, onto tokens A held, while both are mid-stream. With a 2-token staging
    block both move in place; with 8, A's short move is staged."""
    monkeypatch.setattr(manager_module, "STAGE_TOKENS", stage)
    store = _store(36)
    store._stage.fill_(-7)
    one = _run(store, 1, [1] * 3, 0, 1, base=1000)
    a = _run(store, 2, [4] * 6, 2, 8, base=2000)
    three = _run(store, 3, [3] * 3, 0, 3, base=3000)
    b = _run(store, 4, [4] * 6, 2, 8, base=4000)
    for stream in (one, a, three, b):
        next(stream)  # opens them in this order: offsets 0, 1, 11, 14
    for _ in range(2):  # A and B hold 12 tokens each: the sink and a full, shifted window
        next(a), next(b)
    store.remove_request(1)
    store.remove_request(3)
    assert store._free == [(0, 1), (11, 3), (24, 12)]
    newcomer = _run(store, 5, [4] * 4, 2, 12, base=5000)  # 14 tokens: no gap holds it
    total, keys, _ = next(newcomer)
    assert [store._requests[r].offset for r in (2, 4, 5)] == [0, 10, 20] and store._free == [(34, 2)]
    assert bool((store._stage != -7).any()) == (stage == 8)
    assert keys == _expected(total, 2, 12, base=5000)
    for rid, stream in ((2, a), (4, b)):
        for total, keys, values in stream:
            assert keys == values == _expected(total, 2, 8, base=1000 * rid)
    for total, keys, _ in newcomer:
        assert keys == _expected(total, 2, 12, base=5000)


class _NoSelfOverlap:
    """Stands in for the pool's word view in ``_move_left``: a copy within the pool
    must read and write disjoint tokens, since a GPU copies them in parallel. (On
    CPU, an overlapping copy to the left runs front to back and comes out right
    anyway, so the streams alone would not show it.)"""

    def __init__(self, words: torch.Tensor, counts: dict):
        self.words, self.counts = words, counts

    def _tokens(self, t: torch.Tensor) -> range | None:
        if t.untyped_storage().data_ptr() != self.words.untyped_storage().data_ptr():
            return None  # the staging block
        start = (t.storage_offset() - self.words.storage_offset()) // self.words.stride(2)
        return range(start, start + t.shape[2])

    def __getitem__(self, key):
        return self.words[key]

    def __setitem__(self, key, value):
        src, dst = self._tokens(value), self._tokens(self.words[key])
        if src is None:
            self.counts["staged"] += 1
        else:
            assert src.stop <= dst.start or dst.stop <= src.start, f"a copy reads and writes {src} -> {dst}"
        self.words[key] = value


def _random_workload(rng: random.Random, counts: dict) -> None:
    """Requests of random sizes open, step (up to three in one step) and end, in
    random order; every step's rows must attend exactly their retention."""
    store = _store(rng.randint(12, 64))
    store._words = _NoSelfOverlap(store._words, counts)
    rearrange = store._rearrange

    def counted_rearrange():
        counts["rearranges"] += 1
        rearrange()

    store._rearrange = counted_rearrange
    reqs, waiting, running = {}, [], []
    for _ in range(rng.randint(30, 150)):
        r = rng.random()
        if r < 0.25:
            rid = len(reqs) + 1
            sink = rng.randint(0, 4)
            reqs[rid] = SimpleNamespace(sink=sink, window=rng.randint(1, min(15, store.config.pool_tokens - sink)),
                                        written=0)
            store.ingest_request(rid, RollingKVReqConfig(tokens=sink + reqs[rid].window))
            waiting.append(rid)
        elif r < 0.35 and (running or waiting):
            gone = rng.choice(running + waiting)
            store.remove_request(gone)
            (running if gone in running else waiting).remove(gone)
        while waiting and store.reserve(waiting[0], "dit", "rollout", None) is None:  # arrival order
            running.append(waiting.pop(0))
        if not running:
            continue
        rows = rng.sample(running, rng.randint(1, min(3, len(running))))
        spans = [rng.randint(1, reqs[rid].window) for rid in rows]
        step = RollingKVStep(
            segments=tuple(Segment(rid, "main", span) for rid, span in zip(rows, spans, strict=True)),
            positions={rid: RollingPosition(reqs[rid].written, reqs[rid].sink, reqs[rid].window) for rid in rows})
        ctx = StepContext(request_ids=tuple(rows), graph_walk="rollout", slot=0, capture=False)
        assert store.admit(step, ctx).ok
        store.plan(step, ctx)
        tags = torch.full((len(rows), max(spans), HEADS, DIM), -1.0)
        for i, (rid, span) in enumerate(zip(rows, spans, strict=True)):
            tags[i, :span, 0, 0] = 1000 * rid + torch.arange(reqs[rid].written, reqs[rid].written + span)
        for layer in range(LAYERS):
            store.write(layer, tags, tags)
        store.commit(step, ctx)
        for i, (rid, span) in enumerate(zip(rows, spans, strict=True)):
            req = reqs[rid]
            req.written += span
            for layer in range(LAYERS):
                k, v = store.window(layer, row=i)
                assert k.flatten().tolist() == v.flatten().tolist() == \
                    _expected(req.written, req.sink, req.window, base=1000 * rid)


@pytest.mark.parametrize("seed", range(4))
def test_random_requests_keep_their_streams_through_every_rearrange(monkeypatch, seed):
    """The pool fragments, and rearranges slide several regions by different
    distances across gaps of every size, staged and in place."""
    rng = random.Random(seed)
    counts = {"rearranges": 0, "staged": 0}
    for _ in range(50):
        monkeypatch.setattr(manager_module, "STAGE_TOKENS", rng.choice([2, 4, 8, 16]))
        _random_workload(rng, counts)
    assert counts["rearranges"] and counts["staged"], counts  # both paths ran


def test_a_refused_step_gives_back_its_region_and_sets_the_room_aside_again():
    store = _store(36)
    _open(store, 1, 18)
    step, ctx = _step(1, 0, 4, 6, 12)
    assert store.admit(step, ctx).ok and store.num_free_tokens() == 18 and store._kept == 0
    store.rollback_admit(step, ctx)
    assert store._free == [(0, 36)] and store._reserved == {1: 18} and store._kept == 18
    assert store.admit(step, ctx).ok and store._kept == 0
    # a region held from an earlier step stays held
    list(_run(store, 3, [4], 6, 12))
    held, ctx3 = _step(3, 4, 4, 6, 12)
    assert store.admit(held, ctx3).ok
    store.rollback_admit(held, ctx3)
    assert store.num_free_tokens() == 0 and store._kept == 0


def test_a_window_that_cannot_hold_the_step_fails_the_request():
    store = _store(18)
    _open(store, 1, 3)
    step, ctx = _step(1, 0, 4, 0, 3)
    outcome = store.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal
    assert "window must hold the step" in outcome.reason.message


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
    _open(store, 1, 18)
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
    for bad in (0, -1, True, "64", 64.0):
        with pytest.raises(ValueError, match="pool_tokens must be an integer >= 1"):
            spec.apply_yaml_overrides(pool_tokens=bad)
    with pytest.raises(TypeError):
        spec.apply_yaml_overrides(max_slots=2)  # the store has no slots


# Admission: a request that declares its tokens when it opens waits in the engine's queue


def _declared(store: RollingKVManager, rid: int, tokens: int) -> None:
    store.ingest_request(rid, RollingKVReqConfig(tokens=tokens))


def test_ask_queues_only_a_declared_request_that_holds_nothing_yet():
    store = _store(30)
    assert store.ask(1, "dit", "rollout") is None  # declared nothing: no room to wait for
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


def test_room_set_aside_for_a_request_is_not_given_to_another():
    store = _store(30)
    _open(store, 1, 18)
    _declared(store, 2, 13)
    assert store.reserve(2, "dit", "rollout", None) == 12  # 30 free, 18 of them set aside
    assert not store.grew_since(12)
    first, ctx1 = _step(1, 0, 4, 6, 12)
    assert store.admit(first, ctx1).ok and store._kept == 0 and store.num_free_tokens() == 12
    assert not store.grew_since(12)


def test_removing_a_request_drops_what_it_declared_and_reserved():
    store = _store(30)
    _open(store, 1, 18)
    assert not store.grew_since(12)
    store.remove_request(1)
    assert store._wants == {} and store._reserved == {} and store._kept == 0
    assert store.grew_since(12)


def test_a_first_step_that_keeps_other_than_it_declared_fails_the_request():
    store = _store(30)
    _open(store, 1, 18)
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
