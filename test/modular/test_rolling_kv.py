"""Rolling KV window over a request pool: retention, compaction, and K/V that
keep their streams while the pool rearranges, on CPU.

Every token carries its stream index (plus a per-request base) as its key and
value, so what a step attends can be checked against a brute-force reading of
the retention rule: the stream's first ``sink`` tokens and its last ``window``,
the step's included.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.rolling import RollingKVConfig, RollingKVSpec, RollingKVStep, RollingPosition
from mstar.engine.resources.kv.rolling import manager as manager_module
from mstar.engine.resources.kv.rolling.manager import RollingKVManager, row_layout
from mstar.engine.resources.request_pool import RequestPoolConfig, RequestPoolReqConfig, RequestPoolStep
from mstar.engine.resources.request_pool import manager as pool_module
from mstar.engine.resources.request_pool.manager import RequestPool
from mstar.engine.resources.step import AdmitRuntimeError, Segment, StepContext

# One token of a request's K/V, keys and values of every layer, is 2 * 2 * 16 * 4 =
# 256 bytes: one aligned unit of the pool, so pool offsets read in tokens.
LAYERS, HEADS, DIM = 2, 1, 16
TOKEN = LAYERS * 2 * HEADS * DIM * 4


class Pair:
    """A request pool and the rolling window over it, stepped as the engine steps
    them: the pool admits and plans first (the window depends on it)."""

    def __init__(self, pool_tokens: int, inference_tensors: bool = False):
        config = RequestPoolConfig(pool_bytes=pool_tokens * TOKEN, inference_tensors=inference_tensors)
        self.pool = RequestPool(config, torch.device("cpu"), name="pool")
        config = RollingKVConfig(num_layers=LAYERS, num_heads=HEADS, head_dim=DIM, pool="pool", dtype=torch.float32)
        self.store = RollingKVManager(config, torch.device("cpu"), name="kv")
        self.store.bind_dependencies({"pool": self.pool})

    def open(self, rid: int, tokens: int, extra: dict | None = None) -> None:
        """Open a request as the engine does: it declares its K/V buffer (and any
        ``extra`` buffers), and the admission queue sets that room aside."""
        buffers = {"kv": self.store.config.buffer(tokens), **(extra or {})}
        self.pool.ingest_request(rid, RequestPoolReqConfig.of(buffers))
        assert self.pool.reserve(rid, "dit", "rollout", None) is None

    def admit(self, rid, pos: RollingPosition, span: int):
        segs = (Segment(rid, "main", span),)
        steps = RequestPoolStep(segments=segs), RollingKVStep(segments=segs, positions={rid: pos})
        ctx = StepContext(request_ids=(rid,), graph_walk="rollout", slot=0, capture=False)
        outcome = self.pool.admit(steps[0], ctx)
        if outcome.ok:
            outcome = self.store.admit(steps[1], ctx)
        return outcome, steps, ctx

    def plan(self, steps, ctx):
        self.pool.plan(steps[0], ctx)
        return self.store.plan(steps[1], ctx)

    def offset(self, rid) -> int:
        return self.pool._regions[rid].offset // TOKEN

    def remove(self, rid) -> None:
        self.store.remove_request(rid)
        self.pool.remove_request(rid)


def _tags(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, HEADS, DIM)


def _expected(total: int, sink: int, window: int, base: int = 0) -> list[int]:
    keep = range(total) if total <= sink + window else [*range(sink), *range(total - window, total)]
    return [base + i for i in keep]


def _run(pair: Pair, rid, spans, sink, window, passes=1, base=0):
    """Open ``rid`` unless the test already has, then drive one stream through
    admit -> plan -> write (``passes`` times) -> commit, tagging every token with
    ``base`` plus its stream index; yields what each step attended."""
    if rid not in pair.pool._regions and rid not in pair.pool._reserved:
        pair.open(rid, sink + window)
    written = 0
    for span in spans:
        outcome, steps, ctx = pair.admit(rid, RollingPosition(written, sink, window), span)
        assert outcome.ok, outcome.reason
        pair.plan(steps, ctx)
        tags = base + torch.arange(written, written + span, dtype=torch.float32)
        for p in range(passes):
            # a denoise pass rewrites the step's tokens; only the last write must stay
            junk = (passes - 1 - p) * 0.5
            for layer in range(LAYERS):
                pair.store.write(layer, _tags(tags + junk), _tags(tags + junk))
        pair.store.commit(steps[1], ctx)
        k, v = pair.store.window(LAYERS - 1)
        written += span
        yield written, k[0, :, 0, 0].tolist(), v[0, :, 0, 0].tolist()


@pytest.mark.parametrize("inference_tensors", [False, True])
@pytest.mark.parametrize("sink,window,span", [(6, 12, 4), (18, 36, 12), (0, 8, 4), (3, 5, 5), (2, 7, 3), (5, 40, 4)])
def test_a_step_attends_the_sink_and_the_latest_window_in_stream_order(sink, window, span, inference_tensors):
    """Over a pool of inference tensors too: the window writes and shifts in inference mode."""
    pair = Pair(sink + window, inference_tensors)
    for total, keys, values in _run(pair, 1, [span] * 12, sink, window, passes=5):
        assert keys == values == _expected(total, sink, window)


def test_lingbot_frames_keep_six_sink_frames_and_the_latest_twelve():
    """LingBot's retention in frames of 3 tokens: 6 sink frames, 12 more, 4 per chunk."""
    fs = 3
    seen = [len(keys) // fs for _, keys, _ in _run(Pair(18 * fs), 1, [4 * fs] * 8, 6 * fs, 12 * fs)]
    assert seen == [4, 8, 12, 16, 18, 18, 18, 18]
    *_, (total, keys, _) = _run(Pair(18 * fs), 1, [4 * fs] * 8, 6 * fs, 12 * fs)
    assert [int(k) // fs for k in keys[::fs]] == list(range(6)) + list(range(20, 32))


def test_varying_spans_stay_contiguous_and_in_order():
    for total, keys, _ in _run(Pair(30), 1, [1, 7, 3, 9, 2, 8, 8, 1, 5], sink=4, window=26):
        assert keys == _expected(total, 4, 26)


def test_planning_a_step_twice_compacts_once():
    pair = Pair(18)
    list(_run(pair, 1, [4] * 4, sink=6, window=12))  # 16 written: the next step evicts
    outcome, steps, ctx = pair.admit(1, RollingPosition(16, 6, 12), 4)
    assert outcome.ok
    first = pair.plan(steps, ctx).rows
    second = pair.plan(steps, ctx).rows  # the declared step planned again, or a retry
    assert first == second
    pair.store.write(0, _tags(range(16, 20)), _tags(range(16, 20)))
    assert pair.store.window(0)[0][0, :, 0, 0].tolist() == _expected(20, 6, 12)


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


@pytest.mark.parametrize("gap", [1, 2, 3, 4, 7, 20])
def test_a_shift_keeps_every_token_however_short_the_distance(monkeypatch, gap):
    """A shift under a quarter of the staging block goes through it, block by block;
    a longer one in place. Either way every token lands where it should."""
    monkeypatch.setattr(manager_module, "STAGE_TOKENS", 16)
    pair = Pair(48)
    pair.open(1, 48)
    pair.admit(1, RollingPosition(0, 0, 48), 1)
    kv = pair.pool.buffer(1, "kv")
    kv.copy_(torch.arange(kv.numel(), dtype=kv.dtype).view_as(kv))
    pair.store._stage.fill_(-7)
    expect = kv[:, :, gap:gap + 23].clone()
    pair.store._move_left(kv, gap, 0, 23)
    assert torch.equal(kv[:, :, :23], expect)
    assert bool((pair.store._stage != -7).any()) == (4 * gap < 16)


def test_streams_carry_on_after_the_pool_rearranges_their_regions():
    """[gap 1][A][gap 3][B][tail]: a newcomer that fits no gap makes the pool slide A
    by 1 and B by 4, onto tokens A held, while both are mid-stream."""
    pair = Pair(36)
    one = _run(pair, 1, [1] * 3, 0, 1, base=1000)
    a = _run(pair, 2, [4] * 6, 2, 8, base=2000)
    three = _run(pair, 3, [3] * 3, 0, 3, base=3000)
    b = _run(pair, 4, [4] * 6, 2, 8, base=4000)
    for stream in (one, a, three, b):
        next(stream)  # opens them in this order: offsets 0, 1, 11, 14
    for _ in range(2):  # A and B hold 12 tokens each: the sink and a full, shifted window
        next(a), next(b)
    pair.remove(1)
    pair.remove(3)
    assert [(s // TOKEN, n // TOKEN) for s, n in pair.pool._free] == [(0, 1), (11, 3), (24, 12)]
    newcomer = _run(pair, 5, [4] * 4, 2, 12, base=5000)  # 14 tokens: no gap holds it
    total, keys, _ = next(newcomer)
    assert [pair.offset(r) for r in (2, 4, 5)] == [0, 10, 20]
    assert keys == _expected(total, 2, 12, base=5000)
    for rid, stream in ((2, a), (4, b)):
        for total, keys, values in stream:
            assert keys == values == _expected(total, 2, 8, base=1000 * rid)


def _random_workload(rng: random.Random, counts: dict) -> None:
    """Requests of random sizes open, step (one at a time) and end, in random
    order; every step must attend exactly its retention."""
    pair = Pair(rng.randint(12, 64))
    rearrange = pair.pool._rearrange

    def counted():
        counts["rearranges"] += 1
        rearrange()

    pair.pool._rearrange = counted
    reqs, waiting, running = {}, [], []
    for _ in range(rng.randint(30, 150)):
        r = rng.random()
        if r < 0.25:
            rid = len(reqs) + 1
            sink = rng.randint(0, 4)
            window = rng.randint(1, min(15, pair.pool._size // TOKEN - sink))
            reqs[rid] = SimpleNamespace(sink=sink, window=window, written=0)
            pair.pool.ingest_request(rid, RequestPoolReqConfig.of({"kv": pair.store.config.buffer(sink + window)}))
            waiting.append(rid)
        elif r < 0.35 and (running or waiting):
            gone = rng.choice(running + waiting)
            pair.remove(gone)
            (running if gone in running else waiting).remove(gone)
        while waiting and pair.pool.reserve(waiting[0], "dit", "rollout", None) is None:  # arrival order
            running.append(waiting.pop(0))
        if not running:
            continue
        rid = rng.choice(running)
        req = reqs[rid]
        span = rng.randint(1, req.window)
        outcome, steps, ctx = pair.admit(rid, RollingPosition(req.written, req.sink, req.window), span)
        assert outcome.ok, outcome.reason
        pair.plan(steps, ctx)
        tags = _tags(1000 * rid + torch.arange(req.written, req.written + span))
        for layer in range(LAYERS):
            pair.store.write(layer, tags, tags)
        pair.store.commit(steps[1], ctx)
        req.written += span
        for layer in range(LAYERS):
            k, v = pair.store.window(layer)
            assert k[0, :, 0, 0].tolist() == v[0, :, 0, 0].tolist() == \
                _expected(req.written, req.sink, req.window, base=1000 * rid)


@pytest.mark.parametrize("seed", range(4))
def test_random_requests_keep_their_streams_through_every_rearrange(monkeypatch, seed):
    rng = random.Random(seed)
    counts = {"rearranges": 0}
    for _ in range(50):
        monkeypatch.setattr(pool_module, "STAGE_BYTES", rng.choice([2, 4, 16]) * TOKEN)
        _random_workload(rng, counts)
    assert counts["rearranges"], counts  # the workloads do rearrange


def test_a_refused_step_forgets_the_requests_it_started():
    pair = Pair(36)
    pair.open(1, 18)
    outcome, steps, ctx = pair.admit(1, RollingPosition(0, 6, 12), 4)
    assert outcome.ok and 1 in pair.store._requests
    pair.store.rollback_admit(steps[1], ctx)
    pair.pool.rollback_admit(steps[0], ctx)
    assert 1 not in pair.store._requests and pair.pool._reserved == {1: 18 * TOKEN}
    # a request already under way stays
    list(_run(pair, 3, [4], 6, 12))
    outcome, steps, ctx = pair.admit(3, RollingPosition(4, 6, 12), 4)
    assert outcome.ok
    pair.store.rollback_admit(steps[1], ctx)
    assert pair.store._requests[3].written == 4


def test_a_window_that_cannot_hold_the_step_fails_the_request():
    pair = Pair(18)
    pair.open(1, 3)
    outcome, _, _ = pair.admit(1, RollingPosition(0, 0, 3), 4)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal
    assert "window must hold the step" in outcome.reason.message


def test_a_request_must_keep_the_window_it_started_with():
    pair = Pair(64)
    list(_run(pair, 1, [4], sink=6, window=12))
    outcome, _, _ = pair.admit(1, RollingPosition(4, 6, 16), 4)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)


def test_a_position_that_disagrees_with_the_commits_fails_the_request():
    pair = Pair(18)
    list(_run(pair, 1, [4, 4], 6, 12))
    outcome, _, _ = pair.admit(1, RollingPosition(4, 6, 12), 4)  # 8 were committed
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)


def test_a_first_step_that_keeps_other_than_the_declared_buffer_fails_the_request():
    pair = Pair(30)
    pair.open(1, 18)
    outcome, _, _ = pair.admit(1, RollingPosition(0, 6, 14), 4)
    assert not outcome.ok and "RollingKVConfig.buffer" in outcome.reason.message


def test_a_request_without_a_kv_buffer_fails_at_its_first_step():
    pair = Pair(30)
    pair.pool.ingest_request(1, RequestPoolReqConfig.of({"other": ((4, 64), torch.float32)}))
    assert pair.pool.reserve(1, "dit", "rollout", None) is None
    outcome, _, _ = pair.admit(1, RollingPosition(0, 6, 12), 4)
    assert not outcome.ok and "has no kv buffer" in outcome.reason.message


def test_the_kv_share_a_region_with_other_buffers_untouched():
    pair = Pair(64)
    pair.open(1, 18, extra={"cache": ((8, 64), torch.float32)})
    stream = _run(pair, 1, [4] * 8, 6, 12)
    next(stream)
    pair.pool.buffer(1, "cache").fill_(7.0)
    for total, keys, _ in stream:
        assert keys == _expected(total, 6, 12)
    assert bool((pair.pool.buffer(1, "cache") == 7.0).all())


def test_interleaved_requests_keep_their_own_streams():
    pair = Pair(36)
    streams = {1: _run(pair, 1, [4] * 7, 6, 12, base=1000), 2: _run(pair, 2, [3] * 7, 6, 12, base=2000)}
    for _ in range(7):
        for rid, stream in streams.items():
            total, keys, _ = next(stream)
            assert keys == _expected(total, 6, 12, base=1000 * rid), rid


def test_a_removed_request_starts_over():
    pair = Pair(18)
    list(_run(pair, 1, [4] * 6, 6, 12))
    pair.remove(1)
    total, keys, _ = next(_run(pair, 1, [4], 6, 12))
    assert (total, keys) == (4, [0, 1, 2, 3])


def test_writing_more_rows_than_the_step_has_is_refused():
    pair = Pair(18)
    pair.open(1, 18)
    outcome, steps, ctx = pair.admit(1, RollingPosition(0, 6, 12), 4)
    pair.plan(steps, ctx)
    with pytest.raises(ValueError, match="rows"):
        pair.store.write(0, torch.zeros(2, 4, HEADS, DIM), torch.zeros(2, 4, HEADS, DIM))


def test_the_window_depends_on_its_pool_runs_eager_and_does_not_shard_yet():
    spec = RollingKVSpec(resource_key="kv", nodes={"dit"},
                         config=RollingKVConfig(num_layers=1, num_heads=1, head_dim=8, pool="pool"))
    assert spec.depends_on() == {"pool"}
    store = RollingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    assert not store.can_capture() and store.depends_on() == {"pool"}
    with pytest.raises(RuntimeError, match="no request pool 'pool'"):
        store.bind_dependencies({})
    tp2 = SimpleNamespace(world_size=2)
    with pytest.raises(NotImplementedError, match="shard"):
        RollingKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=tp2))
