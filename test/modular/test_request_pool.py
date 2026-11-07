"""Request pool: regions, admission and rearranging, on CPU.

Every byte a request writes into its region carries a tag of its own, so a
rearrange can be checked against what each request wrote.
"""

import logging
import random

import pytest
import torch

from mstar.engine.resources.request_pool import (
    RequestPoolConfig,
    RequestPoolReqConfig,
    RequestPoolSpec,
    RequestPoolStep,
)
from mstar.engine.resources.request_pool import manager as manager_module
from mstar.engine.resources.request_pool.config import ALIGN
from mstar.engine.resources.request_pool.manager import RequestPool
from mstar.engine.resources.step import ASKS, AdmitRuntimeError, Segment, StepContext

UNIT = ALIGN  # one buffer of 64 float32 is one aligned unit


def _pool(units: int) -> RequestPool:
    return RequestPool(RequestPoolConfig(pool_bytes=units * UNIT), torch.device("cpu"))


def _wants(*units: int) -> RequestPoolReqConfig:
    """Buffers of ``units`` aligned units each, float32, named b0, b1, ..."""
    return RequestPoolReqConfig.of({f"b{i}": ((u, 64), torch.float32) for i, u in enumerate(units)})


def _step(*rids):
    step = RequestPoolStep(segments=tuple(Segment(rid, "main", 1) for rid in rids))
    return step, StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _open(pool, rid, *units) -> None:
    """Open a request as the engine does: it declares its buffers, and the admission
    queue sets that room aside once it fits."""
    pool.ingest_request(rid, _wants(*units))
    assert pool.ask(rid, "n", "w") is ASKS
    assert pool.reserve(rid, "n", "w", None) is None


def _admit(pool, *rids):
    step, ctx = _step(*rids)
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    return step, ctx


def _regions(pool) -> dict:
    return {rid: (r.offset // UNIT, r.nbytes // UNIT) for rid, r in pool._regions.items()}


def _free(pool) -> list:
    return [(s // UNIT, n // UNIT) for s, n in pool._free]


def _fill(pool, rid, tag: float) -> None:
    for name in pool._regions[rid].layout:
        pool.buffer(rid, name).fill_(tag)


def _holds(pool, rid, tag: float) -> bool:
    return all(bool((pool.buffer(rid, name) == tag).all()) for name in pool._regions[rid].layout)


def test_requests_take_exact_regions_first_fit_and_name_their_buffers():
    pool = _pool(30)
    _open(pool, 1, 10, 8)
    _open(pool, 2, 7)
    _admit(pool, 1, 2)
    assert _regions(pool) == {1: (0, 18), 2: (18, 7)} and _free(pool) == [(25, 5)]
    b0, b1 = pool.buffer(1, "b0"), pool.buffer(1, "b1")
    assert b0.shape == (10, 64) and b1.dtype == torch.float32
    assert b0.data_ptr() % ALIGN == 0 and b1.data_ptr() - b0.data_ptr() == 10 * UNIT
    assert pool.view("b0", row=1).data_ptr() == pool.buffer(2, "b0").data_ptr()
    with pytest.raises(KeyError, match="no buffer 'b9'"):
        pool.buffer(1, "b9")
    pool.remove_request(1)
    _open(pool, 3, 6)
    _admit(pool, 3)
    assert _regions(pool)[3] == (0, 6) and _free(pool) == [(6, 12), (25, 5)]


def test_buffers_of_any_dtype_start_aligned():
    pool = _pool(64)
    pool.ingest_request(1, RequestPoolReqConfig.of({
        "a": ((3,), torch.bfloat16), "b": ((5, 7), torch.float32), "c": ((1, 2, 3), torch.int64)}))
    assert pool.reserve(1, "n", "w", None) is None
    _admit(pool, 1)
    expected = (("a", (3,), torch.bfloat16), ("b", (5, 7), torch.float32), ("c", (1, 2, 3), torch.int64))
    for name, shape, dtype in expected:
        t = pool.buffer(1, name)
        assert t.shape == shape and t.dtype == dtype and t.data_ptr() % ALIGN == 0 and t.is_contiguous()
    assert _regions(pool)[1] == (0, 3)


def test_a_request_gets_no_room_until_a_region_frees():
    pool = _pool(20)
    _open(pool, 1, 15)
    _admit(pool, 1)
    pool.ingest_request(2, _wants(10))
    assert pool.reserve(2, "n", "w", None) == 5 * UNIT  # the room it found
    pool.remove_request(1)
    assert pool.reserve(2, "n", "w", None) is None


def test_a_request_that_declared_nothing_fails_at_its_first_step():
    pool = _pool(30)
    step, ctx = _step(1)
    outcome = pool.admit(step, ctx)
    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)
    assert "RequestPoolReqConfig" in outcome.reason.message and pool.num_free_bytes() == 30 * UNIT


def test_ask_queues_only_a_declared_request_that_holds_nothing_yet():
    pool = _pool(30)
    assert pool.ask(1, "n", "w") is None  # declared nothing
    pool.ingest_request(1, _wants(5))
    assert pool.ask(1, "n", "w") is ASKS
    assert pool.reserve(1, "n", "w", None) is None
    assert pool.ask(1, "n", "w") is None  # set aside
    _admit(pool, 1)
    assert pool.ask(1, "n", "w") is None  # placed


def test_room_set_aside_for_a_request_is_not_given_to_another():
    pool = _pool(20)
    _open(pool, 1, 15)
    pool.ingest_request(2, _wants(10))
    assert pool.reserve(2, "n", "w", None) == 5 * UNIT
    pool.unreserve(1)
    assert pool.reserve(2, "n", "w", None) is None
    assert not pool.grew_since(10 * UNIT)
    pool.remove_request(2)
    assert pool.grew_since(10 * UNIT)


def test_removing_a_request_drops_what_it_declared_and_reserved():
    pool = _pool(20)
    _open(pool, 1, 15)
    pool.remove_request(1)
    assert not pool._wants and not pool._reserved and pool._kept == 0 and pool._room() == 20 * UNIT
    pool.post_warmup_validate()


def test_a_request_larger_than_the_pool_is_refused_before_it_reaches_a_worker():
    spec = RequestPoolSpec(resource_key="pool", nodes={"n"}, config=RequestPoolConfig(pool_bytes=10 * UNIT))
    with pytest.raises(ValueError, match="resources: pool: pool_bytes"):
        _wants(11).validate(spec)
    _wants(10).validate(spec)
    pool = _pool(10)
    pool.ingest_request(1, _wants(11))
    outcome = pool.ask(1, "n", "w")
    assert not outcome.ok and "pool_bytes" in outcome.reason.message


@pytest.mark.parametrize("bad", [
    RequestPoolReqConfig(names=["a", "a"], shapes=[[1], [1]], dtypes=["float32", "float32"]),
    RequestPoolReqConfig(names=["a"], shapes=[[1], [2]], dtypes=["float32"]),
    RequestPoolReqConfig(names=["a"], shapes=[[0, 3]], dtypes=["float32"]),
    RequestPoolReqConfig(names=["a"], shapes=[[2]], dtypes=["floaty"]),
])
def test_a_malformed_declaration_is_refused(bad):
    with pytest.raises(ValueError):
        bad.validate(None)


def test_yaml_overrides_size_the_pool_and_refuse_nonsense():
    spec = RequestPoolSpec(resource_key="pool", nodes={"n"}, config=RequestPoolConfig(pool_bytes=UNIT))
    spec.apply_yaml_overrides(pool_bytes=64 * UNIT)
    assert spec.config.pool_bytes == 64 * UNIT
    for bad in (0, -1, True, "64", 64.0):
        with pytest.raises(ValueError, match="pool_bytes must be an integer >= 1"):
            spec.apply_yaml_overrides(pool_bytes=bad)


def test_a_refused_step_gives_back_its_region_and_sets_the_room_aside_again():
    pool = _pool(30)
    _open(pool, 1, 10)
    _open(pool, 2, 5)
    _admit(pool, 1)
    step, ctx = _step(1, 2)
    assert pool.admit(step, ctx).ok
    pool.rollback_admit(step, ctx)
    # request 1 keeps the region it already had; request 2's goes back to being set aside
    assert _regions(pool) == {1: (0, 10)} and pool._reserved == {2: 5 * UNIT} and _free(pool) == [(10, 20)]


def test_the_pool_runs_eager_and_does_not_shard_yet():
    from types import SimpleNamespace

    assert not _pool(4).can_capture()
    spec = RequestPoolSpec(resource_key="pool", nodes={"n"}, config=RequestPoolConfig(pool_bytes=UNIT))
    info = SimpleNamespace(joint_comm_group=SimpleNamespace(world_size=2), device=torch.device("cpu"))
    with pytest.raises(NotImplementedError):
        RequestPool.build(spec, info)


def test_rearranging_joins_the_gaps_keeps_every_region_and_logs(caplog):
    pool = _pool(30)
    for rid in (1, 2, 3):
        _open(pool, rid, 10)
    _admit(pool, 1, 2, 3)
    for rid in (1, 2, 3):
        _fill(pool, rid, float(rid))
    pool.remove_request(1)
    pool.remove_request(3)
    assert _free(pool) == [(0, 10), (20, 10)]
    _open(pool, 4, 20)  # neither gap holds it
    with caplog.at_level(logging.INFO, logger=manager_module.__name__):
        _admit(pool, 4)
    assert _regions(pool) == {2: (0, 10), 4: (10, 20)} and _free(pool) == []
    assert _holds(pool, 2, 2.0)
    # serve_concurrent.py counts this line in the server log
    assert [r.getMessage() for r in caplog.records] == [
        "request pool request_pool: rearranged, moving 1 regions (0.00 GiB)"]


class _NoSelfOverlap:
    """Stands in for the pool's words in ``_move_left``: a copy within the pool must
    read and write disjoint words, since a GPU copies them in parallel. (On CPU, an
    overlapping copy to the left runs front to back and comes out right anyway.)"""

    def __init__(self, words: torch.Tensor, counts: dict):
        self.words, self.counts = words, counts

    def _range(self, t: torch.Tensor) -> range | None:
        if t.untyped_storage().data_ptr() != self.words.untyped_storage().data_ptr():
            return None  # the staging block
        start = t.storage_offset() - self.words.storage_offset()
        return range(start, start + t.numel())

    def __getitem__(self, key):
        return self.words[key]

    def __setitem__(self, key, value):
        src, dst = self._range(value), self._range(self.words[key])
        if src is None:
            self.counts["staged"] += 1
        else:
            assert src.stop <= dst.start or dst.stop <= src.start, f"a copy reads and writes {src} -> {dst}"
        self.words[key] = value


@pytest.mark.parametrize("gap", [1, 2, 3, 7, 20])
def test_a_region_moved_left_keeps_every_word_however_short_the_gap(monkeypatch, gap):
    """A gap under a quarter of the staging block is crossed through it, block by
    block; a longer one in place. Either way every word lands where it should."""
    monkeypatch.setattr(manager_module, "STAGE_BYTES", 32 * 8)  # 32 words
    pool = _pool(4)
    words = pool._words
    words.copy_(torch.arange(words.numel()))
    pool._stage.fill_(-7)
    counts = {"staged": 0}
    pool._words = _NoSelfOverlap(words, counts)
    expect = words[gap:gap + 70].clone()
    pool._move_left(gap, 0, 70)
    assert torch.equal(words[:70], expect)
    assert (counts["staged"] > 0) == (4 * gap < 32)


def _random_workload(rng: random.Random, counts: dict) -> None:
    """Requests of random sizes open, are admitted (up to three a step) and end, in
    random order; after every step each region must still hold its own tag."""
    pool = _pool(rng.randint(8, 48))
    pool._words = _NoSelfOverlap(pool._words, counts)
    rearrange = pool._rearrange

    def counted():
        counts["rearranges"] += 1
        rearrange()

    pool._rearrange = counted
    waiting, resident, tags, next_rid = [], [], {}, 1
    for _ in range(rng.randint(20, 80)):
        r = rng.random()
        if r < 0.35:
            units = [rng.randint(1, 6) for _ in range(rng.randint(1, 3))]
            if sum(units) <= pool._size // UNIT:
                pool.ingest_request(next_rid, _wants(*units))
                waiting.append(next_rid)
                next_rid += 1
        elif r < 0.55 and (resident or waiting):
            gone = rng.choice(resident + waiting)
            pool.remove_request(gone)
            (resident if gone in resident else waiting).remove(gone)
        while waiting and pool.reserve(waiting[0], "n", "w", None) is None:  # arrival order
            rid = waiting.pop(0)
            _admit(pool, rid)
            tags[rid] = float(rid)
            _fill(pool, rid, tags[rid])
            resident.append(rid)
        for rid in resident:
            assert _holds(pool, rid, tags[rid])
        spans = sorted((reg.offset, reg.offset + reg.nbytes) for reg in pool._regions.values())
        assert all(a1 <= b0 for (_, a1), (b0, _) in zip(spans, spans[1:], strict=False))


@pytest.mark.parametrize("seed", range(4))
def test_random_requests_keep_their_bytes_through_every_rearrange(monkeypatch, seed):
    rng = random.Random(seed)
    counts = {"rearranges": 0, "staged": 0}
    for _ in range(40):
        monkeypatch.setattr(manager_module, "STAGE_BYTES", rng.choice([2, 8, 32]) * UNIT)
        _random_workload(rng, counts)
    assert counts["rearranges"], counts  # the workloads do rearrange

