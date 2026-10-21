"""Prefix reuse over a recurrent pool: a prompt's state saved at a page boundary, restored for
a later prompt that shares it.

A recurrent layer cannot skip tokens the way attention can: its state after token n is all it
keeps of tokens 0..n. So a hit is a snapshot. An eager long prefill's forward copies its
request's slot into a snapshot slot at page boundaries (reserved at admit, indexed at commit),
and a later request whose page keys reach one restores it at plan. The agreed length is a
boundary some snapshot holds, which can cut a KV hit shorter.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import types

import pytest
import torch

from mstar.engine.resources import AdmitOutcome, AllocationFailed, Resource, StepRunner, SubmoduleStep
from mstar.engine.resources.base import CachedPrefix
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.recurrent import pool as pool_mod
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext

POOL, KV, AFTER, NODE, WALK, PAGE = "state", "kv", "after", "llm", "prefill", 4


def _pool(prefix_slots: int = 2, rank: int = 0, world: int = 1) -> RecurrentStatePool:
    config = RecurrentStateConfig(
        num_layers=2, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)},
        max_slots=4, prefix_slots=prefix_slots, prefix_page_size=PAGE,
    )
    group = None if world == 1 else types.SimpleNamespace(rank=rank, world_size=world)
    pool = RecurrentStatePool(device=torch.device("cpu"), config=config, comm_group=group)
    pool.enable_prefix_cache(b"a root", {"main": (WALK, None)})
    return pool


class _Stub(Resource):
    """A KV cache that answers a fixed length and serves any shorter one, or a resource
    planned after the pool that refuses when told to."""

    def __init__(self, length: int | None = None):
        self.length, self.refuse = length, False

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def resolve_cached_prefix(self, rid, node_name, graph_walk, inputs):
        return None if self.length is None else CachedPrefix("main", self.length, self.length)

    def admit(self, step, ctx):
        if self.refuse:
            return AdmitOutcome(ok=False, reason=AllocationFailed(
                "full", pages_short=1, label="main", request_id=ctx.request_ids[0]))
        return AdmitOutcome(ok=True)


def _runner(pool, **others) -> StepRunner:
    resources = {POOL: pool, **others}
    return StepRunner(resources, node_resources={NODE: list(resources)})


def _ingest(run: StepRunner, rid: str, tokens: list[int]) -> None:
    pages = [tokens[at:at + PAGE] for at in range(0, len(tokens), PAGE)]
    run.ingest_request(rid, {POOL: KVReqConfig(prefix_keys={"main": chain(pages)})})


def _step(rid: str, span: int, snapshots=(), keys=(POOL,), spares=()) -> SubmoduleStep:
    seg = Segment(rid, "main", span)
    st = SubmoduleStep(segments=[seg], steps={
        key: RecurrentStep(segments=(seg,), snapshots=tuple(snapshots),
                           spare_snapshots=tuple(spares)) if key == POOL
        else RecurrentStep(segments=(seg,)) for key in keys})
    st.set_ctx(StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False))
    return st


def _state(pool: RecurrentStatePool, rid: str) -> torch.Tensor:
    return pool.block("state", 0)[pool._slots[rid]["main"].index]


def _prefill(run, pool, rid, tokens, saves, spares=()) -> tuple[int, torch.Tensor]:
    """One prefill step past whatever hit: the forward leaves the state at ``tokens`` -> value
    for each boundary in ``saves`` and saves it there, as a spare at those in ``spares``.
    Returns the hit and the restored state."""
    prefix = run.resolve_cached_prefix(rid, NODE, WALK, None)
    matched = prefix.tokens if prefix is not None else 0
    run.apply_cached_prefix(rid, NODE, WALK, None, prefix)
    points = {spare: [(rid, "main", t) for t in saves if t > matched and (t in spares) == spare]
              for spare in (False, True)}
    st = _step(rid, len(tokens) - matched, points[False], spares=points[True])
    assert run.admit(st).ok
    run.plan(st)
    restored = _state(pool, rid).clone()
    for tokens_at, value in sorted(saves.items()):
        if tokens_at > matched:
            for layer in (0, 1):
                pool.block("state", layer)[pool._slots[rid]["main"].index].fill_(value)
            pool.write_snapshot(rid, "main", tokens_at)
    run.commit(st)
    return matched, restored


def test_a_later_prompt_restores_the_state_a_snapshot_saved():
    pool = _pool()
    run = _runner(pool)
    a = list(range(20))
    _ingest(run, "a", a)
    assert _prefill(run, pool, "a", a, {8: 1.0, 16: 2.0})[0] == 0

    # pages 0-3 shared, page 4 not: the longest snapshot along b's chain is a's at 16
    b = a[:16] + [99, 98, 97, 96, 95]
    _ingest(run, "b", b)
    matched, restored = _prefill(run, pool, "b", b, {})
    assert matched == 16
    assert bool((restored == 2.0).all())
    assert pool._slots["b"]["main"].has_state
    assert not pool._snap_refs  # restored: nothing still holds the snapshot


def test_a_shorter_kv_hit_caps_the_restore_at_a_snapshot_that_fits():
    pool, kv = _pool(), _Stub(length=12)
    run = _runner(pool, **{KV: kv})
    a = list(range(20))
    _ingest(run, "a", a)
    kv.length = 0
    _prefill(run, pool, "a", a, {8: 1.0, 16: 2.0})

    kv.length = 12  # its pages for 12..15 were evicted, say
    _ingest(run, "b", a[:16] + [99, 98, 97, 96, 95])
    assert run.resolve_cached_prefix("b", NODE, WALK, None).tokens == 8
    assert pool._leases["b"][2] == 8


def test_a_keyed_miss_answers_zero_and_an_unkeyed_request_none():
    pool = _pool()
    run = _runner(pool)
    _ingest(run, "a", list(range(12)))
    assert pool.resolve_cached_prefix("a", NODE, WALK, None).tokens == 0
    run.ingest_request("u", {})
    assert pool.resolve_cached_prefix("u", NODE, WALK, None) is None
    # a pool keeping no snapshots still answers 0: another resource's hit must not skip it
    closed = _pool(prefix_slots=0)
    closed.ingest_request("a", KVReqConfig(prefix_keys={"main": chain([[1, 2, 3, 4]] * 3)}))
    assert closed.resolve_cached_prefix("a", NODE, WALK, None).tokens == 0


def test_the_oldest_snapshot_nothing_holds_goes_first():
    pool = _pool(prefix_slots=2)
    run = _runner(pool)
    a = list(range(20))
    _ingest(run, "a", a)
    _prefill(run, pool, "a", a, {8: 1.0, 16: 2.0})
    # b holds a's snapshot at 16 from its probe: c's save must take the one at 8
    _ingest(run, "b", a[:16] + [99] * 5)
    assert run.resolve_cached_prefix("b", NODE, WALK, None).tokens == 16
    c = list(range(100, 120))
    _ingest(run, "c", c)
    _prefill(run, pool, "c", c, {8: 3.0})
    held = {meta[1] for meta in pool._snap_meta.values()}
    assert held == {16, 8} and pool._snap_by_key  # a@16 kept, c@8 took a@8's slot
    _ingest(run, "d", a[:8] + [77] * 9)
    assert pool.resolve_cached_prefix("d", NODE, WALK, None).tokens == 0  # a@8 is gone


def test_a_spare_takes_only_a_free_slot_and_is_evicted_first():
    pool = _pool(prefix_slots=2)
    run = _runner(pool)
    a, c = list(range(20)), list(range(100, 120))
    _ingest(run, "a", a)
    _prefill(run, pool, "a", a, {8: 1.0, 16: 2.0}, spares={8})
    _ingest(run, "e", a[:8] + [77] * 9)
    assert pool.resolve_cached_prefix("e", NODE, WALK, None).tokens == 8  # a free slot took the spare
    run.remove_request("e")
    # c's kept snapshot takes a's spare at 8, though a's kept one at 16 is older; c's spare
    # finds no free slot
    _ingest(run, "c", c)
    _prefill(run, pool, "c", c, {8: 3.0, 16: 4.0}, spares={8})
    assert sorted(meta[1] for meta in pool._snap_meta.values()) == [16, 16]
    for rid, prompt in (("b", a), ("d", c)):
        _ingest(run, rid, prompt[:16] + [99] * 5)
        assert pool.resolve_cached_prefix(rid, NODE, WALK, None).tokens == 16
    _ingest(run, "f", a[:8] + [77] * 9)
    assert pool.resolve_cached_prefix("f", NODE, WALK, None).tokens == 0


def test_a_refused_admit_gives_the_hit_back():
    pool, after = _pool(), _Stub()
    run = _runner(pool, **{AFTER: after})
    a = list(range(20))
    _ingest(run, "a", a)
    _prefill(run, pool, "a", a, {16: 2.0})
    _ingest(run, "b", a[:16] + [99] * 5)
    assert run.resolve_cached_prefix("b", NODE, WALK, None).tokens == 16
    after.refuse = True
    st = _step("b", 5, [("b", "main", 20)], keys=(POOL, AFTER))
    assert not run.admit(st).ok
    # the restore is a lease again, and the reserved snapshot slot is free
    assert pool._leases["b"][2] == 16 and not pool._restores and not pool._snap_writes
    assert pool.resolve_cached_prefix("b", NODE, WALK, None).tokens == 16


def test_rows_admitted_one_at_a_time_each_give_their_own_hit_back():
    """The unbatched fallback admits each row on its own and, when one is refused, unwinds
    the rows admitted ahead of it (#561): each row's restore and snapshot slot come back."""
    pool, after = _pool(prefix_slots=3), _Stub()
    run = _runner(pool, **{AFTER: after})
    a = list(range(20))
    _ingest(run, "a", a)
    _prefill(run, pool, "a", a, {16: 2.0})
    steps = {}
    for rid in ("b", "c"):
        _ingest(run, rid, a[:16] + [99] * 5)
        assert run.resolve_cached_prefix(rid, NODE, WALK, None).tokens == 16
        steps[rid] = _step(rid, 5, [(rid, "main", 20)], keys=(POOL, AFTER))
    assert run.admit(steps["b"]).ok
    after.refuse = True
    assert not run.admit(steps["c"]).ok
    run.rollback_admit(steps["b"])
    assert pool._leases["b"][2] == pool._leases["c"][2] == 16
    assert not pool._restores and not pool._snap_writes and len(pool._snap_free) == 2


def test_dropping_a_pre_plan_gives_back_only_what_its_pass_took():
    """The live step keeps its restore and snapshot slot; the staged one's come back."""
    pool = _pool(prefix_slots=3)
    run = _runner(pool)
    a = list(range(20))
    _ingest(run, "a", a)
    _prefill(run, pool, "a", a, {16: 2.0})
    for rid in ("b", "c"):
        _ingest(run, rid, a[:16] + [99] * 5)
        assert run.resolve_cached_prefix(rid, NODE, WALK, None).tokens == 16
    assert run.admit(_step("b", 5, [("b", "main", 20)])).ok
    seg = Segment("c", "main", 5)
    staged = RecurrentStep(segments=(seg,), snapshots=(("c", "main", 20),))
    ctx = StepContext(request_ids=("c",), graph_walk=WALK, slot=0, capture=False, is_preplan=True)
    assert pool.admit(staged, ctx).ok
    pool.plan(staged, ctx)
    pool.clear_preplan()
    assert pool._leases["c"][2] == 16 and "b" not in pool._leases
    assert set(pool._restores) == {("b", "main")} and set(pool._snap_writes) == {("b", "main", 20)}


def test_under_tp_a_prepare_answers_what_rank_0_decided():
    """Rank 0 probes as it sends the step; a snapshot saved since must not turn the prepare
    of a step the other ranks build without it into a hit."""
    leader = _pool(rank=0, world=2)
    run = _runner(leader)
    a = list(range(20))
    _ingest(run, "a", a)
    _ingest(run, "b", a[:16] + [99] * 5)
    assert leader.resolve_cached_prefix("b", NODE, WALK, None).tokens == 0
    _prefill(run, leader, "a", a, {16: 2.0})
    assert leader.resolve_cached_prefix("b", NODE, WALK, object()).tokens == 0
    assert leader.resolve_cached_prefix("b", NODE, WALK, None).tokens == 16


def test_a_follower_adopts_rank_0s_snapshot_once_its_own_commit_saved_it():
    ranks = leader, follower = _pool(rank=0, world=2), _pool(rank=1, world=2)
    runs = {pool: _runner(pool) for pool in ranks}
    a = list(range(20))
    steps = {}
    for pool in ranks:
        _ingest(runs[pool], "a", a)
        steps[pool] = _step("a", 20, [("a", "main", 16)])
        assert runs[pool].admit(steps[pool]).ok
        runs[pool].plan(steps[pool])
        pool.write_snapshot("a", "main", 16)
    runs[leader].commit(steps[leader])
    for pool in ranks:
        _ingest(runs[pool], "b", a[:16] + [99] * 5)

    matched = leader.resolve_cached_prefix("b", NODE, WALK, None).tokens
    assert matched == 16
    # its own pool holds nothing yet, and only rank 0's word counts anyway
    assert follower.resolve_cached_prefix("b", NODE, WALK, None).tokens == 0
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)
    assert not follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    runs[follower].commit(steps[follower])
    with pytest.raises(RuntimeError, match="diverged"):
        follower.adopt_cached_prefix("b", NODE, WALK, matched, [pages[0] + 1])
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert follower.resolve_cached_prefix("b", NODE, WALK, None).tokens == 16


def test_a_waiting_follower_hashes_the_prompt_once(monkeypatch):
    """The follower polls its adopt until its own commit saves the snapshot; the prompt's
    keys are rooted on the first poll only, and dropped with the request."""
    ranks = leader, follower = _pool(rank=0, world=2), _pool(rank=1, world=2)
    runs = {pool: _runner(pool) for pool in ranks}
    a = list(range(20))
    steps = {}
    for pool in ranks:
        _ingest(runs[pool], "a", a)
        steps[pool] = _step("a", 20, [("a", "main", 16)])
        assert runs[pool].admit(steps[pool]).ok
        runs[pool].plan(steps[pool])
        pool.write_snapshot("a", "main", 16)
    runs[leader].commit(steps[leader])
    for pool in ranks:
        _ingest(runs[pool], "b", a[:16] + [99] * 5)
    matched = leader.resolve_cached_prefix("b", NODE, WALK, None).tokens
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)

    hashed = []
    real = pool_mod.fingerprint
    monkeypatch.setattr(pool_mod, "fingerprint", lambda *f: hashed.append(f) or real(*f))
    for _ in range(3):
        assert not follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert len(hashed) == matched // PAGE
    runs[follower].commit(steps[follower])
    hashed.clear()
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert hashed == []
    runs[follower].remove_request("b")
    assert "b" not in follower._prefix_rooted


def test_a_request_removed_before_its_step_commits_gives_its_snapshot_slots_back():
    """A forward that raises gets neither commit nor rollback: the failed request's removal
    returns the slots its admit reserved, or prefix reuse would stop once all are gone."""
    pool = _pool(prefix_slots=2)
    run = _runner(pool)
    _ingest(run, "a", list(range(20)))
    assert run.resolve_cached_prefix("a", NODE, WALK, None).tokens == 0
    st = _step("a", 20, [("a", "main", PAGE), ("a", "main", 2 * PAGE)])
    assert run.admit(st).ok
    run.plan(st)
    pool.write_snapshot("a", "main", PAGE)
    run.remove_request("a")
    assert len(pool._snap_free) == 2
    assert not pool._snap_writes and not pool._snap_written


def test_a_follower_restores_nothing_where_rank_0_restored_nothing():
    follower = _pool(rank=1, world=2)
    _ingest(_runner(follower), "b", list(range(20)))
    assert follower.adopt_cached_prefix("b", NODE, WALK, 16, None)
    assert "b" not in follower._leases
