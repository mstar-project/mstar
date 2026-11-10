"""Prefix reuse under TP: rank 0 decides each hit, and the other ranks take the same pages."""

from __future__ import annotations

import sys
from array import array
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT, WALK
from test_kv_prefix_layout_lease import ASK, IMAGE, IMAGE_WALK, OTHER_ASK, TEXT, _config, _walks
from test_kv_prefix_lease import _Holding, _StubTransfer
from test_resident_delta import _Commit
from test_tp_async_teardown_order import _rank, _Stop
from test_tp_rebuild_land import _Rank
from test_worker_rebuild import _kept

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication import wire
from mstar.engine.engine import Engine
from mstar.engine.resources import Rebuild, StepRunner, apply_yaml_overrides
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig, opening_prefix_caches
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.worker import make_scheduler, make_worker
from mstar.utils.ipc_format import MoveKind, ScheduleTPNode
from mstar.worker.worker import Worker

WALKS = {"main": KeyedWalks((WALK, IMAGE_WALK), ("decode",))}


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(
    rank: int, max_num_pages: int = 32, opt_in: bool = True,
    world_size: int = 2, shared: bool = False, window: int | None = None, **overrides,
) -> KVManager:
    cfg = dict(
        num_layers=1 if window is None else 3, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2,
        max_num_pages=max_num_pages, page_size=PAGE_SIZE, prefix_cache_tp=opt_in,
    )
    cfg.update(overrides)
    cfg = PagedKVConfig(**cfg)
    if window is not None:
        # layers 1 and 2 keep the last ``window`` tokens, in pages of half the first layer's
        cfg.keep_window(window, (1, 2))
    kv = KVManager(
        cfg=cfg,
        name="kv", joint_comm_group=SimpleNamespace(rank=rank, world_size=world_size),
        transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
        needs_remote_transfer=shared,
    )
    kv.enable_prefix_cache(ROOT, WALKS)
    return kv


def _tokens(n: int, seed: int) -> list[int]:
    return [(seed * 7919 + i * 31) % 1000 for i in range(n)]


def _ingest(kv: KVManager, rid: str, *parts) -> None:
    kv.ingest_request(rid, _config(list(parts)))


def _ctx(rid: str, walk: str = WALK) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)


def _step(kv: KVManager, rid: str, length: int, walk: str = WALK) -> KVStep | None:
    inputs = ARNodeInputs(input_seq_len=length)
    prefix = kv.resolve_cached_prefix(rid, NODE, walk, inputs)
    kv.apply_cached_prefix(rid, NODE, walk, inputs, prefix)
    held = prefix.tokens if prefix is not None else 0
    if held == length:
        kv.complete_cached_walk(rid, NODE, walk)
        return None
    step = KVStep(segments=(Segment(rid, "main", length - held),))
    assert kv.admit(step, _ctx(rid, walk)).ok
    kv.plan(step, _ctx(rid, walk))
    return step


def _run(kv: KVManager, rid: str, *parts) -> None:
    for walk, length in _walks(list(parts)):
        step = _step(kv, rid, length, walk)
        if step is not None:
            kv.commit(step, _ctx(rid, walk))


def _cache(ranks, rid: str, tokens: list[int]) -> None:
    for kv in ranks:
        _ingest(kv, rid, tokens)
        _run(kv, rid, tokens)


def _pages(kv: KVManager, rid: str) -> list[int]:
    return kv._streams[rid]["main"].page_indices


class _Submodule:
    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        return ARNodeSubmodule.split_inputs(self, graph_walk, fwd_info, inputs, start, end)


def _engine(kv: KVManager) -> Engine:
    resources = {"kv": kv}
    return make_engine(
        _resources=resources,
        _resource_nodes={"kv": frozenset({NODE})},
        _runner=StepRunner(resources, node_resources={NODE: ["kv"]}),
        _submodules={NODE: make_mgmt(submodule=_Submodule(), resources=resources)},
        _keyed_walks={NODE: {WALK, IMAGE_WALK}},
        _prefix_model="_Model",
        _graph_runtime=SimpleNamespace(
            get_rid_string=lambda rid: f"wire-{rid}", get_rid_handle=lambda wire_rid: wire_rid.removeprefix("wire-"),
        ),
    )


def _batch(rid: str, walk: str = WALK) -> SimpleNamespace:
    return SimpleNamespace(
        node_name=NODE, request_ids=(rid,), step_context=_ctx(rid, walk),
        per_request_info_wrapped={rid: None}, cached_rids=set(),
    )


def _send(leader: KVManager, rid: str) -> ScheduleTPNode:
    """Rank 0's head for ``rid``'s step, as a follower decodes it."""
    engine = _engine(leader)
    return wire.decode(wire.encode(ScheduleTPNode(
        NODE, WALK, [f"wire-{rid}"],
        prefix_hits=engine.decide_cached_prefix(_batch(rid)), page_trails=engine.page_trails(NODE),
    )))


def _follow(follower: KVManager, head: ScheduleTPNode, caught_up: bool = True) -> bool:
    engine = _engine(follower)
    scheduler = make_scheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine), parallel_leader_nodes=set(),
    )
    scheduler.lockstep.enqueue_follow(head)
    return scheduler.lockstep.settle_delta(caught_up)


def _abandon(kv: KVManager, rid: str, walk: str = WALK) -> None:
    engine = _engine(kv)
    worker = make_worker(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        graph_runtime=SimpleNamespace(commit_speculation=lambda *args, **kwargs: None),
    )
    speculation = SimpleNamespace(
        scheduled_batch=SimpleNamespace(node_name=NODE), node_batch=_batch(rid, walk), spec_id=0,
        consumed_streaming_edges={},
    )
    worker.speculator.settle(speculation, success=False)


def _follower_worker(follower: KVManager) -> Worker:
    engine = _engine(follower)
    engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    sched = make_scheduler(engine_manager=engine_manager, parallel_leader_nodes=set())
    return make_worker(
        engine_manager=engine_manager, scheduler=sched, lockstep=sched.lockstep, admission=sched.admission,
    )


@pytest.mark.parametrize("window", [None, 24])
def test_a_follower_takes_rank_0s_hit_and_never_probes_itself(window):
    ranks = leader, follower = _manager(0, window=window), _manager(1, window=window)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    _cache(ranks, "a", shared)
    for kv in ranks:
        _ingest(kv, "b", prompt)

    assert follower.resolve_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=len(prompt))).tokens == 0
    assert _follow(follower, _send(leader, "b"))

    for kv in ranks:
        _run(kv, "b", prompt)
    assert leader._streams["b"]["main"].pages() == follower._streams["b"]["main"].pages()
    assert _pages(leader, "b")[:5] == _pages(leader, "a")[:5], "b missed the prefix a cached"
    if window is not None:
        # The window page of tokens 72-80: a, at 83 tokens, holds window pages 7-10, and b, at 100, 9-12.
        a, b = (leader._streams[rid]["main"].later[0] for rid in "ab")
        assert a.page_indices[2] == b.page_indices[0], "b's sliding layers missed the window a cached"


def test_ranks_cut_a_window_lease_alike_when_one_index_has_dropped_a_page():
    """A cut below a window lease's end needs window pages the lease doesn't hold, and a follower's index can be
    ahead of rank 0's, so the ranks cut from the lease's own pages rather than looking them up."""
    ranks = leader, follower = _manager(0, window=24), _manager(1, window=24)
    shared = _tokens(6 * PAGE_SIZE + 4, seed=0)
    _cache(ranks, "a", shared)
    for kv in ranks:
        _ingest(kv, "b", shared)
    assert _follow(follower, _send(leader, "b"))
    # the follower's index has dropped the window page of tokens 64-72, which a cut to 80 needs
    rooted = follower._streams["b"]["main"].chain.rooted
    follower._index._remove(follower._index.page_for(follower._window_key(rooted[follower._ends_on(1, 8)], 1, 8)))

    answers = [
        StepRunner({"kv": kv, "other": _Holding(80)}, node_resources={NODE: ["kv", "other"]})
        .resolve_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=len(shared)))
        for kv in ranks
    ]

    assert answers[0] == answers[1], "the ranks cut the lease to different lengths"
    assert leader._streams["b"]["main"].lease == follower._streams["b"]["main"].lease


def test_ranks_keep_the_walks_a_window_lease_served_when_a_later_walk_drops_it():
    """A window lease holds the window's pages at its end. A later walk that writes unprobed cuts it back to the
    walks it served, and above one rank only the lease's own pages count, so it holds the window there too."""
    ranks = leader, follower = _manager(0, window=24, max_num_pages=64), _manager(1, window=24, max_num_pages=64)
    parts = [_tokens(3 * PAGE_SIZE, seed=1), 3 * PAGE_SIZE, _tokens(52, seed=2)]
    for kv in ranks:
        _ingest(kv, "a", *parts)
        _run(kv, "a", *parts)
        kv.remove_request("a")
        _ingest(kv, "b", *parts)
    assert _follow(follower, _send(leader, "b"))
    for kv in ranks:
        for walk, length in _walks(parts)[:2]:
            assert _step(kv, "b", length, walk) is None, "the cache didn't serve the first two walks whole"

        kv.apply_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=52), None)

        stream = kv._streams["b"]["main"]
        assert (stream.stored_len, stream.later[0].stored_len) == (6 * PAGE_SIZE, 24), "the served walks' KV went"
        kv.assert_pages_conserved()
    assert [part.page_indices for part in leader._streams["b"]["main"].groups] == [
        part.page_indices for part in follower._streams["b"]["main"].groups
    ]


def test_ranks_evict_alike_however_their_probes_and_commits_interleave():
    ranks = leader, follower = _manager(0, max_num_pages=12), _manager(1, max_num_pages=12)
    x, y, z = (_tokens(2 * PAGE_SIZE + 1, seed=s) for s in (6, 7, 8))
    hits = {"hx": x[:2 * PAGE_SIZE] + _tokens(3, seed=9), "hy": y[:2 * PAGE_SIZE] + _tokens(3, seed=10)}
    _cache(ranks, "x", x)
    _cache(ranks, "y", y)
    x_pages = {kv: _pages(kv, "x")[:2] for kv in ranks}
    z_steps = {}
    for kv in ranks:
        kv.remove_request("x")
        kv.remove_request("y")
        for rid, tokens in (("z", z), *hits.items()):
            _ingest(kv, rid, tokens)
        z_steps[kv] = _step(kv, "z", len(z))

    heads = [_send(leader, rid) for rid in hits]
    for kv in ranks:
        kv.commit(z_steps[kv], _ctx("z"))
    for head in heads:
        assert _follow(follower, head)
    for kv in ranks:
        _run(kv, "hx", hits["hx"])
        for rid in ("hx", "hy", "z"):
            kv.remove_request(rid)

    big = _tokens(9 * PAGE_SIZE, seed=11)
    for kv in ranks:
        _ingest(kv, "big", big)
        _run(kv, "big", big)
    assert _pages(leader, "big") == _pages(follower, "big"), "the ranks evicted different pages, or in another order"
    for kv in ranks:
        assert not set(x_pages[kv]) & set(_pages(kv, "big")), "big evicted the pages hx ran on before older ones"


def test_a_head_that_wont_run_leaves_both_ranks_evicting_alike():
    ranks = leader, follower = _manager(0, max_num_pages=12), _manager(1, max_num_pages=12)
    a = _tokens(3 * PAGE_SIZE, seed=33)
    for rid, tokens in (("x", _tokens(3 * PAGE_SIZE, seed=32)), ("a", a), ("z", _tokens(3 * PAGE_SIZE, seed=34))):
        _cache(ranks, rid, tokens)
    for kv in ranks:
        for rid in ("x", "a", "z"):
            kv.remove_request(rid)
        _ingest(kv, "b", a[:2 * PAGE_SIZE] + _tokens(7, seed=35))
    head = _send(leader, "b")
    head.speculative, head.spec_from_seq = True, 4
    worker = _follower_worker(follower)
    worker.lockstep.enqueue_follow(head)
    assert worker.lockstep.settle_delta(caught_up=False)
    worker.lockstep.close_follow_step(SimpleNamespace(tp_seq=4))
    _abandon(leader, "b")

    big = _tokens(8 * PAGE_SIZE, seed=36)
    for kv in ranks:
        _ingest(kv, "big", big)
        _run(kv, "big", big)
    assert _pages(leader, "big") == _pages(follower, "big"), "one rank kept pages the other evicted"


def test_ranks_evict_alike_when_a_lease_straddles_a_commit_that_files_under_a_page_only_the_index_holds():
    ranks = leader, follower = _manager(0, max_num_pages=12), _manager(1, max_num_pages=12)
    x = _tokens(2 * PAGE_SIZE + 1, seed=40)
    z, h = (x[:2 * PAGE_SIZE] + _tokens(n, seed=s) for n, s in ((PAGE_SIZE + 3, 41), (3, 42)))
    steps = {}
    for kv in ranks:
        for rid, tokens in (("x", x), ("z", z), ("h", h)):
            _ingest(kv, rid, tokens)
        # x and z prefill in one step, so z finds nothing cached and loses x's keys at commit
        steps[kv] = (_step(kv, "x", len(x)), _step(kv, "z", len(z)))
        kv.commit(steps[kv][0], _ctx("x"))
        kv.remove_request("x")
    head = _send(leader, "h")  # rank 0 leases x's pages before z commits, the follower adopts after
    head.speculative, head.spec_from_seq = True, 4
    for kv in ranks:
        kv.commit(steps[kv][1], _ctx("z"))
    worker = _follower_worker(follower)
    worker.lockstep.enqueue_follow(head)
    assert worker.lockstep.settle_delta(caught_up=True)
    worker.lockstep.close_follow_step(SimpleNamespace(tp_seq=4))
    _abandon(leader, "h")
    for kv in ranks:
        kv.remove_request("z")
        kv.remove_request("h")

    big = _tokens(9 * PAGE_SIZE, seed=43)
    for kv in ranks:
        _ingest(kv, "big", big)
        _run(kv, "big", big)

    assert _pages(leader, "big") == _pages(follower, "big"), "the ranks linked z's page differently and evicted apart"


def test_a_pre_plan_rank_0_throws_away_evicts_no_cached_page_a_follower_keeps():
    """Rank 0 pre-plans r's next page with no page free, and throws the stage away. A follower never planned the
    step. Undoing the admit gives the page back to the free list, so a cached page evicted for it left rank 0's index
    without a key the follower still holds, and a later hit on that key stopped the follower."""
    ranks = leader, follower = _manager(0, max_num_pages=8), _manager(1, max_num_pages=8)
    r = _tokens(4 * PAGE_SIZE, seed=53)
    _cache(ranks, "x", _tokens(3 * PAGE_SIZE, seed=52))
    for kv in ranks:
        kv.remove_request("x")
        kv.ingest_request("r", _config([r], decode=PAGE_SIZE))
        _run(kv, "r", r)
    assert not leader._arena.num_free, "a page is free, so the test checks nothing"
    runner = StepRunner({"kv": leader}, node_resources={NODE: ["kv"]})
    step = SubmoduleStep({"kv": KVStep()}, segments=[Segment("r", "main", 1)])
    step.set_ctx(StepContext(["r"], "decode", 0, False, is_preplan=True))

    runner.pre_admit(step)
    runner.clear_preplan()

    assert sorted(leader._index.pages()) == sorted(follower._index.pages()), (
        "rank 0 evicted a cached page for a stage it threw away, which the follower still holds"
    )
    assert "r" not in leader._ran_short, "a refused pre-admit counted as a grant that ran short"


def test_a_served_walks_lease_stays_when_the_next_walk_wont_run():
    ranks = leader, follower = _manager(0), _manager(1)
    for kv in ranks:
        _ingest(kv, "seed", TEXT, IMAGE, ASK)
        _run(kv, "seed", TEXT, IMAGE, ASK)
        _ingest(kv, "repeat", TEXT, IMAGE, OTHER_ASK)
    assert _follow(follower, _send(leader, "repeat"))
    for kv in ranks:
        assert _step(kv, "repeat", len(TEXT)) is None, "the cache didn't serve the text walk whole"
    leases = [kv._streams["repeat"]["main"].lease for kv in ranks]

    for kv in ranks:
        _abandon(kv, "repeat", IMAGE_WALK)
    assert leases[0] and [kv._streams["repeat"]["main"].lease for kv in ranks] == leases, (
        "giving back the lease would lose the text walk's KV"
    )


def test_under_tp_a_stream_keys_its_prompt_only():
    assert _manager(0, world_size=1)._keyed_walks["main"] == {WALK, IMAGE_WALK, "decode"}
    assert _manager(1)._keyed_walks["main"] == {WALK, IMAGE_WALK}, "every keyed walk stays keyed above one rank"


def test_above_one_rank_a_commit_caches_no_image_end():
    kv = _manager(0)
    _ingest(kv, "seed", TEXT, IMAGE, ASK)
    _run(kv, "seed", TEXT, IMAGE, ASK)

    assert set(kv._index.pages()) <= set(_pages(kv, "seed")), (
        "a copy of the page up to the image's end took a free page, though serving it would allocate on rank 0 alone"
    )


def test_above_one_rank_the_cache_opens_only_where_the_serving_config_opts_in():
    spec = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2, max_num_pages=32,
        page_size=PAGE_SIZE,
    ))
    assert "kv" not in opening_prefix_caches([spec], {NODE: 2}, checkpoint_named=True)
    apply_yaml_overrides([spec], yaml.safe_load("resources:\n  kv:\n    prefix_cache_tp: true\n"))
    assert "kv" in opening_prefix_caches([spec], {NODE: 2}, checkpoint_named=True), (
        "the serving config's prefix_cache_tp never reached the rule"
    )


def test_host_offload_or_a_pool_on_more_than_one_tp_group_is_refused_at_load(monkeypatch):
    # the real host pool pins memory, which needs a device
    monkeypatch.setattr(manager_mod, "CPUPagePool", lambda **kwargs: None)
    with pytest.raises(ValueError, match="cpu_offload_pages=4"):
        _manager(0, cpu_offload_pages=4)
    with pytest.raises(ValueError, match="on one TP group"):
        _manager(0, shared=True)


def _diverged() -> tuple[KVManager, KVManager]:
    ranks = leader, follower = _manager(0), _manager(1)
    x, y = _tokens(3 * PAGE_SIZE, seed=16), _tokens(2 * PAGE_SIZE, seed=17)
    for kv, order in ((leader, (("x", x), ("y", y))), (follower, (("y", y), ("x", x)))):
        for rid, tokens in order:
            _ingest(kv, rid, tokens)
            _run(kv, rid, tokens)
    for kv in ranks:
        _ingest(kv, "b", x[:2 * PAGE_SIZE] + _tokens(5, seed=5))
    return ranks


def test_a_follower_whose_adopt_raises_stops_the_worker(monkeypatch):
    leader, follower = _diverged()
    worker, _ = _rank(monkeypatch, tp_rank=1, inbox=[])
    worker.scheduler.next_batch = lambda **kwargs: _follow(follower, _send(leader, "b"))

    def _caught(error, *args):
        raise _Stop

    worker.pipeline._handle_error = _caught
    with pytest.raises(SystemExit, match="diverged"):
        worker.run()


def _lost_page() -> tuple[KVManager, ScheduleTPNode]:
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(2 * PAGE_SIZE, seed=30)
    _cache(ranks, "a", shared)
    for kv in ranks:
        kv.remove_request("a")
        _ingest(kv, "b", shared + _tokens(7, seed=31))
    assert len(follower._index.evict(1)) == 1
    return follower, _send(leader, "b")


def test_the_async_path_stops_a_follower_that_lost_a_page_once_n_commits():
    follower, head = _lost_page()
    head.speculative, head.spec_from_seq = True, 4
    worker = _follower_worker(follower)
    worker.lockstep.enqueue_follow(head)
    commit = _Commit()
    step_n = SimpleNamespace(
        tp_seq=4, node_name=NODE, graph_walk=WALK, node_batch=SimpleNamespace(commit_done=commit, request_ids=[]),
    )

    with pytest.raises(SystemExit, match=r"here \(1\)"):
        worker.speculator.follow(step_n)
    assert commit.waited, "a page N could still index was called lost"


@pytest.mark.parametrize("cache_open", [True, False])
def test_a_tp_leader_admits_a_yield_away_step_unplanned_as_its_followers_do(cache_open):
    engine = SimpleNamespace(prefix_cache_open=lambda node: cache_open)
    leader = make_worker(
        parallel_nodes={NODE}, parallel_leader_nodes={NODE},
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
    ).speculator

    def spec(node, yield_away):
        return SimpleNamespace(
            is_yield_away=yield_away, node_batch=SimpleNamespace(node_name=node, read_check_rids=None),
        )

    assert leader.plans_ahead(spec(NODE, True)) is not cache_open, "only the cache reads page ids"
    assert leader.plans_ahead(spec(NODE, False)), "the head is pre-planned on every rank"
    assert leader.plans_ahead(spec("other", True)), "a node with no followers keeps its pre-plan"


@pytest.mark.parametrize("rank_0_leases_first", [True, False])
def test_ranks_count_a_shared_window_page_alike_whether_a_lease_or_its_payers_release_lands_first(rank_0_leases_first):
    """Rank 0 leases c's hit, whose window pages a holds, as it sends c's step, and a's commit that rolls those pages
    out of a's window can land before or after that on rank 0, and the other way round on a follower. Once c's admit
    converts the lease, every rank counts the same pages lent and the same allotments."""
    ranks = leader, follower = _manager(0, window=24, max_num_pages=64), _manager(1, window=24, max_num_pages=64)
    shared, rest = _tokens(6 * PAGE_SIZE, seed=3), _tokens(52, seed=4)
    prompt = shared + _tokens(20, seed=5)
    runner = StepRunner({"kv": leader}, node_resources={NODE: ["kv"]})
    for kv in ranks:
        _ingest(kv, "a", shared, rest)
        _ingest(kv, "c", prompt)
    assert runner.admit_retrieve("a", NODE, WALK).ready
    follower.replay(MoveKind.RESERVE, "a", leader._allotments["a"].pages, [])
    rolls = {}
    for kv in ranks:
        kv.commit(_step(kv, "a", len(shared)), _ctx("a"))
        # a's next step rolls the window pages c leases out of a's window
        rolls[kv] = _step(kv, "a", len(rest))
    assert runner.admit_retrieve("c", NODE, WALK).ready
    follower.replay(MoveKind.RESERVE, "c", leader._allotments["c"].pages, [])

    if rank_0_leases_first:
        head = _send(leader, "c")
        lease = head.prefix_hits["wire-c"]["kv"]
        assert set(lease.lent) & set(lease.groups[1]), "rank 0 lent c none of the window pages a holds"
        leader.commit(rolls[leader], _ctx("a"))
        follower.commit(rolls[follower], _ctx("a"))
        assert _follow(follower, head)
    else:
        leader.commit(rolls[leader], _ctx("a"))
        assert _follow(follower, _send(leader, "c"))
        follower.commit(rolls[follower], _ctx("a"))
    for kv in ranks:
        _step(kv, "c", len(prompt))

    window = set(leader._streams["c"]["main"].later[0].page_indices)
    assert leader._index.page_for(leader._window_key(leader._streams["c"]["main"].chain.rooted[5], 1, 11)) in window, (
        "c's window missed the pages a cached"
    )
    assert not window & set(leader._streams["a"]["main"].later[0].page_indices), "a still holds c's window"
    assert not window & leader._streams["c"]["main"].lent, "c lends window pages no other request pays for"
    for rid in "ac":
        assert follower._streams[rid]["main"].lent == leader._streams[rid]["main"].lent, f"{rid} lends other pages"
        assert follower._allotments[rid] == leader._allotments[rid], f"{rid}'s allotment differs between the ranks"
    assert follower._unused == leader._unused, "the follower's room differs from rank 0's"
    for kv in ranks:
        kv.assert_pages_conserved()


@pytest.mark.parametrize("guaranteed", [False, True])
def test_a_hit_counts_alike_on_rank_0_and_on_the_follower_that_adopts_it(guaranteed):
    """The follower lends the pages it adopts as rank 0 lent the pages it leased, a guaranteed request's
    included."""
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    _cache(ranks, "a", shared)
    for kv in ranks:
        kv.ingest_request("b", replace(_config([prompt]), unpausable=guaranteed))
    assert StepRunner({"kv": leader}, node_resources={NODE: ["kv"]}).admit_retrieve("b", NODE, WALK).ready
    follower.replay(MoveKind.RESERVE, "b", leader._allotments["b"].pages, [])

    assert _follow(follower, _send(leader, "b"))

    assert leader._streams["b"]["main"].lent, "rank 0 lent nothing, so the test checks nothing"
    assert follower._streams["b"]["main"].lent == leader._streams["b"]["main"].lent
    assert follower._unused == leader._unused, "the follower's room differs from rank 0's after the hit"


def test_a_follower_lends_the_pages_rank_0_lent_whatever_its_own_owner_counts(monkeypatch):
    """Owner counts can differ between ranks, since above one rank a page is filed under a parent only its
    stream holds. A follower that worked out the lent pages from its own evictable set would then count other room
    than rank 0 and could refuse a grant rank 0 makes. Here a cached page has one more owner on the follower."""
    # the extra owner belongs to no stream, which the debug checks would stop on first
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    _cache(ranks, "a", shared)
    first = _pages(follower, "a")[0]
    for kv in ranks:
        kv.remove_request("a")
        kv.ingest_request("b", _config([prompt]))
    follower._arena.retain([first])
    assert StepRunner({"kv": leader}, node_resources={NODE: ["kv"]}).admit_retrieve("b", NODE, WALK).ready
    follower.replay(MoveKind.RESERVE, "b", leader._allotments["b"].pages, [])

    assert _follow(follower, _send(leader, "b"))

    assert follower._streams["b"]["main"].lent == leader._streams["b"]["main"].lent == set(), (
        "the follower lent a page rank 0 charged to b's allotment"
    )
    assert follower._unused == leader._unused, "the follower's room differs from rank 0's after the hit"


@pytest.mark.parametrize("rank", [0, 1])
def test_a_commit_above_one_rank_keeps_the_room_when_it_files_under_a_page_only_the_index_holds(rank):
    """z prefills beside x, so it finds nothing cached and loses x's keys at commit. Its next page goes in with no
    parent, so the cached pages x left stay evictable and the room the allotments count doesn't drop."""
    kv = _manager(rank, max_num_pages=12)
    x = _tokens(2 * PAGE_SIZE + 1, seed=50)
    z = x[:2 * PAGE_SIZE] + _tokens(PAGE_SIZE + 3, seed=51)
    _ingest(kv, "x", x)
    _ingest(kv, "z", z)
    steps = _step(kv, "x", len(x)), _step(kv, "z", len(z))
    kv.commit(steps[0], _ctx("x"))
    kv.remove_request("x")
    room = kv._room()

    kv.commit(steps[1], _ctx("z"))

    assert kv._room() == room, "the commit pinned cached pages the room counted"


def test_a_chunked_row_starts_past_the_prefix_rank_0_measured_on_every_rank():
    """Rank 0 leases the cached prefix a chunked row's chunks start past as it
    sends the step. The lease goes out with the step, so a
    follower's first chunk finds the same prefix; its own probe would find
    none, and its chunk would not start where rank 0's does."""
    from mstar.engine.chunking import ChunkRange

    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(40, seed=1)
    _cache(ranks, "b0", shared)
    for kv in ranks:
        _ingest(kv, "b", prompt)

    head = _send(leader, "b")  # rank 0 leases the prefix as it sends the step
    start = leader.resolve_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=len(prompt))).tokens
    assert start == 5 * PAGE_SIZE
    assert _follow(follower, head)

    class _Prompt(_Submodule):
        def prepare_inputs(self, **kwargs):
            return ARNodeInputs(input_ids=torch.tensor(prompt), input_seq_len=len(prompt))

    engine = _engine(follower)
    engine._submodules[NODE] = make_mgmt(_Prompt())
    batch = SimpleNamespace(
        node_name=NODE, per_request_info_wrapped={"b": None}, per_request_input_tensors={},
        final_stream_rids=set(), per_request_input_metadata={}, first_chunk_rids=set(),
    )
    end = start + PAGE_SIZE
    first = engine._prepare_chunk(batch, "b", WALK, ChunkRange(start, end, len(prompt), end))

    assert (first.chunk_start, first.input_seq_len) == (start, PAGE_SIZE)


def test_a_follower_keys_a_dropped_request_as_rank_0_did_before_it_takes_rank_0s_hit():
    """Rank 0 starts a dropped request's rebuild before it sends the rebuild's first row, and the rebuild keys the
    request as a prompt of its prompt and its reply. The hit rank 0 sends with the row is what the cache holds under
    those keys. A follower that took the hit before it started the rebuild looked it up under the request's old keys,
    found only the prompt's pages, and stopped."""
    ranks = leader, follower = _manager(0), _manager(1)
    prompt, reply = _tokens(2 * PAGE_SIZE + 3, seed=0), _tokens(2 * PAGE_SIZE, seed=1)
    # an earlier request with the same prompt and reply, whose rebuild cached its pages under the rebuild's keys
    _cache(ranks, "a", prompt + reply[:-1])
    for kv in ranks:
        kv.ingest_request("v", replace(_config([prompt], decode=len(reply)), rebuildable=True))
        _run(kv, "v", prompt)
        _write(kv, "v", len(reply) - 1, "decode")
        kv.drop("v", len(prompt) + len(reply) - 1)
    assert StepRunner({"kv": leader}, node_resources={NODE: ["kv"]}).admit_retrieve("v", NODE, WALK).ready
    follower.replay(MoveKind.RESERVE, "v", leader._allotments["v"].pages, [])
    # as rank 0's readiness check starts the rebuild
    leader.prepare_rebuild("v", Rebuild(array("i", reply)))
    lead = _engine(leader)
    head = ScheduleTPNode(
        NODE, WALK, ["v"], prefix_hits=lead.decide_cached_prefix(_batch("v")), page_trails=lead.page_trails(NODE),
    )
    assert len(leader._streams["v"]["main"].lease.groups[0]) > len(prompt) // PAGE_SIZE, (
        "rank 0's hit is the prompt's alone"
    )
    one = _Rank(follower, _engine(follower))
    one.rebuilds._kept[NODE]["v"] = _kept(*reply, prompt=prompt)
    one.lockstep.enqueue_follow(head)

    batch = one.scheduler._try_schedule_tp_follow()

    assert batch.rebuild_rids == {"v"}, "the follower took v's row as one of its own"
    assert follower._streams["v"]["main"].lease == leader._streams["v"]["main"].lease, (
        "the follower holds other cached pages for v than rank 0"
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_no_rank_above_one_marks_a_request_undroppable_after_admission_on_its_own(rank):
    """No move tells the other ranks that a request lost its log after admission, so a rank that marked it alone
    would reserve host room or guarantee it where the others don't, and their counts would part."""
    kv = _manager(rank)
    _ingest(kv, "r", _tokens(PAGE_SIZE, 1))

    with pytest.raises(AssertionError, match="above one rank"):
        kv.refuse_drops("r")

def _write(kv: KVManager, rid: str, length: int, walk: str) -> None:
    step, ctx = KVStep(segments=(Segment(rid, "main", length),)), _ctx(rid, walk)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
