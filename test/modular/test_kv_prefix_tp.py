"""Prefix reuse under TP: rank 0 decides each hit, and the other ranks take the same pages."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT, WALK
from test_kv_prefix_layout_lease import ASK, IMAGE, IMAGE_WALK, OTHER_ASK, TEXT, _config, _walks
from test_kv_prefix_lease import _StubTransfer
from test_tp_async_teardown_order import _rank, _Stop

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.engine.engine import Engine
from mstar.engine.resources import StepRunner, apply_yaml_overrides
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.worker import Worker

WALKS = {"main": KeyedWalks((WALK, IMAGE_WALK), ("decode",))}


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(
    rank: int, max_num_pages: int = 32, opt_in: bool = True,
    world_size: int = 2, shared: bool = False, **overrides,
) -> KVManager:
    cfg = dict(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2,
        max_num_pages=max_num_pages, page_size=PAGE_SIZE, prefix_cache_tp=opt_in,
    )
    cfg.update(overrides)
    kv = KVManager(
        cfg=PagedKVConfig(**cfg),
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
    engine = Engine.__new__(Engine)
    engine._resources = {"kv": kv}
    engine._runner = StepRunner(engine._resources, node_resources={NODE: ["kv"]})
    engine._submodules = {NODE: SimpleNamespace(submodule=_Submodule())}
    engine._keyed_walks = {NODE: {WALK, IMAGE_WALK}}
    engine._prefix_model = "_Model"
    engine._graph_runtime = SimpleNamespace(
        get_rid_string=lambda rid: f"wire-{rid}", get_rid_handle=lambda wire_rid: wire_rid.removeprefix("wire-"),
    )
    return engine


def _batch(rid: str, walk: str = WALK) -> SimpleNamespace:
    return SimpleNamespace(
        node_name=NODE, request_ids=(rid,), step_context=_ctx(rid, walk),
        per_request_info_wrapped={rid: None}, cached_rids=set(),
    )


def _send(leader: KVManager, rid: str) -> ScheduleTPNode:
    engine = _engine(leader)
    return ScheduleTPNode(
        NODE, WALK, [f"wire-{rid}"],
        prefix_hits=engine.decide_cached_prefix(_batch(rid)), page_trails=engine.page_trails(NODE),
    )


def _follow(follower: KVManager, head: ScheduleTPNode, caught_up: bool = True) -> bool:
    engine = _engine(follower)
    scheduler = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine), parallel_leader_nodes=set(),
    )
    scheduler.register_tp_follow(head)
    return scheduler.settle_tp_follow_delta(caught_up)


def _abandon(kv: KVManager, rid: str, walk: str = WALK) -> None:
    engine = _engine(kv)
    worker = SimpleNamespace(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        _graph_runtime=SimpleNamespace(commit_speculation=lambda *args, **kwargs: None),
    )
    speculation = SimpleNamespace(
        scheduled_batch=SimpleNamespace(node_name=NODE), node_batch=_batch(rid, walk), spec_id=0,
        consumed_streaming_edges={},
    )
    Worker._settle_speculation(worker, speculation, success=False)


def _follower_worker(follower: KVManager) -> Worker:
    engine = _engine(follower)
    worker = Worker.__new__(Worker)
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    worker.scheduler = MicroScheduler(engine_manager=worker.engine_manager, parallel_leader_nodes=set())
    worker._tp_nospec = set()
    return worker


def test_a_follower_takes_rank_0s_hit_and_never_probes_itself():
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    _cache(ranks, "a", shared)
    for kv in ranks:
        _ingest(kv, "b", prompt)

    assert follower.resolve_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=len(prompt))).tokens == 0
    assert _follow(follower, _send(leader, "b"))

    for kv in ranks:
        _run(kv, "b", prompt)
    assert _pages(leader, "b") == _pages(follower, "b")
    assert _pages(leader, "b")[:5] == _pages(leader, "a")[:5], "b missed the prefix a cached"


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
    worker.scheduler.register_tp_follow(head)
    assert worker.scheduler.settle_tp_follow_delta(caught_up=False)
    worker._close_tp_follow_step(SimpleNamespace(tp_seq=4))
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
    worker.scheduler.register_tp_follow(head)
    assert worker.scheduler.settle_tp_follow_delta(caught_up=True)
    worker._close_tp_follow_step(SimpleNamespace(tp_seq=4))
    _abandon(leader, "h")
    for kv in ranks:
        kv.remove_request("z")
        kv.remove_request("h")

    big = _tokens(9 * PAGE_SIZE, seed=43)
    for kv in ranks:
        _ingest(kv, "big", big)
        _run(kv, "big", big)

    assert _pages(leader, "big") == _pages(follower, "big"), "the ranks linked z's page differently and evicted apart"


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


def test_above_one_rank_the_cache_stays_off_unless_the_deployment_opts_in():
    assert _manager(0, opt_in=False)._index is None
    assert _manager(0)._index is not None


def test_the_serving_config_turns_on_prefix_cache_tp():
    spec = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2, max_num_pages=32,
        page_size=PAGE_SIZE,
    ))
    apply_yaml_overrides([spec], yaml.safe_load("resources:\n  kv:\n    prefix_cache_tp: true\n"))
    kv = KVManager(
        cfg=spec.config, name="kv", joint_comm_group=SimpleNamespace(rank=0, world_size=2),
        transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert kv.enable_prefix_cache(ROOT, WALKS), "the serving config's prefix_cache_tp never reached the pool"


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
    worker.scheduler.get_next_batch = lambda request_state, **kwargs: _follow(follower, _send(leader, "b"))

    def _caught(error, *args):
        raise _Stop

    worker._handle_main_loop_error = _caught
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
    worker.scheduler.register_tp_follow(head)
    step_n = SimpleNamespace(
        tp_seq=4, node_name=NODE, graph_walk=WALK, node_batch=SimpleNamespace(commit_done=threading.Event()),
    )

    assert worker._try_follow_speculation(step_n) is None, "a page N could still index was called lost"
    step_n.node_batch.commit_done.set()
    with pytest.raises(SystemExit, match=r"here \(1\)"):
        worker._try_follow_speculation(step_n)


@pytest.mark.parametrize("cache_open", [True, False])
def test_a_tp_leader_admits_a_yield_away_step_unplanned_as_its_followers_do(cache_open):
    leader = SimpleNamespace(parallel_nodes={NODE}, _prefix_cache_open=lambda node: cache_open)

    def spec(node, yield_away):
        return SimpleNamespace(is_yield_away=yield_away, node_batch=SimpleNamespace(node_name=node))

    assert Worker._plans_ahead(leader, spec(NODE, True)) is not cache_open, "only the cache reads page ids"
    assert Worker._plans_ahead(leader, spec(NODE, False)), "the head is pre-planned on every rank"
    assert Worker._plans_ahead(leader, spec("other", True)), "a node with no followers keeps its pre-plan"
