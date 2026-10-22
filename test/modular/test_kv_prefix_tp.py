"""Prefix reuse under TP: rank 0 decides each hit, and the other ranks take the same pages.

Every rank indexes its own copy of the cache, and their page tables have to stay
identical (a sharded prefill gathers cache slots across ranks). So a follower never
looks a prompt up itself. Rank 0 leases the hit as it sends the step. The follower adopts
that lease, and waits until it has also committed the step that wrote those pages. A hit
is stamped when its step commits. Every rank commits in step order, so the indexes age and
evict alike however each rank's probes interleave with its commits.

Above one rank the cache keys no generated tokens. Each rank reads the sampled ids back at
its own step, so the ranks would index those pages at different steps. It neither caches
nor serves the page up to an image's end, since serving it allocates on rank 0 alone. Host
offload and a pool shared across worker instances move pages outside the schedule, so they
are refused at load. Under ``MSTAR_KV_DEBUG_ASSERTS`` a follower checks before each step
that it has allocated the pages rank 0 had. A divergence found by that check or by an adopt
stops the worker.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT, WALK
from test_kv_prefix_layout_lease import ASK, IMAGE, IMAGE_WALK, OTHER_ASK, TEXT, _config, _walks
from test_kv_prefix_lease import _StubTransfer
from test_tp_async_teardown_order import _rank, _Stop

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.wire import decode, encode
from mstar.engine.engine import Engine
from mstar.engine.resources import StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager, KVSequenceInfo, PublishedKVInfo
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule
from mstar.utils.ipc_format import ScheduleTPNode, WorkerMessage, WorkerMessageType
from mstar.worker.micro_scheduler import MicroScheduler, TPDivergence

WALKS = {"main": (WALK, "decode", IMAGE_WALK)}


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
    """Give ``kv`` the keys and layout the preprocess worker sends for a prompt of ``parts``."""
    kv.ingest_request(rid, _config(list(parts)))


def _ctx(rid: str, walk: str = WALK) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)


def _step(kv: KVManager, rid: str, length: int, walk: str = WALK) -> KVStep | None:
    """Prepare one walk as the engine does, then admit and plan what the cache doesn't hold.
    None when the cache serves the walk whole."""
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
    """Run and commit every walk of a prompt of ``parts``."""
    for walk, length in _walks(list(parts)):
        step = _step(kv, rid, length, walk)
        if step is not None:
            kv.commit(step, _ctx(rid, walk))


def _cache(ranks, rid: str, tokens: list[int]) -> None:
    """Every rank prefills ``tokens``, so its index holds their whole pages."""
    for kv in ranks:
        _ingest(kv, rid, tokens)
        _run(kv, rid, tokens)


def _pages(kv: KVManager, rid: str) -> list[int]:
    return kv._streams[rid]["main"].page_indices


class _Submodule:
    """Cuts a prompt as an AR submodule does."""

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


def _batch(rid: str) -> SimpleNamespace:
    return SimpleNamespace(
        node_name=NODE, request_ids=(rid,), step_context=_ctx(rid),
        per_request_info_wrapped={rid: None}, cached_rids=set(),
    )


def _send(leader: KVManager, rid: str) -> ScheduleTPNode:
    """The step rank 0 sends for ``rid``'s prefill, carrying the hit rank 0 leased and its page trails."""
    engine = _engine(leader)
    return ScheduleTPNode(
        NODE, WALK, [f"wire-{rid}"],
        prefix_hits=engine.decide_cached_prefix(_batch(rid)), page_trails=engine.page_trails(NODE),
    )


def _follow(follower: KVManager, head: ScheduleTPNode, caught_up: bool = True) -> bool:
    """Settle ``head`` at the front of the follower's queue, as before building the step. False
    while the follower can't yet take rank 0's hit."""
    engine = _engine(follower)
    scheduler = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine), parallel_leader_nodes=set(),
    )
    scheduler.register_tp_follow(head)
    return scheduler.settle_tp_follow_delta(caught_up)


def _wire(head: ScheduleTPNode) -> ScheduleTPNode:
    return decode(encode(WorkerMessage(WorkerMessageType.SCHEDULE_TP, head))).body


# ── a follower takes rank 0's hit ───────────────────────────────────────


def test_a_follower_takes_rank_0s_hit_and_never_probes_itself():
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    _cache(ranks, "a", shared)
    for kv in ranks:
        _ingest(kv, "b", prompt)

    head = _send(leader, "b")
    # the follower's own index holds the pages, but only rank 0 decides a hit
    assert follower.resolve_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=len(prompt))).tokens == 0
    assert _follow(follower, head)
    # the follower adopts the hit once, so settling it again takes no second reference
    assert _follow(follower, head)

    for kv in ranks:
        _run(kv, "b", prompt)
    assert _pages(leader, "b") == _pages(follower, "b")
    assert _pages(leader, "b")[:5] == _pages(leader, "a")[:5], "b missed the prefix a cached"
    for kv in ranks:
        kv.assert_pages_conserved()


def test_a_follower_waits_until_its_own_commit_indexes_the_pages():
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(4 * PAGE_SIZE + 5, seed=2)
    prompt = shared[:4 * PAGE_SIZE] + _tokens(9, seed=3)
    steps = {}
    for kv in ranks:
        _ingest(kv, "a", shared)
        _ingest(kv, "b", prompt)
        steps[kv] = _step(kv, "a", len(shared))
    leader.commit(steps[leader], _ctx("a"))

    head = _send(leader, "b")
    # rank 0 has committed a, and the follower has only admitted it
    assert not _follow(follower, head, caught_up=False)
    follower.commit(steps[follower], _ctx("a"))
    assert _follow(follower, head)

    for kv in ranks:
        _run(kv, "b", prompt)
    assert _pages(leader, "b") == _pages(follower, "b")


def test_ranks_evict_alike_however_their_probes_and_commits_interleave():
    """Rank 0 leases two hits before z commits, and the follower adopts them after. hx then
    runs, and hy's request leaves before it runs. A stamp taken when either rank looks a
    hit up would age the two indexes differently, and they would evict different pages."""
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

    # hx committed last, so a request that needs most of the pool evicts y's and z's pages first
    big = _tokens(9 * PAGE_SIZE, seed=11)
    for kv in ranks:
        _ingest(kv, "big", big)
        _run(kv, "big", big)
    assert _pages(leader, "big") == _pages(follower, "big"), "the ranks evicted different pages, or in another order"
    for kv in ranks:
        assert not set(x_pages[kv]) & set(_pages(kv, "big")), "big evicted the pages hx ran on before older ones"


def test_rank_0_prepares_the_length_it_decided_though_a_commit_since_holds_more():
    """Rank 0 decides step N+1's hits while step N still runs, and prepares N+1 after N
    commits. The followers were told b missed, so b runs whole on rank 0 too, though a's
    commit has since indexed the prefix they share."""
    leader = _manager(0)
    a = _tokens(4 * PAGE_SIZE + 5, seed=2)
    b = a[:4 * PAGE_SIZE] + _tokens(9, seed=3)
    _ingest(leader, "a", a)
    _ingest(leader, "b", b)
    a_step = _step(leader, "a", len(a))
    assert _send(leader, "b").prefix_hits == {}
    leader.commit(a_step, _ctx("a"))

    inputs = ARNodeInputs(input_ids=torch.tensor(b), input_seq_len=len(b))
    assert _engine(leader)._skip_cached_prefix(_batch("b"), "b", inputs).input_seq_len == len(b), (
        "rank 0 skipped a prefix it told the followers b missed"
    )


def test_a_lease_taken_around_a_commit_leaves_the_ranks_indexes_alike():
    """z shares a's prefix but was sent while a ran, so it missed and wrote a copy of its own.
    a has left when z commits. Rank 0 has by then leased the prefix for h, and the follower
    adopts the lease after it commits z. If z's commit chained its next page under a's only while
    a lease held the shared prefix, z's page would hang under a's on rank 0 only."""
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(4 * PAGE_SIZE, seed=19)
    prompts = {"a": shared + _tokens(PAGE_SIZE, seed=20), "z": shared + _tokens(2 * PAGE_SIZE, seed=21),
               "h": shared + _tokens(PAGE_SIZE, seed=22)}
    a_steps = {}
    for kv in ranks:
        for rid, tokens in prompts.items():
            _ingest(kv, rid, tokens)
        a_steps[kv] = _step(kv, "a", len(prompts["a"]))
    assert _send(leader, "z").prefix_hits == {}, "z was decided after a committed"
    z_steps = {}
    for kv in ranks:
        kv.commit(a_steps[kv], _ctx("a"))
        z_steps[kv] = _step(kv, "z", len(prompts["z"]))
        kv.remove_request("a")

    head = _send(leader, "h")
    for kv in ranks:
        kv.commit(z_steps[kv], _ctx("z"))
    assert head.prefix_hits and _follow(follower, head)

    parents = [[kv._index._parent[page] for page in sorted(kv._index.pages())] for kv in ranks]
    assert parents[0] == parents[1], "z's page hangs under a's on rank 0 only"


def test_a_keyed_miss_answers_zero_so_another_resource_cannot_skip_it():
    kv = _manager(0)
    tokens = _tokens(3 * PAGE_SIZE + 2, seed=11)
    _ingest(kv, "a", tokens)
    assert _send(kv, "a").prefix_hits == {}

    prefix = kv.resolve_cached_prefix("a", NODE, WALK, ARNodeInputs(input_seq_len=len(tokens)))
    assert prefix is not None and prefix.tokens == 0, "a miss with no answer lets another resource's hit cut this walk"


def test_a_readiness_check_stamps_no_cached_page_above_one_rank():
    """A step refused at admit still publishes its empty stream. Rank 0 then checks the
    request's readiness on every scan and a follower once, so a lookup there would age
    rank 0's index alone."""
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(3 * PAGE_SIZE, seed=23)
    _cache(ranks, "old", shared + _tokens(4, seed=24))
    published = PublishedKVInfo(info={rank: {"main": KVSequenceInfo(0, None)} for rank in (0, 1)}, world_size=2)
    for kv in ranks:
        _ingest(kv, "b", shared + _tokens(20, seed=25))

    for kv, scans in ((leader, 3), (follower, 1)):
        for _ in range(scans):
            assert kv.admit_retrieve("b", NODE, WALK, published).ready
    assert leader._index._clock == follower._index._clock, "rank 0's scans stamped the pages b matches"


def test_rank_0s_hits_cross_the_wire():
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(3 * PAGE_SIZE, seed=12)
    _cache(ranks, "a", shared)
    for kv in ranks:
        _ingest(kv, "b", shared[:2 * PAGE_SIZE] + _tokens(7, seed=13))

    assert _follow(follower, _wire(_send(leader, "b")))
    assert follower._streams["b"]["main"].lease == leader._streams["b"]["main"].lease


# ── what the cache keys above one rank ──────────────────────────────────


def test_under_tp_a_stream_keys_its_prompt_only():
    assert _manager(1)._keyed_walks["main"] == _manager(0, world_size=1)._keyed_walks["main"] - {"decode"}


def test_above_one_rank_a_commit_caches_no_image_end():
    kv = _manager(0)
    _ingest(kv, "seed", TEXT, IMAGE, ASK)
    _run(kv, "seed", TEXT, IMAGE, ASK)

    assert set(kv._index.pages()) <= set(_pages(kv, "seed")), (
        "a copy of the page up to the image's end took a free page, though nothing above one rank serves it"
    )


def test_above_one_rank_a_cached_image_end_is_never_served():
    ranks = leader, follower = _manager(0), _manager(1)
    for kv in ranks:
        # seed at world size 1, since above one rank a commit caches no image end
        kv._world_size = 1
        _ingest(kv, "seed", TEXT, IMAGE, ASK)
        _run(kv, "seed", TEXT, IMAGE, ASK)
        kv._world_size = 2
        assert set(kv._index.pages()) - set(_pages(kv, "seed")), "the seed cached no image end to serve"
        _ingest(kv, "repeat", TEXT, IMAGE, OTHER_ASK)

    assert _follow(follower, _send(leader, "repeat")), "the follower can't adopt what rank 0 leased"
    for kv in ranks:
        _run(kv, "repeat", TEXT, IMAGE, OTHER_ASK)
    assert _pages(leader, "repeat") == _pages(follower, "repeat")


# ── what is refused at load ─────────────────────────────────────────────


def test_above_one_rank_the_cache_stays_off_unless_the_deployment_opts_in():
    assert _manager(0, opt_in=False)._index is None
    assert _manager(0)._index is not None


def test_host_offload_is_refused_at_load(monkeypatch):
    # the real host pool pins memory, which needs a device
    monkeypatch.setattr(manager_mod, "CPUPagePool", lambda **kwargs: None)
    with pytest.raises(ValueError, match="cpu_offload_pages=4"):
        _manager(0, cpu_offload_pages=4)
    assert _manager(0, opt_in=False, cpu_offload_pages=4)._index is None, (
        "a TP pool that leaves the cache off may still offload"
    )


def test_a_pool_on_more_than_one_tp_group_is_refused_at_load():
    with pytest.raises(ValueError, match="on one TP group"):
        _manager(0, shared=True)
    assert _manager(0, shared=True, world_size=1)._index is not None, (
        "one rank has nobody to agree with, so sharing the pool is its own business"
    )


# ── a follower that diverged from rank 0 ────────────────────────────────


def _diverged() -> tuple[KVManager, KVManager]:
    """Two ranks whose indexes hold x's prefix on different pages, since the follower ran y
    first. Both have ingested b, which shares x's prefix."""
    ranks = leader, follower = _manager(0), _manager(1)
    x, y = _tokens(3 * PAGE_SIZE, seed=16), _tokens(2 * PAGE_SIZE, seed=17)
    for kv, order in ((leader, (("x", x), ("y", y))), (follower, (("y", y), ("x", x)))):
        for rid, tokens in order:
            _ingest(kv, rid, tokens)
            _run(kv, rid, tokens)
    for kv in ranks:
        _ingest(kv, "b", x[:2 * PAGE_SIZE] + _tokens(5, seed=5))
    return ranks


def test_pages_that_differ_from_rank_0s_are_a_loud_error():
    leader, follower = _diverged()
    pages = _send(leader, "b").prefix_hits["wire-b"]["kv"]
    with pytest.raises(RuntimeError, match="diverged"):
        follower.adopt_cached_prefix("b", NODE, WALK, pages, caught_up=True)


def test_a_follower_whose_adopt_raises_stops_the_worker(monkeypatch):
    """Failing the step's requests would leave the other ranks waiting in the step's collective for good."""
    leader, follower = _diverged()
    worker, _ = _rank(monkeypatch, tp_rank=1, inbox=[])
    worker.scheduler.get_next_batch = lambda request_state, **kwargs: _follow(follower, _send(leader, "b"))

    def _caught(error, *args):
        # the real handler logs the error and runs the next pass
        raise _Stop

    worker._handle_main_loop_error = _caught
    with pytest.raises(TPDivergence, match="diverged"):
        worker.run()


def test_a_follower_that_allocated_other_pages_than_rank_0_is_a_loud_error(monkeypatch):
    """Both ranks have allocated the same number of times, but on different pages, and nothing
    fails until a sharded prefill gathers the wrong slots. So the follower checks before each
    step, and before adopting the step's hit, since a rank out of step may wait on that hit for
    good."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    ranks = leader, follower = _manager(0), _manager(1)
    x, y = _tokens(3 * PAGE_SIZE, seed=16), _tokens(2 * PAGE_SIZE, seed=17)
    for kv in ranks:
        _ingest(kv, "x", x)
        _ingest(kv, "y", y)
        _ingest(kv, "b", x[:2 * PAGE_SIZE] + _tokens(5, seed=5))
    _run(leader, "x", x)
    _run(leader, "y", y)
    _run(follower, "y", y)
    # admitted and not committed, so this rank's index doesn't hold x yet
    _step(follower, "x", len(x))
    assert leader.page_trail()[0] == follower.page_trail()[0]

    in_step = ScheduleTPNode(NODE, WALK, ["wire-b"], page_trails={"kv": follower.page_trail()})
    assert _follow(follower, in_step), "a follower in step with rank 0 was held back"
    head = _send(leader, "b")
    with pytest.raises(TPDivergence, match=rf"KV kv: .*{NODE}/{WALK}, tp seq {head.spec_seq}"):
        _follow(follower, head)


def test_with_the_debug_flag_off_no_trail_is_kept_or_sent(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    leader = _manager(0)
    tokens = _tokens(3 * PAGE_SIZE, seed=18)
    _ingest(leader, "a", tokens)
    _run(leader, "a", tokens)

    assert leader._arena.trail is None
    assert _send(leader, "a").page_trails == {}


def test_a_pool_without_the_tp_cache_keeps_no_trail(monkeypatch):
    """A pool without the TP cache may offload to the host. A reload allocates when rank 0 checks
    readiness and when a follower replays the move, so the two trails would not match."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    assert _manager(0)._arena.trail is not None
    assert _manager(0, opt_in=False)._arena.trail is None


def test_rank_0s_trails_cross_the_wire(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    ranks = leader, follower = _manager(0), _manager(1)
    _cache(ranks, "a", _tokens(3 * PAGE_SIZE, seed=14))
    for kv in ranks:
        _ingest(kv, "b", _tokens(2 * PAGE_SIZE, seed=15))

    head = _wire(_send(leader, "b"))
    assert head.page_trails, "rank 0 sent no trail to compare"
    assert _follow(follower, head), "a trail that came back a list would never compare equal"
