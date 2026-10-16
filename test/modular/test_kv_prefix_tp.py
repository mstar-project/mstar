"""Prefix reuse under TP: rank 0 decides each hit, and the other ranks take the same pages.

Every rank indexes its own copy of the cache, and their page tables have to stay
identical (a sharded DSA prefill gathers cache slots across ranks). So a follower never
looks a prompt up itself: it adopts rank 0's hit, and waits while its own commit of the
step that wrote those pages trails rank 0's. A hit is stamped when it commits, which
every rank does in step order, so the indexes age and evict alike however each rank's
probes interleave with its commits.

Only rank 0 runs the admission gate, and it admits requests whose steps are sent later or
not next. A lease or an eviction order the gate took for one would exist on rank 0 alone,
and the next eviction would take a different page on each rank. So above one rank the gate
only counts: a request is admitted on its whole prompt, and its hit is leased when rank 0
sends its step. What moves pages outside the schedule (host offload, a pool shared across
worker instances) is refused at load, and under ``MSTAR_KV_DEBUG_ASSERTS`` a follower
checks before each step that it has allocated the pages rank 0 had.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402
import torch  # noqa: E402

import mstar.communication.wire_types  # noqa: E402,F401  (registers the tags)
from mstar.communication.wire import decode, encode  # noqa: E402
from mstar.engine.engine import Engine  # noqa: E402
from mstar.engine.resources import StepRunner  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.keys import chain  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager, KVSequenceInfo, PublishedKVInfo  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402
from mstar.model.base import PrefixStream  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    PrefixHit,
    ScheduleTPNode,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return True

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


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
        name="kv", joint_comm_group=types.SimpleNamespace(rank=rank, world_size=world_size),
        transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
        needs_remote_transfer=shared,
    )
    assert kv.enable_prefix_cache(ROOT) == opt_in
    return kv


def test_above_one_rank_the_cache_stays_off_unless_the_deployment_opts_in():
    assert _manager(0, opt_in=False)._index is None
    assert _manager(0)._index is not None


def _tokens(n: int, seed: int) -> list[int]:
    return [(seed * 7919 + i * 31) % 1000 for i in range(n)]


def _ingest(kv: KVManager, rid: str, tokens: list[int], max_tokens: int | None = None) -> None:
    """The keys the preprocess worker sends for ``tokens``. With ``max_tokens``, also the
    count the conductor stamps, which puts the request through the admission gate."""
    whole = len(tokens) // PAGE_SIZE
    pages = [tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)]
    counted = {} if max_tokens is None else dict(
        max_tokens=max_tokens, prompt_slots={"main": len(tokens)}, decode_labels=["main"],
    )
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain(pages)}, prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        **counted,
    ))


def _ctx(rid: str) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)


def _admit(kv: KVManager, rid: str, span: int) -> KVStep:
    step = KVStep(segments=(Segment(rid, "main", span),))
    assert kv.admit(step, _ctx(rid)).ok
    kv.plan(step, _ctx(rid))
    return step


def _run(kv: KVManager, rid: str, tokens: list[int], matched: int = 0) -> None:
    """The rest of one prefill step past a hit: admit, plan, commit."""
    kv.apply_cached_prefix(rid, NODE, WALK, None, matched)
    kv.commit(_admit(kv, rid, len(tokens) - matched), _ctx(rid))


def _pages(kv: KVManager, rid: str) -> list[int]:
    return kv._streams[rid]["main"].page_indices


class _Submodule:
    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        return ARNodeSubmodule.split_inputs(self, graph_walk, fwd_info, inputs, start, end)


def _engine(kv: KVManager, **more: KVManager) -> Engine:
    """An engine over ``kv`` and any pools beside it. Rank 0's engine decides a step's hits
    as it sends the step, then prepares it."""
    engine = Engine.__new__(Engine)
    engine._resources = {"kv": kv, **more}
    engine._runner = StepRunner(engine._resources, node_resources={NODE: list(engine._resources)})
    engine._submodules = {NODE: SimpleNamespace(submodule=_Submodule())}
    engine._keyed_walks = {NODE: {WALK}}
    engine._prefix_model = "_Model"
    engine._graph_runtime = SimpleNamespace(
        get_rid_string=lambda rid: f"wire-{rid}", get_rid_handle=lambda wire_rid: None,
    )
    return engine


def _batch(rid: str):
    return SimpleNamespace(
        node_name=NODE, graph_walk=WALK, request_ids=(rid,), step_context=_ctx(rid),
        per_request_info_wrapped={rid: None}, cached_prefix=None,
    )


def test_a_follower_takes_rank_0s_hit_and_never_probes_itself():
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    for kv in ranks:
        _ingest(kv, "a", shared)
        _run(kv, "a", shared)
        _ingest(kv, "b", prompt)

    matched = leader.resolve_cached_prefix("b", NODE, WALK)
    assert matched == 5 * PAGE_SIZE
    # its own index holds the pages, but only rank 0's word counts
    assert follower.resolve_cached_prefix("b", NODE, WALK) == 0
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    # adopted once: asking again takes no second reference
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert follower.resolve_cached_prefix("b", NODE, WALK) == matched

    for kv in ranks:
        _run(kv, "b", prompt, matched)
    assert _pages(leader, "b") == _pages(follower, "b")
    assert _pages(leader, "b")[:5] == _pages(leader, "a")[:5]
    for kv in ranks:
        kv.assert_pages_conserved()


def test_a_follower_waits_until_its_own_commit_indexes_the_pages():
    leader, follower = _manager(0), _manager(1)
    shared = _tokens(4 * PAGE_SIZE + 5, seed=2)
    prompt = shared[:4 * PAGE_SIZE] + _tokens(9, seed=3)
    steps = {}
    for kv in (leader, follower):
        _ingest(kv, "a", shared)
        steps[kv] = _admit(kv, "a", len(shared))
    leader.commit(steps[leader], _ctx("a"))
    for kv in (leader, follower):
        _ingest(kv, "b", prompt)

    matched = leader.resolve_cached_prefix("b", NODE, WALK)
    assert matched == 4 * PAGE_SIZE
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)
    # rank 0 committed "a" first; this rank has admitted it but not committed it
    assert not follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    follower.commit(steps[follower], _ctx("a"))
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert follower.resolve_cached_prefix("b", NODE, WALK) == matched


def test_pages_that_differ_from_rank_0s_are_a_loud_error():
    leader, follower = _manager(0), _manager(1)
    shared = _tokens(3 * PAGE_SIZE + 1, seed=4)
    prompt = shared[:3 * PAGE_SIZE] + _tokens(5, seed=5)
    for kv in (leader, follower):
        _ingest(kv, "a", shared)
        _run(kv, "a", shared)
        _ingest(kv, "b", prompt)
    matched = leader.resolve_cached_prefix("b", NODE, WALK)
    pages = [page + 1 for page in leader.cached_prefix_pages("b", NODE, WALK, matched)]
    with pytest.raises(RuntimeError, match="diverged"):
        follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)


def test_ranks_evict_alike_however_their_probes_and_commits_interleave():
    """Rank 0 probes a hit before another request commits, rank 1 after: under a stamp
    taken at lookup their indexes would age differently and evict different pages."""
    ranks = leader, follower = _manager(0, max_num_pages=12), _manager(1, max_num_pages=12)
    x, y, z = (_tokens(2 * PAGE_SIZE + 1, seed=s) for s in (6, 7, 8))
    hit = x[:2 * PAGE_SIZE] + _tokens(3, seed=9)
    x_pages = {}
    for kv in ranks:
        for rid, tokens in (("x", x), ("y", y)):
            _ingest(kv, rid, tokens)
            _run(kv, rid, tokens)
            x_pages.setdefault(kv, _pages(kv, rid)[:2])
            kv.remove_request(rid)
        _ingest(kv, "z", z)
        _ingest(kv, "hit", hit)

    z_steps = {kv: _admit(kv, "z", len(z)) for kv in ranks}
    matched = leader.resolve_cached_prefix("hit", NODE, WALK)  # before z commits
    leader.commit(z_steps[leader], _ctx("z"))
    follower.commit(z_steps[follower], _ctx("z"))  # after z commits
    pages = leader.cached_prefix_pages("hit", NODE, WALK, matched)
    assert follower.adopt_cached_prefix("hit", NODE, WALK, matched, pages)
    for kv in ranks:
        _run(kv, "hit", hit, matched)
        kv.remove_request("hit")
        kv.remove_request("z")

    # x was hit last, so a request that needs most of the pool evicts y and z first
    for kv in ranks:
        _ingest(kv, "big", _tokens(9 * PAGE_SIZE, seed=10))
        _run(kv, "big", _tokens(9 * PAGE_SIZE, seed=10))
    assert sorted(leader._index.pages()) == sorted(follower._index.pages())
    assert _pages(leader, "big") == _pages(follower, "big")
    for kv in ranks:
        assert set(x_pages[kv]) <= set(kv._index.pages())


def test_rank_0_prepares_the_length_it_decided_though_a_commit_since_holds_more():
    """Rank 0 decides step N+1's hits while step N still runs, and prepares N+1 after N
    commits. The followers were told b missed, so b runs whole on rank 0 too, though a's
    commit has since indexed the prefix they share."""
    leader = _manager(0)
    engine = _engine(leader)
    a = _tokens(4 * PAGE_SIZE + 5, seed=2)
    b = a[:4 * PAGE_SIZE] + _tokens(9, seed=3)
    _ingest(leader, "a", a)
    a_step = _admit(leader, "a", len(a))
    _ingest(leader, "b", b)
    batch = _batch("b")
    assert engine.decide_cached_prefix(batch) == {}
    leader.commit(a_step, _ctx("a"))
    inputs = ARNodeInputs(input_ids=torch.tensor(b), input_seq_len=len(b))
    assert engine._skip_cached_prefix(batch, "b", inputs).input_seq_len == len(b)
    assert leader._streams["b"]["main"].lease is None


def test_under_tp_a_stream_keys_its_prompt_only():
    kv = _manager(1)
    kv.enable_prefix_cache(ROOT, {"main": (WALK, "decode")})
    assert kv._keyed_walks == {"main": frozenset({WALK})}


def test_a_keyed_miss_answers_zero_so_another_resource_cannot_skip_it():
    kv = _manager(0)
    _ingest(kv, "a", _tokens(3 * PAGE_SIZE + 2, seed=11))
    assert kv.resolve_cached_prefix("a", NODE, WALK) == 0


def test_rank_0s_hits_cross_the_wire():
    body = ScheduleTPNode(
        "LLM", "prefill", ["r1", "r2"],
        prefix_hits={"r2": PrefixHit(tokens=256, pages={"kv": [3, 4], "kv_index": [7, 8]})},
    )
    out = decode(encode(WorkerMessage(WorkerMessageType.SCHEDULE_TP, body))).body
    assert out.prefix_hits == body.prefix_hits
    assert isinstance(out.prefix_hits["r2"], PrefixHit)
    assert ScheduleTPNode("LLM", "decode", ["r1"]).prefix_hits == {}


# ── under the admission gate ────────────────────────────────────────────


def _ranks(max_num_pages: int = 32) -> tuple[KVManager, KVManager]:
    """Two ranks whose pools name the stream's walks, as the engine opens them. The gate
    looks a prompt up only on a walk it knows to be the stream's prefill."""
    ranks = _manager(0, max_num_pages), _manager(1, max_num_pages)
    for kv in ranks:
        assert kv.enable_prefix_cache(ROOT, {"main": (WALK, "decode")})
    return ranks


def _gate(ranks, rid: str, tokens: list[int], max_tokens: int) -> bool:
    """Every rank ingests ``rid`` counted, and rank 0's gate says whether it may run."""
    for kv in ranks:
        _ingest(kv, rid, tokens, max_tokens)
    return ranks[0].admit_retrieve(rid, NODE, WALK, None).ready


def _send(ranks, rid: str, tokens: list[int]) -> int:
    """One prefill step: rank 0 decides the hit as it sends the step, the follower adopts
    it, and both run the rest. Returns the matched length."""
    leader, follower = ranks
    matched = leader.resolve_cached_prefix(rid, NODE, WALK)
    pages = leader.cached_prefix_pages(rid, NODE, WALK, matched)
    assert not matched or follower.adopt_cached_prefix(rid, NODE, WALK, matched, pages), (
        f"the follower never adopts {rid}'s {matched}-token hit"
    )
    for kv in ranks:
        _run(kv, rid, tokens, matched)
    return matched


def _cache(ranks, *prompts: list[int]) -> None:
    """Each prompt prefills and leaves, so every rank's index holds its pages, oldest first."""
    for i, tokens in enumerate(prompts):
        assert _gate(ranks, f"old{i}", tokens, 1)
        _send(ranks, f"old{i}", tokens)
        for kv in ranks:
            kv.remove_request(f"old{i}")


def test_a_request_admitted_before_its_step_is_sent_leaves_the_ranks_evicting_alike():
    """Rank 0 admits b, then sends c first. A lease taken for b at the gate would exist on
    rank 0 alone: c would evict another page there, and the follower could not adopt b's hit."""
    ranks = leader, follower = _ranks(max_num_pages=24)
    shared = _tokens(4 * PAGE_SIZE, seed=1)
    _cache(ranks, shared, _tokens(2 * PAGE_SIZE, seed=2), _tokens(6 * PAGE_SIZE, seed=5))
    b = shared + _tokens(PAGE_SIZE, seed=3)
    assert _gate(ranks, "b", b, PAGE_SIZE)
    # one page more than is free, so each rank evicts a cached page for c
    c = _tokens((leader._arena.num_free + 1) * PAGE_SIZE - 8, seed=4)
    assert _gate(ranks, "c", c, 1)

    _send(ranks, "c", c)
    assert _pages(leader, "c") == _pages(follower, "c"), "the ranks evicted different pages for c"
    assert _send(ranks, "b", b) > 0, "b lost its whole hit"

    assert _pages(leader, "b") == _pages(follower, "b")
    assert leader._reserved["b"].pages == follower._reserved["b"].pages, (
        "the ranks reserved differently for b once its hit was leased"
    )
    for kv in ranks:
        kv.assert_pages_conserved()


def test_a_waiting_request_does_not_reorder_rank_0s_evictions():
    """b waits for room while e decodes. Evicting b's matched pages last on rank 0 alone
    would take a different page there than on the follower."""
    ranks = leader, follower = _ranks(max_num_pages=24)
    shared = _tokens(4 * PAGE_SIZE, seed=1)
    _cache(ranks, shared, _tokens(2 * PAGE_SIZE, seed=2), _tokens(6 * PAGE_SIZE, seed=5))
    e = _tokens(PAGE_SIZE, seed=9)
    assert _gate(ranks, "e", e, 19 * PAGE_SIZE)
    assert not _gate(ranks, "b", shared + _tokens(PAGE_SIZE, seed=3), PAGE_SIZE), (
        "b was admitted into the room e reserved"
    )

    _send(ranks, "e", e)
    cached = len(leader._index.pages())
    # e grows a page a step until the free list runs out and a cached page is evicted
    for _ in range(leader._arena.num_free + 1):
        for kv in ranks:
            kv.commit(_admit(kv, "e", PAGE_SIZE), _ctx("e"))

    assert len(leader._index.pages()) < cached, "nothing was evicted"
    assert _pages(leader, "e") == _pages(follower, "e"), "the ranks evicted different pages for e"


def test_a_follower_that_reserved_before_the_hit_gives_the_same_pages_back():
    """b's image step runs before its keyed prefill, so the follower has already reserved
    b's whole count when it adopts the hit rank 0 leased."""
    ranks = leader, follower = _ranks()
    shared = _tokens(4 * PAGE_SIZE, seed=14)
    _cache(ranks, shared)
    b = shared + _tokens(PAGE_SIZE, seed=15)
    for kv in ranks:
        kv.ingest_request("b", KVReqConfig(
            needed_labels_per_node_walk={(NODE, "image"): ["aux"], (NODE, WALK): ["main"]},
            prefix_keys={"main": chain([b[at:at + PAGE_SIZE] for at in range(0, len(b), PAGE_SIZE)])},
            prefix_tail={"main": []},
            max_tokens=PAGE_SIZE, prompt_slots={"main": len(b), "aux": PAGE_SIZE}, decode_labels=["main"],
        ))
    assert leader.admit_retrieve("b", NODE, "image", None).ready
    image = KVStep(segments=(Segment("b", "aux", PAGE_SIZE),))
    ctx = StepContext(request_ids=("b",), graph_walk="image", slot=0, capture=False)
    for kv in ranks:
        assert kv.admit(image, ctx).ok
        kv.plan(image, ctx)
        kv.commit(image, ctx)

    assert _send(ranks, "b", b) == 4 * PAGE_SIZE

    assert leader._reserved["b"].pages == follower._reserved["b"].pages, (
        "the follower still reserves the pages its adopted hit covers"
    )


def test_a_lease_taken_around_a_commit_leaves_the_ranks_indexes_alike():
    """z shares a's prefix but was sent while a ran, so it missed and wrote a copy of its own.
    a has left when z commits. Rank 0 has by then leased the prefix for h, and the follower
    adopts it after its own commit. Chaining z's next page under a's wherever a lease holds
    it would shape the two indexes differently, and they would evict in different orders."""
    ranks = leader, follower = _ranks()
    shared = _tokens(4 * PAGE_SIZE, seed=19)
    prompts = {"a": shared + _tokens(PAGE_SIZE, seed=20), "z": shared + _tokens(2 * PAGE_SIZE, seed=21),
               "h": shared + _tokens(PAGE_SIZE, seed=22)}
    for kv in ranks:
        for rid, tokens in prompts.items():
            _ingest(kv, rid, tokens)
    a_steps = {kv: _admit(kv, "a", len(prompts["a"])) for kv in ranks}
    assert leader.resolve_cached_prefix("z", NODE, WALK) == 0, "z was decided after a committed"
    z_steps = {}
    for kv in ranks:
        kv.commit(a_steps[kv], _ctx("a"))
        kv.apply_cached_prefix("z", NODE, WALK, None, 0)
        z_steps[kv] = _admit(kv, "z", len(prompts["z"]))
        kv.remove_request("a")

    matched = leader.resolve_cached_prefix("h", NODE, WALK)
    pages = leader.cached_prefix_pages("h", NODE, WALK, matched)
    for kv in ranks:
        kv.commit(z_steps[kv], _ctx("z"))
    assert matched and follower.adopt_cached_prefix("h", NODE, WALK, matched, pages)

    parents = [[kv._index._parent[page] for page in sorted(kv._index.pages())] for kv in ranks]
    assert parents[0] == parents[1], "z's page hangs under a's on rank 0 only"


def test_a_readiness_check_stamps_no_cached_page_above_one_rank():
    """A step refused at admit still publishes its empty stream. Rank 0 then checks the
    request's readiness on every scan and a follower once, so a lookup there would age
    rank 0's index alone."""
    ranks = leader, follower = _ranks()
    shared = _tokens(3 * PAGE_SIZE, seed=23)
    _cache(ranks, shared + _tokens(4, seed=24))
    for kv in ranks:
        _ingest(kv, "b", shared + _tokens(20, seed=25), max_tokens=8)
    published = PublishedKVInfo.build_for_rank(0, 2, {"main": KVSequenceInfo(0, None)})

    for _ in range(3):
        assert leader.admit_retrieve("b", NODE, WALK, published).ready

    assert leader._index._clock == follower._index._clock, (
        "rank 0's scans stamped the pages b matches"
    )
    assert leader._streams["b"]["main"].page_indices == []


# ── what is refused at load ─────────────────────────────────────────────


def test_host_offload_is_refused_at_load(monkeypatch):
    with pytest.raises(ValueError, match="cpu_offload_pages=4"):
        _manager(0, cpu_offload_pages=4)
    # the real host pool pins memory, which needs a device
    monkeypatch.setattr(manager_mod, "CPUPagePool", lambda **kwargs: None)
    assert _manager(0, opt_in=False, cpu_offload_pages=4)._index is None, (
        "a TP pool that leaves the cache off may still offload"
    )


def test_a_pool_on_more_than_one_tp_group_is_refused_at_load():
    with pytest.raises(ValueError, match="on one TP group"):
        _manager(0, shared=True)
    assert _manager(0, shared=True, world_size=1)._index is not None, (
        "one rank has nobody to agree with, so sharing the pool is its own business"
    )


class _KeysTwoPools:
    """Keys the prompt in two pools of one node, as GLM-5.2 does under long context."""

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return type(self).__name__

    def prefix_key_streams(self):
        stream = PrefixStream("text_inputs", "ids", WALK)
        return {"kv": {"main": stream}, "kv_index": {"main": stream}}


def _open_two_pools(world_size: int, rank: int = 0, **kv_index: int) -> dict[str, KVManager]:
    """Open the caches of ``rank``'s engine over two keyed pools, ``kv_index`` overriding
    the second pool's config."""
    engine = Engine.__new__(Engine)
    engine._resources = {
        key: _manager(rank, world_size=world_size, prefix_cache_salt="deployment", **overrides)
        for key, overrides in (("kv", {}), ("kv_index", kv_index))
    }
    sharded = {NODE} if world_size > 1 else set()
    engine._tp_leader_nodes = sharded if rank == 0 else set()
    engine._tp_follower_nodes = sharded - engine._tp_leader_nodes
    engine._open_prefix_caches({
        key: KVSpec(resource_key=key, nodes={NODE}, config=kv.config)
        for key, kv in engine._resources.items()
    }, _KeysTwoPools())
    return engine._resources


def test_a_node_keys_two_pools_above_one_rank_and_is_still_refused_at_one():
    """At one rank the gate admits on a pool's own hit, which the step may cut to the other
    pool's shorter one. Above one rank it admits on the whole prompt, so a cut costs nothing."""
    for rank in (0, 1):
        pools = _open_two_pools(world_size=2, rank=rank)
        assert all(kv._keyed_walks == {"main": frozenset({WALK})} for kv in pools.values())

    with pytest.raises(ValueError, match="more than one KV pool"):
        _open_two_pools(world_size=1)


def test_two_keyed_pools_of_different_page_sizes_are_refused_at_load():
    """A step skips the shortest hit in tokens. A pool with larger pages would keep fewer
    whole pages than that, and its stream would end short of the prompt with no error."""
    with pytest.raises(ValueError, match="different page sizes"):
        _open_two_pools(world_size=2, page_size=2 * PAGE_SIZE)


def test_rank_0_cuts_every_pools_lease_to_the_agreed_length_as_it_decides():
    """kv holds three pages of b's prompt and kv_index one, so the step skips one page.
    Rank 0 leases all three in kv as it probes. Cut only when it prepares the step, kv
    would hold two pages there that no follower adopts."""
    pools = {"kv": _manager(0), "kv_index": _manager(0)}
    shared = _tokens(3 * PAGE_SIZE, seed=26)
    for kv, cached in ((pools["kv"], shared), (pools["kv_index"], shared[:PAGE_SIZE])):
        _ingest(kv, "a", cached + _tokens(4, seed=27))
        _run(kv, "a", cached + _tokens(4, seed=27))
        kv.remove_request("a")
        _ingest(kv, "b", shared + _tokens(20, seed=28))

    hits = _engine(**pools).decide_cached_prefix(_batch("b"))

    assert hits["wire-b"].tokens == PAGE_SIZE
    for name, kv in pools.items():
        assert kv._streams["b"]["main"].lease == hits["wire-b"].pages[name], (
            f"rank 0 holds more of {name} than it told the followers to adopt"
        )


# ── the debug check that every rank allocates what rank 0 did ───────────


def _follow(follower: KVManager, head: ScheduleTPNode) -> bool:
    """The follower settles ``head`` at the front of its queue, as it does before building it."""
    engine = _engine(follower)
    scheduler = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine), parallel_leader_nodes=set(),
    )
    scheduler.register_tp_follow(head)
    return scheduler.settle_tp_follow_delta()


def test_a_follower_that_allocated_other_pages_than_rank_0_is_a_loud_error(monkeypatch):
    """Both ranks have allocated as often and hold as many pages, on different ids: nothing
    fails until a sharded prefill gathers the wrong slots, so the follower says so before
    the next step. It says so before it adopts the step's hits, which a rank out of step
    may wait on for good."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    leader, follower = _manager(0), _manager(1)
    prompts = {"x": _tokens(3 * PAGE_SIZE, seed=16), "y": _tokens(2 * PAGE_SIZE, seed=17)}
    for kv, order in ((leader, "xy"), (follower, "yx")):
        for rid in order:
            _ingest(kv, rid, prompts[rid])
            _run(kv, rid, prompts[rid])
    assert leader.page_trail()[0] == follower.page_trail()[0]
    head = ScheduleTPNode(NODE, WALK, ["wire-z"], spec_seq=7)

    head.page_trails = {"kv": follower.page_trail()}
    assert _follow(follower, head), "a follower in step with rank 0 was held back"

    head.page_trails = _engine(leader).page_trails(NODE)
    head.prefix_hits = {"wire-z": PrefixHit(PAGE_SIZE, {"kv": [1]})}
    with pytest.raises(RuntimeError, match=rf"KV kv: .*{NODE}/{WALK}, tp seq 7"):
        _follow(follower, head)


def test_with_the_debug_flag_off_no_trail_is_kept_or_sent(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    leader = _manager(0)
    tokens = _tokens(3 * PAGE_SIZE, seed=18)
    _ingest(leader, "a", tokens)
    _run(leader, "a", tokens)

    assert leader._arena.trail is None
    assert _engine(leader).page_trails(NODE) == {}
    assert ScheduleTPNode(NODE, WALK, ["r1"]).page_trails == {}


def test_a_pool_without_the_tp_cache_keeps_no_trail(monkeypatch):
    """Such a pool may offload to the host. A reload allocates when rank 0 checks readiness
    and when a follower replays the move, so the two trails would not compare."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    assert _manager(0)._arena.trail == (0, 0)
    assert _manager(0, opt_in=False)._arena.trail is None


def test_rank_0s_trails_cross_the_wire():
    body = ScheduleTPNode(NODE, WALK, ["r1"], page_trails={"kv": (3, -(2 ** 61)), "kv_index": (1, 7)})
    out = decode(encode(WorkerMessage(WorkerMessageType.SCHEDULE_TP, body))).body
    assert out.page_trails == body.page_trails, "a trail that came back a list would never compare equal"
