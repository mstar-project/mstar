"""A request is admitted only once its prompt fits, in the order requests first asked.

"Not ready" is an answer the scheduler already retries, so the pool gives it until the
prompt at the head of its queue fits beside the prompts already admitted. A cache hit on a
page only the index holds counts as a page taken, since eviction could hand that page out.
Nothing passes the head, or a stream of short requests would starve a long one.
"""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_admit_rollback import _HostPoolStub
from test_kv_prefix_layout_lease import ASK, IMAGE, OTHER_ASK, RID, TEXT, _config, _walks
from test_kv_prefix_layout_lease import _Node as _LayoutNode
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _grow, _keys, _layout, _manager, _seed, _StubTransfer

from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import CachedPrefix, EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
)
from mstar.engine.resources.kv.manager import KVManager, KVSequenceInfo, PublishedKVInfo
from mstar.engine.resources.step import AdmitRuntimeError, Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs

SEED = 20261022


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _walk(kv) -> set[int]:
    """Pages only the index holds, with no page below them held."""
    index, owners = kv._index, kv._arena.num_owners
    pinned: set[int] = set()
    for page in index.pages():
        if owners[page] > 1:
            parent = index._parent[page]
            while parent is not None and parent not in pinned:
                pinned.add(parent)
                parent = index._parent[parent]
    return {page for page in index.pages() if owners[page] == 1 and page not in pinned}


def _laid_out(kv, rid: str, tokens: list[int], keyed: bool = False) -> None:
    keys = {"main": _keys(tokens)} if keyed else None
    kv.ingest_request(rid, KVReqConfig(prefix_keys=keys, kv_layout=_layout(len(tokens))))


def _ready(kv, rid: str, published: PublishedKVInfo | None = None) -> bool:
    return kv.admit_retrieve(rid, NODE, WALK, published).ready


def _probe(kv, rid: str, tokens: int) -> CachedPrefix | None:
    inputs = ARNodeInputs(input_seq_len=tokens)
    prefix = kv.resolve_cached_prefix(rid, NODE, WALK, inputs)
    kv.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
    return prefix


def test_a_page_filed_after_a_lost_race_keeps_the_winners_pages_evictable():
    kv = _manager()
    shared = list(range(48))
    _laid_out(kv, "a", shared, keyed=True)
    _grow(kv, "a", 48)
    kv.remove_request("a")
    evictable = set(kv._index.evictable())

    # "b" writes the same three pages without probing, so it loses the race for them, then files its fourth page
    _laid_out(kv, "b", shared + list(range(100, 116)), keyed=True)
    _grow(kv, "b", 64)

    assert evictable <= kv._index.evictable(), "filing under the winner's page took room the pool had counted"


def _queued(max_num_pages: int = 9):
    """"a" fits in 8 pages, "long" doesn't fit beside it, and "short" would."""
    kv = _manager(max_num_pages=max_num_pages)
    for rid, n in (("a", 80), ("long", 64), ("short", 16)):
        _laid_out(kv, rid, list(range(n)))
    assert _ready(kv, "a") and not _ready(kv, "long")
    return kv


def test_nothing_passes_the_head_of_the_queue():
    kv = _queued()

    assert not _ready(kv, "short"), "a later request took the room the head waits for"
    kv.remove_request("a")
    assert _ready(kv, "long") and _ready(kv, "short"), "the queue didn't move once the head fit"


def test_removing_the_head_makes_the_next_request_the_head():
    kv = _queued()
    assert not _ready(kv, "short")

    kv.remove_request("long")

    assert _ready(kv, "short"), "a request that left still heads the queue"


def test_a_prompt_bigger_than_the_pool_fails_as_it_joins():
    kv = _manager(max_num_pages=4)
    _laid_out(kv, "a", list(range(64)))

    outcome = kv.admit_retrieve("a", NODE, WALK, None)

    assert isinstance(outcome.reason, AdmitRuntimeError), "a request that can never fit holds the queue"
    assert not kv._waiting


def _cached(kv, tokens: list[int], held: bool) -> None:
    """Index ``tokens``, held by a live request or by the index alone."""
    _seed(kv, tokens)
    if not held:
        kv.remove_request("seed")


def test_a_held_hit_that_turns_evictable_before_the_probe_leaves_other_allotments_alone():
    kv = _manager(max_num_pages=12)
    tokens = list(range(80))
    _cached(kv, tokens, held=True)
    _laid_out(kv, "a", tokens, keyed=True)
    assert _ready(kv, "a") and kv._allotments["a"].pages == 1
    # the hit pages "a" was credited for go back to the index alone, and "b" is allotted the room that frees
    kv.remove_request("seed")
    _laid_out(kv, "b", list(range(500, 500 + kv._room() * PAGE_SIZE)))
    assert _ready(kv, "b") and kv._room() == 0

    _probe(kv, "a", 80)

    grant = kv.admit(KVStep(segments=(Segment("b", "main", 10 * PAGE_SIZE),)), _ctx("b"))
    assert grant.ok, "the probe leased hit pages past what \"a\" was allotted, out of \"b\"'s allotment"


def test_a_hit_on_a_page_only_the_index_holds_is_allotted_as_if_fresh():
    kv = _manager(max_num_pages=9)
    tokens = list(range(80))
    _cached(kv, tokens, held=False)
    _laid_out(kv, "x", list(range(500, 564)))
    _laid_out(kv, "a", tokens, keyed=True)
    assert _ready(kv, "x")

    # "a" hits 4 cached pages and writes 1, but "x" may evict them
    assert not _ready(kv, "a"), "the head was admitted into room its cache hit had to come from"
    kv.remove_request("x")
    assert _ready(kv, "a") and kv._allotments["a"].pages == 5


def test_a_hit_on_a_page_another_request_holds_costs_nothing():
    kv = _manager(max_num_pages=9)
    tokens = list(range(80))
    _cached(kv, tokens, held=True)
    for rid, prompt in (("a", tokens), ("b", list(range(500, 532))), ("c", list(range(900, 916)))):
        _laid_out(kv, rid, prompt, keyed=rid == "a")
    assert _ready(kv, "a") and kv._allotments["a"].pages == 1, "a hit on pages another request holds was charged"
    assert _ready(kv, "b")
    _grow(kv, "b", 32)
    prefix = _probe(kv, "a", 80)

    # one page is free, and "a" has it allotted for its prompt
    assert not _ready(kv, "c"), "the lease of pages another request holds was counted as pages taken"
    _grow(kv, "a", 80 - prefix.tokens)


def test_a_later_grant_cannot_take_the_heads_cached_hits():
    kv = _manager(max_num_pages=9)
    tokens = list(range(80))
    _cached(kv, tokens[:64], held=False)
    _laid_out(kv, "y", list(range(500, 548)))
    assert _ready(kv, "y")
    _grow(kv, "y", 48)
    _laid_out(kv, "a", tokens, keyed=True)
    assert _ready(kv, "a")

    # "y" growing 4 pages would evict the 4 cached pages "a" is admitted to read
    assert not kv.admit(KVStep(segments=(Segment("y", "main", 64),)), _ctx("y")).ok
    prefix = _probe(kv, "a", 80)
    _grow(kv, "a", 80 - prefix.tokens)

    assert prefix.tokens == 64, "a later grant evicted the head's cache hit"


def test_a_repeat_is_allotted_the_cached_image_end_its_probe_will_take():
    node = _LayoutNode([TEXT, IMAGE, ASK], [TEXT, IMAGE, OTHER_ASK])

    assert _ready(node.kv, RID)

    assert node.kv._allotments[RID].pages == 2, "the cached image end the probe will take was left out of the allotment"


def test_an_image_end_copied_onto_the_repeats_page_leaves_its_allotment():
    node = _LayoutNode([TEXT, IMAGE, ASK], [TEXT, IMAGE, OTHER_ASK])
    assert _ready(node.kv, RID)

    for walk, length in _walks([TEXT, IMAGE, OTHER_ASK]):
        node.walk(walk, length)

    assert node.kv._unused == 0, "the image end's source page stayed in the allotment after its copy"


def test_a_repeat_whose_allotment_is_spent_leaves_its_image_end_to_the_walk():
    node = _LayoutNode([TEXT, IMAGE, ASK], [TEXT, IMAGE, OTHER_ASK])
    kv = node.kv
    # "c" holds the cached image end while the repeat joins, so the repeat is allotted only its question's page
    kv.ingest_request("c", _config([TEXT, IMAGE, list(range(400, 405))]))
    _probe(kv, "c", len(TEXT))
    assert _ready(kv, RID) and kv._allotments[RID].pages == 1
    kv.remove_request("c")
    _laid_out(kv, "b", list(range(kv._room() * PAGE_SIZE)))
    assert _ready(kv, "b") and kv._room() == 0

    _probe(kv, RID, len(TEXT))

    kv.assert_pages_conserved()
    assert node.stream.lease_len == 3 * PAGE_SIZE, "the repeat copied its image end onto a page \"b\" counts on"


def test_a_grant_past_its_allotment_takes_only_room():
    kv = _manager(max_num_pages=9)
    _laid_out(kv, "a", list(range(32)))
    _laid_out(kv, "b", list(range(96)))
    assert _ready(kv, "a") and _ready(kv, "b")
    _grow(kv, "a", 32)

    # "a" grows a page past its prompt while every free page is allotted to "b"'s prompt
    outcome = kv.admit(KVStep(segments=(Segment("a", "main", 16),)), _ctx("a"))

    assert not outcome.ok and outcome.reason.pages_short == 1, "a growing request took a page a prompt counts on"
    _grow(kv, "b", 96)


def test_a_guided_request_with_a_hit_is_allotted_its_whole_prompt_and_keeps_its_chain():
    kv = _manager(max_num_pages=16)
    tokens = list(range(80))
    _cached(kv, tokens, held=True)
    kv.ingest_request("g", KVReqConfig(
        prefix_keys={"main": _keys(tokens)},
        kv_layout={**_layout(80), "cfg_img": StreamLayout(spans=(PrefixSpan(80, 80, WALK),))},
        needed_labels_per_node_walk={(NODE, WALK): ["main", "cfg_img"]},
    ))
    assert _ready(kv, "g") and kv._allotments["g"].pages == 10, "a hit the guided walk never takes was credited"

    # the engine doesn't probe a walk that writes two labels
    kv.apply_cached_prefix("g", NODE, WALK, ARNodeInputs(input_seq_len=80), None)
    step, ctx = KVStep(segments=(Segment("g", "main", 80), Segment("g", "cfg_img", 80))), _ctx("g")
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)

    assert kv._streams["g"]["main"].chain.cursor == 5, "the guided prefill lost its chain and was never indexed"


def test_a_decode_worker_holding_the_whole_prompt_is_allotted_nothing_and_reads_in_nothing():
    kv = _manager(max_num_pages=7)
    tokens = list(range(96))
    _cached(kv, tokens, held=True)
    _laid_out(kv, "d", tokens, keyed=True)
    published = PublishedKVInfo.build_for_rank(0, 1, {"main": KVSequenceInfo(
        seq_len=96, latest_kv_transfer_info="remote", page_indices=list(range(6)),
    )})

    # no page is free, and the local match covers all 6 published pages, one more than a probe takes
    assert _ready(kv, "d", published), "the read-in was charged pages this cache already holds"

    assert kv._streams["d"]["main"].page_indices == kv._streams["seed"]["main"].page_indices


def test_a_cut_lease_refunds_only_the_pages_that_go_back_to_the_evictable_set():
    kv = _manager(max_num_pages=16)
    tokens = list(range(80))
    _cached(kv, tokens, held=False)
    _laid_out(kv, "a", tokens, keyed=True)
    assert _ready(kv, "a")
    inputs = ARNodeInputs(input_seq_len=80)
    kv.resolve_cached_prefix("a", NODE, WALK, inputs)
    # "b" leases the first three of the four pages "a" leased
    _laid_out(kv, "b", tokens[:64], keyed=True)
    kv.resolve_cached_prefix("b", NODE, WALK, ARNodeInputs(input_seq_len=64))
    remaining, room = kv._allotments["a"].remaining, kv._room()

    kv.apply_cached_prefix("a", NODE, WALK, inputs, CachedPrefix("main", 32, 32))

    assert kv._room() == room, "the cut moved room that allotments count on"
    assert kv._allotments["a"].remaining == remaining + 1, "the page \"b\" still holds was allotted again"


def test_a_request_one_pool_holds_back_keeps_no_allotment_in_another_on_its_node():
    # Whisper's decoder asks a self-attention pool and a cross-attention pool in one readiness check
    self_kv, cross_kv = _manager(max_num_pages=4), _manager(max_num_pages=3)
    runner = StepRunner(
        {"kv_cache": self_kv, "cross_kv_cache": cross_kv}, node_resources={NODE: ["kv_cache", "cross_kv_cache"]},
    )

    def ingest(rid: str) -> None:
        self_kv.ingest_request(rid, KVReqConfig(loop_caps={"decode_loop": 32}, kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(PAGE_SIZE, PAGE_SIZE, WALK),), growth=(("decode_loop", 1),),
        )}))
        cross_kv.ingest_request(rid, KVReqConfig(kv_layout=_layout(2 * PAGE_SIZE)))

    ingest("r")
    assert runner.admit_retrieve("r", NODE, WALK).ready
    _grow(self_kv, "r", PAGE_SIZE)
    _grow(cross_kv, "r", 2 * PAGE_SIZE)
    # "c" and "d" wait on the cross pages "r" holds
    for rid in ("c", "d", "c"):
        if rid not in self_kv._overrides:
            ingest(rid)
        assert not runner.admit_retrieve(rid, NODE, WALK).ready

    assert self_kv._allotments.keys() == {"r"}, "a request the cross pool holds back kept its self pool allotment"
    assert next(iter(self_kv._waiting)) == "c", "\"d\" passed \"c\" in the self pool's queue"
    decode = self_kv.admit(KVStep(segments=(Segment("r", "main", 1),)), _ctx("r"))
    assert decode.ok, "requests waiting on the cross pool kept the room the running request decodes into"


def _cfg_parallel_pool(node: str, world_size: int = 1) -> KVManager:
    spec = KVSpec(
        resource_key="kv", nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"}, leader="LLM",
        config=PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096, max_num_pages=9, page_size=16),
    )
    group = None if world_size == 1 else SimpleNamespace(rank=0, world_size=world_size)
    return KVManager.build(spec, EngineResourceInfo(
        device=torch.device("cpu"), kv_dtype=torch.float32, nodes=frozenset({node}), joint_comm_group=group,
    ))


def test_only_the_leaders_pool_holds_a_request_back_and_none_above_one_rank():
    for node, world_size, holds in (("LLM", 1, True), ("LLM_cfg_text", 1, False), ("LLM", 2, False)):
        kv = _cfg_parallel_pool(node, world_size)
        for rid in ("a", "b"):
            _laid_out(kv, rid, list(range(80)))

        assert _ready(kv, "a") and _ready(kv, "b") is not holds, f"{node}'s pool at world size {world_size} decided"
        assert holds or not (kv._allotments or kv._waiting), "a pool that doesn't decide kept admission state"


def test_an_allotment_covers_only_the_labels_its_workers_nodes_open():
    kv = _cfg_parallel_pool("LLM")
    kv.ingest_request("a", KVReqConfig(
        kv_layout={label: _layout(16 * n)["main"] for label, n in (("main", 2), ("cfg_img", 1), ("cfg_text", 3))},
        needed_labels_per_node_walk={
            (NODE, WALK): ["main", "cfg_img"], ("LLM_cfg_text", "image_gen_cfg"): ["cfg_text"],
        },
    ))

    assert _ready(kv, "a") and kv._allotments["a"].pages == 3, "the leader counted a label only another worker opens"


class _StubHostPool(_HostPoolStub):
    """`_HostPoolStub` that also copies back, and lands each copy at once."""

    def copy_in(self, gpu_kv_cache, copies):
        pass

    def record(self):
        return SimpleNamespace(query=lambda: True)


def _run(kv, rng, rid: str, span: int, walk: str, policy: RetentionPolicy | None, preplan: bool = False) -> str:
    """Admit one step of ``rid``, then commit the step, roll it back as a later resource's refusal
    would, or drop the step if it was staged ahead."""
    step = KVStep(
        segments=(Segment(rid, "main", span),),
        post_forks=(("main", "fork"),) if rng.random() < 0.2 else (),
        retention={(rid, "main"): policy} if policy else {},
    )
    ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False, is_preplan=preplan)
    if not kv.admit(step, ctx).ok:
        kv.rollback_admit(step, ctx)
        return "refused"
    if preplan:
        kv.clear_preplan()
        return "dropped"
    if rng.random() < 0.15:
        kv.rollback_admit(step, ctx)
        return "rolled back"
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    return "committed"


# each call, the states of the requests it applies to, and how often it is drawn
_CALLS = {
    "ingest": ((), 1), "ready": (("waiting",), 2), "prefill": (("ready",), 2), "decode": (("running",), 12),
    "preplan": (("running",), 1), "reset": (("ready", "running"), 1), "offload": (("running",), 1),
    "reload": (("offloaded",), 1), "remove": (("waiting", "ready", "running", "offloaded"), 1),
}


def _drive(kv, rng: random.Random, ops: int) -> list[str]:
    """Drive requests that share two prompts, keyed or not and some windowed, through every call
    that moves their pages, and run the pool's debug checks after each."""
    prompts = [list(range(base, base + 90)) for base in (0, 1000)]
    live: dict[str, list] = {}  # rid -> [state, prompt length, policy, decoded]
    log: list[str] = []
    for i in range(ops):
        op = rng.choices(list(_CALLS), weights=[w for _, w in _CALLS.values()])[0]
        rids = [rid for rid, entry in live.items() if entry[0] in _CALLS[op][0]]
        rid = rng.choice(rids) if rids else f"r{i}"
        state, n, policy, decoded = live.get(rid, (None, 0, None, 0))
        if op == "ingest" and len(live) < 5:
            tokens = rng.choice(prompts)[:rng.randrange(20, 90)] + [i] * rng.randrange(0, 20)
            # a window that keeps one page releases prompt pages too, past what the request was allotted
            policy = RetentionPolicy(1, PAGE_SIZE) if rng.random() < 0.3 else None
            n = len(tokens)
            kv.ingest_request(rid, KVReqConfig(
                prefix_keys={"main": _keys(tokens)} if rng.random() < 0.7 else None, loop_caps={"decode_loop": 40},
                kv_layout={"main": StreamLayout((PrefixSpan(n, n, WALK),), (("decode_loop", 1),), 0, policy, 1)},
            ))
            live[rid] = ["waiting", n, policy, 0]
        elif state is None:
            continue
        elif op == "ready" and _ready(kv, rid):
            live[rid][0] = "ready"
        elif op == "prefill":
            inputs = ARNodeInputs(input_seq_len=n)
            prefix = kv.resolve_cached_prefix(rid, NODE, WALK, inputs)
            # a lease the probe took can be cut, but not one a refused admit converted
            if prefix and prefix.tokens >= PAGE_SIZE and not kv._streams[rid]["main"].stored_len and rng.random() < 0.3:
                prefix = CachedPrefix("main", PAGE_SIZE, PAGE_SIZE)
            kv.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
            op = _run(kv, rng, rid, n - (0 if prefix is None else prefix.tokens), WALK, policy)
            live[rid][0] = "running" if op == "committed" else state
        elif op in ("decode", "preplan") and decoded < 40:
            op = _run(kv, rng, rid, 1, "decode", policy, preplan=op == "preplan")
            live[rid][3] += op == "committed"
        elif op == "reset":
            kv.reset_request(rid, free=rng.random() < 0.5)
            live[rid][0], live[rid][3] = "ready", 0
        elif op in ("offload", "reload"):
            allotment = kv._allotments.get(rid)
            left = None if allotment is None else allotment.remaining
            if not (kv.offload(rid) if op == "offload" else kv.reload(rid)):
                continue
            assert op == "offload" or left is None or allotment.remaining == left, (
                f"{op} {rid}: the reload took the pages it brought back out of what the request had left"
            )
            live[rid][0] = "offloaded" if op == "offload" else "running"
        elif op == "remove":
            kv.remove_request(rid)
            del live[rid]
        else:
            continue
        log.append(f"{op} {rid}")
        kv.assert_pages_conserved()
        assert kv._index.evictable() == _walk(kv), f"after {log[-1]}: the evictable set is not what a walk finds"
    return log


def test_allotments_never_exceed_the_free_and_evictable_pages(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=20)
    kv._cpu_pool = _StubHostPool()

    log = _drive(kv, random.Random(SEED), ops=10_000)

    for op in ("committed", "rolled back", "dropped", "refused", "reset", "offload", "reload", "remove"):
        assert sum(line.startswith(op) for line in log) >= 10, f"seed {SEED}: too few {op!r} to say anything"
    for rid in list(kv._streams):
        kv.remove_request(rid)
    assert kv._unused == 0 and kv._arena.num_free + len(kv._index.evictable()) == 19, (
        f"seed {SEED}: pages or allotments outlived every request that held them"
    )


def test_every_page_an_offload_frees_is_room():
    kv = _manager(max_num_pages=9)
    kv._cpu_pool = _StubHostPool()
    _laid_out(kv, "r0", list(range(4 * PAGE_SIZE)))
    assert _ready(kv, "r0")
    _grow(kv, "r0", 2 * PAGE_SIZE)
    room = kv._room()

    assert kv.offload("r0") == 2

    assert kv._room() == room + 2, "the offload gave back its victim's unused allotment, or kept the used part"


def test_a_reload_past_the_room_is_refused_and_leaves_the_request_offloaded():
    kv = _manager(max_num_pages=9)
    kv._cpu_pool = _StubHostPool()
    _laid_out(kv, "r0", list(range(4 * PAGE_SIZE)))
    assert _ready(kv, "r0")
    _grow(kv, "r0", 4 * PAGE_SIZE)
    assert kv.offload("r0") == 4
    _laid_out(kv, "r1", list(range(500, 500 + 5 * PAGE_SIZE)))
    assert _ready(kv, "r1")

    assert not kv.can_reload("r0"), "can_reload counted room an admitted prompt counts on"
    assert kv.reload("r0") is False, "a returning request took pages an admitted prompt counts on"
    assert kv.is_offloaded("r0")
    kv.assert_pages_conserved()
