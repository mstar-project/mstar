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
import test_kv_host_index as host
import torch
from test_kv_prefix_layout_lease import ASK, IMAGE, OTHER_ASK, RID, TEXT, _config, _walks
from test_kv_prefix_layout_lease import _Node as _LayoutNode
from test_kv_prefix_lease import (
    NODE,
    PAGE_SIZE,
    ROOT,
    WALK,
    _ctx,
    _grow,
    _keys,
    _layout,
    _manager,
    _seed,
    _StubTransfer,
)

from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import CachedPrefix, EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
)
from mstar.engine.resources.kv.keys import chain, fingerprint
from mstar.engine.resources.kv.manager import KVManager, KVSequenceInfo, PageArena, PublishedKVInfo
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
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


_RUNNERS: dict[int, tuple[KVManager, StepRunner]] = {}


def _runner(kv) -> StepRunner:
    """The runner that admits for ``kv``, as the engine's does."""
    if id(kv) not in _RUNNERS or _RUNNERS[id(kv)][0] is not kv:
        _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv}, node_resources={NODE: ["kv"]}))
    return _RUNNERS[id(kv)][1]


def _ready(kv, rid: str, published: PublishedKVInfo | None = None) -> bool:
    """A readiness check as the scheduler makes it, after it takes whatever has since been admitted."""
    runner = _runner(kv)
    mine = None if published is None else {"kv": published}
    runner.admit_waiting(lambda other, node: mine if other == rid else None)
    return runner.admit_retrieve(rid, NODE, WALK, mine).ready


def _remove(kv, rid: str) -> None:
    _runner(kv).remove_request(rid)


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
    _remove(kv, "a")
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
    _remove(kv, "a")
    assert _ready(kv, "long") and _ready(kv, "short"), "the queue didn't move once the head fit"


def test_removing_the_head_makes_the_next_request_the_head():
    kv = _queued()
    assert not _ready(kv, "short")

    _remove(kv, "long")

    assert _ready(kv, "short"), "a request that left still heads the queue"


def test_a_prompt_bigger_than_the_pool_fails_as_it_joins():
    kv = _manager(max_num_pages=4)
    _laid_out(kv, "a", list(range(64)))

    outcome = _runner(kv).admit_retrieve("a", NODE, WALK)

    assert isinstance(outcome.reason, AdmitRuntimeError), "a request that can never fit holds the queue"
    assert not _runner(kv)._waiting and not kv._asks


def _cached(kv, tokens: list[int], held: bool) -> None:
    """Index ``tokens``, held by a live request or by the index alone."""
    _seed(kv, tokens)
    if not held:
        _remove(kv, "seed")


def test_a_held_hit_that_turns_evictable_before_the_probe_leaves_other_allotments_alone():
    kv = _manager(max_num_pages=12)
    tokens = list(range(80))
    _cached(kv, tokens, held=True)
    _laid_out(kv, "a", tokens, keyed=True)
    assert _ready(kv, "a") and kv._allotments["a"].pages == 1
    # the hit pages "a" was credited for go back to the index alone, and "b" is allotted the room that frees
    _remove(kv, "seed")
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
    _remove(kv, "x")
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


def test_a_hit_page_another_request_takes_lets_the_head_in():
    """The other request pays for the page out of its own allotment, so the head's need drops while
    the room stays where it was."""
    kv = _manager(max_num_pages=9)
    tokens = list(range(80))
    _cached(kv, tokens, held=False)
    _laid_out(kv, "b", tokens, keyed=True)
    _laid_out(kv, "a", tokens + list(range(900, 932)), keyed=True)
    assert _ready(kv, "b") and not _ready(kv, "a")
    room = kv._room()

    _probe(kv, "b", 80)

    assert kv._room() == room and _ready(kv, "a"), "the head waits for room it no longer needs"


def test_a_hit_page_taken_while_another_page_turns_evictable_lets_the_head_in():
    """The two changes leave the room and the evictable count where they were, but the head needs a
    page less."""
    kv = _manager(max_num_pages=12)
    tokens, other = list(range(80)), list(range(300, 300 + PAGE_SIZE))
    _cached(kv, tokens, held=False)
    _laid_out(kv, "y", other, keyed=True)
    assert _ready(kv, "y")
    _grow(kv, "y", PAGE_SIZE)
    # filed as at commit, while "y" holds it
    kv._index.insert(fingerprint(ROOT, _keys(other)[0]), kv._streams["y"]["main"].page_indices[0])
    _laid_out(kv, "b", tokens[:2 * PAGE_SIZE], keyed=True)
    _laid_out(kv, "a", tokens + list(range(900, 900 + 4 * PAGE_SIZE)), keyed=True)
    assert _ready(kv, "b") and not _ready(kv, "a")

    # "b" takes one of the head's cached hits out of the evictable set
    _probe(kv, "b", 2 * PAGE_SIZE)
    # "y"'s page goes back to the index alone, and its allotment keeps the room
    kv.reset_request("y")

    assert _ready(kv, "a"), "the head waits for room it no longer needs"


def _recurrent(max_slots: int, labels: tuple[str, ...] = ("main",)) -> RecurrentStatePool:
    return RecurrentStatePool.build(
        RecurrentStateSpec(resource_key="state", nodes={NODE}, config=RecurrentStateConfig(
            num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)},
            max_slots=max_slots, labels=labels,
        )),
        EngineResourceInfo(device=torch.device("cpu")),
    )


def test_a_request_with_no_recurrent_slot_keeps_no_kv_pages():
    """Qwen3.5 runs a KV pool and a recurrent pool on one node. A request the KV admitted with no slot
    free would fail its first step, and its allotment would keep room from the requests running."""
    kv = _manager(max_num_pages=9)
    state = _recurrent(max_slots=2)
    runner = StepRunner({"kv": kv, "state": state}, node_resources={NODE: ["kv", "state"]})
    for rid in ("r0", "r1"):
        _laid_out(kv, rid, list(range(16)))
        state.ingest_request(rid)
    assert runner.admit_retrieve("r0", NODE, WALK).ready
    assert state.admit(RecurrentStep(segments=(Segment("r0", "main", 16),)), _ctx("r0")).ok

    assert runner.admit_retrieve("r1", NODE, WALK).queued
    assert "r1" not in kv._allotments, "a request with no slot kept the KV pages it was allotted"


def test_a_request_that_forks_its_recurrent_state_is_admitted_with_a_slot_for_each_label():
    """Two forking requests admitted on one slot each would each find the other's slot kept for it at the
    fork, refuse, and retry forever."""
    state = _recurrent(max_slots=3, labels=("main", "fork"))
    runner = StepRunner({"state": state}, node_resources={NODE: ["state"]})
    for rid in ("r0", "r1"):
        state.ingest_request(rid)
    assert runner.admit_retrieve("r0", NODE, WALK).ready

    assert runner.admit_retrieve("r1", NODE, WALK).queued, "admitted with only one of its slots free"
    fork = RecurrentStep(segments=(Segment("r0", "main", 4),), pre_forks=(("main", "fork"),))
    assert state.admit(fork, _ctx("r0")).ok
    runner.remove_request("r0")
    assert runner.admit_waiting(lambda rid, node: None) == [(NODE, "r1")]


def test_a_step_that_opens_an_undeclared_recurrent_label_fails_its_request():
    state = _recurrent(max_slots=3)
    state.ingest_request("r")
    fork = RecurrentStep(segments=(Segment("r", "main", 4),), pre_forks=(("main", "fork"),))

    outcome = state.admit(fork, _ctx("r"))

    assert not outcome.ok and outcome.reason.terminal, "a slot admission never counted was taken or retried"


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


def test_a_guaranteed_repeat_whose_footprint_fills_the_pool_is_admitted_to_an_idle_pool():
    node = _LayoutNode([TEXT, IMAGE, ASK], [TEXT, IMAGE, OTHER_ASK])
    node.runner.remove_request("seed")
    overrides, node.kv.local_loops = node.kv._overrides[RID], {"decode_loop"}
    prompt = sum(span.length for span in overrides.kv_layout["main"].spans)
    overrides.guaranteed, overrides.loop_caps = True, {"decode_loop": node.kv.config.request_pages * PAGE_SIZE - prompt}

    assert _ready(node.kv, RID), "the cached image end's copy pushed a guaranteed reservation past the pool"
    node.walk(*_walks([TEXT, IMAGE, OTHER_ASK])[0])
    node.kv.assert_pages_conserved()


def _fill(kv, rid: str, spare: int = 0) -> None:
    _laid_out(kv, rid, list(range((kv._room() - spare) * PAGE_SIZE)))
    assert _ready(kv, rid) and kv._room() == spare


@pytest.mark.parametrize("other", ["before", "after", None])
def test_a_guaranteed_repeat_that_copies_an_image_end_grows_to_its_cap_in_a_full_pool(other):
    node = _LayoutNode([TEXT, IMAGE, ASK], [TEXT, IMAGE, OTHER_ASK])
    kv, cap, walks = node.kv, 3 * PAGE_SIZE, _walks([TEXT, IMAGE, OTHER_ASK])
    overrides, kv.local_loops = kv._overrides[RID], {"decode_loop"}
    overrides.guaranteed, overrides.loop_caps = True, {"decode_loop": cap}
    # "c" probes the cached image end before the repeat leases it, or while the repeat holds it
    kv.ingest_request("c", _config([TEXT, IMAGE, list(range(400, 405))]))
    if other == "before":
        _probe(kv, "c", len(TEXT))
    assert _ready(kv, RID)
    # a page another request holds is lent in a full pool, and an evictable one takes a page of room
    _fill(kv, "b", spare=int(other != "before"))
    node.walk(*walks[0])
    if other == "after":
        _probe(kv, "c", len(TEXT))
    assert [node.walk(walk, length) for walk, length in walks[1:]][0] == IMAGE, "the repeat recomputed its image"
    _fill(kv, "d")

    for _ in range(cap):
        node.step(RID, 1, "decode")

    kv.assert_pages_conserved()


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
    _remove(kv, "c")
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
        layout = StreamLayout(spans=(PrefixSpan(PAGE_SIZE, PAGE_SIZE, WALK),), growth=(("decode_loop", 1),))
        self_kv.ingest_request(rid, KVReqConfig(loop_caps={"decode_loop": 32}, kv_layout={"main": layout}))
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
    assert runner._queues["kv_cache"][0] == "c", "\"d\" passed \"c\" in the self pool's queue"
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
        assert holds or not (kv._allotments or kv._asks), "a pool that doesn't decide kept admission state"


def test_an_allotment_covers_only_the_labels_its_workers_nodes_open():
    kv = _cfg_parallel_pool("LLM")
    kv.ingest_request("a", KVReqConfig(
        kv_layout={label: _layout(16 * n)["main"] for label, n in (("main", 2), ("cfg_img", 1), ("cfg_text", 3))},
        needed_labels_per_node_walk={
            (NODE, WALK): ["main", "cfg_img"], ("LLM_cfg_text", "image_gen_cfg"): ["cfg_text"],
        },
    ))

    assert _ready(kv, "a") and kv._allotments["a"].pages == 3, "the leader counted a label only another worker opens"


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
        if i % 100 == 0:
            # copies from the host land in batches, so two can land in one call
            kv._cpu_pool.land()
        op = rng.choices(list(_CALLS), weights=[w for _, w in _CALLS.values()])[0]
        rids = [rid for rid, entry in live.items() if entry[0] in _CALLS[op][0]]
        rid = rng.choice(rids) if rids else f"r{i}"
        state, n, policy, decoded = live.get(rid, (None, 0, None, 0))
        if op == "ingest" and len(live) < 5:
            tokens = rng.choice(prompts)[:rng.randrange(20, 90)] + [i] * rng.randrange(0, 20)
            # a window that keeps one page releases prompt pages as the request decodes, which leaves it
            # holding fewer pages than it was allotted
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
        elif op == "offload" and kv.offload(rid):
            live[rid][0] = "offloaded"
        elif op == "reload":
            left, fits = kv._allotments[rid].remaining, kv.can_reload(rid)
            assert kv.reload(rid) == fits, f"can_reload {rid} answered {fits}, and reload the opposite"
            if not fits:
                continue
            assert kv._allotments[rid].remaining == left, f"reload {rid} changed the unused part of its allotment"
            live[rid][0] = "running"
        elif op == "remove":
            _remove(kv, rid)
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
    kv._cpu_pool = host._HostPool(64)

    log = _drive(kv, random.Random(SEED), ops=10_000)

    for op in ("committed", "rolled back", "dropped", "refused", "reset", "offload", "reload", "remove"):
        assert sum(line.startswith(op) for line in log) >= 10, f"seed {SEED}: too few {op!r} to say anything"
    for rid in list(kv._streams):
        _remove(kv, rid)
    host._land(kv)
    assert kv._unused == 0 and kv._arena.num_free + len(kv._index.evictable()) == 19, (
        f"seed {SEED}: pages or allotments outlived every request that held them"
    )


def test_every_page_an_offload_frees_is_room():
    kv = _manager(max_num_pages=9)
    kv._cpu_pool = host._HostPool(64)
    _laid_out(kv, "r0", list(range(4 * PAGE_SIZE)))
    assert _ready(kv, "r0")
    _grow(kv, "r0", 2 * PAGE_SIZE)
    room = kv._room()

    assert kv.offload("r0") == 2

    assert kv._room() == room + 2, "the offload gave back its victim's unused allotment, or kept the used part"


def test_a_copy_from_the_host_waits_for_the_join_and_comes_out_of_the_allotment():
    kv = host._manager(max_num_pages=20)
    host._on_the_host_only(kv)
    # "c" holds the first four pages of the prompt "b" finds on the host
    host._serve(kv, "c", host.SHARED + list(range(900, 900 + 4 * host.PAGE_SIZE)))
    room = kv._room()
    for rid, prompt in (("big", list(range(500, 500 + (room + 1) * host.PAGE_SIZE))), ("b", host.A_PROMPT + [7, 8])):
        host._ingest(kv, rid, prompt)
        assert not _ready(kv, rid)
    assert kv._host_copies == [], "a request still in the queue started a copy from the host"

    _remove(kv, "big")
    assert not _ready(kv, "b") and kv._host_copies, "\"b\" didn't start its copy once admitted"
    pages = kv._allotments["b"].pages
    assert kv._room() == room - pages, "the copy took room past \"b\"'s allotment"
    host._land(kv)
    assert kv._allotments["b"].pages == pages and kv._room() == room - pages, "the landed copy moved \"b\"'s room"


def _full_of_cache_with_a_prefix_on_the_host():
    kv = host._manager(max_num_pages=20)
    host._serve(kv, "a", host.A_PROMPT + [1, 2])
    assert kv.offload("a") > 0
    # the index keeps the first four of the six pages the host holds, and "c" fills the pool after them, so those
    # four are the leaves an eviction takes first
    kv._index.evict(2)
    host._serve(kv, "c", list(range(500, 500 + 15 * host.PAGE_SIZE)))
    _remove(kv, "c")
    assert kv._arena.num_free == 0 and kv._index.num_pages() == 19
    return kv


def test_a_head_in_a_pool_with_no_free_page_copies_its_prefix_back_from_the_host():
    kv = _full_of_cache_with_a_prefix_on_the_host()
    host._ingest(kv, "b", host.A_PROMPT + [7, 8])

    assert not _ready(kv, "b"), "\"b\" recomputes a prefix the host holds"
    host._land(kv)
    assert host._probe(kv, "b", host.A_PROMPT + [7, 8]) == 6 * host.PAGE_SIZE, (
        "the copy evicted the index's pages of \"b\"'s prefix, or \"b\"'s probe missed the copied pages"
    )


def test_a_guided_head_copies_nothing_from_the_host():
    kv = _full_of_cache_with_a_prefix_on_the_host()
    tokens = host.A_PROMPT + [7, 8]
    spans = (PrefixSpan(len(tokens), len(tokens), WALK),)
    kv.ingest_request("g", KVReqConfig(
        prefix_keys={"main": chain([tokens[at:at + host.PAGE_SIZE] for at in range(0, len(tokens), host.PAGE_SIZE)])},
        kv_layout={"main": StreamLayout(spans=spans, keyed_spans=1), "cfg_img": StreamLayout(spans=spans)},
        needed_labels_per_node_walk={(NODE, WALK): ["main", "cfg_img"]},
    ))

    # the engine doesn't probe a walk that writes two labels
    assert _ready(kv, "g") and not kv._host_copies, "a guided head waits for a copy its walk never reads"
    assert kv._index.num_pages() == 19, "a guided head's copy evicted cached pages"
    assert kv._streams["g"]["main"].chain.looked_up, "every readiness check of a guided head takes the pool's lock"


def test_a_reload_counts_the_evictable_index_pages_it_keeps_as_room():
    kv = host._manager(max_num_pages=16)
    host._serve(kv, "a", host.A_PROMPT)
    assert kv.offload("a") > 0
    host._ingest(kv, "r1", list(range(500, 500 + 10 * host.PAGE_SIZE)))
    # the index still holds all six of "a"'s pages, so the reload copies none, and the room is five
    assert _ready(kv, "r1") and kv._room() == 5

    assert not (kv.can_reload("a") or kv.reload("a")) and kv.is_offloaded("a"), "a reload went past the room"


def test_a_grant_that_evicts_counts_the_pages_the_host_has_no_copy_of():
    kv = host._manager(max_num_pages=8)
    host._serve(kv, "a", host.A_PROMPT + [1])
    _remove(kv, "a")

    host._serve(kv, "b", list(range(500, 500 + 3 * host.PAGE_SIZE)))

    assert kv._cpu_pool.dropped == 2, "the grant's eviction went uncounted"


def test_a_run_filed_after_a_page_only_the_index_holds_leaves_that_page_evictable():
    arena = PageArena(kv_cache=None, allocator=PageAllocator(8))
    arena.index = index = PrefixIndex(arena)
    cached, *run = arena.acquire(3)
    index.insert(b"cached", cached)
    arena.release([cached])

    index.insert_run([b"cached", b"next"], run, None)

    assert cached in index.evictable(), "the run's next page, held by its stream, pinned a page the pool counts as room"


def test_two_copies_under_one_parent_page_release_it_once_each_and_leave_it_cached():
    kv = host._manager(max_num_pages=32)
    other = host.SHARED + list(range(300, 300 + 2 * host.PAGE_SIZE))
    for rid, prompt in (("a", host.A_PROMPT), ("x", other)):
        host._serve(kv, rid, prompt + [1, 2])
        assert kv.offload(rid) > 0
    kv._index.evict(4)
    shared = kv._index.pages_for(kv._streams["a"]["main"].chain.rooted[:4])
    assert len(shared) == 4 and kv._index.num_pages() == 4
    for rid, prompt in (("b", host.A_PROMPT), ("c", other)):
        host._ingest(kv, rid, prompt + [7, 8])
        assert not _ready(kv, rid) and set(shared) <= set(kv._streams[rid]["main"].hold)

    host._land(kv)

    assert all(kv._arena.num_owners[page] == 1 for page in shared), "a landing left the shared parent pages held"
