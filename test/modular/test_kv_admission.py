"""A request is admitted only once its prompt fits, in the order requests first asked unless it passes the head.

"Not ready" is an answer the scheduler already retries, so the pool gives it until the
prompt at the head of its queue fits beside what admitted requests have reserved. A cache hit on a
page only the index holds counts as a page taken, since eviction could hand that page out.
A request passes the head when its footprint fits, less the part of its hit no eviction can take
and the recurrent state can restore. The pages it frees are then kept for the head, or a stream
of short requests would starve a long one.
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
from test_sampler_admission import _resource as _sampler

from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import CachedPrefix, EngineResourceInfo, Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.arena import PageArena
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
from mstar.engine.resources.kv.manager import KVManager, KVSequenceInfo, PublishedKVInfo
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import ASKS, AdmitRuntimeError, Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs

SEED = 20261022
HELD = list(range(48))


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
    """"a" fits in 8 pages, "long" doesn't fit beside it, and "short"'s prompt would but its output wouldn't."""
    kv = _manager(max_num_pages=max_num_pages)
    kv.local_loops = {"decode_loop"}
    for rid, n in (("a", 80), ("long", 64)):
        _laid_out(kv, rid, list(range(n)))
    kv.ingest_request("short", KVReqConfig(loop_caps={"decode_loop": 48}, kv_layout={
        "main": StreamLayout((PrefixSpan(16, 16, WALK),), (("decode_loop", 1),), keyed_spans=1),
    }))
    assert _ready(kv, "a") and not _ready(kv, "long")
    return kv


def test_a_request_with_nothing_cached_passes_the_head_only_when_its_footprint_fits():
    kv = _queued()
    _laid_out(kv, "tiny", list(range(16)))

    assert not _ready(kv, "short"), "a request passed the head with room for its prompt but not its output"
    assert _ready(kv, "tiny"), "a request whose footprint fits waited behind the head because nothing of it is cached"
    _remove(kv, "a")
    assert _ready(kv, "long") and _ready(kv, "short"), "the queue didn't move once the head fit"


def test_removing_the_head_makes_the_next_request_the_head():
    kv = _queued()
    assert not _ready(kv, "short")

    _remove(kv, "long")

    assert _ready(kv, "short"), "a request that left still heads the queue"


def _prefill(kv, rid: str, tokens: list[int]) -> None:
    """Admit ``rid`` and write its prompt past what the cache serves."""
    _laid_out(kv, rid, tokens, keyed=True)
    assert _ready(kv, rid)
    prefix = _probe(kv, rid, len(tokens))
    _grow(kv, rid, len(tokens) - (prefix.tokens if prefix else 0))


def _passing(free: int, head: int, kv=None):
    """"r" runs on ``HELD``, "fill" runs on tokens from 2000 and leaves ``free`` pages, and "head" waits."""
    kv = _manager(max_num_pages=16) if kv is None else kv
    kv._cpu_pool = host._HostPool(64)
    kv.local_loops = {"decode_loop"}
    _prefill(kv, "r", HELD + list(range(1000, 1008)))
    _prefill(kv, "fill", list(range(2000, 2000 + (kv._room() - free) * PAGE_SIZE)))
    _laid_out(kv, "head", list(range(3000, 3000 + head * PAGE_SIZE)))
    assert not _ready(kv, "head")
    return kv


def _decoding(kv, rid: str, tokens: list[int], cap: int) -> None:
    """A request on ``tokens``, decoding up to ``cap`` tokens."""
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": _keys(tokens)}, loop_caps={"decode_loop": cap},
        kv_layout={"main": StreamLayout(
            (PrefixSpan(len(tokens), len(tokens), WALK),), (("decode_loop", 1),), keyed_spans=1,
        )},
    ))


def _short_chat(kv, rid: str, own: list[int], cap: int = 16) -> None:
    """A request on ``HELD``, which "r" holds, plus ``own``, decoding up to ``cap`` tokens."""
    _decoding(kv, rid, HELD + own, cap)


def test_a_passer_pays_for_its_output_and_for_each_page_of_its_hit_eviction_can_take():
    kv = _passing(free=3, head=5)
    # "w" passes and leaves its last page evictable, which the pool keeps for the head
    _prefill(kv, "w", HELD + list(range(100, 116)))
    kv.remove_request("w")
    _laid_out(kv, "partly", HELD + list(range(100, 124)), keyed=True)
    _short_chat(kv, "long", [8] * 8, cap=32)

    assert not _ready(kv, "long"), "a request passed the head with room for its prompt but not its output"
    assert _ready(kv, "partly"), "a request whose footprint fits beside its hit waited"
    assert kv._allotments["partly"].pages == 2, "the passer's hit page eviction can take wasn't charged as fresh"


def test_a_filed_page_tries_no_request_whose_output_cant_fit_beside_the_head(monkeypatch):
    kv = _passing(free=3, head=5)
    for n in range(16):
        _short_chat(kv, f"w{n}", [7000 + n] * 8, cap=512)
        assert not _ready(kv, f"w{n}")
    tried, past = [], KVManager.reserve_past
    monkeypatch.setattr(KVManager, "reserve_past", lambda kv, rid, *args: tried.append(rid) or past(kv, rid, *args))
    kv._index.filed += 1  # a page filed on no waiting request's chain, as a decode key files one

    _runner(kv).admit_waiting(lambda rid, node: None)

    assert not tried, f"one filed page tried {len(tried)} requests whose output doesn't fit whatever their hit"


def test_a_request_that_asks_while_the_heads_mark_is_out_of_date_passes_once_the_head_is_refused_again():
    kv = _passing(free=3, head=5)
    kv._index.filed += 1  # the head may fit now, so nothing passes it until it is measured again
    _short_chat(kv, "p", [7000] * 8)
    assert _runner(kv).admit_retrieve("p", NODE, WALK, None).queued

    started = _runner(kv).admit_waiting(lambda rid, node: None)

    assert (NODE, "p") in started, "a request parked behind a head that still doesn't fit was never tried to pass"


def test_a_filed_page_remeasures_a_waiting_request_whose_output_fits(monkeypatch):
    kv = _passing(free=3, head=5)
    # "p"'s output fits, but not beside its three own pages past what "r" holds
    _short_chat(kv, "p", list(range(7000, 7000 + 3 * PAGE_SIZE)))
    assert not _ready(kv, "p")
    measured = []
    hit = KVManager._hit
    monkeypatch.setattr(KVManager, "_hit", lambda self, rid, *args: measured.append(rid) or hit(self, rid, *args))
    kv._index.filed += 1

    _runner(kv).admit_waiting(lambda rid, node: None)

    assert "p" in measured, "a filed page could have raised the hit of a request refused on it, and it wasn't measured"


def _with_snapshots(kv) -> RecurrentStatePool:
    """A Qwen3.5-shaped node: ``kv`` beside a recurrent pool whose snapshots key off ``kv``'s chains."""
    state = RecurrentStatePool(device=torch.device("cpu"), name="state", config=RecurrentStateConfig(
        num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)}, max_slots=8,
        prefix_slots=4, kv_cache="kv",
    ))
    state.bind_kv(kv)
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "state": state}, node_resources={NODE: ["kv", "state"]}))
    return state


def _snapshot(state: RecurrentStatePool, kv, rid: str, pages: int) -> None:
    """File a snapshot at ``rid``'s chain key ``pages`` pages in, as a prefill that kept that boundary would."""
    assert state._index(state._reserve_snapshot(), kv.chain_key(rid, "main", pages))


def test_a_passer_stays_inside_its_allotment_when_the_recurrent_pool_cuts_its_hit():
    kv = _manager(max_num_pages=16)
    _with_snapshots(kv)
    kv = _passing(free=5, head=6, kv=kv)
    _short_chat(kv, "p", list(range(4000, 4008)))
    assert _ready(kv, "p")
    room = kv._room()
    # no snapshot under the hit, so the runner agrees on none and the prefill writes the KV hit again
    inputs = ARNodeInputs(input_seq_len=len(HELD) + 8)
    runner = _runner(kv)
    runner.apply_cached_prefix("p", NODE, WALK, inputs, runner.resolve_cached_prefix("p", NODE, WALK, inputs))
    _grow(kv, "p", len(HELD) + 8)

    assert kv._room() == room, (
        f"the passer wrote {room - kv._room()} pages past its allotment, out of the room kept for the head")


def test_a_head_is_allotted_the_part_of_its_hit_past_its_newest_snapshot():
    kv = _manager(max_num_pages=16)
    state = _with_snapshots(kv)
    _prefill(kv, "r", HELD + list(range(1000, 1008)))
    _snapshot(state, kv, "r", 1)
    _laid_out(kv, "h", HELD + list(range(2000, 2008)), keyed=True)

    assert _ready(kv, "h")
    assert kv._allotments["h"].pages == 3, "the head's allotment left out hit pages the recurrent pool can't restore"


def test_a_snapshot_filed_under_a_waiting_heads_hit_retries_it():
    kv = _manager(max_num_pages=16)
    state = _with_snapshots(kv)
    _prefill(kv, "r", HELD + list(range(1000, 1008)))
    _prefill(kv, "fill", list(range(5000, 5000 + (kv._room() - 2) * PAGE_SIZE)))
    _laid_out(kv, "h", HELD + list(range(2000, 2008)), keyed=True)
    assert not _ready(kv, "h")

    _snapshot(state, kv, "r", 3)

    assert _ready(kv, "h"), "a snapshot lowered what the head needs, and nothing its mark reads moved"


@pytest.mark.parametrize("leave", [KVManager.remove_request, KVManager.offload], ids=["finishes", "is offloaded"])
def test_the_pages_a_request_that_passed_frees_go_to_the_head(leave):
    kv = _passing(free=1, head=5)
    _prefill(kv, "p", HELD + list(range(4000, 4008)))
    _laid_out(kv, "q", list(range(2000, 2048)) + list(range(5000, 5008)), keyed=True)
    leave(kv, "p")
    assert not _ready(kv, "q"), f"a request behind the head took the page {leave.__name__} freed"
    kv.remove_request("r")

    assert not _ready(kv, "q"), "a request passed a head that the room now covers"
    assert _ready(kv, "head"), "the head waits though the room, the page kept for it included, covers it"
    assert kv._kept == 0, "the pages kept for the head outlived its admission, so the next head's passers wait"


@pytest.mark.parametrize("slots, passes", [(5, True), (4, False)], ids=["a slot left for the head", "none left"])
def test_a_request_passes_beside_a_recurrent_pool_only_while_a_slot_is_left_for_the_head(slots, passes):
    """Qwen3.5 runs a KV pool and a recurrent pool on one node, and a request asks both."""
    kv = _manager(max_num_pages=16)
    state = RecurrentStatePool.build(
        RecurrentStateSpec(resource_key="state", nodes={NODE}, config=RecurrentStateConfig(
            num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)}, max_slots=slots,
        )),
        EngineResourceInfo(device=torch.device("cpu")),
    )
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "state": state}, node_resources={NODE: ["kv", "state"]}))
    kv = _passing(free=3, head=5, kv=kv)
    kv.ingest_request("short", KVReqConfig(prefix_keys={"main": _keys(HELD + [8] * 8)}, loop_caps={"decode_loop": 16},
        kv_layout={"main": StreamLayout((PrefixSpan(56, 56, WALK),), (("decode_loop", 1),), keyed_spans=1)}))

    assert _ready(kv, "short") is passes, "the head would wait for a slot a request passing it took"


@pytest.mark.parametrize("rows, passes", [(4, True), (3, False)], ids=["a row left for the head", "none left"])
def test_a_request_passes_beside_the_samplers_rows_only_while_one_is_left_for_the_head(monkeypatch, rows, passes):
    """Since #938 every request on a captured node asks the sampler for an in-graph row, the head included."""
    kv = _manager(max_num_pages=16)
    sampler = _sampler(monkeypatch, slots=rows, max_bs=1)
    for rid in ("r", "fill", "head", "short"):
        sampler.ingest_request(rid)
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "sampler": sampler}, node_resources={NODE: ["kv", "sampler"]}))
    kv = _passing(free=3, head=5, kv=kv)
    _short_chat(kv, "short", [8] * 8)

    assert _ready(kv, "short") is passes, "a request that passes beside the KV waited behind the head for a row"


class _Picky(Resource):
    """A pool every request asks, which refuses the requests in ``refused``."""

    def __init__(self):
        self.refused, self.held, self.moved = set(), set(), 0

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def ask(self, rid, node_name, graph_walk):
        return None if rid in self.held else ASKS

    def reserve(self, rid, node_name, graph_walk, published):
        if rid in self.refused:
            return self.moved
        self.held.add(rid)
        return None

    reserve_past = reserve

    def unreserve(self, rid):
        self.held.discard(rid)

    def grew_since(self, mark):
        return self.moved != mark


def test_a_request_that_passed_goes_back_behind_the_head_when_another_pool_holds_it_back():
    kv = _manager(max_num_pages=16)
    picky = _Picky()
    picky.refused.add("p")
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "picky": picky}, node_resources={NODE: ["kv", "picky"]}))
    kv = _passing(free=1, head=2, kv=kv)
    _laid_out(kv, "p", HELD + list(range(4000, 4008)), keyed=True)

    assert not _ready(kv, "p"), "p passed though the other pool had no slot for it"
    assert "p" not in kv._allotments and "p" in kv._asks, "the KV kept what it gave a request another pool held back"


def test_the_pages_kept_for_the_head_survive_another_pool_refusing_it():
    kv = _manager(max_num_pages=16)
    picky = _Picky()
    # named to sort after "kv", so the runner reserves the KV first and the other pool refuses after it
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "picky": picky}, node_resources={NODE: ["kv", "picky"]}))
    _prefill(kv, "r", HELD + list(range(1000, 1008)))
    _prefill(kv, "x", list(range(6000, 6064)))
    fill = list(range(2000, 2000 + (kv._room() - 1) * PAGE_SIZE))
    _decoding(kv, "fill", fill, cap=4 * PAGE_SIZE)
    assert _ready(kv, "fill")
    _grow(kv, "fill", len(fill))
    _laid_out(kv, "head", list(range(3000, 3000 + 5 * PAGE_SIZE)))
    assert not _ready(kv, "head")
    # "p" passes into the one free page, writes it and finishes, so that page is kept for the head
    _prefill(kv, "p", HELD + list(range(4000, 4008)))
    kv.remove_request("p")
    # "x" finishes, so the head fits in the KV, but the other pool refuses it and the KV's allotment is undone
    kv.remove_request("x")
    picky.refused.add("head")
    assert not _ready(kv, "head")
    # "fill" grows into all but the kept page, and the other pool lets the head through, which the KV now refuses
    _grow(kv, "fill", 4 * PAGE_SIZE)
    picky.refused.clear()
    picky.moved += 1
    assert not _ready(kv, "head")
    _laid_out(kv, "q", HELD + list(range(5000, 5008)), keyed=True)

    assert not _ready(kv, "q"), "a request passed into the page kept for the head, lost when another pool refused it"


def test_no_request_passes_a_head_that_fits_now():
    kv = _manager(max_num_pages=16)
    head = list(range(3000, 3000 + 5 * PAGE_SIZE))
    _prefill(kv, "r", HELD + list(range(1000, 1008)))
    # "s" is admitted for the head's first two pages and hasn't written them
    _laid_out(kv, "s", head[:2 * PAGE_SIZE], keyed=True)
    assert _ready(kv, "s")
    _prefill(kv, "fill", list(range(2000, 2000 + (kv._room() - 3) * PAGE_SIZE)))
    _laid_out(kv, "head", head, keyed=True)
    assert not _ready(kv, "head")
    # "s" writes and files those pages, which the head now hits, so the head fits in the room it waited for
    _probe(kv, "s", 2 * PAGE_SIZE)
    _grow(kv, "s", 2 * PAGE_SIZE)
    _laid_out(kv, "p", HELD + list(range(4000, 4008)), keyed=True)
    _runner(kv).admit_retrieve("p", NODE, WALK, None)

    assert _ready(kv, "head"), "a request passed the head on the head's last ask, though the head fits now"


def test_a_prompt_bigger_than_the_pool_fails_as_it_joins():
    kv = _manager(max_num_pages=4)
    _laid_out(kv, "a", list(range(64)))

    outcome = _runner(kv).admit_retrieve("a", NODE, WALK)

    assert isinstance(outcome.reason, AdmitRuntimeError), "a request that can never fit holds the queue"
    assert not _runner(kv)._admission._waiting and not kv._asks


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
    overrides.unpausable, overrides.loop_caps = True, {"decode_loop": node.kv.config.request_pages * PAGE_SIZE - prompt}

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
    overrides.unpausable, overrides.loop_caps = True, {"decode_loop": cap}
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
    assert runner._admission._queues["kv_cache"][0] == "c", "\"d\" passed \"c\" in the self pool's queue"
    decode = self_kv.admit(KVStep(segments=(Segment("r", "main", 1),)), _ctx("r"))
    assert decode.ok, "requests waiting on the cross pool kept the room the running request decodes into"


def _cfg_parallel_pool(node: str, world_size: int = 1, rank: int = 0) -> KVManager:
    spec = KVSpec(
        resource_key="kv", nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"}, leader="LLM",
        config=PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096, max_num_pages=9, page_size=16),
    )
    group = None if world_size == 1 else SimpleNamespace(rank=rank, world_size=world_size)
    return KVManager.build(spec, EngineResourceInfo(
        device=torch.device("cpu"), kv_dtype=torch.float32, nodes=frozenset({node}), joint_comm_group=group,
    ))


def test_only_the_leaders_pool_on_rank_0_holds_a_request_back():
    for node, world_size, rank, holds in (
        ("LLM", 1, 0, True), ("LLM_cfg_text", 1, 0, False), ("LLM", 2, 0, True), ("LLM", 2, 1, False),
    ):
        kv = _cfg_parallel_pool(node, world_size, rank)
        for rid in ("a", "b"):
            _laid_out(kv, rid, list(range(80)))

        assert _ready(kv, "a") and _ready(kv, "b") is not holds, f"{node}'s pool on rank {rank} of {world_size}"
        assert holds or not (kv._allotments or kv._asks), "a pool that doesn't decide kept admission state"


def test_an_allotment_covers_only_the_labels_its_workers_nodes_open():
    kv = _cfg_parallel_pool("LLM")
    kv.ingest_request("a", KVReqConfig(
        kv_layout={label: _layout(16 * n)["main"] for label, n in (("main", 2), ("cfg_img", 1), ("cfg_text", 3))},
        needed_labels_per_node_walk={
            (NODE, WALK): ["main", "cfg_img"], ("LLM_cfg_text", "image_gen_cfg2"): ["cfg_text"],
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
    kv.after_output_copy(step)
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
    # the stub comes after `enable_prefix_cache`, so open the writes here
    kv._open_host_writes()

    log = _drive(kv, random.Random(SEED), ops=10_000)

    for op in ("committed", "rolled back", "dropped", "refused", "reset", "offload", "reload", "remove"):
        assert sum(line.startswith(op) for line in log) >= 10, f"seed {SEED}: too few {op!r} to say anything"
    assert kv._cpu_pool.written >= 10, f"seed {SEED}: too few pages written ahead of eviction to say anything"
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
    host._serve(kv, "c", list(range(900, 900 + 8 * host.PAGE_SIZE)))
    room = kv._room()
    host._ingest(kv, "big", list(range(500, 500 + (room + 1) * host.PAGE_SIZE)))
    assert not _ready(kv, "big")
    # only the host holds "b"'s prefix, and its output doesn't fit beside "c", so it can't pass "big"
    kv.local_loops = {"decode_loop"}
    tokens, page = host.A_PROMPT + [7, 8], host.PAGE_SIZE
    kv.ingest_request("b", KVReqConfig(
        prefix_keys={"main": chain([tokens[at:at + page] for at in range(0, len(tokens), page)])},
        prefix_tail={"main": tokens[len(tokens) // page * page:]}, loop_caps={"decode_loop": room * page},
        kv_layout={"main": StreamLayout(
            (PrefixSpan(len(tokens), len(tokens), WALK),), (("decode_loop", 1),), keyed_spans=1,
        )},
    ))
    assert not _ready(kv, "b")
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
