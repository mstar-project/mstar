"""Caching must never be the reason a running request cannot allocate.

The pool is a fixed tensor, so cached pages cost nothing until it is full — and
then they cost everything, because a page held only by the index looks exactly
like a page in use to whoever is asking for one. The rule that keeps the old
behaviour intact is that a shortfall gives those pages back before it becomes a
failure: a request fails to allocate only when the pool is genuinely full of
live requests, which is when it would have failed before any of this existed.

What must not be given back is a page someone is still using. A page the index
shares with a live request or with a lease frees nothing if its entry is
dropped, and the entry was still good, so it is passed over.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"
PAGES_PER_REQUEST = 4
TOKENS = PAGE_SIZE * PAGES_PER_REQUEST


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _tokens(first: int) -> list[int]:
    return list(range(first, first + TOKENS))


def _ingest(kv: KVManager, rid: str, tokens: list[int]) -> None:
    pages = [
        tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
    ]
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain(pages)}, prefix_tail={"main": []},
    ))


def _admit(kv: KVManager, rid: str, span: int):
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
    )
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    return outcome


def _finished(kv: KVManager, rid: str, tokens: list[int]) -> None:
    """A request that ran to the end and left its pages in the index."""
    _ingest(kv, rid, tokens)
    assert _admit(kv, rid, len(tokens)).ok
    kv.remove_request(rid)


# ── the shortfall ───────────────────────────────────────────────────────


def test_a_live_request_never_fails_while_cache_only_pages_exist():
    # room for two requests and the sink, and two requests' worth of cache
    kv = _manager(2 * PAGES_PER_REQUEST + 1)
    _finished(kv, "a", _tokens(0))
    _finished(kv, "b", _tokens(1000))
    assert kv._arena.num_free < PAGES_PER_REQUEST, (
        "the pool was not short, so this would not have exercised anything"
    )

    _ingest(kv, "c", _tokens(2000))
    outcome = _admit(kv, "c", TOKENS)

    assert outcome.ok, (
        "a request failed to allocate while the pool held nothing but cache"
    )
    kv.assert_pages_conserved()


def test_a_request_still_fails_when_the_pool_is_full_of_live_ones():
    kv = _manager(PAGES_PER_REQUEST + 1)
    _ingest(kv, "a", _tokens(0))
    assert _admit(kv, "a", TOKENS).ok, "the request under test never ran"

    _ingest(kv, "b", _tokens(1000))
    outcome = _admit(kv, "b", TOKENS)

    assert not outcome.ok, (
        "eviction found pages that belonged to a request that is still running"
    )
    kv.assert_pages_conserved()


# ── what eviction must not take ─────────────────────────────────────────


def test_a_leased_page_is_never_evicted():
    kv = _manager(2 * PAGES_PER_REQUEST + 1)
    tokens = _tokens(0)
    _finished(kv, "a", tokens)
    _ingest(kv, "b", tokens)
    matched = kv.resolve_cached_prefix("b", NODE, WALK)
    leased = list(kv._streams["b"]["main"].lease)
    assert matched, "the second request did not match, so nothing was leased"

    # somebody else asks for everything the pool can give
    kv._index.evict(kv.config.max_num_pages)

    assert kv._streams["b"]["main"].lease == leased, (
        "the lease lost pages to an eviction it was supposed to hold them from"
    )
    assert all(page not in kv._arena.allocator.free_pages.queue for page in leased), (
        "a page a request was about to be admitted on was handed to someone else"
    )
    kv.assert_pages_conserved()


def test_a_pool_of_live_leaves_gives_up_after_one_look_at_each():
    kv = _manager(2 * PAGES_PER_REQUEST + 1)
    _ingest(kv, "a", _tokens(0))
    assert _admit(kv, "a", TOKENS).ok, "the request under test never ran"
    index = kv._index
    popped: list[int] = []
    real_pop = index._pop_leaf

    def _counting_pop():
        entry = real_pop()
        if entry is not None:
            popped.append(entry[1])
        return entry

    index._pop_leaf = _counting_pop
    freed = index.evict(kv.config.max_num_pages)

    assert freed == 0, "a page a running request holds was counted as freed"
    assert len(popped) == len(set(popped)), (
        f"a leaf that was passed over was looked at again in the same pass: "
        f"{popped}"
    )
    assert set(popped) <= set(kv._streams["a"]["main"].page_indices), (
        "a leaf no running request holds was passed over"
    )
    kv.assert_pages_conserved()


def test_a_leaf_that_was_passed_over_is_still_there_for_the_next_shortfall():
    kv = _manager(2 * PAGES_PER_REQUEST + 1)
    tokens = _tokens(0)
    _ingest(kv, "a", tokens)
    assert _admit(kv, "a", TOKENS).ok, "the request under test never ran"
    indexed = sorted(kv._index.pages())

    assert kv._index.evict(PAGES_PER_REQUEST) == 0, (
        "a pool whose every leaf is live gave something back"
    )
    kv.remove_request("a")

    assert kv._index.evict(PAGES_PER_REQUEST) == PAGES_PER_REQUEST, (
        "the leaves the first pass put back could not be found by the second"
    )
    assert sorted(kv._index.pages()) == [] and indexed, "nothing was indexed"
    kv.assert_pages_conserved()


def test_passing_over_a_leaf_leaves_its_parent_alone():
    kv = _manager(2 * PAGES_PER_REQUEST + 1)
    _ingest(kv, "a", _tokens(0))
    assert _admit(kv, "a", TOKENS).ok, "the request under test never ran"
    index = kv._index
    chain_pages = kv._streams["a"]["main"].page_indices[:PAGES_PER_REQUEST]
    parent, leaf = chain_pages[-2], chain_pages[-1]
    before = index._children[parent]

    index.evict(1)

    assert index._children[parent] == before, (
        "a leaf that was passed over decremented its parent anyway"
    )
    assert index.page_for(index._key[leaf]) == leaf, "the leaf was removed"
    assert index.page_for(index._key[parent]) == parent, (
        "the walk went up from a leaf it never removed"
    )
    kv.assert_pages_conserved()
