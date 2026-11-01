"""Copying cached pages to the host before the device evicts them.

Once the pool is full of cache, every prefill that isn't a whole hit evicts. A
cached page that only the index holds is copied to the host, so the next request
with the same prefix copies it back instead of recomputing it. The copy must never
be filed for a page that eviction reused mid-copy, or a later request would read
another request's bytes.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
from test_kv_host_index import (
    A_PROMPT,
    NODE,
    PAGE_SIZE,
    SHARED,
    WALK,
    _Event,
    _host_pages,
    _ingest,
    _land,
    _manager,
    _offloaded,
    _pages,
    _probe,
    _ready,
    _run,
    _serve,
    _stub_transfer,  # noqa: F401
)
from test_kv_prefix_offload import _commit, _step

from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.submodule_base import ARNodeInputs

B_PROMPT = SHARED + list(range(500, 500 + 2 * PAGE_SIZE))
# shares no page with `_offloaded`'s prompt
C_PROMPT = list(range(600, 600 + 6 * PAGE_SIZE))


def _cached(kv: KVManager, rid: str, prompt: list[int]) -> list[int]:
    """Serve and remove ``rid``, leaving its whole pages to the index, run a step, and return their keys."""
    _serve(kv, rid, prompt + [1, 2])
    keys = kv._streams[rid]["main"].chain.rooted[:len(prompt) // PAGE_SIZE]
    kv.remove_request(rid)
    _step(kv)
    return keys


def _on_host(kv: KVManager, keys: list[bytes]) -> int:
    return sum(kv._cpu_pool.index.page_for(key) is not None for key in keys)


def test_a_page_only_the_index_holds_is_copied_back_from_the_host_after_eviction():
    kv = _manager()
    _serve(kv, "a", A_PROMPT + [1, 2])
    pages = _pages(kv, "a")[:6]
    kv.remove_request("a")
    _step(kv)
    assert kv._cpu_pool.written_out == pages, "a's cached pages were not copied when a released them"
    _land(kv)
    kv._evict(6)

    _ingest(kv, "b", A_PROMPT + [7, 8])
    assert not _ready(kv, "b")
    _land(kv)
    assert _ready(kv, "b") and _probe(kv, "b", A_PROMPT + [7, 8]) == 6 * PAGE_SIZE
    assert kv._streams["b"]["main"].chain.from_host == 6, "b recomputed a prefix the host had a copy of"


def test_a_page_with_a_held_page_below_it_is_copied_once_only_the_index_holds_it():
    kv = _manager()
    _serve(kv, "seed", SHARED + [1])
    prompt = SHARED + list(range(100, 100 + 2 * PAGE_SIZE))
    # both lease the seed's pages before either runs, so b files its later pages under a's
    _probe(kv, "a", prompt + [7])
    _probe(kv, "b", prompt + list(range(300, 300 + 2 * PAGE_SIZE)) + [7])
    _run(kv, "a", 2 * PAGE_SIZE + 1)
    _run(kv, "b", 4 * PAGE_SIZE + 1)
    raced = _pages(kv, "a")[4:6]

    kv.remove_request("a")
    _step(kv)

    # the seed's prefix goes first, so the raced pages land on the host under it
    assert kv._cpu_pool.written_out == _pages(kv, "seed")[:4] + raced, (
        "a page b's held pages hang under was never copied"
    )


def test_a_page_lands_on_the_host_under_a_prefix_another_request_still_holds():
    kv = _manager()
    _serve(kv, "b", SHARED + list(range(300, 300 + 2 * PAGE_SIZE)) + [1])
    prompt = SHARED + list(range(100, 100 + 2 * PAGE_SIZE))
    _serve(kv, "a", prompt + [1, 2])
    pages = _pages(kv, "a")[:6]
    keys = kv._streams["a"]["main"].chain.rooted[:6]
    kv.remove_request("a")
    _step(kv)
    assert kv._cpu_pool.written_out == pages, "a's pages were copied before the prefix b holds"
    _land(kv)

    host = kv._cpu_pool.index
    parents = [host.parent_key(host.page_for(key)) for key in keys]
    assert parents == [None, *keys[:5]], "a page landed on the host with no parent while its prefix was cached"


def test_a_write_never_evicts_the_host_page_its_pages_land_under():
    kv = _manager(host_pages=5)
    shared = _cached(kv, "a", SHARED)
    _land(kv)
    _serve(kv, "c", SHARED + list(range(100, 100 + 2 * PAGE_SIZE)) + [1, 2])
    own = kv._streams["c"]["main"].chain.rooted[4:6]
    kv.remove_request("c")
    # one host page is free, and the shared prefix's last page is the host's only leaf
    _step(kv)
    _land(kv)

    host = kv._cpu_pool.index
    assert host.page_for(shared[-1]) is not None, "a write evicted the host page its first page lands under"
    assert host.parent_key(host.page_for(own[0])) == shared[-1]


def test_a_key_on_the_host_or_on_its_way_there_is_not_copied_again():
    kv = _manager()
    keys = _cached(kv, "a", A_PROMPT)
    _cached(kv, "b", A_PROMPT)
    assert len(kv._cpu_pool.written_out) == 6, "a page was copied again while its first copy was in flight"
    _land(kv)
    _cached(kv, "c", A_PROMPT)
    assert len(kv._cpu_pool.written_out) == 6 and _on_host(kv, keys) == 6, "a page the host holds was copied again"


@pytest.mark.parametrize("grant", ["alloc", "reload"])
def test_a_page_evicted_mid_copy_is_reused_at_once_and_its_copy_files_nothing(grant):
    kv = _manager(max_num_pages=16)
    if grant == "reload":
        _offloaded(kv, "x")
        kv._evict(6)
    keys = _cached(kv, "a", C_PROMPT)
    dropped = kv._cpu_pool.dropped

    kv.ingest_request("hog", KVReqConfig())
    if grant == "alloc":
        _run(kv, "hog", (kv._arena.num_free + 6) * PAGE_SIZE)
    else:
        _run(kv, "hog", (kv._arena.num_free - 1) * PAGE_SIZE)
        assert kv.reload("x"), "the reload waited on a's copies"
    _land(kv)

    assert _on_host(kv, keys) == 0, "the host filed a page eviction reused during its copy"
    assert kv._cpu_pool.dropped - dropped == 6 and kv._cpu_pool.cancelled == 6
    kv.assert_pages_conserved()


def test_a_write_that_finished_before_its_pages_were_evicted_is_filed():
    kv = _manager()
    keys = _cached(kv, "a", A_PROMPT)
    # the copy has run, and no step has admitted since to land it
    kv._cpu_pool.events[-1].done = True
    kv._evict(6)
    _land(kv)

    assert _on_host(kv, keys) == 6, "eviction cancelled a write that had already copied its pages"


def test_a_page_evicted_while_the_copies_launch_files_nothing():
    kv = _manager()
    # another thread evicts the pages once `after_output_copy` has dropped the lock to launch the copies
    kv._cpu_pool.write = lambda gpu_kv_cache, copies: kv._evict(len(copies.gpu))
    keys = _cached(kv, "a", A_PROMPT)
    _land(kv)

    assert _on_host(kv, keys) == 0, "a page evicted before its copy was recorded was filed on the host"
    assert kv._cpu_pool.arena.num_free == 64, "the cancelled write kept its host pages"


def test_writes_land_once_a_step_and_never_in_a_readiness_check(monkeypatch):
    kv = _manager()
    _cached(kv, "a", A_PROMPT)
    queries = []
    monkeypatch.setattr(_Event, "query", lambda event: queries.append(event) or False)
    _ingest(kv, "b", B_PROMPT)

    _ready(kv, "b")
    assert queries == [], "a readiness check polled a write its request doesn't wait on"
    _run(kv, "b", len(B_PROMPT))

    assert len(queries) == 1, f"a step polled one drain of six pages {len(queries)} times"


def test_a_step_writes_at_most_its_share_and_the_rest_waits_for_later_steps():
    kv = _manager()
    kv._host_writes._per_step = 2
    _serve(kv, "a", A_PROMPT + [1, 2])
    pages = _pages(kv, "a")[:6]
    kv.remove_request("a")
    _step(kv)
    assert kv._cpu_pool.written_out == pages[:2], "a step wrote more than its share"

    _step(kv)

    assert kv._cpu_pool.written_out == pages[:4], "the pages over a step's share didn't wait for the next step"


def test_a_page_is_copied_only_once_a_step_committed_after_it_has_finished():
    kv = _manager()
    _cached(kv, "x", C_PROMPT)
    written = len(kv._cpu_pool.written_out)
    earlier = _commit(kv)
    _serve(kv, "a", A_PROMPT + [1, 2])
    kv.remove_request("a")

    kv.after_output_copy(earlier)
    assert len(kv._cpu_pool.written_out) == written, "a page was copied once a step committed before it finished"
    _step(kv)

    assert len(kv._cpu_pool.written_out) == written + 6, "the pages waited past the step committed after them"


def test_a_queued_key_released_again_keeps_its_place_and_its_commit():
    kv = _manager()
    kv._host_writes._per_step = 1
    _serve(kv, "a", A_PROMPT + [1, 2])
    pages = _pages(kv, "a")[:6]
    kv.remove_request("a")
    step = _commit(kv)
    kv.after_output_copy(step)
    kv.after_output_copy(step)
    assert kv._cpu_pool.written_out == pages[:2]
    # b leases the queued pages and drops them after a commit the device hasn't finished
    _probe(kv, "b", A_PROMPT + [7])
    _commit(kv)
    kv.remove_request("b")

    kv.after_output_copy(step)

    assert kv._cpu_pool.written_out == pages[:3], "a key released again waited for the newest commit"


def test_a_copy_from_the_host_that_lands_is_not_copied_back():
    kv = _manager()
    _serve(kv, "a", A_PROMPT + [1, 2])
    kv._arena.writes = None
    assert kv.offload("a") > 0
    kv._arena.writes = kv._host_writes
    kv._evict(6)
    _ingest(kv, "b", A_PROMPT + [7, 8])
    assert not _ready(kv, "b")
    _land(kv)

    assert _ready(kv, "b")
    _step(kv)
    assert kv._cpu_pool.written_out == [], "pages that came from the host were copied back to it"


def test_a_pending_page_evicted_and_reused_before_the_drain_copies_nothing():
    kv = _manager(max_num_pages=16)
    kv._arena.writes = None
    _cached(kv, "a", B_PROMPT)
    kv._arena.writes = kv._host_writes
    # a lease dropped at prepare queues the pages' keys, and no `after_output_copy` runs
    # before the hog evicts them
    _ingest(kv, "b", B_PROMPT + [7])
    inputs = ARNodeInputs(input_seq_len=len(B_PROMPT) + 1)
    assert kv.resolve_cached_prefix("b", NODE, WALK, inputs) is not None
    kv.apply_cached_prefix("b", NODE, WALK, inputs, None)
    kv.ingest_request("hog", KVReqConfig())
    _run(kv, "hog", kv._arena.num_free * PAGE_SIZE + 6 * PAGE_SIZE)

    kv.remove_request("hog")
    _step(kv)

    assert kv._cpu_pool.written_out == [], "a page eviction reused was copied under its old key"
    kv.assert_pages_conserved()


def test_a_copier_that_raises_fails_no_request_files_nothing_and_frees_its_host_pages():
    kv = _manager()

    def fails(gpu_kv_cache, copies):
        raise RuntimeError("copy failed")

    kv._cpu_pool.write = fails
    _serve(kv, "a", A_PROMPT + [1, 2])
    keys = kv._streams["a"]["main"].chain.rooted[:6]
    kv.remove_request("a")
    _step(kv)
    assert kv._cpu_pool.arena.num_free == 58, "the host pages went back before the copies that write them landed"
    _land(kv)

    assert _on_host(kv, keys) == 0 and kv._cpu_pool.arena.num_free == 64


def test_a_full_host_evicts_cached_pages_for_a_write_and_never_an_offloaded_requests():
    kv = _manager(host_pages=8)
    _offloaded(kv, "x")
    held = list(_host_pages(kv, "x"))
    keys = _cached(kv, "a", C_PROMPT)
    assert _host_pages(kv, "x") == held and kv._host_writes.queued == {}, "the drain took x's host pages"
    _land(kv)
    assert _on_host(kv, keys) == 1, "a key with no host page left was copied"

    other = _cached(kv, "c", list(range(900, 900 + PAGE_SIZE)))
    _land(kv)
    assert _on_host(kv, keys) == 0 and _on_host(kv, other) == 1, "the write did not evict the host's cached page"


def test_a_heads_host_run_survives_a_drain_while_its_copy_is_in_flight():
    kv = _manager(host_pages=7)
    _serve(kv, "a", A_PROMPT + [1, 2])
    kv._arena.writes = None
    assert kv.offload("a") > 0
    kv.remove_request("a")
    kv._arena.writes = kv._host_writes
    kv._evict(6)
    # `after_output_copy` needs host pages, and only a's cached prefix could give them
    _serve(kv, "c", B_PROMPT + [1, 2])
    kv._streams["c"]["main"].page_indices, pages = [], kv._streams["c"]["main"].page_indices
    kv._release("c", kv._streams["c"]["main"], pages)
    _ingest(kv, "b", A_PROMPT + [7, 8])
    assert not _ready(kv, "b")

    _step(kv)

    _land(kv)
    assert _ready(kv, "b") and _probe(kv, "b", A_PROMPT + [7, 8]) == 6 * PAGE_SIZE, "a write evicted b's prefix"


def test_an_offload_that_files_a_key_during_its_write_keeps_one_host_page():
    kv = _manager()
    keys = _cached(kv, "a", A_PROMPT)
    _serve(kv, "b", A_PROMPT + [3, 4])
    assert kv.offload("b") > 0
    _land(kv)

    assert _on_host(kv, keys) == 6
    assert kv._cpu_pool.arena.num_free == 64 - 7, "the write's page under a key the offload filed was kept"
    kv.assert_pages_conserved()


def test_a_request_removed_while_its_pages_are_copied_leaks_nothing(monkeypatch):
    monkeypatch.setattr("mstar.engine.resources.kv.manager._DEBUG_ASSERTS", True)
    kv = _manager()
    _cached(kv, "a", A_PROMPT)
    _probe(kv, "b", A_PROMPT + [7])
    kv.remove_request("b")
    _step(kv)
    _land(kv)

    assert kv._host_writes._writing == {} and kv._host_writes._in_flight == []
    kv.assert_pages_conserved()


def test_a_key_written_again_after_a_cancelled_write_is_cancelled_by_its_next_eviction():
    kv = _manager(max_num_pages=16)
    keys = _cached(kv, "a", C_PROMPT)
    first = kv._cpu_pool.events[-1]
    kv._evict(6)
    _cached(kv, "b", C_PROMPT)
    assert len(kv._cpu_pool.written_out) == 12, "a key whose write was cancelled was not written again"
    first.done = True
    with kv._lock:
        kv._host_writes.land()

    kv._evict(6)
    _land(kv)

    assert _on_host(kv, keys) == 0, "the host filed pages eviction reused during their second copy"


def test_eviction_takes_pages_still_waiting_for_their_copy_last():
    kv = _manager()
    take, kv._cpu_pool.take = kv._cpu_pool.take, lambda n: []
    # the drain finds no host page for d's keys and drops them, so no copy is coming
    uncopied = _cached(kv, "d", list(range(700, 700 + 6 * PAGE_SIZE)))
    kv._cpu_pool.take = take
    copied = _cached(kv, "a", B_PROMPT)
    _land(kv)
    # e's write is issued and never lands, and c's is never issued
    writing = _cached(kv, "e", list(range(800, 800 + 6 * PAGE_SIZE)))
    kv._host_writes._per_step = 0
    waiting = _cached(kv, "c", C_PROMPT)
    # b's hit makes the pages of a's prefix the newest, and b's release queues those pages' keys again
    _serve(kv, "b", B_PROMPT + [7, 8])
    kv.remove_request("b")

    kv._evict(6)
    assert kv._index.pages_for(uncopied) == [], "pages with no copy coming outlived pages the host has"
    kv._evict(6)
    left = [len(kv._index.pages_for(keys)) for keys in (copied, writing, waiting)]
    assert left == [0, 6, 6], "eviction dropped pages waiting for their copy while pages the host has were left"
