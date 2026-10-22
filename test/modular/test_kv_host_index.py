"""Offloading pages the prefix index shares."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager, PageArena
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs

PAGE_SIZE = 4
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"
SHARED = list(range(4 * PAGE_SIZE))


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


class _Event:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


class _HostPool(CPUPagePool):
    def __init__(self, max_cpu_pages: int):
        self.arena = PageArena(kv_cache=None, allocator=PageAllocator(max_cpu_pages))
        self.index = PrefixIndex(self.arena)
        self.offloaded = {}
        self._refused_logged_at = None
        self._refused_unlogged = 0
        self.dropped = 0
        self._dropped_logged_at = None
        self.copied_out: list[int] = []
        self.copied_in: list[int] = []
        self.copied_in_from: list[tuple[int, int]] = []
        self.events: list[_Event] = []

    def copy_out(self, gpu_kv_cache, copies):
        self.copied_out.extend(copies.gpu)

    def copy_in(self, gpu_kv_cache, copies):
        self.copied_in.extend(copies.gpu)
        self.copied_in_from.extend(zip(copies.gpu, copies.cpu, strict=True))

    def record(self):
        self.events.append(_Event())
        return self.events[-1]

    def land(self):
        for event in self.events:
            event.done = True

    def sync(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 128, host_pages: int = 64, index: bool = True) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    if index:
        kv.enable_prefix_cache(ROOT)
    kv._cpu_pool = _HostPool(host_pages)
    return kv


def _ingest(kv: KVManager, rid: str, tokens: list[int]) -> None:
    whole = len(tokens) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain([tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        prefix_layout={"main": [PrefixSpan(len(tokens), len(tokens), WALK)]},
    ))


def _probe(kv: KVManager, rid: str, tokens: list[int]) -> int:
    _ingest(kv, rid, tokens)
    inputs = ARNodeInputs(input_seq_len=len(tokens))
    prefix = kv.resolve_cached_prefix(rid, NODE, WALK, inputs)
    kv.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
    return prefix.tokens if prefix is not None else 0


def _run(kv: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _serve(kv: KVManager, rid: str, tokens: list[int]) -> None:
    _run(kv, rid, len(tokens) - _probe(kv, rid, tokens))


def _pages(kv: KVManager, rid: str) -> list[int]:
    return kv._streams[rid]["main"].page_indices


def _host_pages(kv: KVManager, rid: str) -> list[int]:
    return kv._cpu_pool.offloaded[rid]["main"].cpu_page_indices


def test_a_prefix_sixteen_requests_share_is_stored_once():
    kv = _manager(host_pages=64)
    for i in range(16):
        _serve(kv, f"r{i}", SHARED + [1000 * (i + 1) + t for t in range(2 * PAGE_SIZE + 2)])

    for i in range(16):
        assert kv.offload(f"r{i}") > 0, f"r{i} did not move to the host"

    assert len({tuple(_host_pages(kv, f"r{i}")[:4]) for i in range(16)}) == 1, (
        "the requests' shared pages are not the same host pages"
    )
    assert kv._cpu_pool.arena.num_free == 64 - (4 + 16 * 3), (
        "the host holds the shared prefix more than once"
    )
    assert len(kv._cpu_pool.copied_out) == 4 + 16 * 3, "a shared page was copied more than once"


def test_an_offload_whose_keyed_pages_are_all_on_the_host_copies_nothing():
    kv = _manager()
    prompt = SHARED + list(range(100, 100 + 2 * PAGE_SIZE))
    _serve(kv, "a", prompt)
    _serve(kv, "b", prompt + [7])
    assert kv.offload("b") > 0, "b did not move to the host"
    copied = len(kv._cpu_pool.copied_out)

    freed = kv.offload("a")

    assert kv.is_offloaded("a"), "an offload that had nothing to copy was not filed"
    assert len(kv._cpu_pool.copied_out) == copied, "pages already on the host were copied again"
    assert freed == 6, "the offload did not count the pages it left to the index"
    assert _host_pages(kv, "a") == _host_pages(kv, "b")[:6]
    kv.assert_pages_conserved()


def test_a_victim_whose_pages_another_request_holds_reclaims_nothing():
    kv = _manager()
    prompt = SHARED + list(range(100, 100 + 2 * PAGE_SIZE))
    _serve(kv, "a", prompt)
    _serve(kv, "b", prompt + [7])

    assert kv.reclaimable("a") == 0, (
        "a request whose every page b still reads was offered as an eviction victim"
    )
    assert kv.reclaimable("b") == len(_pages(kv, "b")), "b holds a page of its own, so it is a victim"


def test_a_page_that_lost_the_insert_race_goes_by_request_id():
    kv = _manager()
    _serve(kv, "seed", SHARED + [1])
    prompt = SHARED + list(range(100, 100 + 2 * PAGE_SIZE)) + [7]
    # both probe before either runs, so both write the same next two pages
    for rid in ("a", "b"):
        _probe(kv, rid, prompt)
    for rid in ("a", "b"):
        _run(kv, rid, len(prompt) - len(SHARED))
    own = _pages(kv, "b")[4:]

    assert kv.offload("b") > 0, "b did not move to the host"

    state = kv._cpu_pool.offloaded["b"]["main"]
    assert len(state.keys) == 4, "the keyed run went past the page b lost the race for"
    assert kv._cpu_pool.copied_out[4:] == own, "b's own copies of the raced pages were not the ones copied"
    kv.assert_pages_conserved()


def test_a_full_host_evicts_cached_pages_and_never_an_offloaded_requests():
    kv = _manager(host_pages=8)
    _serve(kv, "a", SHARED + list(range(100, 100 + 2 * PAGE_SIZE + 2)))
    assert kv.offload("a") > 0 and kv.reload("a"), "a did not round-trip"
    _serve(kv, "b", list(range(500, 500 + 6 * PAGE_SIZE + 2)))
    assert kv.offload("b") > 0, "b was refused while the host held only cached pages"
    held = list(_host_pages(kv, "b"))

    _serve(kv, "c", list(range(900, 900 + 6 * PAGE_SIZE + 2)))

    assert kv.offload("c") == 0, "c fit on a host that b fills"
    assert _host_pages(kv, "b") == held
    assert all(kv._cpu_pool.arena.num_owners[page] for page in held), (
        "making room for c freed a page b is still offloaded on"
    )


def test_a_stream_with_released_pages_is_filed_by_request():
    kv = _manager()
    _serve(kv, "a", SHARED + list(range(100, 100 + 2 * PAGE_SIZE + 2)))
    kv.protect_prefix("a", 4 * PAGE_SIZE)
    assert kv.release_oldest("a", 2 * PAGE_SIZE) == 2 * PAGE_SIZE

    assert kv.offload("a") > 0, "a did not move to the host"

    assert kv._cpu_pool.offloaded["a"]["main"].keys == [], (
        "pages were filed by key after a release shifted them under the chain"
    )
    assert len(kv._cpu_pool.copied_out) == 5
    assert kv.reload("a")
    assert len(kv._cpu_pool.copied_in) == 5
    kv.assert_pages_conserved()


def test_without_the_index_every_page_is_copied_and_filed_by_request():
    kv = _manager(index=False)
    _serve(kv, "a", SHARED + [1, 2])

    assert kv.offload("a") == 5
    assert kv._cpu_pool.offloaded["a"]["main"].keys == []
    assert kv._cpu_pool.index.num_pages() == 0, "a page was filed by key with the index off"
    assert kv.reload("a")
    assert len(kv._cpu_pool.copied_in) == 5
    assert kv._cpu_pool.arena.num_free == 64, "the reload left host pages behind"


def _offloaded(kv: KVManager, rid: str = "a") -> list[int]:
    _serve(kv, rid, SHARED + list(range(100, 100 + 2 * PAGE_SIZE + 2)))
    pages = list(_pages(kv, rid))
    assert kv.offload(rid) > 0, f"{rid} did not move to the host"
    return pages


def test_a_reload_keeps_the_pages_the_index_still_holds_and_copies_the_rest():
    kv = _manager(max_num_pages=16)
    before = _offloaded(kv)
    # leave one free page, so a reload that evicted first would drop a's cached pages
    kv.ingest_request("hog", KVReqConfig())
    _run(kv, "hog", (kv._arena.num_free - 1) * PAGE_SIZE)

    assert kv.reload("a"), "a did not come back"

    assert _pages(kv, "a")[:6] == before[:6], "the reload did not take back what the index held"
    assert kv._cpu_pool.copied_in == _pages(kv, "a")[6:], "pages the index held were copied again"
    kv.assert_pages_conserved()


def test_a_reload_of_two_labels_gives_each_label_its_own_host_pages():
    kv = _manager()
    _serve(kv, "a", SHARED + list(range(100, 100 + 2 * PAGE_SIZE + 2)))
    step = KVStep(segments=(Segment("a", "cfg", 3 * PAGE_SIZE),))
    ctx = StepContext(request_ids=("a",), graph_walk=WALK, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    assert kv.offload("a") > 0, "a did not move to the host"
    host = {label: list(state.cpu_page_indices) for label, state in kv._cpu_pool.offloaded["a"].items()}
    kv._index.evict(2)

    assert kv.reload("a"), "a did not come back"

    source = dict(kv._cpu_pool.copied_in_from)
    for label, pages in host.items():
        copied = [source[page] for page in kv._streams["a"][label].page_indices if page in source]
        assert copied == pages[len(pages) - len(copied):], f"{label} was copied from another stream's host pages"
    kv.assert_pages_conserved()


def test_keyed_pages_copied_back_enter_the_index_at_once():
    kv = _manager()
    _offloaded(kv)
    keys = list(kv._cpu_pool.offloaded["a"]["main"].keys)
    kv._index.evict(2)

    assert kv.reload("a"), "a did not come back"

    assert [kv._index.page_for(key) for key in keys[4:]] == _pages(kv, "a")[4:6], (
        "the pages copied back were not indexed by the reload"
    )
    assert _probe(kv, "b", SHARED + list(range(100, 100 + 2 * PAGE_SIZE + 2))) == 6 * PAGE_SIZE, (
        "a request with a's prefix did not hit the reloaded pages"
    )
    kv.assert_pages_conserved()


def test_a_reload_undone_at_once_still_offloads():
    kv = _manager()
    _offloaded(kv)
    kv._index.evict(2)
    assert kv.reload("a"), "a did not come back"

    # what `Engine.reload_request` does when a later resource refuses
    assert kv.offload("a") > 0, "the undo of a partial reload moved nothing"

    assert kv.is_offloaded("a")
    kv.assert_pages_conserved()
    assert kv.reload("a"), "a did not come back after the undo"
    kv.assert_pages_conserved()


A_PROMPT = SHARED + list(range(100, 100 + 2 * PAGE_SIZE))


def _on_the_host_only(kv: KVManager) -> None:
    _serve(kv, "a", A_PROMPT + [1, 2])
    assert kv.offload("a") > 0, "a did not move to the host"
    kv._index.evict(6)
    assert kv._index.num_pages() == 0


def _ready(kv: KVManager, rid: str) -> bool:
    out = kv.admit_retrieve(rid, NODE, WALK, None)
    assert out.ok
    return out.ready


def _land(kv: KVManager) -> None:
    kv._cpu_pool.land()
    with kv._lock:
        kv._land_host_copies()


def test_a_prefix_on_the_host_is_copied_back_before_the_request_runs():
    kv = _manager()
    _on_the_host_only(kv)
    _ingest(kv, "b", A_PROMPT + [7, 8])

    assert not _ready(kv, "b"), "b was ready before its prefix came back from the host"
    assert len(kv._cpu_pool.copied_in) == 6
    kv._cpu_pool.land()
    assert _ready(kv, "b"), "b stayed not ready after its copy landed"

    assert _probe(kv, "b", A_PROMPT + [7, 8]) == 6 * PAGE_SIZE, "b's probe missed the pages copied back"
    assert kv._streams["b"]["main"].chain.from_host == 6
    assert kv.prefix_tokens[NODE] == (2 * len(A_PROMPT) + 4, 6 * PAGE_SIZE), "b's hit was not counted"
    assert kv._streams["b"]["main"].hold is None, "the pages copied back are still held for b"
    kv.assert_pages_conserved()


def test_a_second_request_on_keys_still_arriving_waits_for_that_copy():
    kv = _manager()
    _on_the_host_only(kv)
    _ingest(kv, "b", A_PROMPT + [7, 8])
    assert not _ready(kv, "b")
    _ingest(kv, "c", A_PROMPT + [9])

    assert not _ready(kv, "c"), "c was ready while the pages it hits were still arriving"
    assert len(kv._cpu_pool.copied_in) == 6, "c copied pages another request was already copying"
    kv._cpu_pool.land()
    assert _ready(kv, "c")

    assert _probe(kv, "c", A_PROMPT + [9]) == 6 * PAGE_SIZE, "c did not hit the pages b's copy brought in"
    kv.assert_pages_conserved()


def test_a_request_with_no_free_device_page_runs_without_the_host():
    kv = _manager(max_num_pages=16)
    _on_the_host_only(kv)
    kv.ingest_request("hog", KVReqConfig())
    _run(kv, "hog", kv._arena.num_free * PAGE_SIZE)
    _ingest(kv, "b", A_PROMPT + [7, 8])

    assert _ready(kv, "b"), "b waited for room that only eviction would make"
    assert kv._cpu_pool.copied_in == [], "a copy was queued with no free page to land in"


def test_a_request_removed_while_its_prefix_is_arriving_leaks_nothing():
    kv = _manager()
    _on_the_host_only(kv)
    _ingest(kv, "b", A_PROMPT + [7, 8])
    assert not _ready(kv, "b")
    arriving = list(kv._streams["b"]["main"].hold)

    kv.remove_request("b")

    assert all(kv._arena.num_owners[page] for page in arriving), (
        "a page still being copied into went back to the free list"
    )
    _land(kv)
    assert kv._host_copies == [] and kv._arriving == {}
    kv.assert_pages_conserved()


def test_a_copy_whose_parent_left_the_index_is_freed_when_it_lands():
    kv = _manager()
    _offloaded(kv)
    keys = list(kv._cpu_pool.offloaded["a"]["main"].keys)
    # a's own two keyed pages leave the device index, so b's prefix continues on the host
    kv._index.evict(2)
    _ingest(kv, "b", A_PROMPT + [7, 8])
    assert not _ready(kv, "b")
    arriving = kv._host_copies[0].pages
    kv.remove_request("b")
    # the four shared pages are now held by the index alone, and the last of them is the copy's parent
    kv._index.evict(4)
    assert kv._index.page_for(keys[3]) is None

    _land(kv)

    assert [kv._index.page_for(key) for key in keys[4:6]] == [None, None], (
        "the pages were indexed under a parent the index no longer holds"
    )
    assert not any(kv._arena.num_owners[page] for page in arriving), "a page copied in was never freed"
    kv.assert_pages_conserved()


def test_a_cached_page_evicted_with_no_copy_on_the_host_is_counted():
    kv = _manager(max_num_pages=16)
    _serve(kv, "a", A_PROMPT + [1, 2])
    assert kv.offload("a") > 0
    kv.reload("a")
    kv.remove_request("a")
    _serve(kv, "b", list(range(500, 500 + 4 * PAGE_SIZE)))
    kv.remove_request("b")

    kv.ingest_request("hog", KVReqConfig())
    _run(kv, "hog", (kv._arena.num_free + 10) * PAGE_SIZE)

    assert kv._cpu_pool.dropped == 4, "pages the host had a copy of were counted, or b's were missed"


def test_a_leaked_host_page_fails_the_conservation_check():
    kv = _manager()
    _serve(kv, "a", A_PROMPT + [1, 2])
    assert kv.offload("a") > 0
    kv._cpu_pool.arena.acquire(1)

    with pytest.raises(AssertionError, match="host owner counts disagree"):
        kv.assert_pages_conserved()


def test_a_host_page_both_free_and_held_fails_the_conservation_check():
    kv = _manager()
    _serve(kv, "a", A_PROMPT + [1, 2])
    assert kv.offload("a") > 0
    kv._cpu_pool.arena.allocator.free(_host_pages(kv, "a")[-1:])

    with pytest.raises(AssertionError, match="pages both free and owned in the host pool"):
        kv.assert_pages_conserved()
