"""Offloading pages the prefix index shares."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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
        self.events: list[_Event] = []

    def copy_out(self, gpu_kv_cache, copies):
        self.copied_out.extend(copies.gpu)

    def copy_in(self, gpu_kv_cache, copies):
        self.copied_in.extend(copies.gpu)

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
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: SimpleNamespace(
        get_kv_transfer_info=lambda **kw: None, owns_transfer_info=lambda *a, **kw: False,
        remove_request=lambda rid: None, start_async_retrieve=lambda **kw: None, cleanup=lambda: None,
    ))


def _manager(max_num_pages: int = 128, host_pages: int = 64) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
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

    assert kv._cpu_pool.arena.num_free == 64 - (4 + 16 * 3), (
        "the host holds the shared prefix more than once"
    )
    assert len(kv._cpu_pool.copied_out) == 4 + 16 * 3, "a shared page was copied more than once"



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
    assert kv.reload("a")
    kv.assert_pages_conserved()



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



def test_keyed_pages_copied_back_enter_the_index_at_once():
    kv = _manager()
    _offloaded(kv)
    kv._index.evict(2)

    assert kv.reload("a"), "a did not come back"

    assert _probe(kv, "b", SHARED + list(range(100, 100 + 2 * PAGE_SIZE + 2))) == 6 * PAGE_SIZE, (
        "a request with a's prefix did not hit the reloaded pages"
    )
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
    kv._cpu_pool.land()
    assert _ready(kv, "b"), "b stayed not ready after its copy landed"

    assert _probe(kv, "b", A_PROMPT + [7, 8]) == 6 * PAGE_SIZE, "b's probe missed the pages copied back"
    assert kv._streams["b"]["main"].hold is None, "the pages copied back are still held for b"
    kv.assert_pages_conserved()



def test_a_prompt_rooted_before_its_readiness_check_still_copies_its_prefix_back():
    kv = _manager()
    _on_the_host_only(kv)
    _ingest(kv, "b", A_PROMPT + [7, 8])
    chain = kv._streams["b"]["main"].chain
    chain.rooted, chain.rooted_ends = kv._root(list(chain.keys), chain.spans)

    assert not _ready(kv, "b"), "a prompt rooted before its readiness check skipped its copy from the host"
    kv._cpu_pool.land()
    assert _ready(kv, "b"), "b stayed not ready after its copy landed"
    assert _probe(kv, "b", A_PROMPT + [7, 8]) == 6 * PAGE_SIZE, "b's probe missed the pages copied back"
    kv.assert_pages_conserved()


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



