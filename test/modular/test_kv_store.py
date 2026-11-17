"""The KV store under the host pool: pages filed on the host go to the store, and a prompt another process
cached comes back from it before the request runs."""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import wait
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import StepAllowance
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.cpu_page_pool import FiledPage
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.store import KVStoreConfig, MooncakeStoreConfig, StoreIO
from mstar.engine.resources.kv.store import mooncake as mooncake_mod
from mstar.engine.resources.kv.store import tier as tier_mod
from mstar.engine.resources.kv.store.base import GetResult
from mstar.engine.resources.kv.store.mooncake import MooncakeStore
from mstar.engine.resources.kv.store.tier import Phase, StoreTier
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs
from mstar.testing_utils.kv_store import CpuHostPool, MemoryStore

PAGE_SIZE = 4
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"
# five whole pages and a partial one: a lookup matches up to five pages
PROMPT = list(range(5 * PAGE_SIZE + 2))


def _config(**common) -> KVStoreConfig:
    """A store block; the prompts here are a few short pages, so it asks the store for any whole page."""
    block = {"backend": "mooncake", "mooncake": {"master": "unused:1"}, "min_tokens": 0, **common}
    return KVStoreConfig.from_yaml(block)


CONFIG = _config()


class _Clock:
    """The store's clock in tests, moved by hand, so no test depends on how fast a thread runs."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: SimpleNamespace(
        get_kv_transfer_info=lambda **kw: None, owns_transfer_info=lambda *a, **kw: False,
        remove_request=lambda rid: None, start_async_retrieve=lambda **kw: None, cleanup=lambda: None,
    ))


def _manager(store: MemoryStore, config: KVStoreConfig = CONFIG, host_pages: int = 256) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=2, num_kv_heads=1, head_dim=4, max_seq_len=4096,
            max_num_pages=128, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
        host_allowance=StepAllowance(),
    )
    kv._cpu_pool = CpuHostPool(host_pages, tuple(kv.kv_cache.tensor.shape[2:]), kv.kv_cache.tensor.shape[0])
    kv._open_store(store, config)
    # a clock moved by hand, so no test depends on how fast a thread runs
    kv._store.io.clock = _Clock()
    kv.enable_prefix_cache(ROOT)
    return kv


def _ingest(kv: KVManager, rid: str, tokens: list[int]) -> None:
    whole = len(tokens) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain([tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        kv_layout={"main": StreamLayout(spans=(PrefixSpan(len(tokens), len(tokens), WALK),), keyed_spans=1)},
    ))


def _probe(kv: KVManager, rid: str, tokens: list[int]) -> int:
    inputs = ARNodeInputs(input_seq_len=len(tokens))
    prefix = kv.resolve_cached_prefix(rid, NODE, WALK, inputs)
    kv.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
    return prefix.tokens if prefix is not None else 0


def _run(kv: KVManager, rid: str, span: int) -> KVStep:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    return step


def _drain_store(kv: KVManager) -> None:
    """Let the store thread finish everything handed to it so far."""
    store = kv._store
    futures = [read.future for read in store.reads if read.future is not None]
    wait(futures + ([] if store.put is None else [store.put.future]))


def _cache_through_the_store(kv: KVManager, rid: str = "a", prompt: list[int] = PROMPT) -> dict[int, torch.Tensor]:
    """Serve ``rid``'s prompt, end it so its pages go to the host and on to the store, and return what
    each of its pages held on the GPU."""
    _ingest(kv, rid, prompt)
    step = _run(kv, rid, len(prompt))
    pages = list(kv._streams[rid]["main"].page_indices)
    for i, page in enumerate(pages):
        kv.kv_cache.tensor[:, page] = float(i + 1)
    held = {i: kv.kv_cache.tensor[:, page].clone() for i, page in enumerate(pages)}
    kv.remove_request(rid)
    kv.after_output_copy(step)          # copy the released pages to the host
    with kv._lock:
        kv._land_host_writes()          # file them there, which queues them for the store
    kv.after_output_copy(step)          # hand them to the store thread
    _drain_store(kv)
    kv.after_output_copy(step)          # release the host pages the put held
    return held


def _ready(kv: KVManager, rid: str) -> bool:
    out = kv.admit_retrieve(rid, NODE, WALK, None)
    assert out.ok
    return out.ready


def _ready_after(kv: KVManager, rid: str, passes: int = 6) -> int:
    """How many readiness checks ``rid`` waits through, letting the store thread finish between them."""
    for done in range(passes):
        if _ready(kv, rid):
            return done
        _drain_store(kv)
    raise AssertionError(f"{rid} was still not ready after {passes} checks")


def test_the_store_block_takes_known_keys_only():
    config = KVStoreConfig.from_yaml({"backend": "mooncake", "timeout_ms": 50, "mooncake": {"master": "h:1"}})
    assert config.timeout_ms == 50 and config.min_tokens == 256 and config.options.metadata == "P2PHANDSHAKE"
    mooncake = {"master": "h:1"}
    for block, words in (
        ({"backend": "mooncake", "master": "h:1", "mooncake": mooncake}, "unknown keys"),
        ({"backend": "mooncake", "mooncake": {"master": "h:1", "masterr": "h:2"}}, "unknown keys"),
    ):
        with pytest.raises(TypeError, match=words):
            KVStoreConfig.from_yaml(block)
    for block, words in (
        ({"mooncake": mooncake}, "needs a backend"),
        ({"backend": "redis", "mooncake": mooncake}, "unknown backend"),
        ({"backend": "mooncake"}, "needs a mooncake: block"),
        ({"backend": "mooncake", "mooncake": {}}, "needs master"),
        ({"backend": "mooncake", "timeout_ms": "1000", "mooncake": mooncake}, "must be a number"),
        ({"backend": "mooncake", "timeout_ms": 0, "mooncake": mooncake}, "must be positive"),
        ({"backend": "mooncake", "mooncake": {"master": "h:1", "segment_bytes": -1}}, "at least 0"),
        ({"backend": "mooncake", "mooncake": {"master": "h:1", "protocol": "shm"}}, "tcp or rdma"),
        ({"backend": "mooncake", "read_share": 0, "mooncake": mooncake}, "must be positive"),
        ({"backend": "mooncake", "read_share": 0.4, "write_share": 0.2, "mooncake": mooncake}, "at least half"),
    ):
        with pytest.raises(ValueError, match=words):
            KVStoreConfig.from_yaml(block)
    # written as M*'s own transport spells protocols; the backend splits calls only for "tcp"
    upper = KVStoreConfig.from_yaml({"backend": "mooncake", "mooncake": {"master": "h:1", "protocol": "TCP"}})
    assert upper.options.protocol == "tcp"


def test_a_pool_refuses_a_store_it_cant_use():
    def cfg(**kw):
        return PagedKVConfig(num_layers=2, num_kv_heads=1, head_dim=4, max_seq_len=64, page_size=PAGE_SIZE, **kw)

    with pytest.raises(ValueError, match="host pool"):
        KVManager._check_store(cfg(prefix_cache_salt="s"), "kv", None)
    with pytest.raises(ValueError, match="prefix_cache_salt"):
        KVManager._check_store(cfg(cpu_offload_pages=8), "kv", None)
    with pytest.raises(ValueError, match="one rank"):
        KVManager._check_store(cfg(cpu_offload_pages=8, prefix_cache_salt="s"), "kv", SimpleNamespace(world_size=2))
    KVManager._check_store(cfg(cpu_offload_pages=8, prefix_cache_salt="s"), "kv", None)


def test_a_pool_with_recurrent_snapshots_refuses_a_store():
    kv = _manager(MemoryStore())
    with pytest.raises(ValueError, match="snapshots"):
        kv.bind_snapshots(SimpleNamespace())


def test_pages_filed_on_the_host_go_to_the_store():
    store = MemoryStore()
    kv = _manager(store)

    _cache_through_the_store(kv)

    assert len(store.pages) == 5, "the five whole pages the request cached aren't all in the store"
    assert kv._store.put is None, "a finished put still holds its host pages"
    kv._assert_host_pages_conserved()


def test_a_second_process_reads_the_prompt_from_the_store():
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    held = _cache_through_the_store(writer)
    reader = _manager(MemoryStore(shared))
    _ingest(reader, "b", PROMPT)

    waited = _ready_after(reader, "b")

    assert waited >= 2, "b ran before its pages could have come from the store"
    assert _probe(reader, "b", PROMPT) == 5 * PAGE_SIZE, "b's probe missed the pages read from the store"
    lease = reader._streams["b"]["main"].lease
    for i, page in enumerate(lease):
        assert torch.equal(reader.kv_cache.tensor[:, page], held[i]), f"page {i} came back with different bytes"
    reader._assert_host_pages_conserved()
    reader.assert_pages_conserved()


def test_the_store_is_asked_once_per_prompt():
    shared: dict = {}
    _cache_through_the_store(_manager(MemoryStore(shared)))
    reader_store = MemoryStore(shared)
    reader = _manager(reader_store)
    _ingest(reader, "b", PROMPT)

    _ready_after(reader, "b")
    _ready(reader, "b")

    assert reader_store.calls.count("exists") == 1 and reader_store.calls.count("get") == 1


def test_a_short_continuation_doesnt_ask_the_store():
    shared: dict = {}
    _cache_through_the_store(_manager(MemoryStore(shared)))
    store = MemoryStore(shared)
    # five whole pages of 4 tokens past what the GPU and host hold: 20 tokens, under the threshold
    kv = _manager(store, _config(min_tokens=6 * PAGE_SIZE))
    _ingest(kv, "b", PROMPT)

    assert _ready(kv, "b"), "a short continuation waited for the store"
    assert store.calls == [], "a short continuation asked the store"


def test_a_prompt_the_store_lacks_leaves_the_host_alone():
    store = MemoryStore()
    kv = _manager(store)
    _cache_through_the_store(kv, "a")
    kv._index.evict(10)
    host_cached = kv._cpu_pool.index.num_pages()
    _ingest(kv, "b", [9000 + t for t in PROMPT])

    _ready_after(kv, "b")

    assert store.calls.count("get") == 0, "a prompt the store lacks started a get"
    assert kv._cpu_pool.index.num_pages() == host_cached, "a miss evicted the host's cached pages"
    kv._assert_host_pages_conserved()


def _clocked(kv: KVManager) -> _Clock:
    return kv._store.io.clock


class _Gate(MemoryStore):
    """Holds each call named in ``held`` until ``release`` is set, and says when one has started."""

    def __init__(self, pages=None, held=("get",)):
        super().__init__(pages)
        self.held = held
        self.started = threading.Event()
        self.release = threading.Event()
        # every call as it starts; `calls` records one only once it returns
        self.asked: list[str] = []

    def _hold(self, call):
        self.asked.append(call)
        if call in self.held:
            self.started.set()
            assert self.release.wait(10), "the test never released the store"

    def exists(self, keys):
        self._hold("exists")
        return super().exists(keys)

    def get(self, keys, pages):
        self._hold("get")
        return super().get(keys, pages)


def _cached_store() -> dict:
    shared: dict = {}
    _cache_through_the_store(_manager(MemoryStore(shared)))
    return shared


def test_a_read_past_its_deadline_lets_the_request_go():
    gate = _Gate(_cached_store())
    kv = _manager(gate)
    clock = _clocked(kv)
    _ingest(kv, "b", PROMPT)

    assert not _ready(kv, "b")
    wait([read.future for read in kv._store.reads])
    assert not _ready(kv, "b"), "b didn't wait for its get"
    assert gate.started.wait(10)
    clock.now += 3600
    assert _ready(kv, "b"), "b kept waiting on a get past its deadline"
    assert _probe(kv, "b", PROMPT) == 0

    gate.release.set()
    _drain_store(kv)
    with kv._lock:
        kv._store.advance(clock())
    assert kv._store.reads == []
    assert kv._cpu_pool.index.num_pages() == 5, "the late get's pages weren't filed for the next request"
    kv._assert_host_pages_conserved()


def test_a_lookup_nobody_waits_for_any_more_reads_nothing():
    gate = _Gate(_cached_store(), held=("exists",))
    kv = _manager(gate)
    clock = _clocked(kv)
    _ingest(kv, "b", PROMPT)
    assert not _ready(kv, "b")
    assert gate.started.wait(10)
    clock.now += 3600
    assert _ready(kv, "b"), "b kept waiting on a lookup past its deadline"

    gate.release.set()
    _drain_store(kv)
    with kv._lock:
        kv._store.advance(clock())

    assert kv._store.reads == [] and gate.asked.count("get") == 0, "a lookup nobody waits for still read"
    _ingest(kv, "c", PROMPT)
    _ready_after(kv, "c")
    assert gate.asked.count("exists") == 2 and _probe(kv, "c", PROMPT) == 5 * PAGE_SIZE, "c didn't ask again"
    kv._assert_host_pages_conserved()


def test_a_request_removed_mid_read_leaks_nothing():
    gate = _Gate(_cached_store())
    kv = _manager(gate)
    _ingest(kv, "b", PROMPT)
    assert not _ready(kv, "b")
    wait([read.future for read in kv._store.reads])
    assert not _ready(kv, "b")
    assert gate.started.wait(10)

    kv.remove_request("b")
    gate.release.set()
    _drain_store(kv)
    with kv._lock:
        kv._store.advance(kv._store.io.clock())

    assert kv._store.reads == []
    kv._assert_host_pages_conserved()
    kv.assert_pages_conserved()


class _BrokenStore(MemoryStore):
    def exists(self, keys):
        raise ConnectionError("store down")


def test_a_store_error_is_a_miss():
    kv = _manager(_BrokenStore())
    _ingest(kv, "b", PROMPT)

    _ready_after(kv, "b")

    assert _probe(kv, "b", PROMPT) == 0
    assert kv._store.io.errors == 1


def test_a_store_that_keeps_failing_is_skipped():
    io = StoreIO(_BrokenStore(), CONFIG, "layout", page_bytes=1, clock=_Clock())
    for _ in range(5):
        future = io.exists([b"k"])
        wait([future])
        assert future.exception() is not None
    assert not io.available(), "the store was still asked after five failures in a row"
    io.clock.now += 31
    assert io.available()
    future = io.exists([b"k"])
    wait([future])
    assert not io.available(), "a store still failing after its skip was asked four more times"
    io.close()


def test_a_call_waiting_behind_another_isnt_charged_for_the_wait():
    gate = _Gate()
    clock = _Clock()
    io = StoreIO(gate, _config(timeout_ms=1000, floor_gbps=1e-9), "layout", page_bytes=1, clock=clock)
    gate.bind(torch.zeros(1, 8, 1))
    first = io.get([b"a"] * 4, [0, 1, 2, 3])     # 4 bytes at the floor: a 4 s deadline
    assert gate.started.wait(10)
    second = io.get([b"b"], [4])                 # 1 s and 1 byte: queued behind the first
    clock.now += 3                                   # past the second's deadline, not the first's
    gate.release.set()
    wait([first, second])

    assert io.errors == 0 and io.available(), "a call was charged for the time it waited behind another"
    io.close()


def test_a_queued_page_stays_held_until_its_put():
    store = MemoryStore()
    kv = _manager(store)
    _ingest(kv, "a", PROMPT)
    step = _run(kv, "a", len(PROMPT))
    pages = list(kv._streams["a"]["main"].page_indices)
    for i, page in enumerate(pages):
        kv.kv_cache.tensor[:, page] = float(i + 1)
    kv.remove_request("a")
    kv.after_output_copy(step)
    with kv._lock:
        kv._land_host_writes()
    queued = dict(kv._store.queued)
    assert len(queued) == 5

    # the host evicts what it can; a queued page isn't among it, so the put reads the bytes that landed
    kv._index.evict(10)
    kv._cpu_pool.index.evict(100)
    assert all(owners > 0 for owners in kv._cpu_pool.arena.owners_of(list(queued.values())))
    kv._assert_host_pages_conserved()
    kv.after_output_copy(step)
    _drain_store(kv)
    kv.after_output_copy(step)

    stored = sorted(store.pages.values(), key=lambda t: float(t.flatten()[0]))
    assert [float(t.flatten()[0]) for t in stored] == [1.0, 2.0, 3.0, 4.0, 5.0]
    kv._assert_host_pages_conserved()


def test_a_store_whose_lookup_is_overdue_is_not_asked():
    gate = _Gate(held=("exists",))
    kv = _manager(gate)
    clock = _clocked(kv)
    try:
        _ingest(kv, "a", PROMPT)
        assert not _ready(kv, "a"), "a didn't ask the store"
        assert gate.started.wait(10)
        clock.now += 3600
        assert _ready(kv, "a"), "a kept waiting on a store that doesn't answer"
        _ingest(kv, "b", [9000 + t for t in PROMPT])
        assert _ready(kv, "b"), "b waited on a store whose lookup is overdue"
        assert gate.asked.count("exists") == 1, "the store was asked again while its lookup was overdue"
    finally:
        gate.release.set()


def test_a_lookup_doesnt_wait_behind_another_requests_read():
    gate = _Gate(_cached_store())
    kv = _manager(gate)
    try:
        _ingest(kv, "a", PROMPT)
        assert not _ready(kv, "a")
        wait([read.future for read in kv._store.reads])
        assert not _ready(kv, "a")
        assert gate.started.wait(10), "a's get didn't start"
        _ingest(kv, "cold", [9000 + t for t in PROMPT])
        assert not _ready(kv, "cold"), "cold didn't ask the store"
        wait([read.future for read in kv._store.reads if read.phase is Phase.LOOKUP])
        assert _ready(kv, "cold"), "cold's lookup waited behind a's read"
        assert not _ready(kv, "a"), "a ran before its get finished"
    finally:
        gate.release.set()
    assert kv._store.io.errors == 0


def test_a_short_hit_isnt_read():
    shared = _cached_store()
    store = MemoryStore(shared)
    kv = _manager(store, _config(min_tokens=2 * PAGE_SIZE))
    # only the first page is in the store: the hit is one page, under the two-page threshold
    _ingest(kv, "b", PROMPT[:PAGE_SIZE] + [7000 + t for t in PROMPT[PAGE_SIZE:]])

    _ready_after(kv, "b")

    assert store.calls == ["exists"], "a hit under min_tokens was read"
    kv._assert_host_pages_conserved()


def test_a_full_queue_keeps_each_chain_from_its_start():
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    writer._store.max_queued = 3
    _cache_through_the_store(writer)
    for _ in range(3):
        writer.after_output_copy(KVStep(segments=()))
        _drain_store(writer)
    reader_store = MemoryStore(shared)
    reader = _manager(reader_store)

    assert len(shared) == 3, "the store holds pages past the queue's bound"
    _ingest(reader, "b", PROMPT)
    _ready_after(reader, "b")
    assert _probe(reader, "b", PROMPT) == 3 * PAGE_SIZE, "the pages kept weren't the chain's first"
    writer._assert_host_pages_conserved()


def test_a_dropped_page_keeps_the_rest_of_its_chain_out_across_writes():
    """The host files a page whose parent only it holds as a root, so the chain's own parent key decides."""
    kv = _manager(MemoryStore())
    tier, pool = kv._store, kv._cpu_pool
    tier.max_queued = 1
    pages = pool.arena.acquire(4)
    for key, page in zip((b"other", b"k1", b"k2", b"x1"), pages, strict=True):
        pool.index.insert(key, page, None)
    pool.arena.release(pages)

    with kv._lock:
        tier.filed([(b"other", pages[0], None)])
        tier.filed([(b"k1", pages[1], None)])          # the queue is full: k1 is dropped
        pool.arena.release([tier.queued.pop(b"other")])  # a put took it, and the queue has room again
        tier.filed([(b"k2", pages[2], b"k1")])          # a later write lands k2, filed as a root on the host
        tier.filed([(b"x1", pages[3], None)])

    assert list(tier.queued) == [b"x1"], "a page under a dropped one was queued for the store"
    kv._assert_host_pages_conserved()


def test_requests_asking_for_the_same_pages_share_one_read():
    shared: dict = {}
    held = _cache_through_the_store(_manager(MemoryStore(shared)))
    store = MemoryStore(shared)
    kv = _manager(store)
    for rid in ("b", "c"):
        _ingest(kv, rid, PROMPT)
        assert not _ready(kv, rid)

    for rid in ("b", "c"):
        _ready_after(kv, rid)

    assert store.calls.count("exists") == 1 and store.calls.count("get") == 1, "the same pages were read twice"
    for rid in ("b", "c"):
        assert _probe(kv, rid, PROMPT) == 5 * PAGE_SIZE
        lease = kv._streams[rid]["main"].lease
        assert all(torch.equal(kv.kv_cache.tensor[:, page], held[i]) for i, page in enumerate(lease))
    kv._assert_host_pages_conserved()


class _SlowPut(MemoryStore):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def put(self, keys, pages):
        self.release.wait(10)
        return super().put(keys, pages)


def test_a_slow_store_holds_one_put_and_a_bounded_queue():
    slow = _SlowPut()
    kv = _manager(slow, host_pages=256)
    kv._store.max_queued = 4
    try:
        for i in range(30):
            rid = f"r{i}"
            _ingest(kv, rid, [1000 * i + t for t in PROMPT])
            step = _run(kv, rid, len(PROMPT))
            kv.remove_request(rid)
            kv.after_output_copy(step)
            with kv._lock:
                kv._land_host_writes()
            kv.after_output_copy(step)
        held = kv._store.host_pages_held()
        bound = kv._store.max_queued + kv._store.put_pages
        assert len(held) <= bound, f"a store slower than the host writes held {len(held)} host pages"
        assert kv._store.pages_dropped > 0, "the queue never filled, so the test didn't check its bound"
        kv._assert_host_pages_conserved()
    finally:
        slow.release.set()
    _drain_store(kv)
    kv.after_output_copy(step)
    kv._assert_host_pages_conserved()


class _NoRoom(MemoryStore):
    """Answers, but stores nothing: a store with no memory to give."""

    def put(self, keys, pages):
        self.calls.append("put")
        return [False] * len(keys)


def test_a_store_that_refuses_every_put_counts_as_failing():
    kv = _manager(_NoRoom())
    _cache_through_the_store(kv)
    assert kv._store.io.errors == 1 and kv._store.io.pages_refused == 5
    kv._assert_host_pages_conserved()


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the real host pool pins memory and copies on CUDA streams"
)


def _booted(monkeypatch, shared: dict) -> KVManager:
    """A KV cache on the GPU with a real host pool, given its store by the serving config."""
    monkeypatch.setattr(manager_mod, "make_store_backend", lambda config: MemoryStore(shared))
    cfg = PagedKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=4096, max_num_pages=128, page_size=PAGE_SIZE,
    )
    cfg.apply_yaml_overrides(
        cpu_offload_pages=256, prefix_cache_salt="test",
        store={"backend": "mooncake", "min_tokens": 0, "mooncake": {"master": "unused:1"}},
    )
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda"), dtype=torch.float16, host_allowance=StepAllowance(),
    )
    kv._store.io.clock = _Clock()
    kv.enable_prefix_cache(ROOT)
    return kv


@requires_cuda
def test_a_prompt_round_trips_through_a_real_host_pool(monkeypatch):
    shared: dict = {}
    writer = _booted(monkeypatch, shared)
    _ingest(writer, "a", PROMPT)
    step = _run(writer, "a", len(PROMPT))
    pages = list(writer._streams["a"]["main"].page_indices)
    for i, page in enumerate(pages):
        writer.kv_cache.tensor[:, page] = float(i + 1)
    held = {i: writer.kv_cache.tensor[:, page].clone() for i, page in enumerate(pages)}
    writer.remove_request("a")
    writer.after_output_copy(step)
    torch.cuda.synchronize()
    with writer._lock:
        writer._land_host_writes()
    writer.after_output_copy(step)
    _drain_store(writer)
    writer.after_output_copy(step)
    assert len(shared) == 5, "the pages copied to the real host pool didn't reach the store"

    reader = _booted(monkeypatch, shared)
    _ingest(reader, "b", PROMPT)
    for _ in range(8):
        if _ready(reader, "b"):
            break
        _drain_store(reader)
        torch.cuda.synchronize()
    else:
        raise AssertionError("b never became ready")
    assert _probe(reader, "b", PROMPT) == 5 * PAGE_SIZE
    for i, page in enumerate(reader._streams["b"]["main"].lease):
        assert torch.equal(reader.kv_cache.tensor[:, page], held[i]), f"page {i} came back with different bytes"
    reader._assert_host_pages_conserved()
    for kv in (writer, reader):
        kv.cleanup()


def test_a_read_the_page_budget_cuts_under_min_tokens_doesnt_ask(monkeypatch):
    shared = _cached_store()
    store = MemoryStore(shared)
    kv = _manager(store, _config(min_tokens=2 * PAGE_SIZE))
    # the device can take one page for this request: a read of one page is under the two-page threshold
    monkeypatch.setattr(kv, "_allowance", lambda rid: 1)
    monkeypatch.setattr(kv, "_hit_allowance", lambda rid: 10)
    _ingest(kv, "b", PROMPT)

    assert _ready(kv, "b"), "a request waited for a read its page budget cut too short"
    assert store.calls == [], "the store was asked for a read the budget cut under min_tokens"


def test_a_read_the_host_pool_cuts_under_min_tokens_isnt_made():
    store = MemoryStore(_cached_store())
    kv = _manager(store, _config(min_tokens=3 * PAGE_SIZE))
    occupied = kv._cpu_pool.arena.acquire(kv._cpu_pool.arena.num_free - 2)  # two pages left for the read
    _ingest(kv, "b", PROMPT)

    _ready_after(kv, "b")
    assert "get" not in store.calls, "a read of two host pages, under the three-page threshold, was made"
    kv._cpu_pool.arena.release(occupied)
    kv._assert_host_pages_conserved()


def test_a_hit_past_the_reads_share_reads_what_fits_and_says_so_once(caplog):
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    _cache_through_the_store(writer, "a", PROMPT)
    _cache_through_the_store(writer, "z", PROMPT_2)
    store = MemoryStore(shared)
    kv = _manager(store, host_pages=16)   # reads hold at most 2 host pages; each prompt has 5 in the store
    with caplog.at_level(logging.WARNING, logger=tier_mod.__name__):
        for rid, prompt in (("b", PROMPT), ("c", PROMPT_2)):
            _ingest(kv, rid, prompt)
            _ready_after(kv, rid)

    assert store.got == [2, 2], f"reads of {store.got} pages, past the reads' share of 2"
    assert sum("was cut to 2" in record.getMessage() for record in caplog.records) == 1
    kv._assert_host_pages_conserved()


PROMPT_2 = [5000 + t for t in PROMPT]


def test_a_hung_read_holds_only_its_own_host_pages():
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    _cache_through_the_store(writer, "a", PROMPT)
    _cache_through_the_store(writer, "z", PROMPT_2)
    gate = _Gate(shared)
    kv = _manager(gate)
    clock = _clocked(kv)
    try:
        _ingest(kv, "a", PROMPT)
        assert not _ready(kv, "a")
        wait([read.future for read in kv._store.reads])
        assert not _ready(kv, "a")
        assert gate.started.wait(10), "a's get didn't start"
        _ingest(kv, "b", PROMPT_2)
        assert not _ready(kv, "b")
        wait([read.future for read in kv._store.reads if read.future is not None and not read.pages])
        assert not _ready(kv, "b"), "b ran before its turn to read"
        clock.now += 3600
        assert _ready(kv, "a") and _ready(kv, "b"), "a request kept waiting on a hung store"

        held = kv._store.host_pages_held()
        assert len(held) == 5, f"reads behind a hung get hold {len(held) - 5} more host pages"
    finally:
        gate.release.set()
    _drain_store(kv)
    with kv._lock:
        kv._store.advance(clock())
    kv._assert_host_pages_conserved()


def test_refused_puts_stop_writes_not_reads():
    io = StoreIO(_NoRoom(), CONFIG, "layout", page_bytes=1, clock=_Clock())
    io.backend.bind(torch.zeros(1, 8, 1))
    for i in range(5):
        future = io.put([bytes([i])], [i])
        wait([future])
        assert future.exception() is not None
    assert not io.writable(), "five refused puts in a row didn't stop writes"
    assert io.available(), "puts the store refused stopped reads from it too"
    io.close()


def test_a_queued_read_nobody_waits_on_never_runs():
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    for rid, prompt in (("a", PROMPT), ("y", PROMPT_2)):
        _cache_through_the_store(writer, rid, prompt)
    gate = _Gate(shared)
    kv = _manager(gate)
    try:
        _ingest(kv, "a", PROMPT)
        assert not _ready(kv, "a")
        wait([read.future for read in kv._store.reads])
        assert not _ready(kv, "a") and gate.started.wait(10), "a's get didn't start"
        _ingest(kv, "stale", PROMPT_2)
        assert not _ready(kv, "stale")
        wait([read.future for read in kv._store.reads if read.phase is Phase.LOOKUP])
        assert not _ready(kv, "stale"), "stale's read didn't queue behind a's"
        kv.remove_request("stale")          # its read now waits its turn for nobody
    finally:
        gate.release.set()
    _ready_after(kv, "a")

    assert gate.asked.count("get") == 1, "a queued read nobody waits on still ran"
    assert kv._store.reads == []
    kv._assert_host_pages_conserved()


def test_a_request_whose_prompt_runs_past_a_read_gets_its_own():
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    _cache_through_the_store(writer, "a", PROMPT)
    longer = PROMPT[:2 * PAGE_SIZE] + [6000 + t for t in PROMPT[2 * PAGE_SIZE:]]
    _cache_through_the_store(writer, "z", longer)
    store = MemoryStore(shared)
    kv = _manager(store)
    _ingest(kv, "a", PROMPT)
    _ingest(kv, "b", longer)
    assert not _ready(kv, "a") and not _ready(kv, "b")

    _ready_after(kv, "a")
    _ready_after(kv, "b")

    assert _probe(kv, "b", longer) == 5 * PAGE_SIZE, "b got only the pages it shared with a's read"
    kv._assert_host_pages_conserved()


class _FailingGet(MemoryStore):
    """Its transfers fail, reported per key, as Mooncake reports them: no exception."""

    def get(self, keys, pages):
        self.calls.append("get")
        return [GetResult.FAILED] * len(keys)


class _MissingGet(MemoryStore):
    """The pages went between the lookup and the get: nothing was transferred."""

    def get(self, keys, pages):
        self.calls.append("get")
        return [GetResult.MISSING] * len(keys)


class _RaisingGet(MemoryStore):
    """Its gets raise, as a backend does when it couldn't stop a failed read's transfers."""

    def get(self, keys, pages):
        self.calls.append("get")
        raise RuntimeError("the client couldn't be closed")


def test_a_failed_reads_pages_come_back_once_the_get_returns():
    """A get that returns has stopped its transfers (the Mooncake backend restarts its client to make sure)."""
    kv = _manager(_FailingGet(_cached_store()))
    free = kv._cpu_pool.arena.num_free
    _ingest(kv, "b", PROMPT)
    _ready_after(kv, "b")

    assert kv._store.quarantine == [] and kv._cpu_pool.arena.num_free == free, "a failed read's pages stayed held"
    assert kv._store.io.errors == 1, "a get whose transfers failed didn't count as a failure"
    kv._assert_host_pages_conserved()


def test_a_raised_reads_pages_stay_held_until_the_store_closes():
    """A get that raised may still be writing its pages, so they're never reused."""
    kv = _manager(_RaisingGet(_cached_store()))
    clock = _clocked(kv)
    _ingest(kv, "b", PROMPT)
    _ready_after(kv, "b")

    assert len(kv._store.quarantine) == 5
    clock.now += 24 * 3600
    kv.after_output_copy(KVStep(segments=()))
    assert len(kv._store.quarantine) == 5, "a raised read's pages came back while they could still be written"
    kv._assert_host_pages_conserved()


def test_a_store_whose_reads_keep_raising_stops_being_read(caplog):
    # reads hold at most 6 host pages and need 2: after a raised read of 5, a read has 1
    kv = _manager(_RaisingGet(_cached_store()), _config(min_tokens=2 * PAGE_SIZE), host_pages=48)
    _clocked(kv)
    with caplog.at_level(logging.WARNING, logger=tier_mod.__name__):
        _ingest(kv, "b", PROMPT)
        _ready_after(kv, "b")
    assert len(kv._store.quarantine) == 5 and "no longer read" in caplog.text
    store_calls = list(kv._store.io.backend.calls)
    _ingest(kv, "c", PROMPT)
    _ready_after(kv, "c")
    assert "get" not in kv._store.io.backend.calls[len(store_calls):], "a store with no read pages left was read"
    kv._assert_host_pages_conserved()


def test_a_dropped_key_queued_later_lets_its_chain_through():
    kv = _manager(MemoryStore())
    tier, pool = kv._store, kv._cpu_pool
    tier.max_queued = 1
    pages = pool.arena.acquire(3)
    for key, page in zip((b"other", b"k1", b"k2"), pages, strict=True):
        pool.index.insert(key, page, None)
    pool.arena.release(pages)
    with kv._lock:
        tier.filed([(b"other", pages[0], None)])
        tier.filed([(b"k1", pages[1], None)])            # full: dropped
        pool.arena.release([tier.queued.pop(b"other")])
        tier.filed([(b"k1", pages[1], None)])            # filed again later, with room
        pool.arena.release([tier.queued.pop(b"k1")])
        tier.filed([(b"k2", pages[2], b"k1")])
    assert list(tier.queued) == [b"k2"], "a page under a key that was queued later stayed dropped"
    pool.arena.release(list(tier.queued.values()))
    tier.queued.clear()
    kv._assert_host_pages_conserved()


def test_closing_logs_what_the_store_did(caplog):
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    _cache_through_the_store(writer)
    reader = _manager(MemoryStore(shared))
    _ingest(reader, "b", PROMPT)
    _ready_after(reader, "b")

    with caplog.at_level(logging.INFO, logger=tier_mod.__name__):
        writer._store.close()
        reader._store.close()

    lines = [record.getMessage() for record in caplog.records if "pages put" in record.getMessage()]
    assert len(lines) == 2, lines
    assert lines[0].startswith("KV store: 5 pages put (or already held), 0 dropped"), lines[0]
    assert "0 pages put" in lines[1] and ", 5 read, 0 missing" in lines[1], lines[1]


def test_large_pages_queue_a_long_prompt_up_to_the_host_share():
    """Pages of 16 MiB, as Phi-4-mini's are, make a put two pages; a prompt filed at once is still queued whole,
    up to the writes' share of the host pool, not dropped past a few puts' worth."""
    kv = _manager(MemoryStore(), host_pages=512)
    pool = kv._cpu_pool
    io = StoreIO(MemoryStore(), CONFIG, "layout", page_bytes=16 << 20, clock=_Clock())
    tier = StoreTier(io, pool, PAGE_SIZE)
    try:
        assert tier.put_pages == 2
        keys = [b"k%d" % i for i in range(40)]
        parents = [None, *keys[:-1]]
        pages = pool.arena.acquire(len(keys))
        pool.index.insert_run(keys, pages, None)
        pool.arena.release(pages)

        tier.filed([FiledPage(*filed) for filed in zip(keys, pages, parents, strict=True)])

        assert tier.pages_dropped == 0 and list(tier.queued) == keys, (
            f"queued {len(tier.queued)} of a {len(keys)}-page prompt, dropped {tier.pages_dropped}"
        )
        assert tier.max_queued + tier.put_pages <= 512 // 8
    finally:
        pool.arena.release(list(tier.queued.values()))
        io.close()


class _FakeMooncake:
    """The Mooncake client calls `MooncakeStore` makes, each call recorded; a get fails the keys in ``fail``."""

    def __init__(self, page_bytes: int, fail: frozenset[str] = frozenset()):
        self.page_bytes, self.fail, self.calls = page_bytes, fail, []
        self.registered = self.closed = False

    def setup(self, *args) -> int:
        return 0

    def register_buffer(self, base: int, size: int) -> int:
        self.registered = True
        return 0

    def unregister_buffer(self, base: int) -> int:
        self.registered = False
        return 0

    def close(self) -> None:
        self.calls.append(("close", 0))
        self.closed = True

    def batch_is_exist(self, keys) -> list[int]:
        self.calls.append(("exists", len(keys)))
        return [1] * len(keys)

    def batch_put_from_multi_buffers(self, keys, blocks, sizes) -> list[int]:
        self.calls.append(("put", len(keys)))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, blocks, sizes) -> list[int]:
        self.calls.append(("get", len(keys)))
        return [-800 if key in self.fail else self.page_bytes for key in keys]


def _mooncake_over_tcp(fail: frozenset[str] = frozenset(), client=_FakeMooncake) -> tuple[MooncakeStore, list]:
    """A Mooncake backend over TCP on a host pool of 32 layers, so a call takes 2048 // 4 // 32 = 16 pages, with
    the clients it opens, first to last; only the first fails ``fail``."""
    host = torch.zeros((32, 64, 16))
    page_bytes = host[:, 0].numel() * host.element_size()
    clients: list = []

    def open_client():
        clients.append(client(page_bytes, fail if not clients else frozenset()))
        return clients[-1]

    store = MooncakeStore.__new__(MooncakeStore)
    store._client_type, store._lock = open_client, mooncake_mod._SharedLock()
    store._config = MooncakeStoreConfig(master="unused:1", local_hostname="127.0.0.1:1")
    store._store = store._connect()
    store._tcp, store._base, store._registered = True, 0, 0
    store.bind(host)
    return store, clients


def test_mooncake_calls_over_tcp_stay_under_its_queue(monkeypatch):
    """A page is a transfer per layer, and a call past Mooncake's TCP queue (2048 transfers a peer) fails and
    leaves the client failing later transfers."""
    monkeypatch.delenv("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", raising=False)
    store, (fake,) = _mooncake_over_tcp()
    keys, pages = [f"k{i}" for i in range(40)], list(range(40))

    assert store.put(keys, pages) == [True] * 40
    assert store.get(keys, pages) == [GetResult.ARRIVED] * 40
    assert fake.calls == [("put", 16), ("put", 16), ("put", 8), ("get", 16), ("get", 16), ("get", 8)]

    monkeypatch.setenv("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", "8192")
    store, (fake,) = _mooncake_over_tcp()
    store.get(keys, pages)
    assert fake.calls == [("get", 40)], "a deployment's larger queue wasn't used"


def test_a_failed_mooncake_get_ends_the_read_and_restarts_the_client(monkeypatch):
    """Pages after a failed one aren't read, since nothing after it is filed. Mooncake keeps writing a failed
    transfer's bytes after it reports the failure, and closing the client is what stops it, so a get that
    returns has closed the client and opened another."""
    monkeypatch.delenv("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", raising=False)
    store, clients = _mooncake_over_tcp(fail=frozenset({"k20"}))
    keys, pages = [f"k{i}" for i in range(40)], list(range(40))
    results = store.get(keys, pages)

    arrived, failed, missing = GetResult.ARRIVED, GetResult.FAILED, GetResult.MISSING
    assert results == [arrived] * 20 + [failed] + [arrived] * 11 + [missing] * 8
    first, second = clients
    assert first.calls == [("get", 16), ("get", 16), ("close", 0)] and not first.registered
    assert second.registered, "the new client didn't register the host pool"
    assert store.get(keys[:4], pages[:4]) == [arrived] * 4 and second.calls == [("get", 4)]


def test_a_mooncake_restart_waits_for_calls_on_other_threads(monkeypatch):
    """Closing the client under a call another store thread is making would pull it from under that call."""
    monkeypatch.delenv("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", raising=False)
    entered, release = threading.Event(), threading.Event()

    class _SlowLookup(_FakeMooncake):
        def batch_is_exist(self, keys):
            entered.set()
            release.wait(10)
            return super().batch_is_exist(keys)

    store, clients = _mooncake_over_tcp(fail=frozenset({"k0"}), client=_SlowLookup)
    lookup = threading.Thread(target=store.exists, args=(["k0"],))
    lookup.start()
    assert entered.wait(10)
    read = threading.Thread(target=store.get, args=(["k0"], [0]))
    read.start()
    read.join(0.3)
    assert read.is_alive() and not clients[0].closed, "the client was closed under a lookup still running"
    release.set()
    lookup.join(10)
    read.join(10)
    assert clients[0].closed and len(clients) == 2


def test_the_store_block_sets_its_shares_of_the_host_pool():
    """Per-deployment sizing: a deployment that keeps much of its host pool for the store can raise them."""
    kv = _manager(MemoryStore(), _config(read_share=0.25, write_share=0.25), host_pages=64)
    assert kv._store.max_read_pages == 16 and kv._store.max_queued + kv._store.put_pages <= 16


def test_writing_holds_at_most_an_eighth_of_a_small_host_pool():
    kv = _manager(MemoryStore(), host_pages=64)
    assert kv._store.max_queued + kv._store.put_pages <= 64 // 8


def test_pages_missing_at_the_get_come_back_at_once():
    kv = _manager(_MissingGet(_cached_store()))
    _ingest(kv, "b", PROMPT)
    free = kv._cpu_pool.arena.num_free
    _ready_after(kv, "b")

    assert kv._store.quarantine == [] and kv._cpu_pool.arena.num_free == free, "pages nothing was sent to were held"
    assert kv._store.io.errors == 0, "pages the store didn't have counted as a failing store"
    kv._assert_host_pages_conserved()


def test_a_store_whose_gets_keep_failing_is_skipped():
    shared = _cached_store()
    kv = _manager(_FailingGet(shared))
    for i in range(5):
        rid = f"r{i}"
        _ingest(kv, rid, PROMPT)
        _ready_after(kv, rid)
        kv.remove_request(rid)
        with kv._lock:
            kv._index.evict(100)
    assert not kv._store.io.available(), "a store whose every get failed was still asked"
    kv._assert_host_pages_conserved()


def test_a_queued_read_skips_pages_another_read_put_on_the_host():
    shared: dict = {}
    writer = _manager(MemoryStore(shared))
    _cache_through_the_store(writer, "a", PROMPT)
    longer = PROMPT[:2 * PAGE_SIZE] + [6000 + t for t in PROMPT[2 * PAGE_SIZE:]]
    _cache_through_the_store(writer, "z", longer)
    store = MemoryStore(shared)
    kv = _manager(store)
    _ingest(kv, "a", PROMPT)
    _ingest(kv, "b", longer)
    assert not _ready(kv, "a") and not _ready(kv, "b")
    _ready_after(kv, "a")
    _ready_after(kv, "b")

    assert store.got == [5, 3], f"b's read asked again for the pages a's read put on the host: {store.got}"


def test_a_finished_store_call_is_seen_by_the_next_readiness_check():
    store = MemoryStore(_cached_store())
    kv = _manager(store)
    mark = kv._store.io.on_finish

    def slow_mark():
        time.sleep(0.02)  # the mark lands before the call's future completes, however late it runs
        mark()

    kv._store.io.on_finish = slow_mark
    _ingest(kv, "b", PROMPT)
    assert not _ready(kv, "b")
    wait([read.future for read in kv._store.reads])
    assert not _ready(kv, "b")
    assert kv._store.reads[0].phase is Phase.GETTING, "the readiness check after the lookup didn't start the get"


class _TwoGates(MemoryStore):
    """Holds lookups until ``lookups`` is set and gets until ``gets`` is set, recording each call as it starts."""

    def __init__(self, pages=None):
        super().__init__(pages)
        self.lookups, self.gets = threading.Event(), threading.Event()
        self.asked: list[str] = []

    def exists(self, keys):
        self.asked.append("exists")
        assert self.lookups.wait(10)
        return super().exists(keys)

    def get(self, keys, pages):
        self.asked.append("get")
        assert self.gets.wait(10)
        return super().get(keys, pages)


def _started(gates: _TwoGates, call: str) -> bool:
    """Wait until the store thread has started a ``call``, and say whether it did within 10 s."""
    deadline = time.monotonic() + 10
    while call not in gates.asked:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def test_a_lookup_nobody_waits_on_is_cancelled_before_it_runs():
    gates = _TwoGates()
    kv = _manager(gates)
    try:
        for i, rid in enumerate(("a", "b", "c")):
            _ingest(kv, rid, [1000 * i + t for t in PROMPT])
            assert not _ready(kv, rid)
            if i == 0:
                assert _started(gates, "exists"), "a's lookup never started"
        kv.remove_request("b")
        kv.remove_request("c")
    finally:
        gates.lookups.set()
        gates.gets.set()
    _ready_after(kv, "a")
    assert gates.asked.count("exists") == 1, "lookups nobody waited on still ran"


def test_a_slow_store_isnt_asked_while_lookups_pile_up():
    gates = _TwoGates()
    kv = _manager(gates)
    try:
        for i in range(5):  # one running, four waiting to start
            rid = f"r{i}"
            _ingest(kv, rid, [1000 * i + t for t in PROMPT])
            assert not _ready(kv, rid), f"{rid} didn't ask the store"
            if i == 0:
                assert _started(gates, "exists"), "the first lookup never started"
        _ingest(kv, "late", [9000 + t for t in PROMPT])
        assert _ready(kv, "late"), "a new prompt queued a lookup behind four waiting ones"
    finally:
        gates.lookups.set()
        gates.gets.set()


def test_a_requests_wait_counts_from_when_it_asked():
    gates = _TwoGates(_cached_store())
    kv = _manager(gates, _config(timeout_ms=1000, floor_gbps=1e3))
    clock = _clocked(kv)
    try:
        _ingest(kv, "b", PROMPT)
        assert not _ready(kv, "b")                 # asked at t
        clock.now += 0.9                           # the lookup answers late
        gates.lookups.set()
        wait([read.future for read in kv._store.reads])
        assert not _ready(kv, "b")                 # its get starts, and is held
        clock.now += 0.6                           # t + 1.5 s: past t + timeout_ms + the bytes' time
        assert _ready(kv, "b"), "the lookup's answer gave the request a fresh timeout_ms"
    finally:
        gates.gets.set()


def test_reads_hold_at_most_their_share_of_the_host_pool():
    store = MemoryStore(_cached_store())
    kv = _manager(store)
    kv._store.max_read_pages = 3
    _ingest(kv, "b", PROMPT)
    _ready_after(kv, "b")
    assert store.got == [3] and _probe(kv, "b", PROMPT) == 3 * PAGE_SIZE
    kv._assert_host_pages_conserved()


def test_a_request_doesnt_join_a_read_past_its_deadline():
    gates = _TwoGates(_cached_store())
    kv = _manager(gates, _config(timeout_ms=1000, floor_gbps=1e3))
    clock = _clocked(kv)
    try:
        _ingest(kv, "a", PROMPT)
        assert not _ready(kv, "a")                 # asks at t
        clock.now += 0.9                           # the lookup answers late, so the get starts late
        gates.lookups.set()
        wait([read.future for read in kv._store.reads])
        assert not _ready(kv, "a")                 # the get is held; its own limit runs to t + 1.9 s
        clock.now += 0.6                           # t + 1.5 s: past a's deadline, the get not overdue
        assert _ready(kv, "a"), "a waited past its deadline"
        _ingest(kv, "b", PROMPT)
        assert not _ready(kv, "b"), "b didn't ask the store"
        b_read = kv._streams["b"]["main"].store_read
        assert b_read.asked == clock(), "b joined a read whose deadline had passed"
        clock.now += 3600
        assert _ready(kv, "b"), "b waited past its own deadline"
    finally:
        gates.gets.set()


def test_no_request_waits_more_than_30_s():
    from mstar.engine.resources.kv.store.config import MAX_WAIT_S

    config = _config(timeout_ms=1000, floor_gbps=0.001)
    assert config.get_timeout(10**12) == MAX_WAIT_S == 30.0
    assert config.get_timeout(0) == 1.0


def test_removing_one_of_two_requests_on_a_shared_read_leaves_the_other_waiting():
    """Two requests with one prompt, not yet admitted, compare equal; removing one must not unhook the other."""
    shared = _cached_store()
    store = MemoryStore(shared)
    kv = _manager(store)
    for rid in ("b", "c"):
        _ingest(kv, rid, PROMPT)
        assert not _ready(kv, rid)
    assert kv._streams["b"]["main"] == kv._streams["c"]["main"], "the two streams no longer compare equal"

    kv.remove_request("c")

    _ready_after(kv, "b")
    assert _probe(kv, "b", PROMPT) == 5 * PAGE_SIZE
    kv._assert_host_pages_conserved()
