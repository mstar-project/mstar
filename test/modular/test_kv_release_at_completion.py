"""A request that is done gives its pages back before it is removed.

`remove_request` frees a finished request's pages only once the client has read
its outputs, which can be most of a second after the decode loop stops. The
conductor knows when a request is done and says so (RELEASE_KV), and
`KVManager.release_kv` frees the pages and the reservation then, leaving the
rest of the request's state for the `remove_request` that still follows.

Pages the prefix index co-owns stay indexed. The request is never admitted or
given a page again, and a release and a removal, in either order, free
everything once.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import AdmitRuntimeError, Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
PREFILL = "prefill"
DECODE = "decode"


class _StubTransfer:
    """No engine, no bytes moved; remembers which requests it was told to forget."""

    removed: list[str] = []

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        self.removed.append(request_id)

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    _StubTransfer.removed = []
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    for name in ("MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_STEP_TELEMETRY_DIR"):
        monkeypatch.delenv(name, raising=False)


def _manager(
    max_num_pages: int = 16, prefix_cache: bool = False, rank: int = 0, world_size: int = 1,
    fit: str = "sum", order: str = "fifo",
) -> KVManager:
    group = SimpleNamespace(rank=rank, world_size=world_size) if world_size > 1 else None
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            admission_fit=fit, admission_order=order,
        ),
        name="kv", joint_comm_group=group, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    if prefix_cache:
        kv.enable_prefix_cache(ROOT, {"main": (PREFILL, DECODE)})
    return kv


def _request(tokens: list[int], max_tokens: int) -> KVReqConfig:
    """A keyed text request, stamped as Bagel stamps one."""
    whole = len(tokens) // PAGE_SIZE
    return KVReqConfig(
        max_tokens=max_tokens,
        prefix_keys={"main": chain([
            tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
        ])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        prompt_slots={"main": len(tokens)}, decode_labels=["main"],
    )


def _ingest(kv: KVManager, rid: str, prompt: int, max_tokens: int, base: int = 0) -> None:
    kv.ingest_request(rid, _request(list(range(base, base + prompt)), max_tokens))


def _ready(kv: KVManager, rid: str):
    return kv.admit_retrieve(rid, NODE, PREFILL, None)


def _step(kv: KVManager, rid: str, span: int, walk: str = DECODE):
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    return outcome


def _prefill(kv: KVManager, rid: str, prompt: int):
    matched = kv.resolve_cached_prefix(rid, NODE, PREFILL) or 0
    kv.apply_cached_prefix(rid, NODE, PREFILL, None, matched)
    return _step(kv, rid, prompt - matched, PREFILL)


def _running(kv: KVManager, rid: str, prompt: int, max_tokens: int, decoded: int = 3, base: int = 0):
    """``rid`` admitted, prefilled and ``decoded`` tokens in."""
    _ingest(kv, rid, prompt, max_tokens, base)
    assert _ready(kv, rid).ready
    assert _prefill(kv, rid, prompt).ok
    for _ in range(decoded):
        assert _step(kv, rid, 1).ok


def _pages(kv: KVManager, rid: str) -> list[int]:
    return [p for s in kv._streams.get(rid, {}).values() for p in s.page_indices]


def _all_free(kv: KVManager) -> bool:
    cached = len(kv._index.evictable()) if kv._index is not None else 0
    return kv._arena.num_free + cached == kv.config.max_num_pages - 1


# ── what a release frees ────────────────────────────────────────────────


def test_the_pages_and_the_reservation_are_free_right_after_the_release():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    held = len(_pages(kv, "a"))
    assert held and "a" in kv._reserved
    free = kv._arena.num_free

    kv.release_kv("a")

    assert kv._arena.num_free == free + held
    assert _pages(kv, "a") == [] and kv._streams["a"]["main"].stored_len == 0
    assert "a" not in kv._reserved and kv._outstanding() == 0
    assert _all_free(kv)


def test_a_request_waiting_for_the_room_is_admitted_at_the_release_not_at_the_removal():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=100, max_tokens=60)
    _ingest(kv, "b", prompt=100, max_tokens=60, base=1000)
    assert not _ready(kv, "b").ready

    kv.release_kv("a")

    assert _ready(kv, "b").ready
    assert "a" in kv._streams, "the release took the request's state, which the removal clears"


def test_a_release_with_nothing_held_or_reserved_is_no_error():
    kv = _manager()
    _ingest(kv, "never_ran", prompt=20, max_tokens=5)
    kv.release_kv("never_ran")
    kv.release_kv("unknown")

    assert "unknown" not in kv._released and _all_free(kv)


def test_a_second_release_changes_nothing():
    kv = _manager()
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")
    free, epoch = kv._arena.num_free, kv._reserved_epoch

    kv.release_kv("a")

    assert (kv._arena.num_free, kv._reserved_epoch) == (free, epoch)


def test_the_host_pages_of_the_request_are_given_back_too():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    freed = []
    # the host pool pins its memory, so a CPU run stands one in
    kv._cpu_pool = SimpleNamespace(remove_request=freed.append)

    kv.release_kv("a")

    assert freed == ["a"]


def test_a_lease_that_admit_never_converted_is_given_back_too():
    kv = _manager(max_num_pages=32, prefix_cache=True)
    _running(kv, "a", prompt=3 * PAGE_SIZE, max_tokens=4)
    kv.remove_request("a")
    # same prompt: the index has it, so "b" leases it at admission and has not run
    _ingest(kv, "b", prompt=3 * PAGE_SIZE, max_tokens=4)
    assert _ready(kv, "b").ready
    leased = kv._streams["b"]["main"].lease
    assert leased

    kv.release_kv("b")

    assert kv._streams["b"]["main"].lease is None
    assert all(kv._arena.num_owners[page] == 1 for page in leased), "a lease was left held"
    assert len(kv._index.evictable()) == len(kv._index.pages()) == 3


# ── the removal that follows ────────────────────────────────────────────


def test_the_removal_after_a_release_frees_nothing_again_and_clears_the_rest():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")
    free, owners = kv._arena.num_free, list(kv._arena.num_owners)

    kv.remove_request("a")

    assert kv._arena.num_free == free and kv._arena.num_owners == owners
    assert "a" not in kv._streams and "a" not in kv._overrides
    assert "a" not in kv._released and "a" not in kv._opened
    assert _StubTransfer.removed == ["a"], "the transfer state is the removal's to clear"


def test_the_removal_before_a_release_leaves_the_release_nothing_to_do():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.remove_request("a")
    free = kv._arena.num_free

    kv.release_kv("a")

    assert kv._arena.num_free == free and "a" not in kv._released
    assert _all_free(kv)


def test_a_request_id_that_is_reused_after_the_removal_is_admitted_as_new():
    """Worker handles are reused once their request is removed."""
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")
    kv.remove_request("a")

    _running(kv, "a", prompt=40, max_tokens=20, base=5000)

    assert _pages(kv, "a") and "a" in kv._reserved


# ── the prefix index ────────────────────────────────────────────────────


def test_pages_the_index_holds_stay_indexed_and_evictable():
    kv = _manager(max_num_pages=16, prefix_cache=True)
    prompt = 4 * PAGE_SIZE + 5
    _running(kv, "a", prompt=prompt, max_tokens=20, decoded=2)
    indexed = list(kv._index.pages())
    assert len(indexed) == 4 and set(indexed) <= set(_pages(kv, "a"))
    free = kv._arena.num_free
    held = len(_pages(kv, "a"))

    kv.release_kv("a")

    assert set(kv._index.pages()) == set(indexed), "the release unindexed a page"
    assert all(kv._arena.num_owners[page] == 1 for page in indexed)
    assert sorted(kv._index.evictable()) == sorted(indexed)
    assert kv._arena.num_free == free + held - len(indexed), "only the pages the index does not hold are free"

    # and what the index kept is what the next request with that prompt finds
    _ingest(kv, "b", prompt=prompt, max_tokens=20)
    assert _ready(kv, "b").ready
    assert len(kv._streams["b"]["main"].lease) == 4


def test_the_cached_pages_are_evicted_for_a_request_that_needs_the_room():
    kv = _manager(max_num_pages=12, prefix_cache=True)
    _running(kv, "a", prompt=5 * PAGE_SIZE, max_tokens=4, decoded=1)
    kv.release_kv("a")
    cached = len(kv._index.evictable())
    assert cached and kv._arena.num_free < kv.config.max_num_pages - 1

    _running(kv, "b", prompt=9 * PAGE_SIZE, max_tokens=8, decoded=1, base=7000)

    assert len(_pages(kv, "b")) >= 9
    kv.remove_request("b")
    kv.remove_request("a")
    assert _all_free(kv)


# ── a released request is never run again ───────────────────────────────


def test_a_released_request_is_refused_readiness_rather_than_admitted_again():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")

    outcome = _ready(kv, "a")

    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)
    assert "a" not in kv._reserved and "a" not in kv._waiting


def test_a_released_request_is_not_reserved_for_by_a_step_that_reaches_admit():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")
    free = kv._arena.num_free

    outcome = _step(kv, "a", 1)

    assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)
    assert "a" not in kv._reserved and kv._arena.num_free == free and _pages(kv, "a") == []


def test_a_released_request_is_given_no_page_by_a_grant():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")
    free = kv._arena.num_free

    result = kv._alloc("a", "main", 100)

    assert not result.success and isinstance(result.error, AdmitRuntimeError)
    assert kv._arena.num_free == free and _pages(kv, "a") == []


def test_a_released_request_leases_nothing_from_the_index():
    kv = _manager(max_num_pages=32, prefix_cache=True)
    _running(kv, "a", prompt=3 * PAGE_SIZE, max_tokens=4)
    kv.remove_request("a")
    _ingest(kv, "b", prompt=3 * PAGE_SIZE, max_tokens=4)
    kv.release_kv("b")
    owners = list(kv._arena.num_owners)

    assert not kv.resolve_cached_prefix("b", NODE, PREFILL)
    assert kv._streams["b"]["main"].lease is None
    assert kv._arena.num_owners == owners


def test_a_released_request_is_not_gated_to_reserve_again():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    kv.release_kv("a")

    assert not kv._gated("a", kv._overrides["a"])
    assert kv._reserve_new(StepContext(request_ids=("a",), graph_walk=DECODE, slot=0, capture=False)) is None
    assert "a" not in kv._reserved


def test_a_released_request_publishes_nothing():
    kv = _manager()
    _running(kv, "a", prompt=40, max_tokens=20)
    assert kv.publish("a", NODE, DECODE) is not None
    kv.release_kv("a")

    assert kv.publish("a", NODE, DECODE) is None


def test_a_rank_that_follows_the_deciding_one_gives_back_what_it_reserved():
    kv = _manager(max_num_pages=16, rank=1, world_size=2, fit="peak")
    _ingest(kv, "a", prompt=40, max_tokens=20)
    assert _step(kv, "a", 40, PREFILL).ok
    assert "a" in kv._reserved and "a" in kv._seen

    kv.release_kv("a")

    assert "a" not in kv._reserved and "a" not in kv._seen and _pages(kv, "a") == []
    assert not _step(kv, "a", 1).ok


def test_what_the_pool_kept_about_a_released_request_is_forgotten_with_its_reservation():
    kv = _manager(max_num_pages=16, fit="peak", order="backfill")
    _running(kv, "a", prompt=40, max_tokens=20)
    kv._reserved_at["a"] = 1.0
    epoch = kv._reserved_epoch

    kv.release_kv("a")

    assert kv._reserved_epoch > epoch, "a plan or a refusal kept from before the release would stand"
    assert "a" not in kv._reserved_at and "a" not in kv._shape


# ── a release, an abort and a removal, in any order, free once ──────────


@pytest.mark.parametrize("order", [
    ("release", "remove"), ("remove", "release"), ("release", "release", "remove"),
    ("release", "remove", "release"),
])
def test_every_order_frees_every_page_exactly_once(order):
    kv = _manager(max_num_pages=24, prefix_cache=True)
    _running(kv, "a", prompt=3 * PAGE_SIZE + 4, max_tokens=20, decoded=4)
    _running(kv, "b", prompt=2 * PAGE_SIZE, max_tokens=20, decoded=2, base=9000)

    for op in order:
        getattr(kv, f"{op}_kv" if op == "release" else "remove_request")("a")
        kv.assert_pages_conserved()

    # "b" is untouched by what happened to "a"
    assert _pages(kv, "b") and "b" in kv._reserved
    kv.remove_request("b")
    assert "a" not in kv._streams
    assert _all_free(kv)
    kv.assert_pages_conserved()


def test_a_removal_with_pages_still_held_is_what_an_abort_does_and_still_frees_all():
    """An abort sends no release: the removal alone frees what the request holds."""
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)

    kv.remove_request("a")

    assert _all_free(kv) and "a" not in kv._released


# ── in-flight reads ─────────────────────────────────────────────────────


def test_the_release_waits_for_a_read_into_the_pages_it_frees():
    kv = _manager(max_num_pages=16)
    _running(kv, "a", prompt=40, max_tokens=20)
    read = Future()
    kv._streams["a"]["main"].read_future = read
    kv._streams["a"]["main"].read_pending = True
    done = threading.Event()

    threading.Thread(target=lambda: (kv.release_kv("a"), done.set()), daemon=True).start()

    assert not done.wait(0.2), "the pages were freed under a read still writing them"
    # the request is already refused anything new while the read is waited on
    assert "a" in kv._released and not _ready(kv, "a").ok
    read.set_result(None)
    assert done.wait(5)
    assert _pages(kv, "a") == [] and kv._streams["a"]["main"].read_future is None


# ── the log ─────────────────────────────────────────────────────────────


def _rows(directory) -> list[dict]:
    from mstar.engine.resources.kv import admission_log

    admission_log.flush_all()
    return [
        json.loads(line)
        for path in sorted(directory.glob("admission_kv_pid*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_an_early_release_is_logged_once_and_the_removal_logs_no_second_release(monkeypatch, tmp_path):
    monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
    kv = _manager(max_num_pages=24, prefix_cache=True)
    _running(kv, "a", prompt=2 * PAGE_SIZE + 3, max_tokens=20, decoded=2)
    held, indexed = len(_pages(kv, "a")), len(kv._index.pages())
    _running(kv, "b", prompt=20, max_tokens=5, decoded=1, base=9000)

    kv.release_kv("a")
    kv.release_kv("a")
    kv.remove_request("a")
    kv.remove_request("b")

    rows = _rows(tmp_path)
    early = [r for r in rows if r["event"] == "kv_release"]
    assert [r["rid"] for r in early] == ["a"]
    row = early[0]
    assert (row["held"], row["freed"]) == (held, held - indexed)
    assert row["claim"] > 0 and row["reserved_s"] >= 0
    assert {"ts_wall", "ts_mono", "pool", "fit", "order", "supply", "n_reserved", "n_waiting"} <= row.keys()
    assert [r["rid"] for r in rows if r["event"] == "release"] == ["b"], (
        "the removal of a released request logged its reservation again"
    )


def test_a_request_never_released_early_is_logged_as_before(monkeypatch, tmp_path):
    monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
    kv = _manager(max_num_pages=24)
    _running(kv, "a", prompt=20, max_tokens=5, decoded=1)

    kv.remove_request("a")

    events = [r["event"] for r in _rows(tmp_path)]
    assert events.count("release") == 1 and "kv_release" not in events
