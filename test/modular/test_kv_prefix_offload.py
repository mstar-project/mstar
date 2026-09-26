"""Moving a request's pages once some of them are shared with the index.

A read-in pulls a prefill rank's pages across the wire, and the ones it is
pulling are often pages this cache wrote itself for an earlier request. Probing
the index first leaves the transfer with whatever is genuinely new, which for a
second turn of the same conversation can be nothing at all.

An offload meets the same sharing from the other side. A page the suspended
request shares with the index can be evicted while it is away, so the pages it
left are not there to take back: its host copy is the authoritative one, and
reload allocates fresh pages and copies into them.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import (
    KVManager,
    KVSequenceInfo,
    PublishedKVInfo,
)
from mstar.engine.resources.step import Segment, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the host pool pins its memory"
)

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"
PEER = "peer-ipc-handle"


class _StubTransfer:
    """Records retrieves instead of moving bytes."""

    started: list[dict] = []

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs
        return "own-ipc-handle"

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        type(self).started.append(kwargs)

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    _StubTransfer.started = []
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 64, cpu_offload_pages: int = 0) -> KVManager:
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda" if cpu_offload_pages else "cpu"),
        dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _keys(tokens: list[int]) -> list[bytes]:
    return chain([
        tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
    ])


def _ingest(kv: KVManager, rid: str, tokens: list[int]) -> None:
    whole = len(tokens) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": _keys(tokens)},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
    ))


def _run(kv: KVManager, rid: str, tokens: list[int]) -> None:
    step = KVStep(segments=(Segment(rid, "main", len(tokens)),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _published(kv: KVManager, rid: str, seq_len: int, pages: list[int]):
    """What a prefill rank would publish for ``rid``."""
    del kv, rid
    return PublishedKVInfo.build_for_rank(0, 1, {"main": KVSequenceInfo(
        seq_len=seq_len, latest_kv_transfer_info=PEER, page_indices=pages,
    )})


# ── read-in ─────────────────────────────────────────────────────────────


def test_a_read_in_only_moves_what_this_cache_does_not_already_hold():
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", tokens)
    kv.remove_request("a")

    _ingest(kv, "b", tokens + list(range(1000, 1032)))
    out = kv.admit_retrieve("b", NODE, WALK, _published(kv, "b", 96, list(range(6))))

    assert out.ok, "the read-in was refused"
    assert len(_StubTransfer.started) == 1, (
        "the read this request needed was never started"
    )
    assert _StubTransfer.started[0]["start_len"] == 64, (
        "the read moved pages this cache already held"
    )
    assert _StubTransfer.started[0]["end_len"] == 96, (
        "the read stopped short of what the other side published"
    )
    kv.assert_pages_conserved()


def test_a_read_in_hashes_the_prompt_with_the_lock_down(monkeypatch):
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", tokens)
    kv.remove_request("a")
    _ingest(kv, "b", tokens)
    real = manager_mod.fingerprint

    def _fingerprint(*fields):
        free = []

        def _try():
            if kv._lock.acquire(blocking=False):
                kv._lock.release()
                free.append(True)

        attempt = threading.Thread(target=_try)
        attempt.start()
        attempt.join()
        assert free, "a key was hashed with the manager's lock held"
        return real(*fields)

    monkeypatch.setattr(manager_mod, "fingerprint", _fingerprint)

    assert kv.admit_retrieve(
        "b", NODE, WALK, _published(kv, "b", 64, list(range(4))),
    ).ok, "the read-in was refused"
    assert kv._streams["b"]["main"].stored_len == 64, "the read-in matched nothing"
    kv.assert_pages_conserved()


def test_a_read_in_that_matches_everything_published_moves_nothing():
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", tokens)
    kv.remove_request("a")

    _ingest(kv, "b", tokens)
    out = kv.admit_retrieve("b", NODE, WALK, _published(kv, "b", 64, list(range(4))))

    assert out.ok, "the read-in was refused"
    assert _StubTransfer.started == [], (
        "a transfer was started for bytes that were already here"
    )
    assert kv._streams["b"]["main"].stored_len == 64, (
        "the local match was not counted as tokens the stream holds"
    )
    kv.assert_pages_conserved()


def test_a_read_in_with_nothing_local_moves_all_of_it():
    kv = _manager()
    _ingest(kv, "b", list(range(2000, 2064)))

    out = kv.admit_retrieve("b", NODE, WALK, _published(kv, "b", 64, list(range(4))))

    assert out.ok, "the read-in was refused"
    assert _StubTransfer.started[0]["start_len"] == 0, (
        "a cache holding none of this prompt still skipped part of the read"
    )
    kv.assert_pages_conserved()


# ── offload ─────────────────────────────────────────────────────────────


@requires_cuda
def test_a_shared_page_can_be_evicted_while_its_request_is_away():
    kv = _manager(cpu_offload_pages=32)
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", tokens)
    shared = list(kv._streams["a"]["main"].page_indices)
    assert kv._index.pages(), "nothing was indexed, so nothing is shared"

    assert kv.offload("a") > 0, "the request did not move to the host"
    freed = kv._index.evict(len(shared))

    assert freed > 0, (
        "the pages the request left behind could not be given back while it "
        "was away, which is the whole reason its host copy is the good one"
    )
    assert kv.reload("a"), "the request could not come back"
    stream = kv._streams["a"]["main"]
    assert stream.stored_len == len(tokens), (
        "the reload brought back fewer tokens than the request had written"
    )
    assert not set(stream.page_indices) & set(shared), (
        "reload took back pages that had been handed to someone else"
    )
    kv.assert_pages_conserved()


@requires_cuda
def test_a_reload_takes_back_pages_only_the_index_was_holding():
    kv = _manager(max_num_pages=9, cpu_offload_pages=32)
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", tokens)
    assert kv.offload("a") > 0, "the request did not move to the host"
    # a second request fills what is left and ends, so every page in the pool
    # is one only the index holds
    other = list(range(1000, 1064))
    _ingest(kv, "b", other)
    _run(kv, "b", other)
    kv.remove_request("b")
    cached = set(kv._index.pages())
    assert kv._arena.num_free == 0, "the pool had room, so nothing had to be given back"

    assert kv.reload("a"), (
        "a request offloaded to make room could not come back to a pool of "
        "pages nobody was using"
    )
    assert set(kv._streams["a"]["main"].page_indices) <= cached, (
        "the pages the reload took did not come out of the index"
    )
    kv.assert_pages_conserved()


@requires_cuda
def test_a_pool_of_running_requests_still_refuses_a_reload():
    kv = _manager(max_num_pages=9, cpu_offload_pages=32)
    tokens = list(range(64))
    kv.ingest_request("a", KVReqConfig())
    _run(kv, "a", tokens)
    assert kv.offload("a") > 0, "the request did not move to the host"
    for rid, base in (("b", 1000), ("c", 2000)):
        others = list(range(base, base + 64))
        _ingest(kv, rid, others)
        _run(kv, rid, others)
    assert kv._arena.num_free == 0, "the pool had room, so a refusal proves nothing"

    assert not kv.reload("a"), (
        "the reload took pages two running requests are still reading"
    )
    assert kv.is_offloaded("a"), "a refused reload moved the request anyway"
    kv.assert_pages_conserved()
