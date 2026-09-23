"""A page's chain parent is the page the index named, not the one it is holding.

Two things separate a stream from the pages the index holds for it. Two requests
filling the same page in one batch both write a copy, but only the first to
commit gets the entry, so the loser goes on holding a page nobody else can
reach. And a reloaded request is given fresh pages and copied back into them,
while the entries it left behind still name the originals.

In both cases the stream's next page has to chain onto what the index named. Off
its own page list it would name a page the index has never heard of, and the
insert refuses that, because a parent has to outlive its children.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain, fingerprint
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the host pool pins its memory"
)

PAGE_SIZE = 16
ROOT = b"a root"
WALK = "prefill"


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
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


def _ingest(kv: KVManager, rid: str, tokens: list[int]) -> None:
    whole = len(tokens) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain([
            tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
        ])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
    ))


def _batch(kv: KVManager, rids: tuple[str, ...], span: int) -> None:
    """One step carrying a segment for every rid, as a real batch does."""
    step = KVStep(segments=tuple(Segment(rid, "main", span) for rid in rids))
    ctx = StepContext(
        request_ids=rids, graph_walk=WALK, slot=0, capture=False,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _indexed_page(kv: KVManager, rid: str, page: int) -> int | None:
    keys = kv._streams[rid]["main"].chain.keys
    return kv._index.page_for(fingerprint(ROOT, keys[page]))


# ── two requests filling the same page at once ──────────────────────────


def test_the_loser_of_a_race_chains_its_next_page_onto_the_winners():
    kv = _manager()
    tokens = list(range(64))
    for rid in ("r0", "r1"):
        _ingest(kv, rid, tokens)

    # one batch fills page 0 for both; only one of them gets the entry
    _batch(kv, ("r0", "r1"), PAGE_SIZE)
    winner = _indexed_page(kv, "r0", 0)
    loser = next(
        rid for rid in ("r0", "r1")
        if kv._streams[rid]["main"].page_indices[0] != winner
    )
    assert kv._streams[loser]["main"].page_indices[0] != winner, (
        "both requests ended up on one page, so there was no race to lose"
    )

    # the next batch fills page 1, and the loser has to reach past its own
    _batch(kv, ("r0", "r1"), PAGE_SIZE)

    assert _indexed_page(kv, loser, 1) is not None, "page 1 never reached the index"
    assert kv._index._parent[_indexed_page(kv, loser, 1)] == winner, (
        "the loser chained page 1 onto its own copy of page 0, which the index "
        "does not hold"
    )
    kv.assert_pages_conserved()


def test_the_loser_keeps_its_own_page_and_the_index_keeps_one_entry():
    kv = _manager()
    tokens = list(range(64))
    for rid in ("r0", "r1"):
        _ingest(kv, rid, tokens)

    _batch(kv, ("r0", "r1"), PAGE_SIZE)

    assert len(kv._index.pages()) == 1, "two copies of one page were both indexed"
    pages = {rid: kv._streams[rid]["main"].page_indices[0] for rid in ("r0", "r1")}
    assert pages["r0"] != pages["r1"], "the two requests shared a page they filled"
    kv.assert_pages_conserved()


# ── a request that came back onto different pages ───────────────────────


@requires_cuda
def test_a_reloaded_request_chains_its_next_page_onto_the_original():
    kv = _manager(cpu_offload_pages=64)
    tokens = list(range(160))
    _ingest(kv, "a", tokens)
    _batch(kv, ("a",), 80)
    indexed = len(kv._index.pages())
    original = _indexed_page(kv, "a", 4)

    assert kv.offload("a") > 0, "the request did not move to the host"
    assert kv.reload("a"), "the request could not come back"
    _batch(kv, ("a",), 80)

    assert len(kv._index.pages()) == indexed + 5, (
        "the pages filled after the reload never reached the index"
    )
    assert original not in kv._streams["a"]["main"].page_indices, (
        "reload handed back the very pages the index named, so this proves "
        "nothing about chaining onto a copy"
    )
    assert kv._index._parent[_indexed_page(kv, "a", 5)] == original, (
        "the reloaded stream chained onto its own copy rather than onto the "
        "page the index named"
    )
    kv.assert_pages_conserved()
