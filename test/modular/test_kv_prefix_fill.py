"""A page joins the index when it fills, and not a token before.

Pages fill across steps — a prefill arrives in chunks, a decode adds one token a
step — so what a commit can offer is whatever whole pages the stream now holds.
Offering the tail as well would hand a second owner a page that is still being
written into, which is the one thing page ownership cannot survive: the reader
would see bytes change under it.

The other half is who is allowed to offer anything. A capture step writes
whatever the graph was captured with, and a padded row is not a request, so
neither may leave anything behind for a later request to match.
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

PAGE_SIZE = 128
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"


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


def _manager(max_num_pages: int = 64) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _keys(n_tokens: int, first: int = 0) -> list[bytes]:
    tokens = list(range(first, first + n_tokens))
    return chain([
        tokens[at:at + PAGE_SIZE]
        for at in range(0, len(tokens), PAGE_SIZE)
    ])


def _step(
    kv: KVManager, rid: str, span: int, label: str = "main",
    capture: bool = False, padded: bool = False,
) -> None:
    """One step's whole lifecycle for ``rid``."""
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = StepContext(
        request_ids=() if (capture or padded) else (rid,),
        graph_walk=WALK, slot=0, capture=capture,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _indexed(kv: KVManager) -> int:
    return len(kv._index.pages())


# ── as the pages fill ───────────────────────────────────────────────────


def test_a_chunked_prefill_indexes_the_page_the_second_chunk_completes():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": _keys(128)}))

    _step(kv, "r0", 100)
    after_first = _indexed(kv)
    _step(kv, "r0", 28)

    assert after_first == 0, "a page was indexed while it was still being written"
    assert _indexed(kv) == 1, "the page the second chunk filled was not indexed"
    kv.assert_pages_conserved()


def test_only_whole_pages_are_offered():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": _keys(300)}))

    _step(kv, "r0", 300)

    assert _indexed(kv) == 2, "300 tokens is two whole pages and a tail of 44"
    kv.assert_pages_conserved()


def test_a_second_request_with_the_same_prompt_matches_what_was_indexed():
    kv = _manager()
    keys = _keys(512)
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": keys}))
    _step(kv, "r0", 512)

    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))

    assert kv.resolve_cached_prefix("r1", NODE, WALK) == 3 * PAGE_SIZE, (
        "the four indexed pages did not turn into a match of three"
    )
    kv.assert_pages_conserved()


def test_a_request_that_was_served_from_the_cache_does_not_reindex_it():
    kv = _manager()
    keys = _keys(512)
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": keys}))
    _step(kv, "r0", 512)
    indexed = _indexed(kv)

    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    matched = kv.resolve_cached_prefix("r1", NODE, WALK)
    _step(kv, "r1", 512 - matched)

    assert _indexed(kv) == indexed, (
        "the pages the cache handed over were offered back to it"
    )
    kv.assert_pages_conserved()


def test_the_first_request_to_fill_a_page_is_the_one_the_index_names():
    kv = _manager()
    keys = _keys(256)
    for rid in ("r0", "r1"):
        kv.ingest_request(rid, KVReqConfig(prefix_keys={"main": keys}))
    _step(kv, "r0", 256)
    winner = list(kv._index.pages())

    _step(kv, "r1", 256)

    assert list(kv._index.pages()) == winner, (
        "the second request to fill the page took the index entry"
    )
    assert not set(winner) & set(kv._streams["r1"]["main"].page_indices), (
        "the loser's private pages were handed to the index"
    )
    kv.assert_pages_conserved()


def test_a_rewound_stream_indexes_its_rerun_from_page_zero():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": _keys(512)}))
    _step(kv, "r0", 512)
    kv.reset_request("r0")
    # nothing left from the first run, so the rerun is the only thing to index
    kv._index.evict(kv.config.max_num_pages)

    _step(kv, "r0", 512)

    assert set(kv._index.pages()) == set(kv._streams["r0"]["main"].page_indices), (
        "the rerun picked the chain up where the first run left it and filed "
        "none of the pages it wrote"
    )
    kv.assert_pages_conserved()


# ── who may not offer anything ──────────────────────────────────────────


def test_a_capture_step_indexes_nothing():
    kv = _manager()
    kv.ingest_request("d0", KVReqConfig(prefix_keys={"main": _keys(256)}))

    # the rid is in the batch, so only the capture itself can keep it out
    _step(kv, "d0", 256, capture=True)
    captured = _indexed(kv)
    kv.reset_request("d0", free=True)
    _step(kv, "d0", 256)

    assert captured == 0, "a captured graph's dummy data entered the index"
    assert _indexed(kv) == 2, "the same step outside a capture indexed nothing either"


def test_a_padded_row_indexes_nothing():
    kv = _manager()
    kv.ingest_request("pad", KVReqConfig(prefix_keys={"main": _keys(256)}))

    _step(kv, "pad", 256, padded=True)
    padded = _indexed(kv)
    kv.reset_request("pad", free=True)
    _step(kv, "pad", 256)

    assert padded == 0, "a row that is not a request left pages behind"
    assert _indexed(kv) == 2, "the same row as a request indexed nothing either"


def test_a_label_the_request_never_keyed_indexes_nothing():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": _keys(256)}))

    _step(kv, "r0", 256, label="cfg_text")
    unkeyed = _indexed(kv)
    _step(kv, "r0", 256)

    assert unkeyed == 0, "an unkeyed label was indexed under another's keys"
    assert _indexed(kv) == 2, "the keyed label of the same request indexed nothing"
    kv.assert_pages_conserved()


def test_a_request_that_opted_out_indexes_nothing():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": _keys(256)}, prefix_cache=False,
    ))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": _keys(256)}))

    _step(kv, "r0", 256)
    opted_out = _indexed(kv)
    _step(kv, "r1", 256)

    assert opted_out == 0, "a request that turned the cache off was indexed"
    assert _indexed(kv) == 2, "the request beside it, on the same keys, was not"
    kv.assert_pages_conserved()
