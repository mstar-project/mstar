"""A request reserves the pages it can still take, and the pool keeps count.

A running request keeps needing pages as it decodes. When a step finds none,
the only way out is to evict or offload someone, and with neither possible the
worker holds the batch until the clients give up: the 96-page repro holds all
32 of its requests. So each admitted request reserves what its life on the pool
can take from the free list, its uncached prompt plus ``max_tokens`` of growth,
and the pool checks that everything admitted can still be served from what is
free or evictable. A page leased from the index is held, not taken, and a page
counted twice leaves no room for anyone.
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
WALK = "prefill"


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 64, max_seq_len: int = 4096) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _keyed(tokens: list[int], max_tokens: int) -> KVReqConfig:
    whole = len(tokens) // PAGE_SIZE
    return KVReqConfig(
        max_tokens=max_tokens,
        prefix_keys={"main": chain([
            tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
        ])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(request_ids=rids, graph_walk=WALK, slot=0, capture=False)


def _run(kv: KVManager, rid: str, span: int, label: str = "main") -> None:
    """Admit, plan and commit one step extending ``label`` by ``span``."""
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = _ctx(rid)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _pages(tokens: int) -> int:
    return -(-tokens // PAGE_SIZE)


# ── what a request reserves ─────────────────────────────────────────────


def test_a_leased_hit_is_held_rather_than_reserved():
    kv = _manager()
    kv.enable_prefix_cache(ROOT)
    tokens = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _keyed(tokens, max_tokens=32))
    _run(kv, "a", len(tokens))

    kv.ingest_request("b", _keyed(tokens, max_tokens=32))
    # one key short, so three of the four prompt pages come from the index
    leased = kv.resolve_cached_prefix("b", "LLM", WALK) // PAGE_SIZE
    _run(kv, "b", len(tokens) - leased * PAGE_SIZE)

    assert leased == 3
    # 64 of prompt, 32 of decode and the speculative step's token: 7 pages
    assert kv._reserved["b"].pages == 7 - leased, (
        "the pages the lease already holds were reserved again"
    )


def test_a_models_count_past_max_seq_len_is_reserved_whole():
    # Bagel's image tokens: counted by the model, but not positions
    kv = _manager(max_seq_len=64)
    kv.ingest_request("r", KVReqConfig(
        max_tokens=16, prompt_slots={"main": 200}, decode_labels=["main"],
    ))

    _run(kv, "r", 20)

    assert kv._reserved["r"].pages == _pages(200 + 16), (
        "a model's own count was cut down to max_seq_len, leaving the request short"
    )
