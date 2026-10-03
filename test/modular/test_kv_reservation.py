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

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVSpec, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"
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


def _manager(
    max_num_pages: int = 64, max_seq_len: int = 4096, nodes: set[str] | None = None,
) -> KVManager:
    return KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32, nodes=nodes,
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


def _ctx(*rids: str, capture: bool = False) -> StepContext:
    return StepContext(request_ids=rids, graph_walk=WALK, slot=0, capture=capture)


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


def test_a_cold_request_reserves_its_prompt_and_max_tokens_of_growth():
    kv = _manager()
    kv.ingest_request("r", _keyed(list(range(50)), max_tokens=40))

    _run(kv, "r", 50)

    assert kv._reserved["r"].pages == _pages(50 + 40), (
        "a request was admitted on other than its prompt and its growth"
    )
    assert kv._outstanding() == _pages(90) - _pages(50), (
        "what the prompt already took was still counted as owed"
    )


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
    assert kv._reserved["b"].pages == _pages(len(tokens) + 32) - leased, (
        "the pages the lease already holds were reserved again"
    )


def test_removal_gives_the_reservation_back():
    kv = _manager()
    kv.ingest_request("r", _keyed(list(range(50)), max_tokens=40))
    _run(kv, "r", 50)

    kv.remove_request("r")

    assert kv._outstanding() == 0, "a removed request still held room it can never take"


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


def test_decode_adds_at_most_max_seq_len_to_a_models_count():
    kv = _manager(max_seq_len=64)
    kv.ingest_request("r", KVReqConfig(
        max_tokens=444, prompt_slots={"main": 20}, decode_labels=["main"],
    ))

    _run(kv, "r", 20)

    assert kv._reserved["r"].pages == _pages(20 + 64), (
        "decode was reserved past the positions max_seq_len allows"
    )


def test_a_request_nothing_counts_is_left_to_run_as_before():
    # no count from the model and no keys: max_seq_len is all that bounds it,
    # which for most models is far more than the pool can spare per request
    kv = _manager(max_seq_len=64)
    kv.ingest_request("r", KVReqConfig(max_tokens=444))

    _run(kv, "r", 20)

    assert "r" not in kv._reserved, "a request with nothing to size it by was held to a guess"


def test_a_label_opened_only_on_another_node_is_not_reserved():
    # a single-GPU Bagel pool: the guidance nodes' labels live elsewhere
    kv = _manager(nodes={"LLM"})
    kv.ingest_request("r", KVReqConfig(
        max_tokens=16,
        needed_labels_per_node_walk={
            ("LLM", WALK): ["main"], ("LLM_cfg_text", WALK): ["cfg_text"],
        },
        prompt_slots={"main": 32, "cfg_text": 32}, decode_labels=["main"],
    ))

    _run(kv, "r", 32)

    assert kv._reserved["r"].pages == _pages(32 + 16), (
        "a label this pool never holds for the request was reserved on it"
    )


def test_a_worker_reserves_only_what_its_own_nodes_open():
    # CFG parallel: every worker builds the cache from a spec naming all three
    spec = KVSpec(
        resource_key="kv", nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"},
        config=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE_SIZE,
        ),
    )
    kv = KVManager.build(spec, EngineResourceInfo(
        device=torch.device("cpu"), kv_dtype=torch.float32, nodes=frozenset({"LLM"}),
    ))
    kv.ingest_request("r", KVReqConfig(
        max_tokens=16,
        needed_labels_per_node_walk={
            ("LLM", WALK): ["main"],
            ("LLM_cfg_text", "image_gen_cfg"): ["cfg_text"],
            ("LLM_cfg_img", "image_gen_cfg"): ["cfg_img"],
        },
        prompt_slots={"main": 32}, decode_labels=["main"],
    ))

    _run(kv, "r", 32)

    assert kv._reserved["r"].pages == _pages(32 + 16), (
        "a worker reserved labels only the other workers' nodes open"
    )


# ── the check ───────────────────────────────────────────────────────────


def test_the_check_catches_a_request_taking_more_than_it_reserved(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager()
    kv.ingest_request("r", KVReqConfig(
        max_tokens=16, prompt_slots={"main": PAGE_SIZE}, decode_labels=[],
    ))

    with pytest.raises(AssertionError, match="took more pages than they reserved"):
        _run(kv, "r", 3 * PAGE_SIZE)


def test_an_admitted_request_that_finds_no_page_is_an_assertion(monkeypatch):
    kv = _manager(max_num_pages=8)
    kv.ingest_request("r", KVReqConfig(
        max_tokens=16, prompt_slots={"main": PAGE_SIZE}, decode_labels=["main"],
    ))
    assert kv.admit_retrieve("r", "LLM", WALK, None).ready
    # a row with no max_tokens is never admitted by reservation, so it can take the room
    kv.ingest_request("filler", KVReqConfig())
    _run(kv, "filler", 7 * PAGE_SIZE)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)

    with pytest.raises(AssertionError, match="was admitted on a reservation of"):
        kv.admit(KVStep(segments=(Segment("r", "main", PAGE_SIZE),)), _ctx("r"))
