"""KV pages (and the positions over them) held across a session.

A session's state lives under a reserved rid of its own, so the pages a request
built are handed to the session at removal and back to the next request at
ingest — with the arena's owner counts staying exactly as conserved as they are
without sessions. The position counters move with them: a resumed request that
started placing at 0 would write over the context the session holds.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"


class _StubTransfer:
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


def _manager(max_num_pages: int = 32, prefix_cache: bool = False) -> KVManager:
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    if prefix_cache:
        kv.enable_prefix_cache(ROOT, walks={"main": ("prefill", "decode")})
    return kv


def _grow(kv: KVManager, rid: str, span: int, label: str = "main") -> None:
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk="prefill", slot=0, capture=False,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _session_streams(kv: KVManager, session_id: str):
    return kv._streams.get(KVManager.session_rid(session_id), {})


# ── handing state to the session and back ───────────────────────────────────

def test_retain_moves_the_pages_to_the_session_untouched():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    pages = list(kv._streams["r0"]["main"].page_indices)

    kv.retain_session_state("r0", "s")

    assert "r0" not in kv._streams
    stream = _session_streams(kv, "s")["main"]
    assert stream.page_indices == pages
    assert stream.stored_len == 100
    assert [kv._arena.num_owners[p] for p in pages] == [1] * len(pages)


def test_adopt_hands_the_session_s_pages_to_the_next_request():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    pages = list(kv._streams["r0"]["main"].page_indices)
    kv.retain_session_state("r0", "s")

    kv.ingest_request("r1", KVReqConfig())
    kv.adopt_session_state("r1", "s")

    assert kv._streams["r1"]["main"].page_indices == pages
    assert kv._streams["r1"]["main"].stored_len == 100
    assert _session_streams(kv, "s") == {}


def test_the_resumed_request_appends_rather_than_overwrites():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    kv.retain_session_state("r0", "s")
    kv.ingest_request("r1", KVReqConfig())
    kv.adopt_session_state("r1", "s")

    _grow(kv, "r1", 20)

    assert kv._streams["r1"]["main"].stored_len == 120
    kv.assert_pages_conserved()


def test_adopting_nothing_leaves_the_fresh_request_alone():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())

    kv.adopt_session_state("r0", "brand-new")

    assert kv._streams["r0"]["main"].stored_len == 0
    kv.assert_pages_conserved()


def test_a_second_request_in_the_session_replaces_what_it_was_handed():
    kv = _manager()
    for rid in ("r0", "r1"):
        kv.ingest_request(rid, KVReqConfig())
        kv.adopt_session_state(rid, "s")
        _grow(kv, rid, 64)
        kv.retain_session_state(rid, "s")

    assert _session_streams(kv, "s")["main"].stored_len == 128
    kv.assert_pages_conserved()


def test_every_label_the_session_holds_comes_along():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 32, label="main")
    _grow(kv, "r0", 48, label="aux")
    kv.retain_session_state("r0", "s")

    kv.ingest_request("r1", KVReqConfig())
    kv.adopt_session_state("r1", "s")

    assert kv._streams["r1"]["main"].stored_len == 32
    assert kv._streams["r1"]["aux"].stored_len == 48


# ── freeing ─────────────────────────────────────────────────────────────────

def test_remove_session_gives_every_page_back():
    kv = _manager()
    free = kv._arena.num_free
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    kv.retain_session_state("r0", "s")

    kv.remove_session("s")

    assert kv._arena.num_free == free
    assert _session_streams(kv, "s") == {}
    kv.assert_pages_conserved()


def test_clear_session_state_is_the_same_as_removing_it():
    kv = _manager()
    free = kv._arena.num_free
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    kv.retain_session_state("r0", "s")

    kv.clear_session_state("s")

    assert kv._arena.num_free == free


def test_removing_a_request_that_was_retained_does_not_double_free():
    kv = _manager()
    free = kv._arena.num_free
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    kv.retain_session_state("r0", "s")

    kv.remove_request("r0")  # the rid is gone; the session still owns the pages

    assert kv._arena.num_free < free
    kv.remove_session("s")
    assert kv._arena.num_free == free


def test_removing_an_unknown_session_is_a_no_op():
    kv = _manager()
    free = kv._arena.num_free

    kv.remove_session("never-existed")

    assert kv._arena.num_free == free


# ── prefix reuse and sessions ───────────────────────────────────────────────
#
# The two share a page pool, so they have to coexist: a session's first request
# is keyed like any other, and only a *resumed* one steps out of the index —
# its keys cover the new turn alone, so page k of its stream is no longer page k
# of the chain those keys describe.

def _keyed(n_tokens: int, first: int = 0) -> list[bytes]:
    tokens = list(range(first, first + n_tokens))
    return chain([
        tokens[at:at + PAGE_SIZE]
        for at in range(0, len(tokens), PAGE_SIZE)
    ])


def _indexed(kv: KVManager) -> int:
    return len(kv._index.pages())


def test_a_resumed_request_stops_using_the_prefix_cache():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(prefix_cache=True))
    _grow(kv, "r0", 100)
    kv.retain_session_state("r0", "s")

    kv.ingest_request("r1", KVReqConfig(prefix_cache=True))
    kv.adopt_session_state("r1", "s")

    assert kv._overrides["r1"].prefix_cache is False


def test_the_session_s_first_request_keeps_the_prefix_cache():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(prefix_cache=True))

    kv.adopt_session_state("r0", "s")  # the session holds nothing yet

    assert kv._overrides["r0"].prefix_cache is True


def test_a_resumed_request_neither_probes_the_index_nor_files_into_it():
    kv = _manager(prefix_cache=True)
    keys = {"main": _keyed(64)}
    kv.ingest_request("r0", KVReqConfig(prefix_keys=keys))
    _grow(kv, "r0", 64)
    filed = _indexed(kv)
    assert filed == 4, "the session's first request is keyed like any other"
    kv.retain_session_state("r0", "s")

    kv.ingest_request("r1", KVReqConfig(prefix_keys=keys))
    assert kv._keyed_label("r1", "LLM", "prefill") == "main"
    kv.adopt_session_state("r1", "s")

    # one switch shuts the probe, the apply, the extend and the filing
    assert kv._keyed_label("r1", "LLM", "prefill") is None
    assert kv.resolve_cached_prefix("r1", "LLM", "prefill") is None
    _grow(kv, "r1", 64)
    assert _indexed(kv) == filed
    kv.assert_pages_conserved()


def test_a_fresh_request_still_hits_what_the_session_s_first_request_cached():
    kv = _manager(prefix_cache=True)
    keys = {"main": _keyed(64)}
    kv.ingest_request("r0", KVReqConfig(prefix_keys=keys))
    _grow(kv, "r0", 64)
    kv.retain_session_state("r0", "s")

    kv.ingest_request("other", KVReqConfig(prefix_keys=keys))
    matched = kv.resolve_cached_prefix("other", "LLM", "prefill")

    # one key short, so a fully cached prompt still leaves a page to run
    assert matched == 3 * PAGE_SIZE


def test_an_adopted_stream_carries_no_chain():
    kv = _manager(prefix_cache=True)
    keys = {"main": _keyed(64)}
    kv.ingest_request("r0", KVReqConfig(prefix_keys=keys))
    _grow(kv, "r0", 64)
    kv.retain_session_state("r0", "s")
    kv.ingest_request("r1", KVReqConfig(prefix_keys=keys))

    kv.adopt_session_state("r1", "s")

    # nothing the session holds may be filed under this request's keys
    assert kv._streams["r1"]["main"].chain is None


# ── the budget ──────────────────────────────────────────────────────────────

def test_session_state_size_counts_the_pages_it_holds():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 3 * PAGE_SIZE)
    kv.retain_session_state("r0", "s")

    assert kv.session_state_size("s") == 3
    assert kv.session_state_size("other") == 0


def test_kv_declines_to_window_its_state():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig())
    _grow(kv, "r0", 100)
    kv.retain_session_state("r0", "s")

    # the stored keys are already rotated, so keeping a suffix is not sound
    assert kv.trim_session_state("s", 1) is False
    assert kv.session_state_size("s") > 1


# ── positions over the session's pages ──────────────────────────────────────

def _position_manager():
    from mstar.engine.resources.position.config import PositionConfig
    from mstar.engine.resources.position.manager import RopeManager

    return RopeManager(
        config=PositionConfig(kv_cache="kv"),
        device=torch.device("cpu"),
        head_dim=8,
    )


def test_position_counters_travel_with_the_session():
    pos = _position_manager()
    pos.ingest_request("r0")
    pos._counters["r0"]["main"] = 100

    pos.retain_session_state("r0", "s")

    assert "r0" not in pos._counters
    assert pos.session_state_size("s") == 100

    pos.ingest_request("r1")
    pos.adopt_session_state("r1", "s")

    assert pos._counters["r1"]["main"] == 100


def test_position_counters_are_dropped_with_the_session():
    pos = _position_manager()
    pos.ingest_request("r0")
    pos._counters["r0"]["main"] = 100
    pos.retain_session_state("r0", "s")

    pos.remove_session("s")

    assert pos.session_state_size("s") == 0


def test_a_fresh_session_leaves_the_counters_at_zero():
    pos = _position_manager()
    pos.ingest_request("r0")

    pos.adopt_session_state("r0", "brand-new")

    assert pos._counters["r0"] == {}
