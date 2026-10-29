"""A write past a request's layout is logged, and the debug checks report where."""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

import pytest
from test_kv_prefix_lease import NODE, WALK, _ctx, _grow, _keys, _layout, _manager, _seed, _StubTransfer

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs

LOGGER = "mstar.engine.resources.kv.manager"


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


@pytest.fixture
def debug(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


@pytest.fixture
def always_on(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)


def _laid_out(kv, rid: str, spans=(PrefixSpan(20, 20, WALK),), **layout) -> None:
    kv.ingest_request(rid, KVReqConfig(
        kv_layout={"main": StreamLayout(spans=spans, **layout)}, loop_caps={"decode_loop": 3},
    ))


def test_a_label_past_its_footprint_logs_once(always_on, caplog):
    kv = _manager()
    _laid_out(kv, "r0")
    _laid_out(kv, "r1")

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        _grow(kv, "r0", 20)
        assert not caplog.records, "a stream inside its footprint was reported"
        _grow(kv, "r0", 20)
        _grow(kv, "r0", 20)
        _grow(kv, "r1", 60)

    assert len(caplog.records) == 1, "an overrun on one label was reported more than once"
    assert "footprint of 2" in caplog.text and "PrefixSpan(length=20" in caplog.text, (
        "the line doesn't name the layout that under-counted"
    )


def test_a_write_from_another_walk_fails_its_commit(debug):
    kv = _manager()
    _laid_out(kv, "r0")
    step = KVStep(segments=(Segment("r0", "main", 20),))
    ctx = StepContext(request_ids=("r0",), graph_walk="decode", slot=0, capture=False)
    assert kv.admit(step, ctx).ok

    with pytest.raises(AssertionError, match="decode wrote into span 0"):
        kv.commit(step, ctx)


def test_growth_past_its_bound_fails_its_commit(debug):
    kv = _manager()
    _laid_out(kv, "r0", growth=(("decode_loop", 2),))
    _grow(kv, "r0", 20)
    for _ in range(3):
        _grow(kv, "r0", 2)

    with pytest.raises(AssertionError, match="past its layout's spans and growth"):
        _grow(kv, "r0", 1)


def test_scratch_past_its_declaration_fails_its_admit(debug):
    kv = _manager()
    _laid_out(kv, "r0", scratch=4)
    _grow(kv, "r0", 20)

    with pytest.raises(AssertionError, match="past its layout's scratch of 4"):
        kv.admit(KVStep(segments=(Segment("r0", "main", 5),), commit=False), _ctx("r0"))


def test_the_tail_after_a_cache_hit_fires_nothing(debug, monkeypatch):
    checked = []
    check = manager_mod.KVManager._assert_layout
    monkeypatch.setattr(manager_mod.KVManager, "_assert_layout", lambda self, segment, *args: (
        checked.append(segment.request_id), check(self, segment, *args),
    ))
    kv = _manager()
    tokens = list(range(100))
    keys = _seed(kv, tokens)
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": _keys(tokens)}, kv_layout=_layout(100), loop_caps={}))
    inputs = ARNodeInputs(input_seq_len=100)
    prefix = kv.resolve_cached_prefix("r1", NODE, WALK, inputs)
    kv.apply_cached_prefix("r1", NODE, WALK, inputs, prefix)

    _grow(kv, "r1", 100 - prefix.tokens)

    assert "r1" in checked, "the tail's commit was never checked against its layout"
    assert keys and kv._streams["r1"]["main"].span_cursor.done == 1, "the tail never completed the span the hit began"


def test_intake_drops_the_layout_of_a_label_the_request_does_not_open():
    config = KVReqConfig(needed_labels=["main"])
    config.apply_conductor_config(kv_layout={
        "main": [[[20, 20, WALK, 0]], [["decode_loop", 1]], 0, None, 0],
        "cfg_img": [[], [["decode_loop", 1]], 40, None, 0],
    }, loop_caps={"decode_loop": 3})

    assert list(config.kv_layout) == ["main"], "the request would be held to a label it never opens"


def test_a_label_the_request_does_not_open_holds_no_pages(debug, caplog, monkeypatch):
    checked = []
    check = manager_mod.KVManager._assert_layout
    monkeypatch.setattr(manager_mod.KVManager, "_assert_layout", lambda self, segment, *args: (
        checked.append(segment.label), check(self, segment, *args),
    ))
    kv = _manager()
    _laid_out(kv, "r0")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _grow(kv, "r0", 20)
        _grow(kv, "r0", 60, label="cfg_img")
        kv.remove_request("r0")

    assert checked == ["main"], "a label the request doesn't open was checked against its layout"
    assert [r.levelno for r in caplog.records] == [logging.ERROR], "expected one overrun line and no under-use line"
    assert "label cfg_img against a footprint of 0" in caplog.text, (
        "pages a batch took for the request on a label it doesn't open left no line"
    )


def test_a_keyed_label_the_request_does_not_open_gets_no_chain():
    kv = _manager()
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"cfg_img": _keys(list(range(40)))}, kv_layout=_layout(40), loop_caps={},
    ))

    _grow(kv, "r0", 20, label="cfg_img")

    stream = kv._streams["r0"]["cfg_img"]
    assert stream.span_cursor is None and stream.chain is None, "a chain without a cursor fails the next probe"


def test_a_label_left_under_its_footprint_is_reported_at_removal(debug, caplog):
    kv = _manager()
    _laid_out(kv, "r0", growth=(("decode_loop", 16),))
    _grow(kv, "r0", 20)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        kv.remove_request("r0")

    assert "held at most 2 pages on label main against a footprint of 5" in caplog.text, (
        "an over-counted layout left no line to find it by"
    )
