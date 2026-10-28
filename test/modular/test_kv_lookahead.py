"""A draft-verify step commits its whole block and a trim takes back what the verdict
rejected. The layout's lookahead keeps such a request inside its page footprint, its
commit bound and its context."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.base import LabelLayout, Span

LOGGER = "mstar.engine.resources.kv.manager"
PAGE = 16
K = 3  # drafts per verify step
PROMPT = 49  # the prompt plus the cap ends on a page, so a write k slots past it takes one more
MAX_TOKENS = 32
CAPS = {"decode_loop": MAX_TOKENS - 1}  # the prefill emits the first token

# the trim, KVManager.correct_len, arrives with #388 and moves into DV-1's rewind
needs_trim = pytest.mark.skipif(
    not hasattr(KVManager, "correct_len"), reason="no KVManager.correct_len to trim with (#388)",
)


def _manager() -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _layout(lookahead: int) -> StreamLayout:
    return StreamLayout(
        spans=(PrefixSpan(PROMPT, PROMPT, "prefill"),), growth=(("decode_loop", 1),), lookahead=lookahead,
    )


def _step(kv: KVManager, walk: str, span: int, retention: RetentionPolicy | None = None) -> int:
    step = KVStep(
        segments=(Segment("r", "main", span),),
        retention={} if retention is None else {("r", "main"): retention},
    )
    ctx = StepContext(request_ids=("r",), graph_walk=walk, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    held = len(kv._streams["r"]["main"].page_indices)
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    return held


def _verify_to_the_cap(kv: KVManager, accepted: int) -> int:
    """Prefill, then verify steps that each accept ``accepted`` drafts until MAX_TOKENS are
    out: every step commits k + 1 slots and the trim takes back the rejected ones. Returns
    the most pages the stream held."""
    held = _step(kv, "prefill", PROMPT)
    emitted = 1
    while emitted < MAX_TOKENS:
        held = max(held, _step(kv, "decode", K + 1))
        emitted += min(accepted + 1, MAX_TOKENS - emitted)
        if accepted < K:
            kv.correct_len("r", "main", accepted - K)
    return held


@pytest.mark.parametrize("debug", [False, True], ids=["always_on", "debug_asserts"])
@pytest.mark.parametrize(
    "accepted", [pytest.param(0, marks=needs_trim), pytest.param(1, marks=needs_trim), K],
    ids=["accept0", "accept1", "acceptK"],
)
def test_a_verify_loop_to_its_cap_holds_its_footprint(accepted, debug, monkeypatch, caplog):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", debug)
    kv = _manager()
    kv.ingest_request("r", KVReqConfig(kv_layout={"main": _layout(K)}, loop_caps=CAPS))

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        held = _verify_to_the_cap(kv, accepted)

    stream = kv._streams["r"]["main"]
    assert held == len(stream.page_indices) == stream.footprint == 6, (held, stream.footprint)
    assert not caplog.records, caplog.text


@pytest.mark.parametrize("debug", [False, True], ids=["always_on", "debug_asserts"])
def test_without_the_lookahead_the_loop_outgrows_its_layout(debug, monkeypatch, caplog):
    # GLM-5.2's layout before the lookahead: one slot per iteration, and every block goes k past
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", debug)
    kv = _manager()
    kv.ingest_request("r", KVReqConfig(kv_layout={"main": _layout(0)}, loop_caps=CAPS))

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        if debug:
            with pytest.raises(AssertionError, match=r"committed 81 slots, past its layout's spans and growth \(80\)"):
                _verify_to_the_cap(kv, K)
        else:
            _verify_to_the_cap(kv, K)

    if not debug:
        assert "took 6 pages on label main against a footprint of 5" in caplog.text


def test_the_lookahead_counts_in_slots_and_pages_but_not_positions():
    layout = _layout(K)

    assert (layout.slots(CAPS), layout.positions(CAPS), layout.pages(CAPS, PAGE)) == (83, 80, 6)
    assert (_layout(0).slots(CAPS), _layout(0).positions(CAPS), _layout(0).pages(CAPS, PAGE)) == (80, 80, 5)


def test_a_window_counts_a_verify_block_past_its_prefix():
    # blocks of k + 1 = 3, all accepted, so no trim: the window's peak is a block over the
    # budget's last page, which one slot of growth alone doesn't cover
    prefix, k = 8, 2
    policy = RetentionPolicy(context_budget=24, protected_prefix=prefix)
    kv = _manager()
    kv.ingest_request("r")
    peak = _step(kv, "prefill", prefix)
    for _ in range(40):
        peak = max(peak, _step(kv, "decode", k + 1, retention=policy))
    caps = {"decode_loop": 40 * (k + 1)}
    spans = (PrefixSpan(prefix, prefix, "prefill"),)
    layout = StreamLayout(spans=spans, growth=(("decode_loop", 1),), retention=policy, lookahead=k)

    assert peak == layout.pages(caps, PAGE) == 4
    assert StreamLayout(spans=spans, growth=(("decode_loop", 1),), retention=policy).pages(caps, PAGE) == 3


def test_the_row_carries_the_lookahead_and_an_old_row_reads_none():
    spans = (Span("ids", "prefill", PROMPT, PROMPT, "text_inputs"),)
    plain = LabelLayout(spans=spans, growth=(("decode_loop", 1),)).row()
    ahead = LabelLayout(spans=spans, growth=(("decode_loop", 1),), lookahead=K).row()

    assert len(plain) == 5, "a layout without a lookahead changed its row"
    assert ahead == [*plain, K]
    for row, lookahead in ((plain, 0), (ahead, K)):
        config = KVReqConfig()
        config.apply_conductor_config(kv_layout={"main": row}, loop_caps=CAPS)
        assert config.kv_layout["main"] == replace(_layout(lookahead), keyed_spans=1)
        assert config.loop_caps == CAPS


def test_a_lookahead_beside_a_retention_policy_is_refused():
    layout = LabelLayout(growth=(("decode_loop", 1),), lookahead=K, retention=RetentionPolicy(context_budget=64))

    with pytest.raises(ValueError, match="both a lookahead and a retention policy"):
        layout.row()


def test_the_context_holds_the_rows_written_ahead():
    # positions at the caps plus the lookahead fit the pool's max_seq_len
    spec = KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=100, page_size=PAGE,
    ))
    room = 100 - PROMPT - K

    def config(cap: int, lookahead: int = K) -> KVReqConfig:
        return KVReqConfig(kv_layout={"main": _layout(lookahead)}, loop_caps={"decode_loop": cap})

    assert config(1).loop_room(spec) == {"decode_loop": room}
    assert config(1, lookahead=0).loop_room(spec) == {"decode_loop": room + K}
    config(room).validate(spec)
    with pytest.raises(ValueError, match=r"needs 101 positions of 'main' \(3 written ahead and trimmed\)"):
        config(room + 1).validate(spec)
