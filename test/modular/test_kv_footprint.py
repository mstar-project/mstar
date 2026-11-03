"""A stream's footprint is never below the pages the manager gives it."""

from __future__ import annotations

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv.config import (
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.windowing import WindowSchedule

PS = 8


def _manager() -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=512 * PS,
            max_num_pages=512, page_size=PS,
        ),
        name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _grow(
    kv: KVManager, rid: str, span: int, label: str = "main", commit: bool = True,
    retention: RetentionPolicy | None = None,
) -> int:
    step = KVStep(
        segments=(Segment(rid, label, span),), commit=commit,
        retention={} if retention is None else {(rid, label): retention},
    )
    ctx = StepContext(request_ids=(rid,), graph_walk="w", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    held = len(kv._streams[rid][label].page_indices)
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    return held


def _spans(*lengths: int) -> tuple[PrefixSpan, ...]:
    return tuple(PrefixSpan(length, length, "w") for length in lengths)


def test_each_label_rounds_up_on_its_own():
    kv = _manager()
    kv.ingest_request("r")
    held = _grow(kv, "r", 3, "main") + _grow(kv, "r", 3, "uncond")
    layout = StreamLayout(spans=_spans(3))

    assert held == 2 * layout.pages({}, PS), "rounding the labels' sum once is a page short"


def test_growth_is_its_cap_times_its_slots():
    kv = _manager()
    kv.ingest_request("r")
    held = _grow(kv, "r", 5)
    for _ in range(7):
        held = _grow(kv, "r", 2)
    layout = StreamLayout(spans=_spans(5), growth=(("decode_loop", 2),))

    assert held == layout.pages({"decode_loop": 7}, PS) == 3


def test_scratch_counts_once_however_often_it_runs():
    kv = _manager()
    kv.ingest_request("r")
    held = _grow(kv, "r", 6)
    for _ in range(3):
        held = max(held, _grow(kv, "r", 11, commit=False))
    layout = StreamLayout(spans=_spans(6), scratch=11)

    assert held == layout.pages({}, PS) == 3


def test_a_window_holds_its_peak_not_its_sum():
    for prefix, budget, window in [(13, 24, 9), (16, 16, 16), (5, 40, 7), (0, 3, 12)]:
        kv = _manager()
        kv.ingest_request("r")
        policy = RetentionPolicy(context_budget=budget, protected_prefix=prefix)
        peak = _grow(kv, "r", prefix) if prefix else 0
        for _ in range(12):
            peak = max(peak, _grow(kv, "r", window, commit=False))
            peak = max(peak, _grow(kv, "r", window, retention=policy))
        layout = StreamLayout(
            spans=_spans(*([prefix] if prefix else []), *[window] * 12),
            scratch=window, retention=policy,
        )

        assert peak <= layout.pages({}, PS), f"{prefix=} {budget=} {window=}: a window outgrew its footprint"
        assert layout.pages({}, PS) - peak <= 1, f"{prefix=} {budget=} {window=}: the footprint over-counts"


def test_a_window_denoised_before_its_commit_counts_once():
    schedule, per_unit, prompt = WindowSchedule(10, 4, overlap_units=1), 3, 6
    kv = _manager()
    kv.ingest_request("r")
    peak = _grow(kv, "r", prompt)
    windows = []
    for plan in schedule.windows():
        commit = (plan.commit_end - plan.commit_start) * per_unit
        peak = max(peak, _grow(kv, "r", plan.units * per_unit, commit=False), _grow(kv, "r", commit))
        windows.append(PrefixSpan(commit, commit, "w", scratch=plan.units * per_unit))
    layout = StreamLayout(spans=(*_spans(prompt), *windows))

    assert peak == layout.pages({}, PS), "the last window's denoise is counted on top of its commit"
