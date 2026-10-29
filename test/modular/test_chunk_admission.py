"""A chunked prompt admits its whole walk on its first chunk.

Admitting one chunk's worth let eight long prompts in at once on a tight pool,
each holding the pages of the chunks it had written while it waited for pages
the others held, and nothing finished (PR #365 review). Reserving the rest of
the walk up front is what an unchunked prompt does, so the pool behaves as on
main: a prompt that does not fit is refused before it holds anything.

CPU-only: a real ``KVManager``, with the KV transfer engine stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.chunking import ChunkingManagement
from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import (
    ADMIT_OK,
    ALL_SET,
    NONE_SET,
    AllocationFailed,
    PerReqSetting,
    Segment,
    SettingValue,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import ARNodeInputs, ChunkingPolicy

PAGE = 16


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def get_kv_transfer_info(self, **kwargs):
        return None

    def owns_transfer_info(self, *args, **kwargs):
        return False

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _kv(pages: int) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=pages, page_size=PAGE, cpu_offload_pages=0,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids, after=None, reserve_ahead=ALL_SET) -> StepContext:
    ctx = StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False)
    ctx.chunk_tokens_after = after or {}
    ctx.reserve_whole_prefill = reserve_ahead
    return ctx


def _step(**spans) -> KVStep:
    return KVStep(segments=tuple(Segment(int(rid[1:]), "main", n) for rid, n in spans.items()))


def test_a_first_chunk_reserves_the_rest_of_its_walk():
    kv = _kv(pages=32)
    kv.ingest_request(1, None)

    assert kv.admit(_step(r1=PAGE), _ctx(1, after={1: 4 * PAGE})).ok

    assert len(kv._streams[1]["main"].page_indices) == 5


def test_the_manager_reserves_ahead_only_when_the_step_says_so():
    """``chunk_tokens_after`` is what the row has left either way; the flag is
    the walk's say that those tokens are a span this label can reserve."""
    kv = _kv(pages=32)
    kv.ingest_request(1, None)

    assert kv.admit(_step(r1=PAGE), _ctx(1, after={1: 4 * PAGE}, reserve_ahead=NONE_SET)).ok

    assert len(kv._streams[1]["main"].page_indices) == 1


def test_a_partial_setting_reserves_ahead_for_its_rows_alone():
    kv = _kv(pages=32)
    kv.ingest_request(1, None)
    kv.ingest_request(2, None)
    partial = PerReqSetting(SettingValue.PARTIAL, frozenset({1}))

    assert kv.admit(
        _step(r1=PAGE, r2=PAGE), _ctx(1, 2, after={1: 4 * PAGE, 2: 4 * PAGE}, reserve_ahead=partial),
    ).ok

    assert len(kv._streams[1]["main"].page_indices) == 5
    assert len(kv._streams[2]["main"].page_indices) == 1


def test_a_later_chunk_takes_no_more_pages():
    kv = _kv(pages=32)
    kv.ingest_request(1, None)
    first = _step(r1=PAGE)
    ctx = _ctx(1, after={1: 2 * PAGE})
    assert kv.admit(first, ctx).ok
    kv.commit(first, ctx)
    free = kv._arena.num_free

    assert kv.admit(_step(r1=PAGE), _ctx(1, after={1: PAGE})).ok

    assert kv._arena.num_free == free


def test_a_walk_that_does_not_fit_is_refused_before_it_holds_anything():
    kv = _kv(pages=8)  # 7 usable: page 0 is the sink
    kv.ingest_request(1, None)
    free = kv._arena.num_free

    outcome = kv.admit(_step(r1=PAGE), _ctx(1, after={1: 10 * PAGE}))

    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed)
    kv.rollback_admit(_step(r1=PAGE), _ctx(1))
    assert kv._arena.num_free == free


def test_a_refused_reservation_unwinds_with_the_step():
    kv = _kv(pages=32)
    kv.ingest_request(1, None)
    free = kv._arena.num_free
    step, ctx = _step(r1=PAGE), _ctx(1, after={1: 3 * PAGE})
    assert kv.admit(step, ctx).ok

    kv.rollback_admit(step, ctx)

    assert kv._arena.num_free == free
    assert kv._streams[1]["main"].page_indices == []


class _Sub:
    def __init__(self, reserve_whole_prefill=True, other_reserves=True):
        self.policy = ChunkingPolicy(
            walks=("prefill",), chunkable=True, max_batch_tokens=8,
            reserve_whole_prefill=reserve_whole_prefill,
        )
        # a second chunked walk, for a step that mixes the two
        self.other = ChunkingPolicy(
            walks=("other",), chunkable=True, max_batch_tokens=8,
            reserve_whole_prefill=other_reserves,
        )

    def declare_step(self, **kwargs):
        return SubmoduleStep(steps={})


class _Runner:
    def __init__(self):
        self.ctx = None

    def admit(self, step):
        self.ctx = step.ctx
        return ADMIT_OK


def _declare(sub: "_Sub", other_walk_row: bool = False):
    """``_declare_and_admit`` over one non-final chunk, one final one and one
    unchunked row, plus a non-final chunk of walk ``other`` if asked; returns
    the ctx the runner was admitted with."""
    eng = Engine.__new__(Engine)
    eng._runner = _Runner()
    eng._submodules = {"LLM": SimpleNamespace(piecewise_runners={}, submodule=sub)}
    eng._chunking = ChunkingManagement(eng._submodules)
    eng._chunking.set_policies({("LLM", "prefill"): sub.policy, ("LLM", "other"): sub.other})
    rids = [1, 2, 3, 4] if other_walk_row else [1, 2, 3]
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False,
        request_walks={4: "other"} if other_walk_row else {},
    )
    chunk_ranges = {1: (0, 4, 10), 2: (4, 10, 10)}
    inputs = [
        ARNodeInputs(input_seq_len=4, chunk_start=0, chunk_total=10),   # more to come
        ARNodeInputs(input_seq_len=6, chunk_start=4, chunk_total=10),   # its last chunk
        ARNodeInputs(input_seq_len=1),                                  # not chunked
    ]
    if other_walk_row:
        chunk_ranges[4] = (0, 4, 10)
        inputs.append(ARNodeInputs(input_seq_len=4, chunk_start=0, chunk_total=10))
    batch = SimpleNamespace(
        node_name="LLM", graph_walk="prefill", per_request_info={},
        chunk_ranges=chunk_ranges, step_context=ctx,
    )

    outcome, _ = eng._declare_and_admit(
        batch, rids=rids, inputs=inputs, submodule=sub, nvtx=False, ctx=ctx,
    )

    assert outcome.ok
    return eng._runner.ctx


def test_the_engine_reports_only_non_final_chunks():
    ctx = _declare(_Sub())
    assert ctx.chunk_tokens_after == {1: 6}
    assert ctx.reserve_whole_prefill == ALL_SET


def test_a_walk_that_reserves_per_chunk_still_reports_what_is_left():
    """``reserve_whole_prefill=False``: the walk's KV does not grow by the
    tokens a step runs, so the counts stand but no resource reserves by them."""
    ctx = _declare(_Sub(reserve_whole_prefill=False))
    assert ctx.chunk_tokens_after == {1: 6}
    assert ctx.reserve_whole_prefill == NONE_SET


def test_a_step_mixing_the_two_kinds_of_walk_names_the_rows_that_reserve():
    """Each row reserves as its own walk says, rather than one answer for the
    step overriding one of them."""
    ctx = _declare(_Sub(other_reserves=False), other_walk_row=True)
    assert ctx.chunk_tokens_after == {1: 6, 4: 6}
    assert ctx.reserve_whole_prefill == PerReqSetting(SettingValue.PARTIAL, frozenset({1}))
