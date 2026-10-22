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

from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import (
    ADMIT_OK,
    AllocationFailed,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import ARNodeInputs

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


def _ctx(*rids, after=None) -> StepContext:
    ctx = StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False)
    ctx.chunk_tokens_after = after or {}
    return ctx


def _step(**spans) -> KVStep:
    return KVStep(segments=tuple(Segment(int(rid[1:]), "main", n) for rid, n in spans.items()))


def test_a_first_chunk_reserves_the_rest_of_its_walk():
    kv = _kv(pages=32)
    kv.ingest_request(1, None)

    assert kv.admit(_step(r1=PAGE), _ctx(1, after={1: 4 * PAGE})).ok

    assert len(kv._streams[1]["main"].page_indices) == 5


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
    def declare_step(self, **kwargs):
        return SubmoduleStep(steps={})


class _Runner:
    def __init__(self):
        self.ctx = None

    def admit(self, step):
        self.ctx = step.ctx
        return ADMIT_OK


def test_the_engine_reports_only_non_final_chunks():
    eng = Engine.__new__(Engine)
    eng._runner = _Runner()
    eng._submodules = {"LLM": SimpleNamespace(piecewise_runners={})}
    batch = SimpleNamespace(
        node_name="LLM", graph_walk="prefill", per_request_info={},
        chunk_ranges={1: (0, 4), 2: (4, 10)},
    )
    inputs = [
        ARNodeInputs(input_seq_len=4, chunk_start=0, chunk_total=10),   # more to come
        ARNodeInputs(input_seq_len=6, chunk_start=4, chunk_total=10),   # its last chunk
        ARNodeInputs(input_seq_len=1),                                  # not chunked
    ]
    ctx = StepContext(request_ids=(1, 2, 3), graph_walk="prefill", slot=0, capture=False)

    outcome, _ = eng._declare_and_admit(
        batch, rids=[1, 2, 3], inputs=inputs, submodule=_Sub(), nvtx=False, ctx=ctx,
    )

    assert outcome.ok
    assert eng._runner.ctx.chunk_tokens_after == {1: 6}
