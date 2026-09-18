"""Pre-forks under KV pre-planning.

Pre-planning builds the next step's addressing on the plan thread while the
current step is still on the GPU. What has to hold: a promoted plan must be the
plan the step would have made inline, and an abandoned one must leave no trace.

Pre-forks are the sharp edge on both counts. `_apply_fork` copies pages, which
the in-flight step's kernels are still writing, so staging must not issue it —
but a fork target takes its source's length and a fresh generation, and the
step may address that target in the same step (Bagel's `cfg`), so the staged
addressing has to come out as though the copy had already run.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KVManager allocates on device"
)

PAGE_SIZE = 8
RESIDENT = 2 * PAGE_SIZE


class _StubTransferManager:
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
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _ctx(*rids: str, is_preplan: bool = False) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="walk", slot=0, capture=False,
        is_preplan=is_preplan,
    )


def _make_manager() -> KVManager:
    cfg = KVConfig(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=16 * PAGE_SIZE,
        max_num_pages=16,
        page_size=PAGE_SIZE,
        cpu_offload_pages=0,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda"), dtype=torch.float32,
    )


def _forking_step(rid: str, *, address_target: bool) -> KVStep:
    """A pre-fork of ``main`` onto ``cfg``.

    With ``address_target`` the step also runs on ``cfg`` — Bagel's shape, and
    the case where the fork changes what the step's own addressing says.
    """
    labels = ("main", "cfg") if address_target else ("main",)
    return KVStep(
        segments=tuple(Segment(rid, label, 1) for label in labels),
        pre_forks=(("main", "cfg"),),
    )


def _prepared(address_target: bool) -> tuple[KVManager, KVStep]:
    """A manager holding `RESIDENT` committed tokens on ``main``, with the
    forking step admitted (so its target is reserved) but not yet planned."""
    mgr = _make_manager()
    mgr.ingest_request("r0")
    grow = KVStep(segments=(Segment("r0", "main", RESIDENT),))
    assert mgr.admit(grow, _ctx("r0")).ok
    mgr.plan(grow, _ctx("r0"))
    mgr.commit(grow, _ctx("r0"))
    # distinct per page, so the fork's copy can be checked exactly
    for i, page in enumerate(mgr._streams["r0"]["main"].page_indices):
        mgr.kv_cache.tensor[:, page] = float(i + 1)

    step = _forking_step("r0", address_target=address_target)
    assert mgr.admit(step, _ctx("r0")).ok
    return mgr, step


def _snapshot(mgr: KVManager, label: str) -> tuple[int, int, list[int]]:
    stream = mgr._streams["r0"][label]
    return (stream.stored_len, stream.generation, list(stream.page_indices))


@requires_cuda
@pytest.mark.parametrize("address_target", [False, True])
def test_staging_leaves_the_fork_alone(address_target: bool):
    """The copy writes pages the in-flight step is still reading, so staging
    must not issue it — nor move the length and generation that say it ran. An
    abandoned stage then has nothing to rewind."""
    mgr, step = _prepared(address_target)
    was = _snapshot(mgr, "cfg")
    pages = mgr._streams["r0"]["cfg"].page_indices
    mgr.kv_cache.tensor[:, pages] = -99.0

    mgr.plan(step, _ctx("r0", is_preplan=True))
    assert _snapshot(mgr, "cfg") == was, "staging applied the fork"
    assert (mgr.kv_cache.tensor[:, pages] == -99.0).all(), "staging copied pages"

    mgr.clear_preplan()
    assert _snapshot(mgr, "cfg") == was
    assert not mgr._preplanned


@requires_cuda
def test_promotion_applies_the_fork_once():
    """Deferred, not dropped: the copy lands when the real step plans."""
    mgr, step = _prepared(address_target=True)
    generation = mgr._streams["r0"]["cfg"].generation

    mgr.plan(step, _ctx("r0", is_preplan=True))
    mgr.plan(step, _ctx("r0"))

    cfg = mgr._streams["r0"]["cfg"]
    assert cfg.stored_len == RESIDENT
    assert cfg.generation == generation + 1, "the fork should apply exactly once"
    n = RESIDENT // PAGE_SIZE
    torch.testing.assert_close(
        mgr.kv_cache.tensor[:, cfg.page_indices[:n]],
        mgr.kv_cache.tensor[:, mgr._streams["r0"]["main"].page_indices[:n]],
    )


@requires_cuda
def test_staged_addressing_accounts_for_the_pending_fork():
    """A fork target inherits its source's length, and the step may run on that
    target in the same step. Staging defers the copy but must still address the
    stream the way the inline plan does, or attention reads one token of
    context where the step has `RESIDENT` + 1."""
    inline, inline_step = _prepared(address_target=True)
    staged, staged_step = _prepared(address_target=True)

    want = inline.plan(inline_step, _ctx("r0"))
    got = staged.plan(staged_step, _ctx("r0", is_preplan=True))

    inline_cfg = next(v for v in want["cfg"].views if v.label == "cfg")
    assert inline_cfg.length == RESIDENT + 1, (
        "the inline plan must see the fork it just applied"
    )
    assert got["cfg"].views == want["cfg"].views
    for staged_t, inline_t in zip(got["cfg"].cpu_indptrs, want["cfg"].cpu_indptrs):
        torch.testing.assert_close(staged_t, inline_t)

    # and promotion keeps it that way
    promoted = staged.plan(staged_step, _ctx("r0"))
    assert promoted["cfg"].views == want["cfg"].views
