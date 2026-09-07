"""Capture-time padding rows don't keep their pages after warmup.

A capture hands its padding rows real spans — a packed bucket splits its whole
token budget across them — so they take pages proportional to the largest
bucket. A replay pads with zero-length rows (`pad_inputs` asks the config for
0 tokens), so none of that storage is ever read again; left resident it is
just cache the traffic can't have.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    PackedCudaGraphConfig,
)
from mstar.engine.cuda_graph_runner import DummyRowPool
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs


class _StubTransferManager:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


class _RecordingResource:
    def __init__(self):
        self.reset_calls: list[tuple[str, bool]] = []

    def reset_request(self, rid, free=False):
        self.reset_calls.append((rid, free))


def _pool(resource, rids_per_key):
    pool = DummyRowPool(
        prefix="node",
        step_runner=SimpleNamespace(ingest_request=lambda rid: None),
        resources={"kv": resource},
    )
    for key, bs in rids_per_key.items():
        pool.ensure(key, bs)
    return pool


def test_release_all_frees_every_row_it_opened():
    resource = _RecordingResource()
    pool = _pool(resource, {"0_slot0": 2, "1_slot0": 3})

    pool.release_all()

    assert all(free for _rid, free in resource.reset_calls)
    assert len(resource.reset_calls) == 5


def test_release_all_is_a_no_op_on_a_pool_that_opened_nothing():
    resource = _RecordingResource()
    pool = _pool(resource, {})

    pool.release_all()

    assert resource.reset_calls == []


# ── page accounting against a real KV manager ───────────────────────────


def _kv_manager(max_num_pages=64, page_size=16) -> KVManager:
    cfg = KVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1024,
        max_num_pages=max_num_pages, page_size=page_size,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=None, device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _admit(kv: KVManager, rids, spans):
    step = KVStep(segments=tuple(
        Segment(request_id=rid, label="main", span=span)
        for rid, span in zip(rids, spans, strict=True)
    ))
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=True,
    )
    assert kv.admit(step, ctx).ok


def test_capture_spans_come_back_to_the_arena():
    """The whole point: a packed capture's padding rows take pages by the
    bucket's token budget, and warmup hands them back."""
    kv = _kv_manager()
    free_before = kv._arena.num_free
    rids = ["__cg_a_0__", "__cg_a_1__"]
    for rid in rids:
        kv.ingest_request(rid)
    pool = DummyRowPool(
        prefix="a",
        step_runner=SimpleNamespace(ingest_request=lambda rid: None),
        resources={"kv": kv},
    )
    pool._held["0_slot0"] = list(rids)

    _admit(kv, rids, [512, 256])
    assert kv._arena.num_free < free_before, "capture must actually take pages"

    pool.release_all()

    assert kv._arena.num_free == free_before


def test_a_zero_span_padding_row_needs_no_pages_at_replay():
    """Why the pages are safe to drop: `pad_inputs` asks a packed config for
    0 tokens, so a padding row's segment reserves nothing."""
    kv = _kv_manager()
    kv.ingest_request("__cg_a_0__")
    free = kv._arena.num_free

    _admit(kv, ["__cg_a_0__"], [0])

    assert kv._arena.num_free == free


def test_a_batched_padding_row_still_declares_its_token():
    """The other half: a decode-shaped config ignores the token argument, so
    its padding rows re-take one page each on first replay rather than none."""
    device = torch.device("cpu")
    batched = BatchedCudaGraphConfig(
        capture_graph_walk="decode",
        single_request_inputs=ARNodeInputs(
            input_ids=torch.zeros(1, dtype=torch.long, device=device),
            input_seq_len=1,
        ),
    )
    packed = PackedCudaGraphConfig(
        capture_graph_walk="prefill",
        capture_token_lengths=[64],
        make_node_input=lambda n: ARNodeInputs(input_seq_len=n),
    )

    assert [i.input_seq_len for i in batched.get_node_inputs(2, 0)] == [1, 1]
    assert [i.input_seq_len for i in packed.get_node_inputs(2, 0)] == [0, 0]


@pytest.mark.parametrize("page_size", [16, 128])
def test_freed_pages_are_reusable_by_a_real_request(page_size):
    kv = _kv_manager(max_num_pages=8, page_size=page_size)
    kv.ingest_request("__cg_a_0__")
    _admit(kv, ["__cg_a_0__"], [page_size * 6])
    pool = DummyRowPool(
        prefix="a",
        step_runner=SimpleNamespace(ingest_request=lambda rid: None),
        resources={"kv": kv},
    )
    pool._held["0_slot0"] = ["__cg_a_0__"]

    pool.release_all()

    # the arena had 8 pages, 1 reserved as SINK; a request needing all the
    # rest only fits because the capture handed its pages back
    kv.ingest_request("real")
    _admit(kv, ["real"], [page_size * 7])
