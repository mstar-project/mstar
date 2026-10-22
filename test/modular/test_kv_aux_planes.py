"""``PagedKVConfig.aux_planes``: per-token planes on the KV cache's pages."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import AuxPlanes, KVLayout, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager

PAGE_SIZE = 4
WIDTH = 3


class _StubTransferManager:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        del request_id


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _config(**kwargs) -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=6, max_seq_len=8 * PAGE_SIZE, max_num_pages=8,
        page_size=PAGE_SIZE, layout=KVLayout.MLA, kv_lora_rank=4, qk_rope_head_dim=2,
        aux_planes={"idx": AuxPlanes(num_layers=2, width=WIDTH)}, **kwargs,
    )


def _manager(needs_remote_transfer: bool = False, **kwargs) -> KVManager:
    return KVManager(
        cfg=_config(**kwargs), name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
        needs_remote_transfer=needs_remote_transfer,
    )


def _ctx() -> StepContext:
    return StepContext(request_ids=("r",), graph_walk="decode", slot=0, capture=False)


def _step(mgr: KVManager, span: int):
    step = KVStep(segments=(Segment("r", "main", span),))
    assert mgr.admit(step, _ctx()).ok
    out = mgr.plan(step, _ctx())
    mgr.commit(step, _ctx())
    return out["main"].views[0]


def test_aux_planes_sit_on_the_cache_pages():
    cache = KVCache(_config(), torch.device("cpu"), torch.float32)
    assert cache.aux["idx"].shape == (2, 8, PAGE_SIZE, WIDTH)
    assert cache.aux_view("idx", 1).shape == (8, PAGE_SIZE, WIDTH)
    # the transfer engines register the KV tensor alone
    assert cache.nbytes == cache.tensor.nbytes


def test_write_aux_lands_at_the_steps_kv_slots():
    mgr = _manager()
    mgr.ingest_request("r")
    view = _step(mgr, 6)  # two pages
    rows = torch.arange(6 * WIDTH, dtype=torch.float32).view(6, WIDTH)
    mgr.write_kv(torch.ones(6, 6), None, layer_idx=1)
    mgr.write_aux("idx", rows, layer_idx=1)
    pages, offsets = mgr.write_slots()
    assert torch.equal(mgr.aux_view("idx", 1)[pages, offsets], rows)
    assert torch.equal(mgr.layer_view(1)[pages, offsets], torch.ones(6, 6))
    # token t of the request is at its page t // PAGE_SIZE, offset t % PAGE_SIZE
    for t in range(6):
        page = view.page_idxs[t // PAGE_SIZE]
        assert torch.equal(mgr.aux_view("idx", 1)[page, t % PAGE_SIZE], rows[t])
    assert not mgr.aux_view("idx", 0).any()


def test_page_copies_carry_the_aux_planes():
    cache = KVCache(_config(), torch.device("cpu"), torch.float32)
    cache.aux["idx"].normal_()
    cache.tensor.normal_()
    want = cache.aux["idx"][:, 2].clone()
    cache.copy_pages([2], [5])
    assert torch.equal(cache.aux["idx"][:, 5], want)
    cache.copy_slots(3, 6, end=2)
    assert torch.equal(cache.aux["idx"][:, 6, :2], cache.aux["idx"][:, 3, :2])
    assert not torch.equal(cache.aux["idx"][:, 6, 2:], cache.aux["idx"][:, 3, 2:])
    # an MLA page is [token, latent]: whole latents of the first tokens, not their first columns
    assert torch.equal(cache.tensor[:, 6, :2], cache.tensor[:, 3, :2])
    assert not torch.equal(cache.tensor[:, 6, 2:], cache.tensor[:, 3, 2:])


@pytest.mark.parametrize("kwargs", [{"cpu_offload_pages": 4}, {"needs_remote_transfer": True}])
def test_offload_and_transfer_refuse_aux_planes(kwargs):
    with pytest.raises(ValueError, match="aux_planes"):
        _manager(**kwargs)
