"""A packed capture whose rows are all one token long (bs == num_tokens) still
plans prefill attention: its forward reads per-row token offsets (qo_indptr),
which the decode wrapper does not have."""
import types

import pytest
import torch

from mstar.engine.resources import BucketKey, SlotLease
from mstar.engine.resources.attn import flashinfer


class _Decode:
    def __init__(self, **kwargs):
        pass


class _Prefill:
    def __init__(self, **kwargs):
        pass


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(flashinfer, "FlashInferDecodeWrapper", _Decode)
    monkeypatch.setattr(flashinfer, "FlashInferPrefillWrapper", _Prefill)
    kv_config = types.SimpleNamespace(
        head_dim=8, num_qo_heads=1, num_kv_heads=1, page_size=16, max_num_pages=4,
    )
    manager = flashinfer.FlashInferManager("kv", torch.device("cpu"), torch.float32, kv_config)
    manager._workspaces = types.SimpleNamespace(get=lambda label, slot: None)
    return manager


def _lease(walk: str, bs: int, num_tokens: int) -> SlotLease:
    return SlotLease(slot=0, bucket=BucketKey(
        graph_walk=walk, cg_key_info=None, bs=bs, num_tokens=num_tokens,
    ))


def test_a_one_token_per_row_bucket_plans_decode(manager):
    wrapper = manager._cg_wrapper(_lease("decode", 4, 4), "main", 4, force_prefill=False)
    assert isinstance(wrapper, _Decode)


def test_a_packed_bucket_of_one_token_rows_plans_prefill(manager):
    wrapper = manager._cg_wrapper(_lease("prefill", 4, 4), "main", 4, force_prefill=True)
    assert isinstance(wrapper, _Prefill)
