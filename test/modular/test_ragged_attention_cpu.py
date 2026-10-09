"""Attention wrapper contracts and head-dimension checks that need no GPU."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.resources.attn.ragged.wrappers import (
    SUPPORTED_HEAD_DIMS,
    RaggedPrefillWrapper,
    padded_head_dim,
)
from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper, FlashInferPrefillWrapper


@pytest.mark.parametrize(
    ("head_dim", "expected"), [(64, 64), (72, 128), (128, 128), (129, 256), (256, 256)]
)
def test_padded_head_dim_rounds_up(head_dim, expected):
    assert padded_head_dim(head_dim) == expected


def test_padded_head_dim_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the largest supported"):
        padded_head_dim(SUPPORTED_HEAD_DIMS[-1] + 1)


@pytest.mark.parametrize(
    ("wrapper_type", "native_name", "options"),
    [
        (
            FlashInferPrefillWrapper, "BatchPrefillWithPagedKVCacheWrapper",
            dict(page_size=16, batch_size=2, max_total_tokens=16, max_num_pages=4),
        ),
        (
            FlashInferDecodeWrapper, "BatchDecodeWithPagedKVCacheWrapper",
            dict(page_size=16, batch_size=2, max_num_pages=4),
        ),
        (
            RaggedPrefillWrapper, "BatchPrefillWithRaggedKVCacheWrapper",
            dict(max_num_segments=2, max_total_tokens=16),
        ),
    ],
)
def test_accelerator_graph_flag_adapts_to_flashinfer_api(monkeypatch, wrapper_type, native_name, options):
    native = Mock(return_value=SimpleNamespace(plan=Mock()))
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(**{native_name: native}))
    wrapper = wrapper_type(
        workspace_buffer=torch.empty(8, dtype=torch.uint8),
        num_qo_heads=2, num_kv_heads=2, head_dim=64,
        device=torch.device("cpu"), accelerator_graph=True, **options,
    )

    assert wrapper.accelerator_graph
    assert native.call_args.kwargs["use_cuda_graph"] is True
    assert "accelerator_graph" not in native.call_args.kwargs
