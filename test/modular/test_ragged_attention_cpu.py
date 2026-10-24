"""Ragged attention head-dimension padding checks that need no GPU."""

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
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
)


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


@pytest.mark.parametrize("accelerator_graph", [False, True])
@pytest.mark.parametrize("pinned_indices", [False, True])
def test_decode_plan_stages_pinned_indices_only_for_graphs(
    monkeypatch, accelerator_graph, pinned_indices,
):
    native = Mock(return_value=SimpleNamespace(plan=Mock()))
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=native),
    )
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: pinned_indices)
    wrapper = FlashInferDecodeWrapper(
        workspace_buffer=torch.empty(8, dtype=torch.uint8),
        num_qo_heads=2, num_kv_heads=2, head_dim=64, page_size=16,
        batch_size=2, max_num_pages=4, device=torch.device("cpu"),
        accelerator_graph=accelerator_graph,
    )
    indices = torch.tensor([3, 1], dtype=torch.int32)
    wrapper.plan(
        paged_kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.tensor([1, 1], dtype=torch.int32),
    )
    planned_indices = wrapper.attn_wrapper.plan.call_args.kwargs["indices"]
    torch.testing.assert_close(planned_indices, indices)
    if accelerator_graph and pinned_indices:
        assert planned_indices.data_ptr() == wrapper._paged_kv_indices_buf.data_ptr()
    else:
        assert planned_indices is indices


def test_mixed_wrapper_builds_both_halves_in_graph_mode(monkeypatch):
    """The mixed-walk wrapper is only ever built for a captured bucket, so a
    keyword its halves do not take fails every mixed capture."""
    decode = Mock(return_value=SimpleNamespace(plan=Mock()))
    prefill = Mock(return_value=SimpleNamespace(plan=Mock()))
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=decode,
        BatchPrefillWithPagedKVCacheWrapper=prefill,
    ))
    mixed = FlashInferMixedWrapper(
        decode_workspace=torch.empty(8, dtype=torch.uint8),
        prefill_workspace=torch.empty(8, dtype=torch.uint8),
        num_qo_heads=2, num_kv_heads=2, head_dim=64, page_size=16,
        batch_size=8, max_total_tokens=64, max_num_pages=32, max_prefill_rows=2,
        device=torch.device("cpu"),
    )

    assert mixed._decode.accelerator_graph and mixed._prefill.accelerator_graph
    assert mixed._decode.batch_size == 8
    assert mixed._prefill.batch_size == 3
    assert decode.call_args.kwargs["use_cuda_graph"] is True
    assert prefill.call_args.kwargs["use_cuda_graph"] is True
