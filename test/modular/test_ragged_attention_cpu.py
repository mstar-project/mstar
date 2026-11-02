"""Ragged attention checks that need no GPU: head-dimension padding and the
custom op's manager registry."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.resources import AttentionStep, RaggedAttentionConfig, Segment, StepContext
from mstar.engine.resources.attn.ragged.dense import DenseRaggedManager
from mstar.engine.resources.attn.ragged.dense_wrappers import DenseRaggedPrefillWrapper
from mstar.engine.resources.attn.ragged.wrappers import (
    SUPPORTED_HEAD_DIMS,
    FlashInferRaggedPrefillWrapper,
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
            FlashInferRaggedPrefillWrapper, "BatchPrefillWithRaggedKVCacheWrapper",
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


@pytest.mark.parametrize("tensor_cores", [True, False])
def test_mixed_wrapper_builds_both_halves_in_graph_mode(monkeypatch, tensor_cores):
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
        device=torch.device("cpu"), use_tensor_cores=tensor_cores,
    )

    assert mixed._decode.accelerator_graph and mixed._prefill.accelerator_graph
    assert mixed._decode.batch_size == 8
    # a sink row ahead of the prompt rows and one behind them
    assert mixed._prefill.batch_size == 4
    # the tensor-core decode half is the paged prefill kernel (FA2)
    built = prefill.call_args_list + decode.call_args_list
    assert len(built) == 2 and all(c.kwargs["use_cuda_graph"] for c in built)
    assert decode.call_count == (0 if tensor_cores else 1)
    if tensor_cores:
        assert prefill.call_args_list[0].kwargs["backend"] == "fa2"


def test_a_dropped_manager_leaves_the_op_registry():
    """The custom op finds a manager by key; the registry must not keep one
    alive past the engine that built it, nor hand one engine's key to
    another's manager over the same resource key."""
    import gc

    from mstar.engine.resources import RaggedAttentionConfig, RaggedAttentionSpec
    from mstar.engine.resources.attn.ragged.base import _MANAGERS
    from mstar.engine.resources.base import EngineResourceInfo, build_resource

    spec = RaggedAttentionSpec(
        resource_key="dropped", nodes={"encoder"},
        config=RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=64),
    )
    info = EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.bfloat16)
    mgr, other = build_resource(spec, info), build_resource(spec, info)
    assert _MANAGERS[mgr._key] is mgr and _MANAGERS[other._key] is other
    key = mgr._key
    del mgr
    gc.collect()
    assert key not in _MANAGERS


def test_a_ragged_wrapper_does_not_replan_an_unchanged_layout(monkeypatch):
    """A denoise loop declares the same segments every step: the wrapper keeps
    its last schedule, so only a changed layout, causal flag or another slot's
    wrapper plans the kernel again."""
    planned = []
    real_plan = DenseRaggedPrefillWrapper._plan_kernel
    monkeypatch.setattr(
        DenseRaggedPrefillWrapper, "_plan_kernel", lambda self, *a: planned.append(self) or real_plan(self, *a),
    )
    manager = DenseRaggedManager(
        torch.device("cpu"), torch.bfloat16, RaggedAttentionConfig(num_qo_heads=2, num_kv_heads=2, head_dim=64), "x",
    )

    def plan(*spans, causal=False, slot=0):
        segments = tuple(Segment(request_id=f"r{i}", label="main", span=n) for i, n in enumerate(spans))
        ctx = StepContext(request_ids=[s.request_id for s in segments], graph_walk="w", slot=slot, capture=False)
        manager.plan(AttentionStep(segments=segments, causal=causal), ctx)

    plan(64, 32)
    plan(64, 32)
    assert len(planned) == 1 and manager.num_segments() == 2
    plan(64, 32, slot=1)
    assert len(planned) == 2 and planned[1] is not planned[0]
    plan(64, 32, causal=True)
    plan(64, 33)
    assert len(planned) == 4
