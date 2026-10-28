"""Ragged attention checks that need no GPU: head-dimension padding, head
sharding and the custom op's manager registry."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

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


def _load_heads(kind, splits_heads, tp=1, sp=2):
    """Load one resource of ``kind`` (4 heads) for nodes declaring
    ``splits_heads`` (node -> flag) on a tp x sp lockstep instance, and
    return the heads one rank holds."""
    from mstar.engine.engine import Engine
    from mstar.engine.resources import RaggedAttentionConfig, RaggedAttentionSpec
    from mstar.engine.resources.recurrent.config import (
        RecurrentBlockConfig,
        RecurrentStateConfig,
        RecurrentStateSpec,
    )
    from mstar.model.submodule_base import NodeSubmodule

    class _Node(NodeSubmodule):
        def __init__(self, splits):
            super().__init__()
            self.splits_heads = splits
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def prepare_inputs(self, *args, **kwargs):
            raise NotImplementedError

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class _Rank:
        rank = 0

        def __init__(self, world_size):
            self.world_size = world_size

        def barrier(self):
            pass

    class _Groups:
        def get_joint_group_for_node(self, node):
            return SimpleNamespace(world_size=tp * sp, rank=0, tp_group=_Rank(tp), sp_group=_Rank(sp))

        def get_instance_world_size_for_node(self, node):
            return tp * sp

        def get_instance_rank_for_node(self, node):
            return 0

        def resource_needs_remote_transfer(self, nodes, local_nodes):
            return False

    nodes = set(splits_heads)
    if kind == "ragged":
        spec = RaggedAttentionSpec(
            resource_key="res", nodes=nodes,
            config=RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=72, dtype=torch.bfloat16),
        )
    else:
        spec = RecurrentStateSpec("res", nodes, RecurrentStateConfig(
            num_layers=1, max_slots=2,
            blocks={"state": RecurrentBlockConfig(shape=(4, 8), dtype=torch.float32, shard_dims=(0,))},
        ))
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        submodules={node: _Node(splits) for node, splits in splits_heads.items()}, specs=[spec],
        parallel_groups=_Groups(), device=torch.device("cpu"), transfer_engine_info=None,
    )
    if kind == "ragged":
        return engine._resources["res"]._kwargs["num_qo_heads"]
    return spec.config.blocks["state"].shape[0]


@pytest.mark.parametrize("tp,sp,splits_heads,heads", [
    (1, 2, True, 2), (1, 2, False, 4), (2, 1, False, 2), (2, 2, False, 2),
])
@pytest.mark.parametrize("kind", ["ragged", "recurrent"])
def test_a_node_whose_ranks_split_other_work_keeps_every_head(kind, tp, sp, splits_heads, heads):
    """Every resource the engine builds for a multi-rank node holds one rank's
    slice of the heads, whatever the resource kind. A node whose SP ranks
    split something else (``splits_heads = False``, e.g. guidance branches)
    keeps every head across them; its TP ranks still split them."""
    assert _load_heads(kind, {"dit": splits_heads}, tp=tp, sp=sp) == heads


def test_nodes_sharing_a_resource_must_agree_on_splitting_heads():
    """The resource is built once for every node it serves; nodes that would
    size it differently are refused at load rather than one picked by set
    order, which could differ across ranks."""
    with pytest.raises(ValueError, match="split its heads differently"):
        _load_heads("ragged", {"a": True, "b": False})


@pytest.mark.parametrize(("axis", "heads"), [("tp", 2), ("sp", 2), ("cfg", 4)])
def test_resource_info_built_without_head_shards_shards_over_its_group(axis, heads):
    """Built outside the engine (a reference test, the checkpoint validator),
    an ``EngineResourceInfo`` over a two-rank TP or SP group gives one rank's
    heads; a CFG group's ranks each run whole branches, every head."""
    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.engine.resources import RaggedAttentionConfig, RaggedAttentionSpec
    from mstar.engine.resources.base import EngineResourceInfo, build_resource

    pair = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
    axes = {f"{a}_group": pair if a == axis else CommGroup.trivial() for a in ("tp", "sp", "cfg")}
    spec = RaggedAttentionSpec(
        resource_key="by_hand", nodes={"encoder"},
        config=RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=64),
    )
    mgr = build_resource(spec, EngineResourceInfo(
        device=torch.device("cpu"), kv_dtype=torch.bfloat16, joint_comm_group=JointGroups(instance=pair, **axes),
    ))
    assert mgr._kwargs["num_qo_heads"] == heads


def test_a_dropped_manager_leaves_the_op_registry():
    """The custom op finds a manager by key; the registry must not keep one
    alive past the engine that built it."""
    import gc

    from mstar.engine.resources import RaggedAttentionConfig, RaggedAttentionSpec
    from mstar.engine.resources.attn.ragged.base import _MANAGERS
    from mstar.engine.resources.base import EngineResourceInfo, build_resource

    spec = RaggedAttentionSpec(
        resource_key="dropped", nodes={"encoder"},
        config=RaggedAttentionConfig(num_qo_heads=4, num_kv_heads=4, head_dim=64),
    )
    mgr = build_resource(spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.bfloat16))
    assert _MANAGERS["dropped"] is mgr
    del mgr
    gc.collect()
    assert "dropped" not in _MANAGERS
