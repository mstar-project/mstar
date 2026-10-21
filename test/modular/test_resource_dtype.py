"""A resource's storage dtype is its spec's, else its nodes' resolved dtype,
never the engine-wide one (CPU, mock submodules, built through EngineManager)."""
import types

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.resources import AttentionConfig, AttentionSpec, KVSpec
from mstar.engine.resources.attn.config import AttnBackend
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.submodule_base import NodeSubmodule
from mstar.worker.engine_manager import EngineManager


class _Node(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Fp16(_Node):
    def get_autocast_dtype(self):
        return torch.float16


class _OptedOut(_Node):
    disable_autocast = True


_NODES = {"inherit": _Node, "fp16": _Fp16, "opted_out": _OptedOut}


class _Model:
    """What ``EngineManager.build`` reads off a model: one KV cache and its
    attention over ``nodes``."""

    def __init__(self, dtype, nodes, kv_dtype=None):
        self._dtype = dtype
        self._nodes = set(nodes)
        self._kv_dtype = kv_dtype

    def get_autocast_dtype(self):
        return self._dtype

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
        return _NODES[node_name]()

    def get_node_resources(self):
        kv = PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=16,
            max_num_pages=2, page_size=8, dtype=self._kv_dtype,
        )
        return [
            KVSpec(resource_key="kv", nodes=self._nodes, config=kv),
            AttentionSpec(
                resource_key="attn", nodes=self._nodes,
                config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER),
            ),
        ]

    def prefix_key_streams(self):
        return {}

    def get_graph_walk_graphs(self):
        return {}

    def combined_walk_of(self):
        return {}

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return ""


def _build(model, model_config=None) -> EngineManager:
    return EngineManager.build(
        node_names=model._nodes,
        device=torch.device("cpu"),
        model_config=model_config or {},
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="test", my_session_id="test",
            transfer_engine=LocalTransferEngine("test"),
        ),
        model=model,
        graph_runtime=types.SimpleNamespace(),
    )


def _dtypes(model, model_config=None) -> tuple[torch.dtype, torch.dtype]:
    """The KV pages' dtype and the dtype attention is planned in."""
    manager = _build(model, model_config)
    try:
        resources = manager.engine._resources
        return resources["kv"].kv_cache.tensor.dtype, resources["attn"]._dtype
    finally:
        manager.shutdown()


def test_fp32_node_with_bf16_kv_spec_allocates_bf16_pages():
    # attention follows the pages, not the node's fp32
    assert _dtypes(_Model(torch.float32, {"inherit"}, kv_dtype=torch.bfloat16)) == (
        torch.bfloat16, torch.bfloat16,
    )


def test_node_dtype_not_engine_dtype_is_the_fallback():
    assert _dtypes(_Model(torch.bfloat16, {"fp16"})) == (torch.float16, torch.float16)


def test_yaml_sets_the_kv_dtype_per_resource_key():
    config = {"resources": {"kv": {"dtype": "bfloat16"}}}
    assert _dtypes(_Model(torch.float32, {"inherit"}), config) == (
        torch.bfloat16, torch.bfloat16,
    )


def test_nodes_without_a_dtype_must_declare_one():
    with pytest.raises(ValueError, match=r"'kv' declares no dtype .* without a dtype"):
        _build(_Model(torch.bfloat16, {"opted_out"}))
    # the model-wide None is no fallback either
    with pytest.raises(ValueError, match=r"'kv' declares no dtype"):
        _build(_Model(None, {"inherit"}))
    assert _dtypes(_Model(None, {"inherit"}, kv_dtype=torch.bfloat16)) == (
        torch.bfloat16, torch.bfloat16,
    )


def test_shared_resource_with_mismatched_node_dtypes_errors():
    with pytest.raises(ValueError, match=r"'kv' declares no dtype .* different dtypes"):
        _build(_Model(torch.bfloat16, {"inherit", "fp16"}))
    assert _dtypes(_Model(torch.bfloat16, {"inherit", "fp16"}, kv_dtype=torch.float16)) == (
        torch.float16, torch.float16,
    )


def test_yaml_dtype_must_name_a_torch_dtype():
    with pytest.raises(ValueError, match="not a torch dtype"):
        _build(_Model(torch.bfloat16, {"inherit"}), {"resources": {"kv": {"dtype": "bf16"}}})
