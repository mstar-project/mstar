"""A node's precision policy: the load-time cast with fp32 islands, the YAML
tighten/refuse rules, and the engine reading one policy for load, resources and
the autocast scope (CPU, mock submodules)."""
import types

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.precision import (
    NodePrecision,
    apply_precision,
    apply_yaml_precision,
    precision_mismatches,
)
from mstar.engine.resources import KVSpec
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.submodule_base import NodeSubmodule
from mstar.worker.engine_manager import EngineManager


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("index", torch.arange(4))


class _Node(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.head = torch.nn.Linear(4, 2)

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# --- the policy itself ------------------------------------------------------------


def test_autocast_is_off_unless_asked_and_needs_a_param_dtype():
    assert NodePrecision(torch.bfloat16).autocast is False
    assert NodePrecision(None).autocast_dtype is None
    assert NodePrecision(torch.bfloat16, autocast=True).autocast_dtype is torch.bfloat16
    with pytest.raises(ValueError, match="needs a param_dtype"):
        NodePrecision(None, autocast=True)


def test_fp32_modules_and_their_children_stay_fp32():
    node = apply_precision(_Node(), NodePrecision(torch.bfloat16, fp32_modules=("blocks.*.norm", "head")))
    for block in node.blocks:
        assert block.proj.weight.dtype is torch.bfloat16
        assert block.scale.dtype is torch.bfloat16
        assert block.norm.weight.dtype is torch.float32
        # integer buffers are never cast
        assert block.index.dtype is torch.int64
    assert node.head.weight.dtype is torch.float32


def test_a_glob_matching_nothing_is_an_error():
    with pytest.raises(ValueError, match="'blocks.*.nrom' matches no module, param or buffer"):
        apply_precision(_Node(), NodePrecision(torch.bfloat16, fp32_modules=("blocks.*.nrom",)))


def test_params_and_buffers_match_by_name_and_keep_their_identity():
    node = _Node()
    weight = node.blocks[0].proj.weight
    weight.weight_loader = "kept"
    apply_precision(node, NodePrecision(torch.bfloat16, fp32_modules=("blocks.0.proj.bias", "*.scale")))
    assert node.blocks[0].proj.weight is weight and weight.weight_loader == "kept"
    assert weight.dtype is torch.bfloat16
    assert node.blocks[0].proj.bias.dtype is torch.float32
    assert node.blocks[1].proj.bias.dtype is torch.bfloat16
    assert all(block.scale.dtype is torch.float32 for block in node.blocks)


def test_none_keeps_the_build_dtype_but_still_pins_islands():
    node = _Node().to(torch.float16)
    apply_precision(node, NodePrecision(None, fp32_modules=("head",), autocast=False))
    assert node.blocks[0].proj.weight.dtype is torch.float16
    assert node.head.weight.dtype is torch.float32


def test_cast_on_meta_allocates_in_the_final_dtype():
    with torch.device("meta"):
        node = _Node()
    apply_precision(node, NodePrecision(torch.bfloat16, fp32_modules=("head",)))
    node = node.to_empty(device="cpu")
    assert node.blocks[0].proj.weight.dtype is torch.bfloat16
    assert node.head.weight.dtype is torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for peak memory")
def test_meta_cast_never_materialises_bf16_params_in_fp32():
    device = torch.device("cuda")
    with torch.device("meta"):
        node = torch.nn.Linear(4096, 4096, bias=False)
    apply_precision(node, NodePrecision(torch.bfloat16))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    node.to_empty(device=device)
    peak = torch.cuda.max_memory_allocated(device) - before
    assert peak == 4096 * 4096 * 2


# --- YAML -------------------------------------------------------------------------


def test_yaml_may_tighten():
    model = NodePrecision(torch.bfloat16, fp32_modules=("head",))
    tightened = apply_yaml_precision(
        model, "llm",
        {"param_dtype": "float32", "fp32_modules": ["blocks.*.norm", "head"], "autocast": False},
    )
    assert tightened == NodePrecision(
        torch.float32, fp32_modules=("head", "blocks.*.norm"), autocast=False,
    )


def test_yaml_may_not_turn_autocast_on_or_retype_a_pinned_node():
    with pytest.raises(ValueError, match="cannot turn it on"):
        apply_yaml_precision(NodePrecision(torch.bfloat16), "codec", {"autocast": True})
    assert apply_yaml_precision(
        NodePrecision(torch.bfloat16, autocast=True), "llm", {"autocast": False},
    ) == NodePrecision(torch.bfloat16)
    with pytest.raises(ValueError, match="cannot set param_dtype"):
        apply_yaml_precision(NodePrecision(None, autocast=False), "codec", {"param_dtype": "bfloat16"})
    with pytest.raises(ValueError, match="unknown key"):
        apply_yaml_precision(NodePrecision(torch.bfloat16), "llm", {"dtype": "bfloat16"})


# --- through the engine -------------------------------------------------------------


class _Model:
    """A model that declares each node's policy up front; an undeclared node
    gets ``Model``'s default."""

    def __init__(self, precisions, resources=()):
        self._precisions = precisions
        self._resources = list(resources)
        self.hints = {}

    def get_node_precision(self, node_name):
        return self._precisions.get(node_name, NodePrecision(torch.bfloat16))

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
        self.hints[node_name] = autocast_dtype
        return _Node()

    def get_node_resources(self):
        return self._resources

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


def _build(model, nodes, model_config=None) -> EngineManager:
    return EngineManager.build(
        node_names=set(nodes),
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


def test_engine_reads_one_policy_for_load_resources_and_scope():
    kv = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=16, max_num_pages=2, page_size=8)
    model = _Model(
        {
            "llm": NodePrecision(torch.bfloat16, fp32_modules=("head",)),
            "encoder": NodePrecision(torch.float16, autocast=True),
            "codec": NodePrecision(None),
        },
        resources=[KVSpec(resource_key="kv", nodes={"llm"}, config=kv)],
    )
    manager = _build(model, {"llm", "encoder", "codec", "undeclared"})
    try:
        engine = manager.engine
        # the build hint is each node's own param dtype
        assert model.hints == {
            "llm": torch.bfloat16, "encoder": torch.float16, "codec": None, "undeclared": torch.bfloat16,
        }
        llm = engine._submodules["llm"]
        assert engine._autocast_dtype_for("llm") is None
        assert llm.submodule.blocks[0].proj.weight.dtype is torch.bfloat16
        assert llm.submodule.head.weight.dtype is torch.float32
        # the resource inherits the param dtype even with autocast off
        assert engine._resources["kv"].kv_cache.tensor.dtype is torch.bfloat16
        assert engine._autocast_dtype_for("encoder") is torch.float16
        assert engine._submodules["encoder"].submodule.head.weight.dtype is torch.float16
        # None keeps the build dtype (float32 here) and runs without autocast
        assert engine._autocast_dtype_for("codec") is None
        assert engine._submodules["codec"].submodule.head.weight.dtype is torch.float32
        assert engine._autocast_dtype_for("undeclared") is None
    finally:
        manager.shutdown()


def test_model_wide_autocast_dtype_yaml_is_refused_with_its_replacement():
    with pytest.raises(ValueError, match=r"(?s)no longer read.*nodes:"):
        _build(_Model({}), {"llm"}, {"autocast_dtype": "bfloat16"})


def test_yaml_node_block_reaches_the_load_hint_and_the_policy():
    model = _Model({"llm": NodePrecision(torch.bfloat16)})
    config = {"nodes": {"llm": {"precision": {"param_dtype": "float32", "autocast": False}}}}
    manager = _build(model, {"llm"}, config)
    try:
        assert model.hints["llm"] is torch.float32
        assert manager.engine._submodules["llm"].precision == NodePrecision(torch.float32, autocast=False)
    finally:
        manager.shutdown()


def test_yaml_naming_an_unplaced_node_is_an_error():
    with pytest.raises(ValueError, match=r"unknown node\(s\) \['lmm'\]"):
        _build(_Model({}), {"llm"}, {"nodes": {"lmm": {"precision": {"autocast": False}}}})


def test_nodes_sharing_a_module_must_agree():
    shared = torch.nn.Linear(4, 4)

    class _Sharing(_Model):
        def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
            node = _Node()
            node.shared = shared
            return node

    agreeing = _Sharing({n: NodePrecision(torch.bfloat16, autocast=False) for n in ("a", "b")})
    _build(agreeing, {"a", "b"}).shutdown()
    clashing = _Sharing({
        "a": NodePrecision(torch.bfloat16, autocast=False),
        "b": NodePrecision(torch.bfloat16, fp32_modules=("shared",), autocast=False),
    })
    with pytest.raises(ValueError, match=r"node 'a' .*shared\.weight"):
        _build(clashing, {"a", "b"})


def test_a_quantized_param_dtype_is_refused_and_quantized_tensors_survive_the_cast():
    # fp8 weights are a quant config's job, not a cast's
    with pytest.raises(ValueError, match="param_dtype is torch.float8_e4m3fn, a quantized dtype"):
        NodePrecision(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="quantized dtype"):
        apply_yaml_precision(NodePrecision(torch.bfloat16), "llm", {"param_dtype": "float8_e4m3fn"})
    # a tensor a quant pass (or a pre-quantized checkpoint) left in fp8 keeps its dtype
    node = _Node()
    node.blocks[0].proj.weight.data = node.blocks[0].proj.weight.data.to(torch.float8_e4m3fn)
    apply_precision(node, NodePrecision(torch.bfloat16))
    assert node.blocks[0].proj.weight.dtype is torch.float8_e4m3fn
    assert node.blocks[1].proj.weight.dtype is torch.bfloat16
    assert precision_mismatches(node, NodePrecision(torch.bfloat16)) == []
