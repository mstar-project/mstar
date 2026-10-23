"""A node's input specs: resolved against its precision at load, and applied by
the engine to each request's routed tensors before ``prepare_inputs`` (CPU,
mock submodules)."""
import types
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.input_spec import ResolvedSpec, TensorSpec, apply_input_specs, resolve_input_specs
from mstar.engine.precision import NodePrecision
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.submodule_base import NodeSubmodule
from mstar.worker.engine_manager import EngineManager

BF16 = NodePrecision(torch.bfloat16)


def _resolve(specs, precision=BF16):
    return resolve_input_specs("talker", specs, precision)


# --- the spec itself ----------------------------------------------------------------


def test_param_follows_the_node_precision():
    specs = {"hidden": TensorSpec("param")}
    assert _resolve(specs) == {"hidden": ResolvedSpec(torch.bfloat16, "cast")}
    assert _resolve(specs, NodePrecision(torch.float16)) == {"hidden": ResolvedSpec(torch.float16, "cast")}


def test_param_on_a_node_without_a_param_dtype_is_a_load_error():
    with pytest.raises(ValueError, match="'talker'.*'hidden'.*declare the dtype"):
        _resolve({"hidden": TensorSpec("param")}, NodePrecision(None))
    # a concrete dtype needs no param dtype
    assert _resolve({"hidden": TensorSpec(torch.float32)}, NodePrecision(None))["hidden"].dtype is torch.float32


def test_integer_specs_check_and_dtype_none_checks_nothing():
    resolved = _resolve({"ids": TensorSpec(torch.long), "any": TensorSpec(None)})
    assert resolved == {"ids": ResolvedSpec(torch.long, "raise")}
    with pytest.raises(ValueError, match="checked, not cast"):
        TensorSpec(torch.long, on_mismatch="cast")


def test_bad_specs_are_refused():
    with pytest.raises(ValueError, match="quantized dtype"):
        TensorSpec(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="'param' or None"):
        TensorSpec("bfloat16")
    with pytest.raises(ValueError, match="on_mismatch"):
        TensorSpec("param", on_mismatch="ignore")


# --- applying it --------------------------------------------------------------------


def test_a_floating_mismatch_is_cast_and_the_batch_dict_is_left_alone():
    fp32 = torch.randn(2, 4)
    inputs = {"hidden": [fp32], "other": [torch.ones(1)]}
    out = apply_input_specs("talker", inputs, _resolve({"hidden": TensorSpec("param")}))
    assert out["hidden"][0].dtype is torch.bfloat16
    assert torch.equal(out["hidden"][0], fp32.to(torch.bfloat16))
    assert out["other"] is inputs["other"]
    assert inputs["hidden"][0] is fp32


def test_matching_or_absent_inputs_return_the_same_dict():
    specs = _resolve({"hidden": TensorSpec("param"), "ids": TensorSpec(torch.long)})
    inputs = {"hidden": [torch.zeros(2, dtype=torch.bfloat16)], "ids": []}
    assert apply_input_specs("talker", inputs, specs) is inputs
    assert apply_input_specs("talker", {}, specs) == {}


def test_raise_names_the_node_and_the_signal_and_long_is_never_cast():
    specs = _resolve({"ids": TensorSpec(torch.long)})
    with pytest.raises(TypeError, match=r"node 'talker': input 'ids' is torch.int32, its spec wants torch.int64"):
        apply_input_specs("talker", {"ids": [torch.zeros(3, dtype=torch.int32)]}, specs)
    # nor is an integer tensor cast into a floating spec
    with pytest.raises(TypeError, match="'hidden'"):
        apply_input_specs(
            "talker", {"hidden": [torch.zeros(3, dtype=torch.long)]}, _resolve({"hidden": TensorSpec("param")}),
        )
    with pytest.raises(TypeError, match="'hidden'"):
        apply_input_specs(
            "talker", {"hidden": [torch.zeros(3)]}, _resolve({"hidden": TensorSpec("param", on_mismatch="raise")}),
        )


def test_warn_passes_the_tensor_through_and_warns_once(caplog):
    specs = _resolve({"mel": TensorSpec(torch.float32, on_mismatch="warn")})
    mel = torch.zeros(3, dtype=torch.float16)
    with caplog.at_level("WARNING", logger="mstar.engine.input_spec"):
        for _ in range(3):
            out = apply_input_specs("warn_node", {"mel": [mel]}, specs)
            assert out["mel"][0] is mel
    assert len([r for r in caplog.records if "'mel'" in r.getMessage()]) == 1


# --- through the engine -------------------------------------------------------------


class _Batch:
    def __init__(self, tensors):
        self.node_name = "talker"
        self.request_ids = list(tensors)
        self.step_context = SimpleNamespace(graph_walk="decode", request_walks={}, walk_of=lambda _r: "decode")
        self.per_request_info_wrapped = {r: SimpleNamespace() for r in tensors}
        self.per_request_input_tensors = tensors
        self.per_request_input_metadata = {}
        self.final_stream_rids = set()
        self.skipped_rids = set()
        self.failed_requests = {}
        self.inputs = None

    def register_prepare_batch(self, inputs):
        self.inputs = inputs

    def drop_rids(self, rids):
        self.request_ids = [r for r in self.request_ids if r not in rids]

    def register_failure(self, rid, error):
        self.failed_requests[rid] = str(error)


class _Sub:
    def __init__(self):
        self.seen = {}

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        return None

    def prepare_inputs(self, **kw):
        self.seen[len(self.seen)] = kw["inputs"]
        return object()

    def can_batch(self, batch, model_inputs):
        return False


def test_the_engine_applies_specs_before_prepare_inputs_and_fails_only_the_bad_row():
    sub = _Sub()
    eng = Engine.__new__(Engine)
    eng._submodules = {"talker": SimpleNamespace(
        submodule=sub, resources={},
        input_specs=_resolve({"hidden": TensorSpec("param"), "ids": TensorSpec(torch.long)}),
    )}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    batch = _Batch({
        1: {"hidden": [torch.randn(2)], "ids": [torch.zeros(1, dtype=torch.long)]},
        2: {"hidden": [torch.randn(2)], "ids": [torch.zeros(1, dtype=torch.int32)]},
    })
    eng.prepare_inputs(batch)
    assert sub.seen[0]["hidden"][0].dtype is torch.bfloat16
    assert batch.per_request_input_tensors[1]["hidden"][0].dtype is torch.float32
    assert batch.request_ids == [1] and len(batch.inputs) == 1
    assert "'talker'" in batch.failed_requests[2] and "'ids'" in batch.failed_requests[2]


class _SpecNode(NodeSubmodule):
    input_specs = {"hidden": TensorSpec("param")}

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Model:
    def get_node_precision(self, node_name):
        return NodePrecision(torch.bfloat16)

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
        return _SpecNode()

    def get_node_resources(self):
        return []

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


def test_the_load_resolves_param_against_the_yaml_precision():
    manager = EngineManager.build(
        node_names={"talker"},
        device=torch.device("cpu"),
        model_config={"nodes": {"talker": {"precision": {"param_dtype": "float16"}}}},
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="test", my_session_id="test", transfer_engine=LocalTransferEngine("test"),
        ),
        model=_Model(),
        graph_runtime=types.SimpleNamespace(),
    )
    try:
        assert manager.engine._submodules["talker"].input_specs == {"hidden": ResolvedSpec(torch.float16, "cast")}
    finally:
        manager.shutdown()
