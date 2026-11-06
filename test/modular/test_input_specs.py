"""A node's input specs: resolved against its precision at load, and applied by
the engine to each request's routed tensors before ``prepare_inputs`` (CPU,
mock submodules)."""
import types
from types import SimpleNamespace

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.chunking import ChunkingManagement, ChunkRange
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


def test_a_floating_mismatch_is_cast_and_the_batch_dict_is_left_alone():
    fp32 = torch.randn(2, 4)
    inputs = {"hidden": [fp32], "other": [torch.ones(1)]}
    out = apply_input_specs("talker", inputs, _resolve({"hidden": TensorSpec("param")}))
    assert out["hidden"][0].dtype is torch.bfloat16
    assert torch.equal(out["hidden"][0], fp32.to(torch.bfloat16))
    assert out["other"] is inputs["other"]
    assert inputs["hidden"][0] is fp32


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
        self.running_batched = None
        self.chunk_ranges = {}
        self.first_chunk_rids = set()

    def register_prepare_batch(self, inputs, uniform=None):
        self.inputs = inputs

    def drop_rids(self, rids):
        self.request_ids = [r for r in self.request_ids if r not in rids]

    def register_failure(self, rid, error, walk=None):
        self.failed_requests[rid] = str(error)


class _Sub:
    def __init__(self):
        self.seen = {}

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        return None

    def prepare_inputs(self, **kw):
        self.seen[len(self.seen)] = kw["inputs"]
        return SimpleNamespace(input_seq_len=2)

    def split_inputs(self, graph_walk, fwd_info, full, start, end):
        return SimpleNamespace()

    def can_batch(self, batch, model_inputs):
        return False


def _engine(sub) -> Engine:
    eng = Engine.__new__(Engine)
    eng._submodules = {"talker": SimpleNamespace(
        submodule=sub, resources={},
        input_specs=_resolve({"hidden": TensorSpec("param"), "ids": TensorSpec(torch.long)}),
    )}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    eng._keyed_walks = {}
    eng._chunking = ChunkingManagement(eng._submodules)
    return eng


def test_the_engine_applies_specs_before_prepare_inputs_and_fails_only_the_bad_row():
    sub = _Sub()
    eng = _engine(sub)
    batch = _Batch({
        1: {"hidden": [torch.randn(2)], "ids": [torch.zeros(1, dtype=torch.long)]},
        2: {"hidden": [torch.randn(2)], "ids": [torch.zeros(1, dtype=torch.int32)]},
    })
    eng.prepare_inputs(batch)
    assert sub.seen[0]["hidden"][0].dtype is torch.bfloat16
    assert batch.per_request_input_tensors[1]["hidden"][0].dtype is torch.float32
    assert batch.request_ids == [1] and len(batch.inputs) == 1
    assert "'talker'" in batch.failed_requests[2] and "'ids'" in batch.failed_requests[2]


def test_a_chunked_row_is_prepared_through_its_specs_too():
    sub = _Sub()
    eng = _engine(sub)
    batch = _Batch({1: {"hidden": [torch.randn(2)], "ids": [torch.zeros(1, dtype=torch.long)]}})
    batch.chunk_ranges = {1: ChunkRange(start=0, end=1, total=2, stride=1)}
    eng.prepare_inputs(batch)
    assert sub.seen[0]["hidden"][0].dtype is torch.bfloat16
    assert batch.request_ids == [1] and not batch.failed_requests


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

    def kv_labels(self):
        return {}

    def get_graph_walk_graphs(self):
        return {}

    def combined_walk_of(self):
        return {}

    def chunking_policy_of(self):
        return {}

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return ""


def test_the_load_resolves_param_against_the_yaml_precision():
    manager = EngineManager.build(
        node_names={"talker"},
        device=torch.device("cpu"),
        model_config={
            "node_groups": [{"node_names": ["talker"], "ranks": [0]}],
            "nodes": {"talker": {"precision": {"param_dtype": "float16"}}},
        },
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
