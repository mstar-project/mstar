"""Each node's dtype is resolved once at load, and the load cast and every
runtime autocast scope read that one value (CPU, mock submodules)."""
import types

import pytest
import torch

from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import CommGroup, WorkerParallelGroups
from mstar.engine.node_dtype import resolve_model_dtype, resolve_node_dtype
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
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


class _Override(_Node):
    def get_autocast_dtype(self):
        return torch.float16


class _OptedOut(_Override):
    disable_autocast = True


_NODES = {"inherit": _Node, "override": _Override, "opted_out": _OptedOut}


class _Model:
    """What ``EngineManager.build`` reads off a model, with no resources."""

    def __init__(self, dtype):
        self._dtype = dtype

    def get_autocast_dtype(self):
        return self._dtype

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
        return _NODES[node_name]()

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


def _build(model, parallel_groups: WorkerParallelGroups | None = None) -> EngineManager:
    return EngineManager.build(
        node_names=set(_NODES),
        device=torch.device("cpu"),
        model_config={"node_groups": [{"node_names": sorted(_NODES), "ranks": [0]}]},
        parallel_groups=parallel_groups or WorkerParallelGroups(num_workers=1, global_rank=0),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="test", my_session_id="test", transfer_engine=None,
        ),
        model=model,
        graph_runtime=types.SimpleNamespace(),
    )


def test_model_dtype_yaml_may_not_reenable_autocast():
    assert resolve_model_dtype(_Model(torch.bfloat16), {}) is torch.bfloat16
    assert resolve_model_dtype(_Model(torch.bfloat16), {"autocast_dtype": None}) is None
    assert resolve_model_dtype(_Model(None), {"autocast_dtype": None}) is None
    with pytest.raises(ValueError, match="cannot turn it on"):
        resolve_model_dtype(_Model(None), {"autocast_dtype": torch.bfloat16})


def test_node_dtype_opt_out_beats_override_beats_model():
    assert resolve_node_dtype(_Node(), torch.bfloat16) is torch.bfloat16
    assert resolve_node_dtype(_Override(), torch.bfloat16) is torch.float16
    assert resolve_node_dtype(_OptedOut(), torch.bfloat16) is None


def test_runtime_scope_dtype_matches_load_cast():
    manager = _build(_Model(torch.bfloat16))
    try:
        engine = manager.engine
        for name, dtype in {
            "inherit": torch.bfloat16, "override": torch.float16, "opted_out": None,
        }.items():
            assert engine._autocast_dtype_for(name) is dtype, name
            # None keeps the build dtype, float32 here
            weight = engine._submodules[name].submodule.proj.weight
            assert weight.dtype is (dtype or torch.float32), name
    finally:
        manager.shutdown()


def test_a_node_placing_its_kv_positions_over_its_sp_ranks_loads_without_the_sp_group():
    """It runs replicated across those ranks, so its model gets no SP group to run its own
    sequence parallelism with."""
    kwargs = {}

    def get_submodule(name, device, **extra):
        kwargs[name] = extra

    model = _Model(torch.bfloat16)
    model.get_submodule = get_submodule
    model.get_node_resources = lambda: [KVSpec(resource_key="kv", nodes={"inherit"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=64, page_size=16, context_parallel=True,
    ))]
    model.get_default_sharding_config = lambda: ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})
    groups = WorkerParallelGroups(num_workers=2, global_rank=0)
    groups.add_sp("inherit", CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1]))
    _build(model, groups).shutdown()
    assert "sp_group" not in kwargs["inherit"]
