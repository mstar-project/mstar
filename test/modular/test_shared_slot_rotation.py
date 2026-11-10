"""Nodes that share a resource never hold one double-buffer slot back to back.

Resources key their buffers by slot, so a pre-plan of one node landing on the
slot another node's graph is reading overwrites that graph's plan.
"""
import torch

import mstar.engine.engine as engine_module
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, share_slot_rotations
from mstar.engine.resources.base import Resource
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.model.submodule_base import NodeSubmodule
from mstar.testing_utils.engine import make_mgmt


class _PrePlanning(Resource):
    """Stands in for a KV cache two nodes share: it pre-plans, so its nodes double-buffer."""

    @classmethod
    def build(cls, spec, info):
        return cls()

    @property
    def supports_preplan(self):
        return True


def _node(*resources, num_slots=2):
    node = make_mgmt(resources={str(i): resource for i, resource in enumerate(resources)})
    node._slots.num_slots = num_slots
    return node


def test_nodes_sharing_a_resource_alternate_slots():
    shared = _PrePlanning()
    prefix, action = _node(shared), _node(shared)
    share_slot_rotations([prefix, action])
    assert [prefix.lease_slot(), action.lease_slot(), prefix.lease_slot(), action.lease_slot()] == [0, 1, 0, 1]
    assert action.next_slot == prefix.next_slot == 0


def test_nodes_sharing_nothing_rotate_apart():
    vit, llm = _node(_PrePlanning()), _node(_PrePlanning())
    share_slot_rotations([vit, llm])
    assert [vit.lease_slot(), llm.lease_slot()] == [0, 0]


def test_a_chain_of_shared_resources_is_one_rotation():
    a, b = _PrePlanning(), _PrePlanning()
    first, middle, last = _node(a), _node(a, b), _node(b)
    other_order = [last, first, middle]
    share_slot_rotations(other_order)
    assert first._slots is middle._slots is last._slots


def test_the_rotation_covers_the_most_slots_any_sharer_needs():
    shared = _PrePlanning()
    one, two = _node(shared, num_slots=1), _node(shared, num_slots=2)
    share_slot_rotations([one, two])
    assert one.num_slots == two.num_slots == 2


def test_a_later_merge_reaches_every_node_of_the_merged_rotation():
    a, b = _PrePlanning(), _PrePlanning()
    first, second, other, bridge = _node(a), _node(a), _node(b), _node(a, b)
    share_slot_rotations([first, second, other, bridge])
    assert first._slots is second._slots is other._slots is bridge._slots


class _Node(NodeSubmodule):
    """Never stepped: only the resources it declares matter here."""

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        raise AssertionError("not stepped")

    def forward(self, graph_walk, engine_inputs, **kwargs):
        raise AssertionError("not stepped")


def test_an_engine_gives_the_nodes_of_a_shared_resource_one_rotation(monkeypatch):
    monkeypatch.setattr(engine_module, "build_resource", _PrePlanning.build)
    engine = Engine(None)
    engine.load_model(
        {"prefix": _Node(), "action": _Node()},
        [SamplerSpec(resource_key="kv", nodes={"prefix", "action"}, vocab_size=None)],
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"), transfer_engine_info=None,
    )
    prefix, action = engine._submodules["prefix"], engine._submodules["action"]
    assert [prefix.lease_slot(), action.lease_slot()] == [0, 1], "the action node would pre-plan on the prefix's slot"
