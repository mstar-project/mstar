"""Lean captures (`Resource.lean_capture`): a resource whose kernels can be left out of
a step whose rows all allow it gets a twin of each bucket it accepts, keyed by the set
of resources that run lean there (`BucketKey.lean`). Which twins exist, which one a
step leases, and what the engine tells the resources (`StepContext.lean`), with stub
resources; the LoRA pool's and the sampler's own cases are in their test files."""
from __future__ import annotations

from dataclasses import dataclass

from mstar.engine.resources.base import Resource, build_resource
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import BucketKey, StepContext, lean_subsets


class _Plain(Resource):
    build = None


def test_a_bucket_is_plain_unless_it_names_resources_running_lean():
    plain = BucketKey("decode", 4, 4, "cfg")
    assert plain.lean == frozenset() and str(plain) == "decode[bs=4,tokens=4,key=cfg]"
    twin = BucketKey("decode", 4, 4, "cfg", frozenset({"sampler", "lora"}))
    assert twin != plain and str(twin) == "decode[bs=4,tokens=4,key=cfg,lean=lora+sampler]"
    assert StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False).lean == frozenset()


def test_subsets_go_largest_first_and_ties_in_declaration_order():
    assert lean_subsets(()) == ()
    assert lean_subsets(("lora",)) == (frozenset({"lora"}),)
    assert lean_subsets(("lora", "sampler")) == (
        frozenset({"lora", "sampler"}), frozenset({"lora"}), frozenset({"sampler"}),
    )
    assert lean_subsets(("c", "a", "b"))[1:4] == (frozenset("ca"), frozenset("cb"), frozenset("ab"))
    assert lean_subsets(("c", "a", "b"))[4:] == (frozenset("c"), frozenset("a"), frozenset("b"))


def test_a_resource_runs_plain_unless_it_says_otherwise():
    resource = _Plain()
    assert not resource.lean_capture and not resource.lean_for([1, 2])
    assert resource.lean_captures(None) and resource.resource_key is None


@dataclass
class _Spec(NodeResourceSpec):
    @property
    def resource_class(self):
        return _Built


class _Built(_Plain):
    @classmethod
    def build(cls, spec, info):
        return cls()


def test_a_built_resource_knows_its_key():
    assert build_resource(_Spec(resource_key="mine", nodes={"N"}), None).resource_key == "mine"
