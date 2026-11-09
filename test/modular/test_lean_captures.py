"""Lean captures (`Resource.lean_capture`): a resource whose kernels can be left out of
a step whose rows all allow it gets a twin of each bucket it accepts, keyed by the set
of resources that run lean there (`BucketKey.lean`). Which twins exist, which one a
step leases, and what the engine tells the resources (`StepContext.lean`), with stub
resources; the LoRA pool's and the sampler's own cases are in their test files."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.resources.base import CGSlotSpec, Resource, build_resource
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext, SubmoduleStep, lean_subsets
from mstar.model.submodule_base import ARNodeInputs
from mstar.testing_utils.engine import make_engine, make_mgmt

WALK = "decode"


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


# ---- which bucket a step leases ----

class _Lean(Resource):
    """Lean for the rows not in ``busy``."""
    build = None
    lean_capture = True

    def __init__(self, busy=()):
        self.busy = set(busy)

    def lean_for(self, request_ids) -> bool:
        return self.busy.isdisjoint(request_ids)


def _runner(captured):
    """A runner holding ``captured`` (bucket keys) of one batched config, two slots each."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    config = BatchedAcceleratorGraphConfig(
        capture_graph_walk=WALK, capture_batch_sizes=[4, 8], single_request_inputs=ARNodeInputs(input_seq_len=1),
    )
    runner = object.__new__(AcceleratorGraphRunner)
    runner._buckets, runner._twins = {}, {}
    for bucket in captured:
        for slot in range(2):
            runner._register_slot(CGSlotSpec(bucket, slot, config, 0), object())
    return runner


def _bucket(bs, *lean):
    return BucketKey(WALK, bs, bs, "cfg", frozenset(lean))


def _engine(resources, captured, runner=True):
    """An engine whose node keys its captures "cfg" and has ``captured``."""
    return make_engine(_submodules={"N": make_mgmt(
        SimpleNamespace(cg_key_info=lambda walk, info, **kw: "cfg"), resources=resources,
        accelerator_graph_runner=_runner(captured) if runner else None,
    )})


def _reserve(engine, rids):
    batch = SimpleNamespace(
        node_name="N", slot=0, request_ids=list(rids), inputs=None, cg_key_info=None,
        per_request_info_wrapped={}, per_request_input_metadata={},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
    )
    batch.lease_slot = lambda lease: setattr(batch.step_context, "slot_lease", lease)
    lease = engine.reserve_replay_slot(batch)
    return (None if lease is None else lease.bucket), batch.step_context.lean


def test_the_node_keeps_its_lean_resources_in_declaration_order():
    mgmt = make_mgmt(resources={"b": _Lean(), "plain": _Plain(), "a": _Lean(busy=[3])})
    assert mgmt.lean_keys == ("b", "a")
    assert mgmt.lean_of([1, 2]) == {"a", "b"} and mgmt.lean_of([1, 3]) == {"b"}
    assert make_mgmt(resources={"plain": _Plain()}).lean_of([1]) == frozenset()
    assert mgmt.lean_candidates(frozenset("ab")) == (frozenset("ab"), frozenset("b"), frozenset("a"))


def test_a_step_leases_the_twin_of_every_resource_its_rows_let_run_lean():
    every = [_bucket(4), _bucket(4, "a"), _bucket(4, "b"), _bucket(4, "a", "b")]
    engine = _engine({"a": _Lean(busy=[3]), "b": _Lean(busy=[4])}, every)
    assert _reserve(engine, [1, 2]) == (_bucket(4, "a", "b"), {"a", "b"})
    assert _reserve(engine, [1, 3]) == (_bucket(4, "b"), {"b"})
    assert _reserve(engine, [1, 4]) == (_bucket(4, "a"), {"a"})
    # no resource lean: the plain bucket
    assert _reserve(engine, [3, 4]) == (_bucket(4), frozenset())


def test_a_missing_twin_falls_back_to_the_largest_subset_then_the_plain_bucket():
    resources = {"a": _Lean(), "b": _Lean()}
    # the full set's twin missing: the subsets by size, ties in declaration order
    assert _reserve(_engine(resources, [_bucket(4), _bucket(4, "a"), _bucket(4, "b")]), [1]) == (_bucket(4, "a"), {"a"})
    assert _reserve(_engine(resources, [_bucket(4), _bucket(4, "b")]), [1]) == (_bucket(4, "b"), {"b"})
    assert _reserve(_engine(resources, [_bucket(4)]), [1]) == (_bucket(4), frozenset())


def test_the_twin_is_the_tightest_plain_buckets_never_a_wider_one():
    """A step that fits the 4-row bucket replays it plain rather than pad onto an
    8-row twin, as it would have before the twin was captured."""
    engine = _engine({"a": _Lean()}, [_bucket(4), _bucket(8), _bucket(8, "a")])
    assert _reserve(engine, [1, 2, 3]) == (_bucket(4), frozenset())
    assert _reserve(engine, [1, 2, 3, 4, 5]) == (_bucket(8, "a"), {"a"})


def test_an_eager_step_runs_lean_what_its_rows_allow():
    resources = {"a": _Lean(busy=[3]), "b": _Lean()}
    # past the widest bucket
    assert _reserve(_engine(resources, [_bucket(4), _bucket(4, "a")]), range(10, 15)) == (None, {"a", "b"})
    # a node with no captures at all
    assert _reserve(_engine(resources, [], runner=False), [1, 3]) == (None, {"b"})


def test_a_node_without_lean_resources_leases_its_plain_bucket():
    engine = _engine({"plain": _Plain()}, [_bucket(4), _bucket(4, "a")])
    assert _reserve(engine, [1]) == (_bucket(4), frozenset())


def test_the_declaration_check_compares_the_nodes_own_key_whatever_runs_lean():
    """The lease names the node's own ``cg_key_info`` in every twin, so a model whose
    declare_step disagrees with its cg_key_info fails under a lean lease too."""

    def plan(step, phase):
        raise NotImplementedError("reached plan")

    engine = make_engine(_runner=SimpleNamespace(plan=plan))
    lease = SlotLease(slot=0, bucket=_bucket(1, "a"))
    ctx = StepContext(request_ids=[1], graph_walk=WALK, slot=0, capture=False, slot_lease=lease)

    def drive(key):
        engine._drive_step(SimpleNamespace(node_name="N", preplan_event=None), make_mgmt(), [1], [], {},
                           ctx, lease, True, SubmoduleStep(steps={}, cg_key_info=key), False)

    with pytest.raises(RuntimeError, match="disagree"):
        drive("other")
    with pytest.raises(NotImplementedError, match="reached plan"):
        drive("cfg")
