"""Where the engine cuts a matched prefix out of a walk's inputs.

The cut happens once per request, between the submodule preparing its inputs
and the step being declared from them, because everything downstream sizes
itself from ``input_seq_len``: the spans a step declares, the capture bucket,
the padding. Cutting later would leave those describing tokens that never run.

The other half is what is never offered. The default cut only knows how to
slice sequence-shaped fields, and a walk that writes more than one span from
one set of inputs carries something opaque to say so, so those walks are
skipped rather than trimmed to a length only one of their labels agreed to.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import Resource, StepContext, StepRunner
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    NodeInputs,
)

RID = "r0"
NODE = "LLM"
OTHER = "Talker"
WALK = "prefill"
PROMPT = 100


class _Answering(Resource):
    """Answers a fixed match, and records what it was applied with."""

    def __init__(self, matched: int | None):
        self._matched = matched
        self.resolved = 0
        self.applied: list[tuple[int, int]] = []

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def resolve_cached_prefix(self, rid, node_name, graph_walk):
        self.resolved += 1
        return self._matched

    def apply_cached_prefix(self, rid, node_name, graph_walk, inputs, matched_len):
        self.applied.append((inputs.input_seq_len, matched_len))


class _Submodule:
    """Cuts the way a real AR submodule does, without being a torch module."""

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        return ARNodeSubmodule.split_inputs(
            self, graph_walk, fwd_info, inputs, start, end,
        )

    @staticmethod
    def declare_step(inputs: NodeInputs) -> int:
        """The span a step would declare from these inputs."""
        return inputs.input_seq_len


def _engine(resources: dict[str, Resource], node_resources: dict[str, list[str]]):
    engine = Engine.__new__(Engine)
    engine._resources = resources
    engine._runner = StepRunner(resources, node_resources=node_resources)
    engine._submodules = {
        node: SimpleNamespace(submodule=_Submodule())
        for node in node_resources
    }
    # every node keys WALK, as a declaration would name it
    engine._keyed_walks = {node: {WALK} for node in node_resources}
    engine._prefix_model = "_Model"
    return engine


def _batch(node_name: str = NODE):
    return SimpleNamespace(
        node_name=node_name,
        request_ids=(RID,),
        step_context=StepContext(
            request_ids=(RID,), graph_walk=WALK, slot=0, capture=False,
        ),
        per_request_info={RID: None},
    )


def _inputs(**overrides) -> ARNodeInputs:
    return ARNodeInputs(
        input_ids=torch.arange(PROMPT), input_seq_len=PROMPT, **overrides,
    )


def _stage(engine, inputs, node_name: str = NODE):
    return engine._skip_cached_prefix(_batch(node_name), RID, inputs)


# ── the cut ─────────────────────────────────────────────────────────────


def test_a_match_shortens_the_span_a_step_would_declare():
    resource = _Answering(96)
    engine = _engine({"kv": resource}, {NODE: ["kv"]})

    cut = _stage(engine, _inputs())

    assert _Submodule.declare_step(cut) == 4, (
        "the step would still declare the tokens the cache already holds"
    )
    assert cut.input_ids.tolist() == list(range(96, PROMPT)), (
        "the ids left to run are not the ones past the match"
    )


def test_the_smallest_answer_wins():
    engine = _engine(
        {"kv": _Answering(96), "pos": _Answering(32)}, {NODE: ["kv", "pos"]},
    )

    assert _stage(engine, _inputs()).input_seq_len == PROMPT - 32, (
        "a resource holding less than another was overrun"
    )


def test_a_resource_with_no_opinion_is_not_an_answer_of_zero():
    engine = _engine(
        {"kv": _Answering(96), "quiet": _Answering(None)},
        {NODE: ["kv", "quiet"]},
    )

    assert _stage(engine, _inputs()).input_seq_len == PROMPT - 96, (
        "a resource with no opinion was counted as holding nothing"
    )


def test_a_zero_match_leaves_the_inputs_exactly_as_they_were():
    engine = _engine({"kv": _Answering(0)}, {NODE: ["kv"]})
    inputs = _inputs()

    assert _stage(engine, inputs) is inputs, "a miss still rebuilt the inputs"


def test_apply_is_handed_the_untrimmed_inputs():
    resource = _Answering(96)
    engine = _engine({"kv": resource}, {NODE: ["kv"]})

    _stage(engine, _inputs())

    assert resource.applied == [(PROMPT, 96)], (
        "apply saw the inputs after the cut, so it could not read what was skipped"
    )


# ── what is never offered ───────────────────────────────────────────────


@pytest.mark.parametrize("opaque", [
    {"resource_step_info": True},
    {"kwargs": {"guidance": 1.0}},
    {"tensor_inputs": {"extra": torch.zeros(PROMPT)}},
])
def test_a_walk_carrying_anything_opaque_never_probes(opaque):
    resource = _Answering(96)
    engine = _engine({"kv": resource}, {NODE: ["kv"]})
    inputs = _inputs(**opaque)

    assert _stage(engine, inputs) is inputs, "an opaque walk was cut anyway"
    assert resource.resolved == 0, (
        f"a walk carrying {sorted(opaque)} was probed and would have been cut "
        "to a length only one of its labels agreed to"
    )


def test_a_falsy_step_info_is_not_opaque():
    resource = _Answering(96)
    engine = _engine({"kv": resource}, {NODE: ["kv"]})

    cut = _stage(engine, _inputs(resource_step_info=False))

    assert cut.input_seq_len == 4 and resource.resolved == 1, (
        "a walk whose step info is merely falsy was taken for an opaque one"
    )


# ── which walks are probed ──────────────────────────────────────────────


def test_a_walk_the_model_did_not_key_is_never_probed():
    resource = _Answering(96)
    engine = _engine({"kv": resource}, {NODE: ["kv"]})
    engine._keyed_walks = {NODE: {"prefill_text"}}

    _stage(engine, _inputs())

    assert resource.resolved == 0, "a walk the model never named was offered a prefix"


def test_a_node_with_no_keyed_stream_is_never_probed():
    resource = _Answering(96)
    engine = _engine({"kv": resource}, {NODE: ["kv"]})
    engine._keyed_walks = {}

    _stage(engine, _inputs())

    assert resource.resolved == 0, "a node nobody declared was probed"


def test_the_keyed_walk_placing_its_own_positions_names_the_model_and_walk():
    engine = _engine({"kv": _Answering(96)}, {NODE: ["kv"]})

    with pytest.raises(AssertionError, match=f"_Model keys the '{WALK}' walk of {NODE}"):
        _stage(engine, _inputs(custom_pos_ids=torch.arange(PROMPT)))


def test_the_keyed_walk_with_no_ids_to_cut_names_the_model_and_walk():
    engine = _engine({"kv": _Answering(96)}, {NODE: ["kv"]})

    with pytest.raises(AssertionError, match=f"_Model keys the '{WALK}' walk"):
        _stage(engine, NodeInputs(input_seq_len=PROMPT))


# ── scope ───────────────────────────────────────────────────────────────


def test_another_nodes_resource_is_never_swept():
    mine = _Answering(96)
    theirs = _Answering(16)
    engine = _engine(
        {"kv": mine, "talker_kv": theirs},
        {NODE: ["kv"], OTHER: ["talker_kv"]},
    )

    cut = _stage(engine, _inputs())

    assert cut.input_seq_len == 4, "another node's resource shortened this match"
    assert theirs.resolved == 0 and theirs.applied == [], (
        "another node's resource was asked about this request"
    )


# ── what the postprocess hands back ─────────────────────────────────────


class _Chaining(_Answering):
    """Records the outputs it was handed to key a generation from."""

    def __init__(self):
        super().__init__(None)
        self.extended: list[dict] = []

    def extend_prefix_chain(self, rid, node_name, graph_walk, outputs):
        self.extended.append(outputs)


def test_a_declared_nodes_resource_is_handed_the_step_it_sampled_from():
    mine = _Chaining()
    theirs = _Chaining()
    engine = _engine(
        {"kv": mine, "talker_kv": theirs},
        {NODE: ["kv"], OTHER: ["talker_kv"]},
    )
    sampled = {"text_inputs": [torch.tensor([7])]}

    engine.extend_prefix_chains(_batch(), {RID: sampled})

    assert mine.extended == [sampled], (
        "the node's own resource never saw what the step sampled, so nothing "
        "it generates is ever keyed"
    )
    assert theirs.extended == [], "another node's resource was handed the step"


def test_a_step_that_sampled_nothing_for_a_request_hands_back_nothing():
    mine = _Chaining()
    engine = _engine({"kv": mine}, {NODE: ["kv"]})

    engine.extend_prefix_chains(_batch(), {})

    assert mine.extended == [], "a request with no outputs was chained anyway"


# ── one request's failure ───────────────────────────────────────────────


class _RaisingFor(_Answering):
    """Matches every request, and raises when applied to one of them."""

    def __init__(self, matched: int, rid: str):
        super().__init__(matched)
        self._rid = rid

    def apply_cached_prefix(self, rid, node_name, graph_walk, inputs, matched_len):
        if rid == self._rid:
            raise RuntimeError("apply failed")
        super().apply_cached_prefix(rid, node_name, graph_walk, inputs, matched_len)


class _Preparing(_Submodule):
    """Prepares the same prompt for every request, and never batches."""

    def prepare_inputs(self, graph_walk, fwd_info, inputs, resources):
        return _inputs()

    def can_batch(self, batch, model_inputs):
        return False


def test_a_resource_that_raises_in_apply_fails_only_its_own_request():
    engine = _engine({"kv": _RaisingFor(96, RID)}, {NODE: ["kv"]})
    engine._submodules[NODE] = SimpleNamespace(submodule=_Preparing(), resources={})
    batch = ExecutingBatch(
        node_name=NODE, per_request_info={RID: None, "r1": None},
        step_context=StepContext(
            request_ids=(RID, "r1"), graph_walk=WALK, slot=0, capture=False,
        ),
    )

    try:
        engine._prepare_inputs(batch)
    except RuntimeError as error:
        pytest.fail(f"one request's prefix stage failed the whole batch: {error!r}")

    assert set(batch.failed_requests) == {RID}, "the failing request was not failed"
    assert list(batch.request_ids) == ["r1"] and len(batch.inputs) == 1, (
        "the request the resource did not raise for was dropped with it"
    )
    assert batch.inputs[0].input_seq_len == PROMPT - 96, (
        "the surviving request lost its own match"
    )
