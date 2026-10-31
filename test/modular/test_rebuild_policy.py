"""A node declares how its dropped requests are rebuilt, and the worker keeps what that rebuild reads.

The declaration names the walk that re-prefills a request, the input that carries its ids, the output that carries
each sample and the loop that samples. Nothing about a rebuild is inferred from a graph's shape or a KV layout, so a
model either says how its requests come back or never has one dropped. The worker keeps each request's prompt ids
from the policy's input the first time a step runs the policy's walk, and each sample from the policy's output.
"""

from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import Rebuild, SamplerStep, StepContext
from mstar.engine.resources.sampler.config import discard_samples
from mstar.engine.resources.step import SubmoduleStep
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows
from mstar.worker.rebuild import KeptIds, Rebuilds

NODE, PREFILL, DECODE, MIXED = "LLM", "prefill", "decode", "mixed"
R, A, B = 7, 8, 9
POLICY = RebuildPolicy(walk=PREFILL, ids_input="text_inputs", sample_output="new_token", loop="decode_loop")


def _graphs() -> dict:
    """``NODE`` reads ``text_inputs`` and outputs ``new_token``, prefilling in ``PREFILL`` and decoding in a loop."""
    def node():
        return GraphNode(name=NODE, input_names={"text_inputs"}, outputs=[GraphEdge("out", "new_token")])
    return {PREFILL: node(), DECODE: Loop(section=node(), max_iters=8, outputs=[], name="decode_loop")}


@pytest.mark.parametrize("field, value, why", [
    ("walk", "vision", "doesn't run"),
    ("ids_input", "embeds", "carries only"),
    ("sample_output", "codes", "doesn't output"),
    ("loop", "talker_loop", "in no walk"),
])
def test_a_policy_naming_what_the_graph_doesnt_have_is_refused_at_load(field, value, why):
    policy = RebuildPolicy(**{**vars(POLICY), field: value})

    with pytest.raises(ValueError, match=why):
        Rebuilds({NODE: policy}, _graphs())


def test_a_walk_whose_node_reads_more_than_the_ids_is_refused_at_load():
    graphs = _graphs()
    graphs[PREFILL].input_names = {"text_inputs", "pixel_values"}

    with pytest.raises(ValueError, match="carries only"):
        Rebuilds({NODE: POLICY}, graphs)


def _rows(*rids: int) -> dict:
    return {rid: {"text_inputs": [torch.arange(rid, rid + 3)]} for rid in rids}


def test_a_prompt_rows_ids_are_kept_once_and_its_steps_share_one_copy():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())

    rebuilds.keep_prompts(NODE, PREFILL, {}, _rows(R, A))
    rebuilds.keep_prompts(NODE, PREFILL, {}, {R: {"text_inputs": [torch.tensor([0])]}})  # its rebuild row
    rebuilds.keep_prompts(NODE, MIXED, {A: PREFILL, B: DECODE}, _rows(A, B))
    rebuilds.keep_prompts(NODE, DECODE, {}, _rows(B))
    rebuilds.keep_prompts(NODE, PREFILL, {}, {B: {}})  # a TP follower's unread row

    kept = rebuilds._kept[NODE]
    assert kept[R].prompt.tolist() == [7, 8, 9] and kept[A].prompt.tolist() == [8, 9, 10], "it kept another row's ids"
    assert B not in kept, "a decode row's input, or an unread row's, would be taken for its prompt"
    assert kept[R].prompt.untyped_storage().data_ptr() == kept[A].prompt.untyped_storage().data_ptr(), (
        "each row would pay a copy of its own"
    )


def _batch(rids, rebuild=frozenset(), kept_rids=None) -> SimpleNamespace:
    step = SimpleNamespace(steps={"sampler": SamplerStep(kept_rids=kept_rids)})
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), rebuild_rids=rebuild, step=step,
                           step_context=StepContext(request_ids=tuple(rids), graph_walk=DECODE, slot=0, capture=False))


@pytest.mark.parametrize("sampled", [[4, 5, 6], [[4], [5], [6]]])  # the sampler's [B] output, and a [B, 1] one
def test_only_a_rows_kept_sample_is_appended(sampled):
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    rebuilds.keep_prompts(NODE, PREFILL, {}, _rows(R, A, B))
    # A's sampler throws its sample away, and B is a rebuild row
    batch = _batch((R, A, B), rebuild=frozenset({B}), kept_rids=frozenset({R, B}))

    rebuilds.keep_samples(batch, {}, HostRows((R, A, B), {"new_token": torch.tensor(sampled)}))
    rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([9])]}}, None)

    kept = rebuilds._kept[NODE]
    assert list(kept[R].sampled) == [4, 9], "a rebuild would prefill other ids than the request's"
    assert not (kept[A].sampled or kept[B].sampled), "it kept a discarded sample"


@pytest.mark.parametrize("host", [True, False])
def test_a_sample_of_more_than_one_id_is_refused(host):
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    rebuilds.keep_prompts(NODE, PREFILL, {}, _rows(R))
    codes = torch.tensor([[4, 5, 6]])  # as a talker samples several codebooks

    with pytest.raises(ValueError, match="hold 3 ids"):
        if host:
            rebuilds.keep_samples(_batch((R,)), {}, HostRows((R,), {"new_token": codes}))
        else:
            rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [codes[0]]}}, None)


@pytest.mark.parametrize("why, node, walk, sampled, iters", [
    ("declares no rebuild policy", "vocoder", DECODE, 1, 0),
    ("keeps no ids", NODE, DECODE, None, 0),
    ("doesn't run", NODE, PREFILL, 1, 0),
    ("don't cover its KV", NODE, DECODE, 1, 1),
])
def test_a_drop_is_refused_unless_the_kept_ids_cover_the_kv(why, node, walk, sampled, iters):
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    if sampled is not None:
        rebuilds._kept[NODE][R] = KeptIds(torch.tensor([1, 2]), array("i", [5] * sampled))

    assert why in rebuilds.refusal(R, node, walk, {"decode_loop": iters})
    rebuilds._kept[NODE][R] = KeptIds(torch.tensor([1, 2]), array("i", [5, 6]))
    assert rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 1}) is None, "a request its ids cover was refused"


def test_a_rebuild_writes_the_prompt_and_samples_and_its_row_reads_all_but_the_last():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    rebuilds._kept[NODE][R] = KeptIds(torch.tensor([1, 2, 3]), array("i", [4, 5]))

    rebuild, row = rebuilds.rebuild(R, NODE, torch.device("cpu"))

    assert rebuild == Rebuild(PREFILL, array("i", [1, 2, 3, 4, 5])), "the KV would lay out other ids"
    assert (row.node, row.walk) == (NODE, PREFILL) and list(row.inputs) == ["text_inputs"]
    [tokens] = row.inputs["text_inputs"]
    assert tokens.tolist() == [1, 2, 3, 4] and tokens.dtype == torch.long, "decode would read the last id twice"


def test_the_engine_keeps_no_sample_of_a_rebuild_row_whatever_the_model_declared():
    for kept_rids in (None, frozenset({R, A})):
        step = SubmoduleStep(segments=[], steps={"sampler": SamplerStep(kept_rids=kept_rids)})

        discard_samples(step, [R, A], frozenset({R}))

        assert step.steps["sampler"].kept_rids == {A}, "the rebuild row's sample would overwrite decode's next input"
