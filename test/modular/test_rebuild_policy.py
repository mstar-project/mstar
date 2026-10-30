"""A node declares how its dropped requests are rebuilt, and the worker keeps what that rebuild reads.

The declaration names the walk that re-prefills a request, the input that carries its ids, the output that carries
each sample and the loop that samples. Nothing about a rebuild is inferred from a graph's shape or a KV layout, so a
model either says how its requests come back or never has one dropped. The worker keeps each request's prompt ids
from the policy's input the first time a step runs the policy's walk, and each sample from the policy's output.
A walk that reads more than ids, such as one that prefills a prompt with images, names those inputs, and the worker
keeps host copies of them once the request's prefill row has run.
"""

from __future__ import annotations

from array import array
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import Rebuild, SamplerStep, StepContext
from mstar.engine.resources.kv.config import PrefixSpan, StreamLayout
from mstar.engine.resources.sampler.config import discard_samples
from mstar.engine.resources.step import SubmoduleStep
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows
from mstar.worker.rebuild import HostCopies, KeptIds, Rebuilds, policies_run_here

NODE, PREFILL, VISION, DECODE, MIXED = "LLM", "prefill", "vision", "decode", "mixed"
R, A, B = 7, 8, 9
POLICY = RebuildPolicy(walk=PREFILL, ids_input="text_inputs", sample_output="new_token", loop="decode_loop")
# as Qwen3.5 declares its LLM: a prompt with images prefills in a walk that also reads the encoder's output
MEDIA = replace(POLICY, copied_inputs={VISION: ("embeds",)})


def _graphs() -> dict:
    """``NODE`` reads ``text_inputs`` and outputs ``new_token``, prefilling in ``PREFILL``, or in ``VISION`` with
    ``embeds`` as well, and decoding in a loop."""
    def node(*inputs):
        return GraphNode(name=NODE, input_names={"text_inputs", *inputs}, outputs=[GraphEdge("out", "new_token")])
    return {
        PREFILL: node(), VISION: node("embeds"),
        DECODE: Loop(section=node(), max_iters=8, outputs=[], name="decode_loop"),
    }


@pytest.mark.parametrize("field, value, why", [
    ("walk", "audio", "doesn't run"),
    ("ids_input", "embeds", "carries"),
    ("copied_inputs", {VISION: ("pixels",)}, "carries"),
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

    with pytest.raises(ValueError, match="carries"):
        Rebuilds({NODE: POLICY}, graphs)


def test_a_copied_walk_this_worker_doesnt_run_and_every_copied_walk_above_one_rank_is_left_out():
    policies = {NODE: MEDIA}
    walks = {(NODE, walk) for walk in (PREFILL, VISION, DECODE)}

    assert policies_run_here(policies, _graphs(), walks, ()) == policies
    assert policies_run_here(policies, _graphs(), walks - {(NODE, VISION)}, ())[NODE].copied_inputs == {}, (
        "a worker that never runs the walk would keep its requests droppable with no copies to rebuild them"
    )
    assert policies_run_here(policies, _graphs(), walks, {NODE})[NODE].copied_inputs == {}, (
        "rank 0 could drop a request whose copies haven't landed on a follower"
    )


@pytest.mark.parametrize("policy, spans, rebuildable", [
    (MEDIA, (PrefixSpan(3, 3, VISION), PrefixSpan(6, 4, VISION, kind="digest"), PrefixSpan(2, 2, VISION)), True),
    (POLICY, (PrefixSpan(3, 3, VISION), PrefixSpan(6, 4, VISION, kind="digest"), PrefixSpan(2, 2, VISION)), False),
    (MEDIA, (PrefixSpan(3, 3, "vit"), PrefixSpan(6, 4, VISION, kind="digest"), PrefixSpan(2, 2, VISION)), False),
    (POLICY, (PrefixSpan(4, 4, PREFILL),), True),
    (MEDIA, (PrefixSpan(3, 3, VISION), PrefixSpan(6, 4, VISION, kind="rows"), PrefixSpan(2, 2, VISION)), False),
], ids=["copied walk", "walk not copied", "two walks", "ids alone", "video"])
def test_only_a_request_whose_layout_a_rebuild_can_write_again_keeps_its_ids_and_copies(policy, spans, rebuildable):
    rebuilds = Rebuilds({NODE: policy}, _graphs())
    walk = spans[0].walk

    kept = rebuilds.keep(R, NODE, [StreamLayout(spans=spans, growth=(("decode_loop", 1),))])
    rebuilds.keep_prompts(NODE, walk, {}, {R: {"text_inputs": [torch.arange(3)], "embeds": [torch.zeros(6, 2)]}})

    assert kept == rebuildable, "a request no rebuild can write again would be preemptible, or the reverse"
    assert (R in rebuilds._kept[NODE]) == rebuildable, (
        "a request that is never dropped would hold its ids and a host copy of its frames for its life"
    )


def _rows(*rids: int) -> dict:
    return {rid: {"text_inputs": [torch.arange(rid, rid + 3)]} for rid in rids}


def _keep(rebuilds: Rebuilds, *rids: int, walk: str = PREFILL) -> None:
    for rid in rids:
        rebuilds.keep(rid, NODE, [StreamLayout(spans=(PrefixSpan(3, 3, walk),), growth=(("decode_loop", 1),))])


def test_a_prompt_rows_ids_are_kept_once_and_its_steps_share_one_copy():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    _keep(rebuilds, R, A, B)

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


def test_a_prompt_with_images_keeps_its_ids_as_its_walk_read_them_and_waits_for_its_copies():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    rows = {R: {"text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.zeros(6, 2)]}}
    _keep(rebuilds, R, walk=VISION)

    rebuilds.keep_prompts(NODE, VISION, {}, rows)

    kept = rebuilds._kept[NODE][R]
    assert kept.prompt.tolist() == [1, 2, 3, 4, 5] and kept.lengths == (3, 2), "the rebuild row would read other text"
    assert kept.copies == HostCopies(VISION), "a rebuild row would leave out the image"
    kept.sampled.append(6)
    assert "haven't landed" in rebuilds.refusal(R, NODE, DECODE, {}), "a drop would leave nothing to rebuild the image"


def _vision_step(rids, walks: dict, **fields) -> SimpleNamespace:
    infos = {rid: SimpleNamespace(graph_walk=walks.get(rid, VISION), step_metadata={"prefill_order": (rid,)})
             for rid in rids}
    inputs = {rid: {"text_inputs": [torch.tensor([rid])], "embeds": [torch.full((2,), rid)]} for rid in rids}
    ctx = StepContext(request_ids=tuple(rids), graph_walk="mixed", slot=0, capture=False,
                      request_walks={rid: info.graph_walk for rid, info in infos.items()})
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), per_request_info=infos,
                           per_request_input_tensors=inputs, step_context=ctx, **fields)


def test_a_step_of_no_copied_walk_copies_nothing_and_reads_no_row():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    rebuilds._kept[NODE] = {R: KeptIds(torch.tensor([1]), array("i"), None, HostCopies(VISION))}
    ctx = StepContext(request_ids=(R,), graph_walk=DECODE, slot=0, capture=False)

    # the step carries no rows' inputs, so reading one would raise
    rebuilds.keep_copies(SimpleNamespace(node_name=NODE, request_ids=[R], step_context=ctx))

    assert rebuilds._kept[NODE][R].copies.inputs is None, "a decode step would be taken for the image's step"


def test_a_prefill_row_keeps_a_host_copy_of_what_it_read_besides_its_ids_and_of_its_step_metadata():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    rebuilds._kept[NODE] = {rid: KeptIds(torch.tensor([1]), array("i"), None, HostCopies(VISION)) for rid in (R, A)}
    rebuilds._kept[NODE][B] = KeptIds(torch.tensor([1]), array("i"))
    step = _vision_step((R, A, B), {A: DECODE})

    rebuilds.keep_copies(step)

    copies = rebuilds._kept[NODE][R].copies
    assert copies.inputs == {"embeds": [step.per_request_input_tensors[R]["embeds"][0]]}, (
        "a rebuild would need the encoder to run again"
    )
    assert copies.step_metadata == {"prefill_order": (R,)}, "the rebuild row would run under decode's metadata"
    assert copies.step_metadata is not step.per_request_info[R].step_metadata, "a later step could rewrite the copy"
    assert copies.ready(), "the request could never be dropped"
    assert rebuilds._kept[NODE][A].copies.inputs is None, "a row of another walk would stand in for the prefill"


def test_a_rebuild_of_a_prompt_with_images_reads_its_copies_under_its_prefills_metadata():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    embeds, metadata = torch.ones(6, 2), {"prefill_order": ("text", "image", "text")}
    copies = HostCopies(VISION, {"embeds": [embeds]}, metadata)
    rebuilds._kept[NODE][R] = KeptIds(torch.tensor([1, 2, 3, 4, 5]), array("i", [6, 7]), (3, 2), copies)

    rebuild, row = rebuilds.rebuild(R, NODE, torch.device("cpu"))

    assert rebuild == Rebuild(VISION, array("i", range(1, 8))) and row.walk == VISION
    assert [t.tolist() for t in row.inputs["text_inputs"]] == [[1, 2, 3], [4, 5, 6]], (
        "the samples would follow the image instead of the prompt's last text, or decode would read the last id twice"
    )
    assert row.inputs["embeds"] == [embeds] and row.step_metadata is metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_host_copy_lands_whole_before_its_request_can_be_dropped():
    side = torch.cuda.Stream()
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), side_stream=lambda: side)
    rebuilds._kept[NODE][R] = KeptIds(torch.tensor([1]), array("i", [5]), None, HostCopies(VISION))
    rows = 1 << 22
    embeds = torch.arange(1, 1 + rows, dtype=torch.float32, device="cuda")
    expected = embeds.cpu()
    with side:
        # a cached block for the copy back, since allocating a block waits for the device
        torch.full((rows,), -1.0, device="cuda")
    step = _vision_step((R,), {})
    step.per_request_input_tensors[R]["embeds"] = [embeds]
    # a busy side stream keeps the copy queued past the point where the worker lets go of the step's tensors
    with side:
        torch.cuda._sleep(1 << 30)
    rebuilds.keep_copies(step)

    assert "haven't landed" in rebuilds.refusal(R, NODE, DECODE, {}), "the request could be dropped before its copy"
    del step, embeds
    torch.full((rows,), -1.0, device="cuda")
    torch.cuda.synchronize()
    copies = rebuilds._kept[NODE][R].copies
    assert rebuilds.refusal(R, NODE, DECODE, {}) is None
    assert torch.equal(copies.inputs["embeds"][0], expected), "the next step's tensor took the memory before the copy"

    with side:
        torch.cuda._sleep(1 << 30)
    on_device = copies.to_device(torch.device("cuda"), side)
    assert torch.equal(on_device["embeds"][0].cpu(), expected), "the rebuild row would read its inputs before they land"


def _batch(rids, rebuild=frozenset(), kept_rids=None) -> SimpleNamespace:
    step = SimpleNamespace(steps={"sampler": SamplerStep(kept_rids=kept_rids)})
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), rebuild_rids=rebuild, step=step,
                           step_context=StepContext(request_ids=tuple(rids), graph_walk=DECODE, slot=0, capture=False))


@pytest.mark.parametrize("sampled", [[4, 5, 6], [[4], [5], [6]]])  # the sampler's [B] output, and a [B, 1] one
def test_only_a_rows_kept_sample_is_appended(sampled):
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    _keep(rebuilds, R, A, B)
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
    _keep(rebuilds, R)
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
