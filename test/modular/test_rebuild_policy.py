"""A node declares how its dropped requests are rebuilt, and the worker keeps what that rebuild reads.

The declaration names the walk that re-prefills a request, the input that carries its ids, the output that carries
each sample and the loop that samples. Nothing about a rebuild is inferred from a graph's shape, so a model either
says how its requests come back or never has one dropped. A walk that reads more than ids, such as one that prefills
an image, names those inputs. For a request whose layout a rebuild could write again, the worker records each prefill
step once it has run: the ids it read, and host copies of the other inputs of a copied walk. A rebuild runs those
steps again in order, and the last one carries the samples.
"""

from __future__ import annotations

from array import array
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import Rebuild, SamplerStep, StepContext
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.sampler.config import discard_samples
from mstar.engine.resources.step import SubmoduleStep
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows
from mstar.worker.rebuild import HostCopies, KeptIds, PrefillStep, Rebuilds, policies_run_here

NODE, PREFILL, VISION, VIT, DECODE = "LLM", "prefill", "vision", "vit", "decode"
R, A, B = 7, 8, 9
POLICY = RebuildPolicy(walk=PREFILL, ids_input="text_inputs", sample_output="new_token", loop="decode_loop")
# Qwen3.5 prefills a prompt with images in one walk that also reads the encoder's output, and Bagel gives each image
# a walk of its own that reads the encoder's output alone.
MEDIA = replace(POLICY, copied_inputs={VISION: ("embeds",), VIT: ("embeds",)})
TEXT, CLOSE, IMAGE = PrefixSpan(3, 3, PREFILL), PrefixSpan(2, 2, PREFILL), PrefixSpan(6, 1, VIT, kind="digest")


def _graphs() -> dict:
    """``NODE`` reads ``text_inputs`` and outputs ``new_token``, prefilling in ``PREFILL``, in ``VISION`` with
    ``embeds`` as well or in ``VIT`` with ``embeds`` alone, and decoding in a loop."""
    def node(*inputs):
        return GraphNode(name=NODE, input_names=set(inputs), outputs=[GraphEdge("out", "new_token")])
    return {
        PREFILL: node("text_inputs"), VISION: node("text_inputs", "embeds"), VIT: node("embeds"),
        DECODE: Loop(section=node("text_inputs"), max_iters=8, outputs=[], name="decode_loop"),
    }


def _layout(*spans: PrefixSpan) -> StreamLayout:
    return StreamLayout(spans=spans, growth=(("decode_loop", 1),))


def _config(*spans: PrefixSpan, keyed: int | None = None, cached: bool = False) -> KVReqConfig:
    """A request laid out as ``spans``, the first ``keyed`` of them keyed, carrying prefix keys when ``cached``."""
    layout = replace(_layout(*spans), keyed_spans=len(spans) if keyed is None else keyed)
    return KVReqConfig(kv_layout={"main": layout}, prefix_keys={"main": [b"k"]} if cached else None)


def _on_vision(*spans: PrefixSpan) -> tuple[PrefixSpan, ...]:
    return tuple(replace(span, walk=VISION) for span in spans)


def _kept_ids(*steps: PrefillStep, sampled=()) -> KeptIds:
    return KeptIds(list(steps), array("i", sampled))


def _step(walks: dict[int, str], inputs: dict[int, dict], cached=(), rebuild=frozenset()) -> SimpleNamespace:
    """A finished step of the rows in ``walks``, where ``cached`` rows were served whole by the prefix cache."""
    rids = tuple(walks)
    ctx = StepContext(request_ids=rids, graph_walk="mixed", slot=0, capture=False, request_walks=walks)
    return SimpleNamespace(node_name=NODE, request_ids=[rid for rid in rids if rid not in cached],
                           cached_rids=set(cached), rebuild_rids=rebuild, step_context=ctx,
                           per_request_input_tensors=inputs)


def _metadata(rid: int) -> dict:
    return {"prefill_order": (rid,)}


@pytest.mark.parametrize("field, value, why", [
    ("walk", "audio", "doesn't run"),
    ("ids_input", "embeds", "carries only"),
    ("sample_output", "codes", "doesn't output"),
    ("loop", "talker_loop", "in no walk"),
    ("copied_inputs", {VISION: ("pixels",)}, "carries"),
])
def test_a_policy_naming_what_the_graph_doesnt_have_is_refused_at_load(field, value, why):
    policy = RebuildPolicy(**{**vars(MEDIA), field: value})

    with pytest.raises(ValueError, match=why):
        Rebuilds({NODE: policy}, _graphs())


def test_a_copied_walk_that_reads_the_ids_and_doesnt_output_the_sample_is_refused_at_load():
    graphs = _graphs()
    graphs[VISION].outputs = [GraphEdge("out", "hidden")]

    with pytest.raises(ValueError, match="where it reads 'text_inputs'"):
        Rebuilds({NODE: MEDIA}, graphs)


def test_a_walk_whose_node_reads_more_than_the_ids_is_refused_at_load():
    graphs = _graphs()
    graphs[PREFILL].input_names = {"text_inputs", "pixel_values"}

    with pytest.raises(ValueError, match="carries only"):
        Rebuilds({NODE: POLICY}, graphs)


def test_a_copied_walk_this_worker_doesnt_run_and_every_copied_walk_above_one_rank_is_left_out():
    policies = {NODE: replace(MEDIA, copied_inputs={VISION: ("embeds",)})}
    walks = {(NODE, walk) for walk in (PREFILL, VISION, DECODE)}

    assert policies_run_here(policies, _graphs(), walks, ()) == policies
    assert policies_run_here(policies, _graphs(), walks - {(NODE, VISION)}, ())[NODE].copied_inputs == {}, (
        "a worker that never runs the walk would keep its requests droppable with no copies to rebuild them"
    )
    assert policies_run_here(policies, _graphs(), walks, {NODE})[NODE].copied_inputs == {}, (
        "rank 0 could drop a request whose copies haven't landed on a follower"
    )


@pytest.mark.parametrize("policy, config, keeps", [
    (POLICY, _config(PrefixSpan(4, 4, PREFILL)), True),
    (MEDIA, _config(*_on_vision(TEXT, IMAGE, CLOSE)), True),
    (MEDIA, _config(TEXT, IMAGE, IMAGE, CLOSE, cached=True), True),
    (MEDIA, _config(TEXT, IMAGE, CLOSE, keyed=1), True),
    (MEDIA, _config(TEXT, IMAGE, CLOSE, keyed=1, cached=True), False),
    (POLICY, _config(TEXT, IMAGE, CLOSE), False),
    (MEDIA, _config(TEXT, replace(IMAGE, kind="rows"), CLOSE), False),
    (MEDIA, _config(), False),
], ids=[
    "ids alone", "images on one walk", "images on walks of their own", "keys short of the image, cache off",
    "keys short of the image", "walk not copied", "rows", "no prompt",
])
def test_only_a_request_whose_layout_its_rebuild_rows_can_write_again_keeps_its_prefill(policy, config, keeps):
    rebuilds = Rebuilds({NODE: policy}, _graphs())

    assert rebuilds.keep(R, NODE, [config]) == keeps, (
        "a request no rebuild can write again would be preemptible, or the reverse"
    )
    assert (R in rebuilds._accepted[NODE]) == keeps, "a request would record prefill steps no rebuild reads"


def test_each_prefill_step_is_kept_once_it_has_run_and_its_rows_share_one_copy():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    for rid in (R, A):
        rebuilds.keep(rid, NODE, [_config(TEXT)])
    inputs = {rid: {"text_inputs": [torch.arange(rid, rid + 3)]} for rid in (R, A, B)}

    rebuilds.keep_prefill(_step({R: PREFILL, A: PREFILL, B: PREFILL}, inputs), _metadata)
    rebuilds._kept[NODE][R].sampled.append(5)
    rebuilds.keep_prefill(_step({R: PREFILL}, inputs, rebuild=frozenset({R})), _metadata)  # its rebuild row
    rebuilds.keep_prefill(_step({A: DECODE}, inputs), _metadata)

    kept = rebuilds._kept[NODE]
    assert [step.ids[0].tolist() for step in kept[R].steps] == [[7, 8, 9]], "a rebuild would prefill other ids"
    assert [step.ids[0].tolist() for step in kept[A].steps] == [[8, 9, 10]], "a decode row was taken for a prompt"
    assert B not in kept, "a request no rebuild can write again would hold its ids for its life"
    storage = {kept[rid].steps[0].ids[0].untyped_storage().data_ptr() for rid in (R, A)}
    assert len(storage) == 1, "each row would pay a copy of its own"


def test_a_prompt_on_several_walks_keeps_each_step_in_order_with_copies_of_its_image():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    rebuilds.keep(R, NODE, [_config(TEXT, IMAGE, CLOSE)])
    embeds = torch.ones(6, 2)

    rebuilds.keep_prefill(_step({R: PREFILL}, {R: {"text_inputs": [torch.tensor([1, 2, 3])]}}), _metadata)
    # the prefix cache serves the image's step whole, and the step is kept all the same
    rebuilds.keep_prefill(_step({R: VIT}, {R: {"embeds": [embeds]}}, cached={R}), _metadata)
    assert "hasn't sampled" in rebuilds.refusal(R, NODE, DECODE, {}), (
        "a drop would leave the closing text out of the rebuild"
    )
    rebuilds.keep_prefill(_step({R: PREFILL}, {R: {"text_inputs": [torch.tensor([4, 5])]}}), _metadata)

    steps = rebuilds._kept[NODE][R].steps
    assert [step.walk for step in steps] == [PREFILL, VIT, PREFILL], "the rebuild would run its steps out of order"
    assert [[t.tolist() for t in step.ids] for step in steps] == [[[1, 2, 3]], [], [[4, 5]]], (
        "a row would read another step's ids"
    )
    assert steps[1].copies.inputs == {"embeds": [embeds]}, "a rebuild would need the encoder to run again"
    assert steps[1].copies.step_metadata == _metadata(R), "the image's row would run under decode's metadata"
    assert "hasn't sampled" in rebuilds.refusal(R, NODE, DECODE, {}), "the prefill's sample isn't kept yet"
    rebuilds._kept[NODE][R].sampled.append(6)
    assert rebuilds.refusal(R, NODE, DECODE, {}) is None, "a request its kept steps cover was refused"


def test_a_prompt_run_as_two_steps_of_one_walk_keeps_both_until_its_first_sample():
    # Bagel prefills a think-mode generation's text in two steps of one walk
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    rebuilds.keep(R, NODE, [_config(TEXT, CLOSE)])
    for ids in ([1, 2, 3], [4, 5]):
        rebuilds.keep_prefill(_step({R: PREFILL}, {R: {"text_inputs": [torch.tensor(ids)]}}), _metadata)
    rebuilds._kept[NODE][R].sampled.append(6)
    rebuilds.keep_prefill(_step({R: PREFILL}, {R: {"text_inputs": [torch.tensor([7])]}}), _metadata)

    steps = rebuilds._kept[NODE][R].steps
    assert [[t.tolist() for t in step.ids] for step in steps] == [[[1, 2, 3]], [[4, 5]]], (
        "the rebuild would leave out a prefill step, or rerun a step after the first sample"
    )
    assert rebuilds.refusal(R, NODE, DECODE, {}) is None, "a request whose two steps are kept was refused"


def test_a_prompt_with_images_on_one_walk_keeps_its_ids_as_the_walk_read_them():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    rebuilds.keep(R, NODE, [_config(*_on_vision(TEXT, IMAGE, CLOSE))])
    inputs = {R: {"text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.zeros(6, 2)]}}
    metadata = _metadata(R)

    rebuilds.keep_prefill(_step({R: VISION}, inputs), lambda rid: metadata)

    (step,) = rebuilds._kept[NODE][R].steps
    assert [t.tolist() for t in step.ids] == [[1, 2, 3], [4, 5]], "the rebuild row would read other text"
    assert step.copies.step_metadata == metadata and step.copies.step_metadata is not metadata, (
        "a later step could rewrite the copy"
    )


def test_a_rebuild_runs_each_prefill_step_again_in_order_and_the_last_carries_the_samples():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())
    embeds = torch.ones(6, 2)
    copies = HostCopies({"embeds": [embeds]}, _metadata(R))
    rebuilds._kept[NODE][R] = _kept_ids(
        PrefillStep(PREFILL, (torch.tensor([1, 2, 3]),)), PrefillStep(VIT, (), copies),
        PrefillStep(PREFILL, (torch.tensor([4, 5]),)), sampled=[6, 7],
    )

    rebuild, first = rebuilds.rebuild(R, NODE, torch.device("cpu"))
    rows = [first]
    while (row := rebuilds.next_row(R, NODE)) is not None:
        rows.append(row)

    assert rebuild == Rebuild(PREFILL, array("i", range(1, 8))), "the KV would lay out other ids"
    assert [row.walk for row in rows] == [PREFILL, VIT, PREFILL], "a row would write its span in another's place"
    texts = [[t.tolist() for t in row.inputs.get("text_inputs", ())] for row in rows]
    assert texts == [[[1, 2, 3]], [], [[4, 5, 6]]], "the samples would follow another step, or decode read one twice"
    assert rows[1].inputs == {"embeds": [embeds]} and rows[1].step_metadata is copies.step_metadata, (
        "the image's row would run without its copy, or under another step's metadata"
    )
    assert rows[0].step_metadata is None and rows[2].step_metadata is None, "a row of ids alone runs as decode does"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_host_copy_lands_whole_before_its_request_can_be_dropped():
    side = torch.cuda.Stream()
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), side_stream=lambda: side)
    rebuilds.keep(R, NODE, [_config(*_on_vision(TEXT))])
    rows = 1 << 22
    embeds = torch.arange(1, 1 + rows, dtype=torch.float32, device="cuda")
    expected = embeds.cpu()
    with side:
        # a cached block for the copy back, since allocating a block waits for the device
        torch.full((rows,), -1.0, device="cuda")
    step = _step({R: VISION}, {R: {"text_inputs": [torch.tensor([1, 2, 3], device="cuda")], "embeds": [embeds]}})
    # a busy side stream keeps the copy queued past the point where the worker lets go of the step's tensors
    with side:
        torch.cuda._sleep(1 << 30)
    rebuilds.keep_prefill(step, _metadata)
    rebuilds._kept[NODE][R].sampled.append(5)

    assert "haven't landed" in rebuilds.refusal(R, NODE, DECODE, {}), "the request could be dropped before its copy"
    del step, embeds
    torch.full((rows,), -1.0, device="cuda")
    torch.cuda.synchronize()
    copies = rebuilds._kept[NODE][R].steps[0].copies
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


def _prompts(*rids: int) -> Rebuilds:
    """Rebuilds whose ``rids`` each kept a prompt of one step."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    for rid in rids:
        rebuilds.keep(rid, NODE, [_config(TEXT)])
        rebuilds._kept[NODE][rid] = _kept_ids(PrefillStep(PREFILL, (torch.arange(3),)))
    return rebuilds


@pytest.mark.parametrize("sampled", [[4, 5, 6], [[4], [5], [6]]])  # the sampler's [B] output, and a [B, 1] one
def test_only_a_rows_kept_sample_is_appended(sampled):
    rebuilds = _prompts(R, A, B)
    # A's sampler throws its sample away, and B is a rebuild row
    batch = _batch((R, A, B), rebuild=frozenset({B}), kept_rids=frozenset({R, B}))

    rebuilds.keep_samples(batch, {}, HostRows((R, A, B), {"new_token": torch.tensor(sampled)}))
    rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([9])]}}, None)

    kept = rebuilds._kept[NODE]
    assert list(kept[R].sampled) == [4, 9], "a rebuild would prefill other ids than the request's"
    assert not (kept[A].sampled or kept[B].sampled), "it kept a discarded sample"


@pytest.mark.parametrize("host", [True, False])
def test_a_sample_of_more_than_one_id_is_refused(host):
    rebuilds = _prompts(R)
    codes = torch.tensor([[4, 5, 6]])  # as a talker samples several codebooks

    with pytest.raises(ValueError, match="hold 3 ids"):
        if host:
            rebuilds.keep_samples(_batch((R,)), {}, HostRows((R,), {"new_token": codes}))
        else:
            rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [codes[0]]}}, None)


@pytest.mark.parametrize("why, node, walk, sampled, iters", [
    ("declares no rebuild policy", "vocoder", DECODE, 1, 0),
    ("keeps no ids", NODE, DECODE, None, 0),
    ("hasn't sampled", NODE, DECODE, 0, 0),
    ("doesn't run", NODE, PREFILL, 1, 0),
    ("don't cover its KV", NODE, DECODE, 1, 1),
])
def test_a_drop_is_refused_unless_the_kept_ids_cover_the_kv(why, node, walk, sampled, iters):
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    rebuilds.keep(R, NODE, [_config(TEXT)])
    prompt = PrefillStep(PREFILL, (torch.tensor([1, 2]),))
    if sampled is not None:
        rebuilds._kept[NODE][R] = _kept_ids(prompt, sampled=[5] * sampled)

    assert why in rebuilds.refusal(R, node, walk, {"decode_loop": iters})
    rebuilds._kept[NODE][R] = _kept_ids(prompt, sampled=[5, 6])
    assert rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 1}) is None, "a request its ids cover was refused"


def test_a_rebuild_writes_the_prompt_and_samples_and_its_row_reads_all_but_the_last():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    rebuilds._kept[NODE][R] = _kept_ids(PrefillStep(PREFILL, (torch.tensor([1, 2, 3]),)), sampled=[4, 5])

    rebuild, row = rebuilds.rebuild(R, NODE, torch.device("cpu"))

    assert rebuild == Rebuild(PREFILL, array("i", [1, 2, 3, 4, 5])), "the KV would lay out other ids"
    assert (row.node, row.walk) == (NODE, PREFILL) and list(row.inputs) == ["text_inputs"]
    [tokens] = row.inputs["text_inputs"]
    assert tokens.tolist() == [1, 2, 3, 4] and tokens.dtype == torch.long, "decode would read the last id twice"
    assert rebuilds.next_row(R, NODE) is None, "a prompt of one step would have a second row"


def test_the_engine_keeps_no_sample_of_a_rebuild_row_whatever_the_model_declared():
    for kept_rids in (None, frozenset({R, A})):
        step = SubmoduleStep(segments=[], steps={"sampler": SamplerStep(kept_rids=kept_rids)})

        discard_samples(step, [R, A], frozenset({R}))

        assert step.steps["sampler"].kept_rids == {A}, "the rebuild row's sample would overwrite decode's next input"
