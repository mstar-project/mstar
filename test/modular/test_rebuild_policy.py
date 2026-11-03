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

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild, SamplerStep, StepContext
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows, chunk_loop_name
from mstar.worker.host_pool import HostCopy, HostPool
from mstar.worker.rebuild import HostCopies, KeptIds, NodeWrites, Prefilling, Rebuilds, policies_run_here
from mstar.worker.worker import _LazyHostOutputs

NODE, PREFILL, VISION, DECODE, MIXED = "LLM", "prefill", "vision", "decode", "mixed"
R, A, B = 7, 8, 9
POLICY = RebuildPolicy(
    walks=frozenset({PREFILL}), ids_input="text_inputs", sample_output="new_token", loop="decode_loop",
)
# as Qwen3.5 declares its LLM: a prompt with images prefills in a walk that also reads the encoder's output
MEDIA = replace(POLICY, walks=frozenset({PREFILL, VISION}), copy_bytes_per_slot={VISION: 8})


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
    ("walks", frozenset({"audio"}), "doesn't run"),
    ("ids_input", "embeds", "carries"),
    ("sample_output", "codes", "doesn't output"),
    ("loop", "talker_loop", "in no walk"),
    ("copy_bytes_per_slot", {}, "doesn't say"),
])
def test_a_policy_naming_what_the_graph_doesnt_have_is_refused_at_load(field, value, why):
    policy = RebuildPolicy(**{**vars(MEDIA), field: value})

    with pytest.raises(ValueError, match=why):
        Rebuilds({NODE: policy}, _graphs())


def test_a_walk_whose_node_reads_more_than_the_ids_must_say_what_its_copies_take():
    graphs = _graphs()
    graphs[PREFILL].input_names = {"text_inputs", "pixel_values"}

    with pytest.raises(ValueError, match="doesn't say"):
        Rebuilds({NODE: POLICY}, graphs)


def test_the_inputs_a_walk_copies_are_read_off_the_graph():
    """The policy names the walks alone. A list of copied inputs beside the graph's own went stale when the LLM
    stopped reading the image grids as an input, and every Qwen3.5 deployment failed its boot check."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs())

    assert rebuilds._copied[NODE] == {PREFILL: frozenset(), VISION: frozenset({"embeds"})}, (
        "the worker would copy other inputs than the ones the walk reads"
    )


def test_a_copied_walk_this_worker_doesnt_run_and_every_copied_walk_above_one_rank_is_left_out():
    policies = {NODE: MEDIA}
    walks = {(NODE, walk) for walk in (PREFILL, VISION, DECODE)}

    assert policies_run_here(policies, _graphs(), walks, ()) == policies
    assert policies_run_here(policies, _graphs(), walks - {(NODE, VISION)}, ())[NODE].walks == {PREFILL}, (
        "a worker that never runs the walk would keep its requests droppable with no copies to rebuild them"
    )
    assert policies_run_here(policies, _graphs(), walks, {NODE})[NODE].walks == {PREFILL}, (
        "rank 0 could drop a request whose copies haven't landed on a follower"
    )


# text, an image and text, as one row of a walk writes them
IMAGE_CHAT = (
    PrefixSpan(3, 3, VISION), PrefixSpan(6, 4, VISION, kind="digest", opens_step=False),
    PrefixSpan(2, 2, VISION, opens_step=False),
)


@pytest.mark.parametrize("policy, spans, rebuildable", [
    (MEDIA, IMAGE_CHAT, True),
    (POLICY, IMAGE_CHAT, False),
    (MEDIA, (replace(IMAGE_CHAT[0], walk="vit"), *IMAGE_CHAT[1:]), False),
    (POLICY, (PrefixSpan(4, 4, PREFILL),), True),
], ids=["copied walk", "walk not copied", "two walks", "ids alone"])
def test_only_a_request_whose_layout_a_rebuild_can_write_again_keeps_its_ids_and_copies(policy, spans, rebuildable):
    rebuilds = Rebuilds({NODE: policy}, _graphs(), HostPool(1 << 10, torch.device("cpu")))

    kept = _ingest(rebuilds, R, spans)
    rebuilds.keep_prompts(_ran(spans[0].walk, {R: {"text_inputs": [torch.arange(3)], "embeds": [torch.zeros(6, 2)]}}))

    assert kept == rebuildable, "a request no rebuild can write again would be preemptible, or the reverse"
    assert (R in rebuilds._kept[NODE]) == rebuildable, (
        "a request that is never dropped would hold its ids and a host copy of its frames for its life"
    )


def test_a_video_on_a_walk_whose_inputs_are_copied_can_be_rebuilt():
    """A Qwen3.5 video's frame groups are `rows` spans on the vision walk, which reads the encoder's output for them as
    it does for an image. `rows` only says a span has no digest key, so a rebuild from the same copies writes them."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    video = (IMAGE_CHAT[0], replace(IMAGE_CHAT[1], kind="rows"), IMAGE_CHAT[2])

    assert _ingest(rebuilds, R, video), "a video on the copied walk stays guaranteed"


@pytest.mark.parametrize("model_class, attrs", [
    (Qwen3_5DenseModel, {
        "config": SimpleNamespace(max_position_embeddings=4096, hidden_size=64), "vision_config": None,
    }),
    (OrpheusModel, {"config": SimpleNamespace(max_position_embeddings=4096)}),
    (BagelModel, {
        "config": SimpleNamespace(max_position_embeddings=4096, num_timesteps=50),
        "_has_cfg_parallel": False, "_image_gen_remote_handoff": False,
    }),
    (Qwen3OmniModel, {"config": SimpleNamespace(thinker_text=SimpleNamespace(max_position_embeddings=4096))}),
], ids=["qwen3_5", "orpheus", "bagel", "qwen3_omni"])
def test_a_policy_passes_its_check_on_a_walk_that_chunks(model_class, attrs):
    """A chunkable node runs its walk inside a chunk Loop, which adds the Loop's tick to the node's inputs."""
    model = model_class.__new__(model_class)
    vars(model).update(attrs)
    graphs = model.get_graph_walk_graphs()

    for node, policy in model.rebuild_policies().items():
        assert any(chunk_loop_name(node, walk) in graphs[walk].get_loops() for walk in policy.walks), (
            "no walk runs a chunk Loop"
        )
        policy.check(node, graphs)


def test_higgs_is_rebuilt_from_its_audio_step_with_the_text_before_the_audio_and_the_encoders_output_copied():
    from test_model_kv_layouts import _higgs

    model = _higgs()
    policy = model.rebuild_policies().get("LLM")

    assert policy is not None, "every Higgs transcription would stay guaranteed"
    assert policy.check("LLM", model.get_graph_walk_graphs()) == {
        "prefill_audio": frozenset({"text_pre", "audio_embeds"}),
    }, "a rebuild row would run without the text before the audio or the encoder's output"


def test_a_higgs_transcription_can_be_dropped_where_its_llm_runs():
    from test_model_kv_layouts import _higgs

    from mstar.model.higgs_audio.config import KV_CACHE

    model = _higgs()
    out = model.process_prompt("transcribe", ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(16000 * 35)]})
    config = KVReqConfig()
    config.apply_conductor_config(
        kv_layout={"main": out.metadata["kv_layout"][KV_CACHE]["main"].row()}, loop_caps={"decode_loop": 8},
    )
    rebuilds = Rebuilds(model.rebuild_policies(), model.get_graph_walk_graphs(), HostPool(1 << 22, torch.device("cpu")))
    writes = {KV_CACHE: {"LLM": NodeWrites(frozenset({"prefill_audio", "decode"}), frozenset({"decode_loop"}))}}

    rebuildable = rebuilds.ingest_request(R, writes, {KV_CACHE: config.kv_layout.values()}, unpausable=False)

    assert rebuildable[KV_CACHE], "every Higgs transcription would stay guaranteed"


def _rows(*rids: int) -> dict:
    return {rid: {"text_inputs": [torch.arange(rid, rid + 3)]} for rid in rids}


def _ingest(rebuilds: Rebuilds, rid: int, spans: tuple[PrefixSpan, ...]) -> bool:
    """Take ``rid`` in with one label of ``spans`` grown by the decode loop, and return whether a drop on ``NODE``
    can free it."""
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, VISION, DECODE}), frozenset({"decode_loop"}))}}
    layouts = {"kv": [StreamLayout(spans=spans, growth=(("decode_loop", 1),))]}
    return rebuilds.ingest_request(rid, writes, layouts, unpausable=False)["kv"]


def _rebuilds(pool_bytes: int = 1 << 10) -> Rebuilds:
    """`Rebuilds` with ``R``, ``A`` and ``B`` taken in as requests a drop on ``NODE`` can free."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(pool_bytes, torch.device("cpu")))
    for rid in (R, A, B):
        assert _ingest(rebuilds, rid, (PrefixSpan(3, 3, PREFILL),))
    return rebuilds


def _ran(walk: str, inputs: dict, request_walks: dict | None = None) -> SimpleNamespace:
    """A step of ``walk`` that read ``inputs``, run on the host."""
    ctx = SimpleNamespace(graph_walk=walk, request_walks=request_walks or {})
    return SimpleNamespace(node_name=NODE, step_context=ctx, per_request_input_tensors=inputs, completion_event=None)


def _kept(prompt, *sampled: int) -> KeptIds:
    """Ids kept as `Rebuilds.keep_prompts` keeps them, the prompt on the host as int32."""
    view = torch.tensor(prompt, dtype=torch.int32)
    return KeptIds(HostCopy(view, None), PREFILL, torch.long, array("i", sampled))


def test_a_prompt_rows_ids_are_kept_once_on_the_host():
    rebuilds = _rebuilds()

    rebuilds.keep_prompts(_ran(PREFILL, _rows(R, A)))
    rebuilds.keep_prompts(_ran(PREFILL, {R: {"text_inputs": [torch.tensor([0])]}}))  # its rebuild row
    rebuilds.keep_prompts(_ran(MIXED, _rows(A, B), {A: PREFILL, B: DECODE}))
    rebuilds.keep_prompts(_ran(DECODE, _rows(B)))
    rebuilds.keep_prompts(_ran(PREFILL, {B: {}}))  # a TP follower's unread row

    kept = rebuilds._kept[NODE]
    assert kept[R].prompt.view.tolist() == [7, 8, 9] and kept[A].prompt.view.tolist() == [8, 9, 10], (
        "it kept another row's ids"
    )
    assert kept[R].dtype == torch.long, "the rebuild row would read its ids in another dtype than the walk's"
    assert B not in kept, "a decode row's input, or an unread row's, would be taken for its prompt"


def test_a_request_the_full_pool_cant_hold_is_not_rebuildable():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(16, torch.device("cpu")))

    assert _ingest(rebuilds, A, (PrefixSpan(4, 4, PREFILL),)), "a request the pool holds would stay guaranteed"
    assert not _ingest(rebuilds, R, (PrefixSpan(2, 2, PREFILL),)), "two requests' ids would share host memory"
    rebuilds.keep_prompts(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}, R: {"text_inputs": [torch.arange(2)]}}))
    assert A in rebuilds._kept[NODE] and R not in rebuilds._kept[NODE]


def test_removing_a_request_gives_its_slice_back():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(16, torch.device("cpu")))
    _ingest(rebuilds, R, (PrefixSpan(4, 4, PREFILL),))

    rebuilds.remove_request(R)

    assert _ingest(rebuilds, A, (PrefixSpan(4, 4, PREFILL),)), (
        "the pool would fill with removed requests' ids and keep no more"
    )


@pytest.mark.parametrize("why", ["its output can't pause", "its pool can't be rebuilt"])
def test_a_request_no_drop_can_free_keeps_no_ids(why):
    """A guaranteed request kept a host copy of its prompt and appended every sample, each step, though no drop
    could ever use them."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    # a prompt written in a walk the policy doesn't prefill is one no rebuild can write again
    walk = PREFILL if why == "its output can't pause" else "audio"
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE, "audio"}), frozenset({"decode_loop"}))}}
    layouts = {"kv": [StreamLayout(spans=(PrefixSpan(3, 3, walk),), growth=(("decode_loop", 1),))]}
    rebuilds.ingest_request(R, writes, layouts, unpausable=why == "its output can't pause")

    rebuilds.keep_prompts(_ran(PREFILL, _rows(R)))
    rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([9])]}}, None)

    assert R not in rebuilds._kept[NODE], f"a request kept its ids though {why}"


def _media(prompt=(1,), *sampled: int, copies: HostCopies | None = None) -> KeptIds:
    return replace(_kept(prompt, *sampled), walk=VISION, copies=HostCopies() if copies is None else copies)


def _copying(*rids: int) -> Rebuilds:
    """`Rebuilds` whose ``rids`` keep their ids from ``VISION`` and wait for their copies, each with a slice."""
    pool = HostPool(1 << 10, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool)
    for rid in rids:
        rebuilds._kept[NODE][rid], rebuilds._droppable[NODE][rid] = _media(), pool.take(64)
    return rebuilds


def test_a_prompt_with_images_keeps_its_ids_as_its_walk_read_them_and_waits_for_its_copies():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    _ingest(rebuilds, R, IMAGE_CHAT)
    rows = {R: {"text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.zeros(6, 2)]}}

    rebuilds.keep_prompts(_ran(VISION, rows))

    kept = rebuilds._kept[NODE][R]
    assert kept.prompt.view.tolist() == [1, 2, 3, 4, 5] and kept.lengths == (3, 2), (
        "the rebuild row would read other text"
    )
    assert kept.walk == VISION and kept.copies == HostCopies(), "a rebuild row would leave out the image"
    kept.sampled.append(6)
    assert "haven't landed" in rebuilds.refusal(R, NODE, DECODE, {}, None), (
        "a drop would leave nothing to rebuild the image"
    )


def _vision_step(rids, walks: dict, **fields) -> SimpleNamespace:
    infos = {rid: SimpleNamespace(graph_walk=walks.get(rid, VISION), step_metadata={"prefill_order": (rid,)})
             for rid in rids}
    inputs = {rid: {"text_inputs": [torch.tensor([rid])], "embeds": [torch.full((2,), rid)]} for rid in rids}
    ctx = StepContext(request_ids=tuple(rids), graph_walk="mixed", slot=0, capture=False,
                      request_walks={rid: info.graph_walk for rid, info in infos.items()})
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), per_request_info=infos,
                           per_request_input_tensors=inputs, step_context=ctx, **fields)


def test_a_step_of_no_copied_walk_copies_nothing_and_reads_no_row():
    rebuilds = _copying(R)
    ctx = StepContext(request_ids=(R,), graph_walk=DECODE, slot=0, capture=False)

    # the step carries no rows' inputs, so reading one would raise
    rebuilds.keep_copies(SimpleNamespace(node_name=NODE, request_ids=[R], step_context=ctx))

    assert rebuilds._kept[NODE][R].copies.inputs is None, "a decode step would be taken for the image's step"


def test_a_prefill_row_keeps_a_host_copy_of_what_it_read_besides_its_ids_and_of_its_step_metadata():
    rebuilds = _copying(R, A)
    rebuilds._kept[NODE][B] = _kept([1])
    step = _vision_step((R, A, B), {A: DECODE}, completion_event=None)

    rebuilds.keep_copies(step)

    copies = rebuilds._kept[NODE][R].copies
    assert list(copies.inputs) == ["embeds"] and copies.inputs["embeds"][0].tolist() == [R, R], (
        "a rebuild would need the encoder to run again"
    )
    assert copies.step_metadata == {"prefill_order": (R,)}, "the rebuild row would run under decode's metadata"
    assert copies.step_metadata is not step.per_request_info[R].step_metadata, "a later step could rewrite the copy"
    assert copies.ready(), "the request could never be dropped"
    assert rebuilds._kept[NODE][A].copies.inputs is None, "a row of another walk would stand in for the prefill"


def test_a_request_with_images_takes_room_for_its_copies_when_it_arrives_and_gives_back_what_they_didnt_use():
    """Every image request pinned its encoder output in blocks of its own for its whole life, and nothing capped the
    total, so a failed pinned allocation failed every request in its step. The log takes one slice of the startup
    pool when a request arrives, sized by the policy's bytes per slot, so a request the pool can't hold is known
    before it is admitted and stays out of the drops."""
    pool = HostPool(160, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool)

    # five ids at 4 B, aligned to 24, and the image's six slots at 8 B
    assert _ingest(rebuilds, R, IMAGE_CHAT) and rebuilds._droppable[NODE][R].nbytes == 24 + 48
    assert _ingest(rebuilds, A, IMAGE_CHAT)
    assert not _ingest(rebuilds, B, IMAGE_CHAT), "a request the full pool can't hold would be admitted as droppable"
    step = _vision_step((R,), {}, completion_event=None)
    # the image's four rows, without its two sentinels
    step.per_request_input_tensors[R] = {
        "text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.ones(4, 2)],
    }
    rebuilds.keep_prompts(step)
    rebuilds.keep_copies(step)

    assert rebuilds._droppable[NODE][R].nbytes == 24 + 32, "the sentinels' room would stay taken for the request's life"
    assert rebuilds._kept[NODE][R].copies.inputs["embeds"][0].tolist() == [[1.0, 1.0]] * 4


def test_a_rebuild_of_a_prompt_with_images_reads_its_copies_under_its_prefills_metadata():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    embeds, metadata = torch.ones(6, 2), {"prefill_order": ("text", "image", "text")}
    copies = HostCopies({"embeds": [embeds]}, metadata)
    rebuilds._kept[NODE][R] = replace(_media([1, 2, 3, 4, 5], 6, 7, copies=copies), lengths=(3, 2))

    rebuild = rebuilds.start(R, NODE, None)
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {}, inputs, infos)

    assert rebuild == Rebuild(array("i", [6, 7])) and infos[R].graph_walk == VISION
    assert [t.tolist() for t in inputs[R]["text_inputs"]] == [[1, 2, 3], [4, 5, 6]], (
        "the samples would follow the image instead of the prompt's last text, or decode would read the last id twice"
    )
    assert inputs[R]["embeds"] == [embeds] and infos[R].step_metadata is metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_host_copy_lands_whole_before_its_request_can_be_dropped():
    side = torch.cuda.Stream()
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), side_stream=lambda: side)
    rebuilds._kept[NODE][R] = _media([1], 5)
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

    assert "haven't landed" in rebuilds.refusal(R, NODE, DECODE, {}, None), (
        "the request could be dropped before its copy"
    )
    del step, embeds
    torch.full((rows,), -1.0, device="cuda")
    torch.cuda.synchronize()
    copies = rebuilds._kept[NODE][R].copies
    assert rebuilds.refusal(R, NODE, DECODE, {}, None) is None
    assert torch.equal(copies.inputs["embeds"][0], expected), "the next step's tensor took the memory before the copy"

    with side:
        torch.cuda._sleep(1 << 30)
    on_device = copies.to_device(torch.device("cuda"), side)
    assert torch.equal(on_device["embeds"][0].cpu(), expected), "the rebuild row would read its inputs before they land"


def _batch(rids, rebuild=frozenset()) -> SimpleNamespace:
    ctx = StepContext(request_ids=tuple(rids), graph_walk=DECODE, slot=0, capture=False, rebuild_rids=rebuild)
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), step_context=ctx)


@pytest.mark.parametrize("sampled", [[4, 5, 6], [[4], [5], [6]]])  # the sampler's [B] output, and a [B, 1] one
def test_only_a_rows_kept_sample_is_appended(sampled):
    rebuilds = _rebuilds()
    rebuilds.keep_prompts(_ran(PREFILL, _rows(R, A, B)))
    # A runs a chunk short of its prompt's end, and B is a rebuild row
    batch = _batch((R, A, B), rebuild=frozenset({B}))

    rebuilds.keep_samples(batch, {}, HostRows((R, A, B), {"new_token": torch.tensor(sampled)}), skip_rids={A})
    rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([9])]}}, None)

    kept = rebuilds._kept[NODE]
    assert list(kept[R].sampled) == [4, 9], "a rebuild would prefill other ids than the request's"
    assert not (kept[A].sampled or kept[B].sampled), "it kept a discarded sample"


def test_a_sample_is_kept_from_the_host_outputs_the_worker_copies_on_read():
    rebuilds = _rebuilds()
    rebuilds.keep_prompts(_ran(PREFILL, _rows(R)))
    row = {"new_token": [torch.tensor([9])]}
    outputs = _LazyHostOutputs({R: row}, lambda names: {name: {R: row[name]} for name in names})

    rebuilds.keep_samples(_batch((R,)), outputs, None)

    assert list(rebuilds._kept[NODE][R].sampled) == [9], "every drop would be refused for the samples it missed"


@pytest.mark.parametrize("host", [True, False])
def test_a_sample_of_more_than_one_id_is_refused(host):
    rebuilds = _rebuilds()
    rebuilds.keep_prompts(_ran(PREFILL, _rows(R)))
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
    rebuilds = _rebuilds()
    if sampled is not None:
        rebuilds._kept[NODE][R] = _kept([1, 2], *[5] * sampled)

    assert why in rebuilds.refusal(R, node, walk, {"decode_loop": iters}, None)
    rebuilds._kept[NODE][R] = _kept([1, 2], 5, 6)
    assert rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 1}, None) is None, "a request its ids cover was refused"


def test_a_drop_is_refused_until_the_kept_ids_reach_the_host():
    rebuilds = _rebuilds()
    prompt = torch.tensor([1, 2], dtype=torch.int32)
    copying = HostCopy(prompt, SimpleNamespace(query=lambda: False))
    rebuilds._kept[NODE][R] = KeptIds(copying, PREFILL, torch.long, array("i", [5]))

    assert "haven't reached the host" in rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 0}, None), (
        "a rebuild would read ids the copy hasn't written yet"
    )


def test_a_rebuild_writes_the_prompt_and_samples_and_its_row_reads_all_but_the_last():
    rebuilds = _rebuilds()
    rebuilds._kept[NODE][R] = _kept([1, 2, 3], 4, 5)

    rebuild = rebuilds.start(R, NODE, None)
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {}, inputs, infos)

    assert rebuild == Rebuild(array("i", [4, 5])), "the KV would lay out other ids"
    assert infos[R].graph_walk == PREFILL and list(inputs[R]) == ["text_inputs"]
    [tokens] = inputs[R]["text_inputs"]
    assert tokens.tolist() == [1, 2, 3, 4] and tokens.dtype == torch.long, "decode would read the last id twice"


def test_a_request_mid_prefill_is_rebuilt_from_its_prompts_input_up_to_where_its_prompt_stood():
    rebuilds = _rebuilds()
    prefilling = Prefilling(PREFILL, 4, 10)

    why = rebuilds.refusal(R, NODE, PREFILL, {}, prefilling)
    rebuild = rebuilds.start(R, NODE, prefilling)
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), PREFILL, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {R: ChunkRange(0, 4, 4, 4)}, inputs, infos)

    assert why is None, "the newest request would keep its pages while an older one drops itself"
    assert rebuild is None and inputs == {R: {}}, "the rebuild would lay out and read ids the request never kept"
    cursor = rebuilds.cursor(R, NODE)
    assert (cursor.walk, cursor.total, cursor.length, cursor.resumes_prefill) == (PREFILL, 4, 10, True), (
        "the rebuild would run past where the prompt stood, or release the input the prompt's next chunk reads"
    )


def test_the_sampler_commits_no_draw_of_a_rebuild_row_whatever_the_model_declared():
    ctx = StepContext(request_ids=(R, A), graph_walk=PREFILL, slot=0, capture=False, rebuild_rids=frozenset({R}))
    for kept_rids in (None, frozenset({R, A})):
        kept = SamplerResource._kept_rids_of(SamplerStep(kept_rids=kept_rids), ctx)

        assert kept == {A}, "the rebuild row's sample would overwrite decode's next input"
