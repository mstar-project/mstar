"""A node declares how its dropped requests are rebuilt, and the worker keeps what that rebuild reads.

The declaration names the walk that re-prefills a request, the input that carries its ids, the output that carries
each sample and the loop that samples. Nothing about a rebuild is inferred from a graph's shape or a KV layout, so a
model either says how its requests come back or never has one dropped. The worker keeps each request's prompt ids
from the policy's input the first time a step runs the policy's walk, and each sample from the policy's output.
A walk that reads more than ids, such as one that prefills a prompt with images, names those inputs, and the worker
keeps host copies of them once the request's prefill row has run.
"""

from __future__ import annotations

import logging
from array import array
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_kv_drop as kv_drop
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild, SamplerStep, StepAllowance, StepContext
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import Segment
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.multimodal import vision_encoder
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows, chunk_loop_name
from mstar.worker.host_copies import _LazyHostOutputs
from mstar.worker.host_pool import HostCopy, HostPool, HostSlice, aligned
from mstar.worker.rebuild import HostCopies, KeptIds, LogRoom, NodeWrites, Prefilling, Rebuilds, policies_run_here

NODE, PREFILL, VISION, DECODE, MIXED = "LLM", "prefill", "vision", "decode", "mixed"
R, A, B = 7, 8, 9
POLICY = RebuildPolicy(
    walks=frozenset({PREFILL}), ids_input="text_inputs", sample_output="new_token", loop="decode_loop",
)
# as Qwen3.5 declares its LLM: a prompt with images prefills in a walk that also reads the encoder's output
MEDIA = replace(POLICY, walks=frozenset({PREFILL, VISION}), copy_width={VISION: 2})
# what ``NODE`` runs in, so a slot of MEDIA's copies takes 8 B
DTYPES = {NODE: torch.float32}


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
    ("copy_width", {}, "doesn't say"),
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
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance())

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
    rebuilds = Rebuilds({NODE: policy}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)

    kept = _ingest(rebuilds, R, spans)
    ids = torch.arange(sum(span.length for span in spans if span.kind == "ids"))
    rebuilds.keep_prompts(_ran(spans[0].walk, {R: {"text_inputs": [ids], "embeds": [torch.zeros(6, 2)]}}))

    assert kept == rebuildable, "a request no rebuild can write again would be preemptible, or the reverse"
    assert (R in rebuilds._kept[NODE]) == rebuildable, (
        "a request that is never dropped would hold its ids and a host copy of its frames for its life"
    )


def test_a_video_on_a_walk_whose_inputs_are_copied_can_be_rebuilt():
    """A Qwen3.5 video's frame groups are `rows` spans on the vision walk, which reads the encoder's output for them as
    it does for an image. `rows` only says a span has no digest key, so a rebuild from the same copies writes them."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
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
    infos = {rid: SimpleNamespace(step_metadata={"is_prefill": True}) for rid in inputs}
    return SimpleNamespace(
        node_name=NODE, step_context=ctx, request_ids=list(inputs), per_request_input_tensors=inputs,
        per_request_info=infos, cached_rids=set(), completion_event=None,
    )


def _kept(prompt, *sampled: int) -> KeptIds:
    """Ids kept as `Rebuilds.keep_prompts` keeps them, the prompt on the host as the walk read it."""
    view = torch.tensor(prompt, dtype=torch.long)
    return KeptIds(HostCopy(view, None), PREFILL, array("i", sampled), {"is_prefill": True})


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
    assert kept[R].prompt.view.dtype == torch.long, (
        "the rebuild row would read its ids in another dtype than the walk's"
    )
    assert B not in kept, "a decode row's input, or an unread row's, would be taken for its prompt"


def test_only_the_rows_that_ran_keep_their_prompts():
    """The engine takes a row it skipped or failed out of the step's rows and their infos, but not out of its
    inputs."""
    rebuilds = _rebuilds()
    step = _ran(PREFILL, _rows(R, A, B))
    step.cached_rids.add(B)  # the prefix cache served B's walk whole
    for rid in (A, B):
        step.request_ids.remove(rid)
        del step.per_request_info[rid]

    rebuilds.keep_prompts(step)

    assert set(rebuilds._kept[NODE]) == {R}, "a row that never ran would keep a prompt"
    assert B not in rebuilds._keeps_ids[NODE], "a request the cache served would hold its slice and never use it"


def test_a_request_the_full_pool_cant_hold_is_not_rebuildable():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(32, torch.device("cpu")))

    assert _ingest(rebuilds, A, (PrefixSpan(4, 4, PREFILL),)), "a request the pool holds would stay guaranteed"
    assert not _ingest(rebuilds, R, (PrefixSpan(2, 2, PREFILL),)), "two requests' ids would share host memory"
    rebuilds.keep_prompts(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}, R: {"text_inputs": [torch.arange(2)]}}))
    assert A in rebuilds._kept[NODE] and R not in rebuilds._kept[NODE]


def test_a_full_pool_is_logged_once_and_not_per_request(caplog):
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(32, torch.device("cpu")))

    with caplog.at_level(logging.WARNING, logger="mstar.worker.rebuild"):
        for rid in (R, A, B):
            _ingest(rebuilds, rid, (PrefixSpan(8, 8, PREFILL),))

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1 and "the 64 bytes a request's ids and host copies take, with 0 of 32 bytes" in lines[0], (
        "requests that can never be dropped would go unnoticed, or log a line each"
    )


def test_removing_a_request_gives_its_slice_back():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(32, torch.device("cpu")))
    _ingest(rebuilds, R, (PrefixSpan(4, 4, PREFILL),))

    rebuilds.remove_request(R)

    assert _ingest(rebuilds, A, (PrefixSpan(4, 4, PREFILL),)), (
        "the pool would fill with removed requests' ids and keep no more"
    )


def test_a_request_takes_8_bytes_for_each_prompt_id_when_it_arrives():
    """A slice held 4 B for each prompt id, too few to keep a prompt its walk read as int64."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    assert _ingest(rebuilds, R, (PrefixSpan(5, 5, PREFILL),))
    assert rebuilds._keeps_ids[NODE][R].piece.nbytes == 40, "keeping the prompt would overflow its slice"

    rebuilds.keep_prompts(_ran(PREFILL, {R: {"text_inputs": [torch.arange(5)]}}))

    kept = rebuilds._kept[NODE][R].prompt.view
    assert kept.dtype == torch.long and kept.tolist() == [0, 1, 2, 3, 4], "the rebuild would prefill other ids"


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
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance())
    for rid in rids:
        rebuilds._kept[NODE][rid], rebuilds._keeps_ids[NODE][rid] = _media(), LogRoom(pool.take(64), 0, 0)
    return rebuilds


def test_a_prompt_with_images_keeps_its_ids_as_its_walk_read_them_and_waits_for_its_copies():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    _ingest(rebuilds, R, IMAGE_CHAT)
    rows = {R: {"text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.zeros(6, 2)]}}

    rebuilds.keep_prompts(_ran(VISION, rows))

    kept = rebuilds._kept[NODE][R]
    assert kept.prompt.view.tolist() == [1, 2, 3, 4, 5] and kept.lengths == (3, 2), (
        "the rebuild row would read other text"
    )
    assert kept.walk == VISION and kept.copies == HostCopies(6), "a rebuild row would leave out the image"
    kept.sampled.append(6)
    assert "haven't landed" in rebuilds.refusal(R, NODE, DECODE, {}, None), (
        "a drop would leave nothing to rebuild the image"
    )


def test_a_prompt_that_read_a_pad_id_at_each_image_slot_leaves_its_copies_no_slots_of_their_own():
    """A VLM's media walk reads an id at every slot, a pad where the encoder's output goes in. Counting the image's
    slots again for its copies made the rebuild write more slots than the KV held, so every drop refused it."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    _ingest(rebuilds, R, IMAGE_CHAT)
    rows = {R: {"text_inputs": [torch.arange(11)], "embeds": [torch.zeros(6, 2)]}}

    rebuilds.keep_prompts(_ran(VISION, rows))
    rebuilds._kept[NODE][R].sampled.append(6)

    assert rebuilds.tokens(R, NODE, None) == 11, "the rebuild would write other slots than the prompt's KV holds"


def test_a_prompt_whose_ids_cover_neither_its_ids_spans_nor_every_slot_keeps_none():
    """The host copies were counted as the slots the prompt's ids left, so the rebuild matched the KV whatever the
    count of ids. A wrong count passed the drop check and failed `prepare_rebuild`'s assert after the drop had freed
    the pages."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    _ingest(rebuilds, R, IMAGE_CHAT)
    rows = {R: {"text_inputs": [torch.arange(7)], "embeds": [torch.zeros(6, 2)]}}

    rebuilds.keep_prompts(_ran(VISION, rows))

    assert "keeps no ids" in rebuilds.refusal(R, NODE, DECODE, {}, None), (
        "a drop would free the pages of a request whose rebuild writes other slots than its KV holds"
    )
    assert R not in rebuilds._keeps_ids[NODE], "the request would hold its slice for its life"


def _vision_step(rids, walks: dict, **fields) -> SimpleNamespace:
    infos = {rid: SimpleNamespace(graph_walk=walks.get(rid, VISION), step_metadata={"prefill_order": (rid,)})
             for rid in rids}
    inputs = {rid: {"text_inputs": [torch.tensor([rid])], "embeds": [torch.full((2,), rid)]} for rid in rids}
    ctx = StepContext(request_ids=tuple(rids), graph_walk="mixed", slot=0, capture=False,
                      request_walks={rid: info.graph_walk for rid, info in infos.items()})
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), per_request_info=infos,
                           per_request_input_tensors=inputs, cached_rids=set(), step_context=ctx, **fields)


def test_a_step_of_no_copied_walk_copies_nothing_and_reads_no_row():
    rebuilds = _copying(R)
    ctx = StepContext(request_ids=(R,), graph_walk=DECODE, slot=0, capture=False)

    # the step carries no rows' inputs, so reading one would raise
    rebuilds.keep_copies(SimpleNamespace(node_name=NODE, request_ids=[R], step_context=ctx))

    assert rebuilds._kept[NODE][R].copies.inputs is None, "a decode step would be taken for the image's step"


def test_a_prefill_row_keeps_a_host_copy_of_what_it_read_besides_its_ids():
    rebuilds = _copying(R, A)
    rebuilds._kept[NODE][B] = _kept([1])
    step = _vision_step((R, A, B), {A: DECODE}, completion_event=None)

    rebuilds.keep_copies(step)
    rebuilds.flush()

    copies = rebuilds._kept[NODE][R].copies
    assert list(copies.inputs) == ["embeds"] and copies.inputs["embeds"][0].tolist() == [R, R], (
        "a rebuild would need the encoder to run again"
    )
    assert copies.ready(), "the request could never be dropped"
    assert rebuilds._kept[NODE][A].copies.inputs is None, "a row of another walk would stand in for the prefill"


def test_a_request_with_images_takes_room_for_its_copies_when_it_arrives_and_gives_back_what_they_didnt_use():
    """Every image request pinned its encoder output in blocks of its own for its whole life, and nothing capped the
    total, so a failed pinned allocation failed every request in its step. The log takes one slice of the startup
    pool when a request arrives, sized by the policy's bytes per slot, so a request the pool can't hold is known
    before it is admitted and stays out of the drops."""
    pool = HostPool(256, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance(), DTYPES)

    # an id at 8 B for each of the eleven slots, and the image's six slots at two float32 values each
    assert _ingest(rebuilds, R, IMAGE_CHAT) and rebuilds._keeps_ids[NODE][R] == (HostSlice(0, 88 + 48), 11, 5)
    assert not _ingest(rebuilds, A, IMAGE_CHAT), "a request the full pool can't hold would be admitted as droppable"
    step = _vision_step((R,), {}, completion_event=None)
    # the image's four rows, without its two sentinels
    step.per_request_input_tensors[R] = {
        "text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.ones(4, 2)],
    }
    rebuilds.keep_prompts(step)
    rebuilds.keep_copies(step)
    rebuilds.flush()

    assert rebuilds._keeps_ids[NODE][R].piece.nbytes == 40 + 32, (
        "the image's room for ids and its sentinels' room would stay taken for the request's life"
    )
    assert rebuilds._kept[NODE][R].copies.inputs["embeds"][0].tolist() == [[1.0, 1.0]] * 4


def test_a_media_copy_past_the_steps_allowance_waits_for_the_next_steps():
    """The image copies queued every input of every prefill row at once, behind which the next step's output copy
    waited on the device's one copy engine. The image copies share one allowance with the cached pages' writes, and
    go first."""
    allowance = StepAllowance(24)
    rebuilds = _copying(R)
    rebuilds._allowance = allowance
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.arange(8, dtype=torch.float32)]

    rebuilds.keep_copies(step)
    rebuilds.flush()

    assert allowance.left == 0, "the step's copies would run past its allowance"
    assert not rebuilds._kept[NODE][R].copies.ready(), "the request could be dropped with part of its image copied"
    allowance.refill()
    rebuilds.flush()
    copies = rebuilds._kept[NODE][R].copies
    assert copies.ready() and copies.inputs["embeds"][0].tolist() == list(range(8)), "the split copy lost bytes"
    assert allowance.left == 24 - 8, "the cached pages' writes would get less than the copies left them"


class _Landing:
    """A copy's event that lands only once it is waited on."""

    def __init__(self) -> None:
        self.landed = False

    def query(self) -> bool:
        return self.landed

    def synchronize(self) -> None:
        self.landed = True


class _LandingPool(HostPool):
    """Records, at each copy to the host it starts, whether a copy it started earlier is still in flight."""

    def __init__(self) -> None:
        super().__init__(1 << 10, torch.device("cpu"))
        self.events: list[_Landing] = []
        self.overlapped: list[bool] = []

    def copy_out(self, src, into, after):
        self.overlapped.append(not all(event.landed for event in self.events))
        self.events.append(event := _Landing())
        return super().copy_out(src, into, None)._replace(landed=event)


def test_passes_that_run_no_step_start_one_allowance_at_a_time():
    """Each pass of the worker's loop that ran no step refilled the allowance, so a burst of passes, as a burst of
    messages makes, queued several allowances of copies at once behind the next step's output copy."""
    pool = _LandingPool()
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance(8))
    rebuilds._kept[NODE][R], rebuilds._keeps_ids[NODE][R] = _media(), LogRoom(pool.take(64), 0, 0)
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.arange(8, dtype=torch.float32)]
    rebuilds.keep_copies(step)

    for _ in range(4):
        rebuilds.flush_idle()

    assert pool.overlapped == [False] * 4, "a pass started copies while the last allowance's were still in flight"


def test_a_node_that_copies_to_the_host_needs_the_engines_allowance():
    """`Rebuilds` built an allowance of its own when none was passed, so a caller that left out the engine's gave the
    copy engine a second budget."""
    with pytest.raises(AssertionError, match="StepAllowance"):
        Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")))


def test_an_empty_input_queues_no_copy_and_holds_back_none_behind_it():
    """`flush` took a copy off the queue only once it spent some of the step's allowance. A row that read an empty
    tensor queued a copy that spent nothing, so it stayed at the head of the queue and no copy behind it ever
    started."""
    rebuilds = _copying(R)
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.zeros(0, 2), torch.ones(2, 2)]

    rebuilds.keep_copies(step)
    rebuilds.flush()

    copies = rebuilds._kept[NODE][R].copies
    assert copies.ready(), "the request's image would never land, so it could never be dropped"
    assert [view.shape for view in copies.inputs["embeds"]] == [(0, 2), (2, 2)], (
        "the rebuild row would read other inputs than its prefill row"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_media_copy_split_across_two_steps_lands_whole():
    pool = HostPool(1 << 22, torch.device("cuda"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, allowance=(allowance := StepAllowance(1 << 20)))
    rebuilds._kept[NODE][R], rebuilds._keeps_ids[NODE][R] = _media([1]), LogRoom(pool.take(1 << 21), 0, 0)
    embeds = torch.arange(3 << 17, dtype=torch.float32, device="cuda")
    step_done = torch.Event(device="cuda")
    step_done.record()
    step = _vision_step((R,), {}, completion_event=step_done)
    step.per_request_input_tensors[R]["embeds"] = [embeds]

    rebuilds.keep_copies(step)
    rebuilds.flush()
    allowance.refill()
    rebuilds.flush()
    del step
    torch.cuda.synchronize()

    copies = rebuilds._kept[NODE][R].copies
    assert copies.ready() and torch.equal(copies.inputs["embeds"][0], embeds.cpu()), "the second half never landed"


def test_a_rebuild_of_ids_alone_runs_under_its_prefills_metadata():
    """A row of ids alone ran under the request's decode step's metadata and a row with images under its prefill's,
    so the model and the sampler saw the two differently. Every kept request records a copy of its prefill row's
    metadata, since a later step may change the live one."""
    rebuilds = _rebuilds()
    prefill = _ran(PREFILL, _rows(R))
    rebuilds.keep_prompts(prefill)
    prefill.per_request_info[R].step_metadata["is_prefill"] = False
    rebuilds._kept[NODE][R].sampled.append(4)

    rebuilds.start(R, NODE, None)
    infos = {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    infos[R].step_metadata = {"is_prefill": False}
    rebuilds.fill(NODE, {R}, {}, {R: {}}, infos)

    assert infos[R].step_metadata == {"is_prefill": True}, "a row of ids alone runs under the request's decode metadata"


def test_rebuild_tokens_count_the_slots_a_rebuild_runs():
    """`rebuild_tokens` counted only text ids, so it left out the image slots the rebuild recomputes, and it counted
    a cached prefix the rebuild never ran."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    _ingest(rebuilds, R, IMAGE_CHAT)
    rebuilds.keep_prompts(_ran(VISION, {R: {"text_inputs": [torch.arange(3), torch.arange(2)]}}))
    rebuilds._kept[NODE][R].sampled.append(6)
    rebuilds._kept[NODE][R].copies.inputs = {}

    rebuilds.start(R, NODE, None)
    image = rebuilds.land_chunk(R, NODE, None)

    assert image.ran == 3 + 6 + 2, f"the counter says {image.ran} tokens for a rebuild that recomputes 11 slots"
    rebuilds = _rebuilds()
    rebuilds._kept[NODE][R] = _kept(range(9), 9)
    rebuilds.start(R, NODE, None)
    # the first 4 tokens were a cached prefix
    rebuilds.land_chunk(R, NODE, ChunkRange(4, 6, 9, 6))
    text = rebuilds.land_chunk(R, NODE, ChunkRange(6, 9, 9, 3))
    assert text.ran == 5, "the counter would count the cached prefix the rebuild skipped"


class _CountingPool(HostPool):
    """Counts the copies back to the device."""

    def __init__(self) -> None:
        super().__init__(1 << 10, torch.device("cpu"))
        self.copied_in = 0

    def copy_in(self, view):
        self.copied_in += 1
        return view.clone(), SimpleNamespace()


def test_the_copy_back_waits_for_the_step_that_reads_it():
    """The copy back to the device ran when the rebuild was queued and made the current stream wait then, so a step
    still being enqueued stalled behind it, and the copy held device memory outside the KV pool while the row
    waited. The copy back runs when the row's step is built, and only that step waits for it."""
    pool = _CountingPool()
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance())
    copies = HostCopies(inputs={"embeds": [torch.ones(6, 2)]})
    rebuilds._kept[NODE][R] = replace(_media([1, 2, 3, 4, 5], 6, copies=copies), lengths=(3, 2))

    rebuilds.start(R, NODE, None)
    assert not pool.copied_in, "starting the rebuild copied the image back to the device before any step reads it"
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    ready = rebuilds.fill(NODE, {R}, {}, inputs, infos)

    assert pool.copied_in == 1 and len(ready) == 1, "the row's step wouldn't wait for its image"


def test_a_request_with_images_is_not_rebuildable_on_a_node_whose_dtype_is_unknown():
    """A slice sized for bf16 copies overflowed on an fp32 deployment, and the request was left preemptible with no
    way back. The log sizes copies in the dtype the node runs in, and a node that keeps its own dtype copies nothing."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance())

    assert not _ingest(rebuilds, R, IMAGE_CHAT), "a request whose copies can't be sized would be droppable"
    assert _ingest(rebuilds, A, (PrefixSpan(4, 4, VISION),)), "a prompt of ids alone would lose its rebuild"


def test_a_request_removed_with_copies_queued_leaves_the_next_owner_of_its_slice_alone():
    allowance, pool = StepAllowance(0), HostPool(64, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, allowance)
    rebuilds._kept[NODE][R], rebuilds._keeps_ids[NODE][R] = _media(), LogRoom(pool.take(64), 0, 0)
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.full((4,), 7.0)]
    rebuilds.keep_copies(step)

    rebuilds.remove_request(R)
    piece = pool.take(64)
    pool.view(piece, torch.uint8, torch.Size([64])).fill_(1)
    allowance.refill()
    rebuilds.flush()

    assert pool.view(piece, torch.uint8, torch.Size([64])).tolist() == [1] * 64, (
        "a removed request's queued copy wrote into the slice its next owner holds"
    )


def test_a_rebuild_of_a_prompt_with_images_reads_its_copies_under_its_prefills_metadata():
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    embeds, metadata = torch.ones(6, 2), {"prefill_order": ("text", "image", "text")}
    copies = HostCopies(inputs={"embeds": [embeds]})
    rebuilds._kept[NODE][R] = replace(
        _media([1, 2, 3, 4, 5], 6, 7, copies=copies), lengths=(3, 2), step_metadata=metadata,
    )

    rebuild = rebuilds.start(R, NODE, None)
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {}, inputs, infos)

    assert rebuild == Rebuild(VISION, array("i", range(1, 8))) and infos[R].graph_walk == VISION
    assert [t.tolist() for t in inputs[R]["text_inputs"]] == [[1, 2, 3], [4, 5, 6]], (
        "the samples would follow the image instead of the prompt's last text, or decode would read the last id twice"
    )
    assert inputs[R]["embeds"] == [embeds] and infos[R].step_metadata is metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_host_copy_lands_whole_before_its_request_can_be_dropped():
    side = torch.cuda.Stream()
    pool = HostPool(1 << 25, torch.device("cuda"))
    # the pool's copies run on ``side``, so a busy ``side`` holds them back
    pool._stream = pool._in_stream = side
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance())
    rows = 1 << 22
    # room for the prompt's one id and the copy
    room = LogRoom(pool.take(aligned(8) + 4 * rows), 0, 0)
    rebuilds._kept[NODE][R], rebuilds._keeps_ids[NODE][R] = _media([1], 5), room
    embeds = torch.arange(1, 1 + rows, dtype=torch.float32, device="cuda")
    expected = embeds.cpu()
    with side:
        # a cached block for the copy back, since allocating a block waits for the device
        torch.full((rows,), -1.0, device="cuda")
    step_done = torch.Event(device="cuda")
    step_done.record()
    step = _vision_step((R,), {}, completion_event=step_done)
    step.per_request_input_tensors[R]["embeds"] = [embeds]
    # a busy side stream keeps the copy queued past the point where the worker lets go of the step's tensors
    with side:
        torch.cuda._sleep(1 << 30)
    rebuilds.keep_copies(step)
    rebuilds.flush()

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
    on_device, ready = pool.copy_in(copies.inputs["embeds"][0])
    torch.cuda.current_stream().wait_event(ready)
    assert torch.equal(on_device.cpu(), expected), "the rebuild row would read its inputs before they land"


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


def test_a_request_that_samples_a_pad_id_is_never_dropped():
    """A rebuild row's sampled ids run in the packed media step, where the nth pad reads the nth encoder row. A
    sampled pad took the row of the next pad in the step, and every row after it read the wrong encoder rows."""
    model = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
    vars(model).update(
        config=SimpleNamespace(max_position_embeddings=4096, hidden_size=2),
        media=(SimpleNamespace(encoder=vision_encoder((90, 91))),),
    )
    pads = model.rebuild_policies()[NODE].pad_ids
    pool = HostPool(1 << 10, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: replace(MEDIA, pad_ids=pads)}, _graphs(), pool, StepAllowance(), DTYPES)
    for rid in (R, A):
        _ingest(rebuilds, rid, IMAGE_CHAT)
    rebuilds.keep_prompts(_ran(VISION, {
        rid: {"text_inputs": [torch.arange(11)], "embeds": [torch.zeros(6, 2)]} for rid in (R, A)
    }))

    rebuilds.keep_samples(_batch((R, A)), {}, HostRows((R, A), {"new_token": torch.tensor([91, 5])}))

    assert "keeps no ids" in rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 0}, None), (
        "a rebuild row would carry the sampled pad, and the rows after it would read the wrong encoder rows"
    )
    assert list(rebuilds._kept[NODE][A].sampled) == [5], "a request that sampled text would stop being droppable"


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
    prompt = torch.tensor([1, 2], dtype=torch.long)
    copying = HostCopy(prompt, SimpleNamespace(query=lambda: False))
    rebuilds._kept[NODE][R] = KeptIds(copying, PREFILL, array("i", [5]), {})

    assert "haven't reached the host" in rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 0}, None), (
        "a rebuild would read ids the copy hasn't written yet"
    )


def test_a_rebuild_writes_the_prompt_and_samples_and_its_row_reads_all_but_the_last():
    rebuilds = _rebuilds()
    rebuilds._kept[NODE][R] = _kept([1, 2, 3], 4, 5)

    rebuild = rebuilds.start(R, NODE, None)
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {}, inputs, infos)

    assert rebuild == Rebuild(PREFILL, array("i", [1, 2, 3, 4, 5])), "the KV would lay out other ids"
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


@pytest.mark.parametrize("prefilling", [None, Prefilling(PREFILL, 4, 10)], ids=["kept ids", "mid-prefill"])
def test_a_drop_is_checked_against_the_tokens_its_rebuild_writes(prefilling):
    rebuilds = _rebuilds()
    rebuilds._kept[NODE][R] = _kept([1, 2, 3], 4, 5)

    tokens = rebuilds.tokens(R, NODE, prefilling)
    rebuilds.start(R, NODE, prefilling)

    assert tokens == rebuilds.cursor(R, NODE).total, "the drop would check one count and the rebuild write another"


def _write_kv(kv, rid: int, walk: str, *spans: int) -> None:
    """Write ``rid``'s next slots in steps of ``walk``, ``spans[i]`` of them in step ``i``."""
    for span in spans:
        step = KVStep(segments=(Segment(rid, "main", span),))
        ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)


@pytest.mark.parametrize("spans", [(PrefixSpan(11, 11, PREFILL),), IMAGE_CHAT], ids=["text", "image"])
@pytest.mark.parametrize("mid_prefill", [False, True], ids=["mid-decode", "mid-prefill"])
def test_a_drop_takes_the_slots_its_rebuild_writes_and_the_rebuild_writes_them(spans, mid_prefill):
    """The drop counted only a request's kept ids, so it refused a prompt with an image, whose rebuild also writes
    the image's slots from host copies. A count kept apart from the one the rebuild runs could also pass the drop and
    then write other slots than the KV held, which the KV caught only at the rebuild."""
    walk, image = spans[0].walk, len(spans) > 1
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, spans)
    kv = kv_drop._manager()
    layout = StreamLayout(spans=spans, growth=(("decode_loop", 1),))
    kv.ingest_request(R, KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode_loop": 50}))
    if mid_prefill:
        _write_kv(kv, R, walk, 9)
        prefilling, iters = Prefilling(walk, 9, 11), {}
    else:
        step = _ran(walk, {R: {
            "text_inputs": [torch.arange(3), torch.arange(2)] if image else [torch.arange(11)],
            "embeds": [torch.zeros(6, 2)],
        }})
        step.per_request_info[R].graph_walk = walk
        rebuilds.keep_prompts(step)
        rebuilds.keep_copies(step)
        rebuilds.flush()
        # the prompt, then the prefill's sample and one per decode step
        _write_kv(kv, R, walk, 11)
        _write_kv(kv, R, DECODE, 1, 1)
        rebuilds._kept[NODE][R].sampled.extend([100, 101, 102])
        prefilling, iters = None, {"decode_loop": 2}
    assert rebuilds.refusal(R, NODE, DECODE if iters else walk, iters, prefilling) is None

    tokens = rebuilds.tokens(R, NODE, prefilling)
    assert kv.check_drop(R, tokens) is None, f"the drop refused {tokens} slots for a KV that holds them"
    kv.drop(R, tokens)
    held = kv._dropped[R].held
    rebuild = rebuilds.start(R, NODE, prefilling)
    if rebuild is not None:
        kv.prepare_rebuild(R, rebuild)
    _write_kv(kv, R, walk, rebuilds.cursor(R, NODE).total)
    rebuilds.land_chunk(R, NODE, None)
    kv.land_rebuild(R)

    assert kv._streams[R]["main"].stored_len == held, "the rebuild wrote other slots than the drop checked"
    # the rest of the prompt, or the next decode step
    _write_kv(kv, R, *((walk, 2) if mid_prefill else (DECODE, 1)))


def test_the_sampler_commits_no_draw_of_a_rebuild_row_whatever_the_model_declared():
    ctx = StepContext(request_ids=(R, A), graph_walk=PREFILL, slot=0, capture=False, rebuild_rids=frozenset({R}))
    for kept_rids in (None, frozenset({R, A})):
        kept = SamplerResource._kept_rids_of(SamplerStep(kept_rids=kept_rids), ctx)

        assert kept == {A}, "the rebuild row's sample would overwrite decode's next input"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="counts the device's allocations")
# a walk reads a Qwen3.5 prompt with images as one tensor of ids for each run of text
@pytest.mark.parametrize("tensors", [1, 2])
def test_keeping_a_prompt_runs_nothing_on_the_device_but_its_copy(tensors):
    device = torch.device("cuda")
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 12, device))
    assert _ingest(rebuilds, R, (PrefixSpan(256, 256, PREFILL),))
    prompt = torch.arange(256, device=device)
    step_done = torch.Event(device=device)
    step_done.record()
    step = _ran(PREFILL, {R: {"text_inputs": list(prompt.tensor_split(tensors))}})
    step.completion_event = step_done
    allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]

    rebuilds.keep_prompts(step)
    kept = rebuilds._kept[NODE][R]
    kept.prompt.landed.synchronize()

    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated, (
        "a cast on the pool's stream made every later decode step 0.16 ms slower at Qwen3.5-9B TP2"
    )
    assert kept.prompt.view.tolist() == list(range(256)), "the rebuild would prefill other ids"
    assert kept.lengths == (None if tensors == 1 else (128, 128)), "the rebuild would join the runs of text"
