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
from mstar.worker.host_pool import HostPool, HostSlice
from mstar.worker.rebuild import (
    KeptIds,
    LayoutRow,
    NodeWrites,
    Prefilling,
    Rebuilds,
    RowRecord,
    policies_run_here,
)

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
    ("ids_input", "pixels", "carries"),
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
    rebuilds.record(_ran(spans[0].walk, {R: {"text_inputs": [torch.arange(3)], "embeds": [torch.zeros(6, 2)]}}))

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
        "config": SimpleNamespace(max_position_embeddings=4096, num_timesteps=50, hidden_size=64),
        "_has_cfg_parallel": False, "_image_gen_remote_handoff": False,
    }),
    (Qwen3OmniModel, {"config": SimpleNamespace(
        thinker_text=SimpleNamespace(max_position_embeddings=4096, hidden_size=64),
        vision=SimpleNamespace(out_hidden_size=64, deepstack_visual_indexes=(8, 16)),
    )}),
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


def test_higgs_is_rebuilt_from_its_audio_step_with_the_encoders_output_copied():
    from test_model_kv_layouts import _higgs

    model = _higgs()
    policy = model.rebuild_policies().get("LLM")

    assert policy is not None, "every Higgs transcription would stay guaranteed"
    assert policy.check("LLM", model.get_graph_walk_graphs()) == {
        "prefill_audio": frozenset({"audio_embeds"}),
    }, "a rebuild row would run without the encoder's output"


def test_a_higgs_transcription_can_be_dropped_where_its_llm_runs():
    from test_model_kv_layouts import _higgs

    from mstar.model.keys import KV_CACHE

    model = _higgs()
    out = model.process_prompt("transcribe", ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(16000 * 35)]})
    config = KVReqConfig()
    config.apply_conductor_config(
        kv_layout={"main": out.metadata["kv_layout"][KV_CACHE]["main"].row()}, loop_caps={"decode_loop": 8},
    )
    rebuilds = Rebuilds(
        model.rebuild_policies(), model.get_graph_walk_graphs(), HostPool(1 << 22, torch.device("cpu")),
        StepAllowance(), {"LLM": torch.bfloat16},
    )
    writes = {KV_CACHE: {"LLM": NodeWrites(frozenset({"prefill_audio", "decode"}), frozenset({"decode_loop"}))}}

    ingested = rebuilds.ingest_request(R, writes, {KV_CACHE: config.kv_layout.values()}, unpausable=False)

    assert ingested.rebuildable[KV_CACHE], "every Higgs transcription would stay guaranteed"


def _rows(*rids: int) -> dict:
    return {rid: {"text_inputs": [torch.arange(rid, rid + 3)]} for rid in rids}


def _ingest(rebuilds: Rebuilds, rid: int, spans: tuple[PrefixSpan, ...]) -> bool:
    """Take ``rid`` in with one label of ``spans`` grown by the decode loop, and return whether a drop on ``NODE``
    can free it."""
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, VISION, DECODE}), frozenset({"decode_loop"}))}}
    layouts = {"kv": [StreamLayout(spans=spans, growth=(("decode_loop", 1),))]}
    return rebuilds.ingest_request(rid, writes, layouts, unpausable=False).rebuildable["kv"]


def _rebuilds(pool_bytes: int = 1 << 10) -> Rebuilds:
    """`Rebuilds` with ``R``, ``A`` and ``B`` taken in as requests a drop on ``NODE`` can free."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(pool_bytes, torch.device("cpu")))
    for rid in (R, A, B):
        assert _ingest(rebuilds, rid, (PrefixSpan(3, 3, PREFILL),))
    return rebuilds


def _ran(walk: str, inputs: dict, request_walks: dict | None = None, rebuild=frozenset()) -> SimpleNamespace:
    """A step of ``walk`` that read ``inputs``, run on the host, each row whole."""
    ctx = StepContext(
        request_ids=tuple(inputs), graph_walk=walk, slot=0, capture=False, request_walks=request_walks or {},
        rebuild_rids=rebuild,
    )
    infos = {rid: SimpleNamespace(step_metadata={"is_prefill": True}) for rid in inputs}
    return SimpleNamespace(
        node_name=NODE, step_context=ctx, request_ids=list(inputs), per_request_input_tensors=inputs,
        per_request_info=infos, cached_rids=set(), chunk_ranges={}, completion_event=None,
    )


def _record(prompt, landed=None) -> RowRecord:
    """A prefill step that read ``prompt`` as int64 ids, as `Rebuilds.record` keeps it."""
    return RowRecord({"text_inputs": [torch.tensor(prompt)]}, {"is_prefill": True}, landed=landed)


def _kept(prompt, *sampled: int) -> KeptIds:
    """A log of one prefill step that read ``prompt``, and ``sampled``."""
    return KeptIds((LayoutRow(PREFILL, len(prompt)),), [_record(prompt)], array("i", sampled))


def test_a_prompt_rows_ids_are_kept_once_on_the_host():
    rebuilds = _rebuilds()

    rebuilds.record(_ran(PREFILL, _rows(R, A)))
    rebuilds.record(_ran(PREFILL, {R: {"text_inputs": [torch.tensor([0])]}}, rebuild={R}))  # its rebuild row
    rebuilds.record(_ran(MIXED, _rows(A, B), {A: PREFILL, B: DECODE}))
    rebuilds.record(_ran(DECODE, _rows(B)))
    rebuilds.record(_ran(PREFILL, {B: {}}))  # a TP follower's unread row

    kept = rebuilds._kept[NODE]
    [ids_r], [ids_a] = kept[R].records[0].inputs["text_inputs"], kept[A].records[0].inputs["text_inputs"]
    assert ids_r.tolist() == [7, 8, 9] and ids_a.tolist() == [8, 9, 10], "it kept another row's ids"
    assert len(kept[R].records) == 1, "a rebuild row would be recorded as a prefill step"
    assert ids_r.dtype == torch.long, (
        "the rebuild row would read its ids in another dtype than the walk's"
    )
    assert not kept[B].records, "a decode row's input, or an unread row's, would be taken for its prompt"


def test_only_the_rows_that_ran_are_recorded():
    """The engine takes a row it skipped or failed out of the step's rows and their infos, but not out of its
    inputs."""
    rebuilds = _rebuilds()
    step = _ran(PREFILL, _rows(R, A))
    step.request_ids.remove(A)
    del step.per_request_info[A]

    rebuilds.record(step)

    kept = rebuilds._kept[NODE]
    assert len(kept[R].records) == 1 and not kept[A].records, "a row that never ran would be recorded"


def test_a_step_the_prefix_cache_served_whole_is_recorded():
    """Three Bagel requests with the same image and question lost their logs, since the cache served their step
    whole, and the pool still counted on dropping them. Once it ran short nothing could be dropped, and nothing
    moved."""
    rebuilds = _rebuilds()
    step = _ran(PREFILL, _rows(R, B))
    # the engine takes a row the cache served out of the step's rows and their infos
    step.cached_rids.add(B)
    step.request_ids.remove(B)
    info = step.per_request_info.pop(B)

    rebuilds.record(step, {B: info})
    rebuilds.keep_samples(_batch((B,)), {B: {"new_token": [torch.tensor([9])]}}, None)

    [record] = rebuilds._kept[NODE][B].records
    assert record.inputs["text_inputs"][0].tolist() == [9, 10, 11], "the rebuild would prefill other ids"
    assert rebuilds.refusal(B, NODE, DECODE, {}, None) is None, "a request the pool counts on dropping can't be dropped"


def test_a_request_the_full_pool_cant_hold_is_not_rebuildable():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(32, torch.device("cpu")))

    assert _ingest(rebuilds, A, (PrefixSpan(4, 4, PREFILL),)), "a request the pool holds would stay guaranteed"
    assert not _ingest(rebuilds, R, (PrefixSpan(2, 2, PREFILL),)), "two requests' ids would share host memory"
    rebuilds.record(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}, R: {"text_inputs": [torch.arange(2)]}}))
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


PROMPT = (PrefixSpan(4, 4, PREFILL),)


def _ranks() -> tuple[Rebuilds, Rebuilds]:
    """Rank 0 and a follower of ``NODE``, each with a 32-byte pool that r's prompt fills."""
    leader = Rebuilds({NODE: POLICY}, _graphs(), HostPool(32, torch.device("cpu")))
    follower = Rebuilds({NODE: POLICY}, _graphs(), HostPool(32, torch.device("cpu")), followed=frozenset({NODE}))
    for rebuilds in (leader, follower):
        assert _ingest(rebuilds, R, PROMPT)
    return leader, follower


def test_a_follower_keeps_the_log_of_a_request_rank_0_keeps_when_its_own_pool_is_full():
    """Each rank gives a slice back at its own teardown. Rank 0 tears r down before a arrives and the follower
    after, so only rank 0's pool has room for a's log. Rank 0 can then drop a, and a follower that kept no log of a
    stopped. The follower takes a's room past its pool."""
    leader, follower = _ranks()
    leader.remove_request(R)

    assert _ingest(leader, A, PROMPT) and _ingest(follower, A, PROMPT)
    follower.remove_request(R)
    follower.record(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}}))

    assert leader.kept(A, NODE) is not None
    assert follower.kept(A, NODE) is not None, "rank 0 can drop a request the follower keeps no log of"
    [ids] = follower.kept(A, NODE).records[0].inputs["text_inputs"]
    assert ids.tolist() == [0, 1, 2, 3] and follower.pool.nbytes_used() == 0, (
        "the follower kept other ids, or a's log takes room in its pool"
    )


def test_a_full_pool_names_the_node_whose_log_it_couldnt_hold():
    """Rank 0 journals each node its pool refused a log, for its followers. A request no drop can free took no room,
    which every rank works out alike."""
    leader, _ = _ranks()
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"decode_loop"}))}}
    layouts = {"kv": [StreamLayout(spans=PROMPT, growth=(("decode_loop", 1),))]}

    unpausable = leader.ingest_request(B, writes, layouts, unpausable=True)
    full = leader.ingest_request(A, writes, layouts, unpausable=False)

    assert full == ({"kv": False}, {NODE}), "a follower would keep a log rank 0's pool refused"
    assert unpausable.unheld == set(), "rank 0 would journal a refusal every rank makes"


def test_a_follower_gives_back_the_log_of_a_request_rank_0s_pool_couldnt_hold():
    """Rank 0's pool has no room for a's log, so a is not rebuildable there and rank 0 never drops it. The follower
    took a's room past its pool and counted a rebuildable until rank 0's journal said otherwise."""
    leader, follower = _ranks()
    assert not _ingest(leader, A, PROMPT) and _ingest(follower, A, PROMPT)

    assert follower.refuse(A, NODE) == ["kv"], "the follower would admit a in another class than rank 0"
    assert follower.kept(A, NODE) is None and not follower.pool._spilled, (
        "the follower holds a log outside its pool that rank 0 never drops with"
    )


def test_a_request_whose_intake_fails_while_it_takes_room_gives_back_every_slice_at_its_teardown(monkeypatch):
    """The talker's log took a slice past the follower's full pool, and the pinned allocation for the LLM's log
    raised. The talker's slice had no owner, so a's teardown never gave it back."""
    talker, node = "talker", SimpleNamespace(input_names={"text_inputs"}, outputs=[SimpleNamespace(name="new_token")])
    nodes = {NODE: node, talker: node}
    loop = SimpleNamespace(get_nodes=lambda: nodes)
    graphs = {
        PREFILL: SimpleNamespace(get_nodes=lambda: nodes, get_loops=dict),
        DECODE: SimpleNamespace(get_nodes=lambda: nodes, get_loops=lambda: {"decode_loop": loop}),
    }
    policies = {NODE: POLICY, talker: POLICY}
    what = NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"decode_loop"}))
    writes = {"kv": {NODE: what}, "kv2": {talker: what}}
    layouts = dict.fromkeys(writes, [StreamLayout(spans=PROMPT, growth=(("decode_loop", 1),))])
    follower = Rebuilds(policies, graphs, HostPool(16, torch.device("cpu")), followed=frozenset(policies))
    take = follower.pool.take

    def take_one(nbytes, spill=False):
        """Allocates the first slice past the pool, and raises at the second as the pinned allocator would."""
        if follower.pool._spilled:
            raise RuntimeError("CUDA error: out of memory")
        return take(nbytes, spill)
    monkeypatch.setattr(follower.pool, "take", take_one)

    with pytest.raises(RuntimeError, match="out of memory"):
        follower.ingest_request(A, writes, layouts, unpausable=False)
    follower.remove_request(A)

    assert not follower.pool._spilled, "the follower holds a slice outside its pool that no request owns"


def test_a_request_takes_8_bytes_for_each_prompt_id_when_it_arrives():
    """A slice held 4 B for each prompt id, too few to keep a prompt its walk read as int64."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    assert _ingest(rebuilds, R, (PrefixSpan(5, 5, PREFILL),))
    assert rebuilds._keeps_ids[NODE][R].nbytes == 40, "keeping the prompt would overflow its slice"

    rebuilds.record(_ran(PREFILL, {R: {"text_inputs": [torch.arange(5)]}}))

    [kept] = rebuilds._kept[NODE][R].records[0].inputs["text_inputs"]
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

    rebuilds.record(_ran(PREFILL, _rows(R)))
    rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([9])]}}, None)

    assert R not in rebuilds._kept[NODE], f"a request kept its ids though {why}"


def _copying(*rids: int) -> Rebuilds:
    """`Rebuilds` whose ``rids`` were taken in with an image chat on ``VISION``, each with a slice."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    for rid in rids:
        assert _ingest(rebuilds, rid, IMAGE_CHAT)
    return rebuilds


def test_a_prompt_with_images_records_its_ids_as_its_walk_read_them_and_its_image():
    rebuilds = _copying(R)
    rows = {R: {"text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.zeros(6, 2)]}}

    rebuilds.record(_ran(VISION, rows))
    rebuilds.flush()

    [record] = rebuilds._kept[NODE][R].records
    assert [ids.tolist() for ids in record.inputs["text_inputs"]] == [[1, 2, 3], [4, 5]], (
        "the rebuild row would read other text"
    )
    assert torch.equal(record.inputs["embeds"][0], torch.zeros(6, 2)), "a rebuild row would leave out the image"


def test_a_prompt_that_read_a_pad_id_at_each_image_slot_leaves_its_copies_no_slots_of_their_own():
    """A VLM's media walk reads an id at every slot, a pad where the encoder's output goes in. Counting the image's
    slots again for its copies made the rebuild write more slots than the KV held, so every drop refused it, and a log
    with room for ids only at the ids spans overflowed its slice."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    _ingest(rebuilds, R, IMAGE_CHAT)
    rows = {R: {"text_inputs": [torch.arange(11)], "embeds": [torch.zeros(6, 2)]}}

    rebuilds.record(_ran(VISION, rows))
    rebuilds._kept[NODE][R].sampled.append(6)

    assert rebuilds.tokens(R, NODE, None) == 11, "the rebuild would write other slots than the prompt's KV holds"


def _vision_step(rids, walks: dict, **fields) -> SimpleNamespace:
    infos = {rid: SimpleNamespace(graph_walk=walks.get(rid, VISION), step_metadata={"prefill_order": (rid,)})
             for rid in rids}
    inputs = {rid: {"text_inputs": [torch.tensor([rid])], "embeds": [torch.full((2,), rid)]} for rid in rids}
    ctx = StepContext(request_ids=tuple(rids), graph_walk="mixed", slot=0, capture=False,
                      request_walks={rid: info.graph_walk for rid, info in infos.items()})
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), per_request_info=infos,
                           per_request_input_tensors=inputs, cached_rids=set(), step_context=ctx, chunk_ranges={},
                           **fields)


def test_a_step_of_no_prefill_walk_records_nothing():
    rebuilds = _copying(R)

    rebuilds.record(_vision_step((R,), {R: DECODE}, completion_event=None))

    assert not rebuilds._kept[NODE][R].records, "a decode step would be taken for the image's step"


def test_a_prefill_row_keeps_a_host_copy_of_everything_it_read_and_of_its_step_metadata():
    rebuilds = _copying(R, A)
    step = _vision_step((R, A, B), {A: DECODE}, completion_event=None)

    rebuilds.record(step)
    rebuilds.flush()

    [record] = rebuilds._kept[NODE][R].records
    assert sorted(record.inputs) == ["embeds", "text_inputs"] and record.inputs["embeds"][0].tolist() == [R, R], (
        "a rebuild would need the encoder to run again"
    )
    assert record.step_metadata == {"prefill_order": (R,)}, "the rebuild row would run under decode's metadata"
    assert record.step_metadata is not step.per_request_info[R].step_metadata, "a later step could rewrite the copy"
    assert record.ready(), "the request could never be dropped"
    assert not rebuilds._kept[NODE][A].records, "a row of another walk would stand in for the prefill"
    assert B not in rebuilds._kept[NODE], "a request no drop can free would keep a log"


def test_a_request_with_images_takes_room_for_its_copies_when_it_arrives_and_gives_back_what_they_didnt_use():
    """Every image request pinned its encoder output in blocks of its own for its whole life, and nothing capped the
    total, so a failed pinned allocation failed every request in its step. The log takes one slice of the startup
    pool when a request arrives, sized by the policy's bytes per slot, so a request the pool can't hold is known
    before it is admitted and stays out of the drops."""
    pool = HostPool(256, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance(), DTYPES)

    # an id at 8 B for each of the eleven slots, and the image's six slots at two float32 values each
    assert _ingest(rebuilds, R, IMAGE_CHAT) and rebuilds._keeps_ids[NODE][R] == HostSlice(0, 88 + 48)
    assert not _ingest(rebuilds, A, IMAGE_CHAT), "a request the full pool can't hold would be admitted as droppable"
    step = _vision_step((R,), {}, completion_event=None)
    # the image's four rows, without its two sentinels
    step.per_request_input_tensors[R] = {
        "text_inputs": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])], "embeds": [torch.ones(4, 2)],
    }
    rebuilds.record(step)
    rebuilds.flush()

    assert rebuilds._keeps_ids[NODE][R].nbytes == 40 + 32, (
        "the image's room for ids and its sentinels' room would stay taken for the request's life"
    )
    assert rebuilds._kept[NODE][R].records[0].inputs["embeds"][0].tolist() == [[1.0, 1.0]] * 4


def test_a_media_copy_past_the_steps_allowance_waits_for_the_next_steps():
    """The image copies queued every input of every prefill row at once, behind which the next step's output copy
    waited on the device's one copy engine. The image copies share one allowance with the cached pages' writes, and
    go first."""
    allowance = StepAllowance(24)
    rebuilds = _copying(R)
    rebuilds._allowance = allowance
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.arange(8, dtype=torch.float32)]

    rebuilds.record(step)
    rebuilds.flush()

    [record] = rebuilds._kept[NODE][R].records
    assert allowance.left == 0, "the step's copies would run past its allowance"
    assert not record.ready(), "the request could be dropped with part of its image copied"
    allowance.refill()
    rebuilds.flush()
    assert record.ready() and record.inputs["embeds"][0].tolist() == list(range(8)), "the split copy lost bytes"
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
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, StepAllowance(8), DTYPES)
    assert _ingest(rebuilds, R, IMAGE_CHAT)
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.arange(8, dtype=torch.float32)]
    rebuilds.record(step)
    # the ids go to the host at once, outside the allowance
    [ids] = pool.events
    ids.synchronize()
    pool.overlapped.clear()

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

    rebuilds.record(step)
    rebuilds.flush()

    [record] = rebuilds._kept[NODE][R].records
    assert record.ready(), "the request's image would never land, so it could never be dropped"
    assert [view.shape for view in record.inputs["embeds"]] == [(0, 2), (2, 2)], (
        "the rebuild row would read other inputs than its prefill row"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_media_copy_split_across_two_steps_lands_whole():
    pool = HostPool(1 << 22, torch.device("cuda"))
    rebuilds = Rebuilds(
        {NODE: MEDIA}, _graphs(), pool, allowance=(allowance := StepAllowance(1 << 20)), node_dtypes=DTYPES,
    )
    # an image of 3 << 16 slots at the policy's two float32 values a slot holds the 3 << 17 float32 embeds
    assert _ingest(rebuilds, R, (IMAGE_CHAT[0], replace(IMAGE_CHAT[1], length=3 << 16), IMAGE_CHAT[2]))
    embeds = torch.arange(3 << 17, dtype=torch.float32, device="cuda")
    step_done = torch.Event(device="cuda")
    step_done.record()
    step = _vision_step((R,), {}, completion_event=step_done)
    step.per_request_input_tensors[R] = {"text_inputs": [torch.tensor([R], device="cuda")], "embeds": [embeds]}

    rebuilds.record(step)
    rebuilds.flush()
    allowance.refill()
    rebuilds.flush()
    del step
    torch.cuda.synchronize()

    [record] = rebuilds._kept[NODE][R].records
    assert record.ready() and torch.equal(record.inputs["embeds"][0], embeds.cpu()), "the second half never landed"


def test_a_rebuild_of_ids_alone_runs_under_its_prefills_metadata():
    """A row of ids alone ran under the request's decode step's metadata and a row with images under its prefill's,
    so the model and the sampler saw the two differently. Every kept request records a copy of its prefill row's
    metadata, since a later step may change the live one."""
    rebuilds = _rebuilds()
    prefill = _ran(PREFILL, _rows(R))
    rebuilds.record(prefill)
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
    image_chat = {"text_inputs": [torch.arange(3), torch.arange(2)], "embeds": [torch.ones(4, 2)]}
    rebuilds.record(_ran(VISION, {R: image_chat}))
    rebuilds._kept[NODE][R].sampled.append(6)

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
    texts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    record = RowRecord({"embeds": [torch.ones(6, 2)], "text_inputs": texts}, {})
    rebuilds._kept[NODE][R] = KeptIds((LayoutRow(VISION, 11),), [record], array("i", [6]))

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
    allowance, pool = StepAllowance(0), HostPool(256, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), pool, allowance, node_dtypes=DTYPES)
    assert _ingest(rebuilds, R, IMAGE_CHAT)
    step = _vision_step((R,), {}, completion_event=None)
    step.per_request_input_tensors[R]["embeds"] = [torch.full((4,), 7.0)]
    rebuilds.record(step)

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
    texts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    record = RowRecord({"embeds": [embeds], "text_inputs": texts}, metadata)
    rebuilds._kept[NODE][R] = KeptIds((LayoutRow(VISION, 11),), [record], array("i", [6, 7]))

    rebuild = rebuilds.start(R, NODE, None)
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {}, inputs, infos)

    assert rebuild == Rebuild(array("i", [6, 7])) and infos[R].graph_walk == VISION
    assert [t.tolist() for t in inputs[R]["text_inputs"]] == [[1, 2, 3], [4, 5, 6]], (
        "the samples would follow the image instead of the prompt's last text, or decode would read the last id twice"
    )
    assert torch.equal(inputs[R]["embeds"][0], embeds) and infos[R].step_metadata is metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_host_copy_lands_whole_before_its_request_can_be_dropped():
    rows = 1 << 22
    rebuilds = Rebuilds(
        {NODE: MEDIA}, _graphs(), HostPool(8 * rows + (1 << 10), torch.device("cuda")), StepAllowance(), DTYPES,
    )
    # an image of rows // 2 slots at the policy's two float32 values a slot holds the rows float32 embeds
    image = replace(IMAGE_CHAT[1], length=rows // 2)
    assert _ingest(rebuilds, R, (IMAGE_CHAT[0], image, IMAGE_CHAT[2]))
    embeds = torch.arange(1, 1 + rows, dtype=torch.float32, device="cuda")
    expected = embeds.cpu()
    # the step's completion waits behind a busy stream, which keeps the copy queued past the point where the worker
    # lets go of the step's tensors
    busy, completed = torch.cuda.Stream(), torch.cuda.Event()
    with busy:
        torch.cuda._sleep(1 << 30)
    completed.record(busy)
    texts = [torch.tensor([1, 2, 3], device="cuda"), torch.tensor([4, 5], device="cuda")]
    step = _ran(VISION, {R: {"text_inputs": texts, "embeds": [embeds]}})
    step.completion_event = completed
    rebuilds.record(step)
    # what the worker does once the step's outputs are sent
    rebuilds.flush()
    rebuilds._kept[NODE][R].sampled.append(6)

    assert "haven't reached the host" in rebuilds.refusal(R, NODE, DECODE, {}, None), (
        "the request could be dropped before its copy"
    )
    del step, texts, embeds
    torch.full((rows,), -1.0, device="cuda")
    torch.cuda.synchronize()
    assert rebuilds.refusal(R, NODE, DECODE, {}, None) is None
    [record] = rebuilds._kept[NODE][R].records
    assert torch.equal(record.inputs["embeds"][0], expected), "the next step's tensor took the memory before the copy"


def _batch(rids, rebuild=frozenset()) -> SimpleNamespace:
    ctx = StepContext(request_ids=tuple(rids), graph_walk=DECODE, slot=0, capture=False, rebuild_rids=rebuild)
    return SimpleNamespace(node_name=NODE, request_ids=list(rids), step_context=ctx)


@pytest.mark.parametrize("sampled", [[4, 5, 6], [[4], [5], [6]]])  # the sampler's [B] output, and a [B, 1] one
def test_only_a_rows_kept_sample_is_appended(sampled):
    rebuilds = _rebuilds()
    rebuilds.record(_ran(PREFILL, _rows(R, A, B)))
    # A runs a chunk short of its prompt's end, and B is a rebuild row
    batch = _batch((R, A, B), rebuild=frozenset({B}))

    rebuilds.keep_samples(batch, {}, HostRows((R, A, B), {"new_token": torch.tensor(sampled)}), skip_rids={A})
    rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([9])]}}, None)

    kept = rebuilds._kept[NODE]
    assert list(kept[R].sampled) == [4, 9], "a rebuild would prefill other ids than the request's"
    assert not (kept[A].sampled or kept[B].sampled), "it kept a discarded sample"


def test_a_sample_is_kept_from_the_host_outputs_the_worker_copies_on_read():
    rebuilds = _rebuilds()
    rebuilds.record(_ran(PREFILL, _rows(R)))
    row = {"new_token": [torch.tensor([9])]}
    outputs = _LazyHostOutputs({R: row}, lambda names: {name: {R: row[name]} for name in names})

    rebuilds.keep_samples(_batch((R,)), outputs, None)

    assert list(rebuilds._kept[NODE][R].sampled) == [9], "every drop would be refused for the samples it missed"


@pytest.mark.parametrize("host", [True, False])
def test_a_sample_of_more_than_one_id_is_refused(host):
    rebuilds = _rebuilds()
    rebuilds.record(_ran(PREFILL, _rows(R)))
    codes = torch.tensor([[4, 5, 6]])  # as a talker samples several codebooks

    with pytest.raises(ValueError, match="hold 3 ids"):
        if host:
            rebuilds.keep_samples(_batch((R,)), {}, HostRows((R,), {"new_token": codes}))
        else:
            rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [codes[0]]}}, None)


def test_the_vlm_scaffold_keeps_a_request_that_samples_a_pad_id_droppable():
    """The scaffold places a prompt's encoder rows by its layout, not by pad id, so a pad sampled into a rebuild
    row's tail takes no encoder row (`test_qwen3_5_video`), and no sample needs to make the request undroppable."""
    from mstar.model.multimodal_decoder import MediaIntake

    model = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
    model.config = SimpleNamespace(max_position_embeddings=4096, hidden_size=2)
    model.media = (MediaIntake("image", vision_encoder(), 0, None),)
    policy = model.rebuild_policies()[NODE]
    assert policy.walks == {"prefill_text", "prefill_vision"} and not policy.pad_ids


@pytest.mark.parametrize("last, forgotten", [(VISION, True), (PREFILL, False)])
def test_a_pad_id_sampled_after_several_steps_forgets_the_log_only_when_the_last_step_read_more_than_ids(
    last, forgotten,
):
    """The samples go in the rebuild row of a request's last prefill step. A pad there reads an encoder row only
    when that step read an encoder's output, whatever the steps before it read."""
    pool = HostPool(1 << 10, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: replace(MEDIA, pad_ids=frozenset({91}))}, _graphs(), pool, StepAllowance(), DTYPES)
    media = RowRecord({"text_inputs": [torch.tensor([1])], "embeds": [torch.zeros(1, 2)]}, {"is_prefill": True})
    steps = [(VISION, media), (PREFILL, _record((2, 3, 4)))]
    if last == VISION:
        steps.reverse()
    rebuilds._kept[NODE][R] = KeptIds(tuple(LayoutRow(walk, 3) for walk, _ in steps), [r for _, r in steps])

    forgot = rebuilds.keep_samples(_batch((R,)), {}, HostRows((R,), {"new_token": torch.tensor([91])}))

    assert (forgot == [R]) == forgotten and (R not in rebuilds._kept[NODE]) == forgotten, (
        "a rebuild row would give the pad an encoder row" if forgotten else "a request whose rebuild is exact was "
        "never dropped again"
    )


def test_a_request_of_ids_alone_that_samples_a_pad_id_stays_droppable():
    """A walk of ids alone runs with no encoder output, so a pad in its rebuild row is embedded as the decode step
    embedded it. Forgetting its ids left a request its pool admitted as droppable with nothing that could move it."""
    pool = HostPool(1 << 10, torch.device("cpu"))
    rebuilds = Rebuilds({NODE: replace(MEDIA, pad_ids=frozenset({91}))}, _graphs(), pool, StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, (PrefixSpan(3, 3, PREFILL),))
    rebuilds.record(_ran(PREFILL, _rows(R)))

    forgot = rebuilds.keep_samples(_batch((R,)), {}, HostRows((R,), {"new_token": torch.tensor([91])}))

    assert not forgot and rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 0}, None) is None, (
        "a text request whose rebuild would write the KV it holds was never dropped again"
    )


@pytest.mark.parametrize("why, node, walk, sampled, iters", [
    ("declares no rebuild policy", "vocoder", DECODE, 1, 0),
    ("keeps no log", NODE, DECODE, None, 0),
    ("doesn't run", NODE, PREFILL, 1, 0),
    ("don't cover its KV", NODE, DECODE, 1, 1),
])
def test_a_drop_is_refused_unless_the_kept_ids_cover_the_kv(why, node, walk, sampled, iters):
    rebuilds = _rebuilds()
    if sampled is None:
        del rebuilds._kept[NODE][R]
    else:
        rebuilds._kept[NODE][R] = _kept([1, 2], *[5] * sampled)

    assert why in rebuilds.refusal(R, node, walk, {"decode_loop": iters}, None)
    rebuilds._kept[NODE][R] = _kept([1, 2], 5, 6)
    assert rebuilds.refusal(R, NODE, DECODE, {"decode_loop": 1}, None) is None, "a request its ids cover was refused"


def test_a_drop_is_refused_until_the_kept_ids_reach_the_host():
    rebuilds = _rebuilds()
    copying = _record([1, 2], landed=SimpleNamespace(query=lambda: False))
    rebuilds._kept[NODE][R] = KeptIds((LayoutRow(PREFILL, 2),), [copying], array("i", [5]))

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


@pytest.mark.parametrize("prefilling", [None, Prefilling(PREFILL, 4, 10)], ids=["kept ids", "mid-prefill"])
def test_a_drop_is_checked_against_the_tokens_its_rebuild_writes(prefilling):
    rebuilds = _rebuilds()
    rebuilds._kept[NODE][R] = _kept([1, 2, 3], 4, 5)

    tokens = rebuilds.tokens(R, NODE, prefilling)
    rebuilds.start(R, NODE, prefilling)

    assert tokens == rebuilds.cursor(R, NODE).total, "the drop would check one count and the rebuild write another"


def test_the_sampler_commits_no_draw_of_a_rebuild_row_whatever_the_model_declared():
    ctx = StepContext(request_ids=(R, A), graph_walk=PREFILL, slot=0, capture=False, rebuild_rids=frozenset({R}))
    for kept_rids in (None, frozenset({R, A})):
        kept = SamplerResource._kept_rids_of(SamplerStep(kept_rids=kept_rids), ctx)

        assert kept == {A}, "the rebuild row's sample would overwrite decode's next input"


# A Bagel image chat: text in its own step, then the image and the text after it in the image's walk, as a Qwen3-Omni
# prompt with audio lays out each attachment's step.
TWO_STEPS = (PrefixSpan(3, 3, PREFILL), *IMAGE_CHAT[1:])
TEXT_STEP = {"text_inputs": [torch.tensor([1, 2, 3])]}
IMAGE_STEP = {"text_inputs": [torch.tensor([4, 5])], "embeds": [torch.ones(4, 2)]}


def _two_steps(*sampled: int) -> Rebuilds:
    """`Rebuilds` whose ``R`` ran both steps of ``TWO_STEPS``, each under its own metadata, and sampled ``sampled``."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), _CountingPool(), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, TWO_STEPS)
    for walk, inputs, order in ((PREFILL, TEXT_STEP, "text"), (VISION, IMAGE_STEP, "image")):
        step = _ran(walk, {R: inputs})
        step.per_request_info[R].step_metadata = {"prefill_order": (order,)}
        rebuilds.record(step)
    rebuilds.flush()
    rebuilds._kept[NODE][R].sampled.extend(sampled)
    return rebuilds


def _fill(rebuilds: Rebuilds) -> tuple[dict, CurrentForwardPassInfo]:
    inputs, infos = {R: {}}, {R: CurrentForwardPassInfo(str(R), DECODE, 0, 0, 8)}
    rebuilds.fill(NODE, {R}, {}, inputs, infos)
    return inputs[R], infos[R]


def test_a_prompt_of_two_steps_is_rebuilt_a_row_per_step_from_its_layout():
    """The rebuild replayed the steps it had watched go by, so a chunked prompt, recorded once per chunk, rebuilt
    other rows than its layout lays out. The rows are the layout's steps, and the last one carries the samples."""
    rebuilds = _two_steps(6, 7)

    rebuild = rebuilds.start(R, NODE, None)
    cursor = rebuilds.cursor(R, NODE)

    assert [(row.walk, row.total) for row in cursor.rows] == [(PREFILL, 3), (VISION, 9)], (
        "the rows would write other slots than the prompt's steps, or leave out the sample decode wrote"
    )
    assert rebuild == Rebuild(array("i", [6, 7]))


def test_every_rebuild_row_runs_under_its_own_prefill_steps_metadata_and_reads_only_its_own_copies():
    rebuilds = _two_steps(6, 7)
    rebuilds.start(R, NODE, None)

    text, text_info = _fill(rebuilds)
    assert text_info.step_metadata == {"prefill_order": ("text",)}, "the text row would run under decode's metadata"
    assert rebuilds._pool.copied_in == 0 and list(text) == ["text_inputs"], (
        "handing out the text row would copy the image back to the device, outside the KV pool"
    )
    rebuilds.land_chunk(R, NODE, None)
    image, image_info = _fill(rebuilds)

    assert image_info.graph_walk == VISION and image_info.step_metadata == {"prefill_order": ("image",)}
    assert rebuilds._pool.copied_in == 1 and image["text_inputs"][0].tolist() == [4, 5, 6], (
        "the samples would follow another step's text, or the image would stay on the host"
    )


def _write_kv(kv, rid: int, walk: str, *spans: int) -> None:
    """Write ``rid``'s next slots in steps of ``walk``, ``spans[i]`` of them in step ``i``."""
    for span in spans:
        step = KVStep(segments=(Segment(rid, "main", span),))
        ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)


# a prompt of text alone, of text and an image in one step, and of text and then an image in a step of its own,
# each as the steps its layout lays out: their walks, what they read and the slots they write
PROMPTS = {
    "text": ((PrefixSpan(11, 11, PREFILL),), [(PREFILL, {"text_inputs": [torch.arange(11)]}, 11)]),
    "image": (IMAGE_CHAT, [(VISION, {**IMAGE_STEP, "text_inputs": [torch.arange(3), torch.arange(2)]}, 11)]),
    "two steps": (TWO_STEPS, [(PREFILL, TEXT_STEP, 3), (VISION, IMAGE_STEP, 8)]),
}


@pytest.mark.parametrize("prompt", list(PROMPTS))
@pytest.mark.parametrize("mid_prefill", [False, True], ids=["mid-decode", "mid-prefill"])
def test_a_drop_takes_the_slots_its_rebuild_writes_and_the_rebuild_writes_them(prompt, mid_prefill):
    """The drop counted only a request's kept ids, or for a prompt mid-prefill only the chunks of the step in
    progress, so it refused a prompt with an image, whose rebuild also writes the image's slots and reruns the steps
    before the one in progress. A count kept apart from the one the rebuild runs could also pass the drop and then
    write other slots than the KV held, which the KV caught only at the rebuild."""
    spans, steps = PROMPTS[prompt]
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, spans)
    kv = kv_drop._manager()
    layout = StreamLayout(spans=spans, growth=(("decode_loop", 1),))
    kv.ingest_request(R, KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode_loop": 50}))
    # a request dropped mid-prefill stops two slots short of its last step's end
    walk, length = steps[-1][0], steps[-1][2]
    done = length - 2 if mid_prefill else length
    for step_walk, inputs, step_length in steps:
        ran = done if step_walk == walk else step_length
        step = _ran(step_walk, {R: inputs})
        if ran < step_length:
            step.chunk_ranges = {R: ChunkRange(0, ran, step_length, ran)}
        rebuilds.record(step)
        _write_kv(kv, R, step_walk, ran)
    rebuilds.flush()
    if mid_prefill:
        prefilling, iters = Prefilling(walk, done, length), {}
    else:
        # the prefill's sample and one per decode step
        _write_kv(kv, R, DECODE, 1, 1)
        rebuilds._kept[NODE][R].sampled.extend([100, 101, 102])
        prefilling, iters = None, {"decode_loop": 2}
    assert rebuilds.refusal(R, NODE, walk if mid_prefill else DECODE, iters, prefilling) is None

    tokens = rebuilds.tokens(R, NODE, prefilling)
    assert kv.check_drop(R, tokens) is None, f"the drop refused {tokens} slots for a KV that holds them"
    kv.drop(R, tokens)
    held = kv._dropped[R].held
    rebuild = rebuilds.start(R, NODE, prefilling)
    if rebuild is not None:
        kv.prepare_rebuild(R, rebuild)
    while rebuilds.rebuilding(R, NODE):
        cursor = rebuilds.cursor(R, NODE)
        _write_kv(kv, R, cursor.walk, cursor.total)
        rebuilds.land_chunk(R, NODE, None)
    kv.land_rebuild(R)

    assert kv._streams[R]["main"].stored_len == held, "the rebuild wrote other slots than the drop checked"
    # the rest of the prompt, or the next decode step
    _write_kv(kv, R, *((walk, 2) if mid_prefill else (DECODE, 1)))


def test_a_request_dropped_again_after_its_rebuild_is_laid_out_past_what_that_rebuild_wrote():
    """A request back from a drop can be dropped again, since its pool returns it preemptible. Its log holds every
    id it sampled since its prompt, but its first rebuild had already laid out the ids sampled before it, and sending
    them all again laid the label out past what it held."""
    kv, rebuilds = kv_drop._manager(), Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    prompt = list(range(20))
    assert _ingest(rebuilds, R, (PrefixSpan(20, 20, PREFILL),))
    rebuilds.record(_ran(PREFILL, {R: {"text_inputs": [torch.tensor(prompt)]}}))
    sampled = rebuilds._kept[NODE][R].sampled
    kv.ingest_request(R, kv_drop._config(prompt))
    # the prefill's sample and two decode steps, then the drop and the rebuild
    sampled.extend([100, 101, 102])
    kv_drop._write(kv, R, 20, 1, 1)
    kv.drop(R, rebuilds.tokens(R, NODE, None))
    kv.prepare_rebuild(R, rebuilds.start(R, NODE, None))
    rebuilds.land_chunk(R, NODE, None)
    kv_drop._write(kv, R, 22)
    kv.land_rebuild(R)
    # two more decode steps, then the second drop
    sampled.extend([103, 104])
    kv_drop._write(kv, R, 1, 1)
    kv.drop(R, rebuilds.tokens(R, NODE, None))

    kv.prepare_rebuild(R, rebuilds.start(R, NODE, None))

    assert [(row.walk, row.total) for row in rebuilds.cursor(R, NODE).rows] == [(PREFILL, 24)], (
        "the rebuild row would leave out the ids sampled before the first drop"
    )
    fresh = kv_drop._manager()
    fresh.ingest_request("f", kv_drop._config(prompt + [100, 101, 102, 103]))
    fresh._streams["f"]["main"].chain.extend([104], kv_drop.PAGE_SIZE)
    rebuilt, want = kv._streams[R]["main"].chain, fresh._streams["f"]["main"].chain
    assert (rebuilt.keys, rebuilt.unkeyed) == (want.keys, want.unkeyed), (
        "the rebuilt pages would be filed under keys another request with these ids can't match"
    )


def test_a_rebuild_stays_open_until_its_last_row_lands():
    """A request partway through a rebuild of several rows could be picked again and start over. Its rebuild stays
    open, and its KV dropped, from the first row to the last."""
    rebuilds = _two_steps(6)
    rebuilds.start(R, NODE, None)

    first = rebuilds.land_chunk(R, NODE, None)

    assert first.row.walk == PREFILL and rebuilds.rebuilding(R, NODE), "the rebuild would land after its first row"
    assert rebuilds.land_chunk(R, NODE, None).ran == 8 and not rebuilds.rebuilding(R, NODE)


def test_only_the_last_prefill_steps_sample_is_kept():
    """A prefill step before the last draws a token the model throws away, so keeping it would refuse every drop."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, TWO_STEPS)

    for walk, inputs, token in ((PREFILL, TEXT_STEP, 40), (VISION, IMAGE_STEP, 41)):
        rebuilds.record(_ran(walk, {R: inputs}))
        rebuilds.keep_samples(_batch((R,)), {R: {"new_token": [torch.tensor([token])]}}, None)

    assert list(rebuilds._kept[NODE][R].sampled) == [41], "the first step's draw would be kept as a sample"
    assert R not in rebuilds._prefilling[NODE], "decode steps would look for the request's prefill steps"


class _Unread(dict):
    """A step's rows that fail the test when read."""

    def items(self):
        raise AssertionError("a decode step's rows were read")


def test_a_decode_step_reads_no_row_while_a_request_is_still_prefilling():
    """Under steady arrivals some request is always prefilling, and every decode step walked its rows."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, TWO_STEPS)

    rebuilds.record(_ran(DECODE, _Unread({R: {}})))


@pytest.mark.parametrize("step", ["another walk", "another length"])
def test_a_step_other_than_its_layouts_next_leaves_the_request_never_dropped(step):
    """A rebuild from a layout its prefill didn't follow would write other slots."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, TWO_STEPS)
    first = _ran(VISION if step == "another walk" else PREFILL, {R: TEXT_STEP})
    first.chunk_ranges = {} if step == "another walk" else {R: ChunkRange(0, 2, 5, 2)}

    rebuilds.record(first)

    assert "keeps no log" in rebuilds.refusal(R, NODE, DECODE, {}, None), "a rebuild would write other slots"


def test_a_chunked_steps_first_chunk_records_it_and_its_later_chunks_record_nothing():
    """A chunk after the first carries no inputs, since it is cut from what the first prepared, and recording it as
    a step of its own raised for a copied walk."""
    rebuilds = Rebuilds({NODE: MEDIA}, _graphs(), HostPool(1 << 10, torch.device("cpu")), StepAllowance(), DTYPES)
    assert _ingest(rebuilds, R, IMAGE_CHAT)

    chunks = [(ChunkRange(0, 4, 11, 4), IMAGE_STEP), (ChunkRange(4, 8, 11, 4), {}), (ChunkRange(8, 11, 11, 3), {})]
    for chunk, inputs in chunks:
        step = _ran(VISION, {R: inputs})
        step.chunk_ranges = {R: chunk}
        rebuilds.record(step)

    assert len(rebuilds._kept[NODE][R].records) == 1 and R not in rebuilds._prefilling[NODE], (
        "one step run in chunks would be recorded as several, or not at all"
    )


def test_a_request_dropped_in_a_later_prompt_step_reruns_the_steps_before_it_then_its_own_chunks():
    """A request mid-prefill in its second step holds the first step's KV too. Its rebuild reruns the first from the
    log and the second from the input the engine keeps for it, up to where it stood."""
    rebuilds = _two_steps()
    prefilling = Prefilling(VISION, 4, 11)

    why = rebuilds.refusal(R, NODE, VISION, {}, prefilling)
    rebuild = rebuilds.start(R, NODE, prefilling)
    cursor = rebuilds.cursor(R, NODE)

    assert why is None and rebuild is None, "the request's layout already says what its rebuild writes"
    assert [(row.walk, row.total, row.length, row.record is None) for row in cursor.rows] == [
        (PREFILL, 3, 3, False), (VISION, 4, 11, True),
    ], "the rebuild would write the second step's chunks where the first step's KV was"
    assert not cursor.resumes_prefill, "the first row would be cut from the second step's input"


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

    rebuilds.record(step)
    [record] = rebuilds._kept[NODE][R].records
    record.landed.synchronize()

    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated, (
        "a cast on the pool's stream made every later decode step 0.16 ms slower at Qwen3.5-9B TP2"
    )
    assert [ids.tolist() for ids in record.inputs["text_inputs"]] == [
        ids.tolist() for ids in torch.arange(256).tensor_split(tensors)
    ], "the rebuild would prefill other ids, or join the runs of text"
