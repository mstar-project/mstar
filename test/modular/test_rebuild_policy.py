"""A node declares how its dropped requests are rebuilt, and the worker keeps what that rebuild reads.

The declaration names the walk that re-prefills a request, the input that carries its ids, the output that carries
each sample and the loop that samples. Nothing about a rebuild is inferred from a graph's shape or a KV layout, so a
model either says how its requests come back or never has one dropped. The worker keeps each request's prompt ids
from the policy's input the first time a step runs the policy's walk, and each sample from the policy's output.
"""

from __future__ import annotations

import logging
from array import array
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild, SamplerStep, StepContext
from mstar.engine.resources.kv.config import PrefixSpan, StreamLayout
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows, chunk_loop_name
from mstar.worker.host_pool import HostCopy, HostPool, HostSlice
from mstar.worker.rebuild import KeptIds, NodeWrites, Prefilling, Rebuilds
from mstar.worker.worker import _LazyHostOutputs

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


@pytest.mark.parametrize("model_class, attrs", [
    (Qwen3_5DenseModel, {"config": SimpleNamespace(max_position_embeddings=4096), "vision_config": None}),
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
        assert chunk_loop_name(node, policy.walk) in graphs[policy.walk].get_loops(), "the walk runs no chunk Loop"
        policy.check(node, graphs)


def _rows(*rids: int) -> dict:
    return {rid: {"text_inputs": [torch.arange(rid, rid + 3)]} for rid in rids}


def _rebuilds(pool_bytes: int = 1 << 10, device: torch.device = torch.device("cpu")) -> Rebuilds:
    """`Rebuilds` with ``R``, ``A`` and ``B`` taken in as requests a drop on ``NODE`` can free."""
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(pool_bytes, device))
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"decode_loop"}))}}
    layouts = {"kv": [StreamLayout(spans=(PrefixSpan(3, 3, PREFILL),), growth=(("decode_loop", 1),))]}
    for rid in (R, A, B):
        assert rebuilds.ingest_request(rid, writes, layouts, unpausable=False) == {"kv": True}
    return rebuilds


def _ran(walk: str, inputs: dict, request_walks: dict | None = None) -> SimpleNamespace:
    """A step of ``walk`` that read ``inputs``, run on the host."""
    ctx = SimpleNamespace(graph_walk=walk, request_walks=request_walks or {})
    return SimpleNamespace(node_name=NODE, step_context=ctx, per_request_input_tensors=inputs, completion_event=None)


def _kept(prompt, *sampled: int) -> KeptIds:
    """Ids kept as `Rebuilds.keep_prompts` keeps them, the prompt on the host as the walk read it."""
    view = torch.tensor(prompt, dtype=torch.long)
    return KeptIds(HostCopy(HostSlice(0, view.nbytes), view, None), array("i", sampled))


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


def test_a_full_pool_keeps_no_ids():
    rebuilds = _rebuilds(32)

    rebuilds.keep_prompts(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}, R: {"text_inputs": [torch.arange(2)]}}))

    assert A in rebuilds._kept[NODE] and R not in rebuilds._kept[NODE], "two requests' ids would share host memory"


def test_a_full_pool_is_logged_once_and_not_per_request(caplog):
    rebuilds = _rebuilds(32)

    with caplog.at_level(logging.WARNING, logger="mstar.worker.rebuild"):
        for rid in (R, A, B):
            rebuilds.keep_prompts(_ran(PREFILL, {rid: {"text_inputs": [torch.arange(8)]}}))

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1 and "64-byte prompt, with 0 of 32 bytes in use" in lines[0], (
        "requests that can never be dropped would go unnoticed, or log a line each"
    )


def test_removing_a_request_gives_its_slice_back():
    rebuilds = _rebuilds(32)
    rebuilds.keep_prompts(_ran(PREFILL, {R: {"text_inputs": [torch.arange(4)]}}))

    rebuilds.remove_request(R)
    rebuilds.keep_prompts(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}}))

    assert A in rebuilds._kept[NODE], "the pool would fill with removed requests' ids and keep no more"


def _prompts() -> tuple[SimpleNamespace, SimpleNamespace]:
    """Steps that read the prompts of r and a, each of which fills a 32-byte pool."""
    return tuple(_ran(PREFILL, {rid: {"text_inputs": [torch.arange(4)]}}) for rid in (R, A))


def test_a_follower_keeps_the_ids_of_a_request_rank_0_keeps_when_its_own_pool_is_full():
    """Each rank gives a slice back at its own teardown. Rank 0 tears r down before it keeps a's prompt and the
    follower after, so only rank 0's pool has room for a's prompt. Rank 0 can then drop a, and a follower that kept
    no ids for a stopped. The follower keeps a's prompt outside its pool."""
    (leader, follower), (r, a) = (_rebuilds(32), _rebuilds(32)), _prompts()
    leader.keep_prompts(r)
    follower.keep_prompts(r, spill=True)
    leader.remove_request(R)

    assert leader.keep_prompts(a) == [] and follower.keep_prompts(a, spill=True) == []
    follower.remove_request(R)

    kept = follower.kept(A, NODE)
    assert leader.kept(A, NODE) is not None
    assert kept is not None, "rank 0 can drop a request the follower keeps no ids for"
    assert kept.prompt.view.tolist() == [0, 1, 2, 3] and follower.pool.nbytes_used() == 0, (
        "the follower kept other ids, or a's copy takes room in its pool"
    )


def test_a_follower_gives_back_the_ids_of_a_request_rank_0_kept_none_for():
    """Rank 0's pool has no room for a's prompt, so a keeps no ids and rank 0 never drops it. The follower kept a's
    prompt outside its pool until rank 0's journal said so, and kept it again from a later step."""
    (leader, follower), (r, a) = (_rebuilds(32), _rebuilds(32)), _prompts()
    leader.keep_prompts(r)
    follower.keep_prompts(r, spill=True)
    assert leader.keep_prompts(a) == [A]
    follower.keep_prompts(a, spill=True)

    follower.forget(A, NODE)
    follower.keep_prompts(a, spill=True)

    assert follower.kept(A, NODE) is None, "the follower holds ids outside its pool that rank 0 never drops with"


def test_a_request_whose_prompt_the_pool_couldnt_hold_keeps_no_ids_from_a_later_step():
    """Rank 0 decides once whether a request keeps ids, at the first step that reads its prompt. A later chunk that
    found room kept a's ids on rank 0 after rank 0 had journaled that a keeps none, which the follower had replayed."""
    rebuilds, (r, a) = _rebuilds(32), _prompts()
    rebuilds.keep_prompts(r)
    assert rebuilds.keep_prompts(a) == [A]
    rebuilds.remove_request(R)

    assert rebuilds.keep_prompts(a) == [] and rebuilds.kept(A, NODE) is None


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
    prompt = torch.tensor([1, 2], dtype=torch.long)
    copying = HostCopy(HostSlice(0, prompt.nbytes), prompt, SimpleNamespace(query=lambda: False))
    rebuilds._kept[NODE][R] = KeptIds(copying, array("i", [5]))

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


def test_the_sampler_commits_no_draw_of_a_rebuild_row_whatever_the_model_declared():
    ctx = StepContext(request_ids=(R, A), graph_walk=PREFILL, slot=0, capture=False, rebuild_rids=frozenset({R}))
    for kept_rids in (None, frozenset({R, A})):
        kept = SamplerResource._kept_rids_of(SamplerStep(kept_rids=kept_rids), ctx)

        assert kept == {A}, "the rebuild row's sample would overwrite decode's next input"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="counts the device's allocations")
def test_keeping_a_prompt_runs_nothing_on_the_device_but_its_copy():
    device = torch.device("cuda")
    rebuilds = _rebuilds(1 << 12, device)
    prompt = torch.arange(256, device=device)
    step_done = torch.Event(device=device)
    step_done.record()
    step = _ran(PREFILL, {R: {"text_inputs": [prompt]}})
    step.completion_event = step_done
    allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]

    rebuilds.keep_prompts(step)
    kept = rebuilds._kept[NODE][R]
    kept.prompt.landed.synchronize()

    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated, (
        "a cast on the pool's stream made every later decode step 0.16 ms slower at Qwen3.5-9B TP2"
    )
    assert kept.prompt.view.tolist() == list(range(256)), "the rebuild would prefill other ids"
