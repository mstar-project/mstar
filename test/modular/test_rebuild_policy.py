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

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import Rebuild, SamplerStep, StepContext
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import HostRows, chunk_loop_name
from mstar.worker.host_pool import HostCopy, HostPool, HostSlice
from mstar.worker.rebuild import KeptIds, Rebuilds
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


def _rebuilds() -> Rebuilds:
    return Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu")))


def _ran(walk: str, inputs: dict, request_walks: dict | None = None) -> SimpleNamespace:
    """A step of ``walk`` that read ``inputs``, run on the host."""
    ctx = SimpleNamespace(graph_walk=walk, request_walks=request_walks or {})
    return SimpleNamespace(node_name=NODE, step_context=ctx, per_request_input_tensors=inputs, completion_event=None)


def _kept(prompt, *sampled: int) -> KeptIds:
    """Ids kept as `Rebuilds.keep_prompts` keeps them, the prompt on the host as int32."""
    view = torch.tensor(prompt, dtype=torch.int32)
    return KeptIds(HostCopy(HostSlice(0, view.nbytes), view, None), torch.long, array("i", sampled))


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


def test_a_full_pool_keeps_no_ids():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(16, torch.device("cpu")))

    rebuilds.keep_prompts(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}, R: {"text_inputs": [torch.arange(2)]}}))

    assert A in rebuilds._kept[NODE] and R not in rebuilds._kept[NODE], "two requests' ids would share host memory"


def test_removing_a_request_gives_its_slice_back():
    rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(16, torch.device("cpu")))
    rebuilds.keep_prompts(_ran(PREFILL, {R: {"text_inputs": [torch.arange(4)]}}))

    rebuilds.remove_request(R)
    rebuilds.keep_prompts(_ran(PREFILL, {A: {"text_inputs": [torch.arange(4)]}}))

    assert A in rebuilds._kept[NODE], "the pool would fill with removed requests' ids and keep no more"


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
    copying = HostCopy(HostSlice(0, prompt.nbytes), prompt, SimpleNamespace(query=lambda: False))
    rebuilds._kept[NODE][R] = KeptIds(copying, torch.long, array("i", [5]))

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


def test_the_sampler_commits_no_draw_of_a_rebuild_row_whatever_the_model_declared():
    ctx = StepContext(request_ids=(R, A), graph_walk=PREFILL, slot=0, capture=False, rebuild_rids=frozenset({R}))
    for kept_rids in (None, frozenset({R, A})):
        kept = SamplerResource._kept_rids_of(SamplerStep(kept_rids=kept_rids), ctx)

        assert kept == {A}, "the rebuild row's sample would overwrite decode's next input"
