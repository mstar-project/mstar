"""The Praxis backbone uses one episode array for both text streams."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.dense_window import DenseWindowPool
from mstar.engine.resources.step import StepContext
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.engine.state.episodes import EpisodeRowBindings
from mstar.model.praxis1.praxis1_model import TEXT_STATE_NAME, Praxis1Model
from mstar.model.praxis1.submodules import Praxis1BackboneSubmodule


def test_backbone_reuses_cond_and_blank_after_a_new_request():
    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        max_context_chunks=1, max_text_tokens=4,
    )
    model.config = replace(model.config, text_hidden_dim=4)
    text_array = next(array for array in model.declare_state() if array.name == TEXT_STATE_NAME)
    (spec,) = build_specs([text_array])
    pool = DenseWindowPool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    handles = node_handles(SimpleNamespace(declare_state=lambda: [text_array]), "backbone", {TEXT_STATE_NAME: pool})
    node = Praxis1BackboneSubmodule(
        model.config, torch.nn.Identity(), episode_text=True, attention_resources=False,
    )
    node.bind_node_state(handles)
    owners = EpisodeRowBindings()

    def call(rid, text=None, *, reset=False):
        pool.ingest_request(rid)
        binding = owners.bind(rid, 0, "robot")
        pool.bind_episode(rid, 0, "robot", cold=binding.cold, reset=reset)
        info = CurrentForwardPassInfo(
            request_id=rid, graph_walk="chunk", fwd_index=0, random_seed=7, max_tokens=1,
            step_metadata={"text_write": text is not None, "text_tokens": 2, "blank_tokens": 1},
            dynamic_loop_iter_counts={"praxis_denoise": 0},
        )
        inputs = {"obs_latents": [torch.zeros(1, 4, 6, 1, 1)],
                  "anchor_latents": [torch.zeros(1, 4, 1, 1, 1)],
                  "clean_latents": [torch.empty(1, 4, 0, 1, 1)]}
        if text is not None:
            inputs["text_embeds"] = [text[0]]
            inputs["text_embeds_blank"] = [text[1]]
        prepared = node.prepare_inputs("chunk", info, inputs)
        step = node.declare_step("chunk", [rid], [prepared])
        planned = state_steps(step, handles, graph_walk="chunk")
        ctx = StepContext(request_ids=[rid], graph_walk="chunk", slot=0, capture=False)
        planned.set_ctx(ctx)
        assert pool.admit(planned.steps[TEXT_STATE_NAME], ctx).ok
        pool.plan(planned.steps[TEXT_STATE_NAME], ctx)
        planned.begin_calls()
        got = node.episode_text_embeddings(
            prepared.tensor_inputs.get("text_embeds"), prepared.tensor_inputs.get("text_embeds_blank"),
            **{key: prepared.kwargs[key] for key in ("write_text", "text_tokens", "blank_tokens")},
        )
        planned.check_calls()
        pool.commit(planned.steps[TEXT_STATE_NAME], ctx)
        planned.commit_calls()
        planned.end_calls()
        lengths = (pool.logical_length(rid, "cond"), pool.logical_length(rid, "blank"))
        pool.remove_request(rid)
        owners.unbind(rid)
        return prepared, got, lengths

    cond = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=torch.bfloat16)
    blank = torch.tensor([[[9, 10, 11, 12]]], dtype=torch.bfloat16)
    first_inputs, first, first_lengths = call(1, (cond, blank))
    warm_inputs, warm, warm_lengths = call(2)
    assert torch.equal(first[0], cond) and torch.equal(first[1], blank)
    assert torch.equal(warm[0], cond) and torch.equal(warm[1], blank)
    assert "text_embeds" not in warm_inputs.tensor_inputs
    assert torch.equal(first_inputs.tensor_inputs["anchor_latents"], warm_inputs.tensor_inputs["anchor_latents"])
    assert first_lengths == warm_lengths == (2, 1)
    new_cond, new_blank = cond + 10, blank + 10
    _, after_reset, reset_lengths = call(3, (new_cond, new_blank), reset=True)
    assert torch.equal(after_reset[0], new_cond) and torch.equal(after_reset[1], new_blank)
    assert reset_lengths == (2, 1)
