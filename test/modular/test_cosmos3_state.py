"""Cosmos3 on the state contract.

Cosmos3 declares its state rather than hand-writing its resource specs and steps, and what a request carries from one
step to the next rides on ``PROGRESS`` and the loop's edges rather than in node-held request state. The specs built
from the declaration are main's but for the nodes the ``kv`` array spans and the walks the deployment serves; a step
over both guidance branches plans under the contract's combined label; a windowed request's later windows draw the
noise main's kept generator drew; and the DiT alone keeps a session, the decoder taking its context from the DiT.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn.config import AttnBackend
from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.step import StepContext
from mstar.engine.state.adapters import node_resources, state_setup
from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import DIT_NODE, REASONER_NODE, Cosmos3Model
from mstar.model.cosmos3.submodules import (
    ATTN,
    ATTN_GEN,
    COND_LABEL,
    IMAGE_GEN_LOOP,
    PROGRESS,
    SAMPLER,
    SESSION_TAIL,
    UNCOND_LABEL,
    VIDEO_GEN_AR_LOOP,
    WINDOW_COND,
    Cosmos3DiTSubmodule,
    Cosmos3VAEDecoderARSubmodule,
    GenStepInfo,
)
from mstar.model.cosmos3.tests.stubs import StubSampler, bind_progress
from mstar.model.cosmos3.tests.test_edge import _tiny_edge_config
from mstar.model.cosmos3.tests.test_engine_cache import _SdpaResources, dit_handles
from mstar.model.cosmos3.tests.test_rotary_table import stand_in_rotary
from mstar.model.cosmos3.tests.test_video_capture import _dit_with_buckets
from mstar.model.keys import ROPE, VISION_ATTN
from mstar.model.submodule_base import ARNodeInputs, engine_step

CPU = torch.device("cpu")


def _edge(**overrides) -> Cosmos3Model:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    model.config = _tiny_edge_config(**overrides)
    return model


def _specs(model: Cosmos3Model) -> dict:
    setup, problems = state_setup({}, model.declare_state(), model.kernel_settings())
    assert not problems, problems
    return {spec.resource_key: spec for spec in node_resources(model, setup=setup)}


def _dit(config: Cosmos3Config) -> Cosmos3DiTSubmodule:
    dit = Cosmos3DiTSubmodule(transformer=None, config=config, scheduler=None)
    dit.transformer = SimpleNamespace(
        proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)), _rotary=stand_in_rotary(config),
    )
    dit.get_device = lambda: CPU
    return dit


def test_the_declared_specs_are_mains_but_for_the_nodes_the_kv_array_spans():
    specs = _specs(_edge())
    assert set(specs) == {C.KV_CACHE, ATTN, ATTN_GEN, ROPE, C.SEQ_SHARD, SAMPLER, VISION_ATTN}
    # main bound the generation attention to the DiT alone and the positions to the reasoner alone
    assert specs[ATTN_GEN].nodes == specs[ROPE].nodes == specs[C.KV_CACHE].nodes == {DIT_NODE, REASONER_NODE}
    # main listed the windowed and sound walks whether or not the deployment served them
    assert set(specs[C.SEQ_SHARD].walks) == {C.IMAGE_GEN_WALK, C.VIDEO_GEN_WALK, C.ACTION_GEN_WALK,
                                             C.ACTION_VIDEO_GEN_WALK}
    assert specs[VISION_ATTN].config.backend is RaggedAttnBackend.DENSE, "FlashInfer would pad a head dim of 72"


def test_the_windowed_edge_config_resolves_on_dense_gen():
    specs = _specs(_edge(attention_backend="dense_gen", enable_windowed_video=True))
    assert C.VIDEO_GEN_AR_WALK in specs[C.SEQ_SHARD].walks
    assert specs[ATTN_GEN].config.backend is AttnBackend.DENSE, "an uncaptured denoise step would run the paged kernel"


def test_a_guided_denoise_step_plans_both_branches_under_one_combined_label():
    model = _edge()
    dit = _dit(model.config)
    dit.bind_node_state(dit_handles(model, _SdpaResources().as_dict())[0])
    both = ARNodeInputs(input_seq_len=8, resource_step_info=GenStepInfo(cfg=True, cfg_active=True, capture_key=None))
    step = engine_step(dit, graph_walk=C.IMAGE_GEN_WALK, request_ids=["r"], inputs=[both])
    assert step.steps[C.KV_CACHE].combined_labels == {(COND_LABEL, UNCOND_LABEL): "main+uncond"}, (
        "main planned the pair under its own _cfg_batched label"
    )


def test_a_windowed_request_draws_each_window_from_where_progress_left_its_generator(monkeypatch):
    """Main kept a windowed request's generator in node state from one window to the next. A window's first step now
    records where the generator stands under ``PROGRESS``, and the next window's first step draws from there."""
    dit = _dit(Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config)
    monkeypatch.setattr(dit, "_new_scheduler", lambda *a, **k: StubSampler(torch.arange(2)))
    dit._solver_state = ()
    fields = bind_progress(dit, "r")
    (progress,) = [array for array in Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).declare_state()
                   if array.name == PROGRESS]
    md = {
        "window_mode": "chained", "total_latent_units": 6, "window_latent_units": 3, "overlap_latent_units": 0,
        "height": 64, "width": 64, "num_inference_steps": 2, "guidance_scale": 1.0, C.PROMPT_LENGTHS: (7, 0),
    }
    fwd = SimpleNamespace(
        request_id="r", rid_handle="r", random_seed=11, step_metadata=md, dynamic_loop_iter_counts={},
    )
    edges = {WINDOW_COND: [torch.zeros(0)], SESSION_TAIL: [torch.zeros(0)]}
    drawn = []
    for k in range(4):  # two windows of two steps
        fwd.dynamic_loop_iter_counts[VIDEO_GEN_AR_LOOP] = k
        inputs = dit.prepare_inputs(C.VIDEO_GEN_AR_WALK, fwd, {**edges, **({"latents": [torch.zeros(1)]} if k else {})})
        step = dit.declare_step(C.VIDEO_GEN_AR_WALK, ["r"], [inputs])
        if k % 2:
            assert PROGRESS not in step.state, "a step that draws nothing has nothing to record"
            continue
        drawn.append(inputs.tensor_inputs["latents"])
        fields.commit(progress, step.state[PROGRESS], ["r"])
    generator = torch.Generator().manual_seed(11)
    for window, latents in enumerate(drawn):
        assert torch.equal(latents, torch.randn(latents.shape, generator=generator)), f"window {window}'s noise"


def test_a_replayed_distilled_step_leaves_the_generator_where_the_next_step_draws(monkeypatch):
    """A distilled step draws its SDE noise in ``prepare_inputs`` and records where the generator stands under
    ``PROGRESS``. A captured step declares that write on every replay, and a replay that dropped it would draw the
    first step's noise again at every step, with no error."""
    dit, _ = _dit_with_buckets(monkeypatch, guidance_scale=1.0, is_distilled=True, distilled_sigmas=(1.0, 0.75, 0.5))
    dit.config.scheduler.stochastic_sampling = True
    monkeypatch.setattr(dit, "get_device", lambda: CPU)
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    model.config = dit.config
    handles, setup = dit_handles(model, _SdpaResources().as_dict())
    dit.bind_node_state(handles)
    setup.host_fields.open("r")
    md = {"height": 64, "width": 64, "guidance_scale": 1.0, "num_inference_steps": 3, C.PROMPT_LENGTHS: (7, 0)}
    fwd = SimpleNamespace(
        request_id="r", rid_handle="r", random_seed=11, step_metadata=md, dynamic_loop_iter_counts={},
    )
    lease, noise, latents = SimpleNamespace(), [], None
    for k in range(3):
        fwd.dynamic_loop_iter_counts[IMAGE_GEN_LOOP] = k
        inputs = dit.prepare_inputs(C.IMAGE_GEN_WALK, fwd, {"latents": [latents]} if k else {})
        latents = inputs.tensor_inputs["latents"]
        noise.append(inputs.kwargs["settings"]["sde_noise"])
        step = engine_step(dit, graph_walk=C.IMAGE_GEN_WALK, request_ids=["r"], inputs=[inputs], slot_lease=lease)
        ctx = StepContext(request_ids=["r"], graph_walk=C.IMAGE_GEN_WALK, slot=0, capture=False)
        ctx.slot_lease = lease
        step.set_ctx(ctx)
        assert step.cg_key_info is not None and ctx.replays, "the step would run eager"
        # the calls a replay makes around its captured graph, then its commit
        step.begin_calls()
        step.end_calls()
        step.commit_calls()
    generator = torch.Generator().manual_seed(11)
    torch.randn(latents.shape, generator=generator, dtype=latents.dtype)  # the request's initial latents
    for k, drawn in enumerate(noise):
        assert torch.equal(drawn, torch.randn(drawn.shape, generator=generator)), (
            f"step {k} drew other noise than the request's generator gives after {k} steps"
        )


def test_the_dit_alone_keeps_a_session_and_refuses_a_missing_one_at_the_prefill():
    config = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    assert not hasattr(Cosmos3VAEDecoderARSubmodule(None, config), "_sessions"), (
        "a second store would age its sessions apart from the DiT's"
    )
    dit = _dit(config)
    md = {
        "window_mode": "chained", "total_latent_units": 6, "window_latent_units": 3, "overlap_latent_units": 1,
        "height": 64, "width": 64, "session_id": "gone", "resume_latent_units": 1, C.PROMPT_LENGTHS: (7, 0),
        "guidance_scale": 1.0,
    }
    fwd = SimpleNamespace(request_id="r", rid_handle="r", random_seed=0, step_metadata=md)
    with pytest.raises(ValueError, match="unknown or expired session 'gone'"):
        dit.prepare_inputs(C.PREFILL_WALK, fwd, {"text_inputs": [torch.arange(7)]})
