"""CPU node handoff and Euler arithmetic against the pinned Praxis sampler."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("praxis", reason="the Praxis-1 reference is not on PYTHONPATH")
from praxis.sampler import ACTION_SALT, VIDEO_SALT, guidance_combine, seeded_normal, window_time_grid

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state.adapters import build_specs, node_handles, state_steps
from mstar.graph.base import GraphEdge
from mstar.model.base import EpisodeView
from mstar.model.praxis1.components.action_expert import _RESOURCE_MASK
from mstar.model.praxis1.components.encoder import TapFeatures
from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.episode_history import STATE_NAME
from mstar.model.praxis1.praxis1_model import Praxis1Model
from mstar.model.praxis1.submodules import (
    DECODER_BLANK_KV,
    DECODER_COND_KV,
    ENCODER_BLANK_KV,
    ENCODER_COND_KV,
    EXPERT_QUERIES,
    EXPERT_TRUNK_KEYS,
    Praxis1ActionExpertSubmodule,
    Praxis1BackboneSubmodule,
    Praxis1VaeEncoderSubmodule,
)


class _State:
    def __init__(self):
        self.calls = []

    def add(self, layer, *, k, v):
        self.calls.append(("add", layer, k.shape[1], v.shape[1]))

    def read(self, layer, q):
        self.calls.append(("read", layer, q.shape[1]))
        return q


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward_chunk_causal(self, text, anchor, clean, noised, times, *, attend_for_layer):
        self.calls += 1
        k = times.shape[1]
        f = noised.shape[2] // k
        clean_len = 0 if clean is None else clean.shape[2]
        g = 2 + anchor.shape[-2] * anchor.shape[-1] + text.shape[1]
        seq = g + noised.shape[2] + clean_len
        dummy = torch.zeros(1, 1, seq, 2)
        attend_for_layer(0)(dummy, dummy, dummy, None)
        embeddings = torch.full((1, seq, 4), text.mean())
        features = TapFeatures(
            taps=[embeddings.clone()], embeddings=embeddings,
            frames_per_chunk=f, tokens_per_frame=1, rope_ids=torch.arange(seq)[None],
            global_spans={"register": (0, 2), "anchor": (2, 1), "text": (3, text.shape[1])},
            noised_span=(g, k * f), clean_span=(g + k * f, clean_len) if clean_len else None,
            n_chunks=k,
        )
        return torch.full((1, k * f, 1, 1, 4), text.mean()), features


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_chunk_causal(self, noised, cond, times, *, attend_for_layer):
        self.calls += 1
        seq = 2 + noised.shape[2] * 4
        dummy = torch.zeros(1, 1, seq, 2)
        attend_for_layer(0)(dummy, dummy, dummy, None)
        return noised * 0.25 + cond.mean() * 0.5


class _Expert(nn.Module):
    global_order = ("register", "anchor", "text")
    num_future_per_chunk = 2

    def forward_trunk_rope(self, *, noised_action_latents, action_timesteps, proprio_latents,
                           trunk_features, global_tokens, kv_rope_ids, chunk_size,
                           num_clean_frames, num_noised_frames, tokens_per_frame, window_size,
                           cross_mask, self_mask, attend_for_layer):
        assert cross_mask is self_mask is None
        assert len(trunk_features) == 1
        assert num_noised_frames == noised_action_latents.shape[1]
        assert num_clean_frames + num_noised_frames == window_size
        assert kv_rope_ids.shape[1] == global_tokens.shape[1] + sum(
            x.shape[1] for x in trunk_features
        )
        q_len = noised_action_latents.shape[1] // chunk_size * (1 + self.num_future_per_chunk + chunk_size)
        q = torch.zeros(1, 1, q_len, 2)
        kv = torch.zeros(1, 1, kv_rope_ids.shape[1], 2)
        attend_for_layer(0, True)(q, kv, kv, _RESOURCE_MASK)
        attend_for_layer(1, False)(q, q, q, _RESOURCE_MASK)
        return noised_action_latents * 0.1 + action_timesteps[:, :, None] * 0.3 + proprio_latents.mean() * 0.2


def _info(step: int):
    return CurrentForwardPassInfo(
        request_id="r", graph_walk="chunk", fwd_index=step, random_seed=17, max_tokens=1,
        step_metadata={"num_steps": 2, "t_shift": 1.0, "cfg_video_scale": 4.0, "cfg_scale": 1.0,
                       "text_tokens": 3, "blank_tokens": 2},
        dynamic_loop_iter_counts={"praxis_denoise": step},
    )


@pytest.mark.parametrize("depth", [0, 1])
def test_cold_and_warm_node_flow_matches_reference_sampler(depth: int) -> None:
    config = replace(Praxis1Config.for_rig("single_arm"), latent_channels=1, latents_per_chunk=6)
    encoder, decoder = _Encoder(), _Decoder()
    backbone = Praxis1BackboneSubmodule(
        config, SimpleNamespace(encoder=encoder, decoder=decoder),
    )
    expert = Praxis1ActionExpertSubmodule(config, _Expert())
    states = {name: _State() for name in (
        ENCODER_COND_KV, ENCODER_BLANK_KV, DECODER_COND_KV, DECODER_BLANK_KV,
        EXPERT_QUERIES, EXPERT_TRUNK_KEYS,
    )}
    backbone.state_handles = states
    expert.state_handles = states
    k = depth + 1
    text = torch.full((1, 3, 2), 0.5)
    blank = torch.full((1, 2, 2), -0.25)
    proprio = torch.full((1, k, config.proprio_patch_dim), 0.4)
    carried = {
        "video_latents": seeded_normal((1, 4, k * 6, 1, 1), 17, VIDEO_SALT),
        "action_latents": seeded_normal((1, k * 6, config.action_patch_dim), 17, ACTION_SALT),
    }
    for step in range(2):
        input_tensors = {
            "obs_latents": [torch.zeros(1, 4, 1, 1, 1)],
            "text_embeds": [text], "text_embeds_blank": [blank],
            "proprio": [proprio[:, -1:]],
        }
        if depth:
            input_tensors.update(anchor_latents=[torch.zeros(1, 4, 1, 1, 1)],
                                 clean_latents=[torch.zeros(1, 4, depth * 6, 1, 1)],
                                 proprio_all=[proprio])
        input_tensors.update({name: [value] for name, value in carried.items()})
        prepared = backbone.prepare_inputs("chunk", _info(step), input_tensors)
        expected_resources = ({ENCODER_COND_KV} if step == 1 else {
            ENCODER_COND_KV, ENCODER_BLANK_KV, DECODER_COND_KV, DECODER_BLANK_KV,
        })
        assert set(backbone.declare_step("chunk", ["r"], [prepared]).state) == expected_resources
        out = backbone.forward("chunk", None, **prepared.tensor_inputs, **prepared.kwargs)
        expert_inputs = {name: value for name, value in out.items()}
        expert_inputs["proprio"] = [proprio[:, -1:]]
        if depth:
            expert_inputs["proprio_all"] = [proprio]
        prepared_expert = expert.prepare_inputs("chunk", _info(step), expert_inputs)
        assert set(expert.declare_step("chunk", ["r"], [prepared_expert]).state) == {
            EXPERT_QUERIES, EXPERT_TRUNK_KEYS,
        }
        carried = {name: value[0] for name, value in expert.forward(
            "chunk", None, **prepared_expert.tensor_inputs, **prepared_expert.kwargs,
        ).items()}

    ts = window_time_grid(2, 1.0)
    expected_video = seeded_normal((1, 4, k * 6, 1, 1), 17, VIDEO_SALT)
    expected_action = seeded_normal((1, k * 6, config.action_patch_dim), 17, ACTION_SALT)
    for step in range(2):
        cond_velocity = expected_video * 0.25 + text.mean() * 0.5
        blank_velocity = expected_video * 0.25 + blank.mean() * 0.5
        if step != 1:
            expected_video = expected_video + (ts[step + 1] - ts[step]) * guidance_combine(
                cond_velocity, blank_velocity, 4.0,
            )
        action_velocity = expected_action * 0.1 + ts[step] * 0.3 + proprio.mean() * 0.2
        expected_action = expected_action + (ts[step + 1] - ts[step]) * action_velocity
    torch.testing.assert_close(carried["video_latents"], expected_video, rtol=0, atol=0)
    torch.testing.assert_close(carried["action_latents"], expected_action, rtol=0, atol=0)
    assert encoder.calls == 3  # two guided passes, then the final conditional taps
    assert decoder.calls == 2  # both decoder passes belong to the first Euler step
    for state in states.values():
        assert state.calls


@pytest.mark.parametrize("held,rows,clean_frames", [(0, 0, 0), (1, 21, 6)])
def test_episode_layout_plans_both_guidance_branches_on_shared_resource(
    held: int, rows: int, clean_frames: int,
) -> None:
    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        max_context_chunks=1, max_text_tokens=8,
    )
    arrays = [array for array in model.declare_state() if array.name in {
        ENCODER_COND_KV, ENCODER_BLANK_KV, DECODER_COND_KV, DECODER_BLANK_KV,
    }]
    specs = build_specs(arrays)
    assert len(specs) == 2  # one encoder and one decoder resource, with separate branch labels
    resources = {spec.resource_key: SimpleNamespace(run=lambda *args, **kw: None) for spec in specs}
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "backbone", resources)
    node = Praxis1BackboneSubmodule(model.config, SimpleNamespace())
    episode = EpisodeView("session", 0, {STATE_NAME: held} if held else {})
    walk = model.walk_plan(
        "", ["image"], ["action"], {"image_inputs": [object()]},
        {"executed_rows": rows, "text_tokens": 3, "blank_tokens": 2,
         "action_format": "positronic"}, episode,
    )[-1]
    from mstar.conductor.conductor import _with_planned_state_layout

    (walk,) = _with_planned_state_layout([walk], model.declare_state())
    info = CurrentForwardPassInfo(
        request_id="r", graph_walk="chunk", fwd_index=0, random_seed=1, max_tokens=1,
        step_metadata=walk.step_metadata,
    )
    tensors = {
        "obs_latents": [torch.zeros(1, 192, 1, 7, 26)],
        "anchor_latents": [torch.zeros(1, 192, 1, 7, 26)],
        "clean_latents": [torch.zeros(1, 192, clean_frames, 7, 26)],
        "text_embeds": [torch.zeros(1, 3, 2048)],
        "text_embeds_blank": [torch.zeros(1, 2, 2048)],
    }
    prepared = node.prepare_inputs("chunk", info, tensors)
    expanded = state_steps(
        node.declare_step("chunk", ["r"], [prepared]), handles, per_request_info={"r": info},
    )
    assert len(expanded.steps["praxis_encoder_cond_attn"].segments) == 2
    assert len(expanded.steps["praxis_decoder_cond_attn"].segments) == 2


@pytest.mark.parametrize("held,rows,warm", [(0, 0, False), (1, 21, True)])
def test_graph_backbone_is_ready_after_actual_producers(held: int, rows: int, warm: bool) -> None:
    model = Praxis1Model(
        "runwayml/praxis-1-droid", skip_weight_loading=True,
        max_context_chunks=1, max_text_tokens=8,
    )
    episode = EpisodeView("session", 1, {STATE_NAME: held, **({"text_embeds": 5} if warm else {})})
    walk = model.walk_plan(
        "", ["image"], ["action"], {"image_inputs": [object()]},
        {"executed_rows": rows, "text_tokens": 3, "blank_tokens": 2,
         "action_format": "positronic"}, episode,
    )[-1]
    graph = model.get_graph_walk_graphs()[walk.walk]
    backbone = graph.get_nodes()["backbone"]
    names = {edge.name for edge in graph.get_nodes()["vae_encoder"].outputs if edge.next_node == "backbone"}
    if not warm:
        names |= {"text_embeds", "text_embeds_blank"}
    for name in names:
        backbone.ready_signals.update(GraphEdge(name=name, next_node="backbone"))
    assert backbone.ready_signals.is_ready
    external = {name for name, _ in graph.get_inputs_outputs().ext_inputs}
    assert not {"video_latents", "action_latents"} & external
    assert ({"text_embeds", "text_embeds_blank"} <= external) is not warm


@pytest.mark.parametrize("chunks", [1, 2])
def test_vae_seeds_both_loopback_edges_from_reference_streams(chunks: int) -> None:
    config = Praxis1Config.for_rig("single_arm")
    vae = SimpleNamespace(config=SimpleNamespace(z_dim=config.latent_channels))
    node = Praxis1VaeEncoderSubmodule(config, vae)
    latent = torch.zeros(1, 192, 1, 2, 3)
    seeded = node._seed_window(latent, chunks, 17)
    assert torch.equal(seeded["video_latents"][0], seeded_normal((1, 192, chunks * 6, 2, 3), 17, VIDEO_SALT))
    assert torch.equal(seeded["action_latents"][0], seeded_normal((1, chunks * 6, 40), 17, ACTION_SALT))
