"""CPU structural checks for the Cosmos3-Edge serving surface (dummy mode).

A minimal Edge-shaped checkpoint directory (JSON configs only, no weights) is
written to ``tmp_path`` so the model parses the Edge backbone family and the
reasoner without the real snapshot; ``skip_weight_loading`` keeps every node
weightless. Covers the walks, the shared resources, the request state machine
for text (reasoner) vs media (generator) requests, the per-request resource
configs, the worker-graph split of ``configs/cosmos3_edge.yaml`` and the chat
adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs
from mstar.model.cosmos3.cosmos3_model import (
    DIT_NODE,
    REASONER_NODE,
    VISION_ENCODER_NODE,
    Cosmos3Model,
)
from mstar.model.cosmos3.submodules import ATTN, KV_CACHE, REASONER_DECODE_LOOP, SAMPLER

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

_TRANSFORMER = {
    "_class_name": "Cosmos3OmniTransformer",
    "action_dim": 64, "action_gen": True, "attention_bias": False, "backbone_type": "cosmos3_edge_nemotron_dense",
    "base_fps": 24, "enable_fps_modulation": True, "head_dim": 128, "hidden_act": "relu2", "hidden_size": 2048,
    "intermediate_size": 9216, "latent_channel": 48, "latent_patch_size": 2, "num_attention_heads": 16,
    "num_embodiment_domains": 32, "num_hidden_layers": 28, "num_key_value_heads": 8, "patch_latent_dim": 192,
    "qk_norm_for_text": False, "rms_norm_eps": 1e-05, "rope_scaling": {"mrope_section": [24, 20, 20]},
    "rope_theta": 100000000, "sound_dim": None, "sound_gen": False, "sound_latent_fps": 25,
    "temporal_compression_factor": 4, "timestep_scale": 0.001, "unified_3d_mrope_reset_spatial_ids": True,
    "unified_3d_mrope_temporal_modality_margin": 15000, "use_und_k_norm_for_gen": True, "vocab_size": 131072,
}
_TOP = {
    "architectures": ["Cosmos3EdgeForConditionalGeneration"], "model_type": "cosmos3_edge",
    "image_token_id": 19, "video_token_id": 18, "vision_start_token_id": 20, "vision_end_token_id": 21,
    "projector_config": {"input_hidden_size": 1152, "merger_intermediate_size": 11520, "out_hidden_size": 2048,
                         "spatial_merge_size": 2, "use_postshuffle_norm": False},
    "text_config": {"eos_token_id": 11, "hidden_size": 2048, "max_position_embeddings": 131072},
    "vision_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                      "num_hidden_layers": 27, "num_patches": 256, "patch_size": 16, "spatial_merge_size": 2},
}


def _fake_edge_dir(tmp_path: Path) -> Path:
    root = tmp_path / "edge"
    (root / "transformer").mkdir(parents=True)
    (root / "vision_encoder").mkdir()
    (root / "transformer" / "config.json").write_text(json.dumps(_TRANSFORMER))
    (root / "config.json").write_text(json.dumps(_TOP))
    (root / "model_index.json").write_text(json.dumps({"use_native_flow_schedule": True}))
    return root


def _model(tmp_path: Path, **kwargs) -> Cosmos3Model:
    return Cosmos3Model(model_path_hf=str(_fake_edge_dir(tmp_path)), skip_weight_loading=True, **kwargs)


def test_edge_dummy_model_parses_family_and_reasoner(tmp_path) -> None:
    model = _model(tmp_path)
    cfg = model.config
    assert cfg.hidden_act == "relu2" and cfg.use_und_k_norm_for_gen and not cfg.qk_norm_for_text
    assert cfg.use_native_flow_schedule and cfg.serves_reasoner
    walks = model.get_graph_walk_graphs()
    assert {Cosmos3Model.REASONER_PREFILL_WALK, Cosmos3Model.REASONER_PREFILL_VISION_WALK,
            Cosmos3Model.REASONER_DECODE_WALK} <= set(walks)
    # No sound walk: the Edge checkpoint has no sound pathway.
    assert Cosmos3Model.VIDEO_SOUND_GEN_WALK not in walks
    assert set(model.nodes) == {DIT_NODE, REASONER_NODE, VISION_ENCODER_NODE, "vae_encoder", "vae_decoder"}

    prefill_vision = walks[Cosmos3Model.REASONER_PREFILL_VISION_WALK]
    assert isinstance(prefill_vision, Sequential)
    enc, reasoner = prefill_vision.sections
    assert isinstance(enc, GraphNode) and enc.name == VISION_ENCODER_NODE
    assert any(e.next_node == REASONER_NODE and e.name == "vision_embeds" for e in enc.outputs)
    assert reasoner.name == REASONER_NODE
    assert any(e.next_node == EMIT_TO_CLIENT and e.output_modality == "text" and e.persist for e in reasoner.outputs)

    decode = walks[Cosmos3Model.REASONER_DECODE_WALK]
    assert isinstance(decode, Loop) and decode.name == REASONER_DECODE_LOOP
    body = decode.section
    assert body.name == REASONER_NODE and set(body.input_names) == {"text_inputs"}
    assert {e.name for e in body.outputs} == {"new_token", "text_inputs"}


def test_edge_recipe_defaults_from_the_checkpoint(tmp_path) -> None:
    """An Edge checkpoint loads the model card's serving recipe without a
    yaml (480p video at flow shift 12, 640x640 images, the diffusers-0.40
    conditioning crop); yaml model_kwargs still override it."""
    cfg = _model(tmp_path).config
    assert cfg.conditioning_resize == "aspect_crop" and cfg.flow_shift_video == 12.0
    assert cfg.video_size_default == (832, 480) and cfg.image_size_default == (640, 640)
    assert cfg.num_frames_video == 121 and cfg.num_inference_steps_video == 20
    assert cfg.flow_shift_action == 10.0
    over = Cosmos3Model(
        model_path_hf=str(tmp_path / "edge"), skip_weight_loading=True,
        conditioning_resize="stretch", flow_shift_video=9.0,
    ).config
    assert over.conditioning_resize == "stretch" and over.flow_shift_video == 9.0
    # Nano keeps its defaults.
    nano = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    assert nano.conditioning_resize == "stretch" and nano.flow_shift_video is None


def test_edge_resources_shared_between_dit_and_reasoner(tmp_path) -> None:
    model = _model(tmp_path)
    specs = {s.resource_key: s for s in model.get_node_resources()}
    assert specs[KV_CACHE].nodes == {DIT_NODE, REASONER_NODE}
    assert specs[ATTN].nodes == {DIT_NODE, REASONER_NODE}
    assert specs[SAMPLER].nodes == {REASONER_NODE}
    assert specs[SAMPLER].vocab_size == 131072
    # Nano-shaped defaults declare no reasoner resources.
    nano = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    assert SAMPLER not in {s.resource_key for s in nano.get_node_resources()}
    assert Cosmos3Model.REASONER_DECODE_WALK not in nano.get_graph_walk_graphs()


def test_enable_reasoner_false_serves_generator_only(tmp_path) -> None:
    model = _model(tmp_path, enable_reasoner=False)
    assert Cosmos3Model.REASONER_DECODE_WALK not in model.get_graph_walk_graphs()
    assert REASONER_NODE not in model.nodes
    with pytest.raises(ValueError, match="reasoner"):
        model.get_initial_forward_pass_args("default", ["text"], ["text"], {"text_inputs": [], "position_ids": []})


def test_edge_yaml_splits_worker_graphs(tmp_path) -> None:
    model = _model(tmp_path)
    graphs = model.get_worker_graphs(str(CONFIGS / "cosmos3_edge.yaml"))
    by_walk = {}
    for wg in graphs:
        for walk in wg.graph_walks:
            by_walk.setdefault(walk, set()).update(wg.section.get_nodes())
    assert by_walk[Cosmos3Model.REASONER_DECODE_WALK] == {REASONER_NODE}
    assert by_walk[Cosmos3Model.REASONER_PREFILL_VISION_WALK] == {VISION_ENCODER_NODE, REASONER_NODE}
    assert by_walk[Cosmos3Model.IMAGE_GEN_WALK] == {DIT_NODE, "vae_decoder"}
    # Every rank-0 group; the dit + reasoner pair shares one group (a
    # resource spanning both nodes requires it).
    for wg in graphs:
        assert wg.ranks == [0]


def test_reasoner_request_state_machine(tmp_path) -> None:
    model = _model(tmp_path)
    sig = {k: [object()] for k in ("text_inputs", "position_ids", "pixel_values", "vision_grid_thw")}
    fpa = model.get_initial_forward_pass_args("default", ["image", "text"], ["text"], sig, {"max_output_tokens": 32})
    assert fpa.full_metadata.graph_walk == Cosmos3Model.REASONER_PREFILL_VISION_WALK
    assert fpa.full_metadata.is_prefill
    routed = {(e.next_node, e.name) for e in fpa.inputs}
    assert routed == {(REASONER_NODE, "text_inputs"), (REASONER_NODE, "position_ids"),
                      (VISION_ENCODER_NODE, "pixel_values"), (VISION_ENCODER_NODE, "vision_grid_thw")}
    # Text-only prompts skip the encoder.
    text_only = model.get_initial_forward_pass_args(
        "default", ["text"], ["text"], {"text_inputs": [object()], "position_ids": [object()]},
    )
    assert text_only.full_metadata.graph_walk == Cosmos3Model.REASONER_PREFILL_WALK

    token = object()
    nxt = model.get_partition_forward_pass_args("default", fpa.full_metadata, {"new_token": [token]})
    assert nxt.full_metadata.graph_walk == Cosmos3Model.REASONER_DECODE_WALK
    assert not nxt.full_metadata.is_prefill and not nxt.request_done
    assert [(e.next_node, e.name, e.tensor_info) for e in nxt.inputs] == [(REASONER_NODE, "text_inputs", [token])]
    done = model.get_partition_forward_pass_args("default", nxt.full_metadata, {})
    assert done.request_done

    # Text requests open the reasoner label + sampler; media requests keep
    # the two guidance labels and no sampler.
    rc = model.get_request_resource_configs({"default": fpa}, {"temperature": 0.0, "top_k": 5})
    assert set(rc) == {KV_CACHE, SAMPLER}
    assert rc[KV_CACHE].needed_labels == ["main"]
    assert rc[SAMPLER].temperature == 0.0 and rc[SAMPLER].top_k == 5
    gen = model.get_initial_forward_pass_args("default", ["text"], ["image"], {"text_inputs": [object()]}, {})
    assert gen.full_metadata.graph_walk == Cosmos3Model.PREFILL_WALK
    grc = model.get_request_resource_configs({"default": gen}, {})
    assert set(grc) == {KV_CACHE} and grc[KV_CACHE].needed_labels == ["main", "uncond"]


def test_edge_generation_defaults_from_yaml_knobs(tmp_path) -> None:
    model = _model(
        tmp_path, image_size_default=[640, 640], video_size_default=[832, 480],
        num_frames_video=121, num_inference_steps_video=20, guidance_scale=6.0, flow_shift_video=12.0,
        flow_shift_action=10.0,
    )
    p = model._resolve_gen_params({}, ["text"], ["video"])
    assert (p["width"], p["height"], p["num_frames"]) == (832, 480, 121)
    assert p["num_inference_steps"] == 20 and p["guidance_scale"] == 6.0 and p["flow_shift"] == 12.0
    p = model._resolve_gen_params({}, ["image", "text"], ["video"])
    assert p["has_image_condition"] and p["flow_shift"] == 12.0
    p = model._resolve_gen_params({}, ["text"], ["image"])
    assert (p["width"], p["height"]) == (640, 640) and p["flow_shift"] == 3.0
    p = model._resolve_gen_params(
        {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10}, ["image", "text"], ["action"],
    )
    assert p["flow_shift"] == 10.0 and (p["width"], p["height"]) == (832, 480)
    # Explicit request values still win.
    p = model._resolve_gen_params({"size": "320x192", "flow_shift": 5.0, "num_frames": 9}, ["text"], ["video"])
    assert (p["width"], p["height"], p["num_frames"], p["flow_shift"]) == (320, 192, 9, 5.0)


def test_reasoner_submodule_step_and_stop(tmp_path) -> None:
    import types

    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule

    model = _model(tmp_path)
    sub = Cosmos3ReasonerSubmodule(transformer=None, config=model.config)
    assert sub.eos_token_id == 11

    ids = torch.tensor([5, 6, 7, 19, 19, 8])
    pos = torch.zeros(3, 6, dtype=torch.long)
    pos[:, :] = torch.arange(6)
    fwd = types.SimpleNamespace(request_id="r", step_metadata={})
    inp = sub.prepare_inputs(
        Cosmos3Model.REASONER_PREFILL_VISION_WALK, fwd,
        {"text_inputs": [ids], "position_ids": [pos], "vision_embeds": [torch.zeros(2, 8)]},
    )
    assert inp.input_seq_len == 6 and sub.request_state("r")["next_pos"] == 6
    step = sub.declare_step(Cosmos3Model.REASONER_PREFILL_VISION_WALK, ["r"], [inp])
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [("r", "main", 6)]
    assert step.steps[KV_CACHE].commit and step.steps[ATTN].causal
    assert torch.equal(step.steps[SAMPLER].prefill_tracked_tokens["r"], ids)

    dec = sub.prepare_inputs(Cosmos3Model.REASONER_DECODE_WALK, fwd, {"text_inputs": [torch.tensor([42])]})
    assert dec.input_seq_len == 1 and dec.tensor_inputs["position_ids"].tolist() == [[6], [6], [6]]
    assert sub.request_state("r")["next_pos"] == 7
    step = sub.declare_step(Cosmos3Model.REASONER_DECODE_WALK, ["r"], [dec])
    assert step.steps[SAMPLER].prefill_tracked_tokens == {}

    info = types.SimpleNamespace(
        resource_configs={SAMPLER: types.SimpleNamespace(ignore_eos=False)},
        dynamic_loop_iter_counts={REASONER_DECODE_LOOP: 3}, max_tokens=100,
    )
    assert sub.check_stop("r", info, {"new_token": [torch.tensor([11])]}) == {REASONER_DECODE_LOOP}
    assert sub.check_stop("r", info, {"new_token": [torch.tensor([12])]}) == set()
    info.max_tokens = 5
    assert sub.check_stop("r", info, {"new_token": [torch.tensor([12])]}) == {REASONER_DECODE_LOOP}
    info.resource_configs[SAMPLER].ignore_eos = True
    info.max_tokens = 100
    assert sub.check_stop("r", info, {"new_token": [torch.tensor([11])]}) == set()
    out = {"new_token": [torch.tensor([12])]}
    sub.postprocess("r", info, out)
    assert out["text_inputs"] is out["new_token"]


def test_edge_chat_adapter(tmp_path) -> None:
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import ChatCompletionRequest

    adapter = get_adapter("cosmos3_edge")
    assert adapter is not None and adapter.supports_chat and adapter.supports_videos and adapter.supports_images
    req = ChatCompletionRequest(
        model="cosmos3_edge",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Describe."},
        ]}],
        temperature=0.0, max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    args = adapter.chat_to_request(req, upload_dir=tmp_path)
    assert args.output_modalities == ["text"]
    assert args.input_modalities == ["image", "text"]
    assert args.file_paths and args.file_paths["image"]
    assert args.model_kwargs["temperature"] == 0.0
    assert args.model_kwargs["max_output_tokens"] == 64
    assert args.model_kwargs["enable_thinking"] is False
    assert [p.modality for p in args.prompt_parts] == ["image", "text"]


def test_forward_pass_args_type(tmp_path) -> None:
    model = _model(tmp_path)
    fpa = model.get_initial_forward_pass_args(
        "default", ["text"], ["text"], {"text_inputs": [object()], "position_ids": [object()]},
    )
    assert isinstance(fpa, ForwardPassArgs)
