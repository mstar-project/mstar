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
import logging
from pathlib import Path

import pytest
import torch
from jinja2 import TemplateError

from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.cosmos3.cosmos3_model import (
    DIT_NODE,
    REASONER_NODE,
    VISION_ENCODER_NODE,
    Cosmos3Model,
)
from mstar.model.cosmos3.submodules import ATTN, KV_CACHE, REASONER_DECODE_LOOP, SAMPLER
from mstar.model.multimodal import MEDIA_LAYOUT

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


def test_text_request_idles_the_window_decoder_partition(tmp_path) -> None:
    """With windowed serving on, a text (reasoner) request still gets the
    decode walk on the window_decoder partition — never a reasoner walk that
    partition cannot run — and the default partition gets the reasoner
    prefill."""
    from mstar.model.cosmos3 import constants as C

    model = _model(tmp_path, enable_windowed_video=True)
    signals = {"text_inputs": []}
    mk = {"max_output_tokens": 8}
    (dec,) = model.walk_plan(C.WINDOW_DECODER_PARTITION, ["text"], ["text"], signals, mk)
    assert (dec.walk, dict(dec.step_metadata)) == (C.VIDEO_DECODE_AR_WALK, {})
    main = model.walk_plan("default", ["text"], ["text"], signals, mk)
    assert main[0].walk == C.REASONER_PREFILL_WALK


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
    assert nano.conditioning_resize == "aspect_crop" and nano.flow_shift_video is None


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
        model.walk_plan("default", ["text"], ["text"], {"text_inputs": []}, {})


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


@pytest.mark.parametrize("vision", [True, False])
def test_reasoner_plan_prefills_then_decodes_the_first_token(tmp_path, vision) -> None:
    model = _model(tmp_path)
    names = ("text_inputs",) + (("pixel_values", "vision_grid_thw") if vision else ())
    layout = {"offset": [3], "tokens": [8], "output": ["vision_embeds"], "grid": [[1, 4, 8]]}
    knobs = {"max_output_tokens": 32, "enable_thinking": True}
    plan = model.walk_plan("default", ["image", "text"] if vision else ["text"], ["text"],
                           {k: [object()] for k in names}, {**knobs, MEDIA_LAYOUT: layout})
    # the reasoner splices the encoder's tokens and places its positions by the intake's layout
    prefill = (Cosmos3Model.REASONER_PREFILL_VISION_WALK, {**knobs, MEDIA_LAYOUT: layout}) if vision else (
        Cosmos3Model.REASONER_PREFILL_WALK, knobs)
    assert [(step.walk, dict(step.bind), dict(step.step_metadata)) for step in plan] == [
        (prefill[0], {}, prefill[1]),
        (Cosmos3Model.REASONER_DECODE_WALK, {"text_inputs": "new_token"}, knobs),
    ]


def test_reasoner_opens_its_label_and_sampler_and_media_the_guidance_labels(tmp_path) -> None:
    model = _model(tmp_path)
    rc = model.get_request_resource_configs(["text"], ["text"], {"temperature": 0.0, "top_k": 5})
    assert set(rc) == {KV_CACHE, SAMPLER}
    assert rc[KV_CACHE].needed_labels == ["main"]
    assert rc[SAMPLER].temperature == 0.0 and rc[SAMPLER].top_k == 5
    grc = model.get_request_resource_configs(["text"], ["image"], {})
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


def test_edge_windowed_rollout_defaults_to_chained_61_32(tmp_path) -> None:
    """Edge's streamed rollout runs chained 61-frame windows re-pinning 32
    frames (16 and 8 latent frames) unless the request names otherwise; kv
    keeps its 29-frame windows; Nano keeps 29/8."""
    model = _model(tmp_path, enable_windowed_video=True)
    cfg = model.config
    assert (cfg.window_frames_default, cfg.overlap_frames_default, cfg.kv_window_frames_default) == (61, 32, 29)
    for mk in ({"stream_video": True}, {"window_mode": "chained"}):
        p = model._resolve_gen_params({"num_frames": 241, **mk}, ["image", "text"], ["video"])
        assert p["window_mode"] == "chained"
        assert (p["window_latent_units"], p["overlap_latent_units"]) == (16, 8)
        assert p["num_windows"] == 7
    kv = model._resolve_gen_params({"num_frames": 241, "window_mode": "kv"}, ["image", "text"], ["video"])
    assert (kv["window_latent_units"], kv["overlap_latent_units"], kv["context_latent_units"]) == (8, 0, 16)
    own = model._resolve_gen_params(
        {"num_frames": 241, "window_mode": "chained", "window_frames": 29, "overlap_frames": 8},
        ["image", "text"], ["video"],
    )
    assert (own["window_latent_units"], own["overlap_latent_units"]) == (8, 2)
    yaml = _model(tmp_path / "yaml", enable_windowed_video=True, window_frames_default=45, overlap_frames_default=16)
    p = yaml._resolve_gen_params({"num_frames": 241, "stream_video": True}, ["image", "text"], ["video"])
    assert (p["window_latent_units"], p["overlap_latent_units"]) == (12, 4)
    nano = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, enable_windowed_video=True)
    p = nano._resolve_gen_params({"num_frames": 241, "stream_video": True}, ["text"], ["video"])
    assert (p["window_latent_units"], p["overlap_latent_units"]) == (8, 2)


@pytest.mark.parametrize("num_frames", [60, 61, 62, 63, 64, 65, 94, 95, 96, 97])
def test_edge_windowed_schedule_covers_every_requested_frame(tmp_path, num_frames) -> None:
    """The schedule generates at least ``num_frames`` frames (the decoder trims
    the rest); 62-64 used to fit one 61-frame window and came back short."""
    model = _model(tmp_path, enable_windowed_video=True)
    tf = model.config.vae.scale_factor_temporal
    p = model._resolve_gen_params({"num_frames": num_frames, "stream_video": True}, ["image", "text"], ["video"])
    assert 1 + (p["total_latent_units"] - 1) * tf >= num_frames


def test_edge_resumed_session_re_pins_the_chained_overlap(tmp_path) -> None:
    """A resumed session conditions window 0 on the stored tail, as many latent
    frames as the chained overlap (at least 2): 8 under Edge's 61/32 default,
    2 with 29/8, and the schedule grows by them."""
    model = _model(tmp_path, enable_windowed_video=True)
    resume = {"stream_video": True, "session_id": "s", "resume_session": True}
    p = model._resolve_gen_params({"num_frames": 61, **resume}, ["text"], ["video"])
    assert (p["resume_latent_units"], p["overlap_latent_units"]) == (8, 8)
    assert p["total_latent_units"] == 16 + 8
    p = model._resolve_gen_params(
        {"num_frames": 61, "window_frames": 29, "overlap_frames": 8, **resume}, ["text"], ["video"])
    # 16 + 2, padded to whole windows (8 latent frames, stride 6)
    assert (p["resume_latent_units"], p["total_latent_units"]) == (2, 20)
    p = model._resolve_gen_params({"num_frames": 61, "stream_video": True, "session_id": "s"}, ["text"], ["video"])
    assert p["resume_latent_units"] == 0


def test_kv_windows_flagged_experimental_at_startup(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="mstar.model.cosmos3.cosmos3_model"):
        _model(tmp_path / "plain")
    assert "experimental" not in caplog.text
    with caplog.at_level(logging.WARNING, logger="mstar.model.cosmos3.cosmos3_model"):
        _model(tmp_path / "windowed", enable_windowed_video=True)
    assert "window_mode='kv' is experimental" in caplog.text


def test_distilled_checkpoint_refuses_stream_video(tmp_path) -> None:
    root = _fake_edge_dir(tmp_path)
    (root / "modular_model_index.json").write_text(
        json.dumps({"is_distilled": True, "distilled_sigmas": [1.0, 0.75, 0.5, 0.25]})
    )
    model = Cosmos3Model(model_path_hf=str(root), skip_weight_loading=True, enable_windowed_video=True)
    assert model.config.is_distilled
    with pytest.raises(ValueError, match="distilled"):
        model._resolve_gen_params({"num_frames": 121, "stream_video": True}, ["image", "text"], ["video"])


def test_reasoner_submodule_places_positions_and_stops(tmp_path) -> None:
    """A text prompt leaves its positions to the position resource; a vision
    prompt places its own from the intake's layout and moves the counter to
    ``max(position) + 1``. A stop id ends the decode loop, and on either
    prefill walk it, or a one-token budget, ends the partition."""
    import types

    from mstar.conductor.request_info import FINAL_PREFILL
    from mstar.engine.resources.position.config import PositionConfig
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.model.components.mrope import layout_position_ids
    from mstar.model.cosmos3.components.reasoner import MediaGrid, vision_layout
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule
    from mstar.model.keys import ROPE
    from mstar.model.submodule_base import END_PARTITION

    model = _model(tmp_path)
    r = model.config.reasoner
    sub = Cosmos3ReasonerSubmodule(transformer=None, config=model.config, encoders=(model._vision_encoder(),))
    sub.get_device = lambda: torch.device("cpu")
    sub.node_resources = {ROPE: RopeManager(PositionConfig(kv_cache=KV_CACHE), torch.device("cpu"))}

    ids = torch.tensor([5, 6, 7] + [r.image_token_id] * 8 + [8, 9])
    layout = vision_layout(ids, r, [MediaGrid(1, 4, 8)], [], "vision_embeds")
    fwd = types.SimpleNamespace(request_id="r", rid_handle="r", step_metadata={MEDIA_LAYOUT: layout})
    walk = Cosmos3Model.REASONER_PREFILL_VISION_WALK
    inp = sub.prepare_inputs(walk, fwd, {"text_inputs": [ids], "vision_embeds": [torch.zeros(8, 4)]})
    pos, advance = layout_position_ids(layout, r.vision.spatial_merge_size, len(ids))
    assert torch.equal(inp.custom_pos_ids, pos) and inp.resource_step_info == advance == int(pos.max()) + 1
    step = sub.declare_step(walk, ["r"], [inp])
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [("r", "main", 13)]
    assert step.steps[ROPE].advance == (advance,)
    assert torch.equal(step.steps[SAMPLER].prefill_tracked_tokens["r"], ids)

    text = sub.prepare_inputs(Cosmos3Model.REASONER_PREFILL_WALK, fwd, {"text_inputs": [ids[:3]]})
    assert text.custom_pos_ids is None
    assert sub.declare_step(Cosmos3Model.REASONER_PREFILL_WALK, ["r"], [text]).steps[ROPE].advance is None

    def stops(walk, token, ignore_eos=False, max_tokens=100):
        info = types.SimpleNamespace(
            graph_walk=walk, max_tokens=max_tokens,
            resource_configs={SAMPLER: types.SimpleNamespace(ignore_eos=ignore_eos)},
            # the conductor marks the prefill, the step before decode, as the reply's first token's
            step_metadata={} if walk == Cosmos3Model.REASONER_DECODE_WALK else {FINAL_PREFILL: True},
        )
        return sub.check_stop("r", info, {"new_token": [torch.tensor([token])]})

    eos = r.eos_token_id
    assert stops(Cosmos3Model.REASONER_DECODE_WALK, eos) == {REASONER_DECODE_LOOP}
    assert stops(Cosmos3Model.REASONER_DECODE_WALK, 12) == set()
    assert stops(Cosmos3Model.REASONER_DECODE_WALK, eos, ignore_eos=True) == set()
    assert stops(Cosmos3Model.REASONER_PREFILL_WALK, eos) == {END_PARTITION}
    assert stops(Cosmos3Model.REASONER_PREFILL_VISION_WALK, 12, max_tokens=1) == {END_PARTITION}
    assert stops(Cosmos3Model.REASONER_PREFILL_VISION_WALK, 12, max_tokens=2) == set()


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


class _StubRefusingTokenizer:
    """The released template refuses no chat, but a checkpoint can ship its own."""

    def apply_chat_template(self, messages, **kwargs):
        raise TemplateError("System message must be at the beginning.")


def test_a_chat_the_template_refuses_is_a_bad_request(tmp_path) -> None:
    model = _model(tmp_path)
    model.tokenizer = _StubRefusingTokenizer()
    with pytest.raises(ValueError, match="System message must be at the beginning"):
        model.process_prompt("What is ahead?", ["text"], ["text"])

