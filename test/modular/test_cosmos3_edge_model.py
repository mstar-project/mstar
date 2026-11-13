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

from mstar.conductor.request_info import FINAL_PREFILL
from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.components.sampled_token_decoder import MEDIA_EMBEDS, MEDIA_MASK
from mstar.model.cosmos3.cosmos3_model import (
    DIT_NODE,
    REASONER_NODE,
    VISION_ENCODER_NODE,
    Cosmos3Model,
)
from mstar.model.cosmos3.submodules import ATTN, KV_CACHE, REASONER_DECODE_LOOP, SAMPLER
from mstar.model.keys import ROPE
from mstar.model.multimodal import MEDIA_LAYOUT
from mstar.model.submodule_base import chunk_loop_name

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
    # the prompt prefills in chunks: the reasoner runs in its chunk Loop
    assert isinstance(reasoner, Loop)
    assert reasoner.name == chunk_loop_name(REASONER_NODE, Cosmos3Model.REASONER_PREFILL_VISION_WALK)
    assert reasoner.section.name == REASONER_NODE
    # the final chunk's token leaves the Loop for the client
    assert any(e.next_node == EMIT_TO_CLIENT and e.output_modality == "text" and e.persist for e in reasoner.outputs)

    decode = walks[Cosmos3Model.REASONER_DECODE_WALK]
    assert isinstance(decode, Loop) and decode.name == REASONER_DECODE_LOOP
    body = decode.section
    assert body.name == REASONER_NODE and set(body.input_names) == {"text_inputs"}
    # the VLM scaffold's loop: the token loops back and goes to the client as text_inputs
    assert {(e.next_node, e.name) for e in body.outputs} == {
        (REASONER_NODE, "text_inputs"), (EMIT_TO_CLIENT, "text_inputs"),
    }


def test_text_request_idles_the_window_decoder_partition(tmp_path) -> None:
    """With windowed serving on, a text (reasoner) request still gets the
    decode walk on the window_decoder partition — never a reasoner walk that
    partition cannot run — and the default partition gets the reasoner
    prefill."""
    from mstar.model.cosmos3 import constants as C

    model = _model(tmp_path, enable_windowed_video=True)
    signals = {"text_inputs": [object()]}
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
    # the reasoner's position counter is its own; mixed steps split the shared attention
    assert specs[ROPE].nodes == {REASONER_NODE}
    # the vision tower's attention: a segment per frame, down to the smallest video frame (16 patches)
    assert specs["vision_attn"].nodes == {VISION_ENCODER_NODE}
    assert specs["vision_attn"].config.max_segments_per_step == 16384 // 16
    assert [split.combined_walk for split in specs[ATTN].config.splits] == ["mixed"]
    # text and media prompt rows both join decode steps, on the prefill kernels
    assert model.get_combined_graph_walks() == {REASONER_NODE: {"mixed": {"prefill_text", "prefill_vision", "decode"}}}
    assert specs[ATTN].config.splits[0].prefill_walks == ("prefill_text", "prefill_vision")
    # Nano-shaped defaults declare no reasoner resources.
    nano = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    nano_keys = {s.resource_key for s in nano.get_node_resources()}
    assert SAMPLER not in nano_keys and ROPE not in nano_keys
    assert nano.get_combined_graph_walks() == {}
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
    names = ("text_inputs",) + (("pixel_values",) if vision else ())
    # as process_prompt sets it: a text-only prompt's layout places no media
    layout = {"offset": [3], "tokens": [8], "output": ["vision_embeds"], "grid": [[1, 4, 8]]} if vision else {
        "offset": [], "tokens": [], "output": [], "grid": []}
    knobs = {"max_output_tokens": 32, "enable_thinking": True}
    plan = model.walk_plan("default", ["image", "text"] if vision else ["text"], ["text"],
                           {k: [object()] for k in names}, {**knobs, MEDIA_LAYOUT: layout})
    # the reasoner splices the encoder's tokens and places its positions by the intake's layout
    first = {FINAL_PREFILL: True}
    prefill = (Cosmos3Model.REASONER_PREFILL_VISION_WALK, {**knobs, MEDIA_LAYOUT: layout, **first}) if vision else (
        Cosmos3Model.REASONER_PREFILL_WALK, {**knobs, **first})
    assert [(step.walk, dict(step.bind), dict(step.step_metadata)) for step in plan] == [
        (prefill[0], {}, prefill[1]),
        (Cosmos3Model.REASONER_DECODE_WALK, {"text_inputs": "new_token"}, knobs),
    ]
    if vision:
        # media without a layout to place them by is refused
        with pytest.raises(ValueError, match="no media layout"):
            model.walk_plan("default", ["image", "text"], ["text"], {k: [object()] for k in names}, knobs)


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

    from mstar.engine.resources.position.config import PositionConfig
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.model.components.mrope import layout_position_ids
    from mstar.model.cosmos3.components.reasoner import MediaGrid, vision_layout
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule
    from mstar.model.submodule_base import END_PARTITION, StopKind

    model = _model(tmp_path)
    r = model.config.reasoner
    sub = Cosmos3ReasonerSubmodule(transformer=None, config=model.config, encoders=(model._vision_encoder(),))
    sub.get_device = lambda: torch.device("cpu")
    sub.node_resources = {ROPE: RopeManager(PositionConfig(kv_cache=KV_CACHE), torch.device("cpu"))}
    sub.node_resources[ROPE].ingest_request("r")

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
    # a replay would otherwise splice the previous step's rows
    with pytest.raises(ValueError, match="rows for the prompt's 8 pads"):
        sub.prepare_inputs(walk, fwd, {"text_inputs": [ids], "vision_embeds": [torch.zeros(7, 4)]})

    text = sub.prepare_inputs(Cosmos3Model.REASONER_PREFILL_WALK, fwd, {"text_inputs": [ids[:3]]})
    assert text.custom_pos_ids is None
    assert sub.declare_step(Cosmos3Model.REASONER_PREFILL_WALK, ["r"], [text]).steps[ROPE].advance is None
    # a decode step prepared again after a refused admit feeds the same position:
    # preparing moves nothing, the position resource advances when the step commits
    rope = sub.node_resources[ROPE]
    before = rope.position(rid="r", label="main")
    rows = [sub.prepare_inputs(Cosmos3Model.REASONER_DECODE_WALK, fwd, {"text_inputs": [torch.tensor([42])]})
            for _ in range(2)]
    assert all(row.custom_pos_ids is None for row in rows) and rope.position(rid="r", label="main") == before
    # decode rows read their token off the sampler, and the graph runtime checks their stop id
    assert sub.device_loopback.walk == Cosmos3Model.REASONER_DECODE_WALK
    policy = sub.get_stop_policies(Cosmos3Model.REASONER_DECODE_WALK)[REASONER_DECODE_LOOP]
    assert policy.kind == StopKind.EOS and set(policy.eos_tokens) == {r.eos_token_id}

    def stops(walk, token, ignore_eos=False, max_tokens=100):
        info = types.SimpleNamespace(
            graph_walk=walk, max_tokens=max_tokens,
            resource_configs={SAMPLER: types.SimpleNamespace(ignore_eos=ignore_eos)},
            # the walk plan marks the prefill, the step before decode, as the reply's first token's
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


def test_reasoner_rotary_tables_come_from_the_layout_and_the_counter(tmp_path) -> None:
    """The tower takes rotary tables, built by the DiT transformer's own rotary:
    an eager vision prefill's from the positions its layout places, based at
    the counter; a captured decode step's from the counter's planned positions
    on all three axes, inside the graph."""
    import types

    from torch import nn

    from mstar.model.components.mrope import layout_position_ids
    from mstar.model.cosmos3.components.reasoner import MediaGrid, vision_layout
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer, Cosmos3RotaryEmbedding
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule

    model = _model(tmp_path)
    r, cfg = model.config.reasoner, model.config

    class Tower(nn.Module):
        _rotary = Cosmos3OmniTransformer._rotary
        build_cos_sin = Cosmos3OmniTransformer.build_cos_sin

        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 4)
            self.rotary_emb = Cosmos3RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.rope_axes_dim)

    tower = Tower()
    sub = Cosmos3ReasonerSubmodule(transformer=tower, config=cfg, encoders=(model._vision_encoder(),))
    sub.get_device = lambda: torch.device("cpu")
    planned = torch.tensor([40, 41])
    sub.node_resources = {ROPE: types.SimpleNamespace(position=lambda rid, label: 17, pos_ids=lambda label: planned)}

    ids = torch.tensor([5, 6, 7] + [r.image_token_id] * 8 + [8, 9])
    layout = vision_layout(ids, r, [MediaGrid(1, 4, 8)], [], "vision_embeds")
    fwd = types.SimpleNamespace(request_id="r", rid_handle="r", step_metadata={MEDIA_LAYOUT: layout})
    walk = Cosmos3Model.REASONER_PREFILL_VISION_WALK
    row = sub.prepare_inputs(walk, fwd, {"text_inputs": [ids], "vision_embeds": [torch.zeros(8, 4)]})
    eager = types.SimpleNamespace(captured=False)
    tensors = sub.preprocess(walk, eager, [row])
    pos, _ = layout_position_ids(layout, r.vision.spatial_merge_size, len(ids), start=17)
    cos, sin = sub.decoder_kwargs(walk, eager, len(ids), **tensors)["cos_sin"]
    want = tower.build_cos_sin(pos, torch.float32)
    assert torch.equal(cos, want[0]) and torch.equal(sin, want[1])

    cos, sin = sub.decoder_kwargs(Cosmos3Model.REASONER_DECODE_WALK, types.SimpleNamespace(captured=True), 2)["cos_sin"]
    want = tower.build_cos_sin(planned.expand(3, -1), torch.float32)
    assert torch.equal(cos, want[0]) and torch.equal(sin, want[1])


def test_a_mixed_step_carries_media_prompt_rows(tmp_path) -> None:
    """A media prompt row joins a decode step: it advances the position
    counter by its MRoPE advance beside the decode rows' one, its media and
    mask are packed beside the decode rows' loop-back tokens (the mask False
    on them), every mixed step stages a mask, media or not, under the keys
    the mixed capture holds, and a mixed step holds at most the scaffold's cap
    of prompt rows."""
    import types

    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    model = _model(tmp_path)
    sub = Cosmos3ReasonerSubmodule(transformer=None, config=model.config, encoders=(model._vision_encoder(),))
    image_pad = model.config.reasoner.image_token_id
    decode = ARNodeInputs(input_seq_len=1)
    decode.graph_walk = Cosmos3Model.REASONER_DECODE_WALK
    mask = torch.tensor([False] * 5 + [True] * 4 + [False])
    media = ARNodeInputs(
        input_seq_len=10, input_ids=torch.tensor([5, 6, 7, 8, 9] + [image_pad] * 4 + [10]),
        resource_step_info=8, tensor_inputs={MEDIA_EMBEDS: torch.ones(4, 8), MEDIA_MASK: mask},
    )
    media.graph_walk = Cosmos3Model.REASONER_PREFILL_VISION_WALK
    step = sub.declare_step("mixed", ["d", "m"], [decode, media])
    assert step.steps[ROPE].advance == (1, 8)
    assert sub.max_rows_in_combined(Cosmos3Model.REASONER_PREFILL_VISION_WALK) == sub.MIXED_MAX_PREFILL_ROWS

    sampler = types.SimpleNamespace(loopback_tokens=lambda: torch.tensor([42, 0]))
    engine_inputs = types.SimpleNamespace(resources={SAMPLER: sampler}, request_ids=["d", "m"])
    out = SampledTokenDecoder.preprocess(sub, "mixed", engine_inputs, [decode, media])
    assert out["input_ids"].tolist() == [42, 5, 6, 7, 8, 9] + [image_pad] * 4 + [10]
    assert torch.equal(out[MEDIA_EMBEDS], torch.ones(4, 8))
    assert out[MEDIA_MASK].tolist() == [False] + mask.tolist()

    (mixed,) = [c for c in sub.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == "mixed"]
    assert set(out) - {"input_ids"} == set(mixed.input_seq_dims)
    # decode and text rows alone: the replay would otherwise splice the rows a previous step left
    text = ARNodeInputs(input_seq_len=3, input_ids=torch.tensor([5, 6, 7]))
    text.graph_walk = Cosmos3Model.REASONER_PREFILL_WALK
    out = SampledTokenDecoder.preprocess(sub, "mixed", engine_inputs, [decode, text])
    assert MEDIA_EMBEDS not in out and out[MEDIA_MASK].tolist() == [False] * 4


def test_a_mixed_step_that_can_carry_media_keeps_host_positions(tmp_path, monkeypatch) -> None:
    """With ``ingraph_decode_rope`` on, text prefill builds its rope in the graph,
    but a mixed step keeps host-built 3D positions while media rows can join it."""
    import types

    from mstar.model.cosmos3 import submodules as cosmos3_submodules
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule

    model = _model(tmp_path)
    sub = Cosmos3ReasonerSubmodule(transformer=None, config=model.config, encoders=(model._vision_encoder(),))
    monkeypatch.setattr(cosmos3_submodules, "ingraph_decode_rope", lambda: True)
    captured = types.SimpleNamespace(captured=True)
    assert sub._rope_in_graph(captured, "decode") and sub._rope_in_graph(captured, "prefill_text")
    assert not sub._rope_in_graph(captured, "mixed")
    assert not sub._rope_in_graph(types.SimpleNamespace(captured=False), "decode")


def test_a_media_prompt_cuts_into_chunks_that_add_up_to_it(tmp_path) -> None:
    """Chunked prefill of an image prompt: each chunk takes its ids and
    positions, its part of the mask and the vision rows that part marks (cuts
    landing inside an image), and its part of the MRoPE advance, so the
    chunks splice in the whole row's media and advance the position counter
    by the whole prompt's."""
    import types

    from mstar.model.components.mrope import layout_position_ids
    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule
    from mstar.model.multimodal import MEDIA_LAYOUT, splice_media

    model = _model(tmp_path)
    sub = Cosmos3ReasonerSubmodule(transformer=None, config=model.config, encoders=(model._vision_encoder(),))
    sub.get_device = lambda: torch.device("cpu")
    sub.node_resources = {ROPE: types.SimpleNamespace(position=lambda rid, label: 10)}
    pad = model.config.reasoner.image_token_id
    # text 3, a 2x3 merged image, text 2, a 2x2 merged image, text 1
    layout = {"offset": [3, 11], "tokens": [6, 4], "output": ["vision_embeds"] * 2,
              "grid": [[1, 4, 6], [1, 4, 4]]}
    ids = torch.tensor([1, 2, 3] + [pad] * 6 + [4, 5] + [pad] * 4 + [6])
    pos, advance = layout_position_ids(layout, 2, 16, 10)
    rows = torch.arange(10.0).unsqueeze(1).expand(10, 8)
    info = types.SimpleNamespace(rid_handle="r", step_metadata={MEDIA_LAYOUT: layout})
    walk = Cosmos3Model.REASONER_PREFILL_VISION_WALK
    full = sub.prepare_inputs(walk, info, {"text_inputs": [ids], "vision_embeds": [rows]})
    assert sub.get_input_sequence_len(walk, info, {"text_inputs": [ids]}).seq_len == 16

    chunks = [sub.split_inputs(walk, info, full, a, b) for a, b in ((0, 5), (5, 13), (13, 16))]
    assert torch.equal(torch.cat([c.input_ids for c in chunks]), ids)
    assert torch.equal(torch.cat([c.custom_pos_ids for c in chunks], dim=-1), pos)
    assert [c.tensor_inputs[MEDIA_EMBEDS].shape[0] for c in chunks] == [2, 6, 2]
    # each chunk splices in its own rows: together, the whole prompt's
    eager = types.SimpleNamespace(resources={}, request_ids=["r"])
    spliced = [
        splice_media(out["input_ids"].float()[:, None].expand(-1, 8), out[MEDIA_MASK], out[MEDIA_EMBEDS])
        for out in (SampledTokenDecoder.preprocess(sub, walk, eager, [c]) for c in chunks)
    ]
    want = splice_media(ids.float()[:, None].expand(-1, 8), full.tensor_inputs[MEDIA_MASK], rows)
    assert torch.equal(torch.cat(spliced), want)
    # positions 10-14, then up to 19 inside the second image, then 20
    assert advance == 11 and [c.resource_step_info for c in chunks] == [5, 5, 1]
    # an image-free chunk carries no vision rows, and still stages an all-False
    # mask: alone on a captured step, its replay would splice in the last step's rows
    text = sub.split_inputs(walk, info, full, 0, 3)
    assert text.tensor_inputs == {}
    out = SampledTokenDecoder.preprocess(sub, walk, eager, [text])
    assert MEDIA_EMBEDS not in out and out[MEDIA_MASK].tolist() == [False] * 3


def test_the_reasoner_prompts_chunk_under_one_budget(tmp_path) -> None:
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule

    policies = _model(tmp_path).chunking_policy_of()
    budget = Cosmos3ReasonerSubmodule.MAX_BATCH_TOKENS
    for walk in (Cosmos3Model.REASONER_PREFILL_WALK, Cosmos3Model.REASONER_PREFILL_VISION_WALK):
        assert policies["reasoner", walk].chunkable and policies["reasoner", walk].max_batch_tokens == budget
    assert policies["reasoner", "mixed"].max_batch_tokens == budget
    assert not any(node == "dit" for node, _ in policies)


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

