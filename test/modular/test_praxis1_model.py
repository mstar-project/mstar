"""Praxis-1 on CPU: what the checkpoint's config decides, and the graph it builds.

Every width of this model belongs to the checkpoint rather than to a knob, so
the config reader is the part that has to be right: a rig it misreads would
build a graph whose shapes disagree with the weights it then loads. These run
against the published configs of both checkpoints, with no weights and no GPU.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import diffusers
import huggingface_hub
import numpy as np
import pytest
import torch
import yaml

from mstar.api_server.openpi import OpenpiPolicy, unpack
from mstar.api_server.positronic import PositronicPolicy
from mstar.api_server.rust_frontend import openpi_facts
from mstar.conductor.conductor import Conductor
from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import EpisodeView
from mstar.model.praxis1 import praxis1_model
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError
from mstar.model.praxis1.praxis1_model import DENOISE_LOOP_NAME, Praxis1Model
from mstar.model.registry import get_model_class, model_init_kwargs
from mstar.validation import validate_deployment

ROOT = Path(__file__).resolve().parents[2]

# The two published checkpoints, as their config.json describes them.
DROID_CONFIG = {
    "format_version": 2,
    "model_type": "praxis",
    "embodiment": "single_arm",
    "backbone": {
        "dim": 3072,
        "timestep_dim": 128,
        "num_registers": 4,
        "dense_ffn_hidden": 11520,
        "attention": {"num_heads": 24, "num_kv_heads": 8, "head_dim": 128, "flex_block_size": 128},
        # Both pairs, as the real configs carry them: 129 and 9 count the
        # shared expert, 128 and 8 are what the router sees.
        "moe": {
            "num_experts": 129,
            "num_routed_experts": 128,
            "shared_expert": True,
            "top_k": 9,
            "top_k_routed": 8,
            "expert_hidden": 1280,
            "routed_scaling_factor": 2.5,
            # Every published config sets it, and it selects the MoE combine.
            "dispatch_chunk_size": 16384,
        },
        "encoder": {
            "depth": 14,
            "tap_blocks": [0, 2, 4, 6, 7, 9, 11, 13],
            "input_projections": {
                "TextEncodingTensor": {"c_in": 2048},
                "VideoTensor": {"c_in": 192},
            },
        },
        "decoder": {"depth": 6},
        "layout": "chunk_causal",
    },
    "action_expert": {
        "dim": 1024,
        "num_layers": 16,
        "num_heads": 16,
        "head_dim": 64,
        "trunk_dim": 3072,
        "action_patch_dim": 40,
        "proprio_patch_dim": 10,
        "max_action_tokens": 64,
        "attention": "flex_dynamic",
    },
    "chunking": {
        "n_chunks": 4,
        "action_chunk": 24,
        "action_frames_per_token": 4,
        "latents_per_chunk": 6,
        "window_frames": 93,
        "grid_hz": 15.0,
    },
    "video": {
        "height": 224,
        "width": 832,
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "vae": "Wan2.2",
        "latent_channels": 48,
        "vae_stride": [4, 16, 16],
        "patch_size": [1, 2, 2],
        "tile_height": 224,
        "tile_width": 416,
    },
    "text_encoder": {"model": "Qwen3.5-2B", "hidden_dim": 2048, "hidden_layer": -3,
                     "blank_caption": " "},
    "action_space": {"dim": 10},
}


def _yam_config() -> dict:
    """The bimanual checkpoint: three views and twice the action width."""
    config = json.loads(json.dumps(DROID_CONFIG))
    config["embodiment"] = "bimanual"
    config["action_space"]["dim"] = 20
    config["action_expert"]["action_patch_dim"] = 80
    config["action_expert"]["proprio_patch_dim"] = 20
    config["video"]["views"] = ["top_left", "wrist_left", "wrist_right"]
    config["video"]["height"] = 256
    config["video"]["width"] = 1248
    config["video"]["tile_height"] = 256
    return config


def _model(
    tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch, **model_kwargs
) -> Praxis1Model:
    """The model as serving builds it, from the checkpoint's config.json.

    ``_ensure_repo`` is the only download the constructor makes, so pointing
    it at ``tmp_path`` keeps this offline, and lifting ``_SERVABLE`` stands in
    for the compute port. Nothing here calls ``get_submodule``, so no node is
    built.
    """
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(Praxis1Model, "_ensure_repo", lambda self: tmp_path)
    monkeypatch.setattr(praxis1_model, "_SERVABLE", True)
    return Praxis1Model(
        model_path_hf="runwayml/praxis-1-droid", skip_weight_loading=False, **model_kwargs
    )


@pytest.fixture
def no_download(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Fail any test that reaches the Hub; the calls it would have made."""
    calls: list[dict] = []

    def refuse(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"snapshot_download was called: {kwargs}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    return calls


def _loop_conductor(model: Praxis1Model) -> Conductor:
    """A conductor with only what ``_loop_caps`` reads: the model and its loops' ``max_iters``."""
    conductor = Conductor.__new__(Conductor)
    conductor.model = model
    conductor._loop_max_iters = {}
    for walk in model.get_graph_walk_graphs().values():
        for name, loop in walk.get_loops().items():
            conductor._loop_max_iters[name] = loop.max_iters
    return conductor


def test_the_rig_decides_the_wire():
    """The cold reply trims the anchor token while keeping checkpoint widths."""
    droid = Praxis1Config.from_dict(DROID_CONFIG)
    assert droid.views == ("exterior", "wrist")
    assert droid.pose_frame == "flange"
    assert (droid.action_dim, droid.action_horizon) == (10, 21)
    # Six tokens carry 24 subframes, but the depth-0 anchor contributes one row.
    assert droid.action_tokens == 6
    assert droid.action_patch_dim == droid.action_dim * droid.action_frames_per_token

    yam = Praxis1Config.from_dict(_yam_config())
    assert yam.views == ("top_left", "wrist_left", "wrist_right")
    assert yam.pose_frame == "rig"
    assert (yam.action_dim, yam.action_horizon) == (20, 21)
    assert yam.action_patch_dim == 80

    with pytest.raises(Praxis1ConfigError, match="pose_frame"):
        Praxis1Config.from_dict({**DROID_CONFIG, "pose_frame": "rig"})


@pytest.mark.parametrize(
    "name,config,width",
    [
        ("praxis1_droid", DROID_CONFIG, 10),
        ("praxis1_yam", _yam_config(), 20),
    ],
)
def test_cold_openpi_reply_has_21_relative_rows(tmp_path, monkeypatch, name, config, width):
    shipped = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    model = _model(
        tmp_path, config, monkeypatch,
        max_context_chunks=shipped["model_kwargs"]["max_context_chunks"],
        max_text_tokens=shipped["model_kwargs"].get("max_text_tokens"),
    )
    policy = OpenpiPolicy.from_config(shipped["openpi"], model)
    assert policy.action_horizon == 21
    cold_rows = np.zeros((21, width), dtype=np.float32)
    assert policy.actions(cold_rows.tobytes()).shape == cold_rows.shape
    with pytest.raises(RuntimeError, match=f"not 21x{width}"):
        policy.actions(np.zeros((24, width), dtype=np.float32).tobytes())


def test_droid_session_config_is_sized_and_gated(tmp_path, monkeypatch):
    config_path = ROOT / "configs" / "praxis1_droid.yaml"
    shipped = yaml.safe_load(config_path.read_text())
    defaults = shipped["model_kwargs"]
    model = _model(
        tmp_path, DROID_CONFIG, monkeypatch,
        max_context_chunks=defaults["max_context_chunks"],
        max_text_tokens=defaults["max_text_tokens"],
    )
    assert validate_deployment(model, shipped, str(config_path))
    policy = PositronicPolicy.from_config(shipped["positronic"], model)
    openpi_policy = OpenpiPolicy.from_config(shipped["openpi"], model)
    assert policy.observation_format == "praxis1"
    assert policy.control_mode == "droid_impedance"
    assert policy.meta["checkpoint"] == "runwayml/praxis-1-droid"
    assert policy.meta["max_context_chunks"] == defaults["max_context_chunks"] == 1
    assert openpi_policy.metadata["max_context_chunks"] == 0
    for key in ("num_steps", "t_shift", "cfg_scale", "cfg_video_scale"):
        assert policy.meta[key] == openpi_policy.metadata[key] == defaults[key]
    assert policy.meta["checkpoint"] == openpi_policy.metadata["checkpoint"] == defaults["model_path_hf"]
    assert policy.meta["rig"] == openpi_policy.metadata["rig"] == model.config.embodiment
    assert policy.meta["skip_video_decoder"] is openpi_policy.metadata["skip_video_decoder"] is True
    assert policy.meta["attention_backend"] == openpi_policy.metadata["attention_backend"] == "triton"
    assert openpi_policy.action_horizon == 21
    # Both Python and Rust send this same packed first WebSocket frame.
    assert unpack(base64.b64decode(openpi_facts(openpi_policy)["metadata"])) == openpi_policy.metadata

    cold = _model(tmp_path, DROID_CONFIG, monkeypatch, max_context_chunks=0)
    with pytest.raises(ValueError, match="no ready Positronic session adapter"):
        PositronicPolicy.from_config(shipped["positronic"], cold)
    yam = _model(tmp_path, _yam_config(), monkeypatch, max_context_chunks=1, max_text_tokens=256)
    with pytest.raises(ValueError, match="no ready Positronic session adapter"):
        PositronicPolicy.from_config(shipped["positronic"], yam)
    monkeypatch.setattr(praxis1_model, "_SERVABLE", False)
    with pytest.raises(ValueError, match="no ready Positronic session adapter"):
        PositronicPolicy.from_config(shipped["positronic"], model)


def test_droid_session_config_keeps_idless_openpi_cold(tmp_path, monkeypatch):
    shipped = yaml.safe_load((ROOT / "configs" / "praxis1_droid.yaml").read_text())
    model = _model(
        tmp_path, DROID_CONFIG, monkeypatch,
        max_context_chunks=shipped["model_kwargs"]["max_context_chunks"],
        max_text_tokens=shipped["model_kwargs"]["max_text_tokens"],
    )
    steps = model.walk_plan(
        "", ["image", "text"], ["action"], {"image_inputs": [object()]},
        {"action_format": "openpi", "text_tokens": 4, "blank_tokens": 1},
        episode=EpisodeView(None, 0, {}),
    )
    assert steps[-1].walk == model.CHUNK_WALK
    assert steps[-1].step_metadata["history_held"] == 0
    assert steps[-1].step_metadata["history_append"] == 0


@pytest.mark.parametrize("changed", [
    {"max_context_chunks": 2}, {"max_text_tokens": 128},
    {"num_steps": 1}, {"t_shift": 1.0},
    {"cfg_scale": 1.2}, {"cfg_video_scale": 1.0},
    {"cfg_scale": True},
])
def test_untested_droid_session_recipes_stay_closed(tmp_path, monkeypatch, changed):
    shipped = yaml.safe_load((ROOT / "configs" / "praxis1_droid.yaml").read_text())
    kwargs = {
        key: shipped["model_kwargs"][key]
        for key in ("max_context_chunks", "max_text_tokens", "num_steps", "t_shift",
                    "cfg_scale", "cfg_video_scale")
    }
    kwargs.update(changed)
    model = _model(tmp_path, DROID_CONFIG, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match="no ready Positronic session adapter"):
        PositronicPolicy.from_config(shipped["positronic"], model)


def test_the_canvas_takes_the_views_to_latents():
    """One tile per view on one canvas, then the VAE's stride."""
    droid = Praxis1Config.from_dict(DROID_CONFIG)
    assert droid.canvas_width == droid.tile_width * droid.num_cameras
    assert (droid.latent_height, droid.latent_width) == (14, 52)

    yam = Praxis1Config.from_dict(_yam_config())
    assert yam.canvas_width == yam.tile_width * yam.num_cameras
    assert (yam.latent_height, yam.latent_width) == (16, 78)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.update(embodiment="base"), "no action heads"),
        (lambda c: c.update(embodiment="so101"), "not served"),
        (lambda c: c["backbone"].update(layout="full_causal"), "layout"),
        (lambda c: c["video"].update(canvas="single"), "canvas"),
        (lambda c: c["video"].update(vae="Wan2.1"), "VAE"),
        (lambda c: c["video"].update(views=["exterior"]), "views"),
        (lambda c: c["action_space"].update(dim=7), "action rows"),
        (lambda c: c["action_expert"].update(action_patch_dim=24), "action_patch_dim"),
        # The MoE regime. dispatch_chunk_size is passed through, but only as
        # the reference's positive int: a zero, a negative, a bool, a float or
        # a string is refused rather than coerced to a chunk nobody wrote.
        *[
            (
                lambda c, v=v: c["backbone"]["moe"].update(dispatch_chunk_size=v),
                "dispatch_chunk_size must be a positive int",
            )
            for v in (0, -1, True, 16384.0, "16384")
        ],
        (lambda c: c["backbone"]["moe"].update(shared_expert=False), "shared_expert"),
        (lambda c: c["backbone"]["moe"].update(router_activation="softmax"), "router_activation"),
        (lambda c: c["backbone"]["moe"].update(router_dtype="bfloat16"), "router_dtype"),
        (
            lambda c: c["backbone"]["moe"].update(routed_weight_layout="hidden_first"),
            "routed_weight_layout",
        ),
        # The two pairs of expert counts have to differ by exactly the one
        # shared expert. Anything else means a field is misunderstood.
        (lambda c: c["backbone"]["moe"].update(top_k_routed=9), "may differ only by"),
        (lambda c: c["backbone"]["moe"].update(num_routed_experts=100), "may differ only by"),
        (lambda c: c["backbone"]["moe"].pop("top_k_routed"), "must state"),
        (lambda c: c["backbone"]["moe"].pop("num_routed_experts"), "must state"),
    ],
)
def test_a_checkpoint_this_build_does_not_serve_is_refused(mutate, message):
    """Refusing beats loading tensors that would mean something else here."""
    config = json.loads(json.dumps(DROID_CONFIG))
    mutate(config)
    with pytest.raises(Praxis1ConfigError, match=message):
        Praxis1Config.from_dict(config)


def test_the_router_sees_the_routed_counts_and_not_the_totals():
    """128 of 8, never 129 of 9.

    ``backbone.moe`` states both pairs, and the shared-expert-inclusive one is
    what it calls ``num_experts`` and ``top_k``. Handing those to the MoE does
    not raise: a 129-way router picking nine renormalizes nine gates to one and
    returns the right shape, so the only symptom is a policy that is quietly
    wrong. This asserts the config cannot be read that way by accident.
    """
    droid = Praxis1Config.from_dict(DROID_CONFIG)
    assert (droid.num_routed_experts, droid.top_k_routed) == (128, 8)
    assert (droid.num_experts_with_shared, droid.top_k_with_shared) == (129, 9)
    # The totals exceed the routed counts by exactly the always-on expert.
    assert droid.num_experts_with_shared == droid.num_routed_experts + 1
    assert droid.top_k_with_shared == droid.top_k_routed + 1

    # The short names are the ambiguous ones, so they do not exist. A caller
    # reaching for one gets an AttributeError rather than a number off by one.
    assert not hasattr(droid, "num_experts")
    assert not hasattr(droid, "top_k")

    # And the single place the layer's own naming is produced.
    kwargs = droid.moe_kwargs()
    assert (kwargs["num_experts"], kwargs["top_k"]) == (128, 8)
    assert (kwargs["dim"], kwargs["hidden"]) == (3072, 1280)
    assert kwargs["routed_scaling_factor"] == 2.5
    assert kwargs["dispatch_chunk_size"] == 16384


def test_the_totals_are_redundant_and_may_be_omitted():
    """The routed pair is sufficient; the totals are derived when absent.

    Negative control for the consistency check above: dropping the totals must
    not change the routed numbers, which is what shows the check validates the
    two readings against each other rather than picking one of them.
    """
    config = json.loads(json.dumps(DROID_CONFIG))
    for redundant in ("num_experts", "top_k"):
        config["backbone"]["moe"].pop(redundant)
    terse = Praxis1Config.from_dict(config)
    assert terse.moe_kwargs() == Praxis1Config.from_dict(DROID_CONFIG).moe_kwargs()
    assert (terse.num_experts_with_shared, terse.top_k_with_shared) == (129, 9)


@pytest.mark.parametrize(
    "written, read",
    [({}, None), ({"dispatch_chunk_size": None}, None),
     ({"dispatch_chunk_size": 16384}, 16384), ({"dispatch_chunk_size": 7}, 7)],
)
def test_dispatch_chunk_size_is_passed_through_as_written(written, read):
    """Set, absent and null all load, and the layer gets exactly what was read.

    The field selects the MoE combine on ``is not None``, so the reader must
    neither default a missing one to the published 16384 nor drop a set one:
    either serves the weights on the combine the config did not ask for, and
    nothing downstream notices. Absent and null are the reference's default
    and both mean the unchunked combine.
    """
    config = json.loads(json.dumps(DROID_CONFIG))
    config["backbone"]["moe"].pop("dispatch_chunk_size")
    config["backbone"]["moe"].update(written)
    loaded = Praxis1Config.from_dict(config)
    assert loaded.dispatch_chunk_size == read
    assert loaded.moe_kwargs()["dispatch_chunk_size"] == read


@pytest.mark.parametrize("rig", ["single_arm", "bimanual"])
def test_the_dummy_shape_splits_the_experts_the_same_way(rig):
    """``for_rig`` builds the graph the startup checks run, so its expert
    counts have to be the published ones or those checks size a router that
    serving would never build."""
    config = Praxis1Config.for_rig(rig)
    assert config.num_experts_with_shared == config.num_routed_experts + 1
    assert config.top_k_with_shared == config.top_k_routed + 1
    kwargs = config.moe_kwargs()
    assert (kwargs["num_experts"], kwargs["top_k"]) == (128, 8)
    # And the published chunk, so the dummy graph runs the served combine.
    assert kwargs["dispatch_chunk_size"] == 16384


def test_robot_io_uses_cold_horizon_and_checkpoint_width(tmp_path, monkeypatch):
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    state_kwarg, horizon, dim = model.robot_io()
    assert (horizon, dim) == (21, 10)
    assert state_kwarg in model.request_kwargs()
    # One image per camera view the rig has.
    assert model.max_input_files() == {"image": 2}


def test_the_text_walk_persists_what_the_chunk_walk_reads(tmp_path, monkeypatch):
    """Persisted text is available to later walks of the same request."""
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {model.ENCODE_TEXT_WALK, model.CHUNK_WALK}

    encode_text = walks[model.ENCODE_TEXT_WALK]
    assert isinstance(encode_text, GraphNode)
    persisted = {edge.name for edge in encode_text.outputs if edge.persist}
    assert persisted == {"text_embeds", "text_embeds_blank"}

    backbone = _node(walks[model.CHUNK_WALK], "backbone")
    assert persisted <= set(backbone.input_names)


def test_the_denoise_loop_carries_both_latents(tmp_path, monkeypatch):
    """Each Euler step hands the expert its taps and takes back both latents."""
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    chunk = model.get_graph_walk_graphs()[model.CHUNK_WALK]
    assert isinstance(chunk, Sequential)

    loop = next(s for s in chunk.sections if isinstance(s, Loop))
    backbone, expert = loop.section.sections
    assert (backbone.name, expert.name) == ("backbone", "action_expert")

    # The taps never leave the iteration that made them.
    assert "taps" in {edge.name for edge in backbone.outputs}
    assert "taps" in expert.input_names
    # Both latents loop back for the next step.
    assert {edge.name for edge in expert.outputs} == {"video_latents", "action_latents"}
    assert all(edge.next_node == "backbone" for edge in expert.outputs)
    # Only the actions leave once it settles: the predicted video has no reader.
    assert {edge.name for edge in loop.outputs} == {"action_latents"}


def test_the_actions_are_all_that_leaves_the_chunk_walk(tmp_path, monkeypatch):
    """The next call is conditioned on real frames, never on the predicted
    video, so a chunk walk persists nothing and hands the client only rows."""
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    chunk = model.get_graph_walk_graphs()[model.CHUNK_WALK]
    edges = [edge for node in chunk.get_nodes().values() for edge in node.outputs]
    edges += [edge for loop in chunk.get_loops().values() for edge in loop.outputs]

    to_client = [edge for edge in edges if edge.next_node == EMIT_TO_CLIENT]
    assert [(edge.name, edge.output_modality) for edge in to_client] == [("actions", "action")]
    assert [edge.name for edge in edges if edge.persist] == []
    # No edge carries the predicted video past the loop.
    head = _node(chunk, "action_head")
    assert "video_latents" not in head.input_names


def test_the_served_model_reads_the_checkpoint_config(tmp_path, monkeypatch, no_download):
    """Values the published shape does not have reach the model, so the
    tests above ran on config.json and not on the rig defaults."""
    config = json.loads(json.dumps(DROID_CONFIG))
    config["max_denoise_steps"] = 4
    config["action_expert"]["max_action_tokens"] = 32
    served = _model(tmp_path, config, monkeypatch)
    dummy = Praxis1Model(model_path_hf="runwayml/praxis-1-droid", skip_weight_loading=True)

    assert (served.config.max_action_tokens, dummy.config.max_action_tokens) == (32, 64)
    assert (served.max_denoise_steps, dummy.max_denoise_steps) == (4, 8)
    loops = served.get_graph_walk_graphs()[served.CHUNK_WALK].get_loops()
    assert loops[DENOISE_LOOP_NAME].max_iters == 4


@pytest.mark.parametrize(
    "repo, rig, wire",
    [
        ("runwayml/praxis-1-droid", "single_arm", (21, 10, 2)),
        ("runwayml/praxis-1-yam", "bimanual", (21, 20, 3)),
    ],
)
def test_dummy_mode_takes_the_rig_from_the_name_and_reads_nothing(repo, rig, wire, no_download):
    """The startup checks build the model with no checkpoint, so dummy mode
    must never reach the Hub, and must still build the rig's wire."""
    model = Praxis1Model(model_path_hf=repo, skip_weight_loading=True)
    _, horizon, dim = model.robot_io()
    assert model.config.embodiment == rig
    assert (horizon, dim, model.max_input_files()["image"]) == wire
    assert set(model.get_graph_walk_graphs()) == {model.ENCODE_TEXT_WALK, model.CHUNK_WALK}
    assert no_download == []


def test_local_checkpoint_directory_needs_no_hub_fetch(tmp_path, monkeypatch, no_download):
    root = tmp_path / "praxis-1-droid"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(DROID_CONFIG))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"action_expert.marker": "expert.safetensors"},
    }))
    (root / "expert.safetensors").write_bytes(b"")
    monkeypatch.setattr(praxis1_model, "_SERVABLE", True)
    model = Praxis1Model(str(root), skip_weight_loading=False)
    assert model._ensure_repo() == root
    assert model._ensure_component_weights("action_expert.") == root
    assert no_download == []

    (root / "expert.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Praxis weights lack expert.safetensors"):
        model._ensure_component_weights("action_expert.")


def test_missing_local_checkpoint_does_not_turn_into_a_hub_id(tmp_path, monkeypatch, no_download):
    monkeypatch.setattr(praxis1_model, "_SERVABLE", True)
    with pytest.raises(FileNotFoundError, match="Praxis checkpoint directory does not exist"):
        Praxis1Model(str(tmp_path / "praxis-1-droid"), skip_weight_loading=False)
    assert no_download == []


def test_dummy_prompt_has_the_declared_text_padding(monkeypatch, no_download):
    model = Praxis1Model(model_path_hf="runwayml/praxis-1-droid", skip_weight_loading=True)
    model.config = replace(model.config, normalization={
        "proprio_mean": [0.0] * 10, "proprio_std": [1.0] * 10,
    })
    pads = []

    class Tokenizer:
        def __call__(self, *, pad_to_multiple_of, **kwargs):
            pads.append(pad_to_multiple_of)
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

    tokenizer = Tokenizer()

    def fake_tokenizer():
        return tokenizer

    monkeypatch.setattr(model, "_tokenizer", fake_tokenizer)
    model.process_prompt("pick up cup", robot_state=[0, 0, 0, 1, 0, 0, 0, 0])
    assert pads == [32, 32]
    assert no_download == []


@pytest.mark.parametrize("caption", ["", "  "])
def test_served_prompt_strips_text_and_normalizes_blank(tmp_path, monkeypatch, no_download, caption):
    config = json.loads(json.dumps(DROID_CONFIG))
    config["text_encoder"]["blank_caption"] = caption
    model = _model(tmp_path, config, monkeypatch)
    model.config = replace(model.config, normalization={
        "proprio_mean": [0.0] * 10, "proprio_std": [1.0] * 10,
    })
    seen = []

    class Tokenizer:
        def __call__(self, *, text, **kwargs):
            seen.append(text[0])
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

    tokenizer = Tokenizer()

    def fake_tokenizer():
        return tokenizer

    monkeypatch.setattr(model, "_tokenizer", fake_tokenizer)
    model.process_prompt("  pick up cup\n", robot_state=[0, 0, 0, 1, 0, 0, 0, 0])
    assert seen == ["pick up cup", " "]
    assert no_download == []


def test_served_checkpoint_requires_blank_caption():
    config = json.loads(json.dumps(DROID_CONFIG))
    del config["text_encoder"]["blank_caption"]
    with pytest.raises(Praxis1ConfigError, match="blank_caption"):
        Praxis1Config.from_dict(config)


def test_the_config_read_fetches_config_json_only(tmp_path, monkeypatch):
    """Every process builds the graph from config.json; none of them may pull
    the weights, which only get_submodule on a worker reads."""
    calls: list[dict] = []

    def fetch(**kwargs):
        calls.append(kwargs)
        (tmp_path / "config.json").write_text(json.dumps(DROID_CONFIG))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    monkeypatch.setattr(praxis1_model, "_SERVABLE", True)
    Praxis1Model(model_path_hf="runwayml/praxis-1-droid", skip_weight_loading=False)
    assert [call["allow_patterns"] for call in calls] == [["config.json"]]


@pytest.mark.parametrize("name", ["praxis1_droid", "praxis1_yam"])
def test_shipped_vae_config_selects_diffusers_subfolder(tmp_path, monkeypatch, name):
    """The original Wan repo lacks vae/; the Diffusers repo supplies it."""
    shipped = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    vae_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert shipped["model_kwargs"]["vae_weights_hf"] == vae_repo
    config = DROID_CONFIG if name == "praxis1_droid" else _yam_config()
    model_kwargs = {k: v for k, v in shipped["model_kwargs"].items() if k != "model_path_hf"}
    model = _model(tmp_path, config, monkeypatch, **model_kwargs)
    calls = []

    class FakeVae(torch.nn.Module):
        config = type("Config", (), {"z_dim": 48})()

    def from_pretrained(repo, **kwargs):
        calls.append((repo, kwargs))
        return FakeVae()

    monkeypatch.setattr(diffusers.AutoencoderKLWan, "from_pretrained", from_pretrained)
    model._create_submodule("vae_encoder", "cpu")
    assert len(calls) == 1
    assert calls[0][0] == vae_repo
    assert calls[0][1]["subfolder"] == "vae"
    assert calls[0][1]["torch_dtype"] == torch.bfloat16


@pytest.mark.parametrize("name", ["praxis1_droid", "praxis1_yam"])
def test_serving_is_refused_before_anything_is_downloaded(name, no_download):
    """``mstar serve`` builds the model from the registry and the shipped
    config, as here; with no node computation yet that has to stop at once,
    not after a download and every process's startup."""
    shipped = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    with pytest.raises(NotImplementedError, match="not servable yet"):
        get_model_class(name)(**{**model_init_kwargs(name), **shipped["model_kwargs"]})
    assert no_download == []


def test_the_loop_runs_the_recipe_not_its_ceiling(tmp_path, monkeypatch):
    """A 2-step recipe ends the loop on step 2. Undeclared, the conductor
    would give the loop its ceiling of 8, and the expert's async scheduling
    a step to speculate past the last."""
    model = _model(tmp_path, DROID_CONFIG, monkeypatch, num_steps=2)
    conductor = _loop_conductor(model)
    assert conductor._loop_max_iters == {DENOISE_LOOP_NAME: 8}
    assert conductor._loop_caps({}) == {DENOISE_LOOP_NAME: 2}
    # A request may ask for fewer steps.
    assert conductor._loop_caps({"num_steps": 1}) == {DENOISE_LOOP_NAME: 1}
    assert "num_steps" in model.request_kwargs()


@pytest.mark.parametrize(
    "num_steps, message",
    [(3, "never more"), (0, "integer >= 1"), (True, "integer"), ("2", "integer")],
)
def test_a_request_cannot_raise_the_step_count(tmp_path, monkeypatch, num_steps, message):
    model = _model(tmp_path, DROID_CONFIG, monkeypatch, num_steps=2)
    with pytest.raises(ValueError, match=message):
        model.get_loop_caps(num_steps=num_steps)


def test_a_recipe_past_the_loop_ceiling_is_refused_at_construction(tmp_path, monkeypatch):
    """Otherwise every request would be refused at the conductor."""
    with pytest.raises(Praxis1ConfigError, match="num_steps"):
        _model(tmp_path, DROID_CONFIG, monkeypatch, num_steps=9)


def test_nothing_here_decodes_tokens(tmp_path, monkeypatch):
    """No KV cache and no sampler: the actions come from flow matching, and the
    backbone re-reads its window every step."""
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    assert model.get_node_resources() == []


def _node(section, name: str) -> GraphNode:
    """The node called ``name``, anywhere in ``section``."""
    if isinstance(section, GraphNode):
        return section if section.name == name else None
    children = section.sections if isinstance(section, Sequential) else [section.section]
    for child in children:
        found = _node(child, name)
        if found is not None:
            return found
    return None
