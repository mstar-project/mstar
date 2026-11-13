"""Praxis-1 on CPU: what the checkpoint's config decides, and the graph it builds.

Every width of this model belongs to the checkpoint rather than to a knob, so
the config reader is the part that has to be right: a rig it misreads would
build a graph whose shapes disagree with the weights it then loads. These run
against the published configs of both checkpoints, with no weights and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import pytest
import yaml

from mstar.conductor.conductor import Conductor
from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.praxis1 import praxis1_model
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError
from mstar.model.praxis1.praxis1_model import DENOISE_LOOP_NAME, Praxis1Model
from mstar.model.registry import get_model_class, model_init_kwargs

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
        "moe": {"num_experts": 129, "num_routed_experts": 128, "top_k": 9, "expert_hidden": 1280},
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
    "text_encoder": {"model": "Qwen3.5-2B", "hidden_dim": 2048, "hidden_layer": -3},
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
    """A client's reply shape comes from the checkpoint, not from a default."""
    droid = Praxis1Config.from_dict(DROID_CONFIG)
    assert droid.views == ("exterior", "wrist")
    assert (droid.action_dim, droid.action_horizon) == (10, 24)
    # 24 rows carried by 6 tokens of 4 frames, which the head's width agrees with.
    assert droid.action_tokens == 6
    assert droid.action_patch_dim == droid.action_dim * droid.action_frames_per_token

    yam = Praxis1Config.from_dict(_yam_config())
    assert yam.views == ("top_left", "wrist_left", "wrist_right")
    assert (yam.action_dim, yam.action_horizon) == (20, 24)
    assert yam.action_patch_dim == 80


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
    ],
)
def test_a_checkpoint_this_build_does_not_serve_is_refused(mutate, message):
    """Refusing beats loading tensors that would mean something else here."""
    config = json.loads(json.dumps(DROID_CONFIG))
    mutate(config)
    with pytest.raises(Praxis1ConfigError, match=message):
        Praxis1Config.from_dict(config)


def test_robot_io_matches_the_checkpoint(tmp_path, monkeypatch):
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    state_kwarg, horizon, dim = model.robot_io()
    assert (horizon, dim) == (24, 10)
    assert state_kwarg in model.request_kwargs()
    # One image per camera view the rig has.
    assert model.max_input_files() == {"image": 2}


def test_the_text_walk_persists_what_the_chunk_walk_reads(tmp_path, monkeypatch):
    """The task text is constant for an episode, so no policy call re-encodes it."""
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
        ("runwayml/praxis-1-droid", "single_arm", (24, 10, 2)),
        ("runwayml/praxis-1-yam", "bimanual", (24, 20, 3)),
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
