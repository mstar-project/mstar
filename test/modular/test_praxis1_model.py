"""Praxis-1 on CPU: what the checkpoint's config decides, and the graph it builds.

Every width of this model belongs to the checkpoint rather than to a knob, so
the config reader is the part that has to be right: a rig it misreads would
build a graph whose shapes disagree with the weights it then loads. These run
against the published configs of both checkpoints, with no weights and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mstar.graph.base import GraphNode, Loop, Sequential
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError
from mstar.model.praxis1.praxis1_model import Praxis1Model

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


def _model(tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch) -> Praxis1Model:
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(Praxis1Model, "_ensure_repo", lambda self: tmp_path)
    return Praxis1Model(model_path_hf="runwayml/praxis-1-droid", skip_weight_loading=True)


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
        # The MoE regime. dispatch_chunk_size is the one that matters in
        # practice, since every published config sets it: see
        # test_praxis1_moe_equivalence for the measurement behind the refusal.
        (
            lambda c: c["backbone"]["moe"].update(dispatch_chunk_size=16384),
            "dispatch_chunk_size",
        ),
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


def test_an_unset_dispatch_chunk_size_is_served():
    """Negative control for the refusal: absent and explicit null both pass.

    The refusal has to key on the field being *set*, not on it being present,
    or a config that spells out the reference's own default would be rejected
    for agreeing with us.
    """
    for value in ({}, {"dispatch_chunk_size": None}):
        config = json.loads(json.dumps(DROID_CONFIG))
        config["backbone"]["moe"].update(value)
        assert Praxis1Config.from_dict(config).top_k_routed == 8


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
    # and leave the loop once it settles.
    assert {edge.name for edge in loop.outputs} == {"video_latents", "action_latents"}


def test_the_actions_are_what_reaches_the_client(tmp_path, monkeypatch):
    model = _model(tmp_path, DROID_CONFIG, monkeypatch)
    head = _node(model.get_graph_walk_graphs()[model.CHUNK_WALK], "action_head")
    emitted = [edge for edge in head.outputs if edge.output_modality]
    assert [(edge.name, edge.output_modality) for edge in emitted] == [("actions", "action")]


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
