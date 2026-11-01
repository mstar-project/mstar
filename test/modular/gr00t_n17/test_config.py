import json

import pytest

from mstar.model.gr00t_n17.config import Gr00tN17Config

# Trimmed copies of the two checkpoint configs: only the keys the port reads.
GR00T = {
    "select_layer": 16,
    "backbone_embedding_dim": 2048,
    "action_horizon": 40,
    "max_action_dim": 132,
    "max_state_dim": 132,
    "max_num_embodiments": 32,
    "hidden_size": 1024,
    "num_inference_timesteps": 4,
    "use_alternate_vl_dit": True,
    "use_vlln": True,
    "use_vl_self_attention": True,
    "add_pos_embed": True,
    "apply_sincos_state_encoding": False,
    "use_future_tokens": False,
    "reproject_vision": False,
    "diffusion_model_cfg": {
        "attention_head_dim": 48, "interleave_self_attention": True, "norm_type": "ada_norm",
        "num_attention_heads": 32, "num_layers": 32, "output_dim": 1024, "positional_embeddings": None,
    },
    "vl_self_attention_cfg": {
        "attention_head_dim": 64, "num_attention_heads": 32, "num_layers": 4, "positional_embeddings": None,
    },
}
BACKBONE = {
    "image_token_id": 151655,
    "vision_start_token_id": 151652,
    "text_config": {
        "head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144, "num_attention_heads": 16,
        "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-6, "rope_theta": 5000000,
        "vocab_size": 151936,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
    },
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_size": 1024, "in_channels": 3,
        "intermediate_size": 4096, "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 2048,
        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
    },
}


def _load(tmp_path, gr00t=GR00T, backbone=BACKBONE):
    (tmp_path / "g").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "g" / "config.json").write_text(json.dumps(gr00t))
    (tmp_path / "b" / "config.json").write_text(json.dumps(backbone))
    return Gr00tN17Config.from_hf(tmp_path / "g", tmp_path / "b")


def test_reads_the_checkpoint_over_class_defaults(tmp_path):
    c = _load(tmp_path)
    # the LLM is cut to select_layer, not the backbone's 28 layers
    assert c.text.num_hidden_layers == 16
    assert c.text.mrope_section == (24, 20, 20)
    assert (c.dit.num_layers, c.dit.inner_dim, c.dit.output_dim) == (32, 1536, 1024)
    assert (c.bridge.num_layers, c.bridge.inner_dim) == (4, 2048)
    # GR00T owns DeepStack; the reused qwen3_5 tower must see it off
    assert c.deepstack_visual_indexes == (5, 11, 17)
    assert c.vision.deepstack_visual_indexes == ()
    # absent from the checkpoint: the reference's class defaults
    assert (c.input_embedding_dim, c.state_history_length, c.attend_text_every_n_blocks) == (1536, 1, 2)


@pytest.mark.parametrize("key, value", [
    ("use_alternate_vl_dit", False),
    ("apply_sincos_state_encoding", True),
    ("diffusion_model_cfg", {**GR00T["diffusion_model_cfg"], "positional_embeddings": "sinusoidal"}),
    ("select_layer", 29),
    ("vl_self_attention_cfg", {**GR00T["vl_self_attention_cfg"], "num_layers": 0}),
])
def test_rejects_a_variant_it_does_not_implement(tmp_path, key, value):
    with pytest.raises(ValueError, match="unsupported"):
        _load(tmp_path, gr00t={**GR00T, key: value})
