"""Platform use case: prove execution matches Hugging Face Qwen3-VL-MoE."""

from __future__ import annotations

import torch

from mstar.model.qwenvl.components import QwenVLForCausalLM
from mstar.model.qwenvl.submodules import qwen_vl_position_ids
from mstar.model.qwenvl.weight_loader import load_qwen_vl_text_weights, load_qwen_vl_vision_weights

from ._helpers import CausalCache, patch_cpu_rms_norm, qwen_transformers_or_skip, tiny_config


def _text_values():
    return {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 6,
        "norm_topk_prob": True,
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 2, 1], "mrope_interleaved": True},
    }


def _vision_values():
    return {
        "depth": 3,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 16,
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 1,
        "spatial_merge_size": 1,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0, 1, 2],
    }


def test_hf_vision_tower_constructs_with_deepstack_mergers():
    reference = qwen_transformers_or_skip()
    model = reference.vision_model(reference.vision_config(**_vision_values()))
    state = model.state_dict()
    assert "patch_embed.proj.weight" in state
    assert "deepstack_merger_list.2.linear_fc2.weight" in state


def test_text_chat_execution_matches_reference_moe_model(monkeypatch):
    reference = qwen_transformers_or_skip()
    patch_cpu_rms_norm(monkeypatch)
    config = tiny_config()
    oracle = reference.text_model(reference.text_config(**_text_values())).eval()
    model = QwenVLForCausalLM(config).eval()
    assert load_qwen_vl_text_weights(
        model, ((f"model.language_model.{name}", value) for name, value in oracle.state_dict().items())
    ) == set(dict(model.named_parameters())) - {"lm_head.weight"}
    ids, positions = torch.tensor([1, 2, 3]), torch.arange(3).expand(3, -1)
    reference_positions = torch.cat((torch.arange(ids.numel()).unsqueeze(0), positions), dim=0)
    expected = oracle(
        input_ids=ids.unsqueeze(0), position_ids=reference_positions.unsqueeze(1), use_cache=False
    ).last_hidden_state[0]
    actual = model(model.model.embed_tokens(ids), CausalCache(), positions, position_advance=3)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_image_chat_execution_matches_reference_deepstack_model(monkeypatch):
    reference = qwen_transformers_or_skip()
    patch_cpu_rms_norm(monkeypatch)
    text, vision = _text_values(), _vision_values()
    hf_config = reference.config(text_config=text, vision_config=vision, image_token_id=30)
    oracle = reference.model(hf_config).eval()
    config = reference.config(text_config=text, vision_config=vision, image_token_id=30)
    mstar_text, mstar_vision = (
        QwenVLForCausalLM(config).eval(),
        reference.vision_model(oracle.config.vision_config).eval(),
    )
    assert load_qwen_vl_text_weights(
        mstar_text,
        ((f"model.language_model.{name}", value) for name, value in oracle.language_model.state_dict().items()),
    ) == set(dict(mstar_text.named_parameters())) - {"lm_head.weight"}
    assert load_qwen_vl_vision_weights(
        mstar_vision, ((f"model.visual.{name}", value) for name, value in oracle.visual.state_dict().items())
    ) == set(dict(mstar_vision.named_parameters()))
    ids = torch.tensor([1, 30, 30, 30, 30, 2])
    grid, pixels = torch.tensor([[1, 2, 2]]), torch.randn(4, 12)
    positions = qwen_vl_position_ids(ids, grid, config)
    # HF's four-axis input keeps monotonic token positions for the causal
    # mask separate from the three interleaved MRoPE axes. M*'s cache plan
    # already owns causal token order, so only the latter three cross the
    # model boundary.
    reference_positions = torch.cat((torch.arange(ids.numel()).unsqueeze(0), positions), dim=0)
    expected = oracle(
        input_ids=ids.unsqueeze(0),
        pixel_values=pixels,
        image_grid_thw=grid,
        position_ids=reference_positions.unsqueeze(1),
        use_cache=False,
    ).last_hidden_state[0]
    vision_embeds, deepstack_visual_embeds = mstar_vision(pixels, grid_thw=grid)
    embeddings = mstar_text.model.embed_tokens(ids).clone()
    embeddings[ids == 30] = vision_embeds
    reference_hidden = oracle.language_model(
        inputs_embeds=embeddings.unsqueeze(0),
        position_ids=reference_positions.unsqueeze(1),
        visual_pos_masks=(ids == 30).unsqueeze(0),
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=False,
    ).last_hidden_state[0]
    torch.testing.assert_close(reference_hidden, expected, atol=5e-6, rtol=5e-6)
    actual = mstar_text(
        embeddings,
        CausalCache(),
        positions,
        position_advance=4,
        visual_token_mask=ids == 30,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)


def test_image_positions_match_the_official_qwen3_multimodal_contract():
    reference = qwen_transformers_or_skip()
    text, vision = _text_values(), _vision_values()
    config = reference.config(text_config=text, vision_config=vision, image_token_id=30)
    oracle = reference.model(config).eval()
    vision_start = config.vision_start_token_id
    ids = torch.tensor([1, vision_start, 30, 30, 30, 30, 2, vision_start, 30, 30, 30, 30, 3])
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]])

    expected, _ = oracle.get_rope_index(
        ids.unsqueeze(0),
        image_grid_thw=grids,
    )
    actual = qwen_vl_position_ids(ids, grids, config)

    torch.testing.assert_close(actual, expected[:, 0])
