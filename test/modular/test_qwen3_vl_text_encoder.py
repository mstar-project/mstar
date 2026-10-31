"""The scaffold's Qwen3 prompt encoder on a Qwen3-VL checkpoint's language model.

A tiny random Qwen3-VL from transformers is saved and loaded back through the
encoder's own loader (``from_qwen3_vl``, the ``model.language_model.`` remap and the
skip predicate, which must leave the vision tower and the unused layers on disk);
its tapped hidden state must equal transformers' ``hidden_states[k]`` for a
text-only prompt, which is where interleaved MRoPE reduces to plain RoPE, and for a
prompt with two images (the shared vision tower's tokens and deepstack features,
3-D MRoPE positions, ``vl_prompt.qwen3_vl_prompt``).
"""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("this transformers has no Qwen3-VL", allow_module_level=True)

from mstar.model.components.diffusion.qwen3.encoder import Qwen3EncoderConfig, qwen3_inv_freq  # noqa: E402
from mstar.model.components.diffusion.qwen3.vl_prompt import qwen3_vl_prompt  # noqa: E402
from mstar.model.components.diffusion.qwen3.weight_loading import (  # noqa: E402
    make_text_encoder,
    qkv_rules,
    qwen3_vl_text_skip,
    remap_qwen3_vl_text_key,
)
from mstar.model.components.diffusion.weight_loading import (  # noqa: E402
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.components.qwen_vision import (  # noqa: E402
    VISION_ATTN,
    FrameSDPA,
    QwenVisionConfig,
    QwenVisionModel,
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)

TAP = 3
IMAGE, VISION_START, VISION_END = 250, 251, 252


def _tiny_qwen3_vl(path):
    torch.manual_seed(0)
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=IMAGE, vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )
    model = transformers.Qwen3VLForConditionalGeneration(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def _reference_hidden(model, ids):
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
    try:
        out = model.model(mm_token_type_ids=torch.zeros_like(ids), **kwargs)
    except TypeError:  # a transformers without token types
        out = model.model(**kwargs)
    return out.hidden_states[TAP]


def _encoder(path, fused_qkv):
    config = Qwen3EncoderConfig.from_qwen3_vl(
        json.loads((path / "config.json").read_text()), hidden_state_layers=(TAP,), fused_qkv=fused_qkv,
    )
    assert (config.num_hidden_layers, config.rope_theta) == (4, 5_000_000.0)
    with torch.device("meta"):
        encoder = make_text_encoder(config)
    materialize(encoder, torch.float32, "cpu")
    skip = qwen3_vl_text_skip(config)
    load_native(
        encoder, iter_transformers_component(path, "cpu", skip=skip), remap_qwen3_vl_text_key,
        "tiny Qwen3-VL", stacked_params=qkv_rules(config), skip=skip,
    )
    return encoder


@pytest.mark.parametrize("fused_qkv", [True, False])
def test_tapped_hidden_state_matches_transformers(tmp_path, fused_qkv):
    reference = _tiny_qwen3_vl(tmp_path)
    encoder = _encoder(tmp_path, fused_qkv)
    ids = torch.randint(0, 240, (1, 11), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = encoder(ids, torch.ones_like(ids))
        theirs = _reference_hidden(reference, ids)
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


def test_image_prompt_matches_transformers(tmp_path):
    """Two images of different grids between text runs: the tower's merged tokens at the
    pads, deepstack added after layers 0 and 1, 3-D MRoPE positions."""
    reference = _tiny_qwen3_vl(tmp_path)
    encoder = _encoder(tmp_path, fused_qkv=False)
    vision = QwenVisionModel(QwenVisionConfig.from_hf(tmp_path)).eval()
    vision.load_state_dict(reference.model.visual.state_dict())
    grids = [(1, 4, 6), (1, 2, 4)]
    for block in vision.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA(vision_seq_lengths(grids))})

    g = torch.Generator().manual_seed(2)
    text = [torch.randint(0, 240, (n,), generator=g) for n in (3, 2, 4)]
    ids = torch.cat([
        text[0], torch.tensor([VISION_START] + [IMAGE] * 6 + [VISION_END]),
        text[1], torch.tensor([VISION_START] + [IMAGE] * 2 + [VISION_END]), text[2],
    ])
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), 3 * 2 * 16 * 16, generator=g)
    with torch.no_grad():
        theirs = reference.model(
            input_ids=ids[None], attention_mask=torch.ones_like(ids[None]),
            mm_token_type_ids=(ids == IMAGE).long()[None], pixel_values=pixels,
            image_grid_thw=torch.tensor(grids), use_cache=False, output_hidden_states=True,
        ).hidden_states[TAP]
        indices, weights = vision_interpolation(grids, vision.config.num_grid_per_side, 2, torch.device("cpu"))
        merged, deepstack = vision(pixels, indices, weights, vision_position_ids(grids, 2, torch.device("cpu")))
        text_cfg = reference.config.text_config
        prompt = qwen3_vl_prompt(
            ids, IMAGE, grids, 2, merged, deepstack, encoder.embed_tokens,
            qwen3_inv_freq(text_cfg.head_dim, 5_000_000.0), [2, 3, 3],
        )
        ours = encoder(ids[None], torch.ones_like(ids[None]), inputs_embeds=prompt.inputs_embeds,
                       rotary=prompt.rotary, deepstack=prompt.deepstack)
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


def test_rejects_other_rope_types():
    cfg = {"text_config": {
        "model_type": "qwen3_vl_text", "vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 1, "num_key_value_heads": 1,
        "rope_scaling": {"rope_type": "yarn", "factor": 2.0},
    }}
    with pytest.raises(NotImplementedError):
        Qwen3EncoderConfig.from_qwen3_vl(cfg, hidden_state_layers=(1,))
