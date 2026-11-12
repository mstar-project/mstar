"""The scaffold's Qwen prompt encoder on a Qwen-VL checkpoint's language model.

A tiny random Qwen3-VL and Qwen2.5-VL from transformers are saved and loaded back through
the encoder's own loader (the family's config reader, the key remap and the skip predicate,
which must leave the vision tower and the unused layers on disk); the tapped hidden state
must equal transformers' ``hidden_states[k]`` for a text-only prompt and for a prompt with
two images: the family's vision tower (Qwen3-VL's with deepstack, Qwen2.5-VL's windowed one),
whose tokens the encoder splices in and positions with the family's 3-D MRoPE layout.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("this transformers has no Qwen3-VL", allow_module_level=True)

from mstar.model.components.diffusion.attention import FrameSDPA  # noqa: E402
from mstar.model.components.diffusion.qwen.encoder import QwenEncoderConfig, QwenHiddenStateEncoder  # noqa: E402
from mstar.model.components.diffusion.qwen.vl_prompt import VisionTokens  # noqa: E402
from mstar.model.components.diffusion.qwen.weight_loading import (  # noqa: E402
    qkv_rules,
    remap_text_encoder_key,
    text_encoder_skip,
)
from mstar.model.components.diffusion.weight_loading import (  # noqa: E402
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.components.qwen3_vit import (  # noqa: E402
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)
from mstar.model.components.qwen_vision import (  # noqa: E402
    VISION_ATTN,
    Qwen25VisionConfig,
    QwenVisionConfig,
    QwenVisionModel,
    qwen25_tower_inputs,
)
from mstar.model.keys import VISION_FULL  # noqa: E402
from mstar.model.qwen_image.config import QwenImageConfig  # noqa: E402
from mstar.model.qwen_image.weight_loader import build_vision_tower  # noqa: E402

IMAGE, VISION_START, VISION_END = 250, 251, 252
TEXT = {"vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6}


class Family(NamedTuple):
    build: callable          # tmp path -> the saved transformers model
    config: callable         # (config.json dict, overrides) -> QwenEncoderConfig
    tower: callable          # (reference, checkpoint path, grids) -> (pixels -> VisionTokens)
    patch: int
    tap: int


def _save(model, path):
    model.eval().save_pretrained(path, safe_serialization=True)
    return model


def _qwen3_vl(path):
    torch.manual_seed(0)
    return _save(transformers.Qwen3VLForConditionalGeneration(transformers.Qwen3VLConfig(
        text_config={**TEXT, "head_dim": 16, "rope_theta": 5_000_000.0,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
                       "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1]},
        image_token_id=IMAGE, vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )), path)


def _qwen2_5_vl(path):
    torch.manual_seed(0)
    return _save(transformers.Qwen2_5_VLForConditionalGeneration(transformers.Qwen2_5_VLConfig(
        text_config={**TEXT, "rope_theta": 1_000_000.0, "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]}},
        # windows of 2x2 merge units, so the 4x8 patch grid is two windows and the 2x4 one is one
        vision_config={"depth": 4, "hidden_size": 32, "intermediate_size": 48, "num_heads": 2, "out_hidden_size": 64,
                       "patch_size": 4, "temporal_patch_size": 2, "spatial_merge_size": 2, "window_size": 16,
                       "fullatt_block_indexes": [1, 3], "in_channels": 3},
        image_token_id=IMAGE, vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )), path)


def _qwen3_tower(reference, path, grids):
    vision = QwenVisionModel(QwenVisionConfig.from_hf(path)).eval()
    vision.load_state_dict(reference.model.visual.state_dict())
    for block in vision.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA(vision_seq_lengths(grids))})

    def encode(pixels):
        indices, weights = vision_interpolation(grids, vision.config.num_grid_per_side, 2)
        merged, deepstack = vision(pixels, indices, weights, vision_position_ids(grids, 2))
        return VisionTokens(grids, 2, merged, deepstack)

    return encode


def _qwen25_tower(reference, path, grids):
    config = Qwen25VisionConfig.from_dict(json.loads((path / "config.json").read_text())["vision_config"])
    # through Qwen-Image's loader, from the saved checkpoint, which holds no rotary inv_freq: it is derived
    snapshot = path / "snapshot"
    snapshot.mkdir()
    (snapshot / "text_encoder").symlink_to(path)
    vision = build_vision_tower(QwenImageConfig(vision=config), snapshot, "cpu", dtype=torch.float32).eval()
    inputs = qwen25_tower_inputs(grids, config)
    windows, frames = FrameSDPA(inputs.window_lengths), FrameSDPA(inputs.frame_lengths)
    for block in vision.blocks:
        block.attn.bind_resources({VISION_ATTN: frames if block.attn.label == VISION_FULL else windows})

    def encode(pixels):
        merged = vision(pixels, inputs.rope_positions, inputs.window_index, inputs.restore_index)
        # the tower alone, against transformers' (its window order, labels and restore)
        theirs = reference.model.visual(pixels, grid_thw=torch.tensor(grids))
        theirs = getattr(theirs, "pooler_output", theirs)
        torch.testing.assert_close(merged, theirs, rtol=1e-5, atol=1e-5)
        return VisionTokens(grids, 2, merged)

    return encode


FAMILIES = {
    "qwen3_vl": Family(_qwen3_vl, QwenEncoderConfig.from_qwen3_vl, _qwen3_tower, patch=16, tap=3),
    # Qwen-Image's tap: the normed last hidden state
    "qwen2_5_vl": Family(_qwen2_5_vl, QwenEncoderConfig.from_qwen2_5_vl, _qwen25_tower, patch=4, tap=4),
}


def _encoder(path, family, fused_qkv):
    config = family.config(
        json.loads((path / "config.json").read_text()), hidden_state_layers=(family.tap,), fused_qkv=fused_qkv,
    )
    assert config.num_hidden_layers == 4 and config.image_token_id == IMAGE
    with torch.device("meta"):
        encoder = QwenHiddenStateEncoder(config)
    materialize(encoder, torch.float32, "cpu")
    skip = text_encoder_skip(config)
    load_native(
        encoder, iter_transformers_component(path, "cpu", skip=skip), remap_text_encoder_key,
        "tiny Qwen-VL", stacked_params=qkv_rules(config), skip=skip,
    )
    return encoder


def _reference_hidden(reference, tap, ids, **media):
    # transformers 5 positions the images in 3-D only when told which tokens are image tokens
    kwargs = dict(input_ids=ids[None], attention_mask=torch.ones_like(ids[None]), use_cache=False,
                  output_hidden_states=True, mm_token_type_ids=(ids == IMAGE).long()[None], **media)
    try:
        return reference.model(**kwargs).hidden_states[tap]
    except TypeError:  # a transformers without token types
        kwargs.pop("mm_token_type_ids", None)
        return reference.model(**kwargs).hidden_states[tap]


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("fused_qkv", [True, False])
def test_tapped_hidden_state_matches_transformers(tmp_path, name, fused_qkv):
    family = FAMILIES[name]
    reference = family.build(tmp_path)
    encoder = _encoder(tmp_path, family, fused_qkv)
    ids = torch.randint(0, 240, (11,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = encoder(ids[None], torch.ones_like(ids[None]))
        theirs = _reference_hidden(reference, family.tap, ids)
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", FAMILIES)
def test_image_prompt_matches_transformers(tmp_path, name):
    """Two images of different grids between text runs: the tower's merged tokens at the
    pads, deepstack added after the first layers where the family has it, 3-D MRoPE positions."""
    family = FAMILIES[name]
    reference = family.build(tmp_path)
    encoder = _encoder(tmp_path, family, fused_qkv=False)
    grids = [(1, 4, 8), (1, 2, 4)]
    tower = family.tower(reference, tmp_path, grids)

    g = torch.Generator().manual_seed(2)
    text = [torch.randint(0, 240, (n,), generator=g) for n in (3, 2, 4)]
    ids = torch.cat([
        text[0], torch.tensor([VISION_START] + [IMAGE] * 8 + [VISION_END]),
        text[1], torch.tensor([VISION_START] + [IMAGE] * 2 + [VISION_END]), text[2],
    ])
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), 3 * 2 * family.patch**2, generator=g)
    with torch.no_grad():
        theirs = _reference_hidden(reference, family.tap, ids, pixel_values=pixels, image_grid_thw=torch.tensor(grids))
        ours = encoder(ids[None], torch.ones_like(ids[None]), images=tower(pixels))
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_vision_rotary_frequencies_are_the_hosts_on_cuda():
    """transformers builds the tower's inv_freq on the host and moves it; CUDA's pow rounds 4 of Qwen2.5-VL's 20
    differently, which moved Qwen-Image-Edit's text embeddings 2% on an H200."""
    from types import SimpleNamespace

    from mstar.model.components.qwen3_vit import VisionRotaryEmbedding

    with torch.device("meta"):
        rope = VisionRotaryEmbedding(SimpleNamespace(head_dim=80, rope_theta=10000.0), keep_fp32=True)
    rope.to(torch.bfloat16).to_empty(device="cuda")
    want = 1.0 / (10000.0 ** (torch.arange(0, 40, 2, dtype=torch.float) / 40))
    assert rope.inv_freq.dtype == torch.float32 and torch.equal(rope.inv_freq.cpu(), want)


def test_rejects_other_rope_types():
    cfg = {"text_config": {
        "model_type": "qwen3_vl_text", "vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 1, "num_key_value_heads": 1,
        "rope_scaling": {"rope_type": "yarn", "factor": 2.0},
    }}
    with pytest.raises(NotImplementedError):
        QwenEncoderConfig.from_qwen3_vl(cfg, hidden_state_layers=(1,))
