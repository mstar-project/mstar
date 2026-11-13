"""The ``vision_encoder`` node on CPU with a tiny random tower (56 px crops, so
4 x 4 patches and 2 x 2 cells a crop): the packed served path against the
reference's computation written out as ``Phi4MMImageEmbedding`` does it
(per-crop masks, FlashAttention-2's padded patches, ``AvgPool2d``, the HD
transform's reshapes), a step packing two requests against each alone, and
the node's declarations. The attention resource is stood in for by SDPA
within each planned segment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mstar.model.keys import VISION_ATTN
from mstar.model.multimodal import MEDIA_LAYOUT
from mstar.model.phi4mm.components.vision import (
    CHECKPOINT_PREFIX,
    Phi4MMVisionConfig,
    Phi4MMVisionModel,
    unused_weight,
)
from mstar.model.phi4mm.vision import (
    GLOBAL_PIXEL_VALUES,
    PIXEL_VALUES,
    VISION_OUTPUT,
    Phi4MMVisionEncoderSubmodule,
    image_intake,
    vision_resources,
)
from mstar.model.submodule_base import ModelInputsFromEngine

TINY = Phi4MMVisionConfig(
    hidden_size=32, intermediate_size=48, num_attention_heads=2, num_hidden_layers=3,
    crop_size=56, text_hidden_size=24, max_crops=6,
)
# (width, height): 2 x 2 crops with 5 of 8 patch rows valid (an odd edge); 3 crops
# across with 9 of 12 patch columns valid; 3 crops across, unpadded
SIZES = [(100, 60), (125, 56), (150, 40)]


class SegmentSDPA:
    """The ragged attention resource as the blocks see it: attention within
    each segment the step declared."""

    max_segments_per_step = 64

    def __init__(self):
        self.lengths: list[int] = []

    def plan(self, step):
        self.lengths = [s.span for s in step.segments]

    def run(self, q, k, v, label=None):
        out, start = [], 0
        for n in self.lengths:
            sl = slice(start, start + n)
            out.append(F.scaled_dot_product_attention(
                q[sl].transpose(0, 1), k[sl].transpose(0, 1), v[sl].transpose(0, 1),
            ).transpose(0, 1))
            start += n
        assert start == q.shape[0]
        return torch.cat(out)


def _model() -> Phi4MMVisionModel:
    torch.manual_seed(0)
    model = Phi4MMVisionModel(TINY).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model


def _node(model) -> tuple[Phi4MMVisionEncoderSubmodule, SegmentSDPA]:
    node = Phi4MMVisionEncoderSubmodule(model, TINY)
    attn = SegmentSDPA()
    node.bind_node_resources({VISION_ATTN: attn})
    return node, attn


def _images(sizes):
    rng = np.random.RandomState(1)
    return [
        image_intake(torch.from_numpy(rng.randint(0, 256, (3, h, w), dtype=np.uint8)), {}, TINY)
        for w, h in sizes
    ]


def _run(node, attn, requests):
    """One step over ``requests`` (lists of `PreparedMedia`): each request's rows."""
    rids, infos, inputs = [], {}, []
    for i, prepared in enumerate(requests):
        layout = {
            "offset": [0] * len(prepared), "tokens": [p.tokens[0] for p in prepared],
            "output": [VISION_OUTPUT] * len(prepared), "grid": [list(p.grids[0]) for p in prepared],
        }
        rid = f"r{i}"
        info = SimpleNamespace(step_metadata={MEDIA_LAYOUT: layout})
        rids.append(rid)
        infos[rid] = info
        inputs.append(node.prepare_inputs("prefill_vision", info, {
            PIXEL_VALUES: [p.inputs[PIXEL_VALUES] for p in prepared],
            GLOBAL_PIXEL_VALUES: [p.inputs[GLOBAL_PIXEL_VALUES] for p in prepared],
        }))
    step = node.declare_step("prefill_vision", rids, inputs)
    attn.plan(step.get(VISION_ATTN))
    engine = ModelInputsFromEngine(request_ids=rids, per_request_info=infos)
    with torch.no_grad():
        out = node.forward_batched("prefill_vision", engine, **node.preprocess("prefill_vision", engine, inputs))
    return [out[rid][VISION_OUTPUT][0] for rid in rids]


def _reference(model: Phi4MMVisionModel, prepared) -> torch.Tensor:
    """One image the way ``Phi4MMImageEmbedding`` computes it under FlashAttention-2."""
    cfg = model.config
    side, half, c = cfg.patches_per_side, cfg.patches_per_side // 2, cfg.hidden_size
    down, across, rows, cols = prepared.grids[0]
    crops = prepared.inputs[PIXEL_VALUES].float() / 255
    images = torch.cat([prepared.inputs[GLOBAL_PIXEL_VALUES], (crops - 0.5) / 0.5])
    canvas = torch.zeros(side * down, side * across)
    canvas[:rows, :cols] = 1
    masks = torch.cat([
        torch.ones(1, side, side),
        canvas.view(down, side, across, side).permute(0, 2, 1, 3).reshape(-1, side, side),
    ]).bool()
    emb = model.img_processor.embeddings
    x = emb.patch_embedding(images).flatten(2).transpose(1, 2)
    # SiglipVisionEmbeddings' position ids; padded patches take 0
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    position_ids = torch.zeros(len(masks), side * side, dtype=torch.long)
    for i, m in enumerate(masks):
        nb_h, nb_w = m[:, 0].sum(), m[0].sum()
        if nb_h == 0 or nb_w == 0:
            continue
        bh = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / nb_h), boundaries, right=True)
        bw = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / nb_w), boundaries, right=True)
        position_ids[i][m.view(-1)] = (bh[:, None] * side + bw).flatten()
    x = x + emb.position_embedding(position_ids)
    valid = masks.view(len(masks), -1)
    heads, dim = cfg.num_attention_heads, cfg.head_dim
    for layer in model.img_processor.encoder.layers:
        attn = layer.self_attn
        q, k, v = (t.view(len(masks), -1, heads, dim).transpose(1, 2)
                   for t in attn.qkv_proj(layer.layer_norm1(x)).chunk(3, dim=-1))
        bias = torch.zeros(valid.shape).masked_fill(~valid, float("-inf"))[:, None, None]
        o = F.softmax(q @ k.transpose(-1, -2) * dim ** -0.5 + bias, dim=-1) @ v
        # FlashAttention-2 leaves a padded query's output zero
        o = o.transpose(1, 2).reshape(len(masks), -1, c).masked_fill(~valid[..., None], 0)
        x = x + attn.out_proj(o)
        x = x + layer.mlp(layer.layer_norm2(x))
    pooled = F.avg_pool2d(x.view(-1, side, side, c).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    pooled = pooled.reshape(len(masks), -1, c)
    sub_gn, glb_gn = model.sub_GN, model.glb_GN
    glb = torch.cat([pooled[:1].reshape(1, half, half, c), sub_gn.repeat(1, half, 1, 1)], dim=2).reshape(1, -1, c)
    sub = pooled[1:].reshape(1, down, across, half, half, c).permute(0, 1, 3, 2, 4, 5)
    sub = sub.reshape(1, down * half, across * half, c)
    kept = masks[1:, 0::2, 0::2].reshape(1, down, across, half, half).permute(0, 1, 3, 2, 4)
    kept = kept.reshape(1, down * half, across * half)
    h, w = int(kept[0, :, 0].sum()), int(kept[0, 0, :].sum())
    sub = torch.cat([sub[:, :h, :w], sub_gn.repeat(1, h, 1, 1)], dim=2).reshape(1, -1, c)
    return model.img_projection(torch.cat([sub, glb_gn, glb], dim=1))[0]


def test_the_packed_node_is_the_reference_per_image():
    model = _model()
    node, attn = _node(model)
    images = _images(SIZES)
    (rows,) = _run(node, attn, [images])
    with torch.no_grad():
        want = torch.cat([_reference(model, p) for p in images])
    assert rows.shape == (sum(p.tokens[0] for p in images), TINY.text_hidden_size)
    torch.testing.assert_close(rows, want, rtol=1e-5, atol=1e-5)


def test_two_requests_in_one_step_are_each_alone():
    model = _model()
    node, attn = _node(model)
    first, second = _images(SIZES[:2]), _images(SIZES[2:])
    together = _run(node, attn, [first, second])
    alone = [_run(node, attn, [first])[0], _run(node, attn, [second])[0]]
    for got, want in zip(together, alone, strict=True):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(together[0][-10:], together[1][-10:])


def test_the_node_declares_a_segment_per_crop_and_global_view():
    model = _model()
    node, attn = _node(model)
    images = _images(SIZES[:1])
    layout = {"offset": [0], "tokens": images[0].tokens, "output": [VISION_OUTPUT],
              "grid": [list(images[0].grids[0])]}
    info = SimpleNamespace(step_metadata={MEDIA_LAYOUT: layout})
    inp = node.prepare_inputs("prefill_vision", info, {
        PIXEL_VALUES: [images[0].inputs[PIXEL_VALUES]], GLOBAL_PIXEL_VALUES: [images[0].inputs[GLOBAL_PIXEL_VALUES]],
    })
    # a 2 x 2 grid of 4 x 4-patch crops, 5 of 8 patch rows valid: the bottom crops keep one row
    assert images[0].grids == [(2, 2, 5, 8)]
    assert inp.resource_step_info == {"main": [16, 16, 16, 4, 4]} and inp.input_seq_len == 56
    step = node.declare_step("prefill_vision", ["r0"], [inp])
    assert [s.span for s in step.get(VISION_ATTN).segments] == [16, 16, 16, 4, 4]
    assert step.get(VISION_ATTN).causal is False
    with pytest.raises(ValueError, match="crops for its grid"):
        node.prepare_inputs("prefill_vision", info, {
            PIXEL_VALUES: [images[0].inputs[PIXEL_VALUES][:3]],
            GLOBAL_PIXEL_VALUES: [images[0].inputs[GLOBAL_PIXEL_VALUES]],
        })


def test_vision_resources():
    (spec,) = vision_resources(Phi4MMVisionConfig())
    assert spec.resource_key == VISION_ATTN and spec.nodes == {"vision_encoder"}
    assert (spec.config.num_qo_heads, spec.config.head_dim) == (16, 72)
    assert spec.config.max_segments_per_step == Phi4MMVisionEncoderSubmodule.MAX_BATCH_SIZE * 37


def test_the_loader_skips_what_the_feature_tap_never_reads():
    config = Phi4MMVisionConfig()
    assert config.layers_run == 26
    skipped = ["img_processor.encoder.layers.26.mlp.fc1.weight", "img_processor.post_layernorm.weight",
               "img_processor.head.probe", "img_processor.head.attention.in_proj_weight"]
    kept = ["img_processor.encoder.layers.25.mlp.fc1.weight", "img_processor.encoder.layers.2.self_attn.q_proj.bias",
            "sub_GN", "img_projection.2.weight", "img_processor.embeddings.position_embedding.weight"]
    assert all(unused_weight(n, config) for n in skipped)
    assert not any(unused_weight(n, config) for n in kept)
    assert CHECKPOINT_PREFIX == "model.embed_tokens_extend.image_embed."
