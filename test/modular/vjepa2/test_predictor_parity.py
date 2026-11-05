"""Parity tests for the V-JEPA 2 masked latent predictor port.

Validates:
  - Output shape for full-mask and partial-mask cases
  - Sort/unsort invertibility
  - Bit-exact match against an inline reference in the full-mask case
    (where the argsort collapses to identity)
  - State dict key layout matches HF checkpoints

Reference math from:
  - transformers/src/transformers/models/vjepa2/modeling_vjepa2.py (lines 487-635)
"""

from __future__ import annotations

import torch

from mstar.model.vjepa2.components.predictor import (
    VJEPA2Predictor,
    VJEPA2PredictorEmbeddings,
)
from mstar.model.vjepa2.config import VJepa2Config


def _tiny_config() -> VJepa2Config:
    return VJepa2Config(
        patch_size=4,
        crop_size=16,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        pred_hidden_size=24,  # head_dim = 24/4 = 6; third = 2*(6//3//2) = 2
        pred_num_attention_heads=4,
        pred_num_hidden_layers=2,
        pred_num_mask_tokens=4,
        pred_mlp_ratio=2.0,
        hidden_act="gelu",
    )


def test_mask_tokens_match_the_reference_without_reading_a_mask_back():
    """The reference repeats the mask token up to the target mask's largest
    index, then gathers; sizing by a device value is a host read, which fails a
    CUDA graph capture. Same tokens, with no value read back."""
    from mstar.model.vjepa2.components.vit_encoder import apply_masks

    torch.manual_seed(0)
    config = _tiny_config()
    emb = VJEPA2PredictorEmbeddings(config).eval()
    torch.nn.init.normal_(emb.mask_tokens)
    b, n = 2, config.num_patches
    hidden = torch.randn(b, n, config.hidden_size)
    context = [torch.arange(n // 2).unsqueeze(0).repeat(b, 1)]
    target = [torch.arange(n // 2, n).unsqueeze(0).repeat(b, 1)]
    with torch.no_grad():
        embeddings, _ = emb(hidden, context, target)
        token = emb.mask_tokens[1]
        reference = apply_masks(token.repeat(b, int(target[0].max()) + 1, 1), target)
    assert torch.equal(embeddings[:, n:], reference)
    # on the meta device no value can be read back, as in a capture
    meta = [[m.to("meta") for m in masks] for masks in (context, target)]
    assert emb.to("meta")(hidden.to("meta"), *meta)[0].shape == embeddings.shape


def test_predictor_output_shape_full_mask():
    """With full context and full target mask, predictor emits one hidden per target patch."""
    torch.manual_seed(0)
    config = _tiny_config()
    predictor = VJEPA2Predictor(config).eval()

    b, n = 2, config.num_patches
    enc = torch.randn(b, n, config.hidden_size)
    mask_all = torch.arange(n).unsqueeze(0).repeat(b, 1)
    with torch.no_grad():
        out = predictor(enc, [mask_all], [mask_all])
    assert out.shape == (b, n, config.hidden_size)


def test_predictor_output_shape_partial_mask():
    """Context size != target size: output follows the target mask length."""
    torch.manual_seed(0)
    config = _tiny_config()
    predictor = VJEPA2Predictor(config).eval()

    b, n = 2, config.num_patches
    half = n // 2
    enc = torch.randn(b, n, config.hidden_size)
    ctx = torch.arange(half).unsqueeze(0).repeat(b, 1)
    tgt = (torch.arange(n - half) + half).unsqueeze(0).repeat(b, 1)
    with torch.no_grad():
        out = predictor(enc, [ctx], [tgt])
    assert out.shape == (b, n - half, config.hidden_size)


def test_predictor_sort_unsort_is_identity():
    """Sorting then unsorting must recover the input exactly."""
    torch.manual_seed(0)
    b, n, d = 3, 7, 5
    x = torch.randn(b, n, d)
    pos = torch.tensor(
        [
            [3, 1, 4, 1, 5, 9, 2],
            [0, 6, 2, 7, 5, 1, 3],
            [8, 8, 2, 0, 4, 1, 9],
        ],
        dtype=torch.long,
    )
    argsort = torch.argsort(pos, dim=1)
    sorted_x, sorted_pos = VJEPA2Predictor._sort_tokens(x, pos, argsort)
    unsorted = VJEPA2Predictor._unsort_tokens(sorted_x, argsort)
    assert torch.allclose(unsorted, x)


def test_predictor_matches_inline_reference_full_mask_fp32():
    """In the all-tokens case, argsort is a permutation but sort/unsort wrap
    the layers, so the output equals running the layers on the concatenation
    directly when we feed positions in ascending order.

    We use the simplest case where context_mask == target_mask == arange(N),
    which exercises the embedding concat + layer stack + final proj.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    predictor = VJEPA2Predictor(config).to(torch.float32).eval()

    b, n = 1, config.num_patches
    enc = torch.randn(b, n, config.hidden_size, dtype=torch.float32)
    mask_all = torch.arange(n).unsqueeze(0).repeat(b, 1)

    with torch.no_grad():
        ours = predictor(enc, [mask_all], [mask_all])

        # Inline: subselect context, embed, concat with mask tokens, run
        # layers (position_mask = concat(ctx, tgt) = [0..N, 0..N]), layernorm,
        # take target half, proj.
        ctx = predictor.embeddings.predictor_embeddings(enc)
        mask_idx = 1 % config.pred_num_mask_tokens
        target_token = predictor.embeddings.mask_tokens[mask_idx]
        target = target_token.repeat(b, n, 1)
        x = torch.cat([ctx, target], dim=1)  # [B, 2N, pred_hidden]
        pos = torch.cat([mask_all, mask_all], dim=1)  # [B, 2N]

        argsort = torch.argsort(pos, dim=1)
        x, pos_sorted = VJEPA2Predictor._sort_tokens(x, pos, argsort)

        for layer in predictor.layer:
            x = layer(x, position_mask=pos_sorted)
        x = predictor.layernorm(x)
        x = VJEPA2Predictor._unsort_tokens(x, argsort)
        x = x[:, n:]
        ref = predictor.proj(x)

    assert torch.allclose(ours, ref, atol=1e-5), f"max abs diff = {(ours - ref).abs().max().item()}"


def test_predictor_deterministic():
    """Same input, same seed → identical output."""
    config = _tiny_config()
    torch.manual_seed(0)
    predictor = VJEPA2Predictor(config).eval()
    b, n = 1, config.num_patches
    enc = torch.randn(b, n, config.hidden_size)
    mask = torch.arange(n).unsqueeze(0).repeat(b, 1)
    with torch.no_grad():
        a = predictor(enc, [mask], [mask])
        b2 = predictor(enc, [mask], [mask])
    assert torch.equal(a, b2)


def test_predictor_state_dict_keys():
    """Expected HF checkpoint key layout."""
    config = _tiny_config()
    predictor = VJEPA2Predictor(config)
    keys = set(predictor.state_dict().keys())
    assert "embeddings.predictor_embeddings.weight" in keys
    assert "embeddings.predictor_embeddings.bias" in keys
    assert "embeddings.mask_tokens" in keys
    assert "layer.0.attention.query.weight" in keys
    assert "layer.0.attention.proj.weight" in keys
    assert "layernorm.weight" in keys
    assert "proj.weight" in keys


def test_predictor_mask_tokens_zero_init():
    """When ``pred_zero_init_mask_tokens=True`` (default), the mask tokens
    parameter is allocated to zeros."""
    config = _tiny_config()
    predictor = VJEPA2Predictor(config)
    assert torch.equal(
        predictor.embeddings.mask_tokens,
        torch.zeros_like(predictor.embeddings.mask_tokens),
    )


def test_predictor_embeddings_output_layout():
    """Embeddings concatenates context then target; combined mask
    concatenates context_mask then target_mask."""
    torch.manual_seed(0)
    config = _tiny_config()
    emb = VJEPA2PredictorEmbeddings(config).eval()

    b, n_ctx, n_tgt = 2, 5, 3
    ctx = torch.randn(b, n_ctx, config.hidden_size)
    context_mask = [torch.arange(n_ctx).unsqueeze(0).repeat(b, 1)]
    target_mask = [(torch.arange(n_tgt) + n_ctx).unsqueeze(0).repeat(b, 1)]
    with torch.no_grad():
        out_emb, out_mask = emb(ctx, context_mask, target_mask)
    assert out_emb.shape == (b, n_ctx + n_tgt, config.pred_hidden_size)
    assert out_mask.shape == (b, n_ctx + n_tgt)
    # First n_ctx positions should match context_mask[0]
    assert torch.equal(out_mask[:, :n_ctx], context_mask[0])
    assert torch.equal(out_mask[:, n_ctx:], target_mask[0])
