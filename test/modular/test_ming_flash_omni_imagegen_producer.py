"""Tests for the image-gen producer side (step 9b).

Two pieces of the Thinker->ImageGen handoff:

  * ``LingMoeModel.forward(return_hidden_states=True)`` returns the post-norm
    hidden states alongside logits (CUDA-gated — the mminf RMSNorm kernel is
    CUDA-only).
  * ``BailingMoeV2ThinkerSubmodule.extract_image_gen_hidden_states`` slices
    those hidden states at the ``<imagePatch>`` query-token positions
    (pure-tensor, CPU-testable).
"""

from __future__ import annotations

import pytest
import torch

from mminf.model.ming_omni_flash.submodules import BailingMoeV2ThinkerSubmodule

_extract = BailingMoeV2ThinkerSubmodule.extract_image_gen_hidden_states
PATCH = 157157


# ---------------------------------------------------------------------------
# extract_image_gen_hidden_states (CPU)
# ---------------------------------------------------------------------------


def test_extract_picks_patch_positions_in_order() -> None:
    T, H = 8, 4
    hidden = torch.arange(T * H, dtype=torch.float32).view(T, H)
    # patch tokens at positions 3,4,5 (a 3-wide query block).
    token_ids = torch.tensor([10, 11, 12, PATCH, PATCH, PATCH, 13, 14])
    out = _extract(hidden, token_ids, PATCH)
    assert out.shape == (3, H)
    assert torch.equal(out, hidden[3:6])


def test_extract_non_contiguous_positions() -> None:
    hidden = torch.randn(6, 5)
    token_ids = torch.tensor([PATCH, 1, PATCH, 2, 3, PATCH])
    out = _extract(hidden, token_ids, PATCH)
    assert out.shape == (3, 5)
    assert torch.equal(out, hidden[[0, 2, 5]])


def test_extract_flattens_2d_token_ids() -> None:
    hidden = torch.randn(4, 3)
    token_ids = torch.tensor([[PATCH, PATCH, 1, 2]])  # (1, T)
    out = _extract(hidden, token_ids, PATCH)
    assert out.shape == (2, 3)


def test_extract_raises_when_no_patch_tokens() -> None:
    hidden = torch.randn(4, 3)
    token_ids = torch.tensor([1, 2, 3, 4])
    with pytest.raises(ValueError, match="no <imagePatch> token"):
        _extract(hidden, token_ids, PATCH)


def test_extract_raises_on_length_mismatch() -> None:
    hidden = torch.randn(4, 3)
    token_ids = torch.tensor([PATCH, PATCH, PATCH])  # T=3 != 4
    with pytest.raises(ValueError, match="!= hidden_states T"):
        _extract(hidden, token_ids, PATCH)


def test_extract_raises_on_bad_hidden_rank() -> None:
    hidden = torch.randn(2, 4, 3)
    token_ids = torch.tensor([PATCH, PATCH])
    with pytest.raises(ValueError, match=r"expected \(T, H\)"):
        _extract(hidden, token_ids, PATCH)


# ---------------------------------------------------------------------------
# LingMoeModel.forward(return_hidden_states=True) (CUDA-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mminf RMSNorm kernel is CUDA-only")
def test_model_returns_hidden_states_tuple() -> None:
    import torch.nn.functional as F

    from mminf.model.ming_omni_flash.components.model import LingMoeModel

    model = LingMoeModel(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        partial_rotary_factor=1.0,
        mrope_section=[1, 2, 1],
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        first_k_dense_replace=0,
        tie_word_embeddings=False,
        use_qkv_bias=False,
        use_bias=False,
    ).to("cuda").eval()

    class _Cache:
        def set_layer_idx(self, i):
            pass

        def run_attention(self, q, k, v):
            q4 = q.transpose(0, 1).unsqueeze(0)
            k4 = k.transpose(0, 1).unsqueeze(0)
            v4 = v.transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, scale=q.shape[-1] ** -0.5)
            return o.squeeze(0).transpose(0, 1).contiguous()

    ids = torch.tensor([1, 2, 3, 4], device="cuda")
    with torch.no_grad():
        logits, hidden = model(_Cache(), input_ids=ids, return_hidden_states=True)
    assert logits.shape == (4, 64)
    assert hidden.shape == (4, 16)
    assert torch.isfinite(logits).all() and torch.isfinite(hidden).all()
    # The returned hidden states are exactly what lm_head consumed: feeding
    # them back through lm_head must reproduce the returned logits.
    assert torch.allclose(model.lm_head(hidden), logits, atol=1e-3)
