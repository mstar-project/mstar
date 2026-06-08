"""Unit tests for Ling-2.0 MoE block + decoder layer + full thinker model.

Tiny-config tests (vocab=64, hidden=32, layers=2, num_experts=8) that
exercise the routing-mask paths, the dense-vs-MoE layer branch, and the
end-to-end forward shape.

Step-3b scope: no KV cache, no real weights, no batching. The model
takes ``(T,)`` token ids or ``(T, hidden)`` embeds and returns
``(T, vocab_size)`` logits.

CUDA-only tests are gated on ``torch.cuda.is_available()`` because
LingAttention's RMSNorm goes through flashinfer's CUDA kernel — same
constraint as step 3a's attention tests.
"""

from __future__ import annotations

import pytest
import torch

from mminf.model.ming_omni_flash.components.decoder_layer import (
    LingDecoderLayer,
)
from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.ming_omni_flash.components.moe import LingMoeBlock
from mminf.model.ming_omni_flash.components.rope import (
    LingPartialMRotaryEmbedding,
)

torch.manual_seed(2026)


# ---------------------------------------------------------------------------
# LingMoeBlock
# ---------------------------------------------------------------------------


def _make_moe(hidden_size: int = 16) -> LingMoeBlock:
    return LingMoeBlock(
        hidden_size=hidden_size,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        num_shared_experts=1,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
    )


def test_ling_moe_block_text_only_forward_shape() -> None:
    """Vanilla text routing: masks=None, output shape matches input.

    Initialise fused expert + shared expert weights to small randoms so
    the output isn't trivially zero.
    """
    moe = _make_moe()
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(std=0.05)
        moe.experts.down_proj.normal_(std=0.05)
    x = torch.randn(6, 16)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_ling_moe_block_image_mask_routes_through_image_gate() -> None:
    """When ``image_mask`` is True for some positions, those positions
    receive the chosen expert set from ``image_gate`` instead of ``gate``.

    Force the image gate to deterministically pick a known expert by
    spiking one input dim and one image_gate weight column; verify that
    expert is in the per-row selection at masked positions and absent
    at unmasked positions.
    """
    moe = _make_moe()
    # Make the text gate strongly prefer expert 0 across all inputs;
    # make the image gate strongly prefer expert 5.
    with torch.no_grad():
        moe.gate.gate.weight.zero_()
        moe.gate.gate.weight[0, 0] = 10.0
        moe.image_gate.gate.weight.zero_()
        moe.image_gate.gate.weight[5, 0] = 10.0
        moe.audio_gate.gate.weight.zero_()
        moe.experts.gate_up_proj.normal_(std=0.05)
        moe.experts.down_proj.normal_(std=0.05)

    N = 6
    x = torch.zeros(N, 16)
    x[:, 0] = 1.0  # light up the boosted input dim
    image_mask = torch.tensor([True, True, True, False, False, False])

    # Run the routing path directly so we can check the chosen indices,
    # since the forward returns post-dispatch tensors only.
    _, _, text_idx = moe.gate(x)
    _, _, image_idx = moe.image_gate(x)
    image_mask_n = image_mask.reshape(N, 1).bool()
    selected_idx = torch.where(image_mask_n, image_idx, text_idx)

    # Masked rows: expert 5 (image gate's pick) appears.
    assert (selected_idx[:3] == 5).any(dim=-1).all(), selected_idx[:3]
    # Unmasked rows: expert 0 (text gate's pick) appears.
    assert (selected_idx[3:] == 0).any(dim=-1).all(), selected_idx[3:]
    # Masked rows do NOT contain expert 0 (text gate's only pick).
    assert not (selected_idx[:3] == 0).any(), selected_idx[:3]

    # And the forward itself runs through end-to-end with the mask:
    out = moe(x, image_mask=image_mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_ling_moe_block_shared_expert_contributes() -> None:
    """Output differs when the shared expert has non-zero weights vs
    zeroed weights — proves the shared expert isn't dead code."""
    moe = _make_moe()
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(std=0.05)
        moe.experts.down_proj.normal_(std=0.05)
        # Start with shared expert zeroed.
        for p in moe.shared_expert.parameters():
            p.zero_()
    x = torch.randn(4, 16)
    out_zero_shared = moe(x).clone()

    with torch.no_grad():
        for p in moe.shared_expert.parameters():
            p.normal_(std=0.1)
    out_with_shared = moe(x)
    assert not torch.allclose(out_zero_shared, out_with_shared), (
        "shared expert weights had no effect — possibly skipped in forward"
    )


def test_ling_moe_block_rejects_bad_mask_shape() -> None:
    """A mask whose total elements don't match num_tokens raises."""
    moe = _make_moe()
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(std=0.05)
        moe.experts.down_proj.normal_(std=0.05)
    x = torch.randn(5, 16)
    bad = torch.zeros(3, dtype=torch.bool)   # wrong length
    with pytest.raises(ValueError, match="image_mask"):
        moe(x, image_mask=bad)


# ---------------------------------------------------------------------------
# LingMoeModel — input_ids / input_embeds / shape contracts
# ---------------------------------------------------------------------------


def _tiny_model_kwargs() -> dict:
    """Tiny config (~K params, runs on CPU or CUDA in <1s).

    head_dim=8, partial=0.5 → rotary_dim=4, rotary_dim//2=2 → mrope
    section must sum to 2. [1, 1, 0] is the simplest valid split.
    """
    return dict(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4, num_kv_heads=2, head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=128,
        partial_rotary_factor=0.5, mrope_section=[1, 1, 0],
        num_experts=8, num_experts_per_tok=2,
        num_shared_experts=1,
        n_group=2, topk_group=1,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
    )


def _init_dispatch_weights(model: LingMoeModel) -> None:
    """Initialise fused expert tensors so _dispatch produces non-trivial
    output (the constructor allocates them ``torch.empty``)."""
    with torch.no_grad():
        for layer in model.layers:
            if layer.is_moe:
                layer.mlp.experts.gate_up_proj.normal_(std=0.05)
                layer.mlp.experts.down_proj.normal_(std=0.05)


def test_ling_moe_model_input_ids_xor_embeds_required() -> None:
    """Both or neither of input_ids / input_embeds raises."""
    m = LingMoeModel(**_tiny_model_kwargs())
    with pytest.raises(ValueError, match="Exactly one"):
        m(input_ids=None, input_embeds=None)
    with pytest.raises(ValueError, match="Exactly one"):
        m(input_ids=torch.zeros(3, dtype=torch.long),
          input_embeds=torch.zeros(3, 32))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="LingAttention uses mminf RMSNorm (CUDA-only via flashinfer)")
def test_ling_moe_model_forward_with_input_ids_shape() -> None:
    """Forward with (T,) token ids returns (T, vocab_size) finite logits."""
    # bf16 — required by mminf's fused MoE kernel (asserts dtype in
    # {bf16, fp16}). The real model loads bf16 weights, so this matches.
    m = LingMoeModel(**_tiny_model_kwargs()).cuda().to(torch.bfloat16)
    _init_dispatch_weights(m)
    T = 5
    input_ids = torch.randint(0, 64, (T,), device="cuda")
    out = m(input_ids=input_ids)
    assert out.shape == (T, 64)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="LingAttention uses mminf RMSNorm (CUDA-only via flashinfer)")
def test_ling_moe_model_forward_with_input_embeds_shape() -> None:
    """Forward bypassing embed_tokens via (T, hidden) input_embeds."""
    m = LingMoeModel(**_tiny_model_kwargs()).cuda().to(torch.bfloat16)
    _init_dispatch_weights(m)
    T = 4
    embeds = torch.randn(T, 32, device="cuda", dtype=torch.bfloat16)
    out = m(input_embeds=embeds)
    assert out.shape == (T, 64)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="LingAttention uses mminf RMSNorm (CUDA-only via flashinfer)")
def test_ling_decoder_layer_dense_vs_moe_paths_differ() -> None:
    """Layer 0 (dense GatedMLP) and layer 1 (MoE) on the same input must
    produce different outputs — verifies the layer-index branch is wired."""
    rotary = LingPartialMRotaryEmbedding(
        head_dim=8, partial_rotary_factor=0.5,
        mrope_section=[1, 1, 0], rope_theta=10000.0,
        max_position_embeddings=64,
    ).cuda()
    common = dict(
        first_k_dense_replace=1,
        hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_attention_heads=4, num_kv_heads=2, head_dim=8,
        rms_norm_eps=1e-6,
        num_experts=8, num_experts_per_tok=2,
        num_shared_experts=1, n_group=2, topk_group=1,
        routed_scaling_factor=1.0,
        rotary=rotary,
    )
    dense = LingDecoderLayer(layer_idx=0, **common).cuda().to(torch.bfloat16)
    moe = LingDecoderLayer(layer_idx=1, **common).cuda().to(torch.bfloat16)
    with torch.no_grad():
        moe.mlp.experts.gate_up_proj.normal_(std=0.05)
        moe.mlp.experts.down_proj.normal_(std=0.05)
    # Copy attention + norms so any output diff comes from the FFN branch only.
    moe.input_layernorm.load_state_dict(dense.input_layernorm.state_dict())
    moe.post_attention_layernorm.load_state_dict(
        dense.post_attention_layernorm.state_dict()
    )
    moe.self_attn.load_state_dict(dense.self_attn.state_dict())

    assert dense.is_moe is False and moe.is_moe is True
    x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(3, device="cuda")
    out_dense = dense(x, pos)
    out_moe = moe(x, pos)
    assert not torch.allclose(out_dense, out_moe), (
        "dense and MoE layer paths produced identical output"
    )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="LingAttention uses mminf RMSNorm (CUDA-only via flashinfer)")
def test_ling_moe_model_causal() -> None:
    """Appending a later token doesn't change earlier-position logits.

    Strongest end-to-end guard that nothing in the MoE / mask / rope
    plumbing accidentally lets future tokens influence past ones.
    """
    m = LingMoeModel(**_tiny_model_kwargs()).cuda().to(torch.bfloat16).eval()
    _init_dispatch_weights(m)
    input_ids = torch.randint(0, 64, (4,), device="cuda")
    out_a = m(input_ids=input_ids)

    extended = torch.cat([input_ids, torch.randint(0, 64, (1,), device="cuda")])
    out_b = m(input_ids=extended)
    # bf16 tolerance — 2 layers' worth of bf16 ops drift more than fp32.
    assert torch.allclose(out_a, out_b[:4], atol=0.05), (
        "causal mask leaked: appending a token changed earlier-position logits"
    )
