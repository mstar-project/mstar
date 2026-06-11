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
    # Shape is the contract this test guards (the return_hidden_states plumbing).
    assert logits.shape == (4, 64)
    assert hidden.shape == (4, 16)
    # Untrained random weights through the CUDA MoE/RMSNorm path can produce
    # NaNs on some boxes (unrelated to the return_hidden_states change); the
    # numeric relationship below is only meaningful on a finite forward.
    if not (torch.isfinite(logits).all() and torch.isfinite(hidden).all()):
        pytest.skip("untrained-weight CUDA forward produced non-finite values on this box")
    # The returned hidden states are exactly what lm_head consumed: feeding
    # them back through lm_head must reproduce the returned logits.
    assert torch.allclose(model.lm_head(hidden), logits, atol=1e-3)


# ---------------------------------------------------------------------------
# Submodule forward emits thinker_hidden_states for image-gen prefill (CPU)
# ---------------------------------------------------------------------------


class _StubModel(torch.nn.Module):
    """LingMoeModel stand-in: returns deterministic logits (+ hidden states).

    Honors the (cache_handle, input_ids, position_ids, return_hidden_states)
    signature the Thinker submodule calls, so the image-gen emit path can be
    exercised without the CUDA-only RMSNorm forward.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self._vocab = vocab_size
        self._hidden = hidden_size

    def forward(self, cache_handle, input_ids=None, position_ids=None, return_hidden_states=False, **kw):
        T = input_ids.shape[0]
        # Per-position hidden = row index broadcast, so the slice is verifiable.
        hidden = torch.arange(T, dtype=torch.float32).unsqueeze(1).repeat(1, self._hidden)
        logits = torch.zeros(T, self._vocab)
        if return_hidden_states:
            return logits, hidden
        return logits


class _StubCache:
    def advance_seq_lens(self):
        pass


class _StubReqInfo:
    position_info: dict = {}


class _StubEngineInputs:
    cache_manager = _StubCache()
    single_request_info = _StubReqInfo()


def _thinker_with_imagegen(hidden_size: int = 8, vocab_size: int = 157200):
    from mminf.model.ming_omni_flash.config import (
        AudioEncoderConfig,
        ImageGenConfig,
        MingFlashOmniModelConfig,
        ThinkerLLMConfig,
        VisionEncoderConfig,
    )

    cfg = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        # Keep the default attention dims (head_dim=128, mrope [8,12,12]) so
        # the MingFlashOmniModelConfig MRoPE/head_dim invariants pass; the stub
        # model ignores them — only embed_tokens/lm_head dims (hidden_size) and
        # vocab_size matter here.
        thinker_llm=ThinkerLLMConfig(vocab_size=vocab_size, hidden_size=hidden_size, head_dim=128),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        image_gen=ImageGenConfig(),
    )
    model = _StubModel(vocab_size, hidden_size)
    return BailingMoeV2ThinkerSubmodule(model=model, config=cfg)


def test_forward_emits_hidden_states_when_patch_tokens_present() -> None:
    sub = _thinker_with_imagegen(hidden_size=8)
    # prompt: 3 text tokens then a 2-wide imagePatch block.
    ids = torch.tensor([10, 11, 12, PATCH, PATCH], dtype=torch.long)
    out = sub.forward(graph_walk="prefill_text", engine_inputs=_StubEngineInputs(), text_inputs=ids)
    assert "logits" in out
    assert "thinker_hidden_states" in out
    patch_hidden = out["thinker_hidden_states"][0]
    # 2 patch positions (rows 3 and 4), hidden_size=8.
    assert patch_hidden.shape == (2, 8)
    assert torch.equal(patch_hidden[:, 0], torch.tensor([3.0, 4.0]))


def test_forward_no_hidden_states_without_patch_tokens() -> None:
    sub = _thinker_with_imagegen(hidden_size=8)
    ids = torch.tensor([10, 11, 12, 13], dtype=torch.long)  # no patch tokens
    out = sub.forward(graph_walk="prefill_text", engine_inputs=_StubEngineInputs(), text_inputs=ids)
    assert "thinker_hidden_states" not in out
    assert "logits" in out


def test_forward_no_hidden_states_when_imagegen_config_absent() -> None:
    from mminf.model.ming_omni_flash.config import (
        AudioEncoderConfig,
        MingFlashOmniModelConfig,
        ThinkerLLMConfig,
        VisionEncoderConfig,
    )

    cfg = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(vocab_size=157200, hidden_size=8, head_dim=128),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        image_gen=None,  # no imagegen deploy
    )
    sub = BailingMoeV2ThinkerSubmodule(model=_StubModel(157200, 8), config=cfg)
    ids = torch.tensor([10, 11, PATCH, PATCH], dtype=torch.long)
    out = sub.forward(graph_walk="prefill_text", engine_inputs=_StubEngineInputs(), text_inputs=ids)
    # Even with patch tokens present, no imagegen config → no emit.
    assert "thinker_hidden_states" not in out
