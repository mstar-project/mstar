"""mstar::varlen_attention (AITER) vs per-segment SDPA on a ROCm GPU. Covers
the encoder head dims (72 for SigLIP2, 64/80/128), causal and non-causal,
ragged segments, torch.compile and an HF encoder."""

import pytest
import torch
import torch.nn.functional as F

from mstar.model.components import hf_attention

pytestmark = pytest.mark.skipif(not hf_attention.varlen_attention_available(), reason="needs a ROCm GPU with AITER")


def _reference(q, k, v, cu_q, cu_k, causal, scale):
    outs = []
    for i in range(cu_q.numel() - 1):
        qs = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1).float()
        ks = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1).float()
        vs = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1).float()
        if ks.shape[0] != qs.shape[0]:
            ks = ks.repeat_interleave(qs.shape[0] // ks.shape[0], dim=0)
            vs = vs.repeat_interleave(qs.shape[0] // vs.shape[0], dim=0)
        mask = None
        if causal:
            # bottom-right aligned like flash_attn and AITER
            lq, lk = qs.shape[1], ks.shape[1]
            mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=lk - lq)
        outs.append(F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, scale=scale).transpose(0, 1))
    return torch.cat(outs).to(q.dtype)


def _cu(lengths):
    return torch.tensor([0, *lengths], device="cuda").cumsum(0).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 72, 80, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(dtype, head_dim, causal):
    torch.manual_seed(0)
    lengths = [1, 37, 256, 1000, 64]
    total, heads = sum(lengths), 16
    q, k, v = (torch.randn(total, heads, head_dim, device="cuda", dtype=dtype) for _ in range(3))
    cu = _cu(lengths)
    out = hf_attention.varlen_attention(q, k, v, cu, cu, max(lengths), max(lengths), causal, head_dim ** -0.5)
    torch.testing.assert_close(out, _reference(q, k, v, cu, cu, causal, head_dim ** -0.5), atol=2e-2, rtol=2e-2)


def test_gqa_and_cross_lengths():
    torch.manual_seed(0)
    lq, lk = [5, 100, 33], [50, 100, 200]
    q = torch.randn(sum(lq), 8, 64, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(sum(lk), 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    cu_q, cu_k = _cu(lq), _cu(lk)
    for causal in (False, True):
        out = hf_attention.varlen_attention(q, k, v, cu_q, cu_k, max(lq), max(lk), causal, None)
        torch.testing.assert_close(out, _reference(q, k, v, cu_q, cu_k, causal, None), atol=2e-2, rtol=2e-2)


def test_strided_inputs():
    # Qwen3-Omni's vision tower hands over views of one fused qkv projection
    torch.manual_seed(0)
    lengths = [64, 200]
    qkv = torch.randn(sum(lengths), 3, 16, 72, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv.unbind(1)
    cu = _cu(lengths)
    out = hf_attention.varlen_attention(q, k, v, cu, cu, 200, 200, False, None)
    torch.testing.assert_close(out, _reference(q, k, v, cu, cu, False, None), atol=2e-2, rtol=2e-2)


def test_under_torch_compile():
    torch.manual_seed(0)
    lengths = [16, 300, 77]
    q, k, v = (torch.randn(sum(lengths), 4, 80, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    cu = _cu(lengths)

    def f(q, k, v, cu, m):
        return hf_attention.varlen_attention(q * 1.0, k, v, cu, cu, m, m, False, None) + 1

    compiled = torch.compile(f, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, cu, 300), f(q, k, v, cu, 300), atol=1e-2, rtol=1e-2)


def test_hf_audio_tower_matches_sdpa(monkeypatch):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    monkeypatch.setattr(hf_attention, "flash_attn_installed", lambda: False)
    config = dict(
        d_model=256, encoder_attention_heads=4, encoder_layers=2, encoder_ffn_dim=512,
        output_dim=128, num_mel_bins=128, downsample_hidden_size=64,
    )
    torch.manual_seed(0)
    varlen = hf_attention.build_hf_encoder(Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoderConfig(**config))
    sdpa = Qwen3OmniMoeAudioEncoder._from_config(Qwen3OmniMoeAudioEncoderConfig(**config), attn_implementation="sdpa")
    sdpa.load_state_dict(varlen.state_dict())
    varlen, sdpa = (m.to("cuda", torch.bfloat16).eval() for m in (varlen, sdpa))
    feature_lens = torch.tensor([300, 2500], device="cuda")
    features = torch.randn(128, int(feature_lens.sum()), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = varlen(features, feature_lens=feature_lens).last_hidden_state.float()
        want = sdpa(features, feature_lens=feature_lens).last_hidden_state.float()
    cos = F.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
    assert cos > 0.999, cos
