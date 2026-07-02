"""Backend-equivalence parity for the Qwen3-Omni native encoder's varlen attention.

Issue #131 ships several interchangeable varlen-attention backends (selectable via
``MSTAR_VARLEN_BACKEND``) plus a flash-attn fast path and a FlashInfer path. They
are only *correct* if they all compute the SAME block-diagonal (within-segment)
attention. This test pins that: for synthetic packed q/k/v segmented by
``cu_seqlens``, every available backend must match the dense block-diagonal
reference within fp tolerance — across audio-like (many tiny windows), vision-like
(few big segments), single, and ragged layouts, at both encoder head_dims
(audio=64, vision=72→FlashInfer-padded to 128), in fp32 (algorithm) and bf16
(production dtype).

This is the regression guard for any change to backend selection, the SDPA
fallbacks, the FlashInfer head-dim padding, or the adaptive heuristic. Complements
``test_qwen3_omni_native_encoders.py`` (full-encoder vs HF parity).
"""
import pytest
import torch

from mstar.model.qwen3_omni.components import audio_encoder as AE

DEVICE = "cuda:0"

# (name -> per-segment token lengths). Mirrors real encoder layouts.
LAYOUTS = {
    "audio_many_tiny": [50] * 16,        # audio: many ~50-token windows (launch-bound regime)
    "vision_few_big": [728, 728, 512],   # vision: few big segments
    "single_segment": [300],             # one clip / one image
    "ragged": [10, 200, 50, 333, 7],     # uneven
}
# (num_heads, head_dim): audio tower = 20x64, vision tower = 16x72 (FlashInfer pads 72->128)
HEAD_SHAPES = [(20, 64), (16, 72)]


def _make(lengths, H, D, dtype):
    torch.manual_seed(1234)
    total = sum(lengths)
    q = torch.randn(total, H, D, device=DEVICE, dtype=dtype)
    k = torch.randn(total, H, D, device=DEVICE, dtype=dtype)
    v = torch.randn(total, H, D, device=DEVICE, dtype=dtype)
    cu = torch.zeros(len(lengths) + 1, device=DEVICE, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, device=DEVICE, dtype=torch.int32).cumsum(0)
    return q, k, v, cu, max(lengths)


def _rel_cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    rel = (a - b).norm().item() / max(b.norm().item(), 1e-9)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return rel, cos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
@pytest.mark.parametrize("H,D", HEAD_SHAPES, ids=[f"{h}x{d}" for h, d in HEAD_SHAPES])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_varlen_backends_match_dense(layout, H, D, dtype):
    """Every available varlen backend == the dense block-diagonal reference."""
    lengths = LAYOUTS[layout]
    q, k, v, cu, max_seqlen = _make(lengths, H, D, dtype)
    scale = D ** -0.5

    # Reference: dense (total x total) block-diagonal mask SDPA — exact within-segment attention.
    ref = AE._sdpa_varlen_dense(q, k, v, cu, scale)

    # SDPA-family backends (mathematically identical to the dense reference).
    backends = {
        "per_segment": lambda: AE._sdpa_varlen_per_segment(q, k, v, cu, scale),
        "padded": lambda: AE._sdpa_varlen_padded(q, k, v, cu, scale),
        "adaptive": lambda: AE._sdpa_varlen_adaptive(q, k, v, cu, scale),
    }
    # Kernel backends are fp16/bf16 only — skip them in fp32.
    if dtype != torch.float32:
        if AE._FLASHINFER_AVAILABLE:
            backends["flashinfer"] = lambda: AE._flashinfer_varlen(q, k, v, cu, scale)
        if AE._FLASH_ATTN_AVAILABLE:
            from flash_attn import flash_attn_varlen_func
            backends["flash_attn"] = lambda: flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=False, softmax_scale=scale)

    # fp32 isolates the algorithm (tight); bf16 is the production dtype (rounding).
    rel_max, cos_min = (2e-3, 0.9999) if dtype == torch.float32 else (5e-2, 0.999)

    for name, fn in backends.items():
        out = fn()
        assert out.shape == ref.shape, f"{name}: shape {tuple(out.shape)} != {tuple(ref.shape)}"
        rel, cos = _rel_cos(out, ref)
        assert cos > cos_min and rel < rel_max, (
            f"backend={name} layout={layout} {H}x{D} {dtype}: cos={cos:.6f} relL2={rel:.4f}")


@pytest.mark.skipif(not (torch.cuda.is_available() and AE._FLASHINFER_AVAILABLE),
                    reason="flashinfer required")
@pytest.mark.parametrize("D", [64, 72], ids=["d64", "d72"])
def test_flashinfer_headdim_padding_is_exact(D):
    """FlashInfer pads head_dim to {64,128,256}; padding with zeros must be EXACT
    (the encoder relies on this for the graph-capturable path, esp. vision D=72)."""
    q, k, v, cu, _ = _make([64, 64], 16, D, torch.bfloat16)
    scale = D ** -0.5
    ref = AE._sdpa_varlen_dense(q, k, v, cu, scale)
    out = AE._flashinfer_varlen(q, k, v, cu, scale)
    assert out.shape[-1] == D, f"output head_dim {out.shape[-1]} != {D} (padding not sliced back)"
    rel, cos = _rel_cos(out, ref)
    assert cos > 0.999 and rel < 5e-2, f"flashinfer D={D}: cos={cos:.6f} relL2={rel:.4f}"
