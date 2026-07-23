"""Parity for the encoder CUDA-graph path (the production default).

The eager parity tests in ``test_qwen3_omni_native_encoders.py`` exercise only the
eager forward.  The shipping default runs the transformer block loop through a
captured CUDA graph (``MSTAR_ENCODER_CUDA_GRAPH=1`` + FlashInfer varlen backend,
both defaults).  That path had ZERO coverage, and it was in fact silently broken:
``varlen_attention`` preferred flash-attn even while a capture override was live,
so capture threw and the encoder fell back to eager on every request.

This test pins the contract for the graph path:
  1. a graph is ACTUALLY captured (guards the silent eager-fallback regression),
  2. graph replay == eager (must be bit-exact-ish: same kernels, same inputs),
  3. graph replay == HF reference (the encoder is still correct through capture).

Small random-weight encoders => no checkpoint, runs on one GPU in seconds.
Requires CUDA + flashinfer (the only capture-legal varlen backend).
"""
from __future__ import annotations

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
# graph-vs-eager: same kernels/inputs, only replay differs -> essentially exact.
GRAPH_EAGER_MAXABS = 5e-3
# graph-vs-HF (bf16, flashinfer attn vs sdpa): directional check.
HF_COS_MIN = 0.99


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _small_vision_cfg():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig,
    )
    return Qwen3OmniMoeVisionEncoderConfig(
        depth=4, hidden_size=64, num_heads=4, intermediate_size=128,
        out_hidden_size=64, deepstack_visual_indexes=[1, 2])


def _small_audio_cfg():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    cfg = Qwen3OmniMoeAudioEncoderConfig(
        d_model=64, encoder_attention_heads=4, encoder_ffn_dim=128,
        encoder_layers=4, output_dim=64)
    cfg.n_window, cfg.n_window_infer = 50, 800
    return cfg


def _force_flashinfer_backend():
    """Pin the capture-legal backend on the module globals (env is read at import
    time, which may already have happened)."""
    import mstar.model.qwen3_omni.components.audio_encoder as AE
    if not AE._FLASHINFER_AVAILABLE:
        pytest.skip("flashinfer not importable inside encoder module")
    AE._VARLEN_BACKEND = "flashinfer"
    return AE


def _run(encoder, args, cuda_graph: bool):
    import os
    os.environ["MSTAR_ENCODER_CUDA_GRAPH"] = "1" if cuda_graph else "0"
    encoder._cg_cache.clear()
    encoder._cg_warmed = False
    with torch.no_grad():
        out = encoder(*args)
    torch.cuda.synchronize()
    return out, len(encoder._cg_cache)


def test_vision_encoder_graph_eager_hf_parity():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder,
    )

    from mstar.model.qwen3_omni.components.vision_encoder import (
        NativeQwen3OmniVisionEncoder,
    )
    _force_flashinfer_backend()
    torch.manual_seed(0)
    cfg = _small_vision_cfg()
    hf = Qwen3OmniMoeVisionEncoder._from_config(cfg, attn_implementation="sdpa").to(DEVICE, DTYPE).eval()
    nat = NativeQwen3OmniVisionEncoder(cfg).to(DEVICE, DTYPE).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    assert not miss and not unexp

    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    g = torch.tensor([[1, 8, 8]], device=DEVICE)
    pv = torch.randn(8 * 8, rows, device=DEVICE, dtype=DTYPE)

    (emb_eager, _ds_e), ncap_eager = _run(nat, (pv, g), cuda_graph=False)
    (emb_graph, _ds_g), ncap_graph = _run(nat, (pv, g), cuda_graph=True)
    with torch.no_grad():
        o = hf(pv, grid_thw=g)

    assert ncap_graph > 0, ("vision encoder captured NO CUDA graph -> silent eager "
                            "fallback (the default path is broken)")
    maxabs = (emb_graph.float() - emb_eager.float()).abs().max().item()
    assert maxabs < GRAPH_EAGER_MAXABS, f"vision graph vs eager max-abs={maxabs:.3e}"
    cos = _cos(emb_graph, o.pooler_output)
    assert cos > HF_COS_MIN, f"vision graph vs HF cos={cos:.5f}"


def test_audio_encoder_graph_eager_hf_parity():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from mstar.model.qwen3_omni.components.audio_encoder import (
        NativeQwen3OmniAudioEncoder,
    )
    _force_flashinfer_backend()
    torch.manual_seed(0)
    cfg = _small_audio_cfg()
    hf = Qwen3OmniMoeAudioEncoder._from_config(cfg, attn_implementation="sdpa").to(DEVICE, DTYPE).eval()
    nat = NativeQwen3OmniAudioEncoder(cfg).to(DEVICE, DTYPE).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    assert not miss and not unexp

    lens = torch.tensor([800], device=DEVICE)
    feat = torch.randn(cfg.num_mel_bins, 800, device=DEVICE, dtype=DTYPE)

    out_eager, _ = _run(nat, (feat, lens), cuda_graph=False)
    out_graph, ncap_graph = _run(nat, (feat, lens), cuda_graph=True)
    with torch.no_grad():
        ref = hf(feat, feature_lens=lens).last_hidden_state

    assert ncap_graph > 0, ("audio encoder captured NO CUDA graph -> silent eager "
                            "fallback (the default path is broken)")
    e, gph = out_eager.last_hidden_state, out_graph.last_hidden_state
    maxabs = (gph.float() - e.float()).abs().max().item()
    assert maxabs < GRAPH_EAGER_MAXABS, f"audio graph vs eager max-abs={maxabs:.3e}"
    cos = _cos(gph, ref)
    assert cos > HF_COS_MIN, f"audio graph vs HF cos={cos:.5f}"


def test_varlen_attention_uses_flashinfer_under_capture_override():
    """White-box regression guard for the silent-fallback bug.

    flash-attn's varlen kernel is not CUDA-graph-capturable for the production
    encoder head dims, so while a capture override is live ``varlen_attention``
    MUST route to the flashinfer path and MUST NOT call flash-attn.  This catches
    the regression even on tiny shapes (where flash-attn capture happens to work),
    which the end-to-end small-encoder tests above cannot.
    """
    import mstar.model.qwen3_omni.components.audio_encoder as AE

    called = {"flash": False, "flashinfer": False}
    orig_fi = AE._flashinfer_varlen
    orig_override = AE._fi_override

    def fake_flash(*a, **k):
        called["flash"] = True
        raise AssertionError("flash-attn must not be used while a capture override is live")

    def fake_fi(q, k, v, cu_seqlens, scale):
        called["flashinfer"] = True
        return q  # shape-compatible dummy

    AE.flash_attn_varlen_func = fake_flash  # type: ignore[attr-defined]
    AE._flashinfer_varlen = fake_fi
    AE._FLASH_ATTN_AVAILABLE = True
    AE.set_fi_override({"sentinel": True})
    try:
        q = torch.zeros(4, 2, 16)
        AE.varlen_attention(q, q, q, torch.tensor([0, 4]), 4, 0.1)
    finally:
        AE._flashinfer_varlen = orig_fi
        AE.set_fi_override(orig_override)
    assert called["flashinfer"] and not called["flash"]


if __name__ == "__main__":
    test_vision_encoder_graph_eager_hf_parity()
    print("vision graph/eager/HF parity OK")
    test_audio_encoder_graph_eager_hf_parity()
    print("audio graph/eager/HF parity OK")
    test_varlen_attention_uses_flashinfer_under_capture_override()
    print("varlen dispatch-under-capture OK")
