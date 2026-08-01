"""Parity for the encoder CUDA-graph path (the production default).

The eager parity tests in ``test_qwen3_omni_native_encoders.py`` exercise only the
eager forward.  The shipping default runs the transformer block loop through a
``PiecewiseCudaGraphRunner`` with the FlashInfer varlen backend.  That path had
ZERO coverage, and it was in fact silently broken once: ``varlen_attention``
preferred flash-attn even while a capture override was live, so capture threw and
the encoder fell back to eager on every request.

This test pins the contract for the graph path:
  1. a graph is ACTUALLY captured AND the encoder actually replays it -- asserted
     via ``encoder_path_counts()``, not just by counting captured graphs, because
     "a graph exists" and "the forward used it" are different claims and only the
     second one is what production depends on,
  2. graph replay == eager (same kernels, same inputs -> essentially exact),
  3. graph replay == HF reference (still correct through capture).

Small random-weight encoders => no checkpoint, runs on one GPU in seconds.
Requires CUDA + flashinfer (the only capture-legal varlen backend).
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
# graph-vs-eager: same kernels/inputs, only replay differs -> essentially exact.
GRAPH_EAGER_MAXABS = 5e-3
# graph-vs-HF (bf16, flashinfer attn vs sdpa): directional check.
HF_COS_MIN = 0.99

# Deliberately wider and finer than production so a tiny test layout always finds
# a bucket; capture cost is trivial at this model size.  A layout that fits none
# of these would fall back to eager, and the path assertion below catches that.
TEST_TOKEN_BUCKETS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
TEST_CAPTURE_BS = [32]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


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


def _require_flashinfer():
    """flashinfer must be importable *inside the encoder module* -- the encoder
    caches its own availability flag at import time."""
    pytest.importorskip("flashinfer")
    import mstar.model.qwen3_omni.components.audio_encoder as AE
    if not AE._FLASHINFER_AVAILABLE:
        pytest.skip("flashinfer not importable inside encoder module")
    return AE


def _build_runner(encoder):
    """Capture a PiecewiseCudaGraphRunner for one encoder, on test-sized buckets."""
    from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner

    device = torch.device(DEVICE)
    cfg = encoder.get_piecewise_cuda_graph_config(device, DTYPE)
    assert cfg is not None, (
        "encoder declined to provide a piecewise config -- the capture path is "
        "unreachable, so production would silently run eager"
    )
    cfg.total_tokens = list(TEST_TOKEN_BUCKETS)
    cfg.capture_batch_sizes = list(TEST_CAPTURE_BS)
    runner = PiecewiseCudaGraphRunner(
        config=cfg, device=device, autocast_dtype=DTYPE,
    )
    runner.warmup_and_capture()
    assert runner.graphs, "PiecewiseCudaGraphRunner captured no graphs"
    return runner


def _paths_delta(AE, before):
    after = AE.encoder_path_counts()
    return {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}


@requires_cuda
def test_vision_encoder_graph_eager_hf_parity():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder,
    )

    from mstar.model.qwen3_omni.components.vision_encoder import (
        NativeQwen3OmniVisionEncoder,
    )
    AE = _require_flashinfer()
    torch.manual_seed(0)
    cfg = _small_vision_cfg()
    hf = Qwen3OmniMoeVisionEncoder._from_config(cfg, attn_implementation="sdpa").to(DEVICE, DTYPE).eval()
    nat = NativeQwen3OmniVisionEncoder(cfg).to(DEVICE, DTYPE).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    assert not miss and not unexp

    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    g = torch.tensor([[1, 8, 8]], device=DEVICE)
    pv = torch.randn(8 * 8, rows, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        emb_eager, _ds_e = nat(pv, g)                      # piecewise_runner=None
    runner = _build_runner(nat)
    before = AE.encoder_path_counts()
    with torch.no_grad():
        emb_graph, _ds_g = nat(pv, g, piecewise_runner=runner)
    torch.cuda.synchronize()
    delta = _paths_delta(AE, before)
    with torch.no_grad():
        o = hf(pv, grid_thw=g)

    assert delta.get("vision.piecewise", 0) == 1 and delta.get("vision.eager", 0) == 0, (
        f"vision encoder did NOT replay the captured graph (paths={delta}) -> "
        "silent eager fallback, the default path is broken"
    )
    maxabs = (emb_graph.float() - emb_eager.float()).abs().max().item()
    assert maxabs < GRAPH_EAGER_MAXABS, f"vision graph vs eager max-abs={maxabs:.3e}"
    cos = _cos(emb_graph, o.pooler_output)
    assert cos > HF_COS_MIN, f"vision graph vs HF cos={cos:.5f}"


@requires_cuda
def test_audio_encoder_graph_eager_hf_parity():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from mstar.model.qwen3_omni.components.audio_encoder import (
        NativeQwen3OmniAudioEncoder,
    )
    AE = _require_flashinfer()
    torch.manual_seed(0)
    cfg = _small_audio_cfg()
    hf = Qwen3OmniMoeAudioEncoder._from_config(cfg, attn_implementation="sdpa").to(DEVICE, DTYPE).eval()
    nat = NativeQwen3OmniAudioEncoder(cfg).to(DEVICE, DTYPE).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    assert not miss and not unexp

    lens = torch.tensor([800], device=DEVICE)
    feat = torch.randn(cfg.num_mel_bins, 800, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_eager = nat(feat, lens)                        # piecewise_runner=None
    runner = _build_runner(nat)
    before = AE.encoder_path_counts()
    with torch.no_grad():
        out_graph = nat(feat, lens, piecewise_runner=runner)
    torch.cuda.synchronize()
    delta = _paths_delta(AE, before)
    with torch.no_grad():
        ref = hf(feat, feature_lens=lens).last_hidden_state

    assert delta.get("audio.piecewise", 0) == 1 and delta.get("audio.eager", 0) == 0, (
        f"audio encoder did NOT replay the captured graph (paths={delta}) -> "
        "silent eager fallback, the default path is broken"
    )
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

    Runs on CPU with zero tensors and a stubbed flashinfer, so it needs neither a
    GPU nor flashinfer -- deliberately NOT under ``requires_cuda``, since this is
    the only standing guard for the regression and it must run in CI.
    """
    import mstar.model.qwen3_omni.components.audio_encoder as AE

    called = {"flash": False, "flashinfer": False}
    # Every global this test mutates must be restored: leaking
    # ``flash_attn_varlen_func``/``_FLASH_ATTN_AVAILABLE`` makes every later
    # varlen test in the same session dispatch into the stub below and fail.
    saved = (
        AE._flashinfer_varlen,
        AE._fi_override,
        AE.flash_attn_varlen_func,
        AE._FLASH_ATTN_AVAILABLE,
    )

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
        (
            AE._flashinfer_varlen,
            _restore_override,
            AE.flash_attn_varlen_func,
            AE._FLASH_ATTN_AVAILABLE,
        ) = saved
        AE.set_fi_override(_restore_override)
    assert called["flashinfer"] and not called["flash"]
