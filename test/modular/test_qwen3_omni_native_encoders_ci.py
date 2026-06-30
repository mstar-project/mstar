"""CI-runnable structural parity for the native Qwen3-Omni encoders.

The full parity test (``test_qwen3_omni_native_encoders.py``) needs the 30B
checkpoint + CUDA + flash-attn, so it auto-skips in CI and provides no standing
guarantee. This module fills that gap: it builds SMALL HF + native encoders from
seeded random weights, on CPU, with the flash-attn import forced off (so the
native path takes its SDPA fallback), and asserts implementation equivalence —
no checkpoint, no GPU, no flash-attn required.

It also asserts parity at EVERY layer (not just the final output), which
characterizes the intermediate residual-stream / DeepStack drift that the
end-only assertion under-reports. In fp32 the native impl is mathematically
identical to HF, so per-layer cosine should be ~1.0.

Skips only if ``transformers`` is unavailable.
"""
from __future__ import annotations

import sys

import pytest
import torch

# Force the native encoders' SDPA fallback (no flash-attn in CI).
sys.modules.setdefault("flash_attn", None)

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def _force_sdpa_cpu_path():
    """``sys.modules.setdefault`` above only blocks a *future* import; if the GPU
    parity test file already imported ``audio_encoder`` in the same session,
    ``_FLASH_ATTN_AVAILABLE`` / the flashinfer backend are already baked in and the
    CPU structural tests would route to a CUDA-only kernel and error. Force the
    pure-SDPA path for the duration of each CI test regardless of import order."""
    import mstar.model.qwen3_omni.components.audio_encoder as AE
    saved = (AE._FLASH_ATTN_AVAILABLE, AE._VARLEN_BACKEND)
    AE._FLASH_ATTN_AVAILABLE = False
    AE._VARLEN_BACKEND = "per_segment"   # SDPA, CPU-capable, no flashinfer/flash-attn
    try:
        yield
    finally:
        AE._FLASH_ATTN_AVAILABLE, AE._VARLEN_BACKEND = saved

DEVICE = "cpu"
DTYPE = torch.float32          # fp32 => native is bit-for-bit equal to HF
COS_MIN = 0.9999
RELL2_MAX = 5e-3


def _cmp(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (a - b).norm().item() / max(b.norm().item(), 1e-9)
    return cos, rel


def _hook_residuals(modules, store):
    for i, m in enumerate(modules):
        m.register_forward_hook(
            lambda _m, _in, out, i=i: store.__setitem__(
                i, out[0] if isinstance(out, tuple) else out))


def _small_vision_cfg():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig,
    )
    return Qwen3OmniMoeVisionEncoderConfig(
        depth=4,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        out_hidden_size=64,
        deepstack_visual_indexes=[1, 2],
    )


def _small_audio_cfg():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    cfg = Qwen3OmniMoeAudioEncoderConfig(
        d_model=64,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        encoder_layers=4,
        output_dim=64,
    )
    cfg.n_window, cfg.n_window_infer = 50, 800
    return cfg


def test_vision_structural_parity_cpu():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder,
    )

    from mstar.model.qwen3_omni.components.vision_encoder import (
        NativeQwen3OmniVisionEncoder,
    )

    torch.manual_seed(0)
    cfg = _small_vision_cfg()
    hf = Qwen3OmniMoeVisionEncoder._from_config(cfg, attn_implementation="sdpa").to(DEVICE, DTYPE).eval()
    nat = NativeQwen3OmniVisionEncoder(cfg).to(DEVICE, DTYPE).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    assert not miss and not unexp, f"structural mismatch: {len(miss)} missing / {len(unexp)} unexpected"

    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    g = torch.tensor([[1, 8, 8]], device=DEVICE)
    pv = torch.randn(8 * 8, rows, device=DEVICE, dtype=DTYPE)
    hcap, ncap = {}, {}
    _hook_residuals(hf.blocks, hcap)
    _hook_residuals(nat.blocks, ncap)
    with torch.no_grad():
        o = hf(pv, grid_thw=g)
        emb_n, ds_n = nat(pv, grid_thw=g)

    # every block's residual stream
    for i in range(len(hf.blocks)):
        cos, rel = _cmp(ncap[i], hcap[i])
        assert cos > COS_MIN and rel < RELL2_MAX, f"vision block {i}: cos={cos:.6f} relL2={rel:.2e}"
    # merged pooler + every DeepStack level
    cos, rel = _cmp(emb_n, o.pooler_output)
    assert cos > COS_MIN and rel < RELL2_MAX, f"vision pooler: cos={cos:.6f} relL2={rel:.2e}"
    assert len(ds_n) == len(o.deepstack_features)
    for k, (dn, dh) in enumerate(zip(ds_n, o.deepstack_features, strict=False)):
        cos, rel = _cmp(dn, dh)
        assert cos > COS_MIN and rel < RELL2_MAX, f"vision deepstack[{k}]: cos={cos:.6f} relL2={rel:.2e}"


def test_audio_structural_parity_cpu():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from mstar.model.qwen3_omni.components.audio_encoder import (
        NativeQwen3OmniAudioEncoder,
    )

    torch.manual_seed(0)
    cfg = _small_audio_cfg()
    hf = Qwen3OmniMoeAudioEncoder._from_config(cfg, attn_implementation="sdpa").to(DEVICE, DTYPE).eval()
    nat = NativeQwen3OmniAudioEncoder(cfg).to(DEVICE, DTYPE).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    assert not miss and not unexp, f"structural mismatch: {len(miss)} missing / {len(unexp)} unexpected"

    lens = torch.tensor([800], device=DEVICE)
    feat = torch.randn(cfg.num_mel_bins, 800, device=DEVICE, dtype=DTYPE)
    hcap, ncap = {}, {}
    _hook_residuals(hf.layers, hcap)
    _hook_residuals(nat.layers, ncap)
    with torch.no_grad():
        ref = hf(feat, feature_lens=lens).last_hidden_state
        out = nat(feat, lens).last_hidden_state

    for i in range(len(hf.layers)):
        cos, rel = _cmp(ncap[i], hcap[i])
        assert cos > COS_MIN and rel < RELL2_MAX, f"audio layer {i}: cos={cos:.6f} relL2={rel:.2e}"
    cos, rel = _cmp(out, ref)
    assert cos > COS_MIN and rel < RELL2_MAX, f"audio final: cos={cos:.6f} relL2={rel:.2e}"


def test_native_audio_output_is_named_tuple():
    """The native audio encoder must return a stable typed output (not an ad-hoc
    per-call class) and accept HF's return_dict kwarg."""
    from mstar.model.qwen3_omni.components.audio_encoder import (
        AudioEncoderOutput,
        NativeQwen3OmniAudioEncoder,
    )

    torch.manual_seed(0)
    cfg = _small_audio_cfg()
    nat = NativeQwen3OmniAudioEncoder(cfg).to(DEVICE, DTYPE).eval()
    lens = torch.tensor([800], device=DEVICE)
    feat = torch.randn(cfg.num_mel_bins, 800, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out = nat(feat, lens, return_dict=True)
    assert isinstance(out, AudioEncoderOutput)
    assert out.last_hidden_state.dim() == 2
