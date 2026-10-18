"""hf_attention without flash_attn. Checks which implementation gets picked
and that HF's packed path reaches varlen_attention with the right segment
boundaries. A per-segment SDPA reference stands in for the kernel here, the
kernel itself is tested in test/rocm/test_varlen_attention.py."""

import types

import pytest
import torch
import torch.nn.functional as F

from mstar.model.components import hf_attention


def _per_segment_sdpa(q, k, v, cu_q, cu_k, max_q, max_k, causal=False, softmax_scale=None):
    outs = []
    for i in range(cu_q.numel() - 1):
        qs = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)
        ks = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1)
        vs = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1)
        outs.append(F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal, scale=softmax_scale).transpose(0, 1))
    return torch.cat(outs)


@pytest.fixture
def no_flash_attn(monkeypatch):
    monkeypatch.setattr(hf_attention, "flash_attn_installed", lambda: False)


@pytest.mark.parametrize("flash_attn,varlen,expected", [
    (True, True, "flash_attention_2"),
    (True, False, "flash_attention_2"),
    (False, True, hf_attention.HF_VARLEN_ATTENTION),
    (False, False, "sdpa"),
])
def test_implementation_choice(monkeypatch, flash_attn, varlen, expected):
    monkeypatch.setattr(hf_attention, "flash_attn_installed", lambda: flash_attn)
    monkeypatch.setattr(hf_attention, "varlen_attention_available", lambda: varlen)
    assert hf_attention.hf_encoder_attn_implementation() == expected


@pytest.mark.parametrize("platform,aiter,expected", [
    ("rocm", True, True), ("rocm", False, False), ("cuda", True, False), ("cpu", True, False),
])
def test_varlen_kernel_only_on_rocm_with_aiter(monkeypatch, platform, aiter, expected):
    import importlib.util

    import mstar.platforms

    fake = types.SimpleNamespace(is_rocm=lambda: platform == "rocm")
    monkeypatch.setattr(mstar.platforms, "current_platform", fake)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: (object() if aiter else None) if name == "aiter" else real_find_spec(name, *a),
    )
    hf_attention.varlen_attention_available.cache_clear()
    try:
        assert hf_attention.varlen_attention_available() is expected
    finally:
        hf_attention.varlen_attention_available.cache_clear()


def _audio_config():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )

    return Qwen3OmniMoeAudioEncoderConfig(
        d_model=64, encoder_attention_heads=4, encoder_layers=2, encoder_ffn_dim=128,
        output_dim=32, num_mel_bins=32, downsample_hidden_size=16, n_window=50, n_window_infer=200,
    )


def _vision_config():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig,
    )

    return Qwen3OmniMoeVisionEncoderConfig(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=128, patch_size=4,
        out_hidden_size=32, num_position_embeddings=64, deepstack_visual_indexes=[1],
    )


def _same_weights(a, b):
    b.load_state_dict(a.state_dict())
    return a.eval(), b.eval()


def test_audio_tower_on_varlen_matches_sdpa(monkeypatch, no_flash_attn):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    calls = []

    def reference(*args, **kwargs):
        calls.append(args[3].tolist())
        return _per_segment_sdpa(*args, **kwargs)

    monkeypatch.setattr(hf_attention, "varlen_attention_available", lambda: True)
    monkeypatch.setattr(hf_attention, "varlen_attention", reference)
    torch.manual_seed(0)
    # same config object for both, building the varlen model must not change it
    config = _audio_config()
    varlen = hf_attention.build_hf_encoder(Qwen3OmniMoeAudioEncoder, config)
    sdpa = Qwen3OmniMoeAudioEncoder._from_config(config, attn_implementation="sdpa")
    varlen, sdpa = _same_weights(varlen, sdpa)
    assert all(
        m.config._attn_implementation == hf_attention.HF_VARLEN_ATTENTION
        for m in varlen.modules() if hasattr(m, "config")
    )
    # two clips, the second spanning several attention windows
    feature_lens = torch.tensor([150, 640])
    features = torch.randn(32, int(feature_lens.sum()))
    with torch.no_grad():
        got = varlen(features, feature_lens=feature_lens).last_hidden_state
        want = sdpa(features, feature_lens=feature_lens).last_hidden_state
    assert calls and len(calls[0]) > 3  # one packed call per layer, several segments each
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_vision_tower_on_varlen_matches_sdpa(monkeypatch, no_flash_attn):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    monkeypatch.setattr(hf_attention, "varlen_attention_available", lambda: True)
    monkeypatch.setattr(hf_attention, "varlen_attention", _per_segment_sdpa)
    torch.manual_seed(0)
    varlen = hf_attention.build_hf_encoder(Qwen3OmniMoeVisionEncoder, _vision_config())
    sdpa = Qwen3OmniMoeVisionEncoder._from_config(_vision_config(), attn_implementation="sdpa")
    varlen, sdpa = _same_weights(varlen, sdpa)
    # an image and a 4-frame video: five segments
    grid_thw = torch.tensor([[1, 4, 6], [4, 4, 4]])
    patches = torch.randn(int(grid_thw.prod(dim=1).sum()), 3 * 2 * 4 * 4)
    with torch.no_grad():
        got = varlen(patches, grid_thw=grid_thw)
        want = sdpa(patches, grid_thw=grid_thw)
    got = got.pooler_output if hasattr(got, "pooler_output") else got[0]
    want = want.pooler_output if hasattr(want, "pooler_output") else want[0]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_hf_forward_refuses_unpacked_input():
    q = torch.randn(2, 4, 8, 16)
    module = types.SimpleNamespace(is_causal=False, training=False)
    with pytest.raises(ValueError, match="packed"):
        hf_attention._hf_varlen_forward(module, q, q, q, attention_mask=None)


def test_fake_kernel_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(10, 4, 72)
        v = torch.empty(12, 2, 72)
        cu = torch.empty(3, dtype=torch.int32)
        out = torch.ops.mstar.varlen_attention(q, v, v, cu, cu, 6, 7, False, None)
    assert out.shape == (10, 4, 72)
