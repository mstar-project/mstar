"""window_attention: SDPA off ROCm, AITER's FMHA on ROCm where it applies."""

import pytest
import torch
import torch.nn.functional as F

from mstar.model.components import window_attention as wa


def _bshd(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


def _reference(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2)


def test_default_is_sdpa_on_bshd_views():
    qkv = _bshd(2, 9, 3 * 4 * 8)
    q, k, v = (t.view(2, 9, 4, 8) for t in qkv.split(32, dim=-1))
    impl = wa._window_attention._default()
    torch.testing.assert_close(impl(q, k, v), _reference(q, k, v), rtol=0, atol=0)


def test_rocm_without_aiter_runs_sdpa(monkeypatch):
    monkeypatch.setattr(wa.importlib.util, "find_spec", lambda name: None)
    assert wa._rocm() is wa._sdpa


def test_rocm_routes_supported_inputs_to_aiter(monkeypatch):
    monkeypatch.setattr(wa.importlib.util, "find_spec", lambda name: object())
    calls = []
    monkeypatch.setattr(wa, "_aiter_window_attention", lambda q, k, v: calls.append(q.dtype) or _reference(q, k, v))
    attend = wa._rocm()
    half = _bshd(1, 5, 2, 8, dtype=torch.bfloat16)
    attend(half, half, half)
    assert calls == [torch.bfloat16]
    # float32 and too wide a head stay on SDPA
    single = _bshd(1, 5, 2, 8)
    torch.testing.assert_close(attend(single, single, single), _reference(single, single, single))
    wide = _bshd(1, 3, 1, 512, dtype=torch.bfloat16)
    attend(wide, wide, wide)
    assert calls == [torch.bfloat16]


def test_whisper_encoder_attention_matches_sdpa():
    from mstar.model.whisper.components.encoder import WhisperEncoderAttention

    torch.manual_seed(0)
    attn = WhisperEncoderAttention(d_model=32, num_heads=4).eval()
    with torch.no_grad():
        for p in attn.parameters():  # the fused projection is allocated uninitialized
            p.normal_(0, 0.2)
    x = torch.randn(2, 11, 32)
    with torch.no_grad():
        q, k, v = attn.qkv_proj(x).split(32, dim=-1)
        q, k, v = (t.view(2, 11, 4, 8).transpose(1, 2) for t in (q, k, v))
        want = attn.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 11, 32))
        torch.testing.assert_close(attn(x), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_matches_float32_reference():
    q, k, v = (_bshd(2, 1500, 20, 64, seed=i).cuda() for i in range(3))
    want = _reference(q, k, v)
    got = wa.window_attention_kernel()(q.bfloat16(), k.bfloat16(), v.bfloat16())
    assert got.shape == want.shape
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_whisper_encoder_layer_traces_as_one_graph():
    """The layer binds its kernels when built, so dynamo never meets the
    dispatch's first pick (a lock it can't trace, a graph break mid-encoder)."""
    from torch._dynamo.testing import CompileCounter

    from mstar.model.whisper.components.encoder import WhisperEncoderLayer
    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.platforms.dispatch import PlatformDispatch

    PlatformDispatch.reset_all()
    layer = WhisperEncoderLayer(WhisperModelConfig(d_model=32, encoder_attention_heads=4, encoder_ffn_dim=64)).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.2)
        counter = CompileCounter()
        torch._dynamo.reset()
        torch.compile(layer, backend=counter, fullgraph=True)(torch.randn(2, 7, 32))
    assert counter.frame_count == 1
