"""CPU checks for the shared Wan VAE decode helpers (mstar/model/components/
wan_vae.py) on tiny randomly initialized ``AutoencoderKLWan``s of both decoder
families: the Wan 2.1 one (plain up blocks) and the Wan2.2 one (residual up
blocks, 2x2 patch). No weights, no GPU."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402

KINDS = ("wan21", "wan22")


def tiny_wan_vae(kind: str, seed: int = 0) -> AutoencoderKLWan:
    if kind == "wan21":
        cfg = dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
    else:
        cfg = dict(
            base_dim=8, decoder_base_dim=16, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
            is_residual=True, in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
        )
    z = cfg["z_dim"]
    vae = AutoencoderKLWan(**cfg, latents_mean=[0.0] * z, latents_std=[1.0] * z).eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return vae


def _latents(vae, frames: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, vae.config.z_dim, frames, 4, 6, generator=gen)


@pytest.mark.parametrize("kind", KINDS)
def test_decode_causal_from_an_empty_cache_is_vae_decode(kind) -> None:
    vae = tiny_wan_vae(kind)
    z = _latents(vae, 5)
    step = wan_vae.DecoderStep(vae, compile=False)
    with torch.no_grad():
        ref = vae.decode(z).sample
        out = wan_vae.decode_causal(vae, z, wan_vae.new_feat_cache(vae), step, clip_start=True)
    assert out.shape == ref.shape == (1, 3, 17, *ref.shape[3:])
    assert torch.equal(out, ref)


@pytest.mark.parametrize("kind", KINDS)
def test_warm_cache_stream_is_the_whole_stream_decode(kind) -> None:
    """Decoding a stream piece by piece with one cache carried across the
    pieces reproduces one decode of the whole stream, bit for bit."""
    vae = tiny_wan_vae(kind)
    z = _latents(vae, 9)
    step = wan_vae.DecoderStep(vae, compile=False)
    cache = wan_vae.new_feat_cache(vae)
    with torch.no_grad():
        ref = vae.decode(z).sample
        parts = [wan_vae.decode_causal(vae, z[:, :, :4], cache, step, clip_start=True)]
        parts += [wan_vae.decode_causal(vae, z[:, :, a:b], cache, step, clip_start=False) for a, b in ((4, 7), (7, 9))]
    assert [p.shape[2] for p in parts] == [13, 12, 8]
    assert torch.equal(torch.cat(parts, dim=2), ref)
    assert wan_vae.feat_cache_bytes(cache) > 0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("per_call, channels_last", [(1, False), (0, False), (0, True)])
def test_clip_start_alone_and_a_ragged_tail(kind, per_call, channels_last) -> None:
    """A stream whose first piece is the clip-start latent alone and whose
    length does not divide into 3-latent chunks (pieces of 1, 3, 3, 2, 1
    latents): bit for bit with one latent per call in the default layout, to
    float rounding with several latents per call or channels-last."""
    vae = tiny_wan_vae(kind)
    z = _latents(vae, 10)
    with torch.no_grad():
        ref = vae.decode(z).sample
    if channels_last:
        wan_vae.to_channels_last(vae)
    step = wan_vae.DecoderStep(vae, compile=False)
    cache = wan_vae.new_feat_cache(vae)
    bounds = (0, 1, 4, 7, 9, 10)
    with torch.no_grad():
        parts = [
            wan_vae.decode_causal(vae, z[:, :, a:b], cache, step, clip_start=a == 0, latents_per_call=per_call)
            for a, b in zip(bounds, bounds[1:], strict=False)
        ]
    assert [p.shape[2] for p in parts] == [1, 12, 12, 8, 4]
    out = torch.cat(parts, dim=2)
    if per_call == 1 and not channels_last:
        assert torch.equal(out, ref)
    else:
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("per_call, calls_expected", [(0, [1, 3, 5]), (2, [1, 2, 1, 2, 2, 1]), (3, [1, 3, 3, 2])])
def test_several_latents_per_call_is_the_same_decode(kind, per_call, calls_expected) -> None:
    vae = tiny_wan_vae(kind)
    z = _latents(vae, 9)
    step = wan_vae.DecoderStep(vae, compile=False)
    calls = []
    decoder = step._decoder
    step._decoder = lambda x, **kw: calls.append(x.shape[2]) or decoder(x, **kw)
    cache = wan_vae.new_feat_cache(vae)
    with torch.no_grad():
        ref = vae.decode(z).sample
        parts = [wan_vae.decode_causal(vae, z[:, :, :4], cache, step, clip_start=True, latents_per_call=per_call)]
        parts.append(wan_vae.decode_causal(vae, z[:, :, 4:], cache, step, clip_start=False, latents_per_call=per_call))
    assert calls == calls_expected
    torch.testing.assert_close(torch.cat(parts, dim=2), ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_channels_last_stream_feeds_every_conv_channels_last(kind, monkeypatch) -> None:
    """After ``to_channels_last`` (idempotent) every conv kernel of a warm
    stream sees a channels-last input, cached context included, and the
    frames match the default layout's to float rounding."""
    import torch.nn.functional as F

    vae = tiny_wan_vae(kind)
    z = _latents(vae, 7)
    with torch.no_grad():
        ref = vae.decode(z).sample
    wan_vae.to_channels_last(vae)
    wan_vae.to_channels_last(vae)
    assert wan_vae.is_channels_last(vae)
    assert all(
        type(m) is not wan_vae.WanCausalConv3d and m.weight.is_contiguous(memory_format=torch.channels_last_3d)
        for m in vae.decoder.modules() if isinstance(m, torch.nn.Conv3d)
    )
    seen = []
    for name, fmt in (("conv3d", torch.channels_last_3d), ("conv2d", torch.channels_last)):
        def conv(x, *args, _orig=getattr(F, name), _fmt=fmt, **kw):
            seen.append(x.is_contiguous(memory_format=_fmt))
            return _orig(x, *args, **kw)
        monkeypatch.setattr(F, name, conv)
    step = wan_vae.DecoderStep(vae, compile=False)
    cache = wan_vae.new_feat_cache(vae)
    with torch.no_grad():
        parts = [wan_vae.decode_causal(vae, z[:, :, :4], cache, step, clip_start=True)]
        parts.append(wan_vae.decode_causal(vae, z[:, :, 4:], cache, step, clip_start=False))
    assert len(seen) > 30 and all(seen)
    torch.testing.assert_close(torch.cat(parts, dim=2), ref, rtol=0, atol=1e-5)


def _storage_bytes(cache: list) -> int:
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in cache if torch.is_tensor(t)}
    return sum(storages.values())


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_channels_last_stream_cache_owns_its_storage(kind) -> None:
    """Compiled on channels-last activations, Inductor drops diffusers' cache
    clone and returns each slot as a view of a whole conv input; the stream
    must still hold only ``feat_cache_bytes``, and the same frames. Eager and
    compiled writes leave the same strides, so a stream's compiled calls are
    not recompiled for the layout its eager first calls left."""
    vae = tiny_wan_vae(kind)
    z = _latents(vae, 8)
    wan_vae.to_channels_last(vae)
    outs, strides = {}, {}
    for compile in (False, True):
        step, cache = wan_vae.DecoderStep(vae, compile=compile), wan_vae.new_feat_cache(vae)
        with torch.no_grad():
            outs[compile] = [wan_vae.decode_causal(vae, z[:, :, a:b], cache, step, clip_start=a == 0)
                             for a, b in ((0, 2), (2, 5), (5, 8))]
        slots = [t for t in cache if torch.is_tensor(t)]
        assert len(slots) > 10
        assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in slots)
        assert _storage_bytes(cache) == wan_vae.feat_cache_bytes(cache)
        strides[compile] = [t.stride() for t in slots]
    assert strides[True] == strides[False]
    torch.testing.assert_close(torch.cat(outs[True], 2), torch.cat(outs[False], 2), rtol=0, atol=1e-5)


def test_compact_feat_cache_replaces_only_views() -> None:
    owned = torch.randn(1, 4, 2, 3, 5).contiguous(memory_format=torch.channels_last_3d)
    big = torch.randn(1, 4, 6, 3, 5).contiguous(memory_format=torch.channels_last_3d)
    first = torch.randn(1, 4, 6, 3, 5)
    cache = [None, "Rep", owned, big[:, :, -2:], first[:, :, -2:]]
    before = [t.clone() if torch.is_tensor(t) else t for t in cache]
    wan_vae.compact_feat_cache(cache)
    assert cache[:3] == [None, "Rep", owned] and cache[2] is owned
    assert all(torch.equal(cache[i], before[i]) for i in (3, 4))
    assert cache[3].is_contiguous(memory_format=torch.channels_last_3d) and cache[4].is_contiguous()
    assert cache[3].stride() == owned.stride()
    assert _storage_bytes(cache) == wan_vae.feat_cache_bytes(cache)


@pytest.mark.parametrize("before", [False, True])
def test_cudnn_benchmark_is_scoped(before, torch_flag) -> None:
    torch_flag(torch.backends.cudnn, "benchmark", before)
    with wan_vae.cudnn_benchmark(True):
        assert torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.benchmark is before
    with pytest.raises(RuntimeError), wan_vae.cudnn_benchmark(True):
        raise RuntimeError
    assert torch.backends.cudnn.benchmark is before
    with wan_vae.cudnn_benchmark(False):
        assert torch.backends.cudnn.benchmark is before
    assert torch.backends.cudnn.benchmark is before


def test_compiled_step_runs_steady_calls_compiled_after_a_warm_up_on_a_cache_copy() -> None:
    """A compiled step runs a stream's start eagerly (the clip-start call, and
    calls that still find the "Rep" marker in the cache). A steady-state bf16
    call with a new signature first runs once on a scratch copy of the cache,
    so the served cache advances exactly once per call."""
    vae = tiny_wan_vae("wan21")
    step = wan_vae.DecoderStep(vae, compile=False)
    eager, compiled = [], []

    def fake(log):
        def fn(x, feat_cache, feat_idx, first_chunk=False):
            log.append(feat_cache)
            feat_cache[0] = x.clone() if feat_cache[0] is None else feat_cache[0] + 1
            return x
        return fn

    step.compiled, step._decoder, step._fn = True, fake(eager), fake(compiled)
    cache = [None, "Rep"]
    x = torch.zeros(1, 2, 1, 2, 2, dtype=torch.bfloat16)
    step(x, cache, first_chunk=True)
    step(x, cache)
    assert len(eager) == 2 and not compiled
    cache[1] = torch.zeros(1)
    step(x, cache)  # a new signature: warm-up on a copy, then the served call
    step(x, cache)  # same signature: no warm-up
    assert len(compiled) == 3 and compiled[0] is not cache and compiled[1] is cache and compiled[2] is cache
    assert torch.equal(cache[0], torch.full_like(x, 3))
    step(x.float(), cache)  # fp32 never warms
    assert len(compiled) == 4
