"""CPU checks for ``vae_decode_fast_math`` (``wan_vae.rewrite_decoder``) on tiny randomly initialized
``AutoencoderKLWan``s of both families (Wan2.1: 8x spatial, no residual up blocks; Wan2.2: residual up blocks, patch
2): the rewritten decoder is the stock decoder's math (fp64, to float rounding), its clip start leaves the zero frames
"Rep" stands for (bit for bit, eager), a clip decoded in pieces is the whole decode bit for bit, every call can run
compiled, switching back gives stock ``vae.decode`` exactly, and the knob reaches the node. No weights, no GPU."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock, WanResample  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.wan22 import submodules as wan22_submodules  # noqa: E402
from mstar.model.wan22.config import WAN22_VARIANT_T2V_A14B, Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule  # noqa: E402
from mstar.model.wan22.wan22_model import Wan22Model  # noqa: E402


def _init(vae: AutoencoderKLWan) -> AutoencoderKLWan:
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return vae.eval()


def _wan21() -> AutoencoderKLWan:
    return _init(AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1))


def _wan22() -> AutoencoderKLWan:
    return _init(AutoencoderKLWan(
        base_dim=8, decoder_base_dim=16, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1, is_residual=True,
        in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
        latents_mean=[0.1 * i for i in range(8)], latents_std=[0.5 + 0.1 * i for i in range(8)],
    ))


FAMILIES = {"wan21": _wan21, "wan22": _wan22}


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def vae(request):
    return FAMILIES[request.param]()


def _latents(vae, frames: int = 6, seed: int = 1, h: int = 4, w: int = 6, dtype=torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, vae.config.z_dim, frames, h, w, generator=gen).to(dtype)


def _decode(fd: wan_vae.FastDecoder, z: torch.Tensor, pieces: int = 0) -> torch.Tensor:
    with torch.inference_mode():
        if not pieces:
            return fd.decode(z)
        x, cache = fd.post_quant(z), wan_vae.new_feat_cache(fd.vae)
        return torch.cat([fd.decode_stream(x[:, :, s:s + pieces], cache, clip_start=s == 0, post_quantized=True)
                          for s in range(0, x.shape[2], pieces)], dim=2)


def _stock(vae, z: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return vae.decode(z, return_dict=False)[0]


@pytest.mark.parametrize("size", [(1, 1), (3, 5), (8, 7)])
def test_phase_upsample_is_nearest_x2_then_the_conv(size) -> None:
    gen = torch.Generator().manual_seed(2)
    conv = torch.nn.Conv2d(6, 3, 3, padding=1).double()
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=gen))
    x = torch.randn(2, 6, *size, generator=gen, dtype=torch.float64)
    ref = conv(F.interpolate(x, scale_factor=2.0, mode="nearest-exact"))
    out = F.conv_transpose2d(x, wan_vae.phase_upsample_weight(conv), conv.bias, stride=2, padding=1)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)


def test_rewritten_decoder_is_the_stock_math(vae) -> None:
    """fp64, eager, default layout: every rewrite (zero frames for "Rep", the transposed-conv upsample, the GEMM
    attention) only moves the frames by float rounding."""
    stock = copy.deepcopy(vae).double()
    fd = wan_vae.FastDecoder(copy.deepcopy(vae).double(), channels_last=False, compile=False, rewrite=True)
    assert wan_vae.is_rewritten(fd.vae)
    kinds = {type(m) for m in fd.vae.decoder.modules()}
    assert wan_vae.RewrittenResample in kinds and wan_vae.RewrittenAttentionBlock in kinds
    assert WanResample not in kinds and WanAttentionBlock not in kinds
    z = _latents(vae)
    # 1e-6, not fp64's 1e-12: diffusers' WanUpsample upsamples in fp32 whatever the input dtype, so the stock decode
    # rounds there. A rewrite that changed the math would move these frames by ~1e-1.
    torch.testing.assert_close(_decode(fd, z), _stock(stock, z), rtol=0, atol=1e-6)


class _ZeroCacheOnly(wan_vae.RewrittenResample):
    """The rewritten temporal upsample, with the stock spatial one (``self.resample``)."""

    def _upsample_space(self, x):
        return self.resample(x)


def test_clip_start_leaves_the_zero_frames_rep_stands_for(vae) -> None:
    """With only the temporal upsamplers' cache rewritten, an eager fp32 decode is stock ``vae.decode`` bit for bit:
    the zero frames are exactly the padding "Rep" asks for."""
    zeroed = copy.deepcopy(vae)
    for m in zeroed.decoder.modules():
        if type(m) is WanResample and m.mode == "upsample3d":
            m.__class__ = _ZeroCacheOnly
    fd = wan_vae.FastDecoder(zeroed, channels_last=False, compile=False)
    z = _latents(vae, dtype=torch.float32)
    cache = wan_vae.new_feat_cache(zeroed)
    with torch.inference_mode():
        out = torch.cat([fd.decode_stream(z[:, :, :1], cache, clip_start=True),
                         fd.decode_stream(z[:, :, 1:], cache, clip_start=False)], dim=2)
    assert not any(isinstance(t, str) for t in cache)
    assert torch.equal(out, _stock(vae, z))


@pytest.mark.parametrize("pieces", [1, 2, 4])
def test_pieces_are_the_whole_decode(vae, pieces) -> None:
    fd = wan_vae.FastDecoder(copy.deepcopy(vae), compile=False, rewrite=True)
    z = _latents(vae, frames=7, dtype=torch.float32)
    assert torch.equal(_decode(fd, z, pieces=pieces), _decode(fd, z))


def test_every_call_runs_compiled(vae) -> None:
    """The rewritten decoder leaves no "Rep" in the cache, so neither clip-start call falls back to eager; the
    compiled decode matches the eager rewritten one to float rounding."""
    fd = wan_vae.FastDecoder(copy.deepcopy(vae), compile=True, rewrite=True)
    eager_calls = []
    decoder = fd.step._decoder
    fd.step._decoder = lambda x, **kw: eager_calls.append(x.shape[2]) or decoder(x, **kw)
    z = _latents(vae, frames=4, dtype=torch.float32)
    out = _decode(fd, z)
    assert eager_calls == []
    ref = _decode(wan_vae.FastDecoder(copy.deepcopy(vae), compile=False, rewrite=True), z)
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-4)


def test_stock_layout_is_stock_vae_decode(vae) -> None:
    fd = wan_vae.FastDecoder(copy.deepcopy(vae), compile=False, rewrite=True)
    z = _latents(vae, dtype=torch.float32)
    _decode(fd, z)
    fd.use_stock_layout()
    assert not wan_vae.is_rewritten(fd.vae) and not wan_vae.is_channels_last(fd.vae)
    assert torch.equal(_stock(fd.vae, z), _stock(vae, z))
    _decode(fd, z)  # and back
    assert wan_vae.is_rewritten(fd.vae) and wan_vae.is_channels_last(fd.vae)


def test_the_knob(monkeypatch) -> None:
    assert Wan22Config().vae_decode_fast_math is False
    assert Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, vae_decode_fast_math=True).vae_decode_fast_math is True
    with pytest.raises(ValueError, match="true or false"):
        Wan22Config(vae_decode_fast_math="true")
    with pytest.raises(ValueError, match="vae_decode_fast too"):
        Wan22Config(vae_decode_fast_math=True, vae_decode_fast=False)
    # the model passes it on: an unnamed model kwarg would be dropped into **kwargs
    assert Wan22Model(model_path_hf="test/wan22", vae_decode_fast_math=True).config.vae_decode_fast_math is True
    monkeypatch.setattr(wan22_submodules, "wan_vae_decode_dtype", lambda: torch.float32)
    on = Wan22VaeDecoderSubmodule(_wan22(), Wan22Config(vae_decode_fast_math=True))
    off = Wan22VaeDecoderSubmodule(_wan22(), Wan22Config())
    assert on._fast.rewrite and on._fast.step.compile_clip_start and wan_vae.is_rewritten(on.vae)
    assert not off._fast.rewrite and not wan_vae.is_rewritten(off.vae)
