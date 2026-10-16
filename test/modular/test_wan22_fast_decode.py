"""CPU checks for the wan22 vae_decoder's ``vae_decode_fast`` path (the shared
Wan VAE decoder in mstar/model/components/wan_vae.py) on a tiny randomly
initialized Wan2.2-family ``AutoencoderKLWan``: off is stock ``vae.decode``
exactly, the eager default-layout path is too, and channels-last or compiled
match it to float rounding. No weights, no GPU."""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.wan22.config import Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule  # noqa: E402
from mstar.model.wan22.wan22_model import Wan22Model  # noqa: E402

Z_DIM = 8


def _tiny_vae() -> AutoencoderKLWan:
    vae = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=16, z_dim=Z_DIM, dim_mult=[1, 2, 2, 2], num_res_blocks=1, is_residual=True,
        in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
        latents_mean=[0.1 * i for i in range(Z_DIM)], latents_std=[0.5 + 0.1 * i for i in range(Z_DIM)],
    ).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return vae


@pytest.fixture(scope="module")
def vae():
    return _tiny_vae()


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    # The tiling gate reads free VRAM; on CPU there is none to read.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2**40, 2**40))


def _latents(frames: int = 6, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, Z_DIM, frames, 4, 6, generator=gen)


def _decoder(monkeypatch, vae, *, fast: bool, dtype=torch.float32, tiling: str = "untiled"):
    monkeypatch.setenv("WAN22_VAE_DECODE_FP32", "1" if dtype == torch.float32 else "0")
    cfg = Wan22Config(vae_z_dim=Z_DIM, vae_decode_tiling=tiling, vae_decode_fast=fast)
    return Wan22VaeDecoderSubmodule(copy.deepcopy(vae), cfg)


def _forward(sub, latents: torch.Tensor) -> torch.Tensor:
    return sub.forward("video_gen", engine_inputs=None, latents=latents)["video_output"][0]


def _stock(vae, latents: torch.Tensor, dtype=torch.float32, tiled: bool = False) -> torch.Tensor:
    """The node's decode before ``vae_decode_fast``: de-normalize, stock
    ``vae.decode`` (or ``tiled_decode``), quantize."""
    vae = copy.deepcopy(vae).to(dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, Z_DIM, 1, 1, 1).to(dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, Z_DIM, 1, 1, 1).to(dtype)
    z = latents.to(dtype) / inv_std + mean
    with torch.inference_mode():
        video = (vae.tiled_decode if tiled else vae.decode)(z, return_dict=False)[0]
    return (video / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)


def _u8_close(a: torch.Tensor, b: torch.Tensor, tol: int) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.uint8
    assert (a.int() - b.int()).abs().max() <= tol


def _spy_eager_calls(step) -> list[int]:
    """The latents per call of ``step``'s eager decoder calls."""
    calls = []
    decoder = step._decoder
    step._decoder = lambda x, **kw: calls.append(x.shape[2]) or decoder(x, **kw)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_off_switch_is_stock_vae_decode(monkeypatch, vae, dtype) -> None:
    sub = _decoder(monkeypatch, vae, fast=False, dtype=dtype)
    assert sub._fast_step is None
    assert not wan_vae.is_channels_last(sub.vae)
    assert {type(m) for m in sub.vae.modules()} == {type(m) for m in vae.modules()}
    z = _latents()
    assert torch.equal(_forward(sub, z), _stock(vae, z, dtype))


def test_eager_default_layout_is_stock_vae_decode(monkeypatch, vae) -> None:
    """The fast path's wiring with an eager step on the default layout issues
    ``vae.decode``'s decoder calls, one latent each, and its frames bit for bit."""
    sub = _decoder(monkeypatch, vae, fast=False)
    sub._fast_step = wan_vae.DecoderStep(sub.vae, compile=False)
    calls = _spy_eager_calls(sub._fast_step)
    z = _latents()
    assert torch.equal(_forward(sub, z), _stock(vae, z))
    assert calls == [1] * z.shape[2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_default_fast_path_is_stock_to_rounding(monkeypatch, vae, dtype) -> None:
    """On by default: channels-last and compiled. In fp32 the frames match
    stock ``vae.decode`` to float rounding; in bf16 they are no further from
    an fp32 decode than stock bf16's are. Repeat calls (past the compiled
    step's warm-up) return the same frames."""
    sub = _decoder(monkeypatch, vae, fast=True, dtype=dtype)
    assert Wan22Config().vae_decode_fast
    assert wan_vae.is_channels_last(sub.vae) and sub._fast_step.compiled
    z = _latents()
    first = _forward(sub, z)
    if dtype == torch.float32:
        _u8_close(first, _stock(vae, z), 1)
    else:
        ref = _stock(vae, z).float()
        err, stock_err = ((x.float() - ref).abs() for x in (first, _stock(vae, z, dtype)))
        assert err.mean() <= 1.1 * stock_err.mean() and err.max() <= stock_err.max() + 1
    assert torch.equal(_forward(sub, z), first)


def test_compiled_calls_see_one_layout_whatever_the_clip_length(monkeypatch, vae) -> None:
    """The compiled step's inputs have the same shape and strides for every
    clip length at one resolution, so a new frame count does not recompile."""
    sub = _decoder(monkeypatch, vae, fast=True)
    step = sub._fast_step
    seen = []

    def spy(x, **kw):
        seen.append((tuple(x.shape), x.stride()))
        return step._decoder(x, **kw)

    step._fn = spy
    for frames in (4, 7):
        _forward(sub, _latents(frames))
    assert len(seen) == (4 - 2) + (7 - 2)
    assert len(set(seen)) == 1


def test_channels_last_eager_is_stock_to_rounding(monkeypatch, vae) -> None:
    sub = _decoder(monkeypatch, vae, fast=False)
    wan_vae.to_channels_last(sub.vae)
    sub._fast_step = wan_vae.DecoderStep(sub.vae, compile=False)
    z = _latents()
    _u8_close(_forward(sub, z), _stock(vae, z), 1)


def test_tiled_fallback_keeps_tiled_decode(monkeypatch, vae) -> None:
    """A tiled decode stays ``tiled_decode`` with the fast path on (on the
    channels-last weights, so to float rounding); off, it is exact."""
    z = _latents()
    ref = _stock(vae, z, tiled=True)
    assert torch.equal(_forward(_decoder(monkeypatch, vae, fast=False, tiling="tiled"), z), ref)
    sub = _decoder(monkeypatch, vae, fast=True, tiling="tiled")
    calls = _spy_eager_calls(sub._fast_step)
    _u8_close(_forward(sub, z), ref, 1)
    assert calls == []


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_vae_decode_fast_must_be_a_bool(value) -> None:
    with pytest.raises(ValueError, match="vae_decode_fast"):
        Wan22Config(vae_decode_fast=value)
    with pytest.raises(ValueError, match="vae_decode_fast"):
        Wan22Model(model_path_hf="unused", skip_weight_loading=True, vae_decode_fast=value)


def test_vae_decode_fast_reaches_the_config_from_model_kwargs() -> None:
    for kwargs, expected in (({}, True), ({"vae_decode_fast": False}, False)):
        model = Wan22Model(model_path_hf="unused", skip_weight_loading=True, **kwargs)
        assert model.config.vae_decode_fast is expected
