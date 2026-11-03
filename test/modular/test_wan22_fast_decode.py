"""CPU checks for the wan22 vae_decoder's ``vae_decode_fast`` path (the shared
Wan VAE decoder in mstar/model/components/wan_vae.py) on a tiny randomly
initialized Wan2.2-family ``AutoencoderKLWan``: off is stock ``vae.decode``
exactly, the eager default-layout path is too, and channels-last or compiled
match it to float rounding; only listed sizes compile, and every other decode
is stock. No weights, no GPU."""

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


def _latents(frames: int = 6, seed: int = 1, h: int = 4, w: int = 6) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, Z_DIM, frames, h, w, generator=gen)


# _latents' 4x6 latent grid decodes to 96x64
LISTED = ("96x64",)


def _decoder(monkeypatch, vae, *, fast: bool, dtype=torch.float32, tiling: str = "untiled", sizes=LISTED):
    monkeypatch.setenv("WAN22_VAE_DECODE_FP32", "1" if dtype == torch.float32 else "0")
    cfg = Wan22Config(vae_z_dim=Z_DIM, vae_decode_tiling=tiling, vae_decode_fast=fast, vae_decode_fast_sizes=sizes)
    return Wan22VaeDecoderSubmodule(copy.deepcopy(vae), cfg)


def _forward(sub, latents: torch.Tensor) -> torch.Tensor:
    """The node's emitted ``[F, H, W, 3]`` frames, as the decode's ``[1, 3, F, H, W]``."""
    frames = sub.forward("video_gen", engine_inputs=None, latents=latents)["video_output"][0]
    assert frames.is_contiguous() and frames.shape[-1] == 3
    return frames.permute(3, 0, 1, 2)[None]


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
    assert sub._fast is None
    assert not wan_vae.is_channels_last(sub.vae)
    assert {type(m) for m in sub.vae.modules()} == {type(m) for m in vae.modules()}
    z = _latents()
    assert torch.equal(_forward(sub, z), _stock(vae, z, dtype))


def test_eager_default_layout_is_stock_vae_decode(monkeypatch, vae) -> None:
    """The fast path's wiring with an eager step on the default layout issues
    ``vae.decode``'s decoder calls, one latent each, and its frames bit for bit."""
    sub = _decoder(monkeypatch, vae, fast=False)
    sub._fast = wan_vae.FastDecoder(sub.vae, channels_last=False, compile=False)
    calls = _spy_eager_calls(sub._fast.step)
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
    assert wan_vae.is_channels_last(sub.vae) and sub._fast.step.compiled
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
    step = sub._fast.step
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
    sub._fast = wan_vae.FastDecoder(sub.vae, compile=False)
    z = _latents()
    _u8_close(_forward(sub, z), _stock(vae, z), 1)


def test_tiled_fallback_is_stock_tiled_decode(monkeypatch, vae) -> None:
    """A tiled decode with the fast path on is stock ``tiled_decode`` on the VAE
    as loaded, bit for bit, as it is with the fast path off."""
    z = _latents()
    ref = _stock(vae, z, tiled=True)
    assert torch.equal(_forward(_decoder(monkeypatch, vae, fast=False, tiling="tiled"), z), ref)
    sub = _decoder(monkeypatch, vae, fast=True, tiling="tiled")
    calls = _spy_eager_calls(sub._fast.step)
    assert torch.equal(_forward(sub, z), ref)
    assert calls == [] and not wan_vae.is_channels_last(sub.vae)


def _stub_compile(monkeypatch) -> list[tuple]:
    """Replace ``torch.compile`` with an eager stand-in that records each new
    input signature, which is what a ``dynamic=False`` graph recompiles on."""
    compiles = []

    def compile(fn, **kwargs):
        def run(x, **kw):
            key = (tuple(x.shape), x.stride())
            if key not in compiles:
                compiles.append(key)
            return fn(x, **kw)

        return run

    monkeypatch.setattr(torch, "compile", compile)
    return compiles


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_only_a_listed_size_compiles(monkeypatch, vae, dtype) -> None:
    """A listed size compiles once whatever the frame count; an unlisted size
    decodes eagerly on the VAE as loaded, the frames ``vae_decode_fast: false``
    gives, with no compile; back at the listed size nothing recompiles."""
    compiles = _stub_compile(monkeypatch)
    sub = _decoder(monkeypatch, vae, fast=True, dtype=dtype)
    off = _decoder(monkeypatch, vae, fast=False, dtype=dtype)
    listed = _forward(sub, _latents(4))
    _forward(sub, _latents(7))
    assert len(compiles) == 1 and wan_vae.is_channels_last(sub.vae)
    for h, w in ((2, 4), (6, 4)):  # 64x32 and 64x96, neither listed
        z = _latents(5, h=h, w=w)
        assert torch.equal(_forward(sub, z), _forward(off, z))
        assert len(compiles) == 1 and not wan_vae.is_channels_last(sub.vae)
    assert torch.equal(_forward(sub, _latents(4)), listed)
    assert len(compiles) == 1 and wan_vae.is_channels_last(sub.vae)


def test_switching_layouts_does_not_recompile_the_listed_size(monkeypatch, vae) -> None:
    """Real Dynamo guards (eager backend, so no Inductor build): after an
    unlisted size switches the VAE back to its loaded layout, the listed size
    reuses its graphs. Weights rebuilt as inference tensors would fail the
    parameter guards and recompile every graph."""
    import torch._dynamo.utils as dynamo_utils

    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, **{**kw, "backend": "eager"}))
    torch._dynamo.reset()
    sub = _decoder(monkeypatch, vae, fast=True)
    listed = _forward(sub, _latents(4))
    graphs = dynamo_utils.counters["stats"]["unique_graphs"]
    assert graphs > 0
    _forward(sub, _latents(4, h=2, w=4))
    assert not wan_vae.is_channels_last(sub.vae)
    assert torch.equal(_forward(sub, _latents(4)), listed)
    assert dynamo_utils.counters["stats"]["unique_graphs"] == graphs
    torch._dynamo.reset()


def test_the_fast_decode_compiles_in_deterministic_mode(monkeypatch, vae) -> None:
    """Inductor's deterministic mode where torch has it: a cold-cache
    benchmark would otherwise pick reduction configs per process."""
    import torch._inductor.config as inductor_config

    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: seen.append(kw) or fn)
    _decoder(monkeypatch, vae, fast=True)
    expected = {"comprehensive_padding": False} if hasattr(inductor_config, "comprehensive_padding") else {}
    if hasattr(inductor_config, "deterministic"):
        expected["deterministic"] = True
    assert [kw.get("options") for kw in seen] == [expected or None]


def test_default_layout_undoes_channels_last_exactly(vae) -> None:
    ref = copy.deepcopy(vae)
    sub = copy.deepcopy(vae)
    wan_vae.to_channels_last(sub)
    wan_vae.to_default_layout(sub)
    assert not wan_vae.is_channels_last(sub)
    assert [type(m) for m in sub.modules()] == [type(m) for m in ref.modules()]
    for (name, p), q in zip(sub.named_parameters(), ref.parameters(), strict=True):
        assert torch.equal(p, q) and p.stride() == q.stride(), name


def test_fast_decode_sizes_default_to_the_default_size() -> None:
    assert Wan22Config().fast_decode_sizes() == {(1280, 704)}
    assert Wan22Config(vae_decode_fast_sizes=["1280x704", "832x480"]).fast_decode_sizes() == {(1280, 704), (832, 480)}
    model = Wan22Model(model_path_hf="unused", skip_weight_loading=True, vae_decode_fast_sizes=["832x480"])
    assert model.config.fast_decode_sizes() == {(832, 480)}


@pytest.mark.parametrize("sizes", ["1280x704", ["1280"], ["1280x700"], ["0x704"], [1280]])
def test_fast_decode_sizes_must_be_aligned_wxh(sizes) -> None:
    with pytest.raises(ValueError, match="vae_decode_fast_sizes"):
        Wan22Config(vae_decode_fast_sizes=sizes)


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
