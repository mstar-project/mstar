"""Every compiled Wan VAE decode turns Inductor's comprehensive padding off.

On torch 2.13 the padded strides it picks for the Wan 2.2 decoder's
``up_blocks[1]`` shortcut clone make a compiled decode read wrong values past
the end of the buffer. torch 2.12.1 graph-breaks before that graph forms, so
nothing fails today; these tests keep the option on every compile site before
the torch pin moves.
"""
import pytest
import torch

pytest.importorskip("diffusers")

import torch._inductor.config as inductor_config  # noqa: E402
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config  # noqa: E402
from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderARSubmodule, Cosmos3VAEDecoderSubmodule  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(inductor_config, "comprehensive_padding"), reason="this torch has no comprehensive_padding"
)


def _tiny_vae():
    return AutoencoderKLWan(
        base_dim=8, decoder_base_dim=16, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1, is_residual=True,
        in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
        latents_mean=[0.0] * 8, latents_std=[1.0] * 8,
    ).eval()


@pytest.fixture(scope="module")
def vae():
    return _tiny_vae()


@pytest.fixture
def compiles(monkeypatch):
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: seen.append(kw) or fn)
    return seen


def _padding_off(kw: dict) -> bool:
    return (kw.get("options") or {}).get("comprehensive_padding") is False


def test_decoder_step_compiles_without_comprehensive_padding(vae, compiles):
    wan_vae.DecoderStep(vae, compile=True)
    assert len(compiles) == 1 and _padding_off(compiles[0])


def test_deterministic_decoder_step_keeps_both_options(vae, compiles):
    wan_vae.DecoderStep(vae, compile=True, deterministic=True)
    assert len(compiles) == 1 and _padding_off(compiles[0])
    if hasattr(inductor_config, "deterministic"):
        assert compiles[0]["options"]["deterministic"] is True


def test_eager_decoder_step_compiles_nothing(vae, compiles):
    wan_vae.DecoderStep(vae, compile=False)
    assert compiles == []


@pytest.mark.parametrize("warm", [False, True])
def test_cosmos3_vae_decode_nodes_compile_without_comprehensive_padding(vae, compiles, monkeypatch, warm):
    monkeypatch.delenv("COSMOS3_COMPILE_VAE", raising=False)
    cfg = Cosmos3Config()
    cfg.vae_decode_fast = False  # the stock compiled decode; the fast path is tested below
    cfg.windowed_decode_warm_cache = warm
    cfg.vae_decode_channels_last = False
    Cosmos3VAEDecoderSubmodule(vae, cfg)
    Cosmos3VAEDecoderARSubmodule(vae, cfg)
    # the stateless node, then the AR node's stateless decode and, when warm, its DecoderStep
    assert len(compiles) == (3 if warm else 2)
    assert all(_padding_off(kw) for kw in compiles)


def test_cosmos3_fast_decode_compiles_only_its_decoder_step(compiles, monkeypatch):
    monkeypatch.delenv("COSMOS3_COMPILE_VAE", raising=False)
    # Its own VAE: the fast path casts and converts the node's instance in place.
    sub = Cosmos3VAEDecoderSubmodule(_tiny_vae(), Cosmos3Config(vae_decode_fast=True))
    assert sub._fast_step is not None
    assert len(compiles) == 1 and _padding_off(compiles[0])
