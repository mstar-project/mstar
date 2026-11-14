"""Wan2.2's opt-in ``vae_decode_kernel`` (``mstar.model.components.wan22_vae_kernel``).

CPU: the kernels read the decoder's tensors by position, so their order and shapes are pinned against diffusers'
TI2V-5B ``AutoencoderKLWan`` (built on the meta device); the knob is refused where it cannot apply, the kernels build
for Hopper and Blackwell only, and a node that cannot run them (no sm_90 / sm_100 / sm_103 GPU, or not the TI2V-5B VAE)
decodes exactly as with the knob off.

GPU (compute capability 9.0 / 10.0 / 10.3 and the Wan2.2-TI2V-5B VAE in the HF cache): the kernels' video against an
fp32 decode of the same latents, and a stream cut into pieces against the whole decode, bit for bit."""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components import wan22_vae_kernel  # noqa: E402
from mstar.model.wan22 import submodules as wan22_submodules  # noqa: E402
from mstar.model.wan22.config import WAN22_VARIANT_T2V_A14B, WAN22_VARIANT_TI2V_5B, Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule  # noqa: E402

TI2V_5B_VAE = dict(
    base_dim=160, decoder_base_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
    temperal_downsample=[False, True, True], is_residual=True, in_channels=12, out_channels=12, patch_size=2,
    scale_factor_spatial=16, scale_factor_temporal=4,
)


def test_weight_order_is_the_ti2v_5b_decoders_parameters():
    with torch.device("meta"):
        vae = AutoencoderKLWan(**TI2V_5B_VAE)
    params = [(n, tuple(p.shape)) for n, p in vae.named_parameters() if n.startswith(("post_quant_conv.", "decoder."))]
    assert params == list(wan22_vae_kernel.WEIGHT_SHAPES)
    assert len(params) == 110


def test_the_model_passes_the_decoder_knobs_to_its_config():
    """A serving yaml's ``model_kwargs`` reach ``Wan22Model.__init__``, which builds the config from named arguments
    (an unnamed one would land in ``**kwargs`` unread)."""
    from mstar.model.wan22.wan22_model import Wan22Model

    assert Wan22Model(model_path_hf="test/wan22", vae_decode_kernel=True).config.vae_decode_kernel is True
    assert Wan22Model(model_path_hf="test/wan22").config.vae_decode_kernel is False


def test_knob_is_a_bool_and_ti2v_5b_only():
    assert Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B).vae_decode_kernel is False
    assert Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B, vae_decode_kernel=True).vae_decode_kernel is True
    with pytest.raises(ValueError, match="true or false"):
        Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B, vae_decode_kernel="false")
    with pytest.raises(ValueError, match="Wan2.1 VAE"):
        Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, vae_decode_kernel=True)


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


@pytest.mark.parametrize(
    ("capability", "arch"),
    [((9, 0), "sm_90a"), ((10, 0), "sm_100a"), ((10, 3), "sm_103a"), ((8, 0), None), ((12, 0), None)],
)
def test_the_kernels_build_for_hopper_and_blackwell_only(monkeypatch, capability, arch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(wan22_vae_kernel, "_cutlass_dir", lambda: "/cutlass")
    monkeypatch.setattr(wan22_vae_kernel.shutil, "which", lambda name: f"/usr/bin/{name}")
    reason = wan22_vae_kernel.unsupported_reason()
    if arch is None:
        assert reason is not None and f"not {capability[0]}.{capability[1]}" in reason
        return
    assert reason is None
    assert wan22_vae_kernel._arch(torch.device("cuda")) == arch
    cflags, cuda_flags = wan22_vae_kernel.build_flags(arch)
    assert f"-gencode=arch=compute_{arch[3:]},code={arch}" in cuda_flags
    # Hopper's configuration (csrc/arch.h, csrc/binding.cpp) is chosen at build time, and only for sm_90a
    assert ("-DWAN22_VAE_SM90" in cflags) == ("-DWAN22_VAE_SM90" in cuda_flags) == (arch == "sm_90a")


def test_a_vae_the_kernels_do_not_fit_is_named():
    reason = wan22_vae_kernel.unsupported_reason(vae=_tiny_vae())
    assert reason is not None and "TI2V-5B" in reason


def test_where_the_kernels_cannot_run_the_knob_changes_nothing(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2**40, 2**40))
    monkeypatch.setattr(wan22_submodules, "wan_vae_decode_dtype", lambda: torch.float32)
    vae = _tiny_vae()
    z = torch.randn(1, Z_DIM, 5, 4, 6, generator=torch.Generator().manual_seed(1))

    def frames(kernel: bool):
        cfg = Wan22Config.for_variant(
            WAN22_VARIANT_TI2V_5B, vae_decode_tiling="untiled", vae_decode_fast_sizes=("96x64",),
            vae_decode_kernel=kernel,
        )
        sub = Wan22VaeDecoderSubmodule(copy.deepcopy(vae), cfg)
        return sub.forward("video_gen", SimpleNamespace(request_ids=["r"]), z)["video_output"][0], sub

    want, _ = frames(False)
    with caplog.at_level(logging.WARNING):
        got, sub = frames(True)
        frames_again = sub.forward("video_gen", SimpleNamespace(request_ids=["s"]), z)["video_output"][0]
    assert torch.equal(got, want)
    assert frames_again.shape[1:] == want.shape[1:]
    assert sum("vae_decode_kernel is on but ignored" in r.message for r in caplog.records) == 1


def _gpu_vae():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in wan22_vae_kernel.ARCHS:
        pytest.skip("needs a compute capability 9.0 / 10.0 / 10.3 GPU")
    if wan22_vae_kernel.unsupported_reason() is not None:
        pytest.skip(wan22_vae_kernel.unsupported_reason())
    try:
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
        )
    except OSError:
        pytest.skip("Wan-AI/Wan2.2-TI2V-5B-Diffusers's VAE is not in the HF cache")
    return vae.to("cuda").eval()


def _smooth_latents(t: int, h: int, w: int) -> torch.Tensor:
    """Raw latents with a real VAE's statistics: a smooth random field, de-normalized with the VAE's mean / std."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, 48, t, h // 4 + 1, w // 4 + 1, device="cuda", generator=gen)
    x = torch.nn.functional.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=True)
    return x


@pytest.mark.parametrize(("h", "w"), [(44, 80), (30, 52)])  # 1280x704 and 832x480
def test_gpu_kernels_match_an_fp32_decode_and_a_stream_the_whole_decode(h, w):
    vae = _gpu_vae()
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    z = _smooth_latents(6, h, w) * std + mean
    with torch.no_grad():
        ref = vae.decode(z, return_dict=False)[0].float()
        stock_bf16 = copy.deepcopy(vae).to(torch.bfloat16).decode(z.bfloat16(), return_dict=False)[0].float()
        kernel = wan22_vae_kernel.Wan22VaeKernel(vae, torch.device("cuda"))
        whole = kernel.decode(z)
        stream = kernel.stream(h, w)
        pieces = torch.cat([stream.step(z[:, :, i:i + 2]) for i in range(0, 6, 2)], dim=2)
    assert whole.shape == ref.shape == (1, 3, 21, 16 * h, 16 * w)
    assert torch.equal(pieces, whole)

    def psnr(a):
        return float(10 * torch.log10(4.0 / (a - ref).pow(2).mean()))  # [-1, 1]: peak-to-peak 2

    # the kernels err against fp32 about as much as diffusers' own bf16 decode does
    assert psnr(whole.float()) > psnr(stock_bf16) - 3.0
