"""Wan2.2's opt-in ``vae_decode_tiny`` (``mstar.model.components.taehv_decoder``): the decoder half of TAEHV's
``taew2_2`` / ``taew2_2_super``, loaded from upstream's checkpoint layout, decoding the DiT's normalized latents to the
full VAE's frame layout, in the node. CPU, random weights."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components.taehv_decoder import TinyWanDecoder  # noqa: E402
from mstar.model.wan22.config import WAN22_VARIANT_T2V_A14B, WAN22_VARIANT_TI2V_5B, Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule  # noqa: E402


def _latents(t=5, h=4, w=6, seed=1):
    return torch.randn(1, 48, t, h, w, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("super_", [False, True])
def test_decodes_to_the_full_vaes_layout(super_):
    torch.manual_seed(0)
    tiny = TinyWanDecoder(super_=super_).eval()
    with torch.no_grad():
        out = tiny(_latents())
    assert out.shape == (1, 3, 17, 64, 96) and out.dtype == torch.float32  # 4T - 3 frames, 16x space
    assert out.min() >= -1 and out.max() <= 1


def test_causal_in_time():
    """A frame never sees later latents (TAEHV's memory is the previous frame), as a streamed decode needs."""
    torch.manual_seed(0)
    tiny = TinyWanDecoder(super_=True).eval()
    z = _latents()
    z2 = z.clone()
    z2[:, :, 3:] = torch.randn_like(z2[:, :, 3:])
    with torch.no_grad():
        a, b = tiny(z), tiny(z2)
    frames_of_first_3 = 4 * 3 - 3
    assert torch.equal(a[:, :, :frames_of_first_3], b[:, :, :frames_of_first_3])
    assert not torch.equal(a, b)


@pytest.mark.parametrize("name", ["taew2_2.pth", "taew2_2_super.pth"])
def test_loads_upstreams_checkpoint_layout(tmp_path, name):
    """Upstream saves the encoder too and may give a TGrow more timesteps than it grows (the last ones are kept)."""
    torch.manual_seed(0)
    want = TinyWanDecoder(super_="_super" in name)
    sd = {"encoder.0.weight": torch.randn(64, 12, 3, 3)}
    for k, v in want.state_dict().items():
        sd[k] = v.clone()
    grow = next(i for i, b in enumerate(want.decoder) if type(b).__name__ == "_TGrow" and b.stride == 2)
    key = f"decoder.{grow}.conv.weight"
    sd[key] = torch.cat([torch.randn_like(sd[key]), sd[key]])
    torch.save(sd, tmp_path / name)
    got = TinyWanDecoder.load(str(tmp_path / name))
    for k, v in want.state_dict().items():
        assert torch.equal(got.state_dict()[k], v), k
    assert not got.training


def test_knob_is_a_path_for_ti2v_5b_and_one_decoder():
    from mstar.model.wan22.wan22_model import Wan22Model

    assert Wan22Model(model_path_hf="test/wan22", vae_decode_tiny="taew2_2.pth").config.vae_decode_tiny == "taew2_2.pth"
    assert Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B).vae_decode_tiny is None
    with pytest.raises(ValueError, match="checkpoint path"):
        Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B, vae_decode_tiny="")
    with pytest.raises(ValueError, match="A14B"):
        Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, vae_decode_tiny="taew2_2.pth")
    with pytest.raises(ValueError, match="set one"):
        Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B, vae_decode_tiny="taew2_2.pth", vae_decode_kernel=True)


def test_the_node_decodes_with_the_tiny_vae(tmp_path):
    torch.manual_seed(0)
    tiny = TinyWanDecoder(super_=False)
    torch.save(tiny.state_dict(), tmp_path / "taew2_2.pth")
    vae = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=16, z_dim=48, dim_mult=[1, 2, 2, 2], num_res_blocks=1, is_residual=True,
        in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
    ).eval()
    cfg = Wan22Config.for_variant(WAN22_VARIANT_TI2V_5B, vae_decode_tiny=str(tmp_path / "taew2_2.pth"))
    sub = Wan22VaeDecoderSubmodule(copy.deepcopy(vae), cfg)
    z = _latents()
    with torch.no_grad():
        video = tiny.to(torch.bfloat16)(z)
    want = Wan22VaeDecoderSubmodule._frames(video)
    got = sub.forward("video_gen", SimpleNamespace(request_ids=["r"]), z)["video_output"][0]
    assert got.shape == (17, 64, 96, 3) and got.dtype == torch.uint8
    assert torch.equal(got, want)
