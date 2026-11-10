"""The native keyframe encoder against the reference's ``encode_vae_condition`` over
diffusers' ``AutoencoderKLMiniMaxH3``, tiny and on CPU.

A random tiny VAE in diffusers' layout is re-keyed to the original checkpoint's names
(``encoder.down.<i>.block.<j>``, ``nin_shortcut``, ``downsample``) and loaded through this
port's loader. Both encode the same uint8 keyframe, untiled and over 2 x 2 blended tiles,
and sample the posterior under the same seed.
"""

from __future__ import annotations

import re

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLMiniMaxH3"):
    pytest.skip("this diffusers has no MiniMax-H3", allow_module_level=True)
encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")

from safetensors.torch import save_file  # noqa: E402

from mstar.model.minimax_h3 import weight_loader  # noqa: E402
from mstar.model.minimax_h3.config import (  # noqa: E402
    KEYFRAME_ENCODE_SEED,
    PIXEL_MEAN,
    PIXEL_STD,
    H3Config,
    H3VideoEncoderConfig,
    H3VideoVAEConfig,
)

LATENT = 4
MEAN, STD = (0.1, -0.2, 0.3, 0.0), (1.5, 0.5, 2.0, 1.0)


def _to_original(name: str) -> str:
    name = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", name)
    name = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", name)
    return name.replace(".conv_shortcut.", ".nin_shortcut.")


@pytest.fixture(scope="module")
def encoders_pair(tmp_path_factory):
    torch.manual_seed(0)
    reference = diffusers.AutoencoderKLMiniMaxH3(
        latent_channels=LATENT, block_out_channels=(8, 8, 8, 8, 8, 16), norm_num_groups=4,
        decoder_num_layers=1, decoder_num_attention_heads=2, decoder_attention_head_dim=32,
        latents_mean=MEAN, latents_std=STD,
    ).eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if name.startswith(("encoder.", "quant_conv.")):
                param.copy_(torch.randn(param.shape, generator=g) * 0.2)
    state = {_to_original(k): v.contiguous() for k, v in reference.state_dict().items()
             if k.startswith(("encoder.", "quant_conv."))}
    root = tmp_path_factory.mktemp("h3enc")
    (root / "FL2VA" / "video_vae" / "source").mkdir(parents=True)
    save_file(state, str(root / "FL2VA" / "video_vae" / "source" / "model.safetensors"))
    config = H3Config(
        video_vae=H3VideoVAEConfig(latent_channels=LATENT, latents_mean=MEAN, latents_std=STD),
        video_encoder=H3VideoEncoderConfig(base_channels=8, channel_mult=(1, 1, 1, 1, 1, 2), latent_channels=LATENT,
                                           norm_groups=4),
    )
    return weight_loader.build_keyframe_encoder(config, root, "cpu"), reference


@pytest.mark.parametrize("size", [(64, 96), (320, 288)])
def test_keyframe_latents_match_the_reference(encoders_pair, size):
    """One tile, and 2 x 2 tiles blended in latent space."""
    ours, reference = encoders_pair
    frame = torch.randint(0, 256, (3, *size), dtype=torch.uint8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = encoders.encode_vae_condition(reference, frame[None, :, None], PIXEL_MEAN, PIXEL_STD,
                                             KEYFRAME_ENCODE_SEED)
        got = ours.encode(frame, KEYFRAME_ENCODE_SEED)
    assert got.shape == want.shape == (1, LATENT, 1, size[0] // 16, size[1] // 16)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
