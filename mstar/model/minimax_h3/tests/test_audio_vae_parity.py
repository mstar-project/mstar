"""The native MiniMax-H3 audio decoder against diffusers' ``AutoencoderKLMiniMaxH3Audio``,
tiny and on CPU.

diffusers keeps the original checkpoint's key names (weight-normed convolutions as
``weight_g`` / ``weight_v``, the resampling filters as buffers), so its random state dict
is a checkpoint in the original layout: saved as ``FL2VA/audio_vae/model.safetensors``,
it loads through this port's loader, and both decode the same latents.
"""

from __future__ import annotations

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLMiniMaxH3Audio"):
    pytest.skip("this diffusers has no MiniMax-H3", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from mstar.model.minimax_h3 import weight_loader  # noqa: E402
from mstar.model.minimax_h3.config import H3AudioVAEConfig, H3Config  # noqa: E402

LATENT, LATENT_DIM, DECODER_DIM = 4, 16, 32
RATES, KERNELS = (2, 3), (4, 5)


@pytest.fixture(scope="module")
def decoders(tmp_path_factory):
    reference = diffusers.AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4, encoder_rates=(3, 2), latent_dim=LATENT_DIM, latent_channels=LATENT, num_attention_heads=2,
        decoder_dim=DECODER_DIM, decoder_rates=RATES, decoder_kernel_sizes=KERNELS,
    ).eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in reference.parameters():
            param.copy_(torch.randn(param.shape, generator=g) * 0.3)
    root = tmp_path_factory.mktemp("h3audio")
    (root / "FL2VA" / "audio_vae").mkdir(parents=True)
    state = {k: v.contiguous() for k, v in reference.state_dict().items()}
    save_file(state, str(root / "FL2VA" / "audio_vae" / "model.safetensors"))
    config = H3Config(audio_vae=H3AudioVAEConfig(
        latent_channels=LATENT, latents_mean=(0.0,) * LATENT, latents_std=(1.0,) * LATENT,
        latent_dim=LATENT_DIM, decoder_dim=DECODER_DIM, upsample_rates=RATES, upsample_kernel_sizes=KERNELS,
    ))
    ours = weight_loader.build_audio_vae(config, root, "cpu")
    return ours, reference


def test_decode_matches_diffusers(decoders):
    ours, reference = decoders
    z = torch.randn(2, LATENT, 9, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = reference.decode(z, return_dict=False)[0]
        got = ours(z)
    assert got.shape == want.shape == (2, 1, 9 * 6)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
