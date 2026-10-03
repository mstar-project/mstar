"""BAGEL's encoders must leave the request's image as they found it. A resize
that keeps the size hands back its input, so a normalize in place wrote into
the request's own tensor, and the next walk to read it normalized it twice."""

from types import SimpleNamespace

import pytest
import torch

from mstar.model.bagel.components.modeling_utils import ImageTransform
from mstar.model.bagel.submodules import VAEEncoderSubmodule, ViTEncoderSubmodule


class _Posterior:
    def posterior_noise(self, height, width, generator=None, device=None):
        return torch.zeros(1, 16, height // 8, width // 8)


def _vae():
    sub = VAEEncoderSubmodule.__new__(VAEEncoderSubmodule)
    sub.vae_model = _Posterior()
    sub.latent_patch_size = 2
    sub.latent_channel = 16
    sub.latent_downsample = 16
    sub.max_latent_size = 64
    sub.transform = ImageTransform(1024, 512, 16)
    return sub


def _vit():
    return ViTEncoderSubmodule(None, None, None, 14, 70)


# sizes every resize of the encoder keeps, so each hands back its input
@pytest.mark.parametrize(
    ("encoder", "walk", "preprocess", "size"),
    [
        (_vae, "prefill_vae", "default", (512, 512)),
        (_vae, "prefill_vae", "vllm", (512, 512)),
        (_vit, "prefill_vit", "default", (560, 784)),
        (_vit, "prefill_vit", "vllm", (980, 980)),
    ],
    ids=["vae", "vae-vllm", "vit", "vit-vllm"],
)
def test_an_encoder_leaves_the_requests_image_as_it_found_it(encoder, walk, preprocess, size):
    image = torch.rand(3, *size)
    before = image.clone()

    fwd_info = SimpleNamespace(random_seed=0, step_metadata={"image_preprocess": preprocess})
    encoder().prepare_inputs(walk, fwd_info, {"image_inputs": [image]})

    assert torch.equal(image, before), (
        "the encoder normalized the request's own image, so the next walk to read it normalizes it twice"
    )
