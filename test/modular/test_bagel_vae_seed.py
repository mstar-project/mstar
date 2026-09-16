"""BAGEL image-to-image must be repeatable per seed. The VAE encoder samples
its posterior; the draw comes from the request seed, not the global RNG."""

from types import SimpleNamespace

import torch

from mstar.model.bagel.components.autoencoder import BagelAutoEncoder, DiagonalGaussian
from mstar.model.bagel.components.modeling_utils import ImageTransform
from mstar.model.bagel.config import BagelAutoEncoderConfig
from mstar.model.bagel.submodules import VAEEncoderSubmodule


def test_posterior_sample_uses_the_given_noise():
    reg = DiagonalGaussian()
    z = torch.randn(1, 4, 3, 3)
    noise = torch.randn(1, 2, 3, 3)
    mean, logvar = torch.chunk(z, 2, dim=1)
    assert torch.equal(reg(z, noise=noise), mean + torch.exp(0.5 * logvar) * noise)
    assert torch.equal(reg(z, noise=noise), reg(z, noise=noise))
    assert not torch.equal(reg(z), reg(z))  # global RNG: a fresh draw each call
    assert torch.equal(DiagonalGaussian(sample=False)(z), mean)


def _small_ae():
    return BagelAutoEncoder(BagelAutoEncoderConfig(
        resolution=64, ch=32, ch_mult=(1, 1, 1, 1), num_res_blocks=1, z_channels=4,
    ))


def test_seeded_encode_is_repeatable():
    ae = _small_ae().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        first = ae.encode(x, noise=ae.posterior_noise(64, 64, torch.Generator().manual_seed(7)))
        again = ae.encode(x, noise=ae.posterior_noise(64, 64, torch.Generator().manual_seed(7)))
        other = ae.encode(x, noise=ae.posterior_noise(64, 64, torch.Generator().manual_seed(8)))
    assert first.shape == (1, 4, 8, 8)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


class _NoiseRecorder:
    def __init__(self):
        self.calls = []

    def posterior_noise(self, height, width, generator=None, device=None):
        noise = torch.randn(1, 16, height // 8, width // 8, generator=generator, device=device)
        self.calls.append(noise)
        return noise


def _submodule(vae_model):
    sub = VAEEncoderSubmodule.__new__(VAEEncoderSubmodule)
    sub.vae_model = vae_model
    sub.latent_patch_size = 2
    sub.latent_channel = 16
    sub.latent_downsample = 16
    sub.max_latent_size = 64
    sub.transform = ImageTransform(1024, 512, 16)
    return sub


def _prepare(sub, seed):
    fwd_info = SimpleNamespace(random_seed=seed, step_metadata={})
    return sub.prepare_inputs("prefill_vae", fwd_info, {"image_inputs": [torch.rand(3, 64, 64)]})


def test_vae_noise_comes_from_the_request_seed():
    recorder = _NoiseRecorder()
    sub = _submodule(recorder)
    a = _prepare(sub, seed=99).tensor_inputs["vae_noise"]
    b = _prepare(sub, seed=99).tensor_inputs["vae_noise"]
    c = _prepare(sub, seed=100).tensor_inputs["vae_noise"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert len(recorder.calls) == 3
    padded = _prepare(sub, seed=99).tensor_inputs["padded_images"]
    assert a.shape == (1, 16, padded.shape[-2] // 8, padded.shape[-1] // 8)
