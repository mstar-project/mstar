"""Shared pieces for the Wan VAE streaming-decode benchmark.

Wan 2.1 VAE decoder, batch 1, one steady chunk = 3 latent frames
``[1, 16, 3, 60, 104]`` -> 12 RGB frames at 480x832, bf16, causal conv cache
warm across chunks, timed as decode + conversion to 8-bit frames, median /
p90 over 75 steady chunks.
"""

import os
from pathlib import Path

import torch

WAN21_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN22_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

CHUNK_LATENTS = 3      # latent frames per steady chunk (z in R^{16x3x60x104})
FRAMES_PER_CHUNK = 12  # pixel frames per steady chunk (4x temporal)


def hf_snapshot(repo: str) -> Path:
    """Local snapshot dir of ``repo`` from the HF cache; never downloads."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=["vae/*"], local_files_only=True))


def load_vae(repo: str, dtype: torch.dtype, device: str, keep_encoder: bool = False):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(str(hf_snapshot(repo) / "vae"), torch_dtype=dtype)
    if not keep_encoder:
        vae.encoder = None
        vae.quant_conv = None
    return vae.to(device).eval()


def latent_stats(vae, device, dtype=torch.float32):
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, z, 1, 1, 1)
    return mean, std


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> uint8, the formula both M* decoder nodes use."""
    return (video / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)


class StreamingDecoder:
    """Chunked causal decode of a Wan VAE with the conv feature cache kept warm
    across chunks. Mirrors ``AutoencoderKLWan._decode`` minus ``clear_cache``
    between calls.

    per_frame=True is stock diffusers behaviour (one latent frame per decoder
    call); per_frame=False decodes the whole chunk in one call.
    """

    def __init__(self, vae, *, per_frame: bool, compile_decoder: bool):
        self.vae = vae
        self.per_frame = per_frame
        self.patch_size = vae.config.patch_size
        self.decoder = vae.decoder
        self.step = torch.compile(vae.decoder, fullgraph=False, dynamic=False) if compile_decoder else vae.decoder
        self.reset()

    def reset(self):
        self.feat = [None] * self.vae._cached_conv_counts["decoder"]

    def _finish(self, out):
        if self.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            out = unpatchify(out, patch_size=self.patch_size)
        return out.clamp(-1.0, 1.0)

    def first(self, z1):
        """The stream's first latent frame (decodes to 1 pixel frame)."""
        out = self.step(self.vae.post_quant_conv(z1), feat_cache=self.feat, feat_idx=[0], first_chunk=True)
        return self._finish(out)

    def chunk(self, z, feat=None):
        feat = self.feat if feat is None else feat
        x = self.vae.post_quant_conv(z)
        if self.per_frame:
            out = torch.cat([self.step(x[:, :, i:i + 1], feat_cache=feat, feat_idx=[0]) for i in range(x.shape[2])], 2)
        else:
            out = self.step(x, feat_cache=feat, feat_idx=[0])
        return self._finish(out)


def mstar_path() -> str:
    """M* checkout whose model code the M* variants import: $MSTAR, else the repo
    this script lives in, else ~/oss/mstar."""
    if "MSTAR" in os.environ:
        return os.environ["MSTAR"]
    here = Path(__file__).resolve().parents[2]
    if (here / "mstar" / "model").is_dir():
        return str(here)
    return str(Path.home() / "oss" / "mstar")
