"""The decoder of TAEHV's tiny autoencoders for the Wan2.2-TI2V-5B latent space (``taew2_2``, ``taew2_2_super``).

Vendored from madebyollin/taehv (MIT, ``taehv.py`` @ 2026-11): the decoder half only, in its parallel-over-time form.
A tiny VAE decodes a 1280x704x81 clip in 31 ms (``taew2_2``) or 113 ms (``taew2_2_super``) on a B300, against 1.15 s
for the full Wan2.2 VAE, and is lossy: 35.0 / 38.4 dB PSNR from an fp32 full decode, LPIPS 0.027 / 0.014 (the CRF 18
mp4 encode itself is ~38.6 dB). A preview / draft option, never a default.

It decodes the DiT's latents as they come out of the denoise loop (normalized: no ``latents_mean`` / ``latents_std``),
``[B, 48, T, h, w]``, to ``[B, 3, 4T - 3, 16h, 16w]`` in [-1, 1], as the full VAE's decode is laid out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

LATENT_CHANNELS = 48
PATCH = 2  # pixel-shuffle patch of the Wan2.2 checkpoints


def _conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), nn.ReLU(inplace=True), _conv(n_out, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _SuperMemBlock(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_f * 2, n_f * 2, 7, padding=3, groups=n_f * 2, bias=False),
            nn.Conv2d(n_f * 2, n_f * 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(n_f * 4, n_f, 1, bias=False),
        )

    def forward(self, x, past):
        return self.conv(torch.cat([x, past], 1)) + x


class _TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


def _decoder(super_: bool) -> nn.Sequential:
    """The Wan2.2 checkpoints' decoder: 2x space in every block, 2x time in the last two."""
    time_up = (1, 2, 2)
    if super_:
        n = [512, 256, 128, 64]
        layers = [nn.Conv2d(LATENT_CHANNELS, n[0], 1, bias=False)]
        for i in range(3):
            layers += [_SuperMemBlock(n[i]), _SuperMemBlock(n[i]), _SuperMemBlock(n[i]), _conv(n[i], n[i + 1] * 4),
                       nn.ReLU(inplace=True), nn.PixelShuffle(2), _TGrow(n[i + 1], time_up[i])]
        layers.append(_conv(n[3], 3 * PATCH**2))
    else:
        n = [256, 128, 64, 64]
        layers = [_Clamp(), _conv(LATENT_CHANNELS, n[0]), nn.ReLU(inplace=True)]
        for i in range(3):
            layers += [_MemBlock(n[i], n[i]), _MemBlock(n[i], n[i]), _MemBlock(n[i], n[i]), nn.Upsample(scale_factor=2),
                       _TGrow(n[i], time_up[i]), _conv(n[i], n[i + 1], bias=False)]
        layers += [nn.ReLU(inplace=True), _conv(n[3], 3 * PATCH**2)]
    return nn.Sequential(*layers)


class TinyWanDecoder(nn.Module):
    """``taew2_2`` (``super_=False``) or ``taew2_2_super``; ``load`` takes upstream's checkpoint file."""

    def __init__(self, super_: bool = False):
        super().__init__()
        self.decoder = _decoder(super_)
        self.frames_to_trim = 2 ** sum(isinstance(b, _TGrow) and b.stride == 2 for b in self.decoder) - 1

    @classmethod
    def load(cls, path: str) -> "TinyWanDecoder":
        """From upstream's ``taew2_2.pth`` / ``taew2_2_super.pth`` (or ``.safetensors``); "_super" in the file name
        selects the larger decoder, as upstream reads it."""
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(super_="_super" in path.rsplit("/", 1)[-1])
        own = model.state_dict()
        sd = {k: v for k, v in sd.items() if k.startswith("decoder.")}
        for i, block in enumerate(model.decoder):  # a TGrow may carry more timesteps than it grows: keep the last
            k = f"decoder.{i}.conv.weight"
            if isinstance(block, _TGrow) and k in sd and sd[k].shape[0] > own[k].shape[0]:
                sd[k] = sd[k][-own[k].shape[0]:]
        model.load_state_dict(sd)
        return model.eval()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, 48, T, h, w]`` normalized latents -> ``[B, 3, 4T - 3, 16h, 16w]`` fp32 in [-1, 1]."""
        b, _, t = latents.shape[:3]
        x = latents.permute(0, 2, 1, 3, 4).to(next(self.parameters()).dtype).flatten(0, 1)  # [B*T, C, h, w]
        for block in self.decoder:
            if isinstance(block, (_MemBlock, _SuperMemBlock)):
                bt, c, h, w = x.shape
                frames = x.reshape(b, bt // b, c, h, w)
                past = F.pad(frames, (0, 0, 0, 0, 0, 0, 1, 0))[:, : bt // b].reshape(x.shape)  # each frame's previous
                x = block(x, past)
            else:
                x = block(x)
        x = F.pixel_shuffle(x, PATCH).clamp_(0, 1)
        x = x.reshape(b, -1, *x.shape[1:])[:, self.frames_to_trim:]  # [B, F, 3, H, W]
        return x.permute(0, 2, 1, 3, 4).float() * 2 - 1
