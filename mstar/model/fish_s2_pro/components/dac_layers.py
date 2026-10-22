# Copyright (c) 2023-present, Descript
# SPDX-License-Identifier: MIT
#
# This file has been modified for M*: adapted from descript-audio-codec 1.0.0
# (https://github.com/descriptinc/descript-audio-codec, ``dac/nn/layers.py``
# and ``dac/nn/quantize.py``) through the inference-only copy vendored by
# vLLM-Omni (Apache-2.0, ``vllm_omni/model_executor/models/fish_speech/
# dac_modules/layers.py`` at f69b1f2b1).
#
# Original file was released under the MIT License, reproduced below. This
# modified file is released under the same license.
#
# MIT License
#
# Copyright (c) 2023-present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Snake activation and the factorized vector quantizers of the DAC codec.

Inference only. The 1x1 projections are plain ``Conv1d``: ``codec.pth``
stores them weight-normed (``weight_g`` / ``weight_v``), and the loader in
``dac.py`` folds those into ``weight`` once, with the same
``torch._weight_norm`` the weight-norm hook evaluates on every call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Snake1d(nn.Module):
    """``x + sin(alpha * x)^2 / alpha``, per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


class VectorQuantize(nn.Module):
    """Factorized, l2-normalized codebook lookup (Improved VQGAN)."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, T]`` codes -> ``[B, codebook_dim, T]`` codebook vectors."""
        return F.embedding(codes, self.codebook.weight).transpose(1, 2)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, D, T]`` -> (quantized ``[B, D, T]``, codes ``[B, T]``)."""
        latents = self.in_proj(z)
        batch, dim, time = latents.shape
        encodings = F.normalize(latents.permute(0, 2, 1).reshape(batch * time, dim))
        codebook = F.normalize(self.codebook.weight)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        codes = (-dist).max(1)[1].reshape(batch, time)
        # The reference's straight-through form; it can round differently
        # from the bare codebook vector, and the next residual sees that.
        z_q = latents + (self.decode_code(codes) - latents)
        return self.out_proj(z_q), codes


class ResidualVectorQuantize(nn.Module):
    """SoundStream residual VQ over ``n_codebooks`` factorized quantizers."""

    def __init__(self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)]
        )

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, D, T]`` -> (quantized ``[B, D, T]``, codes ``[B, N, T]``)."""
        z_q = torch.zeros_like(z)
        residual = z
        codes = []
        for quantizer in self.quantizers:
            z_q_i, codes_i = quantizer.quantize(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(codes_i)
        return z_q, torch.stack(codes, dim=1)

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, N, T]`` codes -> ``[B, D, T]`` quantized representation."""
        z_q = 0.0
        for i, quantizer in enumerate(self.quantizers[: codes.shape[1]]):
            z_q = z_q + quantizer.out_proj(quantizer.decode_code(codes[:, i]))
        return z_q
