# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# Copyright (c) 2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
# Copyright (c) 2022 Xingchen Song
# Copyright (c) 2024 Alibaba Inc (Xiang Lyu)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: reimplemented on Chatterbox's
# flow-matching solver, settling whole 25-token chunks so a stream equals a
# whole-utterance solve.
#
# Original files (FunAudioLLM/CosyVoice, cosyvoice/flow/flow.py,
# cosyvoice/flow/flow_matching.py and
# cosyvoice/transformer/upsample_encoder.py) were released under Apache-2.0,
# with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""CosyVoice3's token-to-mel flow: speech tokens -> ``mu`` -> mel.

Tokens (prompt then generated) are embedded to 80 channels, passed through a
look-ahead layer (a convolution that reads three tokens ahead, then a causal
one) and repeated x2 to the 50 Hz mel rate. The mel is integrated from a fixed
noise field by Chatterbox's classifier-free-guided Euler solver, with the DiT
as its estimator; the prompt mel is written into ``cond`` so the generated
mel continues the reference voice.

A row of a batched solve settles ``settle`` generated tokens. Unless the row
is final, the three tokens after them must already exist: they are the
look-ahead layer's right context and are not decoded. A streaming row attends
block-causally over 25-token chunks, so its frames depend on nothing past the
end of their chunk; settling whole chunks makes a stream's mel equal to one
whole-utterance streaming solve.

Ported from ``cosyvoice/flow/flow.py`` (``CausalMaskedDiffWithDiT``),
``cosyvoice/transformer/upsample_encoder.py`` (``PreLookaheadLayer``) and
``cosyvoice/flow/flow_matching.py`` (``CausalConditionalCFM``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.components.s3gen_cfm import CausalConditionalCFM
from mstar.model.chatterbox.loader import load_component
from mstar.model.cosyvoice3.components.dit import DiT, chunk_attention_mask
from mstar.model.cosyvoice3.config import FlowConfig
from mstar.utils.h2d import PinnedStager


@dataclass
class ReferenceVoice:
    """What the flow (and the LM's prompt) needs from a reference clip."""

    prompt_tokens: torch.Tensor  # [P] long, speech tokens of the clip
    prompt_feat: torch.Tensor  # [2P, 80] float, 24 kHz mel of the clip
    embedding: torch.Tensor  # [192] float, CAMPPlus x-vector

    @property
    def num_prompt_tokens(self) -> int:
        return int(self.prompt_tokens.shape[0])


@dataclass
class CosyVoice3FlowRow:
    """One request's share of a batched solve: its tokens so far, how many to
    settle, whether the stream has ended (the look-ahead is zero padding
    instead of real tokens) and whether it attends block-causally."""

    voice: ReferenceVoice
    tokens: torch.Tensor  # [n] long, generated speech tokens
    settle: int
    final: bool
    streaming: bool


class PreLookaheadLayer(nn.Module):
    """``conv1`` reads ``pre_lookahead_len`` tokens ahead (zeros past the end),
    ``conv2`` two tokens back; residual."""

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """``[B, L, C] -> [B, L, C]``; position ``i`` reads inputs ``i .. i+3``."""
        x = F.pad(inputs.transpose(1, 2), (0, self.pre_lookahead_len))
        # In the flow solve these run in float32 with TF32 off. There, for
        # B * L under about 500, cuDNN picks a plan that reserves about 40 GiB
        # of workspace and takes about 155 ms a call (H200, cuDNN 9); the
        # native convolution takes about 0.1 ms and is as close to float64.
        with torch.backends.cudnn.flags(enabled=False):
            x = F.leaky_relu(self.conv1(x))
            x = self.conv2(F.pad(x, (self.conv2.kernel_size[0] - 1, 0)))
        return x.transpose(1, 2) + inputs


class CosyVoice3Flow(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        self.input_embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.output_size)
        self.pre_lookahead_layer = PreLookaheadLayer(
            config.input_size,
            config.pre_lookahead_channels,
            config.pre_lookahead_len,
        )
        self.decoder = CausalConditionalCFM(config.cfm, DiT(config.dit))
        generator = torch.Generator().manual_seed(config.noise_seed)
        noise = torch.randn(1, config.output_size, config.noise_frames, generator=generator)
        self.register_buffer("noise", noise, persistent=False)
        self._row_stager = PinnedStager(torch.long)

    @property
    def device(self) -> torch.device:
        return self.input_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.input_embedding.weight.dtype

    def set_estimator_dtype(self, dtype: torch.dtype) -> None:
        """Run the DiT estimator in ``dtype``. The solve keeps passing it float32
        state and time steps, which the DiT casts itself (the time step only
        after its sinusoid), so ``CausalConditionalCFM``'s own input casts stay
        off."""
        self.decoder.estimator.to(dtype)

    def project_speaker(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.spk_embed_affine_layer(F.normalize(embedding, dim=1))

    @torch.no_grad()
    def solve_rows(self, rows: Sequence[CosyVoice3FlowRow], n_timesteps: int | None = None) -> list[torch.Tensor]:
        """Each row's generated mel ``[80, 2 * settle]`` (frames from the first
        generated token on), from one right-padded batched solve.

        Padding cannot reach a row's frames: embeddings past a row's tokens are
        zeroed, the look-ahead reads at most three tokens past the settled
        ones (which exist unless the row is final), the DiT's convolutions are
        causal and its attention masks the padded keys.
        """
        cfg = self.config
        ratio = cfg.token_mel_ratio
        lookahead = cfg.pre_lookahead_len
        device, dtype = self.device, self.dtype
        seqs, valid_tokens = [], []
        for row in rows:
            n = int(row.tokens.numel())
            if row.final:
                if row.settle != n:
                    raise ValueError(f"a final row settles all {n} tokens, not {row.settle}")
                used = n
            else:
                if row.settle + lookahead > n:
                    raise ValueError(f"settling {row.settle} of {n} tokens leaves no look-ahead")
                used = row.settle + lookahead
            prompt = row.voice.prompt_tokens.to(device, torch.long)
            seqs.append(torch.cat([prompt, row.tokens[:used].to(device, torch.long)]))
            valid_tokens.append(prompt.numel() + row.settle)
        batch = len(rows)
        valid_frames = [v * ratio for v in valid_tokens]
        # every per-row count is known here: one copy for the lot
        counts = torch.empty(3, batch, dtype=torch.long, device=device)
        self._row_stager.copy_(counts, [s.numel() for s in seqs] + valid_frames + [row.streaming for row in rows])
        lens, valid, streaming = counts[0], counts[1], counts[2] != 0
        tokens = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        width = tokens.shape[1]
        in_row = torch.arange(width, device=device)[None, :] < lens[:, None]
        emb = self.input_embedding(tokens.clamp_min(0)) * in_row.unsqueeze(-1).to(dtype)
        mu = self.pre_lookahead_layer(emb).repeat_interleave(ratio, dim=1).transpose(1, 2)

        total = max(valid_frames)
        if total > cfg.noise_frames:
            raise ValueError(f"the flow noise covers {cfg.noise_frames} frames; this solve needs {total}")
        mu = mu[:, :, :total].contiguous()
        cond = torch.zeros(batch, cfg.output_size, total, device=device, dtype=dtype)
        for i, row in enumerate(rows):
            feat = row.voice.prompt_feat.to(device, dtype)
            cond[i, :, : feat.shape[0]] = feat.transpose(0, 1)
        spks = self.project_speaker(torch.stack([row.voice.embedding.to(device, dtype) for row in rows]))
        mask = chunk_attention_mask(valid, total, cfg.dit.static_chunk_size, streaming)
        noise = self.noise[:, :, :total].to(device, dtype).expand(batch, -1, -1).contiguous()
        mel = self.decoder.solve(mu, mask, spks, cond, noise, n_timesteps or cfg.cfm.n_timesteps)
        return [mel[i, :, row.voice.num_prompt_tokens * ratio : valid_frames[i]] for i, row in enumerate(rows)]

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    @staticmethod
    def _remap(name: str) -> str | None:
        if name == "decoder.estimator.rotary_embed.inv_freq":
            return None  # recomputed
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``flow.pt`` into this module, exactly."""
        return load_component(self, weights, component="CosyVoice3 flow", name_remapper=self._remap)
