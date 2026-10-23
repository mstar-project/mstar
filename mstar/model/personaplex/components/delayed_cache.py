# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Adapted for M* on 2026-09-28 from NVIDIA/personaplex models/lm.py at
# 3428dfd95309a7f3c84fd93259ded0f810d1ff91: independent integer cache component.
"""Integer state of LMGen, independent of transformers and M* resources.

``prepare`` returns False only for token-frame priming (no forward). Otherwise
read ``model_input``, ``target`` and ``target_provided``, run the model, and
``commit`` its raw sampled text/audio. Forced depth inputs must use the live
target mask. Commit returns the delayed output and advances the offset.
"""

import torch

from ..config import (
    AUDIO_CODEBOOKS_PER_STREAM,
    AUDIO_INITIAL_TOKEN_ID,
    CACHE_SIZE,
    DELAYS,
    MAX_DELAY,
    NUM_AUDIO_CODEBOOKS,
    NUM_CODEBOOKS,
    TEXT_INITIAL_TOKEN_ID,
    TEXT_PAD_TOKEN_ID,
    UNGENERATED_TOKEN_ID,
)


class DelayedTokenCache:
    """Mirror [PP] LMGen.prepare_step_input/process_transformer_output.

    Token tensors are int64 on the cache device: agent/user [B,8,1], text
    [B], sampled audio [B,16]. Properties are live views for teacher forcing;
    callers must clone them if they need snapshots after ``commit``.
    """

    def __init__(self, batch_size: int = 1, device: str | torch.device = "cpu"):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.cache = torch.full(
            (batch_size, NUM_CODEBOOKS, CACHE_SIZE), UNGENERATED_TOKEN_ID, dtype=torch.long, device=device
        )
        self.provided = torch.zeros_like(self.cache, dtype=torch.bool)
        self.initial = torch.full(
            (batch_size, NUM_CODEBOOKS, 1), AUDIO_INITIAL_TOKEN_ID, dtype=torch.long, device=device
        )
        self.initial[:, 0] = TEXT_INITIAL_TOKEN_ID
        self.offset = 0
        self._prepared = False
        self._delays = torch.tensor(DELAYS, device=device)

    def _validate(self, name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if value.shape != shape or value.dtype != torch.long or value.device != self.cache.device:
            raise ValueError(f"{name} must be int64 {shape} on {self.cache.device}")

    def prepare(
        self, *, user: torch.Tensor | None = None, agent: torch.Tensor | None = None, text: torch.Tensor | None = None
    ) -> bool:
        """Write provided tokens before the initial-token refill ([PP] lm.py:741-802).

        At offset 0, delay-0 values are overwritten immediately. Delay-1
        values survive until the offset-1 refill. Missing streams are sampled.
        """
        if self._prepared:
            raise RuntimeError("commit the prepared step before preparing another")
        batch = self.cache.shape[0]
        # Validate all inputs before mutating state.
        for name, tokens in (("user", user), ("agent", agent)):
            if tokens is not None:
                self._validate(name, tokens, (batch, AUDIO_CODEBOOKS_PER_STREAM, 1))
        if text is not None:
            self._validate("text", text, (batch,))
        for start, tokens in ((1 + AUDIO_CODEBOOKS_PER_STREAM, user), (1, agent)):
            if tokens is None:
                continue
            for q in range(AUDIO_CODEBOOKS_PER_STREAM):
                k = start + q
                column = (self.offset + DELAYS[k]) % CACHE_SIZE
                self.cache[:, k, column] = tokens[:, q, 0]
                self.provided[:, k, column] = True
        if text is not None:
            column = (self.offset + DELAYS[0]) % CACHE_SIZE
            self.cache[:, 0, column] = text
            self.provided[:, 0, column] = True
        for k, delay in enumerate(DELAYS):
            if self.offset <= delay:
                self.cache[:, k, self.offset % CACHE_SIZE] = self.initial[:, k, 0]
                self.provided[:, k, self.offset % CACHE_SIZE] = True
        if self.offset == 0:
            self.cache[:, :, 0] = self.initial[:, :, 0]
            self.offset += 1
            return False
        self._prepared = True
        return True

    def prepare_embedding(self) -> None:
        """Prepare dummy tokens exactly as step_embeddings ([PP] lm.py:853-863).

        The first embedding consumes two dummy prepares; later ones consume
        one. Embeddings themselves belong to the temporal transformer.
        """
        dummy = self.initial[:, 1 : 1 + AUDIO_CODEBOOKS_PER_STREAM]
        text = torch.full_like(self.initial[:, 0, 0], TEXT_PAD_TOKEN_ID)
        while not self.prepare(user=dummy, agent=dummy, text=text):
            pass

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare a forward step first")

    @property
    def model_input(self) -> torch.Tensor:
        self._require_prepared()
        column = (self.offset - 1) % CACHE_SIZE
        return self.cache[:, :, column : column + 1]

    @property
    def target(self) -> torch.Tensor:
        self._require_prepared()
        column = self.offset % CACHE_SIZE
        return self.cache[:, :, column : column + 1]

    @property
    def target_provided(self) -> torch.Tensor:
        self._require_prepared()
        column = self.offset % CACHE_SIZE
        return self.provided[:, :, column : column + 1]

    def target_all_provided(self) -> bool:
        return bool(self.target_provided.all())

    def commit(self, sampled_text: torch.Tensor, sampled_audio: torch.Tensor) -> torch.Tensor | None:
        """Preserve forced values, clear the consumed mask and gather ([PP] lm.py:899-952).

        Gather mixes delay-0 tokens from offset-1 and delay-1 from offset.
        There is no final flush: the last delay-0 samples are never emitted.
        """
        self._require_prepared()
        batch = self.cache.shape[0]
        self._validate("sampled_text", sampled_text, (batch,))
        self._validate("sampled_audio", sampled_audio, (batch, NUM_AUDIO_CODEBOOKS))
        input_column = (self.offset - 1) % CACHE_SIZE
        target_column = self.offset % CACHE_SIZE
        self.provided[:, :, input_column] = False
        samples = torch.cat((sampled_text[:, None], sampled_audio), dim=1)
        self.cache[:, :, target_column] = torch.where(
            self.provided[:, :, target_column], self.cache[:, :, target_column], samples
        )
        output = None
        if self.offset > MAX_DELAY:
            columns = ((self.offset - MAX_DELAY + self._delays) % CACHE_SIZE).view(1, -1, 1)
            output = self.cache.gather(2, columns.expand(batch, -1, 1))
        self.offset += 1
        self._prepared = False
        return output

    def state_tensor(self, step: int) -> torch.Tensor:
        """Pack cache, mask, offset and a walk step counter as ``[B, 2*17*4 + 2]`` int64.

        This is the LM's ``lm_state`` graph edge; a prepared-but-uncommitted
        cache is never packed.
        """
        if self._prepared:
            raise RuntimeError("commit the prepared step before packing the cache")
        batch = self.cache.shape[0]
        scalars = torch.tensor([self.offset, step], dtype=torch.long, device=self.cache.device).expand(batch, 2)
        return torch.cat((self.cache.flatten(1), self.provided.long().flatten(1), scalars), dim=1)

    @classmethod
    def from_state_tensor(cls, state: torch.Tensor) -> tuple["DelayedTokenCache", int]:
        """Inverse of ``state_tensor``; returns the cache and the step counter."""
        size = NUM_CODEBOOKS * CACHE_SIZE
        if state.dim() != 2 or state.shape[1] != 2 * size + 2 or state.dtype != torch.long:
            raise ValueError(f"lm_state must be int64 [B, {2 * size + 2}], got {state.dtype} {tuple(state.shape)}")
        scalars = state[:, 2 * size :]
        if not torch.equal(scalars, scalars[:1].expand_as(scalars)) or bool((scalars < 0).any()):
            raise ValueError("lm_state rows disagree on offset/step or hold negative values")
        cache = cls(state.shape[0], device=state.device)
        cache.cache.copy_(state[:, :size].view_as(cache.cache))
        provided = state[:, size : 2 * size]
        if bool(((provided != 0) & (provided != 1)).any()):
            raise ValueError("lm_state provided mask must be 0/1")
        cache.provided.copy_(provided.view_as(cache.provided).bool())
        cache.offset = int(scalars[0, 0])
        return cache, int(scalars[0, 1])

    def load(self, cache: torch.Tensor) -> None:
        """Restore only voice tokens ([PP] lm.py:1031-1038), retaining offset/mask.

        This is the reference .pt replay operation, not a full state restore.
        The preset batch must match the live batch. Broadcasting a real B=1
        preset into a B=2 cache is not supported; the served path runs batch size 1.
        """
        if self._prepared:
            raise RuntimeError("commit the prepared step before loading a voice cache")
        self._validate("voice cache", cache, tuple(self.cache.shape))
        self.cache.copy_(cache)
