"""S3Gen: S3 speech tokens -> mel (flow matching) -> 24 kHz waveform (HiFT),
conditioned on a reference utterance.

Reference: ``chatterbox/models/s3gen/s3gen.py`` (``S3Token2Mel`` /
``S3Token2Wav``) and ``chatterbox/models/s3gen/flow.py``
(``CausalMaskedDiffWithXvec.inference``). The S3 tokenizer that turns the
reference audio into prompt tokens is a separate component; callers pass its
output in.

Frame bookkeeping a caller must respect:

* one token is ``token_mel_ratio`` (2) mel frames and ``2 * 480`` samples;
* the flow decoder is run over ``[prompt tokens | new tokens]`` and the prompt
  mel is written into the ``cond`` channels, so the generated mel starts at
  frame ``2 * len(prompt)``; only those frames are returned;
* the token encoder looks ``pre_lookahead_len`` (3) tokens ahead, so the last
  three tokens of a partial sequence are not final;
* ``mel_to_wav`` zeroes the first ``trim_fade_frames`` samples and fades the
  next ``trim_fade_frames`` in; apply it to the head of an utterance only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from mstar.model.chatterbox.components.audio import MelSpectrogram24k, kaldi_fbank_80
from mstar.model.chatterbox.components.s3gen_cfm import CausalConditionalCFM, ConditionalDecoder
from mstar.model.chatterbox.components.s3gen_flow import FlowTokenEncoder, lengths_to_mask
from mstar.model.chatterbox.components.s3gen_graphs import SolveGraphs
from mstar.model.chatterbox.components.s3gen_hift import HiFTGenerator
from mstar.model.chatterbox.components.s3gen_xvector import CAMPPlus
from mstar.model.chatterbox.config import S3GenConfig
from mstar.model.chatterbox.loader import fold_weight_norm, load_component


@dataclass
class ReferenceConditioning:
    """What S3Gen needs from a reference utterance."""

    prompt_tokens: torch.Tensor  # [1, N] long, S3 tokens of the reference (16 kHz)
    prompt_feat: torch.Tensor    # [1, 2N, 80] float, mel of the reference (24 kHz)
    embedding: torch.Tensor      # [1, 192] float, CAMPPlus x-vector

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "ReferenceConditioning":
        return ReferenceConditioning(
            prompt_tokens=self.prompt_tokens.to(device),
            prompt_feat=self.prompt_feat.to(device, dtype) if dtype else self.prompt_feat.to(device),
            embedding=self.embedding.to(device, dtype) if dtype else self.embedding.to(device),
        )

    @property
    def num_prompt_tokens(self) -> int:
        return int(self.prompt_tokens.shape[1])


@dataclass
class FlowRow:
    """One request's share of a batched flow solve: its tokens so far, its
    reference, whether the sequence is complete (else the look-ahead tokens are
    provisional) and, for a stream, the fixed noise field it was started with
    (``[1, 80, >= 2(N + n)]``) so every chunk denoises the same draw. Rows
    without a field draw fresh noise from ``generator`` (the offline path)."""
    tokens: torch.Tensor  # [n] long, S3 speech tokens
    ref: ReferenceConditioning
    finalize: bool = True
    noise: torch.Tensor | None = None
    generator: torch.Generator | None = None


class S3Gen(nn.Module):
    def __init__(self, config: S3GenConfig):
        super().__init__()
        self.config = config
        self.mel_extractor = MelSpectrogram24k(config.mel)
        self.speaker_encoder = CAMPPlus(config.xvector)
        self.flow_encoder = FlowTokenEncoder(config)
        self.decoder = CausalConditionalCFM(
            config.cfm, ConditionalDecoder(config.estimator, meanflow=config.meanflow),
        )
        self.vocoder = HiFTGenerator(config.hift)
        # optional CUDA-graph replay of whole solves (``enable_graphs``)
        self.solver: SolveGraphs | None = None

        n_trim = config.trim_fade_frames
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.flow_encoder.encoder_proj.weight.dtype

    # ------------------------------------------------------------------
    # Flow solve dispatch
    # ------------------------------------------------------------------

    def solve(
        self, mu: torch.Tensor, mask: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor,
        noise: torch.Tensor, n_timesteps: int,
    ) -> torch.Tensor:
        """The decoder's Euler solve for this variant (mean flow or CFG)."""
        if self.config.meanflow:
            return self.decoder.solve_meanflow(mu, mask, spks, cond, noise, n_timesteps)
        return self.decoder.solve(mu, mask, spks, cond, noise, n_timesteps)

    def _run_solve(self, mu, mask, spks, cond, noise, n_timesteps: int) -> torch.Tensor:
        if self.solver is not None:
            return self.solver(mu, mask, spks, cond, noise, n_timesteps)
        return self.solve(mu, mask, spks, cond, noise, n_timesteps)

    def enable_graphs(self, rows: Sequence[int] = (1, 2, 4, 8), max_graphs: int = 64) -> "SolveGraphs":
        """Replay whole flow solves from CUDA graphs, one per (rows, frames, steps)."""
        self.solver = SolveGraphs(self.solve, rows=rows, max_graphs=max_graphs)
        return self.solver

    @property
    def device(self) -> torch.device:
        return self.flow_encoder.encoder_proj.weight.device

    # ------------------------------------------------------------------
    # Reference conditioning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_reference(
        self,
        ref_wav_24k: torch.Tensor,
        ref_wav_16k: torch.Tensor,
        ref_tokens_16k: torch.Tensor,
    ) -> ReferenceConditioning:
        """Mel of the 24 kHz reference, x-vector of the 16 kHz one, and the
        reference tokens trimmed so that ``mel frames == 2 * tokens``."""
        if ref_wav_24k.dim() == 1:
            ref_wav_24k = ref_wav_24k[None]
        if ref_wav_16k.dim() == 1:
            ref_wav_16k = ref_wav_16k[None]
        ref_wav_24k = ref_wav_24k.to(self.device, self.dtype)
        ref_mels = self.mel_extractor(ref_wav_24k).transpose(1, 2)  # [1, F, 80]

        fbank = kaldi_fbank_80(ref_wav_16k.to(self.device, torch.float32), self.config.xvector.feat_dim)
        embedding = self.speaker_encoder(fbank[None].to(self.dtype))

        tokens = torch.atleast_2d(ref_tokens_16k).to(self.device)
        n = min(int(tokens.shape[1]), int(ref_mels.shape[1]) // self.config.token_mel_ratio)
        tokens = tokens[:, :n]
        ref_mels = ref_mels[:, : n * self.config.token_mel_ratio]
        return ReferenceConditioning(prompt_tokens=tokens, prompt_feat=ref_mels, embedding=embedding)

    # ------------------------------------------------------------------
    # Tokens -> mel
    # ------------------------------------------------------------------

    def _draw_noise(
        self, batch: int, prompt_len: int, gen_len: int, generator: torch.Generator | None,
    ) -> torch.Tensor:
        """The reference's noise, drawn in its order: the mean-flow path
        samples the generated frames first, then a full-length field whose
        tail it replaces; the standard path samples one full-length field."""
        shape = (batch, self.config.output_size, prompt_len + gen_len)
        kwargs = dict(dtype=self.dtype, device=self.device, generator=generator)
        if not self.config.meanflow:
            return torch.randn(shape, **kwargs)
        generated = torch.randn((batch, self.config.output_size, gen_len), **kwargs)
        z = torch.randn(shape, **kwargs)
        z[..., prompt_len:] = generated
        return z

    @torch.no_grad()
    def tokens_to_mel(
        self,
        tokens: torch.Tensor,
        token_lens: torch.Tensor,
        ref: ReferenceConditioning,
        *,
        n_timesteps: int | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        finalize: bool = True,
    ) -> torch.Tensor:
        """``tokens`` ``[B, L]`` (right-padded to ``token_lens``) -> mel ``[B, 80, 2L]``
        of the generated part only (padding frames are zero).

        ``finalize=False`` is the streaming call on a sequence that will still
        grow: the encoder's look-ahead makes the last ``pre_lookahead_len``
        tokens provisional, so their ``2 * pre_lookahead_len`` frames are cut
        before the flow solve and the mel returned is ``[B, 80, 2(L - 3)]``
        (reference ``CausalMaskedDiffWithXvec.inference``). ``noise`` then must
        cover the full ``2(N + L)`` frames; its head is used.
        """
        n_timesteps = n_timesteps or self.config.cfm.n_timesteps
        batch = tokens.shape[0]
        ref = ref.to(self.device, self.dtype)
        prompt_tokens = ref.prompt_tokens.expand(batch, -1)
        prompt_len = prompt_tokens.shape[1]
        full_tokens = torch.cat([prompt_tokens, tokens.to(self.device)], dim=1)
        full_lens = token_lens.to(self.device) + prompt_len

        spk = self.flow_encoder.project_speaker(ref.embedding.expand(batch, -1))
        mu, h_masks = self.flow_encoder(full_tokens, full_lens)
        h_lens = h_masks.sum(dim=-1).squeeze(-1)
        if not finalize:
            cut = self.config.encoder.pre_lookahead_len * self.config.token_mel_ratio
            mu = mu[:, :, :-cut]
            h_lens = (h_lens - cut).clamp_min(0)
        mel_prompt = prompt_len * self.config.token_mel_ratio
        total = mu.shape[-1]

        cond = torch.zeros(batch, self.config.output_size, total, device=self.device, dtype=mu.dtype)
        cond[:, :, :mel_prompt] = ref.prompt_feat.expand(batch, -1, -1).transpose(1, 2)
        mask = lengths_to_mask(h_lens, total).unsqueeze(1).to(mu.dtype)

        if noise is None:
            noise = self._draw_noise(batch, mel_prompt, total - mel_prompt, generator)
        elif noise.shape[-1] != total:
            noise = noise[:, :, :total]
        mel = self._run_solve(mu, mask, spk, cond, noise, n_timesteps)
        return mel[:, :, mel_prompt:]

    @torch.no_grad()
    def tokens_to_mel_rows(
        self, rows: Sequence[FlowRow], *, n_timesteps: int | None = None, frame_bucket: int = 0,
    ) -> list[torch.Tensor]:
        """Several requests, each with its own reference, look-ahead state and
        noise, solved as one right-padded batch. Returns each row's generated
        mel ``[1, 80, 2 * usable]`` (``usable = n`` when final, ``n - 3`` while
        streaming), the same frames ``tokens_to_mel`` produces for it alone.

        Right padding is invisible to the valid frames: the encoder zeroes the
        padding before its look-ahead convolution and masks attention, and the
        estimator's convolutions are causal or masked, so a row's result does
        not depend on the longer rows it shares the batch with. ``frame_bucket``
        pads the solve to a multiple of that many frames for the same reason,
        so a compiled or graph-captured estimator meets few distinct shapes.
        """
        n_timesteps = n_timesteps or self.config.cfm.n_timesteps
        ratio = self.config.token_mel_ratio
        lookahead = self.config.encoder.pre_lookahead_len * ratio
        refs = [row.ref.to(self.device, self.dtype) for row in rows]
        full = [
            torch.cat([ref.prompt_tokens[0], row.tokens.to(self.device, torch.long).reshape(-1)])
            for ref, row in zip(refs, rows, strict=True)
        ]
        full_lens = torch.tensor([f.numel() for f in full], dtype=torch.long, device=self.device)
        tokens = torch.nn.utils.rnn.pad_sequence(full, batch_first=True)

        spk = self.flow_encoder.project_speaker(torch.cat([ref.embedding for ref in refs], dim=0))
        mu, h_masks = self.flow_encoder(tokens, full_lens)
        cuts = torch.tensor([0 if row.finalize else lookahead for row in rows], device=self.device)
        h_lens = (h_masks.sum(dim=-1).squeeze(-1) - cuts).clamp_min(0)
        total = mu.shape[-1]
        if frame_bucket > 0 and total % frame_bucket:
            padded = -(-total // frame_bucket) * frame_bucket
            mu = torch.nn.functional.pad(mu, (0, padded - total))
            total = padded

        cond = torch.zeros(len(rows), self.config.output_size, total, device=self.device, dtype=mu.dtype)
        noise = torch.zeros_like(cond)
        for i, (ref, row) in enumerate(zip(refs, rows, strict=True)):
            mel_prompt = ref.num_prompt_tokens * ratio
            valid = int(h_lens[i])
            cond[i, :, :mel_prompt] = ref.prompt_feat[0].transpose(0, 1)
            if row.noise is not None:
                noise[i, :, :valid] = row.noise[0, :, :valid].to(noise.dtype)
            else:
                noise[i, :, :valid] = self._draw_noise(1, mel_prompt, valid - mel_prompt, row.generator)[0]
        mask = lengths_to_mask(h_lens, total).unsqueeze(1).to(mu.dtype)
        mel = self._run_solve(mu, mask, spk, cond, noise, n_timesteps)
        return [
            mel[i : i + 1, :, ref.num_prompt_tokens * ratio : int(h_lens[i])]
            for i, ref in enumerate(refs)
        ]

    # ------------------------------------------------------------------
    # Mel -> waveform
    # ------------------------------------------------------------------

    @torch.no_grad()
    def vocode(
        self,
        mel: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        cache_source: torch.Tensor | None = None,
        fade_in: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, 80, T]`` mel -> (waveform ``[B, 480T]``, harmonic source ``[B, 1, 480T]``).

        ``cache_source`` (``[B, 1, n]``) replaces the head of the freshly drawn
        excitation so a chunk continues the previous chunk's harmonics
        (reference ``HiFTGenerator.inference(cache_source=...)``). ``fade_in``
        applies the utterance-head trim/fade and belongs on the first chunk only.
        """
        wav, source = self.vocoder.vocode(mel.to(self.dtype), generator=generator, cache_source=cache_source)
        if fade_in:
            n = self.trim_fade.shape[0]
            wav[:, :n] = wav[:, :n] * self.trim_fade
        return wav, source

    @torch.no_grad()
    def mel_to_wav(
        self, mel: torch.Tensor, *, generator: torch.Generator | None = None, fade_in: bool = True,
    ) -> torch.Tensor:
        return self.vocode(mel, generator=generator, fade_in=fade_in)[0]

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    @staticmethod
    def _remap(name: str) -> str | None:
        if name.startswith("tokenizer."):
            return None
        if name.startswith("flow.decoder.estimator."):
            return "decoder.estimator." + name[len("flow.decoder.estimator."):]
        if name.startswith("flow."):
            return "flow_encoder." + name[len("flow."):]
        if name.startswith("mel2wav."):
            return "vocoder." + name[len("mel2wav."):]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``s3gen*.safetensors`` (``flow.*``, ``mel2wav.*``,
        ``speaker_encoder.*``; the tokenizer's keys are another component's)."""
        return load_component(
            self, fold_weight_norm(weights), component="S3Gen", name_remapper=self._remap,
        )
