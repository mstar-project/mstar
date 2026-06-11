"""TalkerGenerator: orchestrates Qwen2 + CFM + Aggregator + AudioVAE (step 6e-1).

Port of vllm-omni's ``MingAudioGenerator`` (``talker_module.py:854-1146``)
plus the streaming-decode utilities (`silence_holder`,
`trim_trailing_silence`). Stateless across requests — one ``__init__``
binds the model components, then each call to `generate_latents` runs a
fresh per-request AR loop.

Skipped from upstream:
  * `CFMGraphExecutorPool` / `CFMGraphExecutor` — vllm-specific CUDA-graph
    batching infrastructure. We always run `cfm_sample_step` through the
    manual path; mstar's engine layer handles graph capture separately.
  * `build_tts_input` / `_looks_like_music_prompt` — prompt-construction
    helpers that go alongside the eventual `process_prompt` audio-out path.
    Lives in step 8 (TTS caption template).

The generator's outputs feed directly into the mstar graph wiring in
step 6e-2:
  * `generate_latents()` is what `TalkerSubmodule.forward` will call per
    request, returning the list of CFM-generated latent patches.
  * `decode_to_waveform()` is what the audio-output submodule will call
    to produce the final waveform tensor for `EMIT_TO_CLIENT`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from mstar.model.ming_omni_flash.components.talker_dit import (
    CFM,
    Aggregator,
    get_epss_timesteps,
)

if TYPE_CHECKING:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    from mstar.model.ming_omni_flash.components.audio_vae import AudioVAE
    from mstar.model.ming_omni_flash.config import TalkerConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# Streaming silence / trim utilities
# ===========================================================================


def trim_trailing_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    sil_th: float = 1e-3,
    tail_silence_s: float = 0.3,
) -> torch.Tensor:
    """Drop low-energy frames off the tail; keep a short trailing silence.

    Accepts 2-D ``(C, T)`` or 3-D ``(B, C, T)`` waveforms. Anything else
    is passed through unchanged (defensive: rather than raise, leave the
    output untouched so a misshaped tensor doesn't crash decode).
    """
    if waveform.numel() == 0:
        return waveform

    original_dim = waveform.dim()
    if original_dim == 3:
        speech = waveform[:, 0, :]
    elif original_dim == 2:
        speech = waveform
    else:
        return waveform

    frame_size = int(sample_rate * 0.1)
    frame_step = int(sample_rate * 0.1)
    if speech.shape[-1] < frame_size:
        keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
        trimmed = speech[..., :keep]
    else:
        num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
        cur_len = (num_frame - 1) * frame_step + frame_size
        speech = speech[..., :cur_len]
        spe_frames = speech.unfold(-1, frame_size, frame_step)
        scores = spe_frames.abs().mean(dim=-1)
        scores = scores.mean(dim=list(range(scores.dim() - 1)))
        idx = int(scores.shape[0]) - 1
        while idx >= 0 and scores[idx] <= sil_th:
            idx -= 1
        if idx < 0:
            keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
            trimmed = speech[..., :keep]
        else:
            non_sil_len = idx * frame_step + frame_size + int(tail_silence_s * sample_rate)
            non_sil_len = min(non_sil_len, speech.shape[-1])
            trimmed = speech[..., :non_sil_len]

    if original_dim == 3:
        return trimmed.unsqueeze(1)
    return trimmed


def silence_holder(
    speech: torch.Tensor,
    sample_rate: int,
    sil_cache: dict | None = None,
    last_chunk: bool = True,
    sil_th: float = 1e-3,
    last_sil: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Streaming silence holder used during chunked VAE decode.

    Buffers low-energy chunks until a non-silent frame arrives (or the
    stream ends), so the client doesn't see long silent runs that would
    later get trimmed anyway. ``sil_cache`` carries state across chunks:
    ``{"holder": [tensors], "buffer": [tensors]}``.

    Same algorithm as upstream's ``silence_holder``. The leading-silence
    holder lets you defer emission of long silent regions; the
    short-chunk buffer concatenates chunks smaller than one frame.
    """
    if speech.numel() == 0:
        return speech, sil_cache or {"holder": [], "buffer": []}

    frame_step = int(sample_rate * 0.1)
    frame_size = int(sample_rate * 0.1)
    if sil_cache is None:
        sil_cache = {"holder": [], "buffer": []}

    if sil_cache["buffer"]:
        speech = torch.cat([*sil_cache["buffer"], speech], dim=-1)
        sil_cache["buffer"] = []

    if speech.shape[-1] < frame_size:
        sil_cache["buffer"].append(speech)
        if last_chunk:
            out = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return out[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
    cur_len = (num_frame - 1) * frame_step + frame_size
    if speech.shape[-1] > cur_len:
        sil_cache["buffer"].append(speech[..., cur_len:])
        speech = speech[..., :cur_len]

    spe_frames = speech.unfold(-1, frame_size, frame_step)
    scores = spe_frames.abs().mean(dim=-1)
    scores = scores.mean(dim=list(range(scores.dim() - 1)))
    idx = int(scores.shape[0]) - 1
    while idx >= 0 and scores[idx] <= sil_th:
        idx -= 1

    if idx < 0:
        sil_cache["holder"].append(speech)
        if last_chunk:
            out = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return out[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    non_sil_len = idx * frame_step + frame_size
    if last_chunk:
        non_sil_len += int(last_sil * sample_rate)
    non_sil_len = min(non_sil_len, speech.shape[-1])
    speech_out = torch.cat([*sil_cache["holder"], speech[..., :non_sil_len]], dim=-1)
    sil_cache["holder"] = []
    if non_sil_len < speech.shape[-1]:
        sil_cache["holder"].append(speech[..., non_sil_len:])
    return speech_out, sil_cache


# ===========================================================================
# TalkerGenerator
# ===========================================================================


class TalkerGenerator:
    """Drives prefill → AR decode → VAE decode for a single TTS request.

    Stateless across requests: bind the model components once at
    construction, then each `generate_latents` / `decode_to_waveform`
    call runs a fresh per-request flow. The eventual `TalkerSubmodule`
    (step 6e-2) instantiates one per worker and calls into it once per
    request.

    Field naming mirrors upstream `MingAudioGenerator.__init__` so the
    eventual graph-walk wiring + tests can reference the same surface.
    """

    def __init__(
        self,
        talker_config: "TalkerConfig",
        llm: "Qwen2Model",
        cfm: CFM,
        aggregator: Aggregator,
        stop_head: nn.Module,
        audio_vae: "AudioVAE | None" = None,
        cfg_strength: float | None = None,
    ) -> None:
        self.config = talker_config
        self.llm = llm
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.audio_vae = audio_vae
        self.patch_size = talker_config.patch_size
        self.his_patch_size = talker_config.history_patch_size
        self.latent_dim = talker_config.vae.latent_dim
        self.cfg_strength = (
            cfg_strength if cfg_strength is not None else talker_config.cfg_strength
        )
        # Trailing latent frames prepended on each VAE-decode chunk so the
        # Qwen2 backbone sees enough context for FA2 to be happy.
        self._vae_decode_pad_frames = 32

    # ------------------------------------------------------------------
    # Step entry points (mirror upstream MingAudioGenerator)
    # ------------------------------------------------------------------

    def llm_step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        step: int,
        past_key_values=None,
        use_static_cache: bool,
    ) -> torch.Tensor:
        """Single Qwen2 forward step; returns the last hidden state row.

        On step 0 (or when no static cache is in use), call the LLM
        without an explicit `cache_position`. On subsequent decode
        steps with a `StaticCache`, supply `cache_position` so the
        cache knows where to write the new K/V.
        """
        if step == 0 or not use_static_cache:
            outputs = self.llm(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
        else:
            past_seen = int(past_key_values.get_seq_length())
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            outputs = self.llm(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                cache_position=cache_position,
            )
        return outputs.last_hidden_state[:, -1:, :]

    def cfm_sample_step(
        self,
        last_hidden_state: torch.Tensor,
        his_lat: torch.Tensor,
        *,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One CFM sampling step.

        Returns ``(gen_lat, next_inputs_embeds, stop_out)`` where:
          * `gen_lat`: ``(B, patch_size, latent_dim)`` — the new
            latent patch.
          * `next_inputs_embeds`: ``(B, 1, llm_hidden)`` — what to feed
            the LLM on the next step (Aggregator output).
          * `stop_out`: ``(B, 2)`` — softmaxed stop classifier output.
        """
        if cfg is None:
            cfg = self.cfg_strength

        bat_size, _, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.patch_size, z_dim),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        t = get_epss_timesteps(
            self.config.steps,
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        sde_rnd = torch.randn(
            (self.config.steps, *randn_tensor.shape),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        sde_args = torch.tensor(
            [cfg, sigma, temperature],
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )

        gen_lat = self.cfm.sample(last_hidden_state, his_lat, randn_tensor, t, sde_args, sde_rnd)
        inputs_embeds = self.aggregator(gen_lat)
        stop_out = self.stop_head(last_hidden_state[:, -1, :]).softmax(dim=-1)
        return gen_lat, inputs_embeds, stop_out

    # ------------------------------------------------------------------
    # AR generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_latents(
        self,
        inputs_embeds: torch.Tensor,
        *,
        prompt_wav_lat: torch.Tensor | None = None,
        min_new_token: int = 10,
        max_steps: int = 1000,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        use_static_cache: bool = True,
    ) -> list[torch.Tensor]:
        """AR loop: prefill → repeated (LLM step → CFM sample → stop check).

        Returns the list of per-step CFM-generated latent patches in
        emission order. Each entry is ``(B, patch_size, latent_dim)``;
        caller concatenates along dim=1 before feeding to `decode_to_waveform`
        for the one-shot decode path, or hands them in one at a time for
        the streaming path.
        """
        if cfg is None:
            cfg = self.cfg_strength
        device = next(self.llm.parameters()).device
        dtype = next(self.llm.parameters()).dtype

        his_lat = self._init_his_lat(prompt_wav_lat, device, dtype)
        past_key_values, max_cache_len = self._init_kv_cache(use_static_cache, device, dtype)
        prefill_len = inputs_embeds.shape[1]
        all_latents: list[torch.Tensor] = []

        steps_budget = min(max_steps, max_cache_len - prefill_len) if max_cache_len else max_steps
        for step in range(steps_budget):
            last_hs = self.llm_step(
                inputs_embeds,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=use_static_cache,
            )
            gen_lat, inputs_embeds, stop_out = self.cfm_sample_step(
                last_hs, his_lat, cfg=cfg, sigma=sigma, temperature=temperature,
            )
            his_lat = self._update_his_lat(his_lat, gen_lat)
            all_latents.append(gen_lat)

            stop_prob = float(stop_out[0, 1].detach().cpu().item())
            if step > min_new_token and stop_prob > 0.5:
                logger.debug("TalkerGenerator: stop at step=%d (prob=%.4f)", step, stop_prob)
                break

        return all_latents

    # ------------------------------------------------------------------
    # KV cache + history-latent bookkeeping
    # ------------------------------------------------------------------

    def _init_his_lat(
        self,
        prompt_wav_lat: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the initial history-latent buffer (shape (1, his_patch_size, latent_dim)).

        If `prompt_wav_lat` is supplied (e.g. voice-prompt conditioning),
        right-align it inside the his_patch_size window; otherwise the
        buffer starts as zeros.
        """
        his_lat = torch.zeros(
            1, self.his_patch_size, self.latent_dim, device=device, dtype=dtype,
        )
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat
        return his_lat

    def _init_kv_cache(
        self,
        use_static_cache: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object | None, int]:
        """Allocate a `StaticCache` for the Qwen2 LLM when requested.

        Returns ``(cache_or_None, max_cache_len)``. `StaticCache` is the
        upstream choice; matches what the released ckpt's serving path
        uses and lets us pass `cache_position` through `llm_step` on
        step > 0.
        """
        max_cache_len = 2048
        if not use_static_cache:
            return None, max_cache_len
        from transformers import Qwen2Config, StaticCache
        # Build a Qwen2Config from our TalkerLLMConfig dataclass so
        # StaticCache can read the layer / head dims it needs.
        llm_cfg = Qwen2Config(
            hidden_size=self.config.llm.hidden_size,
            num_hidden_layers=self.config.llm.num_hidden_layers,
            num_attention_heads=self.config.llm.num_attention_heads,
            num_key_value_heads=self.config.llm.num_key_value_heads,
            vocab_size=self.config.llm.vocab_size,
            max_position_embeddings=self.config.llm.max_position_embeddings,
        )
        cache = StaticCache(
            config=llm_cfg,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        return cache, max_cache_len

    def _update_his_lat(
        self, his_lat: torch.Tensor, gen_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Slide the his_patch_size window forward by patch_size."""
        if self.his_patch_size == self.patch_size:
            return gen_lat
        if self.his_patch_size > self.patch_size:
            return torch.cat(
                [his_lat[:, self.patch_size - self.his_patch_size:], gen_lat], dim=1,
            )
        raise NotImplementedError(
            f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})",
        )

    # ------------------------------------------------------------------
    # Duration cap heuristic (port of upstream `duration_capped_steps`)
    # ------------------------------------------------------------------

    def duration_capped_steps(
        self, text_len: int, requested_max_steps: int,
    ) -> int:
        """Cap requested max_steps by a duration heuristic.

        Mirrors upstream: each generation step yields
        ``(patch_size * vae_patch_size * vae_hop_length) / sample_rate``
        seconds of audio. The max-duration budget per turn is
        ``max(2.0, text_len * 5818/16000)`` seconds (the 5818/16000
        constant is a duration-per-token estimate matched against
        the released ckpt's prosody).
        """
        if self.audio_vae is None:
            return requested_max_steps
        sample_rate = float(self.audio_vae.config.sample_rate)
        vae_patch_size = float(self.audio_vae.config.patch_size)
        hop_size = float(self.audio_vae.decoder.hop_length)
        seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
        if seconds_per_step <= 0:
            return requested_max_steps
        max_duration_s = max(2.0, float(text_len) * (5818.0 / 16000.0))
        max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
        return min(requested_max_steps, max_steps_by_duration)

    # ------------------------------------------------------------------
    # Audio decode (one-shot + streaming)
    # ------------------------------------------------------------------

    def decode_to_waveform(
        self, latents: list[torch.Tensor], stream_decode: bool = True,
    ) -> torch.Tensor:
        """Decode latents → waveform via `AudioVAE.decode`.

        ``stream_decode=True`` runs the chunked path (matches the live
        serving topology where each CFM step's latent is decoded as it
        emits); False concatenates everything and runs one decode.
        """
        if self.audio_vae is None:
            raise RuntimeError("TalkerGenerator: audio_vae is None — cannot decode.")
        if not latents:
            device = next(self.llm.parameters()).device
            dtype = next(self.llm.parameters()).dtype
            return torch.zeros((1, 1, 0), device=device, dtype=dtype)

        if stream_decode:
            return self._stream_decode(latents)
        all_lat = torch.cat(latents, dim=1)
        waveform, _, _ = self.audio_vae.decode(
            all_lat, use_cache=False, stream_state=(None, None, None), last_chunk=True,
        )
        return waveform

    def _stream_decode(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Chunked VAE decode with sliding-window pad + silence holder."""
        sr = int(self.audio_vae.config.sample_rate)
        decode_pad: torch.Tensor | None = None
        sil_cache: dict | None = None
        wav_chunks: list[torch.Tensor] = []

        for i, lat in enumerate(latents):
            last_chunk = (i == len(latents) - 1)
            if decode_pad is not None:
                vae_input = torch.cat([decode_pad, lat], dim=1)
                pad_frames = decode_pad.shape[1]
            else:
                vae_input = lat
                pad_frames = 0

            speech, _, _ = self.audio_vae.decode(
                vae_input,
                use_cache=False,
                stream_state=(None, None, None),
                last_chunk=True,
            )
            total_frames = vae_input.shape[1]
            dcs = speech.shape[-1] // total_frames
            speech_chunk = speech[:, :, pad_frames * dcs:][0].detach().float()
            speech_chunk, sil_cache = silence_holder(
                speech_chunk, sr, sil_cache=sil_cache, last_chunk=last_chunk,
            )
            if speech_chunk.numel() > 0:
                wav_chunks.append(speech_chunk)
            decode_pad = vae_input[:, -self._vae_decode_pad_frames:, :].detach()

        if not wav_chunks:
            device = next(self.llm.parameters()).device
            dtype = next(self.llm.parameters()).dtype
            return torch.zeros((1, 1, 0), device=device, dtype=dtype)
        return torch.cat(wav_chunks, dim=-1).unsqueeze(0)

    def trim_trailing_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Tail-silence trim using the audio VAE's sample rate."""
        if self.audio_vae is None:
            return waveform
        return trim_trailing_silence(waveform, int(self.audio_vae.config.sample_rate))


__all__ = [
    "TalkerGenerator",
    "silence_holder",
    "trim_trailing_silence",
]
