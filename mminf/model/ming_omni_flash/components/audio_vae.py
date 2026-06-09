"""AudioVAE for Ming-flash-omni-2.0 (step 6d).

Self-contained port of vllm-omni's
``vllm_omni/model_executor/models/ming_flash_omni/audio_vae.py`` (392 LOC).
The released ckpt ships the VAE under ``talker/vae/model.safetensors``
with the top-level prefixes ``encoder.*`` and ``decoder.*``; we mirror
the upstream module tree so the eventual loader is a plain prefix-strip
+ load_state_dict.

Topology (released ckpt):

  AudioVAE
    .encoder (Encoder)                            # waveform → latent
      .encoder (Qwen2Model, sliding-window=64)    # main backbone
      .aggregator (Qwen2Model, 4 layers)          # patch-summarisation
      .fc1 (Linear 882 → 896)
      .fc2 (Linear 896 → 896)
      .fc3 (Linear 896 → 128)                     # latent_dim*2 (mean+scale)
      .norm (LayerNorm 896)
      .cls_embed (Parameter (1, 1, 896))
    .decoder (Decoder)                            # latent → waveform
      .decoder (Qwen2Model, sliding-window=64)
      .fc1 (Linear 64 → 896)
      .head (ISTFTHead)
        .out (Linear 896 → 3530 = n_fft + 2)
        .istft (ISTFT, n_fft=3528, hop=882, win=3528)
      .upsampling (StreamingLinearUpsample)       # only when patch_size != -1

Two simplifications vs vllm-omni:

  * `encode_latent` uses an inline `_oobleck_sample()` instead of
    `diffusers.OobleckDiagonalGaussianDistribution` — same math
    (mean/scale split, softplus on scale, reparameterised sample) but
    no diffusers dep.  The full diffusers class also exposes
    `kl_divergence` / `mode` for training; we only need `sample` at
    inference, so the minimal helper is enough.

  * `Decoder.low_level_reconstruct`'s streaming KV-cache fill path uses
    HF `Cache` instances; the upstream's `past_key_values` tuple
    fallback isn't needed on transformers >= 4.43.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ===========================================================================
# Inline Oobleck-style Gaussian sampler (replaces diffusers dep)
# ===========================================================================


def _oobleck_sample(parameters: torch.Tensor) -> torch.Tensor:
    """Sample from a diagonal Gaussian parameterised by ``[mean, scale]``.

    Matches the inference-time behaviour of
    ``diffusers.models.autoencoders.autoencoder_oobleck.OobleckDiagonalGaussianDistribution.sample``:

      mean, scale_raw = parameters.chunk(2, dim=1)
      scale = softplus(scale_raw) + 1e-4
      sample = mean + scale * eps

    Args:
        parameters: ``(B, 2 * latent_dim, T)`` tensor — first half is
            the mean, second half is the raw scale.

    Returns:
        ``(B, latent_dim, T)`` sample.
    """
    mean, scale_raw = parameters.chunk(2, dim=1)
    scale = F.softplus(scale_raw) + 1e-4
    eps = torch.randn_like(mean)
    return mean + scale * eps


# ===========================================================================
# ISTFT — inverse-STFT reconstruction with optional streaming buffers
# ===========================================================================


class _ISTFT(nn.Module):
    """Sliding-window OLA inverse STFT used by ISTFTHead.

    Two padding modes:

      * ``"center"`` — wraps ``torch.istft`` directly.
      * ``"same"`` — hand-rolled F.fold reconstruction so we can
        manage chunk boundaries via ``audio_buffer`` / ``window_buffer``
        (essential for the streaming decode path).

    The streaming variant preserves the trailing ``win_length - hop_length``
    samples of audio + window envelope across chunks so adjacent chunks
    sum-of-window-envelope-normalise correctly when concatenated.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ) -> None:
        super().__init__()
        if padding not in ("center", "same"):
            raise ValueError(f"Padding must be 'center' or 'same'; got {padding!r}.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.buffer_len = win_length - hop_length
        self.register_buffer("window", torch.hann_window(win_length))

    # ------------------------------------------------------------------
    # Per-chunk buffer plumbing
    # ------------------------------------------------------------------

    def _buffer_process(
        self,
        x: torch.Tensor,
        buffer: torch.Tensor | None,
        pad: int,
        last_chunk: bool,
        streaming: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply OLA buffering for the ``same`` padding mode.

        Non-streaming: trim ``pad`` samples off both ends.
        Streaming: add the previous chunk's tail into the current head;
        retain the new tail unless this is the last chunk (in which case
        trim ``pad`` off the end).
        """
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            else:
                x = x.clone()
                x[:, : self.buffer_len] = x[:, : self.buffer_len] + buffer
            buffer = x[:, -self.buffer_len :]
            if not last_chunk:
                x = x[:, : -self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
        return x, buffer

    def forward(
        self,
        spec: torch.Tensor,
        audio_buffer: torch.Tensor | None = None,
        window_buffer: torch.Tensor | None = None,
        streaming: bool = False,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Inverse-STFT reconstruction.

        Args:
            spec: ``(B, n_fft//2 + 1, T)`` complex STFT magnitudes.

        Returns:
            Tuple of ``(waveform, audio_buffer, window_buffer)``.
            Buffers are None when ``streaming=False`` and the centre
            padding mode is in use.
        """
        if self.padding == "center":
            y = torch.istft(
                spec, self.n_fft, self.hop_length, self.win_length, self.window,
                center=True,
            )
            return y, None, None

        # same-padding path
        pad = (self.win_length - self.hop_length) // 2
        B, N, T = spec.shape

        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        output_size = (T - 1) * self.hop_length + self.win_length
        y = F.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        y, audio_buffer = self._buffer_process(
            y, audio_buffer, pad, last_chunk=last_chunk, streaming=streaming,
        )

        # Compute the per-position sum-of-window-squared so OLA averages
        # correctly. Same fold over a (1, T, win_length) tile of the
        # squared window.
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = (
            F.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )
            .squeeze(0)
            .squeeze(0)
        )
        window_envelope, window_buffer = self._buffer_process(
            window_envelope, window_buffer, pad,
            last_chunk=last_chunk, streaming=streaming,
        )
        window_envelope = window_envelope.squeeze()

        if not (window_envelope > 1e-11).all():
            raise RuntimeError(
                "ISTFT window envelope has near-zero positions; "
                "check hop_length / win_length / window choice."
            )
        y = y / window_envelope

        return y, audio_buffer, window_buffer


# ===========================================================================
# ISTFTHead — Linear → STFT magnitude/phase → ISTFT → waveform
# ===========================================================================


class _ISTFTHead(nn.Module):
    """Projects DiT hidden states to STFT mag+phase then runs an ISTFT.

    Output Linear emits ``n_fft + 2`` channels; the first half is the
    log-magnitude (exp'd + clipped to 1e2) and the second half is the
    phase. Reassembled as a complex spectrogram for the ISTFT.
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ) -> None:
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = _ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        audio_buffer: torch.Tensor | None = None,
        window_buffer: torch.Tensor | None = None,
        streaming: bool = False,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Returns ``(audio, x_pred, audio_buffer, window_buffer)``.

        ``audio`` is ``(B, 1, T_samples)``; ``x_pred`` is the raw
        (B, n_fft+2, T_frames) projection (useful for adversarial /
        spec-disc training paths; harmless at inference).
        """
        x_pred = self.out(x).transpose(1, 2)
        mag, phase = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        audio, audio_buffer, window_buffer = self.istft(
            spec, audio_buffer=audio_buffer, window_buffer=window_buffer,
            streaming=streaming, last_chunk=last_chunk,
        )
        return audio.unsqueeze(1), x_pred, audio_buffer, window_buffer


# ===========================================================================
# StreamingLinearUpsample — chunked linear upsample for patched latents
# ===========================================================================


class _StreamingLinearUpsample(nn.Module):
    """Linear upsampling that produces consistent output across chunks.

    Non-streaming: ``upsampler(x)`` directly.
    Streaming: defer emit until we have a 1-step lookahead so the
    upsample boundary matches the non-chunked result. Internal ``state``
    dict tracks: ``prev_chunk``, ``history_last`` (the last frame of the
    PREVIOUS prev_chunk, kept so the upsample window has left context),
    ``is_first``.
    """

    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(
            scale_factor=scale_factor, mode="linear", align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor | None,
        state: dict[str, Any] | None = None,
        is_last: bool = False,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        if state is None:
            state = {"prev_chunk": None, "history_last": None, "is_first": True}

        if x is None and not is_last:
            return None, state

        # Single-chunk fast path: first AND last.
        if state["is_first"] and is_last:
            out = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            return out, None

        output_chunks: list[torch.Tensor] = []

        if state["is_first"]:
            state["prev_chunk"] = x
            state["is_first"] = False
            if not is_last:
                return None, state

        # Emit the deferred prev_chunk now that we have a right lookahead.
        if state["prev_chunk"] is not None:
            p = state["prev_chunk"].transpose(1, 2)
            if state["history_last"] is None:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([p, lookahead], dim=2)
                up = self.upsampler(inp)
                out_prev = up[:, :, : p.size(2) * self.scale_factor]
            else:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([state["history_last"], p, lookahead], dim=2)
                up = self.upsampler(inp)
                start = self.scale_factor
                end = start + p.size(2) * self.scale_factor
                out_prev = up[:, :, start:end]
            output_chunks.append(out_prev.transpose(1, 2))
            state["history_last"] = p[:, :, -1:]
            state["prev_chunk"] = x

        if is_last:
            p = state["prev_chunk"].transpose(1, 2)
            inp = torch.cat([state["history_last"], p], dim=2)
            up = self.upsampler(inp)
            out_last = up[:, :, self.scale_factor :]
            output_chunks.append(out_last.transpose(1, 2))
            state = None

        final = torch.cat(output_chunks, dim=1) if output_chunks else None
        return final, state


# ===========================================================================
# Encoder / Decoder (Qwen2-backed)
# ===========================================================================


def _build_vae_qwen2_config(backbone: dict, attn_implementation: str) -> "Qwen2Config":
    """Build a Qwen2Config from the VAE backbone dict, stripping fields HF doesn't accept."""
    from transformers import Qwen2Config
    # Drop fields that Qwen2Config doesn't accept as kwargs (HF would
    # store them as custom attrs, but cleaner to drop). `is_causal` is
    # the only field upstream adds that HF's Qwen2 ignores.
    accepted = {
        k: v for k, v in backbone.items()
        if k not in ("is_causal", "transformers_version", "torch_dtype",
                     "_attn_implementation", "_attn_implementation_autoset",
                     "attn_implementation", "model_type", "architectures")
    }
    cfg = Qwen2Config(**accepted, attn_implementation=attn_implementation)
    return cfg


def _resolve_attn_implementation() -> str:
    """Prefer FA2 when available; else sdpa."""
    try:
        from transformers.utils import is_flash_attn_2_available
        return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except Exception:
        return "sdpa"


class _Decoder(nn.Module):
    """Latent → waveform via Qwen2 backbone + ISTFTHead.

    Module-tree mirrors upstream so the released ckpt's
    ``decoder.decoder.layers.N.*`` (Qwen2Model), ``decoder.fc1``,
    ``decoder.head.out``, ``decoder.head.istft.window`` keys all
    land via plain state-dict equality.
    """

    def __init__(
        self,
        decoder_args: dict,
        output_dim: int = 882,
        latent_dim: int = 64,
        patch_size: int = -1,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import Qwen2Model
        if attn_implementation is None:
            attn_implementation = _resolve_attn_implementation()
        cfg = _build_vae_qwen2_config(decoder_args, attn_implementation=attn_implementation)
        logger.info("AudioVAE Decoder: using attn_implementation=%r", cfg._attn_implementation)

        self.decoder = Qwen2Model(cfg)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hop_length = output_dim
        self.fc1 = nn.Linear(latent_dim, cfg.hidden_size)
        self.head = _ISTFTHead(
            dim=cfg.hidden_size,
            n_fft=self.hop_length * 4,
            hop_length=self.hop_length,
            padding="same",
        )
        self.patch_size = patch_size
        if self.patch_size != -1:
            self.upsampling = _StreamingLinearUpsample(scale_factor=patch_size)

    def low_level_reconstruct(
        self,
        x: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        stream_state: tuple[Any, Any, Any] = (None, None, None),
        last_chunk: bool = False,
    ):
        """Reconstruct ``(B, 1, T_samples)`` waveform from latent ``(B, T, latent_dim)``.

        Non-streaming path runs the full upsample + backbone + head.
        Streaming path threads ``stream_state = (upsample_state,
        audio_buffer, window_buffer)`` and the Qwen2 backbone's
        ``past_key_values`` across chunks; bridges the sliding-window
        boundary with the partial-fill trick from upstream when the
        first chunk would exceed ``sliding_window``.
        """
        upsample_state, audio_buffer, window_buffer = stream_state
        bsz, device, dtype = x.size(0), x.device, x.dtype
        x = self.fc1(x)
        if self.patch_size != -1:
            if use_cache:
                x, upsample_state = self.upsampling(
                    x, state=upsample_state, is_last=last_chunk,
                )
                if x is None:
                    stream_state = (upsample_state, audio_buffer, window_buffer)
                    return torch.empty(bsz, 1, 0, device=device, dtype=dtype), stream_state, past_key_values
            else:
                x = self.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)

        hidden_states_list: list[torch.Tensor] = []

        # Sliding-window bridge: when the cache is empty and this chunk
        # would push past `sliding_window`, fill the cache with the
        # first (sw_size - 1) tokens first so the second pass benefits
        # from the cached prefix.
        if use_cache and getattr(self.decoder.config, "sliding_window", None) is not None:
            sw_size = self.decoder.config.sliding_window
            target_len = sw_size - 1
            past_len = _get_past_len(past_key_values)
            curr_len = x.shape[1]
            if past_len < target_len and (past_len + curr_len) >= sw_size:
                fill_len = target_len - past_len
                x_fill = x[:, :fill_len, :]
                outputs = self.decoder(
                    inputs_embeds=x_fill, past_key_values=past_key_values, use_cache=True,
                )
                hidden_states_list.append(outputs.last_hidden_state)
                past_key_values = outputs.past_key_values
                x = x[:, fill_len:, :]

        outputs = self.decoder(
            inputs_embeds=x, past_key_values=past_key_values, use_cache=use_cache,
        )
        hidden_states_list.append(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values

        full_hidden = (
            torch.cat(hidden_states_list, dim=1)
            if len(hidden_states_list) > 1
            else hidden_states_list[0]
        )
        x_out, _x_pred, audio_buffer, window_buffer = self.head(
            full_hidden,
            streaming=use_cache,
            audio_buffer=audio_buffer,
            window_buffer=window_buffer,
            last_chunk=last_chunk,
        )
        stream_state = (upsample_state, audio_buffer, window_buffer)
        return x_out, stream_state, past_key_values


def _get_past_len(past_key_values) -> int:
    """Recover past-seq-len across the various HF cache shapes."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        return int(past_key_values[0][0].shape[-2])
    return 0


class _Encoder(nn.Module):
    """Waveform → latent via Qwen2 backbone + optional patch aggregator.

    With ``patch_size != -1`` the encoder runs a second short Qwen2
    backbone (4 layers) over each patch concatenated with a learnable
    [CLS] embedding and outputs the [CLS] row only — same shape as
    the Aggregator (`components/talker_dit.Aggregator`) but inside the
    VAE encoder rather than at the talker output.
    """

    def __init__(
        self,
        encoder_args: dict,
        input_dim: int = 882,
        hop_size: int = 882,
        latent_dim: int = 64,
        patch_size: int = -1,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import Qwen2Model
        if attn_implementation is None:
            attn_implementation = _resolve_attn_implementation()
        cfg = _build_vae_qwen2_config(encoder_args, attn_implementation=attn_implementation)
        logger.info("AudioVAE Encoder: using attn_implementation=%r", cfg._attn_implementation)

        self.encoder = Qwen2Model(cfg)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, cfg.hidden_size, bias=False)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fc3 = nn.Linear(cfg.hidden_size, latent_dim * 2)
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.patch_size = patch_size
        if patch_size != -1:
            # Aggregator is a 4-layer Qwen2 backbone (upstream
            # explicitly overrides num_hidden_layers to 4).
            agg_cfg = _build_vae_qwen2_config(
                {**encoder_args, "num_hidden_layers": 4},
                attn_implementation=attn_implementation,
            )
            self.aggregator = Qwen2Model(agg_cfg)
            # Learnable CLS embedding prepended to each patch.
            self.cls_embed = nn.Parameter(torch.empty(1, 1, cfg.hidden_size))
            # Match upstream's normal_(0, 0.02) init so eager-init
            # weights match if the loader is bypassed in tests.
            nn.init.normal_(self.cls_embed, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Waveform → frames windowed slicing
    # ------------------------------------------------------------------

    def get_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Slide a ``(input_dim,)`` window over the waveform with stride hop_size.

        Pads the right edge so the final window doesn't overshoot.
        Returns ``(B, num_frames, input_dim)``.
        """
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        padding_needed = expected_len - x.size(-1)
        waveform = F.pad(x, (0, padding_needed), value=0.0)
        frames = waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)
        return frames

    def pad_patch_insert_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Group frames into patches of ``patch_size`` and append a CLS row to each."""
        bsz, num_frame, dim = x.size()
        r = num_frame % self.patch_size
        pad_num = self.patch_size - r if r else 0
        x = F.pad(x, (0, 0, 0, pad_num), value=0.0)
        x = x.reshape(-1, self.patch_size, dim)
        cls = self.cls_embed.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls), dim=1)
        x = x.reshape(bsz, -1, dim)
        return x

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(latent_params, waveform.unsqueeze(1))``.

        ``latent_params`` is ``(B, T_latents, latent_dim*2)`` — the
        first half is the Gaussian mean and the second half is the
        raw scale; pass through `_oobleck_sample` to draw a latent.
        """
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        h = self.encoder(inputs_embeds=x).last_hidden_state

        if self.patch_size != -1:
            h = self.pad_patch_insert_cls(h)
            h = self.aggregator(inputs_embeds=h).last_hidden_state
            bsz, _, dim = h.size()
            h = h.reshape(-1, self.patch_size + 1, dim)
            h = h[:, -1:, :].reshape(bsz, -1, dim)

        h = self.fc3(h)
        return h, waveform.unsqueeze(1)


# ===========================================================================
# AudioVAE — wraps Encoder + Decoder
# ===========================================================================


class AudioVAE(nn.Module):
    """Top-level Audio VAE.

    Plain nn.Module (not PreTrainedModel) so we don't inherit HF
    config machinery — the dataclass `AudioVAEConfig` carries the dims
    and the loader handles weights directly.
    """

    def __init__(
        self,
        audio_vae_config,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        self.config = audio_vae_config
        self.encoder = _Encoder(
            encoder_args=audio_vae_config.enc_backbone,
            input_dim=audio_vae_config.encoder_input_dim,
            hop_size=audio_vae_config.encoder_hop_size,
            latent_dim=audio_vae_config.latent_dim,
            patch_size=audio_vae_config.patch_size,
            attn_implementation=attn_implementation,
        )
        self.decoder = _Decoder(
            decoder_args=audio_vae_config.dec_backbone,
            output_dim=audio_vae_config.decoder_output_dim,
            latent_dim=audio_vae_config.latent_dim,
            patch_size=audio_vae_config.patch_size,
            attn_implementation=attn_implementation,
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def encode_latent(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and sample a latent (Gaussian re-parameterised).

        Returns ``(latent, frame_num)``.  ``latent`` is
        ``(B, latent_dim, T_latents)``; ``frame_num`` is the per-clip
        latent count after patching.
        """
        frame_num = torch.ceil(
            waveform_length / self.config.encoder_input_dim,
        ).to(torch.int32)
        if self.config.patch_size != -1:
            frame_num = torch.ceil(frame_num / self.config.patch_size)
        h, _y = self.encoder(waveform)
        # encoder.fc3 emits (B, T, latent_dim*2) — transpose to channels-second
        # for `_oobleck_sample` (chunks on dim=1).
        h = h.transpose(1, 2)
        latent = _oobleck_sample(h)
        latent = latent.transpose(1, 2)
        return latent, frame_num

    def decode(
        self,
        latent: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        stream_state: tuple[Any, Any, Any] = (None, None, None),
        last_chunk: bool = False,
    ):
        """Decode latent → waveform; threads the streaming state for chunked TTS."""
        waveform, stream_state, past_key_values = self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )
        return waveform, stream_state, past_key_values


def build_audio_vae(
    audio_vae_config,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    attn_implementation: str | None = None,
) -> AudioVAE:
    """Construct an `AudioVAE` from `AudioVAEConfig`.

    ``attn_implementation`` defaults to ``"sdpa"`` on CPU and FA2 when
    flash-attn is importable AND the target device is CUDA. Caller can
    pin to ``"eager"`` for debugging or ``"sdpa"`` to mirror what
    vllm-omni's talker actually uses at runtime (it forces sdpa on the
    talker LLM regardless of FA2 availability).
    """
    if attn_implementation is None:
        device_str = str(device)
        if device_str == "cpu" or device_str.startswith("cpu"):
            attn_implementation = "sdpa"
        else:
            attn_implementation = _resolve_attn_implementation()
    vae = AudioVAE(audio_vae_config, attn_implementation=attn_implementation)
    vae = vae.to(dtype=dtype, device=device)
    vae.eval()
    return vae


__all__ = ["AudioVAE", "build_audio_vae"]
