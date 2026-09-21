"""Speaker (voice-clone) embedding extractor for Zonos2.

This is a port of ``../ZONOS2/python/zonos2/models/speaker_cloning.py``.

The Zonos2 checkpoint does not contain the encoder. ``model.pth`` holds only
the two layers downstream of it: ``speaker_lda_projection`` and
``speaker_projection``. The encoder is an external HF model whose architecture
lives in remote code on the hub, and it loads with ``trust_remote_code``. This
module is therefore a mel front-end and a thin ``AutoModel`` wrapper, as the
reference is.

The repo id says ``...-1.7B``, but the remote class is an
``EcapaTdnnSpeakerEncoder`` of about 12M parameters (about 48 MB in fp32). The
name refers to the model that it was distilled against. The encoder is small
enough to keep on the GPU next to the TTS model.

The mel front-end must agree with the reference exactly. A projection trained
against these features consumes the embedding. A different window, scale, or
padding therefore moves the vector away from the trained distribution. The
clone then degrades, but the code does not fail.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Qwen3SpeakerEncoder(nn.Module):
    """Convert reference audio into a speaker embedding.

    The result is a ``(1, embedding_dim)`` float32 tensor. ``embedding_dim`` is
    the model's ``speaker_embedding_dim``, which is 2048 for the released
    checkpoint. The backbone can return more than one candidate output, so this
    class selects the one with the expected width, as the reference consumers
    do.
    """

    TARGET_SAMPLE_RATE = 24_000
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 128
    F_MIN = 0.0
    F_MAX = 12_000.0

    def __init__(
        self,
        model_id: str,
        embedding_dim: int,
        cache_dir: str | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        import torchaudio
        from transformers import AutoModel

        self.model_id = model_id
        self.embedding_dim = int(embedding_dim)
        self.encoder_device = torch.device(device)

        self.model = AutoModel.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True,
        )
        self.model.to(self.encoder_device).eval()

        # power=1.0 gives a magnitude mel, not a power mel. center=False,
        # because ``_make_mel`` pads manually.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=self.N_FFT,
            win_length=self.WIN_LENGTH,
            hop_length=self.HOP_LENGTH,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            n_mels=self.N_MELS,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(self.encoder_device)

        self.requires_grad_(False).eval()
        self._resamplers: dict[int, nn.Module] = {}

    def _get_resampler(self, orig_sample_rate: int) -> nn.Module:
        import torchaudio

        resampler = self._resamplers.get(orig_sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                orig_sample_rate, self.TARGET_SAMPLE_RATE,
            ).to(self.encoder_device)
            self._resamplers[orig_sample_rate] = resampler
        return resampler

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Downmix to mono, move to the encoder device, and resample to 24 kHz.

        The method does no loudness normalization, no silence trimming, and no
        length limit. It embeds the full clip, as the reference does.
        """
        if wav.ndim > 2:
            raise ValueError(
                f"Reference audio must be (samples,) or (channels, samples); got {tuple(wav.shape)}."
            )
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(self.encoder_device, torch.float32)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            wav = self._get_resampler(int(sample_rate))(wav)
        return wav

    def _make_mel(self, wav: torch.Tensor) -> torch.Tensor:
        # Reflect-pad by half of the analysis window. The frame centres then
        # agree with center=True. Then compress with a log.
        pad = (self.N_FFT - self.HOP_LENGTH) // 2
        wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        mel = self.mel_transform(wav)                    # (1, n_mels, T)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(1, 2)                       # (1, T, n_mels)

    def _select_embedding(self, output) -> torch.Tensor:
        """Select the output whose width agrees with ``embedding_dim``.

        The backbone returns a sequence of hidden states. This method mean-pools
        a time-varying result into one speaker vector, as the reference does in
        ``tts/emotion.py:340``. The backbone can return more than one candidate,
        so the one with the expected width wins.
        """
        candidates = list(output) if isinstance(output, tuple) else [output]

        pooled: list[torch.Tensor] = []
        for candidate in candidates:
            if candidate is None:
                continue
            tensor = candidate.to(torch.float32).squeeze(0)
            if tensor.ndim >= 2:
                # (T, D) -> (D). Average over every leading (time) axis.
                tensor = tensor.reshape(-1, tensor.shape[-1]).mean(0)
            pooled.append(tensor.reshape(-1))

        for tensor in pooled:
            if tensor.numel() == self.embedding_dim:
                return tensor.contiguous()

        produced = ", ".join(str(t.numel()) for t in pooled) or "nothing"
        raise ValueError(
            f"Speaker encoder {self.model_id!r} produced {produced}, but the model "
            f"expects a {self.embedding_dim}-dim speaker embedding."
        )

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Return the ``(1, embedding_dim)`` float32 speaker embedding."""
        wav = self.prepare_input(wav, sample_rate)
        mel = self._make_mel(wav)
        hidden = self.model(input_values=mel).last_hidden_state
        return self._select_embedding(hidden).unsqueeze(0)
