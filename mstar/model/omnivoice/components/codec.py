"""Higgs-Audio-v2 codec wrappers: waveform <-> 8-row token canvas.

``HiggsAudioV2TokenizerModel`` ships in ``transformers``, so nothing is
re-implemented here — these are the two thin seams M* needs, plus the
post-processing the reference applies after decode.

The codec weights live in the OmniVoice checkpoint under ``audio_tokenizer/``.
The standalone mirror is the fallback the reference falls back to, kept for
checkpoints published without the subfolder.
"""

import logging

import numpy as np
import torch
from torch import nn

from mstar.model.omnivoice.config import (
    CODEC_FALLBACK_HF,
    CODEC_SUBFOLDER,
    OmniVoiceConfig,
)

logger = logging.getLogger(__name__)


def load_codec(model_path: str, cache_dir: str | None = None) -> nn.Module:
    """Load the codec from the checkpoint's subfolder, else the mirror."""
    from transformers import HiggsAudioV2TokenizerModel

    try:
        return HiggsAudioV2TokenizerModel.from_pretrained(
            model_path, subfolder=CODEC_SUBFOLDER, cache_dir=cache_dir
        ).eval()
    except (OSError, ValueError) as exc:
        logger.warning(
            "No %s/ in %s (%s); falling back to %s",
            CODEC_SUBFOLDER, model_path, exc, CODEC_FALLBACK_HF,
        )
        return HiggsAudioV2TokenizerModel.from_pretrained(
            CODEC_FALLBACK_HF, cache_dir=cache_dir
        ).eval()


@torch.inference_mode()
def encode_reference(codec: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """``[1, T]`` waveform at the codec's rate -> ``[C, T_tok]`` tokens.

    The tail is clipped to a whole number of hops first, as the reference does:
    a partial hop would encode to a frame the decoder cannot place.
    """
    hop = int(codec.config.hop_length)
    remainder = int(waveform.shape[-1] % hop)
    if remainder:
        waveform = waveform[:, :-remainder]
    if waveform.shape[-1] == 0:
        raise ValueError(
            f"Reference audio is shorter than one codec hop ({hop} samples)"
        )
    # Follow the codec's own dtype rather than pinning float32: the checkpoint
    # is loaded at the configured dtype and the codec is cast with it, so a
    # hardcoded float32 input would be the same mixed-dtype fault the backbone
    # hit.
    codec_dtype = next(codec.parameters()).dtype
    waveform = waveform.to(device=codec.device, dtype=codec_dtype)
    return codec.encode(waveform.unsqueeze(0)).audio_codes.squeeze(0)


@torch.inference_mode()
def decode_canvas(codec: nn.Module, tokens: torch.Tensor) -> np.ndarray:
    """``[C, T_tok]`` tokens -> 1-D waveform at ``codec.config.sample_rate``."""
    tokens = tokens.to(codec.device)
    return codec.decode(tokens.unsqueeze(0)).audio_values[0].float().cpu().numpy()


def post_process(
    waveform: np.ndarray,
    config: OmniVoiceConfig,
    ref_rms: float | None,
    pad_duration: float,
    fade_duration: float,
    enabled: bool,
) -> np.ndarray:
    """Silence trim, level match, fade and pad — the reference's tail.

    Level handling differs by mode and is not cosmetic: a cloned voice is
    matched to the reference's RMS (quiet references stay quiet), while a
    designed or automatic voice has no reference to match and is peak-normalised
    to 0.5 instead.

    Imported from the ``omnivoice`` package rather than re-ported: the silence
    trim is pydub-based and re-implementing it would drift from the reference
    for no gain.
    """
    from omnivoice.utils.audio import fade_and_pad_audio, remove_silence

    audio = waveform
    if audio.ndim == 1:
        audio = audio[None, :]

    # A decode that is digitally silent is a fault, not a quiet result, and it
    # has a specific cause worth naming: a codec running at a dtype it
    # underflows in. Saying so beats the numpy error the silence used to
    # produce three frames later.
    if audio.size and not np.any(audio):
        raise RuntimeError(
            f"Codec decoded {audio.shape[-1]} samples of digital silence. "
            "The usual cause is the codec running in float16; upstream keeps "
            "it in float32."
        )

    if enabled:
        trimmed = remove_silence(
            audio, config.sample_rate, mid_sil=500, lead_sil=100, trail_sil=100
        )
        # Over-trimming is a degraded result, not a server error: keep the
        # untrimmed audio rather than letting an empty array reach .max().
        if np.asarray(trimmed).size == 0:
            logger.warning(
                "Silence trim removed the whole utterance (%d samples); "
                "returning it untrimmed.", audio.shape[-1],
            )
        else:
            audio = trimmed

    if ref_rms is not None and ref_rms < 0.1:
        audio = audio * ref_rms / 0.1
    elif ref_rms is None:
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.5

    audio = fade_and_pad_audio(
        audio,
        pad_duration=pad_duration,
        fade_duration=fade_duration,
        sample_rate=config.sample_rate,
    )
    return np.asarray(audio).squeeze(0)
