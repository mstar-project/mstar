"""Higgs-Audio says how many slots its prefill writes, because no one tensor does.

The LLM's prefill is three walks into ``main``: the ChatML text before the
audio, the encoder's embeddings of every chunk, and the text after. The request
carries the chunks as mel frames, about eight per embedding the encoder keeps,
so a pool sized by any one tensor either refuses requests that fit or admits one
whose audio walk then finds no page. An encoder a few dimensions wide gives the
length the walk writes.
"""

import sys
from dataclasses import replace

sys.path.insert(0, ".")

import pytest
import torch
from api_uploads import decode_uploads, preprocess, wav
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from mstar.model.higgs_audio.components.audio_tower import HiggsAudioFeatureProjector, HiggsAudioTower
from mstar.model.higgs_audio.config import KV_CACHE, HiggsAudioModelConfig
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
from mstar.model.higgs_audio.submodules import HiggsAudioEncoderSubmodule


@pytest.fixture(scope="module")
def higgs() -> HiggsAudioModel:
    try:
        snapshot_download("bosonai/higgs-audio-v3-stt", local_files_only=True)
    except LocalEntryNotFoundError as exc:
        pytest.skip(f"bosonai/higgs-audio-v3-stt is not cached: {exc}")
    return HiggsAudioModel("bosonai/higgs-audio-v3-stt")


def _encoder(config: HiggsAudioModelConfig) -> HiggsAudioEncoderSubmodule:
    """The real tower and projector, narrowed; their strides alone set the output length."""
    tiny = replace(
        config, audio_d_model=16, audio_encoder_layers=1, audio_encoder_attention_heads=2,
        audio_encoder_ffn_dim=32, hidden_size=16,
    )
    return HiggsAudioEncoderSubmodule(HiggsAudioTower(tiny), HiggsAudioFeatureProjector(tiny), tiny)


# one 4 s chunk; three; and three plus a short tail padded to the longest, from a stereo
# 44.1 kHz file the loader resamples and downmixes
@pytest.mark.parametrize(("prompt", "seconds", "sample_rate", "channels"), [
    (None, 3.0, 16000, 1),
    ("Transcribe this meeting, keeping the speaker's filler words.", 12.0, 16000, 1),
    ("Transcribe this meeting, keeping the speaker's filler words.", 12.5, 44100, 2),
], ids=["one_chunk_no_prompt", "three_chunks", "padded_tail_stereo"])
def test_main_reserves_both_text_spans_and_the_audio_the_encoder_keeps(
    higgs, prompt, seconds, sample_rate, channels, monkeypatch, tmp_path,
):
    decode_uploads(monkeypatch)
    clip = wav(seconds, sample_rate, channels)
    tensors, model_kwargs = preprocess(higgs, None, prompt, [clip], ["text"], {}, tmp_path)

    with torch.no_grad():
        audio = _encoder(higgs.config).forward(
            "prefill_audio", None, tensors["audio_features"][0], tensors["audio_feature_lens"][0],
        )["audio_embeds"][0]
    written = sum(span.numel() for span in tensors["text_inputs"]) + audio.shape[0]
    assert (model_kwargs["prompt_slots"], model_kwargs["decode_labels"]) == (
        {KV_CACHE: {"main": written}}, {KV_CACHE: ["main"]},
    ), "main reserves other than its text and audio walks write for an uploaded clip"
