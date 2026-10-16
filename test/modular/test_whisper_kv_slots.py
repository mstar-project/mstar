"""Whisper says how many slots each of its caches holds, because no tensor does.

The decoder's prefill writes the forced prompt to ``kv_cache`` and the
encoder's whole window to ``cross_kv_cache``, which the request carries as mel
frames, two per row the encoder keeps. Whatever clip a client uploads, the
feature extractor pads or cuts it to one 30 s window, so the context count is
fixed; a count that followed the clip would refuse a long upload that fits or
admit a short one whose prefill then finds no page. An encoder a few
dimensions wide gives the length the context walk writes.
"""

import sys

sys.path.insert(0, ".")

import pytest
import torch
from api_uploads import decode_uploads, preprocess, wav
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE
from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel


@pytest.fixture(scope="module")
def whisper() -> WhisperModel:
    try:
        snapshot_download("openai/whisper-large-v3", local_files_only=True)
    except LocalEntryNotFoundError as exc:
        pytest.skip(f"openai/whisper-large-v3 is not cached: {exc}")
    return WhisperModel("openai/whisper-large-v3")


def _written(model: WhisperModel, tensors: dict) -> dict:
    """The spans the decoder's prefill declares on each cache, over a narrowed real encoder."""
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    # the real conv strides and window; the width does not set the length
    hf_config = WhisperConfig.from_pretrained(
        model.local_dir, d_model=16, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32,
    )
    encoder = WhisperEncoderSubmodule(WhisperEncoder(hf_config), model.config)
    with torch.no_grad():
        states = encoder.forward("prefill", None, **encoder.prepare_inputs("prefill", None, tensors).tensor_inputs)
    decoder = WhisperDecoderSubmodule(torch.nn.Linear(1, 1), model.config)
    inp = decoder.prepare_inputs("prefill", None, {"text_inputs": tensors["text_inputs"], **states})
    step = decoder.declare_step("prefill", ["r"], [inp])
    return {
        KV_CACHE: {s.label: s.span for s in step.segments},
        CROSS_KV_CACHE: {s.label: s.span for s in step.steps[CROSS_KV_CACHE].segments},
    }


# a short clip the extractor pads; one past the window it cuts, with a forced language and task;
# and a stereo 44.1 kHz file the loader resamples
@pytest.mark.parametrize(("seconds", "sample_rate", "channels", "kwargs"), [
    (2.0, 16000, 1, {}),
    (45.0, 16000, 1, {"language": "fr", "task": "translate"}),
    (5.0, 44100, 2, {}),
], ids=["padded_clip", "cut_clip_translated", "stereo_44k"])
def test_each_cache_reserves_what_the_prefill_writes(
    whisper, seconds, sample_rate, channels, kwargs, monkeypatch, tmp_path,
):
    decode_uploads(monkeypatch)
    clip = wav(seconds, sample_rate, channels)
    tensors, model_kwargs = preprocess(whisper, None, None, [clip], ["text"], kwargs, tmp_path)

    assert (model_kwargs["prompt_slots"], model_kwargs["decode_labels"]) == (
        _written(whisper, tensors), {KV_CACHE: ["main"], CROSS_KV_CACHE: []},
    ), "a cache reserves other than the decoder's prefill writes for an uploaded clip, or decode grows the wrong one"
