"""Chatterbox says how many slots T3's prefill writes, because its text does not.

Every stream T3 opens starts with the conditioning rows (speaker, reference
prompt, emotion) and ends on the speech BOS, which Chatterbox writes twice, and
guidance opens a second stream as long as the first. A pool sized by the text
alone admits a request whose prefill then finds no page. Turbo's reference
prompt is as many S3 tokens as the voice encoder makes of the clip's first
``enc_cond_seconds``, so a count at the tokenizer's cap refuses requests that
fit.
"""

import sys

sys.path.insert(0, ".")

import pytest
import torch
from api_uploads import preprocess, wav
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from mstar.model.chatterbox.chatterbox_model import PROMPT_TOKENS, SPEAKER_EMB, TEXT_INPUTS, ChatterboxModel
from mstar.model.chatterbox.components import t3 as t3_components
from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
from mstar.model.chatterbox.config import T3_KV, ChatterboxConfig, S3TokenizerConfig
from mstar.model.chatterbox.submodules import BuiltinT3Voice, T3Submodule, VoiceEncoderSubmodule

PROMPT = "The quick brown fox jumps over the lazy dog, twice."


class _StubTokenizer:
    """One id per character, for the Turbo checkpoint this machine does not cache."""

    def __call__(self, text):
        return torch.tensor([ord(c) % 50 for c in text], dtype=torch.long)


class _StubSpeakerEncoder:
    """A speaker embedding without the LSTM: it counts no tokens."""

    def embed_utterance(self, wav):
        return torch.zeros(256)


def _model(variant: str) -> ChatterboxModel:
    if variant == "turbo":
        model = object.__new__(ChatterboxModel)
        model.config = ChatterboxConfig.from_variant(variant)
        model.tokenizer = _StubTokenizer()
        model.voices_dir = None
        return model
    try:
        snapshot_download("ResembleAI/chatterbox", local_files_only=True)
    except LocalEntryNotFoundError as exc:
        pytest.skip(f"ResembleAI/chatterbox is not cached: {exc}")
    return ChatterboxModel("ResembleAI/chatterbox", variant=variant)


def _t3(config: ChatterboxConfig, monkeypatch) -> T3Submodule:
    # the real embeddings and conditioning encoder; prefill's span never reaches the backbone
    monkeypatch.setattr(t3_components, "T3Backbone", lambda *args, **kwargs: torch.nn.Identity())
    voice = BuiltinT3Voice(
        speaker_emb=torch.zeros(config.t3.speaker_embed_size),
        prompt_tokens=torch.zeros(config.t3.speech_cond_prompt_len, dtype=torch.long),
    )
    return T3Submodule(t3_components.T3Model(config.t3), config, builtin_voice=voice)


def _written(model: ChatterboxModel, sub: T3Submodule, tensors: dict, knobs: dict) -> dict:
    """The span T3's prefill declares writing to each stream, its clip conditioned by the voice encoder's own code."""
    t3_inputs = {TEXT_INPUTS: tensors[TEXT_INPUTS]}
    if "ref_audio" in tensors:
        # one encoder layer, not six: depth changes no token count
        encoder = VoiceEncoderSubmodule(_StubSpeakerEncoder(), S3Tokenizer(S3TokenizerConfig(n_layer=1)), model.config)
        t3_inputs[SPEAKER_EMB] = [torch.randn(sub.t3.speaker_embed_size)]
        t3_inputs[PROMPT_TOKENS] = [encoder.condition(tensors["ref_audio"][0])[1]]
    info = type("Info", (), {
        "rid_handle": "r", "request_id": "wire-r",
        "step_metadata": {**model.resolve_generation_kwargs(knobs), "is_prefill": True},
    })()
    step = sub.declare_step("prefill", ["r"], [sub.prepare_inputs("prefill", info, t3_inputs)])
    return {s.label: s.span for s in step.segments}


@pytest.mark.parametrize(("variant", "reference", "cfg_weight"), [
    ("chatterbox", None, 0.5),
    # a stereo 44.1 kHz clip, which the loader resamples, with guidance off
    ("multilingual", (6.0, 44100, 2), 0.0),
    ("turbo", None, 0.5),
    # short of enc_cond_seconds, and past it, which the voice encoder trims
    ("turbo", (9.7, 44100, 2), 0.5),
    ("turbo", (20.0, 24000, 1), 0.5),
], ids=["builtin_voice", "clip_no_guidance", "turbo_builtin_voice", "turbo_short_clip", "turbo_long_clip"])
def test_each_stream_reserves_what_its_prefill_writes(variant, reference, cfg_weight, monkeypatch, tmp_path):
    model = _model(variant)
    sub = _t3(model.config, monkeypatch)
    knobs = {"cfg_weight": cfg_weight}
    files = [wav(*reference)] if reference else []

    tensors, declared = preprocess(model, None, PROMPT, files, ["audio"], knobs, tmp_path)

    written = _written(model, sub, tensors, knobs)
    assert (declared["prompt_slots"], declared["decode_labels"]) == ({T3_KV: written}, {T3_KV: list(written)}), (
        "a stream reserved other than the conditioning, text and BOS its prefill writes for the request"
    )
