"""Qwen3-TTS says how many slots the Talker's prefill writes, because its text does not.

``_build_prefill`` drops the ChatML suffix, puts a codec tag in front for each
think, language and speaker slot, writes an in-context clone's reference frames,
and with streamed text holds back all but the first text token. A pool sized by
``text_inputs`` alone over-reserves some prompts and under-reserves a clone
with a long reference, whose prefill then finds no page. Each request is checked
against the length the real prefill builder returns.
"""

import sys
from dataclasses import replace

sys.path.insert(0, ".")

import pytest
import torch
from api_uploads import preprocess, wav
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from mstar.model.qwen3_tts.config import TALKER_KV, Qwen3TTSCodePredictorConfig
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.qwen3_tts.submodules import TalkerSubmodule

PROMPT = "Hello there, this is a fairly short sentence to speak."
REF_TEXT = "and this is what the reference clip says out loud"


def _model(repo: str) -> Qwen3TTSModel:
    try:
        path = snapshot_download(f"Qwen/{repo}", local_files_only=True)
    except LocalEntryNotFoundError as exc:
        pytest.skip(f"Qwen/{repo} is not cached: {exc}")
    return Qwen3TTSModel(path)


def _voice_design() -> Qwen3TTSModel:
    # no VoiceDesign checkpoint here: the CustomVoice tokenizer under VoiceDesign's config
    model = _model("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    talker = replace(model.config.talker, spk_id={}, spk_is_dialect={})
    model.config = replace(model.config, tts_model_type="voice_design", talker=talker)
    return model


def _talker(model: Qwen3TTSModel) -> TalkerSubmodule:
    """The real prefill builder over a Talker a few dimensions wide; the length does not depend on the width."""
    code_predictor = Qwen3TTSCodePredictorConfig(
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        hidden_size=16, intermediate_size=32, head_dim=16,
    )
    talker = replace(
        model.config.talker, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        hidden_size=16, intermediate_size=32, head_dim=8, text_hidden_size=16, code_predictor=code_predictor,
    )
    config = replace(model.config, talker=talker, speaker_encoder=None)
    return TalkerSubmodule(Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config)


def _written(model: Qwen3TTSModel, tensors: dict, cloned: bool) -> int:
    """The rows `_build_prefill` writes for the request."""
    sub = _talker(model)
    layout = tensors["prompt_layout"][0]
    frames = int(layout[4])
    embeds = sub._build_prefill(
        "r", tensors["text_inputs"][0], layout, int(tensors["speaker_id"][0]), int(tensors["language_id"][0]),
        speaker_embed=torch.randn(sub.talker_config.hidden_size) if cloned else None,
        ref_codes=torch.randint(0, 100, (frames, sub.talker_config.num_code_groups)) if frames else None,
    )
    return embeds.shape[0]


@pytest.mark.parametrize(("variant", "reference", "kwargs"), [
    (
        "custom_voice", None,
        {"voice": "ryan", "language": "english", "instruct": "speak slowly and calmly", "non_streaming_mode": True},
    ),
    ("voice_design", None, {"instruct": "a low, gravelly voice with a slight rasp"}),
    # an in-context clone from a stereo 44.1 kHz file the loader resamples and downmixes, its text streamed
    ("base", (3.0, 44100, 2), {"ref_text": REF_TEXT}),
    # the longest clip the loader takes, its text written whole
    ("base", (30.0, 24000, 1), {"ref_text": REF_TEXT, "language": "english", "non_streaming_mode": True}),
    ("base", (3.0, 24000, 1), {"x_vector_only_mode": True}),
], ids=["custom_voice", "voice_design", "clone_streamed", "clone_long_reference", "x_vector_only"])
def test_the_talker_reserves_what_its_prefill_writes(variant, reference, kwargs, tmp_path):
    model = {
        "custom_voice": lambda: _model("Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "voice_design": _voice_design,
        "base": lambda: _model("Qwen3-TTS-12Hz-1.7B-Base"),
    }[variant]()
    files = [wav(*reference)] if reference else []
    tensors, declared = preprocess(model, None, PROMPT, files, ["audio"], kwargs, tmp_path)

    assert (declared["prompt_slots"], declared["decode_labels"]) == (
        {TALKER_KV: {"main": _written(model, tensors, bool(files))}}, {TALKER_KV: ["main"]},
    ), "the Talker reserved other than its prefill writes for the request, or left its decode unreserved"
