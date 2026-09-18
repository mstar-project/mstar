"""Real-weight loading checks for the Qwen3-TTS 1.7B family.

Never downloads. Each variant is exercised only when its checkpoint is already
in the local Hugging Face cache and CUDA is available. The parity harness
(``test/qwen3-tts/parity_qwen3_tts.py``) is the functional check; this file
pins what loading must get right for every variant: complete checkpoint
coverage, the Talker-to-predictor projection, and the prefill layouts.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

VARIANTS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
# talker.* minus code_predictor.*  |  code_predictor.*  |  speech_tokenizer decoder.*
TALKER_PARAMS = 1_741_550_592
CODE_PREDICTOR_PARAMS = 175_125_760
CODEC_PARAMS = 114_323_137


def _find_cached_snapshot(repo: str) -> Path | None:
    repo_dir = f"models--{repo.replace('/', '--')}"
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if (
                (snapshot / "model.safetensors").is_file()
                and (snapshot / "speech_tokenizer" / "model.safetensors").is_file()
            ):
                return snapshot
    return None


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def loaded(request):
    variant = request.param
    snapshot = _find_cached_snapshot(VARIANTS[variant])
    if snapshot is None:
        pytest.skip(f"{VARIANTS[variant]} is not in the local Hugging Face cache")
    model = Qwen3TTSModel(model_path_hf=str(snapshot))
    talker = model.get_submodule("Talker", device="cuda:0", autocast_dtype=torch.bfloat16)
    codec = model.get_submodule("Codec", device="cuda:0")
    yield variant, model, talker, codec
    del talker, codec
    model._submodule_cache.clear()
    torch.cuda.empty_cache()


def test_variant_metadata_and_weights_load_completely(loaded):
    variant, model, talker, codec = loaded
    assert model.config.tts_model_type == variant
    assert model.config.tts_model_size == "1b7"
    assert model.config.talker.hidden_size == 2048
    assert model.config.code_predictor.hidden_size == 1024
    # The coverage check in get_submodule already failed loudly on any missing
    # or unused tensor; the counts pin the architecture the checkpoint carries.
    assert sum(p.numel() for p in talker.model.parameters()) == TALKER_PARAMS
    assert sum(p.numel() for p in talker.code_predictor.parameters()) == CODE_PREDICTOR_PARAMS
    assert sum(p.numel() for p in codec.decoder.parameters()) == CODEC_PARAMS
    projection = talker.code_predictor.small_to_mtp_projection
    assert isinstance(projection, torch.nn.Linear)
    assert projection.weight.shape == (1024, 2048)
    assert next(talker.model.parameters()).dtype == torch.bfloat16
    assert next(codec.decoder.parameters()).dtype == torch.float32


def test_prefill_layout_matches_variant(loaded):
    variant, model, talker, _ = loaded
    kwargs = {"input_modalities": ["text"], "output_modalities": ["audio"]}
    if variant == "base":
        with pytest.raises(ValueError, match="reference audio"):
            model.process_prompt("Testing the base model.", **kwargs)
        return
    if variant == "custom_voice":
        tensors = model.process_prompt(
            "Testing Qwen three TTS.", voice="Vivian", language="English",
            instruct="Speak slowly and clearly.", **kwargs,
        )
        assert tensors["speaker_id"][0].item() == model.config.talker.spk_id["vivian"]
    else:
        tensors = model.process_prompt(
            "Testing Qwen three TTS.", instruct="A calm male voice.", **kwargs,
        )
        assert tensors["speaker_id"][0].item() == -1
    instruct_len, text_len, stream_text = tensors["prompt_layout"][0].tolist()
    assert instruct_len > 0 and text_len > 0 and stream_text == 0

    prepared = talker.prepare_inputs(
        "talker_prefill", SimpleNamespace(request_id=f"prefill-{variant}"), tensors,
    )
    assert prepared.input_embeds.shape[1] == model.config.talker.hidden_size
    # instruct + role(3) + codec tags + (text + eos) + closing pad/bos
    tags = 3 + 1 + (1 if variant == "custom_voice" else 0) + 1  # think..., [speaker], pad
    assert prepared.input_seq_len == instruct_len + 3 + tags + (text_len + 1) + 1
    state = talker.request_state(f"prefill-{variant}")
    assert state["trailing_text_hidden"].shape == (0, model.config.talker.hidden_size)


def test_depth_loop_runs_through_projection(loaded):
    variant, model, talker, _ = loaded
    del variant
    batch = 2
    hidden = torch.randn(
        batch, model.config.talker.hidden_size, device="cuda:0", dtype=torch.bfloat16
    )
    layer0 = torch.randint(0, 2048, (batch,), device="cuda:0")
    codes, embed_sum = talker._depth_loop(hidden, layer0, lambda logits: logits.argmax(-1))
    torch.cuda.synchronize()
    assert codes.shape == (batch, model.config.num_code_groups)
    assert (codes[:, 0] == layer0).all()
    assert (codes[:, 1:] < model.config.code_predictor.vocab_size).all()
    assert embed_sum.shape == (batch, model.config.talker.hidden_size)
