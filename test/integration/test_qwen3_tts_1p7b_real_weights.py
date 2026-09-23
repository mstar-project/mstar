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
        # Base clones a voice: a text-only request has no reference clip.
        with pytest.raises(ValueError, match="reference clip"):
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
            "Testing Qwen three TTS.", language="English", instruct="A calm male voice.", **kwargs,
        )
        assert tensors["speaker_id"][0].item() == -1
    instruct_len, text_len, stream_text, ref_text_len, ref_frames = tensors["prompt_layout"][0].tolist()
    assert instruct_len > 0 and text_len > 0 and stream_text == 0
    assert ref_text_len == 0 and ref_frames == 0

    prepared = talker.prepare_inputs(
        "talker_prefill", SimpleNamespace(request_id=f"prefill-{variant}"), tensors,
    )
    assert prepared.input_embeds.shape[1] == model.config.talker.hidden_size
    # instruct + role(3) + codec tags + (text + eos) + closing pad/bos; with an
    # explicit language the tags are think, think_bos, language, think_eos,
    # [speaker], pad.
    tags = 3 + 1 + (1 if variant == "custom_voice" else 0) + 1
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


def test_base_reference_clip_becomes_xvector_frames_and_clone_prefill(loaded):
    """Base only: the shared reference clip runs through load_audio -> RefEncoder -> clone prefill.

    ``QWEN3_TTS_REF_AUDIO`` names a 24 kHz mono clip (the benchmark's
    ``clone_2.wav``); its transcript is fixed here.
    """
    variant, model, talker, _ = loaded
    if variant != "base":
        pytest.skip("voice cloning is a Base feature")
    clip_path = os.environ.get("QWEN3_TTS_REF_AUDIO")
    if not clip_path or not Path(clip_path).is_file():
        pytest.skip("set QWEN3_TTS_REF_AUDIO to a reference clip")
    from mstar.model.submodule_base import ModelInputsFromEngine

    clip = model.load_audio(clip_path, "cuda:0")
    assert clip.metadata["sample_rate"] == 24000 and clip.data.ndim == 1
    frames = model.config.codec.frames_for_samples(clip.data.shape[0])
    tensors = model.process_prompt(
        "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
        input_modalities=["audio", "text"], output_modalities=["audio"],
        tensors={"audio_inputs": [clip.data]},
        ref_text="Okay. Yeah. I resent you. I love you. I respect you. "
                 "But you know what? You blew it! And thanks to you.",
    )
    assert tensors["prompt_layout"][0].tolist()[4] == frames and tensors["ref_frames"][0].item() == frames

    ref_encoder = model.get_submodule("RefEncoder", device="cuda:0", autocast_dtype=torch.bfloat16)
    prepared = ref_encoder.prepare_inputs(
        "talker_prefill_clone", SimpleNamespace(request_id="clone"),
        {"audio_inputs": [clip.data], "prompt_layout": tensors["prompt_layout"]},
    )
    engine_inputs = ModelInputsFromEngine(request_ids=["clone"], per_request_info={})
    with torch.no_grad():
        encoded = ref_encoder.forward(
            "talker_prefill_clone", engine_inputs,
            **ref_encoder.preprocess("talker_prefill_clone", engine_inputs, [prepared]),
        )
    torch.cuda.synchronize()
    xvec, codes = encoded["speaker_embed"][0], encoded["ref_codes"][0]
    assert xvec.shape == (model.config.talker.hidden_size,) and torch.isfinite(xvec.float()).all()
    assert codes.shape == (frames, model.config.num_code_groups)
    assert (codes >= 0).all() and (codes < model.config.codec.codebook_size).all()

    prefill = talker.prepare_inputs(
        "talker_prefill_clone", SimpleNamespace(request_id="clone-prefill"),
        {**tensors, "speaker_embed": [xvec], "ref_codes": [codes]},
    )
    # role(3) + [think, think_bos, lang?, think_eos, x-vector, pad] + in-context span (streaming text default)
    assert prefill.input_embeds.shape[1] == model.config.talker.hidden_size
    assert prefill.input_seq_len > 3 + 5 + frames
    state = talker.request_state("clone-prefill")
    assert torch.equal(state["reference_frames"], codes)
    model._submodule_cache.pop("RefEncoder", None)
    del ref_encoder
    torch.cuda.empty_cache()
