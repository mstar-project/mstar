"""Real-weight checks for the CosyVoice3 port, on CPU.

Never downloads: skips unless ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512`` is
already in the Hugging Face cache. Every node the model builds must load its
checkpoint exactly (the loaders refuse missing or unexpected keys) and carry
the computed buffers a direct construction makes; the reference front end and
Token2Wav must produce the documented geometry.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mstar.model.chatterbox.loader import resolve_snapshot  # noqa: E402
from mstar.model.cosyvoice3.components.flow import CosyVoice3Flow, CosyVoice3FlowRow  # noqa: E402
from mstar.model.cosyvoice3.components.hift import CausalHiFTGenerator  # noqa: E402
from mstar.model.cosyvoice3.config import END_OF_PROMPT, END_OF_PROMPT_ID, SYSTEM_PROMPT  # noqa: E402
from mstar.model.cosyvoice3.cosyvoice3_model import CosyVoice3Model  # noqa: E402

HF_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


@pytest.fixture(scope="module")
def model() -> CosyVoice3Model:
    try:
        resolve_snapshot(HF_REPO)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"checkpoint not available: {exc}")
    return CosyVoice3Model(model_path_hf=HF_REPO)


def _clip(seconds: float) -> torch.Tensor:
    t = torch.arange(int(seconds * 24000)) / 24000
    return 0.3 * torch.sin(2 * math.pi * 180 * t) * (1 + 0.5 * torch.sin(2 * math.pi * 3 * t))


def test_text_tokenizer_carries_the_trained_delimiter(model):
    ids = model.tokenizer(f"{SYSTEM_PROMPT}{END_OF_PROMPT}hello")
    assert (ids == END_OF_PROMPT_ID).sum() == 1
    out = model.process_prompt(
        "Hello there.",
        ["text", "audio"],
        ["audio"],
        tensors={"audio_inputs": [_clip(2.0)]},
        ref_text="A reference transcript.",
    )
    assert (out.new_input_tensors["text_inputs"][0] == END_OF_PROMPT_ID).sum() == 1


def test_reference_encoder_tokens_mel_and_xvector(model):
    encoder = model._reference_encoder("cpu")
    voice = encoder.voice(_clip(2.0))
    assert voice.num_prompt_tokens == 50 and voice.prompt_feat.shape == (100, 80)
    assert voice.embedding.shape == (192,) and torch.isfinite(voice.embedding).all()
    assert int(voice.prompt_tokens.max()) < 6561


def test_nodes_load_exactly_and_keep_their_computed_buffers(model):
    flow = model.get_submodule("flow", "cpu").flow
    torch.testing.assert_close(flow.noise, CosyVoice3Flow(model.config.flow).noise, rtol=0, atol=0)
    hift = model.get_submodule("hift", "cpu").hift
    fresh = CausalHiFTGenerator(model.config.hift)
    assert torch.equal(hift.stft_window, fresh.stft_window) and torch.equal(hift.source_noise, fresh.source_noise)
    assert hift.f0_predictor.classifier.weight.dtype == torch.float64
    lm = model.get_submodule("llm", "cpu").model
    assert lm.llm_decoder.weight.dtype == torch.float32 and lm.speech_embedding.weight.shape == (6761, 896)


def test_token2wav_geometry_with_real_weights(model):
    flow = model.get_submodule("flow", "cpu").flow
    hift = model.get_submodule("hift", "cpu").hift
    voice = model._reference_encoder("cpu").voice(_clip(2.0))
    tokens = voice.prompt_tokens[:30].clone()
    with torch.no_grad():
        mel = flow.solve_rows([CosyVoice3FlowRow(voice, tokens, 30, final=True, streaming=True)], n_timesteps=2)[0]
        wav, _ = hift.vocode(mel[None].float(), finalize=True)
    assert mel.shape == (80, 60) and torch.isfinite(mel).all()
    assert wav.shape == (1, 60 * 480) and wav.abs().max() > 1e-3
