"""Real-weight integration tests for the Qwen3-TTS M* port.

The module never downloads weights. It skips unless the 0.6B CustomVoice
checkpoint is already present in a standard Hugging Face cache and CUDA is
available.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

HF_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def _find_cached_snapshot() -> Path | None:
    repo_dir = f"models--{HF_REPO.replace('/', '--')}"
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


SNAPSHOT = _find_cached_snapshot()
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        SNAPSHOT is None,
        reason=f"{HF_REPO} is not present in the local Hugging Face cache",
    ),
]


@pytest.fixture(scope="module")
def model() -> Qwen3TTSModel:
    assert SNAPSHOT is not None
    return Qwen3TTSModel(model_path_hf=str(SNAPSHOT))


@pytest.fixture(scope="module")
def talker(model):
    submodule = model.get_submodule(
        "Talker",
        device="cuda:0",
        autocast_dtype=torch.bfloat16,
    )
    assert submodule is not None
    return submodule


@pytest.fixture(scope="module")
def codec(model):
    submodule = model.get_submodule("Codec", device="cuda:0")
    assert submodule is not None
    return submodule


def test_real_checkpoint_loads_all_components(model, talker, codec):
    assert model.get_submodule("Talker", device="cuda:0") is talker
    assert model.get_submodule("Codec", device="cuda:0") is codec
    assert sum(p.numel() for p in talker.model.parameters()) == 764_218_368
    assert sum(p.numel() for p in talker.code_predictor.parameters()) == 141_570_304
    assert sum(p.numel() for p in codec.decoder.parameters()) == 114_323_137
    assert next(talker.model.parameters()).dtype == torch.bfloat16
    assert next(codec.decoder.parameters()).dtype == torch.float32


def test_real_tokenizer_and_prefill_build_expected_hidden_width(model, talker):
    tensors = model.process_prompt(
        "Testing Qwen three TTS.",
        input_modalities=["text"],
        output_modalities=["audio"],
        voice="Vivian",
        language="English",
    )
    prepared = talker.prepare_inputs(
        "talker_prefill",
        SimpleNamespace(request_id="integration-prefill"),
        tensors,
    )

    assert prepared.input_embeds.ndim == 2
    assert prepared.input_embeds.shape[1] == model.config.talker.hidden_size
    assert prepared.input_seq_len == prepared.input_embeds.shape[0]


def test_real_codec_decodes_expected_number_of_pcm_samples(codec, model):
    frames = 2
    codes = torch.zeros(
        1,
        model.config.codec.num_quantizers,
        frames,
        dtype=torch.long,
        device="cuda:0",
    )
    pcm = codec._decode(codes)
    torch.cuda.synchronize()

    assert pcm.shape == (1, frames * model.config.codec.decode_upsample_rate)
    assert pcm.dtype == torch.int16
