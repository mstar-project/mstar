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
from mstar.utils.attention import apply_rope_pos_ids, decode_attn_nhd

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


def test_fused_code_predictor_rope_matches_checkpoint_formula():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    head_dim = 128
    rope_theta = 1_000_000.0
    position_ids = torch.tensor([7], dtype=torch.int32, device=device)
    q = torch.randn(1, 16, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, 8, head_dim, dtype=dtype, device=device)

    inv_freq = 1.0 / (
        rope_theta ** (
            torch.arange(
                0, head_dim, 2, dtype=torch.float32, device=device
            ) / head_dim
        )
    )
    angles = position_ids.to(torch.float32).unsqueeze(1) * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype).unsqueeze(1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype).unsqueeze(1)

    def reference(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cos + torch.cat([-second, first], dim=-1) * sin

    expected_q = reference(q)
    expected_k = reference(k)
    actual_q, actual_k = apply_rope_pos_ids(
        q.clone(), k.clone(), position_ids, rope_theta
    )

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("cache_len", [1, 8, 16])
def test_code_predictor_decode_attention_matches_sdpa(cache_len: int):
    generator = torch.Generator(device="cuda:0").manual_seed(1234 + cache_len)
    q = torch.randn(
        2, 1, 16, 128,
        dtype=torch.bfloat16,
        device="cuda:0",
        generator=generator,
    )
    k_cache = torch.randn(
        2, 16, 8, 128,
        dtype=torch.bfloat16,
        device="cuda:0",
        generator=generator,
    )
    v_cache = torch.randn(
        2, 16, 8, 128,
        dtype=torch.bfloat16,
        device="cuda:0",
        generator=generator,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k_cache[:, :cache_len].transpose(1, 2),
        v_cache[:, :cache_len].transpose(1, 2),
        is_causal=False,
        enable_gqa=True,
    ).transpose(1, 2)
    actual = decode_attn_nhd(q, k_cache, v_cache, cache_len)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


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
