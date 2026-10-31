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
from mstar.utils.attention import apply_rope_pos_ids

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


def test_real_tokenizer_and_prefill_build_expected_hidden_width(model, talker):
    result = model.process_prompt(
        "Testing Qwen three TTS.",
        input_modalities=["text"],
        output_modalities=["audio"],
        voice="Vivian",
        language="English",
    )
    prepared = talker.prepare_inputs(
        "talker_prefill",
        # rid_handle, not request_id: the talker keys its per-request
        # state by the worker-local handle. The layout and tags arrive as
        # step metadata.
        SimpleNamespace(request_id="integration-prefill", rid_handle=0, step_metadata=result.metadata),
        result.new_input_tensors,
    )

    assert prepared.input_embeds.ndim == 2
    assert prepared.input_embeds.shape[1] == model.config.talker.hidden_size
    assert prepared.input_seq_len == prepared.input_embeds.shape[0]


def _served_decode(model, codec, codes: torch.Tensor) -> torch.Tensor:
    """One codec step over fresh streams, as the engine runs it: each row on a
    slot of the model's declared state pool, gathered and scattered around
    the compiled decode. ``codes [B, Q, frames]`` -> PCM16 ``[B, samples]``."""
    from mstar.engine.resources import StepContext
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool
    from mstar.engine.resources.runner import StepRunner
    from mstar.model.qwen3_tts.config import CODEC_STATE
    from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine

    spec = next(s for s in model.get_node_resources() if s.resource_key == CODEC_STATE)
    spec.apply_yaml_overrides(max_slots=codes.shape[0] + 1)
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=codes.device))
    runner = StepRunner({CODEC_STATE: pool})
    rids = [f"r{i}" for i in range(codes.shape[0])]
    for rid in rids:
        pool.ingest_request(rid)
    rows = [ARNodeInputs(tensor_inputs={"codec_tokens": codes[i].t()}) for i in range(len(rids))]
    step = codec.declare_step("codec_chunk", rids, rows)
    step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="codec_chunk", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    engine_inputs = ModelInputsFromEngine(
        request_ids=rids, per_request_info={}, resources={CODEC_STATE: pool}, step=step,
    )
    lengths = torch.full((len(rids),), codes.shape[-1], dtype=torch.long, device=codes.device)
    with torch.no_grad():
        return codec._decode(engine_inputs, codes, lengths)


def test_real_codec_decodes_expected_number_of_pcm_samples(codec, model):
    frames = 2
    codes = torch.zeros(1, model.config.codec.num_quantizers, frames, dtype=torch.long, device="cuda:0")
    pcm = _served_decode(model, codec, codes)
    torch.cuda.synchronize()

    assert pcm.shape == (1, frames * model.config.codec.decode_upsample_rate)
    assert pcm.dtype == torch.int16


def test_real_bf16_codec_stays_close_to_float32(codec, model):
    """The bf16 codec (``codec_dtype: bfloat16``) against the float32 one on
    the same codes: a sanity floor that a broken cast or a bf16-only kernel
    falls through, not the WER gate the deployment was measured with."""
    bf16_model = Qwen3TTSModel(model_path_hf=str(SNAPSHOT), codec_dtype="bfloat16")
    bf16_codec = bf16_model.get_submodule("Codec", device="cuda:0")
    torch.manual_seed(0)
    codes = torch.randint(
        0, model.config.codec.codebook_size, (2, model.config.codec.num_quantizers, 25), device="cuda:0",
    )
    want = _served_decode(model, codec, codes).double()
    got = _served_decode(bf16_model, bf16_codec, codes).double()
    snr = 10 * torch.log10(want.pow(2).sum() / (want - got).pow(2).sum().clamp_min(1e-20))
    assert snr > 15, f"bf16 codec SNR {snr:.1f} dB against float32"
