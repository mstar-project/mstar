"""Tests for TalkerSubmodule + node registration + construction (step 6e-2).

Two layers:

  * Pure-Python: a tiny TalkerGenerator wrapped in TalkerSubmodule —
    prepare_inputs embeds talker text ids, forward runs the full
    AR-decode + VAE-decode and returns an audio_chunk. Plus the
    model's get_node_engine_types / get_submodule wiring.

  * Snapshot-gated: MingFlashOmniModel._create_talker_submodule builds
    the full talker stack and loads real weights, then runs a tiny
    end-to-end generation. Heavy (~5 GB CPU); skipped without a snapshot.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mminf.engine.base import EngineType
from mminf.model.ming_omni_flash.components.audio_vae import build_audio_vae
from mminf.model.ming_omni_flash.components.talker_dit import (
    build_aggregator,
    build_talker_cfm,
    build_talker_heads,
    build_talker_llm,
)
from mminf.model.ming_omni_flash.components.talker_generator import TalkerGenerator
from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    AudioVAEConfig,
    DiTBlockConfig,
    MingFlashOmniModelConfig,
    TalkerConfig,
    TalkerLLMConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel
from mminf.model.ming_omni_flash.submodules import TalkerSubmodule

# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------


def _find_local_snapshot() -> str | None:
    def _has(p: Path) -> bool:
        return (
            (p / "talker" / "config.json").exists()
            and (p / "talker" / "model.safetensors").exists()
            and (p / "talker" / "vae" / "model.safetensors").exists()
        )
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and _has(Path(override)):
        return override
    hybrid = Path("/dev/shm/ming-hybrid")
    if _has(hybrid):
        return str(hybrid)
    return None


# ---------------------------------------------------------------------------
# Tiny config + generator (CPU, fast)
# ---------------------------------------------------------------------------


def _tiny_qwen2_backbone(hidden_size: int = 32, num_layers: int = 1) -> dict:
    return {
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "vocab_size": 1,
        "use_sliding_window": True,
        "sliding_window": 32,
        "max_window_layers": 0,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    }


def _tiny_talker_config() -> TalkerConfig:
    return TalkerConfig(
        steps=2,
        patch_size=2,
        history_patch_size=2,
        cfg_strength=2.0,
        llm=TalkerLLMConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, sliding_window=64, max_window_layers=0,
            use_sliding_window=False,
        ),
        flowmodel=DiTBlockConfig(
            depth=1, hidden_size=32, num_heads=2, mlp_ratio=2,
            in_channels=4, dropout=0.0, attn_mask_enabled=False,
        ),
        aggregator=DiTBlockConfig(
            depth=1, hidden_size=32, num_heads=2, mlp_ratio=2,
            in_channels=4, dropout=0.0, attn_mask_enabled=False,
        ),
        vae=AudioVAEConfig(
            sample_rate=8000, patch_size=-1, latent_dim=4,
            encoder_input_dim=16, encoder_hop_size=16, decoder_output_dim=16,
            enc_backbone=_tiny_qwen2_backbone(), dec_backbone=_tiny_qwen2_backbone(),
        ),
    )


def _tiny_model_config() -> MingFlashOmniModelConfig:
    return MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        talker=_tiny_talker_config(),
    )


def _build_tiny_submodule() -> TalkerSubmodule:
    cfg = _tiny_talker_config()
    llm = build_talker_llm(cfg.llm, dtype=torch.float32, device="cpu")
    cfm = build_talker_cfm(cfg, dtype=torch.float32, device="cpu")
    agg = build_aggregator(cfg, dtype=torch.float32, device="cpu")
    heads = build_talker_heads(cfg, dtype=torch.float32, device="cpu")
    vae = build_audio_vae(cfg.vae, dtype=torch.float32, device="cpu", attn_implementation="sdpa")
    gen = TalkerGenerator(
        talker_config=cfg, llm=llm, cfm=cfm, aggregator=agg,
        stop_head=heads["stop_head"], audio_vae=vae,
    )
    model_cfg = _tiny_model_config()
    return TalkerSubmodule(generator=gen, config=model_cfg, max_steps=3, min_new_token=1000)


# ---------------------------------------------------------------------------
# TalkerSubmodule — prepare_inputs / forward
# ---------------------------------------------------------------------------


def test_talker_submodule_stateless_flavor_is_audio_codec() -> None:
    sub = _build_tiny_submodule()
    assert sub.get_stateless_flavor() == "audio_codec"


def test_talker_submodule_prepare_inputs_embeds_text_ids() -> None:
    sub = _build_tiny_submodule()
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    out = sub.prepare_inputs(
        graph_walk="talker", fwd_info=None,
        inputs={"talker_text_inputs": [token_ids]},
    )
    embeds = out.tensor_inputs["inputs_embeds"]
    # (1, T, hidden) after embedding.
    assert embeds.shape == (1, 4, sub.config.talker.llm.hidden_size)
    assert out.tensor_inputs["prompt_wav_lat"] is None


def test_talker_submodule_prepare_inputs_raises_on_missing_text() -> None:
    sub = _build_tiny_submodule()
    with pytest.raises(ValueError, match="missing 'talker_text_inputs'"):
        sub.prepare_inputs(graph_walk="talker", fwd_info=None, inputs={})


def test_talker_submodule_forward_returns_audio_chunk() -> None:
    """End-to-end tiny generation: text ids -> waveform."""
    sub = _build_tiny_submodule()
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    prep = sub.prepare_inputs(
        graph_walk="talker", fwd_info=None,
        inputs={"talker_text_inputs": [token_ids]},
    )
    out = sub.forward(
        graph_walk="talker", engine_inputs=None,
        inputs_embeds=prep.tensor_inputs["inputs_embeds"],
        prompt_wav_lat=prep.tensor_inputs["prompt_wav_lat"],
    )
    assert "audio_chunk" in out
    wf = out["audio_chunk"][0]
    assert wf.dim() == 3
    assert wf.shape[0] == 1 and wf.shape[1] == 1
    assert torch.isfinite(wf).all()


def test_talker_submodule_prepare_inputs_accepts_2d_token_ids() -> None:
    """Already-batched (1, T) token ids work too (no double-unsqueeze)."""
    sub = _build_tiny_submodule()
    token_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    out = sub.prepare_inputs(
        graph_walk="talker", fwd_info=None,
        inputs={"talker_text_inputs": [token_ids]},
    )
    assert out.tensor_inputs["inputs_embeds"].shape == (1, 3, sub.config.talker.llm.hidden_size)


# ---------------------------------------------------------------------------
# Model node registration
# ---------------------------------------------------------------------------


def test_get_node_engine_types_registers_talker_when_config_present() -> None:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = _tiny_model_config()
    types = inst.get_node_engine_types()
    assert types["Talker"] == EngineType.STATELESS
    assert types["Thinker"] == EngineType.KV_CACHE


def test_get_node_engine_types_omits_talker_for_thinker_only() -> None:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    cfg = _tiny_model_config()
    cfg.talker = None
    inst.config = cfg
    types = inst.get_node_engine_types()
    assert "Talker" not in types


def test_get_submodule_talker_raises_without_talker_config() -> None:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    cfg = _tiny_model_config()
    cfg.talker = None
    inst.config = cfg
    inst._submodule_cache = {}
    with pytest.raises(RuntimeError, match="no talker/ subdir"):
        inst._create_talker_submodule(device="cpu")


# ---------------------------------------------------------------------------
# Snapshot-gated end-to-end construction + generation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _find_local_snapshot() is None,
    reason="Need Ming-flash-omni-2.0 snapshot with talker/.",
)
def test_create_talker_submodule_loads_real_weights_and_generates() -> None:
    """Full talker construction (real weights) + tiny TTS generation.

    Heavy (~5 GB on CPU). Builds the LLM + CFM + Aggregator + heads +
    AudioVAE, loads every subtree via the step-6f loaders, and runs a
    short generation to confirm the wiring produces a finite waveform.
    """
    snap = _find_local_snapshot()
    code_dir = os.environ.get("MING_CODE_DIR", "/tmp/ming_repo")
    model = MingFlashOmniModel(model_path_hf=snap, ming_code_dir=code_dir)

    # bf16 on CPU is slow for matmuls; override autocast dtype to fp32
    # for the test by monkeypatching get_autocast_dtype.
    model.get_autocast_dtype = lambda: torch.float32  # type: ignore

    sub = model.get_submodule("Talker", device="cpu")
    assert isinstance(sub, TalkerSubmodule)

    # Cap generation hard so the test is fast.
    sub.max_steps = 2
    sub.min_new_token = 1000   # force max_steps cap (no early stop)

    # A short token sequence in the talker LLM's vocab.
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    prep = sub.prepare_inputs(
        graph_walk="talker", fwd_info=None,
        inputs={"talker_text_inputs": [token_ids]},
    )
    out = sub.forward(
        graph_walk="talker", engine_inputs=None,
        inputs_embeds=prep.tensor_inputs["inputs_embeds"],
        prompt_wav_lat=None,
    )
    wf = out["audio_chunk"][0]
    assert wf.dim() == 3 and wf.shape[1] == 1
    assert torch.isfinite(wf).all()
