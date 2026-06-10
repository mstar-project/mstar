"""Snapshot-gated tests for the Talker + AudioVAE weight loaders (step 6f).

The talker checkpoint lives in two safetensors files:

  talker/model.safetensors        — model.* / cfm.* / aggregator.*
                                    / stop_head.* / spk_head.*
  talker/vae/model.safetensors    — encoder.* + decoder.* (AudioVAE)

Each loader is non-TP and just does prefix-strip + load_state_dict
via the shared `_load_prefixed_state_dict` helper. These tests skip
cleanly when no snapshot is available, so CI machines without the
~5GB talker download still pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mminf.model.ming_omni_flash.components.audio_vae import build_audio_vae
from mminf.model.ming_omni_flash.components.talker_dit import (
    build_aggregator,
    build_talker_cfm,
    build_talker_heads,
    build_talker_llm,
)
from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
from mminf.model.ming_omni_flash.loader import (
    load_talker_aggregator_weights,
    load_talker_audio_vae_weights,
    load_talker_cfm_weights,
    load_talker_heads_weights,
    load_talker_llm_weights,
)

# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------


def _find_local_snapshot() -> str | None:
    def _has(p: Path) -> bool:
        return (
            (p / "talker" / "config.json").exists()
            and (p / "talker" / "model.safetensors").exists()
        )
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and _has(Path(override)):
        return override
    hybrid = Path("/dev/shm/ming-hybrid")
    if _has(hybrid):
        return str(hybrid)
    return None


pytestmark = pytest.mark.skipif(
    _find_local_snapshot() is None,
    reason="Need Ming-flash-omni-2.0 snapshot with talker/.",
)


@pytest.fixture(scope="module")
def snap_and_config() -> tuple[str, MingFlashOmniModelConfig]:
    snap = _find_local_snapshot()
    cfg = MingFlashOmniModelConfig.from_pretrained(snap)
    if cfg.talker is None:
        pytest.skip("Snapshot has no talker/ subdir.")
    return snap, cfg


# ---------------------------------------------------------------------------
# Talker LLM (Qwen2)
# ---------------------------------------------------------------------------


def test_load_talker_llm_weights_strict(snap_and_config) -> None:
    """``model.*`` from talker/model.safetensors loads cleanly into Qwen2Model."""
    snap, cfg = snap_and_config
    llm = build_talker_llm(cfg.talker.llm, dtype=torch.float32, device="cpu")
    loaded = load_talker_llm_weights(llm, snap, device="cpu", strict=True)
    # 24 layers × ~12 params each + embed + final norm = many keys; just
    # spot-check representative entries.
    assert "embed_tokens.weight" in loaded
    assert "layers.0.self_attn.q_proj.weight" in loaded
    assert "layers.0.mlp.gate_proj.weight" in loaded
    assert f"layers.{cfg.talker.llm.num_hidden_layers - 1}.input_layernorm.weight" in loaded
    assert "norm.weight" in loaded
    # Sanity-check that the embed table actually got overwritten.
    assert (llm.embed_tokens.weight.abs().sum() > 0).item()


# ---------------------------------------------------------------------------
# CFM
# ---------------------------------------------------------------------------


def test_load_talker_cfm_weights_strict(snap_and_config) -> None:
    """``cfm.*`` loads into `CFM(DiT)` by state-dict equality."""
    snap, cfg = snap_and_config
    cfm = build_talker_cfm(cfg.talker, dtype=torch.float32, device="cpu")
    loaded = load_talker_cfm_weights(cfm, snap, device="cpu", strict=True)
    # CFM module wraps a DiT under `.model`, so the loaded keys are
    # ``model.<...>`` after stripping the ``cfm.`` prefix.
    assert "model.x_embedder.weight" in loaded
    assert "model.c_embedder.cond_embedder.weight" in loaded
    assert "model.t_embedder.time_embed.dim" not in loaded   # buffer-free
    assert "model.blocks.0.attn.to_q.weight" in loaded
    assert "model.blocks.0.mlp.ff.0.0.weight" in loaded
    assert "model.final_layer.linear.weight" in loaded


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_load_talker_aggregator_weights_strict(snap_and_config) -> None:
    snap, cfg = snap_and_config
    agg = build_aggregator(cfg.talker, dtype=torch.float32, device="cpu")
    loaded = load_talker_aggregator_weights(agg, snap, device="cpu", strict=True)
    assert "x_embedder.weight" in loaded
    assert "word_embedder.weight" in loaded
    assert "blocks.0.attn.to_q.weight" in loaded
    assert "final_layer.linear.weight" in loaded


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def test_load_talker_heads_weights_strict(snap_and_config) -> None:
    """stop_head and spk_head both load by leaf prefix."""
    snap, cfg = snap_and_config
    heads = build_talker_heads(cfg.talker, dtype=torch.float32, device="cpu")
    loaded = load_talker_heads_weights(heads, snap, device="cpu", strict=True)
    assert loaded["stop_head"] == {"weight", "bias"}
    assert loaded["spk_head"] == {"weight", "bias"}
    # Sanity: head weights are not the init values.
    assert (heads["stop_head"].weight.abs().sum() > 0).item()
    assert (heads["spk_head"].weight.abs().sum() > 0).item()


def test_load_talker_heads_weights_rejects_missing_key() -> None:
    """KeyError if the heads dict is missing one of the required entries.

    Use a dict missing ``stop_head`` (the first entry the loader checks)
    so the missing-key guard fires before we attempt any disk I/O.
    """
    with pytest.raises(KeyError, match="missing required key 'stop_head'"):
        load_talker_heads_weights({"spk_head": torch.nn.Linear(1, 1)}, "/tmp/x")


# ---------------------------------------------------------------------------
# AudioVAE
# ---------------------------------------------------------------------------


def test_load_talker_audio_vae_weights_strict(snap_and_config) -> None:
    """Full AudioVAE state_dict round-trips from talker/vae/model.safetensors."""
    snap, cfg = snap_and_config
    vae = build_audio_vae(
        cfg.talker.vae, dtype=torch.float32, device="cpu",
        attn_implementation="sdpa",
    )
    loaded = load_talker_audio_vae_weights(vae, snap, device="cpu", strict=True)
    # Encoder + decoder subtrees both present.
    assert "encoder.fc1.weight" in loaded
    assert "encoder.encoder.embed_tokens.weight" in loaded
    assert "encoder.aggregator.embed_tokens.weight" in loaded
    assert "encoder.cls_embed" in loaded
    assert "decoder.fc1.weight" in loaded
    assert "decoder.head.out.weight" in loaded
    assert "decoder.head.istft.window" in loaded
    assert "decoder.decoder.embed_tokens.weight" in loaded


def test_audio_vae_decode_runs_with_loaded_weights(snap_and_config) -> None:
    """End-to-end CPU smoke after a real-weights load.

    Constructs a small latent and decodes; checks the output is finite.
    Catches catastrophic dtype / weight-layout misloads that wouldn't
    surface from key-name parity alone.
    """
    snap, cfg = snap_and_config
    vae = build_audio_vae(
        cfg.talker.vae, dtype=torch.float32, device="cpu",
        attn_implementation="sdpa",
    )
    load_talker_audio_vae_weights(vae, snap, device="cpu", strict=True)

    # One latent frame at latent_dim=64.
    latent = torch.randn(1, 1, cfg.talker.vae.latent_dim) * 0.1
    with torch.no_grad():
        waveform, state, pkv = vae.decode(latent, use_cache=False)
    assert waveform.dim() == 3
    assert waveform.shape[0] == 1 and waveform.shape[1] == 1
    assert torch.isfinite(waveform).all()
