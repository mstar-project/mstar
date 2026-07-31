"""CPU-only checks for serving Cosmos3-Nano-Policy-DROID.

The registry/config wiring checks always run. The checkpoint-structure checks
parse the DROID checkpoint's JSON files and build the backbone on the ``meta``
device (shapes only, zero storage) — point ``COSMOS3_DROID_DIR`` at a
Cosmos3-Nano-Policy-DROID directory to run them (skipped otherwise).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.loader import (
    DROP_KEYS,
    cosmos3_name_remapper,
    read_transformer_weight_keys,
    read_transformer_weight_shapes,
)

DROID_DIR = Path(os.environ.get("COSMOS3_DROID_DIR", "/nonexistent-cosmos3-droid"))

needs_droid = pytest.mark.skipif(
    not DROID_DIR.exists(), reason="set COSMOS3_DROID_DIR to a Cosmos3-Nano-Policy-DROID dir"
)


def test_droid_registry_and_config_wiring() -> None:
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.cli.main import DEFAULT_CONFIGS
    from mstar.model.registry import HF_MODELS, MODEL_REGISTRY

    assert MODEL_REGISTRY["cosmos3_droid"] is Cosmos3Model
    assert HF_MODELS["cosmos3_droid"]["model_path_hf"] == "nvidia/Cosmos3-Nano-Policy-DROID"

    adapter = get_adapter("cosmos3_droid")
    assert adapter is not None and adapter.supports_images and adapter.supports_videos

    import mstar

    yaml_path = Path(mstar.__file__).resolve().parent.parent / "configs" / DEFAULT_CONFIGS["cosmos3_droid"]
    assert yaml_path.exists(), yaml_path

    import yaml

    cfg = yaml.safe_load(yaml_path.read_text())
    assert cfg["model"] == "cosmos3_droid"
    mk = cfg["model_kwargs"]
    # The released DROID policy sampling defaults, and the explicit statement
    # that this checkpoint serves no sound.
    assert mk["enable_sound"] is False
    assert mk["num_inference_steps_action"] == 4
    assert mk["guidance_scale_action"] == 3.0
    node_names = {n for g in cfg["node_groups"] for n in g["node_names"]}
    assert "audio_decoder" not in node_names
    assert {"dit", "vae_encoder", "vae_decoder"} <= node_names


@needs_droid
def test_droid_config_roundtrip() -> None:
    cfg = Cosmos3Config.from_pretrained(DROID_DIR)

    # Nano-identical transformer dimensions.
    assert cfg.num_hidden_layers == 36
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 12288

    # Capability flags: the action pathway stays, the sound pathway is gone.
    assert cfg.action_gen is True
    assert cfg.max_action_dim == 64 and cfg.num_embodiment_domains == 32
    assert cfg.sound_gen is False
    assert cfg.sound_dim is None

    # VAE + scheduler match the Nano components.
    assert cfg.vae.z_dim == 48
    assert cfg.vae.scale_factor_spatial == 16 and cfg.vae.scale_factor_temporal == 4
    assert len(cfg.vae.latents_mean) == 48 and len(cfg.vae.latents_std) == 48
    assert cfg.scheduler.scheduler_type == "unipc"
    assert cfg.scheduler.flow_shift == 1.0


@needs_droid
def test_droid_loader_key_coverage() -> None:
    cfg = Cosmos3Config.from_pretrained(DROID_DIR)
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)

    model_keys = set(model.state_dict().keys())
    index_keys = read_transformer_weight_keys(DROID_DIR)

    # No audio projections in this checkpoint or in the built backbone.
    assert not any("audio" in k for k in index_keys)
    assert not any("audio" in k for k in model_keys)

    dropped = {k for k in index_keys if cosmos3_name_remapper(k) is None}
    assert dropped == set(DROP_KEYS), dropped

    mapped = {cosmos3_name_remapper(k) for k in index_keys}
    mapped.discard(None)
    missing = model_keys - mapped
    unexpected = mapped - model_keys
    assert not missing, f"backbone params not covered by checkpoint: {sorted(missing)[:20]}"
    assert not unexpected, f"checkpoint keys with no backbone param: {sorted(unexpected)[:20]}"

    # Nano's counts minus the five audio keys (modality embed + proj_in/out
    # weight and bias): 814 - 5 == 809 in the index, 813 - 5 == 808 built.
    assert len(index_keys) == 809, len(index_keys)
    assert len(model_keys) == 808, len(model_keys)


@needs_droid
def test_droid_loader_shape_coverage() -> None:
    cfg = Cosmos3Config.from_pretrained(DROID_DIR)
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)

    try:
        ckpt_shapes = read_transformer_weight_shapes(DROID_DIR)
    except Exception as exc:  # noqa: BLE001 — LFS pointer / missing shards
        pytest.skip(f"transformer shards unreadable: {exc}")

    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    mismatched = {
        k: {"model": s, "ckpt": ckpt_shapes.get(k)}
        for k, s in model_shapes.items()
        if s != ckpt_shapes.get(k)
    }
    assert not mismatched, mismatched


@needs_droid
def test_droid_sound_pathway_disabled() -> None:
    from mstar.engine.base import EngineType
    from mstar.model.cosmos3 import constants

    model = Cosmos3Model(model_path_hf=str(DROID_DIR))
    assert model.config.sound_gen is False
    assert model._sound_serving_enabled() is False

    # The sound walk and the audio_decoder node are not declared, so a config
    # without them serves every remaining walk.
    walks = model.get_graph_walk_graphs()
    assert constants.VIDEO_SOUND_GEN_WALK not in walks
    assert constants.ACTION_GEN_WALK in walks and constants.ACTION_VIDEO_GEN_WALK in walks
    types = model.get_node_engine_types()
    assert "audio_decoder" not in types
    assert types["dit"] is EngineType.KV_CACHE

    # A sound-requesting video request is rejected up front.
    with pytest.raises(ValueError, match="sound"):
        model._resolve_gen_params(
            {"num_frames": 17, "generate_sound": True}, ["text"], ["video"]
        )
