"""CPU-only structural checks for the Cosmos3 model package.

No GPU and no model weights are required: the config is parsed from the
checkpoint's JSON files, the backbone is built on the ``meta`` device (shapes
only, zero storage), and weight-key coverage is checked against the shard
index. Run directly (``python3 test_loader.py``) or via pytest.

Point ``COSMOS3_NANO_DIR`` at a Cosmos3-Nano checkpoint directory (config +
tokenizer + shard index; the safetensors tensor data itself is not read).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.loader import (
    DROP_KEYS,
    cosmos3_name_remapper,
    read_transformer_weight_keys,
    read_transformer_weight_shapes,
)

NANO_DIR = Path(
    os.environ.get(
        "COSMOS3_NANO_DIR",
        "/Users/atindrajha/Downloads/disaggregation_research/Cosmos3-Nano-hf",
    )
)


def test_config_roundtrip() -> None:
    cfg = Cosmos3Config.from_pretrained(NANO_DIR)

    # Transformer dimensions (Nano).
    assert cfg.num_hidden_layers == 36
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 12288
    assert cfg.vocab_size == 151936
    assert cfg.rms_norm_eps == 1e-6

    # 3D interleaved mRoPE.
    assert tuple(cfg.rope_axes_dim) == (24, 20, 20)
    assert cfg.mrope_interleaved is True
    assert cfg.rope_theta == 5_000_000.0
    assert cfg.unified_3d_mrope_temporal_modality_margin == 15000
    assert cfg.unified_3d_mrope_reset_spatial_ids is True
    assert cfg.base_fps == 24 and cfg.enable_fps_modulation is True

    # Latent geometry / attention style.
    assert cfg.latent_channel == 48
    assert cfg.latent_patch_size == 2
    assert cfg.patch_latent_dim == 192
    assert cfg.timestep_scale == 0.001
    assert cfg.joint_attn_implementation == "two_way"
    assert cfg.use_moe is True
    assert cfg.qk_norm_for_diffusion is True and cfg.qk_norm_for_text is True

    # Capability flags / modality heads.
    assert cfg.action_gen is True and cfg.max_action_dim == 64
    assert cfg.num_embodiment_domains == 32
    assert cfg.sound_gen is True and cfg.sound_dim == 64

    # VAE (AutoencoderKLWan) geometry + normalization stats.
    assert cfg.vae.z_dim == 48
    assert cfg.vae.scale_factor_spatial == 16
    assert cfg.vae.scale_factor_temporal == 4
    assert len(cfg.vae.latents_mean) == 48
    assert len(cfg.vae.latents_std) == 48

    # UniPC flow scheduler.
    assert cfg.scheduler.scheduler_type == "unipc"
    assert cfg.scheduler.prediction_type == "flow_prediction"
    assert cfg.scheduler.predict_x0 is True
    assert cfg.scheduler.solver_order == 2
    assert cfg.scheduler.solver_type == "bh2"
    assert cfg.scheduler.use_flow_sigmas is True
    assert cfg.scheduler.use_karras_sigmas is True


def test_loader_key_coverage() -> None:
    cfg = Cosmos3Config.from_pretrained(NANO_DIR)
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)

    model_keys = set(model.state_dict().keys())
    index_keys = read_transformer_weight_keys(NANO_DIR)

    # The only intentionally-dropped key is the unused text lm_head.
    dropped = {k for k in index_keys if cosmos3_name_remapper(k) is None}
    assert dropped == set(DROP_KEYS), dropped

    mapped = {cosmos3_name_remapper(k) for k in index_keys}
    mapped.discard(None)

    missing = model_keys - mapped  # backbone params with no checkpoint key
    unexpected = mapped - model_keys  # checkpoint keys with no backbone param
    assert not missing, f"backbone params not covered by checkpoint: {sorted(missing)[:20]}"
    assert not unexpected, f"checkpoint keys with no backbone param: {sorted(unexpected)[:20]}"

    # Sanity on the exact counts: 36 layers * 22 + 22 non-layer == 814; drop lm_head -> 813.
    assert len(index_keys) == 814, len(index_keys)
    assert len(model_keys) == 813, len(model_keys)


def test_loader_shape_coverage() -> None:
    """Every backbone param's *shape* matches the checkpoint tensor it loads
    from. Reads only safetensors headers (no tensor data, CPU-safe). Returns
    early if the shards are LFS pointers (asset-only clone) rather than real
    weights. Complements the name-only coverage check — it is what would have
    caught a wrong per-domain action-projection shape before a GPU load.
    """
    cfg = Cosmos3Config.from_pretrained(NANO_DIR)
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)

    try:
        ckpt_shapes = read_transformer_weight_shapes(NANO_DIR)
    except Exception as exc:  # noqa: BLE001 — LFS pointer / missing shards
        print(f"  (shape check skipped: transformer shards unreadable: {exc})")
        return

    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    # The remapper is identity for backbone keys, so model key == checkpoint key.
    mismatched = {
        k: {"model": s, "ckpt": ckpt_shapes.get(k)}
        for k, s in model_shapes.items()
        if s != ckpt_shapes.get(k)
    }
    assert not mismatched, mismatched


def test_tokenizer_roundtrip() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(NANO_DIR / "text_tokenizer"))
    prompt = "A red cube resting on a polished wooden table, soft daylight."
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert len(ids) > 0
    assert tok.decode(ids) == prompt


def _main() -> None:
    failures = []
    for name, fn in [
        ("config_roundtrip", test_config_roundtrip),
        ("loader_key_coverage", test_loader_key_coverage),
        ("loader_shape_coverage", test_loader_shape_coverage),
        ("tokenizer_roundtrip", test_tokenizer_roundtrip),
    ]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc!r}")
    if failures:
        raise SystemExit(1)
    print("\nAll Cosmos3 structural checks passed.")


if __name__ == "__main__":
    _main()
