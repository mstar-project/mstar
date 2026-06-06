"""Smoke tests for Ming-flash-omni-2.0 config loading.

These tests run against the released checkpoint
(``inclusionAI/Ming-flash-omni-2.0``). They skip cleanly when no local
snapshot is available, so CI / dev machines without the 222 GB download
still pass.

Snapshot discovery order:
  1. ``MING_FLASH_OMNI_DIR`` env var (explicit override)
  2. The default HF Hub cache layout under ``~/.cache/huggingface/hub/``
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    ImageGenConfig,
    MingFlashOmniModelConfig,
    TalkerConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)


def _find_local_snapshot() -> str | None:
    """Locate a Ming-flash-omni-2.0 snapshot on disk, or None."""
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and (Path(override) / "config.json").exists():
        return override

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_root / "models--inclusionAI--Ming-flash-omni-2.0" / "snapshots"
    if not repo_dir.exists():
        return None
    # Pick the first snapshot dir that has a config.json (HF stores one per
    # commit revision; usually there's only one).
    for snap in sorted(repo_dir.iterdir()):
        if (snap / "config.json").exists():
            return str(snap)
    return None


@pytest.fixture(scope="module")
def snapshot_dir() -> str:
    snap = _find_local_snapshot()
    if snap is None:
        pytest.skip(
            "Ming-flash-omni-2.0 snapshot not found. Set MING_FLASH_OMNI_DIR "
            "or download with `huggingface-cli download "
            "inclusionAI/Ming-flash-omni-2.0`."
        )
    return snap


@pytest.fixture(scope="module")
def config(snapshot_dir: str) -> MingFlashOmniModelConfig:
    return MingFlashOmniModelConfig.from_pretrained(snapshot_dir)


def test_from_pretrained_loads_thinker_dims(config: MingFlashOmniModelConfig) -> None:
    """Released ckpt: Ling-2.0 32L, 4096-hidden, 256-expert MoE, head_dim=128."""
    llm = config.thinker_llm
    assert llm.vocab_size == 157184
    assert llm.hidden_size == 4096
    assert llm.intermediate_size == 9216
    assert llm.num_hidden_layers == 32
    assert llm.num_attention_heads == 32
    assert llm.num_key_value_heads == 4
    assert llm.head_dim == 128
    assert llm.rope_theta == 2_400_000.0
    assert llm.num_experts == 256
    assert llm.num_experts_per_tok == 8
    assert llm.moe_intermediate_size == 1024
    assert llm.first_k_dense_replace == 1
    assert llm.router_type == "MultiRouter"
    assert llm.use_qk_norm is True

    # Convenience accessors used by the rest of mminf
    assert config.thinker_hidden_size == 4096
    assert config.thinker_num_layers == 32
    assert config.thinker_head_dim == 128
    assert config.thinker_num_kv_heads == 4
    assert config.vocab_size == 157184


def test_from_pretrained_loads_vision_audio(config: MingFlashOmniModelConfig) -> None:
    """Released ckpt: Qwen3-MoE ViT (27L, out_hidden=4096) + Whisper-style audio."""
    assert config.vision.depth == 27
    assert config.vision.hidden_size == 1152
    assert config.vision.out_hidden_size == 4096
    assert config.vision.deepstack_visual_indexes == (8, 16, 24)
    assert config.vision.spatial_merge_size == 2
    assert config.vision.patch_size == 16
    assert config.vision.hidden_act == "gelu_pytorch_tanh"

    audio = config.audio_encoder
    assert audio.encoder_layers == 32
    assert audio.d_model == 1280
    assert audio.encoder_attention_heads == 20
    assert audio.n_mels == 128
    assert audio.ds_kernel_size == 3
    assert audio.ds_stride == 2
    assert audio.norm_query_embeds is True


def test_mrope_section_sums_to_half_rotary_dims(config: MingFlashOmniModelConfig) -> None:
    """Regression guard on the MRoPE arithmetic.

    sum(mrope_section) must equal (head_dim * partial_rotary_factor) / 2 —
    the rotary subset of each head is paired (cos, sin), so the section
    partitions one half. For Ming-flash-omni-2.0: 128 * 0.5 / 2 = 32, and
    the released ckpt sets mrope_section = [8, 12, 12].
    """
    llm = config.thinker_llm
    assert llm.head_dim is not None
    rotary_pair_dims = int(llm.head_dim * llm.partial_rotary_factor) // 2
    assert sum(llm.mrope_section) == rotary_pair_dims, (
        f"mrope_section {llm.mrope_section} sums to {sum(llm.mrope_section)}, "
        f"expected {rotary_pair_dims}"
    )


def test_subdir_configs_load_when_present(config: MingFlashOmniModelConfig) -> None:
    """talker/ and the imagegen subdir family populate when present."""
    assert config.talker is not None, "talker/config.json should have populated"
    assert config.talker.vae_sample_rate == 44100
    assert config.talker.patch_size == 4
    assert config.talker.history_patch_size == 32
    # llm/ dict load
    assert config.talker.llm is not None
    assert config.talker.llm.get("model_type") == "qwen2"
    # vae/ dict load
    assert config.talker.vae is not None
    assert config.talker.vae.get("sample_rate") == 44100

    assert config.image_gen is not None, "imagegen subdirs should have populated"
    assert config.image_gen.num_query_tokens == 256  # img_gen_scales=[16] => 16*16
    assert config.image_gen.diffusion_c_input_dim == 2560
    assert config.image_gen.text_encoder_norm is True


def test_subdir_configs_absent_returns_none() -> None:
    """A snapshot dir with only a stripped-down config.json yields
    talker=None and image_gen=None."""
    minimal = {
        "llm_config": {"hidden_size": 4096, "num_attention_heads": 32, "vocab_size": 157184},
        "vision_config": {"depth": 27, "out_hidden_size": 4096},
        "audio_config": {
            "ds_kernel_size": 3, "ds_stride": 2, "norm_query_embeds": True,
            "whisper_encoder_config": {
                "n_ctx": 15000, "n_head": 20, "n_layer": 32, "n_mels": 128, "n_state": 1280,
            },
        },
        "mlp_depth": 2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(minimal))
        c = MingFlashOmniModelConfig.from_pretrained(tmp)
    assert c.talker is None
    assert c.image_gen is None


def test_sub_config_from_dict_filters_unknown_keys() -> None:
    """from_dict should silently drop keys the dataclass doesn't declare,
    so checkpoints that add new fields don't break loading."""
    # Released ThinkerLLMConfig doesn't carry e.g. ``some_future_field``; that
    # key must be silently dropped, not raise.
    cfg = ThinkerLLMConfig.from_dict({
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "some_future_field": "ignored",
    })
    assert cfg.hidden_size == 4096
    assert not hasattr(cfg, "some_future_field")

    vis = VisionEncoderConfig.from_dict({"depth": 27, "deepstack_visual_indexes": [1, 2, 3]})
    assert vis.deepstack_visual_indexes == (1, 2, 3)

    aud = AudioEncoderConfig.from_dict({"ds_stride": 4, "irrelevant": True})
    assert aud.ds_stride == 4


def test_invariant_check_rejects_out_of_vocab_multimodal_tokens() -> None:
    """__post_init__ should refuse a config whose multimodal token IDs
    are outside the vocabulary range — that pattern silently causes a
    CUDA device-side assert at embedding-lookup time."""
    bad = ThinkerLLMConfig(
        vocab_size=1000,
        image_patch_token=2000,  # > vocab_size
    )
    with pytest.raises(ValueError, match="image_patch_token"):
        MingFlashOmniModelConfig(thinker_llm=bad)


def test_invariant_check_rejects_bad_mrope_section() -> None:
    """Wrong mrope_section partition is exactly the kind of silent miswire
    we want loud failure on."""
    bad_llm = ThinkerLLMConfig(
        rope_scaling={"type": "video_rope", "mrope_section": [16, 16, 16]},  # sums to 48, expected 32
    )
    with pytest.raises(ValueError, match="MRoPE section"):
        MingFlashOmniModelConfig(thinker_llm=bad_llm)


def test_imagegen_skeleton_defaults() -> None:
    """The image-gen skeleton should produce a usable instance even before
    any subdir reads (downstream code may want to read default subfolder
    names / sampling defaults without touching disk)."""
    ig = ImageGenConfig()
    assert ig.num_query_tokens == 256
    assert ig.transformer_subfolder == "transformer"
    assert ig.byt5_subfolder == "byt5"
    assert ig.num_inference_steps == 30
    assert ig.guidance_scale == 2.0


def test_talker_from_subdir_returns_none_for_missing_dir() -> None:
    """Missing talker/ subdir must return None, not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        assert TalkerConfig.from_subdir(Path(tmp) / "talker") is None
