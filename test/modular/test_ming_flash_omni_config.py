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
    # Step 6a: llm + vae are typed dataclasses (used to be raw dicts).
    assert config.talker.llm.hidden_size == 896
    assert config.talker.llm.num_hidden_layers == 24
    assert config.talker.llm.num_key_value_heads == 2
    assert config.talker.vae.sample_rate == 44100
    assert config.talker.vae.latent_dim == 64
    # flowmodel + aggregator share shape; only dropout differs (0 vs 0.1).
    assert config.talker.flowmodel.depth == 8
    assert config.talker.flowmodel.hidden_size == 1024
    assert config.talker.aggregator.dropout == pytest.approx(0.1)

    assert config.image_gen is not None, "imagegen subdirs should have populated"
    assert config.image_gen.num_query_tokens == 256  # img_gen_scales=[16] => 16*16
    assert config.image_gen.diffusion_c_input_dim == 2560
    assert config.image_gen.text_encoder_norm is True
    # Step 9a: typed sub-configs parsed from the imagegen subdir tree.
    assert config.image_gen.dit.dim == 3840
    assert config.image_gen.dit.n_layers == 30
    assert config.image_gen.dit.in_channels == 16
    assert config.image_gen.dit.axes_dims == (32, 48, 48)
    assert config.image_gen.vae.latent_channels == 16
    assert config.image_gen.vae.scaling_factor == pytest.approx(0.3611)
    assert config.image_gen.scheduler.shift == pytest.approx(3.0)
    assert config.image_gen.byt5.sdxl_channels == 2560
    assert config.image_gen.byt5.byt5_name == "google/byt5-small"
    # Connector is a Qwen2 LLM kept as a raw dict.
    assert config.image_gen.connector is not None
    assert config.image_gen.connector.get("model_type") == "qwen2"


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


def test_invariant_check_covers_audio_and_end_tokens() -> None:
    """The vocab-bounds check must cover every multimodal token field,
    not just the four the ckpt ships. Regression for the audio + *_end
    tokens added alongside the vision/audio encoder port."""
    for field, bad_value in [
        ("audio_patch_token", 200_000),
        ("audio_start_token", 200_000),
        ("audio_end_token", 200_000),
        ("image_end_token", 200_000),
        ("video_end_token", 200_000),
    ]:
        bad = ThinkerLLMConfig(vocab_size=160_000, **{field: bad_value})
        with pytest.raises(ValueError, match=field):
            MingFlashOmniModelConfig(thinker_llm=bad)


def test_video_start_token_mislabel_auto_repaired(caplog: pytest.LogCaptureFixture) -> None:
    """The inclusionAI ckpt's llm_config.video_start_token=157159 is
    actually `</image>` per the tokenizer; the real `<video>` token is
    157160. ThinkerLLMConfig.__post_init__ must repair the bogus value
    AND emit a warning so the user sees what happened.
    """
    import logging
    with caplog.at_level(logging.WARNING):
        cfg = ThinkerLLMConfig.from_dict({
            # Mimic the on-disk inclusionAI llm_config (minus head_dim noise).
            "hidden_size": 4096, "num_attention_heads": 32, "vocab_size": 160_000,
            "image_start_token": 157158,
            "video_start_token": 157159,  # bogus per ckpt
        })
    # Repaired in place to the tokenizer-truth value.
    assert cfg.video_start_token == 157160, (
        f"video_start_token should auto-repair from 157159 to 157160; got {cfg.video_start_token}"
    )
    assert any("video_start_token=157159" in rec.message for rec in caplog.records), \
        "expected a warning about the ckpt mislabel"

    # If video_start_token is set to anything else (whether the corrected
    # 157160 or a custom value), the repair must NOT fire and the value
    # must pass through untouched.
    cfg_ok = ThinkerLLMConfig(video_start_token=157160)
    assert cfg_ok.video_start_token == 157160
    cfg_custom = ThinkerLLMConfig(video_start_token=99_999, image_end_token=42)
    assert cfg_custom.video_start_token == 99_999


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


# ---------------------------------------------------------------------------
# Step 6a: TalkerLLMConfig / DiTBlockConfig / AudioVAEConfig
# ---------------------------------------------------------------------------


def test_talker_llm_config_defaults_match_released_ckpt() -> None:
    """Defaults track the released talker/llm/config.json values."""
    from mminf.model.ming_omni_flash.config import TalkerLLMConfig
    llm = TalkerLLMConfig()
    assert llm.vocab_size == 151936
    assert llm.hidden_size == 896
    assert llm.intermediate_size == 4864
    assert llm.num_hidden_layers == 24
    assert llm.num_attention_heads == 14
    assert llm.num_key_value_heads == 2
    assert llm.head_dim == 64  # 896 / 14
    assert llm.rope_theta == 1_000_000.0
    assert llm.tie_word_embeddings is True


def test_talker_llm_config_from_dict_filters_unknown_keys() -> None:
    """Released config has fields we don't model (`transformers_version` etc.)
    — `from_dict` must silently ignore them."""
    from mminf.model.ming_omni_flash.config import TalkerLLMConfig
    llm = TalkerLLMConfig.from_dict({
        "hidden_size": 1024,
        "transformers_version": "4.43.1",
        "_attn_implementation": "flash_attention_2",
    })
    assert llm.hidden_size == 1024
    assert llm.num_hidden_layers == 24  # default preserved


def test_dit_block_config_intermediate_size_and_head_dim() -> None:
    from mminf.model.ming_omni_flash.config import DiTBlockConfig
    blk = DiTBlockConfig()
    assert blk.intermediate_size == 1024 * 4
    assert blk.head_dim == 1024 // 16


def test_audio_vae_config_lifts_latent_and_output_dims_from_kwargs() -> None:
    """`enc_kwargs.latent_dim` and `dec_kwargs.output_dim` get pulled out."""
    from mminf.model.ming_omni_flash.config import AudioVAEConfig
    cfg = AudioVAEConfig.from_dict({
        "sample_rate": 44100,
        "patch_size": 4,
        "enc_kwargs": {
            "latent_dim": 64,
            "input_dim": 80,
            "hop_size": 320,
            "backbone": {"hidden_size": 896, "num_hidden_layers": 24},
        },
        "dec_kwargs": {
            "latent_dim": 64,
            "output_dim": 882,
            "backbone": {"hidden_size": 896, "num_hidden_layers": 24},
        },
        "init_method": "kaiming",
        "lambda_mel_loss": 1.0,
    })
    assert cfg.sample_rate == 44100
    assert cfg.latent_dim == 64
    assert cfg.encoder_input_dim == 80
    assert cfg.encoder_hop_size == 320
    assert cfg.decoder_output_dim == 882
    assert cfg.enc_backbone["hidden_size"] == 896
    assert cfg.dec_backbone["num_hidden_layers"] == 24
    assert cfg.lambda_mel_loss == 1.0


def test_audio_vae_config_falls_back_when_enc_kwargs_missing_latent() -> None:
    """If enc_kwargs has no latent_dim, fall back to dec_kwargs.latent_dim."""
    from mminf.model.ming_omni_flash.config import AudioVAEConfig
    cfg = AudioVAEConfig.from_dict({
        "enc_kwargs": {"input_dim": 80, "hop_size": 320},
        "dec_kwargs": {"latent_dim": 128, "output_dim": 512},
    })
    assert cfg.latent_dim == 128


def test_talker_config_from_subdir_typed_subfields() -> None:
    """from_subdir produces typed TalkerLLMConfig / DiTBlockConfig / AudioVAEConfig."""
    from mminf.model.ming_omni_flash.config import (
        AudioVAEConfig,
        DiTBlockConfig,
        TalkerConfig,
        TalkerLLMConfig,
    )
    with tempfile.TemporaryDirectory() as tmp:
        talker_dir = Path(tmp) / "talker"
        talker_dir.mkdir()
        # Minimal valid config.json (top-level scalars + flowmodel + aggregator).
        (talker_dir / "config.json").write_text(json.dumps({
            "steps": 12,
            "patch_size": 8,
            "history_patch_size": 64,
            "cfg_strength": 1.5,
            "flowmodel": {"depth": 4, "hidden_size": 512, "num_heads": 8, "dropout": 0.0},
            "aggregator": {"depth": 4, "hidden_size": 512, "num_heads": 8, "dropout": 0.1},
        }))
        (talker_dir / "llm").mkdir()
        (talker_dir / "llm" / "config.json").write_text(json.dumps({
            "hidden_size": 512, "num_hidden_layers": 12,
        }))
        (talker_dir / "vae").mkdir()
        (talker_dir / "vae" / "config.json").write_text(json.dumps({
            "sample_rate": 22050,
            "patch_size": 2,
            "enc_kwargs": {"latent_dim": 32, "input_dim": 80, "hop_size": 256},
            "dec_kwargs": {"latent_dim": 32, "output_dim": 401},
        }))

        cfg = TalkerConfig.from_subdir(talker_dir)
        assert cfg is not None
        assert cfg.steps == 12
        assert isinstance(cfg.llm, TalkerLLMConfig)
        assert cfg.llm.hidden_size == 512
        assert isinstance(cfg.flowmodel, DiTBlockConfig)
        assert cfg.flowmodel.depth == 4
        assert cfg.aggregator.dropout == pytest.approx(0.1)
        assert isinstance(cfg.vae, AudioVAEConfig)
        assert cfg.vae.sample_rate == 22050
        assert cfg.vae.latent_dim == 32
        # Convenience accessor still works.
        assert cfg.vae_sample_rate == 22050


def test_talker_config_default_factories_yield_real_dataclasses() -> None:
    """``TalkerConfig()`` with no args still produces typed sub-configs."""
    from mminf.model.ming_omni_flash.config import (
        AudioVAEConfig,
        DiTBlockConfig,
        TalkerConfig,
        TalkerLLMConfig,
    )
    t = TalkerConfig()
    assert isinstance(t.llm, TalkerLLMConfig)
    assert isinstance(t.flowmodel, DiTBlockConfig)
    assert isinstance(t.aggregator, DiTBlockConfig)
    assert isinstance(t.vae, AudioVAEConfig)
    assert t.vae_sample_rate == 44100   # convenience property


# ---------------------------------------------------------------------------
# Step 9a: ImageGen typed sub-configs (pure-Python)
# ---------------------------------------------------------------------------


def test_zimage_dit_config_from_dict_coerces_tuples_and_filters() -> None:
    from mminf.model.ming_omni_flash.config import ZImageDiTConfig
    dit = ZImageDiTConfig.from_dict({
        "dim": 3840, "n_layers": 30, "in_channels": 16,
        "axes_dims": [32, 48, 48], "axes_lens": [1536, 512, 512],
        "all_patch_size": [2], "_class_name": "ZImageTransformer2DModel",
    })
    assert dit.dim == 3840
    assert dit.axes_dims == (32, 48, 48)
    assert dit.axes_lens == (1536, 512, 512)
    assert dit.all_patch_size == (2,)
    assert not hasattr(dit, "_class_name")


def test_image_vae_config_defaults_and_from_dict() -> None:
    from mminf.model.ming_omni_flash.config import ImageVAEConfig
    vae = ImageVAEConfig.from_dict({
        "latent_channels": 16, "scaling_factor": 0.3611, "shift_factor": 0.1159,
        "act_fn": "silu", "ignored": 1,
    })
    assert vae.latent_channels == 16
    assert vae.scaling_factor == pytest.approx(0.3611)
    assert vae.shift_factor == pytest.approx(0.1159)
    assert not hasattr(vae, "ignored")


def test_imagegen_scheduler_config_from_dict() -> None:
    from mminf.model.ming_omni_flash.config import ImageGenSchedulerConfig
    s = ImageGenSchedulerConfig.from_dict({
        "num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": False,
        "_class_name": "FlowMatchEulerDiscreteScheduler",
    })
    assert s.num_train_timesteps == 1000
    assert s.shift == pytest.approx(3.0)
    assert s.use_dynamic_shifting is False


def test_byt5_mapper_config_from_nested_json() -> None:
    from mminf.model.ming_omni_flash.config import ByT5MapperConfig
    b = ByT5MapperConfig.from_json({
        "byt5_mapper_type": "T5EncoderBlockByT5Mapper",
        "byt5_mapper_config": {"num_layers": 4, "sdxl_channels": 2560},
        "byt5_config": {"byt5_name": "google/byt5-small", "multilingual": True},
        "byt5_max_length": 256,
    })
    assert b.byt5_mapper_type == "T5EncoderBlockByT5Mapper"
    assert b.mapper_num_layers == 4
    assert b.sdxl_channels == 2560
    assert b.byt5_name == "google/byt5-small"
    assert b.byt5_max_length == 256
    assert b.multilingual is True


def test_imagegen_config_default_factories_yield_typed_subconfigs() -> None:
    from mminf.model.ming_omni_flash.config import (
        ByT5MapperConfig,
        ImageGenConfig,
        ImageGenSchedulerConfig,
        ImageVAEConfig,
        ZImageDiTConfig,
    )
    ig = ImageGenConfig()
    assert isinstance(ig.dit, ZImageDiTConfig)
    assert isinstance(ig.vae, ImageVAEConfig)
    assert isinstance(ig.scheduler, ImageGenSchedulerConfig)
    assert isinstance(ig.byt5, ByT5MapperConfig)
    assert ig.connector is None  # only populated by from_subdirs
    assert ig.use_identity_mlp is True
    assert ig.dit_type == "zimage"


def test_imagegen_from_subdirs_returns_none_without_transformer() -> None:
    """No transformer/ subdir → None (thinker-only / talker-only ckpt)."""
    from mminf.model.ming_omni_flash.config import ImageGenConfig
    with tempfile.TemporaryDirectory() as tmp:
        assert ImageGenConfig.from_subdirs(Path(tmp)) is None


def test_imagegen_from_subdirs_parses_synthetic_tree() -> None:
    """from_subdirs reads each subdir's config into the typed fields."""
    from mminf.model.ming_omni_flash.config import ImageGenConfig
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "transformer").mkdir()
        (root / "transformer" / "config.json").write_text(json.dumps({
            "_class_name": "ZImageTransformer2DModel",
            "dim": 1024, "n_layers": 4, "in_channels": 16,
            "axes_dims": [8, 12, 12], "axes_lens": [128, 64, 64],
        }))
        (root / "vae").mkdir()
        (root / "vae" / "config.json").write_text(json.dumps({
            "latent_channels": 16, "scaling_factor": 0.5, "shift_factor": 0.1,
        }))
        (root / "scheduler").mkdir()
        (root / "scheduler" / "scheduler_config.json").write_text(json.dumps({
            "num_train_timesteps": 1000, "shift": 2.5,
        }))
        (root / "byt5").mkdir()
        (root / "byt5" / "byt5.json").write_text(json.dumps({
            "byt5_mapper_config": {"num_layers": 2, "sdxl_channels": 1024},
            "byt5_config": {"byt5_name": "google/byt5-small"},
        }))
        (root / "connector").mkdir()
        (root / "connector" / "config.json").write_text(json.dumps({
            "model_type": "qwen2", "hidden_size": 1536,
        }))
        (root / "mlp").mkdir()
        (root / "mlp" / "config.json").write_text(json.dumps({
            "img_gen_scales": [16], "diffusion_c_input_dim": 2560,
            "use_identity_mlp": True, "dit_type": "zimage",
        }))

        ig = ImageGenConfig.from_subdirs(root)
        assert ig is not None
        assert ig.dit.dim == 1024
        assert ig.dit.axes_dims == (8, 12, 12)
        assert ig.vae.scaling_factor == pytest.approx(0.5)
        assert ig.scheduler.shift == pytest.approx(2.5)
        assert ig.byt5.mapper_num_layers == 2
        assert ig.connector["hidden_size"] == 1536
        assert ig.num_query_tokens == 256
