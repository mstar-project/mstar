"""A serving YAML's ``platform_overrides``: what each platform serves, and the
Qwen3-Omni, FLUX.2 klein, Z-Image and Qwen3.5 configs that rely on them."""

from pathlib import Path

import pytest
import yaml

from mstar.platforms import CudaPlatform, RocmPlatform, serving_config
from mstar.platforms.serving_config import OVERRIDES_KEY, apply_platform_overrides, load_serving_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _shipped(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def test_an_entry_replaces_top_level_keys_on_its_platform_only():
    config = {
        "max_seq_len": 8,
        "node_groups": ["base"],
        "platform_overrides": {"rocm": {"node_groups": ["amd"], "rank_devices": {2: 0}}},
    }

    assert apply_platform_overrides(config, "rocm") == {
        "max_seq_len": 8, "node_groups": ["amd"], "rank_devices": {2: 0},
    }
    assert apply_platform_overrides(config, "cuda") == {"max_seq_len": 8, "node_groups": ["base"]}


def test_a_config_without_the_block_is_untouched():
    config = {"node_groups": ["base"]}
    assert apply_platform_overrides(config, "rocm") is config
    assert apply_platform_overrides(None, "rocm") is None


@pytest.mark.parametrize("overrides, match", [
    ({"amd": {"mps": True}}, "names"),
    ({"rocm": ["mps"]}, "must be a mapping"),
    (["rocm"], "must be a mapping"),
])
def test_a_malformed_block_fails_on_every_platform(overrides, match):
    config = {"platform_overrides": overrides}
    for platform in ("rocm", "cuda"):
        with pytest.raises(ValueError, match=match):
            apply_platform_overrides(config, platform)


@pytest.mark.parametrize("platform, expected", [
    (RocmPlatform, "qwen3omni_2gpu_ramp4g.yaml"),
    (CudaPlatform, "qwen3omni_2gpu_stock.yaml"),
])
def test_the_2gpu_config_serves_its_platforms_layout(monkeypatch, platform, expected):
    """On AMD GPUs Code2Wav has its own worker on the Talker's GPU and audio
    starts on the gentle ramp; CUDA keeps one worker and the default chunks. The
    opt-in configs stay equal to what the default resolves to."""
    monkeypatch.setattr(serving_config, "_built_for", platform)
    assert load_serving_config(str(CONFIGS / "qwen3omni_2gpu.yaml")) == _shipped(expected)


@pytest.mark.parametrize("platform, expected", [
    (RocmPlatform, "qwen3omni_colocated_split3.yaml"),
    (CudaPlatform, "qwen3omni_colocated_stock.yaml"),
])
def test_the_colocated_config_serves_its_platforms_layout(monkeypatch, platform, expected):
    """On AMD GPUs three workers share the one GPU; CUDA keeps one worker."""
    monkeypatch.setattr(serving_config, "_built_for", platform)
    assert load_serving_config(str(CONFIGS / "qwen3omni_colocated.yaml")) == _shipped(expected)


@pytest.mark.parametrize("ramp, base", [
    ("qwen3omni_2gpu_ramp4g.yaml", "qwen3omni_2gpu_split.yaml"),
    ("qwen3omni_colocated_split3_ramp4g.yaml", "qwen3omni_colocated_split3.yaml"),
])
def test_the_gentle_ramp_configs_are_their_base_with_a_chunk_schedule(ramp, base):
    config = _shipped(ramp)
    assert config.pop("model_kwargs") == {"codec_chunk_schedule": [4, 5, 6, 8, 10, 13, 16, 20]}
    assert config == _shipped(base)


@pytest.mark.parametrize("name, bf16, schedule", [
    ("flux2_klein.yaml", "flux2_klein_bf16.yaml", "flux2_klein_fp8_bpff.yaml"),
    ("flux2_klein_9b.yaml", "flux2_klein_9b_bf16.yaml", "flux2_klein_9b_fp8_bbpf.yaml"),
])
def test_the_klein_configs_serve_bf16_on_cuda_and_a_step_schedule_on_rocm(monkeypatch, name, bf16, schedule):
    """On AMD GPUs the override serves the named schedule file with the solo text encoder and
    the knob fp8 needs, and drops none of the file's settings. CUDA serves the bf16 file
    without its override, so its encoder stays batched."""
    monkeypatch.setattr(serving_config, "_built_for", CudaPlatform)
    expected = load_serving_config(str(CONFIGS / bf16))
    assert OVERRIDES_KEY not in expected
    assert "text_encoder_max_batch_size" not in expected["model_kwargs"]
    assert load_serving_config(str(CONFIGS / name)) == expected

    monkeypatch.setattr(serving_config, "_built_for", RocmPlatform)
    resolved = load_serving_config(str(CONFIGS / name))
    # the schedule file resolved on AMD GPUs, plus the solo text encoder
    assert resolved["model_kwargs"] == {
        **load_serving_config(str(CONFIGS / schedule))["model_kwargs"], "text_encoder_max_batch_size": 1,
    }
    assert resolved["knobs"] == {"dit_exact_ops": True}
    assert {k: v for k, v in resolved.items() if k not in ("model_kwargs", "knobs")} == {
        k: v for k, v in expected.items() if k != "model_kwargs"
    }
    # a bf16 deployment on AMD GPUs differs from the CUDA one by the attention kernel and the solo text encoder alone
    assert load_serving_config(str(CONFIGS / bf16)) == {
        **expected,
        "model_kwargs": {**expected["model_kwargs"], "attention_backend": "auto", "text_encoder_max_batch_size": 1},
    }


@pytest.mark.parametrize("platform", [CudaPlatform, RocmPlatform])
@pytest.mark.parametrize("fp8, bf16", [
    ("flux2_klein_fp8.yaml", "flux2_klein_bf16.yaml"),
    ("flux2_klein_9b_fp8_keep.yaml", "flux2_klein_9b_bf16.yaml"),
])
def test_the_named_fp8_klein_configs_are_their_bf16_file_with_fp8_and_the_knob(monkeypatch, platform, fp8, bf16):
    monkeypatch.setattr(serving_config, "_built_for", platform)
    expected = load_serving_config(str(CONFIGS / bf16))
    resolved = load_serving_config(str(CONFIGS / fp8))
    # the keep list is the fp8 file's business: at least the text stream, no faster in fp8
    keep = resolved["model_kwargs"].pop("linear_keep_bf16")
    assert any(pattern.startswith("double_blocks.") for pattern in keep)
    bf16_kwargs = {k: v for k, v in expected["model_kwargs"].items() if k != "linear_keep_bf16"}
    assert resolved == {
        **expected,
        "model_kwargs": {**bf16_kwargs, "linear_dtype": "fp8"},
        "knobs": {"dit_exact_ops": True},
    }


def test_z_image_serves_sdpa_on_cuda_and_the_ragged_resource_on_rocm(monkeypatch):
    """On AMD GPUs the override moves Z-Image's DiT attention to ``auto`` and
    changes nothing else; CUDA serves the file as written."""
    shipped = _shipped("z_image_turbo.yaml")
    expected = {k: v for k, v in shipped.items() if k != OVERRIDES_KEY}
    assert expected["model_kwargs"]["attention_backend"] == "sdpa"
    monkeypatch.setattr(serving_config, "_built_for", CudaPlatform)
    assert load_serving_config(str(CONFIGS / "z_image_turbo.yaml")) == expected

    monkeypatch.setattr(serving_config, "_built_for", RocmPlatform)
    resolved = load_serving_config(str(CONFIGS / "z_image_turbo.yaml"))
    assert resolved["model_kwargs"] == {**expected["model_kwargs"], "attention_backend": "auto"}
    assert {k: v for k, v in resolved.items() if k != "model_kwargs"} == {
        k: v for k, v in expected.items() if k != "model_kwargs"
    }


QWEN3_5_NO_STEP_BUDGET_ON_ROCM = [
    "qwen3_5_0.8b.yaml",
    "qwen3_5_0.8b_bf16state.yaml",
    "qwen3_5_27b_tp4.yaml",
    "qwen3_5_2b.yaml",
    "qwen3_5_2b_bf16state.yaml",
    "qwen3_5_4b.yaml",
    "qwen3_5_9b.yaml",
    "qwen3_5_9b_bf16state.yaml",
    "qwen3_5_9b_dp2.yaml",
    "qwen3_5_9b_tp2.yaml",
]


@pytest.mark.parametrize("name", QWEN3_5_NO_STEP_BUDGET_ON_ROCM)
def test_qwen3_5_keeps_the_step_budget_on_cuda_and_drops_it_on_rocm(monkeypatch, name):
    """On AMD GPUs the override turns the LLM's chunked prefill budget off, so prompts run
    whole in pure prefill steps; CUDA serves the file as written, with the model's default."""
    shipped = _shipped(name)
    expected = {k: v for k, v in shipped.items() if k != OVERRIDES_KEY}
    assert "max_batch_tokens" not in expected
    monkeypatch.setattr(serving_config, "_built_for", CudaPlatform)
    assert load_serving_config(str(CONFIGS / name)) == expected

    monkeypatch.setattr(serving_config, "_built_for", RocmPlatform)
    resolved = load_serving_config(str(CONFIGS / name))
    assert resolved["max_batch_tokens"] == {"LLM": None}
    # the rest of the file as written, plus the rest of the ROCm entry (eager yield where the config has it)
    assert resolved == {**expected, **shipped[OVERRIDES_KEY]["rocm"]}
