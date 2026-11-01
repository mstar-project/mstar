"""A serving YAML's ``platform_overrides``: what each platform serves, and the
Qwen3-Omni and FLUX.2 klein configs that rely on them."""

from pathlib import Path

import pytest
import yaml

from mstar.platforms import CudaPlatform, RocmPlatform, serving_config
from mstar.platforms.serving_config import apply_platform_overrides, load_serving_config

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


@pytest.mark.parametrize("name, bf16", [
    ("flux2_klein.yaml", "flux2_klein_bf16.yaml"),
    ("flux2_klein_9b.yaml", "flux2_klein_9b_bf16.yaml"),
])
def test_the_klein_configs_serve_bf16_on_cuda_and_fp8_on_rocm(monkeypatch, name, bf16):
    """On AMD GPUs the override adds fp8 and the knob it needs to the file's
    model_kwargs and drops none of them. CUDA serves the bf16 file unchanged."""
    expected = _shipped(bf16)
    monkeypatch.setattr(serving_config, "_built_for", CudaPlatform)
    assert load_serving_config(str(CONFIGS / name)) == expected

    monkeypatch.setattr(serving_config, "_built_for", RocmPlatform)
    resolved = load_serving_config(str(CONFIGS / name))
    assert resolved["model_kwargs"] == {**expected["model_kwargs"], "linear_dtype": "fp8"}
    assert resolved["knobs"] == {"dit_exact_ops": True}
    assert {k: v for k, v in resolved.items() if k not in ("model_kwargs", "knobs")} == {
        k: v for k, v in expected.items() if k != "model_kwargs"
    }
