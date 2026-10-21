"""``test/conftest.py`` gives GPU tests on ROCm the env serving runs with."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.platforms import RocmPlatform

_spec = importlib.util.spec_from_file_location("root_conftest", Path(__file__).parents[1] / "conftest.py")
root_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(root_conftest)

MIOPEN_SPLIT_K = (
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC",
)
ROCM_DEFAULTS = tuple(RocmPlatform().process_env_defaults())


@pytest.fixture
def env(monkeypatch):
    """Every ROCm default unset. Teardown restores the real values."""
    for key in ROCM_DEFAULTS:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    return monkeypatch


@pytest.fixture
def build(monkeypatch):
    """Pretend torch was built for ``hip`` or ``cuda``. Counting devices
    returns ``gpus`` and records the MIOpen variables at that moment; with
    ``gpus=None`` it fails the test."""
    seen = []

    def run(*, hip=None, cuda=None, gpus=None):
        def is_available():
            assert gpus is not None, "counted devices"
            seen.append({key: os.environ.get(key) for key in MIOPEN_SPLIT_K})
            return gpus

        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch.version, "cuda", cuda)
        monkeypatch.setattr(torch.version, "xpu", None, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", is_available)
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: False))
        platforms.resolve_platform.cache_clear()
        return seen

    yield run
    platforms.resolve_platform.cache_clear()


@pytest.mark.parametrize("cuda", ["12.8", None], ids=["cuda", "cpu"])
def test_other_builds_are_left_alone(env, build, cuda):
    build(cuda=cuda)
    root_conftest.apply_rocm_env_defaults()
    assert not set(ROCM_DEFAULTS) & set(os.environ)


def test_a_rocm_build_with_its_gpus_hidden_is_left_alone(env, build):
    build(hip="7.0", gpus=False)
    root_conftest.apply_rocm_env_defaults()
    assert not set(ROCM_DEFAULTS) & set(os.environ)


def test_a_rocm_gpu_gets_the_miopen_defaults_before_devices_are_counted(env, build):
    seen = build(hip="7.0", gpus=True)
    root_conftest.apply_rocm_env_defaults()
    assert seen == [dict.fromkeys(MIOPEN_SPLIT_K, "0")]
    assert all(os.environ[key] == "0" for key in MIOPEN_SPLIT_K)


@pytest.mark.parametrize("gpus", [True, False], ids=["gpu", "gpus_hidden"])
def test_an_exported_value_is_kept(env, build, gpus):
    env.setenv(MIOPEN_SPLIT_K[0], "1")
    build(hip="7.0", gpus=gpus)
    root_conftest.apply_rocm_env_defaults()
    assert os.environ[MIOPEN_SPLIT_K[0]] == "1"
    assert os.environ.get(MIOPEN_SPLIT_K[1]) == ("0" if gpus else None)
