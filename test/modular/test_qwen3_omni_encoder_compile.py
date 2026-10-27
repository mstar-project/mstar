"""Qwen3-Omni's encoders run eager on AMD GPUs and compiled elsewhere. No
accelerator is needed: ``current_platform`` and ``torch.compile`` are stubbed."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar import platforms
from mstar.engine.engine import Engine
from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule, VisionEncoderSubmodule
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform


def _encoders():
    config = Qwen3OmniModelConfig()
    return {"audio_encoder": AudioEncoderSubmodule(nn.Identity(), config),
            "vision_encoder": VisionEncoderSubmodule(nn.Identity(), config)}


@pytest.mark.parametrize(
    ("platform", "eager"), [(CudaPlatform(), False), (RocmPlatform(), True), (CpuPlatform(), False)],
)
def test_encoders_run_eager_on_amd_gpus_only(monkeypatch, platform, eager):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)

    assert {name: sub.disable_torch_compile for name, sub in _encoders().items()} == {
        "audio_encoder": eager, "vision_encoder": eager,
    }


@pytest.mark.parametrize(("platform", "compiled"), [(CudaPlatform(), True), (RocmPlatform(), False)])
def test_the_engine_compiles_the_encoders_only_where_they_ask_for_it(monkeypatch, platform, compiled):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: "compiled")
    engine = Engine(graph_runtime=None, device_capability=SimpleNamespace(is_available=True))
    engine._item_encoders = set()
    engine._submodules = {
        name: SimpleNamespace(submodule=sub, forward="eager", forward_batched="eager")
        for name, sub in _encoders().items()
    }

    engine._compile_submodules()

    expected = "compiled" if compiled else "eager"
    assert {name: m.forward for name, m in engine._submodules.items()} == {
        "audio_encoder": expected, "vision_encoder": expected,
    }
