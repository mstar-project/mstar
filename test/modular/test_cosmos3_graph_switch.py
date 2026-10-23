"""Cosmos3 graph switches cover denoise, prefill and reasoner decode."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule, Cosmos3ReasonerSubmodule

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
GRAPH_CONFIGS = [
    path for path in sorted(CONFIGS.glob("cosmos3*.yaml"))
    if "accelerator_graph" in yaml.safe_load(path.read_text()).get("model_kwargs", {})
]


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(1, 1, bias=False)
        self.sp_group = SimpleNamespace(world_size=1)
        self.comm_group = SimpleNamespace(world_size=1)

    def text_forward(self, *args, **kwargs):
        raise AssertionError("declaring captures should not execute the model")


def _submodules(enabled):
    model = Cosmos3Model(
        model_path_hf="unused", skip_weight_loading=True,
        accelerator_graph=enabled, compile_denoise=False,
    )
    assert model.config.accelerator_graph is enabled
    transformer = _FakeTransformer()
    return (
        Cosmos3DiTSubmodule(transformer, model.config),
        Cosmos3ReasonerSubmodule(transformer, model.config),
    )


@pytest.mark.parametrize("path", GRAPH_CONFIGS, ids=lambda path: path.name)
@pytest.mark.parametrize("enabled", [False, True])
def test_shipped_configs_use_accelerator_graph_switch(path, enabled):
    deployment = yaml.safe_load(path.read_text())
    assert deployment["model_kwargs"]["accelerator_graph"] is True
    deployment["model_kwargs"]["accelerator_graph"] = enabled
    model = Cosmos3Model(
        model_path_hf="unused", skip_weight_loading=True, **deployment["model_kwargs"],
    )
    assert model.config.accelerator_graph is enabled


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("disable_env", [None, "", "1"])
def test_switch_gates_all_cosmos3_captures(monkeypatch, enabled, disable_env):
    if disable_env is None:
        monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    else:
        monkeypatch.setenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", disable_env)
    monkeypatch.delenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", raising=False)
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_BS", "1")
    monkeypatch.setenv("COSMOS3_GRAPH_MAX_LATENT_AREA", "2000")
    monkeypatch.setenv("COSMOS3_PREFILL_CAPTURE_BS", "1")
    monkeypatch.setenv("COSMOS3_REASONER_CAPTURE_BS", "1")
    dit, reasoner = _submodules(enabled)
    dit_configs = dit.get_accelerator_graph_configs(torch.device("cpu"))
    reasoner_configs = reasoner.get_accelerator_graph_configs(torch.device("cpu"))
    if enabled and not disable_env:
        assert [config.capture_graph_walk for config in dit_configs] == [
            C.IMAGE_GEN_WALK, C.VIDEO_GEN_WALK, C.PREFILL_WALK,
        ]
        assert [config.capture_graph_walk for config in reasoner_configs] == [
            C.REASONER_DECODE_WALK,
        ]
    else:
        assert dit_configs == reasoner_configs == []


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_disabled_switch_skips_gpu_templates(monkeypatch, device_type):
    monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    modules = _submodules(False)
    monkeypatch.setattr(
        torch, "zeros", lambda *args, **kwargs: pytest.fail("disabled capture allocated a template"),
    )
    for module in modules:
        assert module.get_accelerator_graph_configs(torch.device(device_type)) == []


def test_prefill_environment_switch_leaves_other_captures_enabled(monkeypatch):
    monkeypatch.delenv("COSMOS3_DISABLE_ACCELERATOR_GRAPH", raising=False)
    monkeypatch.setenv("COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH", "1")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_RES", "64x64")
    monkeypatch.setenv("COSMOS3_GEN_CAPTURE_VIDEO", "64x64x9")
    monkeypatch.setenv("COSMOS3_GRAPH_MAX_LATENT_AREA", "2000")
    dit, reasoner = _submodules(True)
    assert {config.capture_graph_walk for config in dit.get_accelerator_graph_configs("cpu")} == {
        C.IMAGE_GEN_WALK, C.VIDEO_GEN_WALK,
    }
    assert {config.capture_graph_walk for config in reasoner.get_accelerator_graph_configs("cpu")} == {
        C.REASONER_DECODE_WALK,
    }
