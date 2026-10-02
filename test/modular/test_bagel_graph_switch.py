"""BAGEL deployment graph switches without checkpoints or GPU allocation."""

import json
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from mstar.model.bagel import bagel_model
from mstar.model.bagel.config import load_bagel_config
from mstar.model.bagel.submodules import LLMSubmodule


@pytest.fixture
def model_factory(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    monkeypatch.setattr(bagel_model, "hf_hub_download", lambda **kwargs: config_path)
    monkeypatch.setattr(
        bagel_model.BagelTokenizer, "from_pretrained", lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        bagel_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}, None),
    )
    return lambda **kwargs: bagel_model.BagelModel("test/bagel", **kwargs)


@pytest.mark.parametrize(
    ("options", "enabled"),
    [({}, True), ({"accelerator_graph": False}, False),
     ({"accelerator_graph": True}, True)],
)
def test_model_resolves_graph_switch(model_factory, options, enabled):
    assert model_factory(**options).config.accelerator_graph is enabled


def test_yaml_graph_switch_reaches_model(model_factory):
    deployment = yaml.safe_load("model: bagel\nmodel_kwargs:\n  accelerator_graph: false\n")
    assert model_factory(**deployment["model_kwargs"]).config.accelerator_graph is False


def test_checkpoint_graph_setting_is_overridden_by_deployment(model_factory, monkeypatch):
    config = load_bagel_config({"accelerator_graph": False})
    monkeypatch.setattr(bagel_model, "load_bagel_config", lambda value: config)
    assert model_factory(accelerator_graph=True).config.accelerator_graph is True


@pytest.mark.parametrize(
    "value",
    ["false", 0],
)
def test_invalid_graph_options_fail_before_downloading(monkeypatch, value):
    monkeypatch.setattr(
        bagel_model, "hf_hub_download",
        lambda **kwargs: pytest.fail("invalid switches should fail before downloading"),
    )
    with pytest.raises(ValueError, match="boolean"):
        bagel_model.BagelModel("test/bagel", accelerator_graph=value)


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
@pytest.mark.parametrize("node_name", ["LLM", "LLM_cfg_text", "LLM_cfg_img"])
def test_disabled_llm_capture_skips_template_allocation(device_type, node_name):
    module = LLMSubmodule.__new__(LLMSubmodule)
    nn.Module.__init__(module)
    module.config = load_bagel_config({"accelerator_graph": False})
    module.node_name = node_name
    # No weights or other attributes: a disabled getter must return before
    # reading them or allocating any capture templates on the requested GPU.
    assert module.get_accelerator_graph_configs(torch.device(device_type)) == []


def test_enabled_cuda_capture_keeps_existing_recipes(monkeypatch):
    module = LLMSubmodule.__new__(LLMSubmodule)
    nn.Module.__init__(module)
    module.config = load_bagel_config({})
    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros",
        lambda *args, **kwargs: zeros(*args, **{**kwargs, "device": "cpu"}),
    )
    configs = module.get_accelerator_graph_configs(torch.device("cuda"))
    assert [config.capture_graph_walk for config in configs] == [
        "decode", "decode", "prefill_text",
    ]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("vit_enabled", [False, True])
@pytest.mark.parametrize("batching", [False, True])
def test_model_graph_switch_gates_optional_vit_capture(
    model_factory, monkeypatch, enabled, vit_enabled, batching,
):
    monkeypatch.setenv("MSTAR_VIT_ACCELERATOR_GRAPH", str(int(vit_enabled)))
    monkeypatch.setenv("MSTAR_VIT_BATCHING", str(int(batching)))
    monkeypatch.setenv("MSTAR_VIT_ACCELERATOR_GRAPH_TOKEN_BUCKETS", "1024,512")
    monkeypatch.setenv("MSTAR_VIT_ACCELERATOR_GRAPH_BATCH_SIZES", "4,1,2")
    model = model_factory(accelerator_graph=enabled)
    model.vit_model = nn.Identity()
    model.vit_model.vision_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8, num_attention_heads=2, rope=False),
    )
    model.connector = nn.Identity()
    model.vit_pos_embed = nn.Identity()
    monkeypatch.setattr(model, "_init_vit_components", lambda *args, **kwargs: None)
    vit = model._create_submodule("vit_encoder", "cpu")
    configs = vit.get_piecewise_accelerator_graph_configs(
        torch.device("cpu"), torch.float32,
    )
    assert bool(configs) is (enabled and vit_enabled)
    if configs:
        config = next(iter(configs.values()))
        assert config.total_tokens == [512, 1024]
        assert config.capture_batch_sizes == ([1, 2, 4] if batching else [1])
