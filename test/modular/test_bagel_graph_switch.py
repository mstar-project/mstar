
import json
from pathlib import Path
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


@pytest.mark.parametrize(
    ("name", "enabled"),
    [("bagel.yaml", True), ("bagel_cfg_parallel.yaml", True),
     ("bagel_xpu_tp2.yaml", False), ("bagel_xpu_cfg_tp2.yaml", False)],
)
def test_default_deployments_preserve_cuda_captures_and_disable_xpu(model_factory, name, enabled):
    path = Path(__file__).resolve().parents[2] / "configs" / name
    deployment = yaml.safe_load(path.read_text())
    assert model_factory(**deployment.get("model_kwargs", {})).config.accelerator_graph is enabled


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
    assert module.get_accelerator_graph_configs(torch.device(device_type)) == []


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_enabled_capture_keeps_existing_recipes(monkeypatch, device_type):
    module = LLMSubmodule.__new__(LLMSubmodule)
    nn.Module.__init__(module)
    module.config = load_bagel_config({"accelerator_graph": True})
    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros",
        lambda *args, **kwargs: zeros(*args, **{**kwargs, "device": "cpu"}),
    )
    configs = module.get_accelerator_graph_configs(torch.device(device_type))
    assert [config.capture_graph_walk for config in configs] == [
        "decode", "decode", "prefill_text",
    ]
    assert [config.additional_key_info for config in configs] == [False, True, False]
    assert all(config.compile for config in configs)
    assert [config.capture_batch_sizes for config in configs] == [
        None, None, [1, 2, 4, 8, 16, 32, 64],
    ]
    assert configs[2].capture_token_lengths == [128, 256, 512, 1024, 2048]


@pytest.mark.parametrize("graph_walk", ["image_gen", "image_gen_cfg"])
def test_image_generation_uses_the_single_request_forward(graph_walk):
    module = LLMSubmodule.__new__(LLMSubmodule)
    nn.Module.__init__(module)
    assert not module.can_batch(SimpleNamespace(graph_walk=graph_walk), [])
    assert module.max_batch_size(graph_walk) is None


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
