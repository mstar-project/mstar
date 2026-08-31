"""Platform use case: serve an image-plus-text chat completion request."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.graph.base import Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.qwenvl.qwenvl_model import QwenVLModel
from mstar.model.qwenvl.submodules import QwenVLLLMSubmodule, QwenVLVisionSubmodule

from ._helpers import FakeProcessor, FixedLanguageModel, install_fake_transformers, tiny_config


def test_customer_can_submit_image_plus_text_chat_request():
    config = tiny_config()
    ids = torch.tensor(
        [1, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 2]
    )
    processor = FakeProcessor(ids, grid=torch.tensor([[1, 4, 4]]))
    model = object.__new__(QwenVLModel)
    model.config, model.processor, model._submodule_cache = config, processor, {}
    result = model.process_prompt("describe", ["image", "text"], ["text"], {"image_inputs": [torch.ones(3, 4, 4)]})
    assert processor.call_kwargs["images"][0].dtype == "uint8"
    assert result["position_ids"][0][:, -1].tolist() == [3, 3, 3]
    assert model.postprocess(torch.tensor([4, 5]), "text") == b"decoded:[4, 5]"
    with pytest.raises(ValueError, match="Unsupported"):
        model.postprocess(torch.tensor([4]), "image")
    with pytest.raises(NotImplementedError, match=r"image\+text"):
        model.process_prompt("video", ["video"], ["text"], {"video_inputs": [torch.ones(1)]})


def test_cli_exposes_qwen3_vl_single_gpu_correctness_baseline():
    import yaml

    from mstar.cli.main import DEFAULT_CONFIGS, _next_steps, _resolve_config

    assert DEFAULT_CONFIGS["qwenvl"] == "qwenvl.yaml"
    config_path = _resolve_config("qwenvl", None)
    assert config_path.endswith("configs/qwenvl.yaml")
    with open(config_path) as stream:
        deployment = yaml.safe_load(stream)
    assert deployment["max_seq_len"] == 16_384
    assert deployment["kv_cache"]["max_num_pages"] == 128
    assert deployment["node_groups"] == [{"node_names": ["vision_encoder", "LLM"], "ranks": [0]}]
    next_steps = _next_steps("qwenvl", "0.0.0.0", 8000)
    assert "client.chat" in next_steps
    assert "OpenAI-compatible" not in next_steps


def test_platform_can_resolve_local_or_hub_model_snapshot(tmp_path, monkeypatch):
    model = object.__new__(QwenVLModel)
    model.config, model.processor = tiny_config(), FakeProcessor(torch.tensor([1, 2]))
    assert set(model.process_prompt("text", ["text"], ["text"])) == {"text_inputs", "position_ids"}
    model.model_path_hf = str(tmp_path)
    assert model._resolve_snapshot() == str(tmp_path)
    captured = {}
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: captured.update(kwargs) or "/resolved"),
    )
    model.model_path_hf, model.cache_dir = "Qwen/Qwen3-VL-30B-A3B-Instruct", "/cache"
    assert model._resolve_snapshot() == "/resolved"
    assert captured == {"repo_id": model.model_path_hf, "cache_dir": "/cache"}
    assert model.get_node_engine_types() == {"vision_encoder": EngineType.STATELESS, "LLM": EngineType.KV_CACHE}
    assert model.get_kv_cache_config()[0].num_layers == 3
    assert model.get_sampling_config("LLM", {"temperature": 0.5}).temperature == 0.5


def test_chat_request_runs_prefill_then_bounded_decode():
    model = object.__new__(QwenVLModel)
    model.config = tiny_config()
    initial = model.get_initial_forward_pass_args(
        "default",
        ["image"],
        ["text"],
        {"pixel_values": [], "image_grid_thw": [], "text_inputs": [], "position_ids": []},
    )
    assert initial.full_metadata.graph_walk == "prefill_vision"
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["text"], graph_walk="prefill", is_prefill=True
    )
    decode = model.get_partition_forward_pass_args("default", metadata, {"new_token": []})
    assert decode.full_metadata.graph_walk == "decode"
    assert model.get_partition_forward_pass_args("default", decode.full_metadata, {}).request_done
    graph = model.get_graph_walk_graphs()["decode"]
    assert isinstance(graph, Loop)
    assert {edge.next_node for edge in graph.section.outputs} == {"LLM", EMIT_TO_CLIENT}


def test_tp2_topology_remains_an_explicit_later_slice():
    model = object.__new__(QwenVLModel)
    model.config = tiny_config()
    assert model.get_default_sharding_config().tp_enabled_nodes == {"LLM"}
    assert model.get_sharding_config("configs/qwenvl.yaml").groups == []
    assert any(group.tp_size == 2 for group in model.get_sharding_config("configs/qwenvl_tp2.yaml").groups)


def test_worker_can_materialize_serving_nodes_from_onboarded_snapshot(tmp_path, monkeypatch):
    config = tiny_config()
    config.save_pretrained(tmp_path)

    class Vision(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, pixels, grid_thw):
            return pixels

    install_fake_transformers(monkeypatch, vision_class=Vision, config=config)
    import mstar.model.qwenvl.weight_loader as loader
    from mstar.model.qwenvl import components

    class Language(FixedLanguageModel):
        def __init__(self, config, comm_group=None):
            super().__init__(config)

    monkeypatch.setattr(components, "QwenVLForCausalLM", Language)
    monkeypatch.setattr(loader, "iter_qwen_vl_text_weights", lambda *args: iter(()))
    monkeypatch.setattr(loader, "iter_qwen_vl_vision_weights", lambda *args: iter(()))
    monkeypatch.setattr(
        loader, "load_qwen_vl_text_weights", lambda module, weights: set(dict(module.named_parameters()))
    )
    monkeypatch.setattr(
        loader, "load_qwen_vl_vision_weights", lambda module, weights: set(dict(module.named_parameters()))
    )
    model = QwenVLModel(str(tmp_path))
    llm = model.get_submodule("LLM", device="cpu", autocast_dtype=torch.float32)
    assert isinstance(llm, QwenVLLLMSubmodule)
    assert model.get_submodule("LLM", device="cpu") is llm
    assert isinstance(
        model.get_submodule("vision_encoder", device="cpu", autocast_dtype=torch.float32),
        QwenVLVisionSubmodule,
    )
    assert model.get_submodule("not-a-node", device="cpu") is None
