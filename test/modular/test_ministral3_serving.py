"""Registration, deployment and Python chat contracts; no model weights or GPU."""

import base64
from pathlib import Path

import pytest
import yaml
from test_ministral3_model import checkpoint as _checkpoint

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY, Ministral3Adapter
from mstar.api_server.openai.protocol import ChatCompletionRequest
from mstar.cli.main import DEFAULT_CONFIGS, _next_steps
from mstar.model.ministral3.ministral3_model import Ministral3Model
from mstar.model.registry import HF_MODELS, MODEL_REGISTRY
from mstar.validation import DeploymentError, validate_deployment

checkpoint = _checkpoint
ROOT = Path(__file__).resolve().parents[2]
PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360f8cfc0f01f0005000201e7a5b6"
    "2a0000000049454e44ae426082")).decode()


def test_registration_and_cli():
    assert MODEL_REGISTRY["ministral3_3b"] == ("mstar.model.ministral3.ministral3_model", "Ministral3Model")
    assert HF_MODELS["ministral3_3b"] == {"model_path_hf": "mistralai/Ministral-3-3B-Instruct-2512"}
    assert DEFAULT_CONFIGS["ministral3_3b"] == "ministral3_3b.yaml"
    assert isinstance(ADAPTER_REGISTRY["ministral3_3b"], Ministral3Adapter)
    example = _next_steps("ministral3_3b", "localhost", 8000)
    assert "client.chat" in example and "oai.chat.completions.create" in example


def test_default_deployment_and_independent_vision_placement(checkpoint, tmp_path):
    model = Ministral3Model(str(checkpoint))
    path = ROOT / "configs/ministral3_3b.yaml"
    config = yaml.safe_load(path.read_text())
    validate_deployment(model, config, str(path))
    assert config["resources"]["kv_cache"] == {"page_size": 128, "max_num_pages": 128}
    config["node_groups"] = [{"node_names": ["LLM"], "ranks": [0]},
                             {"node_names": ["vision_encoder"], "ranks": [1]}]
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(config))
    validate_deployment(model, config, str(path))
    config["node_groups"][0].update(ranks=[0, 2], tp_size=2)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DeploymentError, match="tp_enabled_nodes"):
        validate_deployment(model, config, str(path))


@pytest.mark.parametrize("limit", [1, 2])
def test_image_order_roles_sampling_and_token_limits(tmp_path, limit):
    request = ChatCompletionRequest(messages=[
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [
            {"type": "text", "text": "Before"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
            {"type": "text", "text": "After"},
        ]},
    ], temperature=0., max_tokens=100, max_completion_tokens=limit, stream=True, top_k=5)
    args = Ministral3Adapter().chat_to_request(request, tmp_path)
    assert args.input_modalities == ["text", "text", "image", "text"]
    assert args.output_modalities == ["text"]
    assert args.model_kwargs == {"temperature": 0., "max_output_tokens": limit, "top_k": 5}
    assert [p.role for p in args.prompt_parts] == ["system", "user", "user", "user"]
    assert Path(args.file_paths["image"][0]).read_bytes() == base64.b64decode(PNG)


@pytest.mark.parametrize("fields", [{"modalities": ["audio"]}, {"modalities": []}, {"audio": {}},
                                      {"tools": [{"type": "function"}]}, {"response_format": {"type": "json_object"}}])
def test_refuses_unserved_output(fields, tmp_path):
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}], **fields)
    with pytest.raises(ValueError):
        Ministral3Adapter().chat_to_request(request, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["audio_url", "video_url", "input_audio", "unknown"])
def test_unsupported_input_is_refused_before_saving_any_image(kind, tmp_path):
    request = ChatCompletionRequest(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        {"type": kind},
    ]}])
    with pytest.raises(ValueError, match="only text and image_url"):
        Ministral3Adapter().chat_to_request(request, tmp_path)
    assert not list(tmp_path.iterdir())


def test_image_part_cannot_smuggle_audio_mime(tmp_path):
    request = ChatCompletionRequest(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:audio/wav;base64,aGk="}},
    ]}])
    with pytest.raises(ValueError, match="text and image inputs only"):
        Ministral3Adapter().chat_to_request(request, tmp_path)
    assert not list(tmp_path.iterdir())
