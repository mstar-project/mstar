"""Dense Qwen3-8B is registered, takes the OpenAI chat route, and the CLI serves
it from a shipped config."""
from __future__ import annotations

from pathlib import Path

import pytest

# the model's resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")


def test_registry_cli_and_adapter_agree():
    from mstar.api_server.openai.adapters import TextChatAdapter, get_adapter
    from mstar.cli.main import DEFAULT_CONFIGS
    from mstar.model.qwen3.qwen3_model import Qwen3Model
    from mstar.model.registry import HF_MODELS, get_model_class

    assert get_model_class("qwen3_8b") is Qwen3Model
    assert HF_MODELS["qwen3_8b"]["model_path_hf"] == "Qwen/Qwen3-8B"
    assert isinstance(get_adapter("qwen3_8b"), TextChatAdapter)
    configs = Path(__file__).resolve().parents[4] / "configs"
    assert DEFAULT_CONFIGS["qwen3_8b"] == "qwen3_8b.yaml" and (configs / "qwen3_8b.yaml").is_file()
