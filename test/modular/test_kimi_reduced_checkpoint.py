"""The reduced random-weight checkpoint writer and loader, on CPU only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))

import torch
from kimi_reference import write_checkpoint
from safetensors import safe_open

from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model


def test_write_checkpoint_on_cpu(tmp_path):
    write_checkpoint(tmp_path, KimiK2Config.reduced(), device=torch.device("cpu"))

    checkpoint = tmp_path / "model.safetensors"
    assert checkpoint.exists()
    with safe_open(str(checkpoint), framework="pt") as f:
        keys = f.keys()
        assert "model.embed_tokens.weight" in keys
        assert f.get_slice("model.embed_tokens.weight").get_shape() == [256, 128]
        assert "lm_head.weight" in keys


def test_reduced_model_loads_written_checkpoint_on_cpu(tmp_path):
    write_checkpoint(tmp_path, KimiK2Config.reduced(), device=torch.device("cpu"))

    model = KimiK2Model(
        model_path_hf=str(tmp_path), config_variant="reduced", tokenizer_mode="byte"
    )
    sub = model.get_submodule("LLM", device="cpu")
    assert sub is not None

    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as f:
        expected = f.get_tensor("model.embed_tokens.weight")
    actual = sub.language_model.model.embed_tokens.weight
    assert torch.equal(actual, expected.to(actual.dtype))
