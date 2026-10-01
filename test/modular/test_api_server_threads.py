"""The API server sizes torch's intra-op pool from the model / config.yaml."""
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server import entrypoint
from mstar.model.base import Model
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel


@pytest.fixture
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(16)
    yield
    torch.set_num_threads(before)


def test_model_default_applies(threads):
    entrypoint._set_preprocess_threads(SimpleNamespace(PREPROCESS_TORCH_THREADS=4), {})
    assert torch.get_num_threads() == 4


def test_config_overrides_the_model(threads):
    model = SimpleNamespace(PREPROCESS_TORCH_THREADS=4)
    entrypoint._set_preprocess_threads(model, {"preprocess_torch_threads": 2})
    assert torch.get_num_threads() == 2


def test_none_leaves_torch_alone(threads):
    entrypoint._set_preprocess_threads(SimpleNamespace(PREPROCESS_TORCH_THREADS=None), {})
    assert torch.get_num_threads() == 16
    # an explicit null in the yaml turns a model default off
    entrypoint._set_preprocess_threads(
        SimpleNamespace(PREPROCESS_TORCH_THREADS=4), {"preprocess_torch_threads": None},
    )
    assert torch.get_num_threads() == 16


def test_only_qwen3_5_opts_in():
    assert Model.PREPROCESS_TORCH_THREADS is None
    assert Qwen3_5DenseModel.PREPROCESS_TORCH_THREADS == 4
