"""load_weights_into reports checkpoint keys that matched no parameter."""

import logging

import pytest
import torch
from torch import nn

from mstar.model.loader.base import LLAMA_STACKED_PARAMS, load_weights_into


def _module():
    m = nn.Module()
    m.inner = nn.Linear(2, 2)
    return m


def _weights(prefix):
    return [(f"{prefix}.weight", torch.ones(2, 2)), (f"{prefix}.bias", torch.ones(2))]


def test_drift_with_unloaded_params_warns(caplog):
    with caplog.at_level(logging.INFO, logger="mstar.model.loader.base"):
        loaded = load_weights_into(_module(), _weights("outer.inner"))
    assert loaded == set()
    (rec,) = caplog.records
    assert rec.levelno == logging.WARNING
    assert "2 checkpoint keys matched no parameter" in rec.getMessage()
    assert "2 of 2 parameters got no weight" in rec.getMessage()


def test_extra_keys_with_everything_loaded_stay_quiet(caplog):
    weights = _weights("inner") + [("lm_head.weight", torch.ones(1))]
    with caplog.at_level(logging.INFO, logger="mstar.model.loader.base"):
        loaded = load_weights_into(_module(), weights)
    assert loaded == {"inner.weight", "inner.bias"}
    (rec,) = caplog.records
    assert rec.levelno == logging.INFO


def test_clean_load_logs_nothing(caplog):
    with caplog.at_level(logging.INFO, logger="mstar.model.loader.base"):
        load_weights_into(_module(), _weights("inner"))
    assert not caplog.records


def test_strict_refuses_a_stacked_parameter_missing_a_shard():
    """``up_proj`` alone fills ``gate_up_proj``'s name but leaves its gate
    rows at their initial values."""
    m = nn.Module()
    m.mlp = nn.Module()
    m.mlp.gate_up_proj = nn.Parameter(torch.zeros(4, 2))
    m.mlp.gate_up_proj.weight_loader = lambda p, w, shard: p.data[2 * shard:2 * shard + 2].copy_(w)
    up = [("mlp.up_proj", torch.ones(2, 2))]
    with pytest.raises(ValueError, match="mlp.gate_up_proj shard 0"):
        load_weights_into(m, up, stacked_params=LLAMA_STACKED_PARAMS, strict=True)
    load_weights_into(m, [("mlp.gate_proj", torch.ones(2, 2)), *up], stacked_params=LLAMA_STACKED_PARAMS, strict=True)
