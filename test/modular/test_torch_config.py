"""Dynamo settings must reach every thread that compiles (#167).

Since torch 2.12, ``torch._dynamo.config`` keeps overrides in a ContextVar, so
a value set on the main thread is invisible to the worker's GPU executor
thread, which falls back to torch's defaults. Every test here reads the config
on a fresh thread, the way the executor sees it. That also keeps these tests
from leaking settings into the rest of the suite.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from mstar.engine.torch_config import (
    DEFAULT_RECOMPILE_LIMIT,
    RECOMPILE_LIMIT_ENV,
    RECOMPILE_LIMIT_MAX,
    RECOMPILE_LIMIT_MIN,
    apply_torch_config,
    recompile_limit,
)
from mstar.worker.worker import Worker

KEYS = (
    "recompile_limit",
    "accumulated_recompile_limit",
    "allow_unspec_int_on_nn_module",
    "specialize_int",
)


def _read_on_fresh_thread(initializer=None) -> dict:
    with ThreadPoolExecutor(max_workers=1, initializer=initializer) as ex:
        return ex.submit(
            lambda: {key: getattr(torch._dynamo.config, key) for key in KEYS}
        ).result()


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(RECOMPILE_LIMIT_ENV, raising=False)


def test_apply_torch_config_sets_the_calling_thread():
    flags = _read_on_fresh_thread(apply_torch_config)
    assert flags == {
        "recompile_limit": DEFAULT_RECOMPILE_LIMIT,
        "accumulated_recompile_limit": 256,
        "allow_unspec_int_on_nn_module": True,
        "specialize_int": False,
    }


def test_worker_executor_initializer_applies_the_settings(monkeypatch):
    """The bug itself: the GPU executor thread must see mstar's settings."""
    monkeypatch.setenv(RECOMPILE_LIMIT_ENV, "123")
    # __new__ skips Worker.__init__, which opens sockets and process groups
    worker = Worker.__new__(Worker)
    worker.device = torch.device("cpu")

    flags = _read_on_fresh_thread(worker._init_cuda_executor_thread)

    assert flags["recompile_limit"] == 123
    assert flags["allow_unspec_int_on_nn_module"] is True
    assert flags["specialize_int"] is False


def test_override_above_the_accumulated_cap_raises_it_too(monkeypatch):
    monkeypatch.setenv(RECOMPILE_LIMIT_ENV, "500")
    flags = _read_on_fresh_thread(apply_torch_config)
    assert flags["recompile_limit"] == 500
    assert flags["accumulated_recompile_limit"] == 500


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, DEFAULT_RECOMPILE_LIMIT),
        ("300", 300),
        ("2", RECOMPILE_LIMIT_MIN),
        ("99999", RECOMPILE_LIMIT_MAX),
        ("not-a-number", DEFAULT_RECOMPILE_LIMIT),
    ],
)
def test_env_override(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv(RECOMPILE_LIMIT_ENV, raw)
    assert recompile_limit() == expected
