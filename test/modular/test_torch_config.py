import importlib
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import mstar.engine
from mstar.worker.worker import Worker


@pytest.fixture
def reload_engine(monkeypatch):
    yield
    monkeypatch.delenv("MSTAR_RECOMPILE_LIMIT", raising=False)
    importlib.reload(mstar.engine)


def test_executor_thread_gets_torch_config():
    worker = Worker.__new__(Worker)  # skips __init__, which opens sockets
    worker.device = torch.device("cpu")
    with ThreadPoolExecutor(1, initializer=worker._init_cuda_executor_thread) as ex:
        limit, unspec = ex.submit(lambda: (
            torch._dynamo.config.recompile_limit,
            torch._dynamo.config.allow_unspec_int_on_nn_module,
        )).result()
    assert (limit, unspec) == (mstar.engine.RECOMPILE_LIMIT, True)


@pytest.mark.parametrize(
    "raw, expected", [(None, 84), ("", 84), ("123", 123), ("2", 8), ("5000", 256)]
)
def test_recompile_limit_env(monkeypatch, reload_engine, raw, expected):
    if raw is None:
        monkeypatch.delenv("MSTAR_RECOMPILE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("MSTAR_RECOMPILE_LIMIT", raw)
    assert importlib.reload(mstar.engine).RECOMPILE_LIMIT == expected
