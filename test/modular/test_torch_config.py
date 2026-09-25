from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from mstar.engine import recompile_limit
from mstar.worker.worker import Worker


def test_executor_thread_gets_torch_config():
    worker = Worker.__new__(Worker)  # skips __init__, which opens sockets
    worker.device = torch.device("cpu")
    with ThreadPoolExecutor(1, initializer=worker._init_engine_thread) as ex:
        limit, unspec = ex.submit(lambda: (
            torch._dynamo.config.recompile_limit,
            torch._dynamo.config.allow_unspec_int_on_nn_module,
        )).result()
    assert (limit, unspec) == (recompile_limit(), True)


@pytest.mark.parametrize(
    "raw, expected", [(None, 84), ("", 84), ("123", 123), ("2", 8), ("5000", 256)]
)
def test_recompile_limit_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("MSTAR_RECOMPILE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("MSTAR_RECOMPILE_LIMIT", raw)
    assert recompile_limit() == expected
