import os
import subprocess
import sys
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
    "raw, expected",
    [(None, 84), ("", 84), ("123", 123), ("2", 8), ("5000", 256), ("lots", 84)],
)
def test_recompile_limit_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("MSTAR_RECOMPILE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("MSTAR_RECOMPILE_LIMIT", raw)
    assert recompile_limit() == expected


# the API server's process, and the module a spawned worker imports to find its target
@pytest.mark.parametrize("entry", ["mstar.api_server.entrypoint", "mstar.conductor.conductor"])
def test_compile_cache_defaults_under_the_user_cache(tmp_path, entry):
    """torch writes its /tmp default into the environment when ``torch._dynamo``
    is first imported, so mstar's default has to land before any import does."""
    env = {k: v for k, v in os.environ.items() if k not in ("TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME")}
    env["XDG_CACHE_HOME"] = str(tmp_path)
    code = f"import os, {entry}; print(os.environ['TORCHINDUCTOR_CACHE_DIR'])"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300, check=True,
    ).stdout.strip().splitlines()[-1]
    assert out == str(tmp_path / "mstar" / "torchinductor")
