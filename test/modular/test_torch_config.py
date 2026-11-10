import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from mstar.engine import _default_compile_caches, recompile_limit
from mstar.testing_utils.worker import make_worker


def test_executor_thread_gets_torch_config():
    worker = make_worker(device=torch.device("cpu"))
    with ThreadPoolExecutor(1, initializer=worker.executor._init_engine_thread) as ex:
        limit, unspec = ex.submit(lambda: (
            torch._dynamo.config.recompile_limit,
            torch._dynamo.config.allow_unspec_int_on_nn_module,
        )).result()
    assert (limit, unspec) == (recompile_limit(), True)


def test_the_engine_threads_start_before_a_compiled_call_can_raise_the_limit():
    """Before torch 2.12 dynamo's config is process-wide: a thread whose initializer ran only at
    its first job would read and write the limit beside another thread's ``with_recompile_limit``,
    so the pool's thread runs its initializer when the pool is built."""
    worker = make_worker(device=torch.device("cpu"))
    started = []
    init = worker.executor._init_engine_thread
    worker.executor._init_engine_thread = lambda: (started.append(1), init())
    pool = worker.executor._engine_thread_pool("mstar-plan")
    try:
        assert started == [1], "the engine thread's initializer waited for its first job"
    finally:
        pool.shutdown()


def test_a_raised_recompile_limit_holds_inside_its_calls_only():
    from mstar.engine import with_recompile_limit

    config = torch._dynamo.config
    seen = []
    before = (config.recompile_limit, config.accumulated_recompile_limit)
    wrapped = with_recompile_limit(
        lambda: seen.append((config.recompile_limit, config.accumulated_recompile_limit)), 500)
    wrapped()
    # past dynamo's per-code-object total too, or the entries over it would run uncompiled
    assert seen == [(500, 500)]
    assert (config.recompile_limit, config.accumulated_recompile_limit) == before

    def plain():
        return None

    assert with_recompile_limit(plain, None) is plain


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


# the API server's process, the module a spawned worker imports to find its target, and
# `python -m mstar.integrations.dynamo`
@pytest.mark.parametrize(
    "entry", ["mstar.api_server.entrypoint", "mstar.conductor.conductor", "mstar.integrations.dynamo.worker"],
)
def test_compile_cache_defaults_under_the_user_cache(tmp_path, entry):
    """torch writes its /tmp default into the environment when ``torch._dynamo``
    is first imported, so mstar's default has to land before any import does."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")}
    env["XDG_CACHE_HOME"] = str(tmp_path)
    code = f"import os, {entry}; print(os.environ['TORCHINDUCTOR_CACHE_DIR'], os.environ['TRITON_CACHE_DIR'])"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300, check=True,
    ).stdout.strip().splitlines()[-1]
    inductor = tmp_path / "mstar" / "torchinductor"
    assert out == f"{inductor} {inductor / 'triton'}"


def _no_home(cls):
    raise RuntimeError("Could not determine home directory.")


@pytest.mark.parametrize("inductor, xdg, has_home", [
    (None, "relative/cache", True),  # a relative XDG_CACHE_HOME is ignored, as the XDG spec says
    (None, None, False),             # no home keeps torch's and Triton's defaults
    ("exported", None, True),        # Triton's kernels follow an exported Inductor cache
])
def test_compile_cache_fallbacks(tmp_path, monkeypatch, inductor, xdg, has_home):
    # a copy, so what _default_compile_caches writes can't outlive the test
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    if inductor is None:
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / inductor))
    if xdg is None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    if has_home:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
    else:
        monkeypatch.setattr(Path, "home", classmethod(_no_home))
    _default_compile_caches()
    want = tmp_path / (inductor or "home/.cache/mstar/torchinductor") if has_home else None
    got = (os.environ.get("TORCHINDUCTOR_CACHE_DIR"), os.environ.get("TRITON_CACHE_DIR"))
    assert got == ((str(want), str(want / "triton")) if want else (None, None))
