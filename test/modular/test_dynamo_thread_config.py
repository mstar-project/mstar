"""Regression: dynamo config must apply on worker threads (#167).

torch._dynamo ConfigModule stores user overrides in a ContextVar (thread-local
as of torch 2.13+). Import-time assignments on the main thread do not propagate
to the dedicated GPU executor thread. apply_dynamo_config() must be called on
any thread that may trigger dynamo tracing.
"""

from concurrent.futures import ThreadPoolExecutor

import torch

from mstar.engine import RECOMPILE_LIMIT, apply_dynamo_config

_KEYS = ("recompile_limit", "allow_unspec_int_on_nn_module", "specialize_int")


def _read_dynamo_flags() -> dict:
    return {k: getattr(torch._dynamo.config, k) for k in _KEYS}


def test_apply_dynamo_config_on_worker_thread():
    """GPU-thread initializer path: flags match RECOMPILE_LIMIT / expected defaults."""
    # Main thread already applied at import; re-apply for a clean baseline.
    apply_dynamo_config()
    main = _read_dynamo_flags()
    assert main["recompile_limit"] == RECOMPILE_LIMIT
    assert main["allow_unspec_int_on_nn_module"] is True
    assert main["specialize_int"] is False

    with ThreadPoolExecutor(max_workers=1, initializer=apply_dynamo_config) as ex:
        worker = ex.submit(_read_dynamo_flags).result()

    assert worker["recompile_limit"] == RECOMPILE_LIMIT
    assert worker["allow_unspec_int_on_nn_module"] is True
    assert worker["specialize_int"] is False


def test_import_time_flags_do_not_leak_to_fresh_thread_without_initializer():
    """Document the torch 2.13+ ContextVar pitfall the fix addresses.

    On builds where ConfigModule is thread-local, a fresh thread without
    apply_dynamo_config() does not see main-thread overrides. Skip when the
    installed torch still shares config across threads (pre-2.13 behavior).
    """
    apply_dynamo_config()
    main = _read_dynamo_flags()

    with ThreadPoolExecutor(max_workers=1) as ex:
        worker = ex.submit(_read_dynamo_flags).result()

    if worker == main:
        # Older torch: config is process-global — nothing to regress against.
        return

    # Thread-local: worker should have fallen back away from our overrides.
    assert worker["recompile_limit"] != RECOMPILE_LIMIT or worker[
        "allow_unspec_int_on_nn_module"
    ] is not True
