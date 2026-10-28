import logging
import os

from mstar.utils.cache_home import user_cache_dir


def _default_compile_caches() -> None:
    """Keep Inductor's cache (compiled graphs, autotune picks) and Triton's
    (compiled kernels, Inductor's and others' alike) under ``~/.cache/mstar``
    rather than torch's ``/tmp/torchinductor_<user>`` and Triton's
    ``~/.triton/cache``, so a new container or a reboot that keeps ``~/.cache``
    starts warm. Importing this package creates the directories.

    Must run before anything imports ``torch._dynamo``, which writes torch's
    own default into the environment; ``Platform.process_env_defaults`` runs
    too late for that. An exported value wins, and without a writable home
    torch's and Triton's defaults stay.
    """
    for var, name in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("TRITON_CACHE_DIR", "triton")):
        if os.environ.get(var):
            continue
        try:
            path = user_cache_dir() / "mstar" / name
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError):
            return
        if os.access(path, os.W_OK):
            os.environ[var] = str(path)


_default_compile_caches()

import torch  # noqa: E402

logger = logging.getLogger(__name__)


def recompile_limit() -> int:
    raw = os.environ.get("MSTAR_RECOMPILE_LIMIT") or "84"
    try:
        requested = int(raw)
    except ValueError:
        logger.warning("MSTAR_RECOMPILE_LIMIT=%r is not an integer; using 84", raw)
        return 84
    limit = min(max(requested, 8), 256)
    if limit != requested:
        logger.warning("MSTAR_RECOMPILE_LIMIT=%d is outside [8, 256]; using %d", requested, limit)
    return limit


def apply_dynamo_config() -> None:
    # dynamo config is per-thread since torch 2.12, so every compiling thread calls this
    torch._dynamo.config.recompile_limit = recompile_limit()
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.specialize_int = False


def apply_torch_config() -> None:
    apply_dynamo_config()
    # Process-wide, so set once here and never from a thread initializer: a
    # thread starting mid-step would overwrite a node's fp32_precision_scope.
    torch.set_float32_matmul_precision('high')


apply_torch_config()
