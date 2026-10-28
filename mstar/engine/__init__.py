import logging
import os

from mstar.utils.cache_home import user_cache_dir


def _default_compile_cache() -> None:
    """Keep Inductor's cache (compiled graphs, autotune picks, Triton kernels)
    under ``~/.cache`` beside the weights and FlashInfer's JIT cache, rather
    than torch's ``/tmp/torchinductor_<user>``, so a restart that keeps
    ``~/.cache`` starts warm. Importing this package creates the directory.

    Must run before anything imports ``torch._dynamo``, which writes torch's
    own default into the environment. An exported value wins, and without a
    writable home torch's default stays.
    """
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return
    try:
        path = user_cache_dir() / "mstar" / "torchinductor"
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError):
        return
    if os.access(path, os.W_OK):
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(path)


_default_compile_cache()

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
