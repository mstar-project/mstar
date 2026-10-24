import logging
import os
from pathlib import Path


def _default_compile_cache() -> None:
    """Keep Inductor's cache (compiled graphs, autotune picks, Triton kernels)
    under ``~/.cache`` beside the weights and FlashInfer's JIT cache, rather
    than torch's ``/tmp/torchinductor_<user>``, so a restart that keeps
    ``~/.cache`` starts warm.

    Must run before anything imports ``torch._dynamo``, which writes torch's
    own default into the environment. An exported value wins, and a home that
    can't be written keeps torch's default.
    """
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return
    path = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mstar" / "torchinductor"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
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


def apply_torch_config() -> None:
    # dynamo config is per-thread since torch 2.12, so every compiling thread calls this
    torch._dynamo.config.recompile_limit = recompile_limit()
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.specialize_int = False
    torch.set_float32_matmul_precision('high')


apply_torch_config()
