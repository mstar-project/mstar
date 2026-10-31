import logging
import os

from mstar.utils.cache_home import user_cache_dir


def _default_compile_caches() -> None:
    """Keep Inductor's cache (compiled graphs, autotune picks) under
    ``~/.cache/mstar`` rather than torch's ``/tmp/torchinductor_<user>``, so a
    new container or a reboot that keeps ``~/.cache`` starts warm, and Triton's
    kernels (Inductor's and others' alike) in its ``triton`` subdirectory, so
    where they land doesn't depend on what compiled first and one directory
    holds them all. Importing this package creates the directory.

    Must run before anything imports ``torch._dynamo``, which writes torch's
    own default into the environment; ``Platform.process_env_defaults`` runs
    too late for that. Exported values win, and without a writable home
    torch's and Triton's defaults stay.
    """
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        try:
            path = user_cache_dir() / "mstar" / "torchinductor"
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError):
            return
        if not os.access(path, os.W_OK):
            return
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(path)
    if not os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton")


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


def with_recompile_limit(fn, submodule):
    """Wrap ``fn``, one of ``submodule``'s torch.compile'd callables, so dynamo compiles it under
    the submodule's ``dynamo_recompile_limit()`` where that is above this thread's limit. The
    limit is that submodule's alone: other nodes, and calls outside it, keep the engine's. A
    submodule that doesn't override the hook gets ``fn`` back as is."""
    from mstar.model.submodule_base import NodeSubmodule

    hook = getattr(type(submodule), "dynamo_recompile_limit", None)
    if hook is None or hook is NodeSubmodule.dynamo_recompile_limit:
        return fn

    def forward(*args, **kwargs):
        need = submodule.dynamo_recompile_limit()
        if need is None or need <= torch._dynamo.config.recompile_limit:
            return fn(*args, **kwargs)
        with torch._dynamo.config.patch(recompile_limit=need):
            return fn(*args, **kwargs)

    return forward


def apply_torch_config() -> None:
    apply_dynamo_config()
    # Process-wide, so set once here and never from a thread initializer: a
    # thread starting mid-step would overwrite a node's fp32_precision_scope.
    torch.set_float32_matmul_precision('high')


apply_torch_config()
