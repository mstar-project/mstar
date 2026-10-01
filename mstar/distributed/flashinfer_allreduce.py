"""FlashInfer-backed all-reduce for ``CommGroup.all_reduce`` (opt-in via
``MSTAR_ALLREDUCE_BACKEND=flashinfer``).

NCCL's ring all-reduce costs ~13 us per call at a (1, 7168) bf16 decode
shape; FlashInfer's ``allreduce_fusion`` (pattern ``kAllReduce``, "trtllm"
backend) costs ~4 us at the same shape (measured in
``perf/allreduce_alternatives.py``). A decode step issues on the order of a
hundred all-reduces, so the saving adds up to milliseconds per step at
TP4/TP8.

FlashInfer needs a workspace per comm group: a collective call that exchanges
IPC handles across ranks and allocates buffers sized for
``MSTAR_FLASHINFER_ALLREDUCE_MAX_TOKENS`` tokens (default 1024). That
exchange allocates memory and must run on every rank together, so it is only
attempted outside CUDA-graph capture -- in practice the first eager forward
before capture. A call before the workspace exists, or one whose shape the
workspace can't hold, falls back to NCCL.

Only the shape- and config-level decisions (backend flag, dtype, token cap)
are made in Python. Everything that depends on process state -- does the
workspace exist, are we capturing, is the buffer sufficient -- lives inside
the custom op ``mstar::flashinfer_all_reduce``. The model forward is traced
by dynamo and captured under ``fail_on_recompile``; a Python branch on
``is_current_stream_capturing()`` becomes a guard that flips between the
eager warm-up and the capture and fails every graph. The op is opaque to
dynamo (``register_fake`` gives it a shape-only meta kernel), so its body
runs for real on every call and can branch freely. Custom ops can't take a
``CommGroup``, so each group registers itself in a process-global table under
its process group's name once ``init_dist`` has created that group (the
``CommGroup`` objects themselves are built in the parent and pickled to the
workers). The name is identical across ranks and across restarts, so compiled
code that bakes it in stays valid in inductor's on-disk cache.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

BACKEND_ENV = "MSTAR_ALLREDUCE_BACKEND"
MAX_TOKENS_ENV = "MSTAR_FLASHINFER_ALLREDUCE_MAX_TOKENS"

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# Resolved at import so traced code reads plain constants. The process
# environment is fixed before the worker starts, so this loses nothing.
_ENABLED = os.environ.get(BACKEND_ENV, "nccl") == "flashinfer"
_MAX_TOKENS = int(os.environ.get(MAX_TOKENS_ENV, "1024"))

_groups: dict[str, object] = {}
_workspaces: dict[str, object] = {}
_backend_logged = False
_capture_warned = False
_flashinfer_available: bool | None = None


def register(comm_group) -> str:
    """Register ``comm_group`` (whose ``device_group`` must exist) for the
    custom op and return its handle."""
    handle = comm_group.device_group.group_name
    _groups[handle] = comm_group
    return handle


def enabled() -> bool:
    """Whether the FlashInfer backend is selected and importable. Logs the
    resolved backend once. Runtime only: not called from traced code."""
    global _backend_logged, _flashinfer_available
    if not _ENABLED:
        if not _backend_logged:
            _backend_logged = True
            logger.info("CommGroup.all_reduce backend: nccl")
        return False
    if _flashinfer_available is None:
        try:
            import flashinfer.comm  # noqa: F401
        except ImportError:
            _flashinfer_available = False
        else:
            _flashinfer_available = True
    if not _backend_logged:
        _backend_logged = True
        logger.info(
            "CommGroup.all_reduce backend: %s",
            "flashinfer" if _flashinfer_available else "nccl (flashinfer unavailable)",
        )
    return _flashinfer_available


def get_or_create(handle: str, tokens: int, hidden: int, dtype: torch.dtype):
    """The FlashInfer workspace for ``handle``, creating it on first use.

    Returns ``None`` (caller falls back to NCCL) when no workspace exists yet
    and creation isn't legal right now, i.e. under CUDA-graph capture, where
    the collective IPC exchange can't run.
    """
    global _capture_warned
    workspace = _workspaces.get(handle)
    if workspace is not None:
        return workspace
    if torch.cuda.is_current_stream_capturing():
        if not _capture_warned:
            _capture_warned = True
            logger.warning(
                "CUDA-graph capture reached comm_group %s before its FlashInfer "
                "all-reduce workspace existed; the graph will replay NCCL instead",
                handle,
            )
        return None
    from flashinfer import comm

    comm_group = _groups[handle]
    workspace = comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=comm_group.world_size,
        rank=comm_group.rank,
        max_token_num=_MAX_TOKENS,
        hidden_dim=hidden,
        dtype=dtype,
        group=comm_group.device_group,
    )
    logger.info(
        "created FlashInfer all-reduce workspace for comm_group %s: world_size=%d "
        "hidden=%d dtype=%s max_token_num=%d (first call carried %d tokens)",
        handle, comm_group.world_size, hidden, dtype, _MAX_TOKENS, tokens,
    )
    _workspaces[handle] = workspace
    return workspace


@torch.library.custom_op("mstar::flashinfer_all_reduce", mutates_args=())
def flashinfer_all_reduce(input_: torch.Tensor, handle: str) -> torch.Tensor:
    """All-reduce ``input_`` (tokens, hidden) over the registered group,
    through FlashInfer when its workspace can serve the call and through
    NCCL otherwise. Out of place either way: a custom op may not return an
    alias of its input, so the NCCL path reduces a clone."""
    comm_group = _groups[handle]
    tokens, hidden = input_.shape
    workspace = None
    if enabled():
        workspace = get_or_create(handle, tokens, hidden, input_.dtype)
    if workspace is None or not workspace.is_buffer_size_sufficient(
        comm_group.world_size, tokens, hidden, input_.dtype,
    ):
        out = input_.clone()
        dist.all_reduce(out, group=comm_group.device_group)
        return out
    from flashinfer import comm

    return comm.allreduce_fusion(
        input_, workspace, pattern=comm.AllReduceFusionPattern.kAllReduce,
    )


@flashinfer_all_reduce.register_fake
def _flashinfer_all_reduce_fake(input_: torch.Tensor, handle: str) -> torch.Tensor:
    return torch.empty_like(input_)


def all_reduce(comm_group, input_: torch.Tensor) -> torch.Tensor:
    """Replacement for ``dist.all_reduce`` behind ``CommGroup.all_reduce``.
    Routes to the custom op when the backend is on, the dtype is supported
    and the call fits the workspace's token cap; plain in-place NCCL
    otherwise (so long prefills never pay the op's out-of-place copy)."""
    if comm_group.world_size == 1:
        return input_
    orig_shape = input_.shape
    hidden = orig_shape[-1]
    tokens = input_.numel() // hidden
    if (
        not _ENABLED
        or input_.dtype not in _SUPPORTED_DTYPES
        or tokens > _MAX_TOKENS
    ):
        dist.all_reduce(input_, group=comm_group.device_group)
        return input_
    flat = input_.reshape(tokens, hidden)
    out = torch.ops.mstar.flashinfer_all_reduce(flat, comm_group.allreduce_handle)
    return out.view(orig_shape)
