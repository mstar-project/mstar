"""Per-node float32 math precision, applied by the engine around a node's forward.

``apply_torch_config`` sets ``float32_matmul_precision('high')`` (TF32) for the
whole process. A submodule that needs full fp32 for its own work declares it
(``NodeSubmodule.float32_matmul_precision`` / ``cudnn_allow_tf32``) instead of
flipping the process-wide flags itself, and the engine sets them around that
node's forward launch and its warmup and capture only.

TF32 is a kernel choice made on the host when a kernel is launched, compiled or
captured, so the scope has to cover the launch, the first (compiling) call and
the capture; a replayed graph keeps what it was captured with. Dynamo cannot
trace a flag change inside a compiled region, which is why this wraps the
compiled callable rather than living in the model's forward.

The flags are process-wide, not per thread. While a node's forward is being
launched on the GPU thread, other threads of the worker (the main thread's
postprocess of the previous step, the plan thread's pre-plan) see the node's
setting too. The engine only changes them on the thread that runs forwards
(and the one that captures, before serving), so its save and restore never
interleave with each other. The process default is set once at import; the
executor threads' initializers apply only the per-thread dynamo config
(``apply_dynamo_config``), so a thread starting mid-step cannot reset a node's
setting. A model that flips the flags itself still can.
"""
import contextlib
from typing import NamedTuple

import torch

_MATMUL_PRECISIONS = ("highest", "high", "medium")
_NO_CHANGE = contextlib.nullcontext()


class Fp32Precision(NamedTuple):
    """A node's float32 settings; None leaves the process default in place."""

    # ``torch.set_float32_matmul_precision``: "highest" is full fp32
    matmul: str | None
    # ``torch.backends.cudnn.allow_tf32``: cuDNN convolutions and RNNs
    cudnn_allow_tf32: bool | None


def node_fp32_precision(node_name: str, submodule) -> Fp32Precision | None:
    """The submodule's declared settings, checked when the engine loads it.
    None when it declares neither."""
    matmul = getattr(submodule, "float32_matmul_precision", None)
    cudnn = getattr(submodule, "cudnn_allow_tf32", None)
    if matmul is not None and matmul not in _MATMUL_PRECISIONS:
        raise ValueError(
            f"{node_name}: float32_matmul_precision must be one of "
            f"{_MATMUL_PRECISIONS} or None, got {matmul!r}"
        )
    if cudnn is not None and not isinstance(cudnn, bool):
        raise ValueError(f"{node_name}: cudnn_allow_tf32 must be a bool or None, got {cudnn!r}")
    if matmul is None and cudnn is None:
        return None
    return Fp32Precision(matmul, cudnn)


@contextlib.contextmanager
def cudnn_benchmark(enabled: bool):
    """Within the block, let cuDNN time its conv algorithms per new shape
    (once, then cached) instead of taking its heuristic pick; restores the
    process-wide flag on exit. Not safe against convs on other threads, like
    the scope below: the engine applies it around a node's capture warm-up
    (``NodeSubmodule.capture_cudnn_benchmark``), and the VAE decoders around
    their forward on the worker's one GPU thread."""
    if not enabled:
        yield
        return
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous


def fp32_precision_scope(precision: Fp32Precision | None):
    """Apply ``precision`` for the duration of the block, then restore what
    was there before, also when the block raises."""
    if precision is None:
        return _NO_CHANGE
    return _applied(precision)


# torch keeps a legacy flag next to the per-backend ones, and the legacy getters
# raise once the two disagree: ``matmul.allow_tf32`` alone changes only the CUDA
# backend, for one. Then the legacy value is read off the CUDA backend, which is
# what the cuBLAS getter checks it against.
def _legacy_matmul() -> str:
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:
        return "high" if torch.backends.cuda.matmul.fp32_precision == "tf32" else "highest"


def _legacy_cudnn() -> bool:
    try:
        return torch.backends.cudnn.allow_tf32
    except RuntimeError:
        return torch.backends.cudnn.conv.fp32_precision == "tf32"


@contextlib.contextmanager
def _applied(precision: Fp32Precision):
    backends = torch.backends
    saved_matmul = saved_cudnn = None
    try:
        if precision.matmul is not None:
            saved_matmul = (
                _legacy_matmul(),
                backends.cuda.matmul.fp32_precision,
                backends.mkldnn.matmul.fp32_precision,
            )
            # the legacy setter, which keeps the legacy flag and both backends
            # in agreement: inductor reads the legacy getters while it compiles
            torch.set_float32_matmul_precision(precision.matmul)
        if precision.cudnn_allow_tf32 is not None:
            saved_cudnn = (
                _legacy_cudnn(),
                backends.cudnn.conv.fp32_precision,
                backends.cudnn.rnn.fp32_precision,
            )
            backends.cudnn.allow_tf32 = precision.cudnn_allow_tf32
            # the legacy setter writes "none" for False, which inherits a "tf32"
            # set on cudnn.fp32_precision, so name the ops' precision outright
            op = "tf32" if precision.cudnn_allow_tf32 else "ieee"
            backends.cudnn.conv.fp32_precision = op
            backends.cudnn.rnn.fp32_precision = op
        yield
    finally:
        # the legacy flag first: its setters also write the backends, which
        # then go back to exactly what they were
        if saved_matmul is not None:
            legacy, cuda, mkldnn = saved_matmul
            torch.set_float32_matmul_precision(legacy)
            backends.cuda.matmul.fp32_precision = cuda
            backends.mkldnn.matmul.fp32_precision = mkldnn
        if saved_cudnn is not None:
            legacy, conv, rnn = saved_cudnn
            backends.cudnn.allow_tf32 = legacy
            backends.cudnn.conv.fp32_precision = conv
            backends.cudnn.rnn.fp32_precision = rnn
