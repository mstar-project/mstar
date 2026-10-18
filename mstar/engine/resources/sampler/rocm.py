"""Sampling kernels on AMD GPUs: FlashInfer's ROCm fork.

The fork keeps the per-row device ``seed``/``offset`` tensors that let a captured
graph draw fresh numbers on every replay, so the CUDA call sites carry over
unchanged. Imported only on ROCm.
"""

import inspect
from collections.abc import Callable

import torch

from mstar.utils.flashinfer_rocm import import_flashinfer_rocm


def sampling_from_probs(name: str) -> Callable:
    """The fork's ``flashinfer.sampling.<name>``, checked to take tensor seeds.

    The fork guards its ops against dynamo by raising when traced, where upstream
    graph-breaks; ``torch.compiler.disable`` restores the graph break, so a caller
    traced by dynamo runs the kernel eagerly rather than failing.
    """
    flashinfer = import_flashinfer_rocm("sampling")
    fn = getattr(flashinfer.sampling, name)
    params = inspect.signature(fn).parameters
    if "seed" not in params or "offset" not in params:
        raise RuntimeError(
            f"flashinfer.sampling.{name} in this amd-flashinfer build takes no "
            "seed/offset tensors, which captured sampling needs (0.6.18+amd.1 has them)"
        )
    return torch.compiler.disable(fn)
