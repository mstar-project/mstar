"""What the harness asks of a backend.

A backend is a class with ``from_args(args)`` and, when it has flags of its own, ``add_arguments(parser)``; the
runner (``benchmark.step_cache.run``) finds it by name in :data:`BACKENDS`, lets it add its flags, and builds it
from the parsed arguments. The harness holds no GPU claim: whoever starts a run holds the GPU.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any, NamedTuple, Protocol

import torch

from benchmark.step_cache.methods import Method

# name -> "module:class"; imported on first use, so a backend's own dependencies load only when it runs
BACKENDS = {
    "server": "benchmark.step_cache.backends.server:ServerBackend",
    "inproc": "benchmark.step_cache.backends.inproc:InprocBackend",
}


class Generated(NamedTuple):
    # uint8 [T, H, W, 3] on the CPU
    frames: torch.Tensor
    # wall time of the request (server), or of the denoise loop and the decode (inproc)
    seconds: float
    # the denoise loop alone, when the backend can time it
    dit_seconds: float | None = None
    # steps that ran blocks
    computed_steps: int | None = None
    # block-stack runs in units of one guidance branch through every block (``methods.plan_cost``)
    branch_forwards: float | None = None
    # anything else worth keeping in the records (a plan string, a latents path)
    extra: dict[str, Any] | None = None


class Backend(Protocol):
    name: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Backend: ...

    def describe(self) -> dict[str, Any]:
        """The hardware, torch and CUDA versions, the model, and the tree's sha when known."""

    def configure(self, method: Method) -> None:
        """Make the next :meth:`generate` calls run ``method``."""

    def generate(self, prompt: str, seed: int, *, size: str, frames: int, steps: int) -> Generated: ...

    def close(self) -> None: ...


def backend_class(name: str) -> type:
    """The backend class registered as ``name``."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; the backends are {sorted(BACKENDS)}")
    module, _, attribute = BACKENDS[name].partition(":")
    try:
        return getattr(importlib.import_module(module), attribute)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        raise ValueError(f"the {name!r} backend ({module}) is not in this tree") from error
