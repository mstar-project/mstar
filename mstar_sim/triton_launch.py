"""Triton kernels on the emulated GPU.

``TRITON_INTERPRET=1`` makes every ``@triton.jit`` function an interpreted one,
whose launch this replaces: nothing runs, and the launch is one kernel of the
simulated device, charged by its memory traffic (the tensor arguments it
touches, an upper bound).
"""

from __future__ import annotations

import torch

from mstar_sim import clock


def _traffic(*args, **kwargs) -> tuple[float, float]:
    nbytes = sum(
        a.numel() * a.element_size()
        for a in (*args, *kwargs.values())
        if isinstance(a, torch.Tensor)
    )
    return 0.0, float(nbytes)


def install() -> None:
    from triton.runtime import interpreter

    run = interpreter.InterpretedFunction.run
    if getattr(run, "__mstar_sim_kernel__", None):
        return

    def launch(self, *args, grid, warmup, **kwargs):
        name = f"triton.{getattr(self.fn, '__name__', 'kernel')}"

        @clock.kernel_fn(name, cost=_traffic, writes_args=True)
        def body(*a, **k):
            return None  # its outputs are value-less: nothing to run

        return body(*args, **kwargs)

    launch.__mstar_sim_kernel__ = True
    interpreter.InterpretedFunction.run = launch
    from triton.runtime import autotuner

    # nothing to benchmark on the emulated GPU (the real path asks for a CUDA driver):
    # every config ties, so the autotuner keeps the first
    autotuner.Autotuner._bench = lambda self, *args, config, **meta: [0.0, 0.0, 0.0]
