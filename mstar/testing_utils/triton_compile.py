"""Triton kernels in CPU tests: a kernel's launches recorded, then compiled for a GPU.

Triton type-checks a kernel, and the helpers it calls, when a GPU first launches it, so a helper
called with the wrong arguments or a pointer of a type its loads cannot take fails only there.
A test stands ``KernelLaunches`` in for the kernel, runs the real launcher, and
``compile_launches`` builds every launch it made as the GPU's first launch would: the argument
types Triton derives there, the launch's own ``num_warps`` and ``num_stages``, and every tensor
aligned as a GPU allocation is (the compiler picks other layouts for aligned pointers, and with
them a launch can need more shared memory than the GPU has, which it refuses at launch).
Import after ``pytest.importorskip("triton.backends.nvidia")``.
"""
from __future__ import annotations

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import mangle_type


class KernelLaunches:
    """Stands in for a ``@triton.jit`` kernel: records each launch's grid and arguments
    instead of running it."""

    def __init__(self, fn) -> None:
        self.fn, self.calls, self.grids = fn, [], []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.grids.append(grid)
            self.calls.append((args, kwargs))

        return launch

    def arguments(self, call: int = -1) -> dict:
        """A launch's arguments by the kernel's parameter names."""
        args, kwargs = self.calls[call]
        names = self.fn.arg_names
        return dict(zip(names, args, strict=False)) | {k: v for k, v in kwargs.items() if k in names}


def record_launches(monkeypatch, module, name: str) -> KernelLaunches:
    """``module.name`` replaced by a ``KernelLaunches`` of it for the test."""
    launches = KernelLaunches(getattr(module, name))
    monkeypatch.setattr(module, name, launches)
    return launches


# shared memory a block of threads may take, by compute capability (90: H100, H200)
MAX_SHARED_MEMORY = {90: 232448}


def compile_launches(launches: KernelLaunches, capability: int = 90) -> int:
    """Compile each distinct launch in ``launches`` for an NVIDIA GPU of ``capability``
    (90: Hopper) and check it fits the GPU's shared memory; the number compiled. Skips the test
    on a host that cannot run ptxas."""
    assert launches.calls, launches.fn
    names, done = launches.fn.arg_names, set()
    for call in range(len(launches.calls)):
        values = launches.arguments(call)
        options = {k: v for k, v in launches.calls[call][1].items() if k not in names}
        consts = {names[i]: values[names[i]] for i in launches.fn.constexprs}
        signature = {n: "constexpr" if n in consts else mangle_type(v) for n, v in values.items()}
        # what the JIT tells the compiler of a GPU tensor: its address is a multiple of 16
        attrs = {(names.index(n),): [["tt.divisibility", 16]] for n, v in values.items() if torch.is_tensor(v)}
        key = repr((signature, consts, options))
        if key in done:
            continue
        done.add(key)
        try:
            kernel = triton.compile(ASTSource(fn=launches.fn, signature=signature, constexprs=consts, attrs=attrs),
                                    target=GPUTarget("cuda", capability, 32), options=options)
        except (RuntimeError, FileNotFoundError) as exc:
            if "ptxas" in str(exc).lower():
                pytest.skip(f"this host can't run ptxas: {exc}")
            raise
        limit = MAX_SHARED_MEMORY.get(capability)
        assert limit is None or kernel.metadata.shared <= limit, (
            f"{launches.fn.__name__} needs {kernel.metadata.shared} bytes of shared memory, past the {limit} "
            f"of compute capability {capability}: {consts} {options}")
    return len(done)
