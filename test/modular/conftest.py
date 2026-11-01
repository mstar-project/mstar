"""Shared fixtures for modular tests.

Two stubs that exist purely so test imports survive on local macOS dev:

1. ``torch._dynamo.config`` — mstar.engine.__init__ writes flags
   (recompile_limit, specialize_int, etc.) that don't exist on older
   CUDA-less torch builds. We replace the config object with a sink.

2. ``triton`` — mstar.engine.resources.sampler.utils does ``import triton`` at module
   load. Triton is GPU-only and not installed on macOS. We inject a
   minimal stub so the import succeeds; tests that exercise the kernels
   themselves still need a real GPU.

And one autouse fixture that clears the cached platform around every test.
"""
import sys
import types
from typing import NamedTuple

import pytest
import torch

from mstar.platforms import resolve_platform

# --- torch._dynamo.config stub --------------------------------------------

class _DynamoConfigSink:
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return None


try:
    torch._dynamo.config.recompile_limit = 64
except (AttributeError, RuntimeError):
    torch._dynamo.config = _DynamoConfigSink()


# --- triton stub -----------------------------------------------------------

if "triton" not in sys.modules:
    triton = types.ModuleType("triton")
    triton.language = types.ModuleType("triton.language")
    triton.language.constexpr = int
    triton.jit = lambda *a, **k: (lambda f: f)
    triton.cdiv = lambda a, b: -(-a // b)
    triton.Config = lambda *a, **k: a
    triton.autotune = lambda *a, **k: (lambda f: f)
    triton.heuristics = lambda *a, **k: (lambda f: f)
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton.language


# --- Dynamo counts ------------------------------------------------------------

class DynamoCounts(NamedTuple):
    compiles: int  # frames Dynamo compiled, recompiles and empty graphs included
    graph_calls: int  # compiled graphs one call runs once everything is compiled
    traced: frozenset[int]  # data_ptrs of the tensors the compiled graphs take


@pytest.fixture
def dynamo_counts():
    """``count(fn, *args) -> DynamoCounts`` for ``torch.compile(fn,
    dynamic=False)`` called twice on ``args``, the second time under the capture
    stance (``fail_on_recompile``), as the graph runner warms a bucket and then
    captures it: a frame Dynamo calls back into on a warm call fails it. A graph
    count would miss a frame that only guards on an int before a graph break: it
    compiles an empty graph per value."""
    from torch._dynamo.callback import callback_handler

    def count(fn, *args) -> DynamoCounts:
        compiles, calls, traced = [0], [0], set()

        def on_start(*_args):
            compiles[0] += 1

        def backend(gm, example_inputs):
            traced.update(t.data_ptr() for t in example_inputs if isinstance(t, torch.Tensor))

            def run(*inputs):
                calls[0] += 1
                return gm.forward(*inputs)
            return run

        torch._dynamo.reset()
        callback_handler.register_start_callback(on_start)
        try:
            compiled = torch.compile(fn, backend=backend, dynamic=False)
            with torch.no_grad():
                compiled(*args)
                calls[0] = 0
                with torch.compiler.set_stance("fail_on_recompile"):
                    compiled(*args)
        finally:
            callback_handler.remove_start_callback(on_start)
            torch._dynamo.reset()
        return DynamoCounts(compiles[0], calls[0], frozenset(traced))

    return count


# --- torch flag proxies ------------------------------------------------------

@pytest.fixture
def torch_flag():
    """``monkeypatch.setattr`` for torch's flags (``torch._inductor.config``,
    ``torch.backends.*``), restored by assignment. Some of them are served through
    ``__getattr__`` / ``__setattr__``, which monkeypatch restores by deleting the
    attribute since pytest 9.2, removing the flag for every later test in the
    process; one fixture for all of them means nobody has to know which."""
    saved = []

    def set_flag(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    yield set_flag
    for obj, name, value in reversed(saved):
        setattr(obj, name, value)


# --- platform cache ----------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_platform():
    """Clear the cached platform, so a test that stubs ``torch.cuda`` can't
    leak its platform into later tests."""
    resolve_platform.cache_clear()
    yield
    resolve_platform.cache_clear()
