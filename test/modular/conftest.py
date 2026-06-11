"""Shared fixtures for modular tests.

Two stubs that exist purely so test imports survive on local macOS dev:

1. ``torch._dynamo.config`` — mstar.engine.__init__ writes flags
   (recompile_limit, specialize_int, etc.) that don't exist on older
   CUDA-less torch builds. We replace the config object with a sink.

2. ``triton`` — mstar.utils.sampling does ``import triton`` at module
   load. Triton is GPU-only and not installed on macOS. We inject a
   minimal stub so the import succeeds; tests that exercise the kernels
   themselves still need a real GPU.
"""
import sys
import types

import torch

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
