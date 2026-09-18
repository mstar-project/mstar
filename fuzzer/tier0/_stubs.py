"""Stubs that let tier 0 run on a computer with no GPU.

Import this module before you import ``mstar``.

These are the same two stubs as in ``test/modular/conftest.py``:

* ``mstar.engine`` writes flags into ``torch._dynamo.config``. An older torch
  build without CUDA does not have some of those flags.
* ``mstar.engine.resources.sampler.utils`` imports triton when it loads.

Tier 0 uses neither kernel. Thus an object that accepts everything and does
nothing is sufficient.
"""

import sys
import types

import torch


class _DynamoConfigSink:
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return None


try:
    torch._dynamo.config.recompile_limit = 64
except (AttributeError, RuntimeError):
    torch._dynamo.config = _DynamoConfigSink()


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


# mstar logs a warning for each rejected admit, and an error for each
# unservable request. A fuzz run makes thousands of both on purpose. Only the
# failure report is important, so make the logger quiet.
import logging  # noqa: E402

logging.getLogger("mstar").setLevel(logging.CRITICAL)
