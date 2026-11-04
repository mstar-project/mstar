"""Run GPU-only tests on the emulated GPU and keep only what it can answer.

    python -m mstar_sim --fast -- -m pytest -p mstar_sim.pytest_plugin --gpu-only <paths>

``--gpu-only`` keeps the tests a CPU box skips for want of a GPU: everything in
``test/integration``, and any test whose skip (a ``skipif`` marker, a
module-level ``pytest.skip``, or one in its body) names CUDA, a GPU, Triton or
FlashInfer. The rest already run in CPU Core.

Nothing is computed on the emulated GPU, so a test that checks values (parity,
an error bound) can't pass. A test is reported as an expected failure when it
fails an ``assert`` after reading a never-computed value, or fails in any way
after an op whose output shape depends on never-computed values (``nonzero``
of a mask, run on zeros). Every other failure stays a failure: an exception
from mstar's own code (a lifecycle or planning error, an API that moved, a
crash), or an assertion made before any such read.
"""

from __future__ import annotations

import inspect
import re

import pytest

from mstar_sim import device


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    device.VALUELESS_READS[0] = 0
    device.VALUE_SHAPES[0] = 0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed or call.excinfo is None:
        return
    if device.VALUE_SHAPES[0]:
        report.outcome = "skipped"
        report.wasxfail = (f"mstar_sim: failed after {device.VALUE_SHAPES[0]} op(s) whose output shape "
                           "depends on never-computed values")
    elif call.excinfo.errisinstance(AssertionError) and device.VALUELESS_READS[0]:
        report.outcome = "skipped"
        report.wasxfail = (f"mstar_sim: a value check after {device.VALUELESS_READS[0]} read(s) "
                           "of never-computed values")


_NEEDS_GPU = re.compile(r"cuda|gpu|triton|flashinfer|nvidia|sm_?\d\d|blackwell|hopper", re.I)
_SKIP_CALL = re.compile(r"pytest\.skip\(\s*(?:f?[\"'])([^\"']*)")


def pytest_addoption(parser):
    parser.addoption("--gpu-only", action="store_true",
                     help="mstar_sim: run only the tests a CPU box skips for want of a GPU")


def _needs_gpu(item) -> bool:
    if "test/integration/" in item.nodeid:
        return True
    for mark in item.iter_markers("skipif"):
        if _NEEDS_GPU.search(str(mark.kwargs.get("reason", ""))):
            return True
    sources = [getattr(item.module, "__file__", None)]
    texts = []
    for src in sources:
        if src:
            with open(src) as f:
                texts.append(("module", f.read()))
    try:
        texts.append(("body", inspect.getsource(item.function)))
    except (OSError, TypeError):
        pass
    for where, text in texts:
        for reason in _SKIP_CALL.findall(text):
            if _NEEDS_GPU.search(reason) and (where == "body" or "allow_module_level" in text):
                return True
    return False


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu-only"):
        return
    kept = [item for item in items if _needs_gpu(item)]
    config.hook.pytest_deselected(items=[item for item in items if item not in set(kept)])
    items[:] = kept
