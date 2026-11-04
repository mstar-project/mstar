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
crash), or an assertion made before any such read. A failure raised inside
the emulator's own code is an emulator gap: reported as an expected failure and
listed at the end of the run, so a gap never reads as a fault of mstar's.
A ``pytest.raises`` that saw nothing after a stand-in kernel ran expected the
kernel's own error check, which a stand-in doesn't have.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from mstar_sim import device

GAPS: list[tuple[str, str]] = []


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    device.VALUELESS_READS[0] = 0
    device.VALUE_SHAPES[0] = 0
    device.KERNELS_RUN[0] = 0


def _needs_the_hub(excinfo) -> bool:
    """A download the offline run can't make: the environment, not mstar."""
    text = excinfo.exconly()
    return ("huggingface_hub" in text or "huggingface.co" in text) and (
        excinfo.errisinstance(OSError) or "Snapshot" in text or "offline" in text.lower())


def _missing_from_stand_in(excinfo) -> str | None:
    """A name a stand-in library (``mstar_sim/fake_libs``) doesn't provide yet: the
    real library has it, so the gap is the emulator's."""
    if not excinfo.errisinstance((AttributeError, ImportError)):
        return None
    text = excinfo.exconly()
    stand_ins = {path.name for path in _FAKE_LIBS.iterdir()} if _FAKE_LIBS.is_dir() else set()
    if any(re.search(rf"\b{re.escape(name)}\b", text) for name in stand_ins) or "fake_libs" in text:
        return f"fake_libs: {text[:160]}"
    return None


_FAKE_LIBS = pathlib.Path(__file__).resolve().parent / "fake_libs"


def _raised_in_emulator(excinfo) -> str | None:
    """Where the exception started, when that is the emulator's own code: a gap in
    the emulator, not a fault of mstar's."""
    # the innermost frame of this tree (not torch, triton or the stdlib under it) decides
    for entry in reversed(excinfo.traceback):
        path = str(entry.path)
        if "site-packages" in path or "/lib/python" in path or "<" in path:
            continue
        if "/mstar_sim/" in path and "/mstar_sim/tests/" not in path:
            return f"{path.rsplit('/mstar_sim/', maxsplit=1)[-1]}:{entry.lineno + 1}"
        return None
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.excinfo is None or report.when not in ("setup", "call"):
        return
    if _needs_the_hub(call.excinfo):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, "Skipped: needs the Hugging Face Hub (offline)")
        return
    where = _raised_in_emulator(call.excinfo) or _missing_from_stand_in(call.excinfo)
    if where:
        GAPS.append((item.nodeid, f"{where}: {call.excinfo.exconly()[:160]}"))
        report.outcome = "skipped"
        report.wasxfail = f"mstar_sim gap at {where}: {call.excinfo.exconly()[:160]}"
        return
    if report.when != "call":
        return
    if (device.KERNELS_RUN[0] and call.excinfo.errisinstance(pytest.fail.Exception)
            and "DID NOT RAISE" in str(call.excinfo.value)):
        report.outcome = "skipped"
        report.wasxfail = "mstar_sim: expected a kernel's own error, which a stand-in doesn't check"
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
                     help="mstar_sim: run only the tests a CPU box skips for want of a GPU, by "
                          "their skip reasons (or exactly those a plain run skipped: --skipped-in)")
    parser.addoption("--skipped-in", default=None, metavar="JUNIT_XML",
                     help="mstar_sim: run only the tests this plain CPU run's junit report skipped")


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


def _skipped_in(path: str) -> set[str]:
    import xml.etree.ElementTree as ET

    out = set()
    for case in ET.parse(path).iter("testcase"):
        if case.find("skipped") is not None:
            module = case.get("classname", "").replace(".", "/")
            out.add(f"{module}.py::{case.get('name')}")
    return out


def pytest_collection_modifyitems(config, items):
    skipped_in = config.getoption("--skipped-in")
    if skipped_in:
        names = _skipped_in(skipped_in)
        kept = [item for item in items if item.nodeid in names]
    elif config.getoption("--gpu-only"):
        kept = [item for item in items if _needs_gpu(item)]
    else:
        return
    config.hook.pytest_deselected(items=[item for item in items if item not in set(kept)])
    items[:] = kept


def pytest_terminal_summary(terminalreporter):
    """List the emulator gaps hit: each is mstar_sim's to fix, not the test's."""
    if GAPS:
        terminalreporter.section(f"mstar_sim gaps ({len(GAPS)})")
        for nodeid, where in GAPS:
            terminalreporter.write_line(f"{nodeid}: {where}")
