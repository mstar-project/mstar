"""Run GPU-only tests on the emulated GPU and keep only what it can answer.

    python -m mstar_sim --fast -- -m pytest -p mstar_sim.pytest_plugin --gpu-only <paths>

``--gpu-only`` keeps the tests a CPU box skips for want of a GPU: everything in
``test/integration``, and any test whose skip (a ``skipif`` marker, a
module-level ``pytest.skip``, or one in its body) names CUDA, a GPU, Triton or
FlashInfer, or that checks for one itself (``torch.cuda.is_available()`` in its
body, a fixture or a same-module helper it reaches, or its skipif). The rest
already run in CPU Core. ``--skipped-in`` keeps those a plain run skipped, plus
the ones that check for a GPU in their body
(on a CPU they ``print`` and ``return``, and pass).

Nothing is computed on the emulated GPU, so a test that checks values (parity,
an error bound) can't be answered. A test is reported as an expected failure when:

- its own code (directly, or through ``torch.testing``) read a never-computed value,
  and it then passes or fails an ``assert``: such a check can't answer;
- an ``assert`` or torch's own error fails it at or after an op whose output shape depends
  on never-computed values (``nonzero`` of a mask, run on zeros, or a mask that holds such
  values, as a table a Triton launch was handed does), or that refused them (``multinomial``
  of zeros).

A value is never computed when no kernel computed it (``device.never_computed``): a
large tensor's, one a stand-in wrote, and what kept ops derive from those, until a real
write covers it. So a pass means every value the test itself read was computed, as far
as the emulator can tell. It can't tell of a host value mstar read from one and handed on
(a sampled token in a list, whose check then passes on zeros or fails as mstar's), of an
allocation nothing wrote (it reads as zeros), of another rank's value (the fact doesn't
cross ranks), or whether a real value could show what the test is for (a race, when values
land at launch). The fact is per storage, so a real value beside a never-computed one reads
as never computed, until a whole write.

Every other failure stays a failure: an exception mstar's own code raised (a lifecycle
or planning error, an API that moved, a crash), whatever ran before it, or an assertion
made before any such read. A failure raised inside the emulator's own code, or in a test listed in
``_STAND_IN_LACKS``, is an emulator gap: reported as an expected failure and listed at
the end of the run, so a gap doesn't read as a fault of mstar's. Torch's own error on an
op the interposer ran is mstar's: a ``RuntimeError`` or ``IndexError`` only, so a
``TypeError`` from a bad call stays a gap, and an emulator bug raising a plain
``RuntimeError`` from the interposer reads as mstar's. A ``pytest.raises`` that saw nothing
after a stand-in kernel ran expected the kernel's own error check, which a stand-in doesn't
have.
"""

from __future__ import annotations

import ast
import functools
import inspect
import os
import re
import sys
import textwrap

import pytest

from mstar_sim import device

GAPS: list[tuple[str, str]] = []
# tests of what a stand-in doesn't do, which no rule can tell from a failure of mstar's, so
# each is a gap; one that passes fails the run, so an entry can't go stale
_STAND_IN_LACKS = dict.fromkeys(
    ("test_a_sealed_buffer_reports_flashinfers_number",
     "test_a_captured_paged_wrapper_grows_on_its_first_plan_and_its_later_plans_go_direct",
     "test_every_wrapper_grows_on_the_real_planner_and_its_next_plan_fits"),
    "flashinfer's stand-in planner never runs out of workspace")
# host reads of never-computed values by the test's own code
TEST_READS = [0]


def _uncomputed() -> None:
    frame = sys._getframe(1)
    while frame is not None:  # the reader: the innermost frame that isn't torch's or the emulator's
        path = frame.f_code.co_filename
        if not _library(path) and (_EMULATOR not in path or "/mstar_sim/tests/" in path):
            TEST_READS[0] += _test_code(path)
            return
        frame = frame.f_back


device.uncomputed = _uncomputed


def _library(path: str) -> bool:
    return "site-packages" in path or "/lib/python" in path or "<" in path


def _test_code(path: str) -> bool:
    return "/test/" in path or "/tests/" in path or os.path.basename(path).startswith(("test_", "conftest"))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    TEST_READS[0] = 0
    device.VALUE_SHAPES[0] = 0
    device.KERNELS_RUN[0] = 0
    # a device-side assert poisons the process, as on CUDA; each test reports its own
    device.FAULT[:] = ["", 0.0]


def _needs_the_hub(excinfo) -> bool:
    """A download the offline run can't make: the environment, not mstar."""
    text = excinfo.exconly()
    return ("huggingface_hub" in text or "huggingface.co" in text) and (
        excinfo.errisinstance(OSError) or "Snapshot" in text or "offline" in text.lower())


def _missing_from_stand_in(excinfo) -> str | None:
    """A name an emulated object (a ``torch.cuda`` stand-in, a ``fake_libs`` library)
    doesn't provide yet: the real one has it, so the gap is the emulator's."""
    error = excinfo.value
    if isinstance(error, AttributeError):
        owner = getattr(error, "obj", None)  # what lacked the attribute
        if isinstance(owner, device.SimTensor):
            return None  # it has all a tensor has: the name is mstar's mistake
        module = owner if inspect.ismodule(owner) else sys.modules.get(
            getattr(owner if isinstance(owner, type) else type(owner), "__module__", None) or "")
    elif isinstance(error, ImportError):
        module = sys.modules.get((error.name or "").split(".")[0])  # what it was imported from
    else:
        return None
    if _EMULATOR in str(getattr(module, "__file__", "") or ""):
        return f"{module.__name__}: {excinfo.exconly()[:160]}"
    return None


_EMULATOR = "/mstar_sim/"


_INTERPOSERS = ("/mstar_sim/device.py", "/mstar_sim/dist.py")  # what runs torch's ops for mstar's code


def _torchs_verdict(excinfo) -> bool:
    """A torch error on an op's operands, which a GPU's op raises too (a shape, an index):
    torch raised it, not a ``raise`` of this tree's, and not about what the emulator made of
    them (no shape-only kernel, a meta payload, a limit of torch's CPU kernels or of gloo, the
    backend standing in for NCCL, a device mix among payloads: one CUDA refuses is the
    emulator's own check, and tagged)."""
    raised_in = str(excinfo.traceback[-1].path)
    return (excinfo.errisinstance((RuntimeError, IndexError)) and not excinfo.errisinstance(NotImplementedError)
            and (_library(raised_in) or raised_in.endswith(_INTERPOSERS))
            and not re.search("meta|mstar_sim|two devices|Unsupported dtype|not implemented for|"
                              "Gloo.*support|does not support", str(excinfo.value)))


def _raised_in_emulator(excinfo) -> str | None:
    """Where the exception started, when that is the emulator's own code: a gap in
    the emulator, not a fault of mstar's. What the emulator raises as the GPU stack
    would (out of memory, a device-side assert, a capture refusal) is mstar's, and so
    is torch's verdict on an op the interposer ran for mstar's code."""
    if getattr(excinfo.value, "_mstar_sim_gpu", False):
        return None
    verdict = _torchs_verdict(excinfo)
    # the innermost frame of this tree (not torch, triton or the stdlib under it) decides
    for entry in reversed(excinfo.traceback):
        path = str(entry.path)
        if _library(path):
            continue
        if _EMULATOR not in path or "/mstar_sim/tests/" in path:
            return None
        if not (verdict and path.endswith(_INTERPOSERS)):  # its caller decides
            return f"{path.rsplit(_EMULATOR, maxsplit=1)[-1]}:{entry.lineno + 1}"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    lacks = _STAND_IN_LACKS.get(item.name.split("[")[0])
    if lacks and report.passed and report.when == "call":
        report.outcome = "failed"
        report.longrepr = f"mstar_sim: passed, though listed in _STAND_IN_LACKS ({lacks}): delete its entry"
        return
    if report.passed and report.when == "call" and TEST_READS[0]:
        report.outcome = "skipped"
        report.wasxfail = f"mstar_sim: passed, but it read {TEST_READS[0]} never-computed value(s): unanswered"
        return
    if not report.failed or call.excinfo is None or report.when not in ("setup", "call"):
        return
    if _needs_the_hub(call.excinfo):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, "Skipped: needs the Hugging Face Hub (offline)")
        return
    where = lacks or _raised_in_emulator(call.excinfo) or _missing_from_stand_in(call.excinfo)
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
    if device.VALUE_SHAPES[0] and (call.excinfo.errisinstance(AssertionError) or _torchs_verdict(call.excinfo)):
        report.outcome = "skipped"
        report.wasxfail = (f"mstar_sim: failed after {device.VALUE_SHAPES[0]} op(s) whose output shape "
                           "depends on, or that refused, never-computed values")
    elif call.excinfo.errisinstance(AssertionError) and TEST_READS[0]:
        report.outcome = "skipped"
        report.wasxfail = f"mstar_sim: a value check after it read {TEST_READS[0]} never-computed value(s)"


_NEEDS_GPU = re.compile(r"cuda|gpu|triton|flashinfer|nvidia|sm_?\d\d|blackwell|hopper", re.I)
_SKIP_CALL = re.compile(r"pytest\.skip\(\s*(?:f?[\"'])([^\"']*)")


def pytest_addoption(parser):
    parser.addoption("--gpu-only", action="store_true",
                     help="mstar_sim: run only the tests a CPU box skips for want of a GPU, by "
                          "their skip reasons (or exactly those a plain run skipped: --skipped-in)")
    parser.addoption("--skipped-in", default=None, metavar="JUNIT_XML",
                     help="mstar_sim: run only the tests this plain CPU run's junit report skipped")


# a GPU check whose CPU branch passes (``print(...); return``) or a skipif whose reason names no GPU
_CHECKS_GPU = re.compile(r"cuda\.(?:is_available|device_count)\(")
_SKIPIF_GPU = re.compile(r"skipif\(\s*[^,)]*cuda\.(?:is_available|device_count)\(")


def _checks_gpu(item) -> bool:
    """The test, a fixture it takes or a same-module helper it reaches (through any
    number of calls) checks for a GPU."""
    module = getattr(item, "module", None)
    todo, seen = [item.function, *(getattr(module, n, None) for n in getattr(item, "fixturenames", ()))], set()
    while todo:
        fn = todo.pop()
        fn = getattr(fn, "__wrapped__", fn)  # a fixture's own function
        if not inspect.isfunction(fn) or fn in seen or (
                fn is not item.function and fn.__module__ != getattr(module, "__name__", None)):
            continue
        seen.add(fn)
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if _CHECKS_GPU.search(source):
            return True
        todo.extend(getattr(module, n, None) for n in set(re.findall(r"\b(\w+)\(", source)))
    return any(_SKIPIF_GPU.search(_marks_source(node.obj, module))
               for node, _ in item.iter_markers_with_node("skipif"))


@functools.cache
def _marks_source(obj, module) -> str:
    """The marks a test function, class or module sets: its decorators and ``pytestmark``,
    with the module-level marks they name (``requires_cuda = pytest.mark.skipif(...)``)."""
    tree = _parse(obj)
    if tree is None:
        return ""
    scope = tree if inspect.ismodule(obj) else tree.body[0]
    marks = [*getattr(scope, "decorator_list", ()), *_assigns(scope).get("pytestmark", ())]
    named = _assigns(_parse(module)) if module is not None else {}
    marks += [a for m in marks for n in ast.walk(m) if isinstance(n, ast.Name) for a in named.get(n.id, ())]
    return "\n".join(ast.unparse(m) for m in marks)


@functools.cache
def _parse(obj) -> ast.Module | None:
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError):
        return None


def _assigns(scope) -> dict[str, list[ast.Assign]]:
    out: dict[str, list[ast.Assign]] = {}
    for s in getattr(scope, "body", ()):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(s)
    return out


def _needs_gpu(item) -> bool:
    if "test/integration/" in item.nodeid or _checks_gpu(item):
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


def _skipped_in(path: str) -> set[tuple[str, ...]]:
    """The tests a junit report skipped, as junitxml names them (``(classname, name)``);
    a module skipped at collection is one case named by its module (``("", module)``)."""
    import xml.etree.ElementTree as ET

    return {(case.get("classname", ""), case.get("name")) for case in ET.parse(path).iter("testcase")
            if case.find("skipped") is not None}


def _junit_names(item) -> set[tuple[str, ...]]:
    """The junit names that skip ``item``: its own case, and its module's."""
    from _pytest.junitxml import mangle_test_address

    names = mangle_test_address(item.nodeid)
    return {(".".join(names[:-1]), names[-1]), ("", names[0])}


def pytest_collection_modifyitems(config, items):
    skipped_in = config.getoption("--skipped-in")
    if skipped_in:
        names = _skipped_in(skipped_in)
        kept = [item for item in items if _junit_names(item) & names or _checks_gpu(item)]
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
