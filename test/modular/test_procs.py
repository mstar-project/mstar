"""Process-lifetime helpers: SIGTERM unwinds the interpreter (atexit runs), and a child asked to
die with its parent goes away when the parent is killed outright."""
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time

import pytest

from mstar.utils.procs import die_with_parent, graceful_sigterm

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl/fork semantics")


@linux_only
def test_graceful_sigterm_runs_atexit(tmp_path):
    # a real interpreter (not a forked mp child, which exits through os._exit without atexit)
    marker = tmp_path / "marker"
    code = (
        "import atexit, os, signal, time; from mstar.utils.procs import graceful_sigterm; graceful_sigterm();"
        f"atexit.register(lambda: open({str(marker)!r}, 'w').write('atexit ran'));"
        "os.kill(os.getpid(), signal.SIGTERM); time.sleep(30)"
    )
    r = subprocess.run([sys.executable, "-c", code], timeout=60, check=False)
    assert r.returncode == 0 and marker.read_text() == "atexit ran"


def _grandchild():
    assert die_with_parent(signal.SIGTERM)
    graceful_sigterm()
    time.sleep(60)


def _parent(q):
    g = mp.get_context("fork").Process(target=_grandchild)
    g.start()
    q.put(g.pid)
    time.sleep(60)  # killed by the test with SIGKILL: no cleanup of its own


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # reaped-but-zombie children of *this* process count as gone
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


@linux_only
def test_child_dies_when_its_parent_is_killed():
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_parent, args=(q,))
    p.start()
    gpid = q.get(timeout=20)
    assert _alive(gpid)
    os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=10)
    deadline = time.time() + 10
    while time.time() < deadline and _alive(gpid):
        time.sleep(0.1)
    assert not _alive(gpid), "orphaned grandchild kept running after its parent was SIGKILLed"
