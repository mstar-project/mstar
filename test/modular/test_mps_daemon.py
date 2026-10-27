"""``mps: true``: the conductor's MPS control daemon, against a fake nvidia-cuda-mps-control."""

import os
import signal
import stat
import subprocess
import sys
import time

import pytest

from mstar.conductor.mps import MpsDaemon

# Daemon mode (-f) runs until a client sends quit; client mode answers only while it runs.
FAKE_CONTROL = """#!/bin/bash
d=$CUDA_MPS_PIPE_DIRECTORY
if [ "$1" = "-f" ]; then
  [ -n "$FAKE_MPS_FAIL" ] && exit 1
  echo $$ > "$d/daemon"
  while [ ! -e "$d/quit" ]; do sleep 0.02; done
  rm -f "$d/daemon"; exit 0
fi
read cmd
[ -e "$d/daemon" ] || exit 1
[ -n "$FAKE_MPS_HANG" ] && exec sleep 30
[ "$cmd" = quit ] && [ -z "$FAKE_MPS_IGNORE_QUIT" ] && touch "$d/quit"
exit 0
"""


@pytest.fixture
def fake_control(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-cuda-mps-control"
    script.write_text(FAKE_CONTROL)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_the_daemon_runs_in_its_own_directory_until_stopped(fake_control, tmp_path):
    mps = MpsDaemon(str(tmp_path / "mps"))
    mps.start()
    pipe = mps.client_env["CUDA_MPS_PIPE_DIRECTORY"]
    assert pipe.startswith(str(tmp_path / "mps"))
    assert os.path.exists(os.path.join(pipe, "daemon"))
    proc = mps._proc

    mps.stop()
    assert proc.poll() == 0  # quit through the pipe, not killed
    assert not os.path.exists(tmp_path / "mps")
    mps.stop()  # idempotent: shutdown can run twice


def test_a_daemon_that_does_not_come_up_is_an_error(fake_control, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MPS_FAIL", "1")
    with pytest.raises(RuntimeError, match="did not come up"):
        MpsDaemon(str(tmp_path / "mps")).start(timeout_s=5)


def test_no_control_binary_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="nvidia-cuda-mps-control"):
        MpsDaemon(str(tmp_path / "mps")).start()


def test_a_daemon_that_ignores_quit_is_killed_and_its_directory_removed(fake_control, tmp_path, monkeypatch):
    mps = MpsDaemon(str(tmp_path / "mps"))
    mps.start()
    proc = mps._proc
    monkeypatch.setenv("FAKE_MPS_IGNORE_QUIT", "1")
    mps.stop(timeout_s=0.5)
    assert proc.poll() is not None
    assert not os.path.exists(tmp_path / "mps")


def test_a_control_command_that_hangs_fails_cleanly(fake_control, tmp_path, monkeypatch):
    """A hung control client is a failed command, not an exception that
    leaves the daemon running and its directory behind."""
    real_run = subprocess.run

    def run(*args, **kwargs):
        kwargs["timeout"] = 0.2
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("FAKE_MPS_HANG", "1")
    mps = MpsDaemon(str(tmp_path / "mps"))
    with pytest.raises(RuntimeError, match="did not come up"):
        mps.start(timeout_s=1)
    assert mps._proc is None
    assert not os.path.exists(tmp_path / "mps")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the parent-death signal is Linux's")
def test_the_daemon_does_not_outlive_a_killed_conductor(fake_control, tmp_path):
    pid_file = tmp_path / "pid"
    conductor = subprocess.Popen([sys.executable, "-c", f"""
import time
from mstar.conductor.mps import MpsDaemon
mps = MpsDaemon({str(tmp_path / "mps")!r})
mps.start()
open({str(pid_file)!r}, "w").write(str(mps._proc.pid))
time.sleep(120)
"""])
    try:
        deadline = time.monotonic() + 60
        while not (pid_file.exists() and pid_file.read_text()):
            assert time.monotonic() < deadline, "the daemon did not start"
            time.sleep(0.05)
        daemon = int(pid_file.read_text())
        conductor.send_signal(signal.SIGKILL)
        conductor.wait()
        deadline = time.monotonic() + 10
        while True:
            try:
                os.kill(daemon, 0)
            except ProcessLookupError:
                break
            # reaped by init once it exits; a zombie still answers kill(0)
            if _is_zombie(daemon):
                break
            assert time.monotonic() < deadline, "the MPS daemon outlived the conductor"
            time.sleep(0.05)
    finally:
        conductor.kill()


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False
