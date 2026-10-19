"""``mps: true``: the conductor's MPS control daemon, against a fake nvidia-cuda-mps-control."""

import os
import stat

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
[ "$cmd" = quit ] && touch "$d/quit"
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
