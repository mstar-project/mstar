"""CUDA MPS for a deployment that runs several workers on one GPU (``mps: true``).

Without MPS, kernels from different processes on one GPU time-slice; under MPS they
overlap. Each deployment runs its own control daemon in a private pipe directory, so
only its workers become MPS clients and other processes on the GPU are unaffected.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

CONTROL = "nvidia-cuda-mps-control"

# Execs the control daemon with the parent-death signal set (Linux), so it gets
# SIGTERM if the conductor dies without stopping it. The signal follows the
# thread that started it, so only the main thread's daemon is tied this way.
_EXEC_TIED_TO_PARENT = """
import ctypes, os, signal, sys
parent = int(sys.argv[1])
ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
if os.getppid() != parent:
    sys.exit(1)  # the conductor died before the signal was set
os.execvp(sys.argv[2], sys.argv[2:])
"""


class MpsDaemon:
    """An ``nvidia-cuda-mps-control`` child of the conductor, and the environment
    its clients need."""

    def __init__(self, directory: str | None = None):
        self.directory = directory or tempfile.mkdtemp(prefix="mstar_mps_")
        self.client_env = {
            "CUDA_MPS_PIPE_DIRECTORY": os.path.join(self.directory, "pipe"),
            "CUDA_MPS_LOG_DIRECTORY": os.path.join(self.directory, "log"),
        }
        self._proc: subprocess.Popen | None = None

    def start(self, timeout_s: float = 10.0) -> None:
        if shutil.which(CONTROL) is None:
            raise RuntimeError(f"mps: true needs {CONTROL} on PATH (it ships with the NVIDIA driver)")
        for path in self.client_env.values():
            os.makedirs(path, exist_ok=True)
        # Foreground mode keeps the daemon our child, so stop() can reap it by PID.
        command = [CONTROL, "-f"]
        if sys.platform.startswith("linux") and threading.current_thread() is threading.main_thread():
            command = [sys.executable, "-c", _EXEC_TIED_TO_PARENT, str(os.getpid()), *command]
        try:
            with open(os.path.join(self.client_env["CUDA_MPS_LOG_DIRECTORY"], "stderr.log"), "w") as err:
                self._proc = subprocess.Popen(
                    command,
                    env={**os.environ, **self.client_env},
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                )
            deadline = time.monotonic() + timeout_s
            while not self._command("get_server_list"):
                if self._proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"MPS control daemon did not come up; see {self.client_env['CUDA_MPS_LOG_DIRECTORY']}")
                time.sleep(0.05)
        except BaseException:
            self.stop()
            raise
        logger.info("MPS control daemon %d started in %s", self._proc.pid, self.directory)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Quit the daemon (and the MPS server it spawned), killing it if it
        does not quit, and remove its directory. Call after the workers exit."""
        proc, self._proc = self._proc, None
        try:
            if proc is not None and proc.poll() is None and self._command("quit"):
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    logger.warning("MPS control daemon %d did not quit; killing", proc.pid)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("MPS control daemon %d survived SIGKILL", proc.pid)
            shutil.rmtree(self.directory, ignore_errors=True)

    def _command(self, command: str) -> bool:
        """Send one command to the daemon; False when it fails or hangs."""
        try:
            result = subprocess.run(
                [CONTROL], input=f"{command}\n", text=True, capture_output=True,
                env={**os.environ, **self.client_env}, timeout=10, check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("MPS control command %r timed out", command)
            return False
        return result.returncode == 0
