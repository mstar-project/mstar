"""CUDA MPS for a deployment that runs several workers on one GPU (``mps: true``).

Without MPS, kernels from different processes on one GPU time-slice; under MPS they
overlap. Each deployment runs its own control daemon in a private pipe directory, so
only its workers become MPS clients and other processes on the GPU are unaffected.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)

CONTROL = "nvidia-cuda-mps-control"


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
        with open(os.path.join(self.client_env["CUDA_MPS_LOG_DIRECTORY"], "stderr.log"), "w") as err:
            self._proc = subprocess.Popen(
                [CONTROL, "-f"],
                env={**os.environ, **self.client_env},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
            )
        deadline = time.monotonic() + timeout_s
        while not self._command("get_server_list"):
            if self._proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(
                    f"MPS control daemon did not come up; see {self.client_env['CUDA_MPS_LOG_DIRECTORY']}")
            time.sleep(0.05)
        logger.info("MPS control daemon %d started in %s", self._proc.pid, self.directory)

    def stop(self) -> None:
        """Quit the daemon (and the MPS server it spawned). Call after the workers exit."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            self._command("quit")
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("MPS control daemon %d did not quit; killing", proc.pid)
                proc.kill()
                proc.wait(timeout=5)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _command(self, command: str) -> bool:
        result = subprocess.run(
            [CONTROL], input=f"{command}\n", text=True, capture_output=True,
            env={**os.environ, **self.client_env}, timeout=10, check=False,
        )
        return result.returncode == 0
