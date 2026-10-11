"""Start one M* API server in its own process group and tear the whole group down."""
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


class ServerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Server:
    """``with Server(...) as url:`` serves ``config`` from ``source``. The server
    and its children share a process group that is signalled as a whole on exit
    or interrupt; exit raises ServerError if any member survives."""

    def __init__(self, source: Path, config: Path, log_path: Path, *, port: int | None = None,
                 env: dict[str, str] | None = None, startup_timeout: float = 1200,
                 tensor_comm_protocol: str = "SHM", command: list[str] | None = None):
        self.source, self.config, self.log_path = source, config, log_path
        self.tensor_comm_protocol = tensor_comm_protocol
        self.command = command  # default: the M* API server for ``config``
        self.port = port or free_port()
        self.env = env or {}
        self.startup_timeout = startup_timeout
        self.startup_seconds: float | None = None
        self._proc: subprocess.Popen | None = None
        self._scratch: Path | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> str:
        if port_in_use(self.port):
            raise ServerError(f"port {self.port} is already in use")
        self._scratch = Path(tempfile.mkdtemp(prefix="mstar-e2e-"))
        pythonpath = os.pathsep.join(filter(None, [str(self.source), os.environ.get("PYTHONPATH")]))
        env = {**os.environ, "PYTHONPATH": pythonpath, **self.env}
        cmd = self.command or [
            sys.executable, "-u", "-m", "mstar.api_server.entrypoint", "--config", str(self.config),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--socket-path-prefix", str(self._scratch / "ipc"), "--upload-dir", str(self._scratch / "upload"),
            "--tensor-comm-protocol", self.tensor_comm_protocol]
        cmd = [arg.replace("{port}", str(self.port)) for arg in cmd]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w")
        self._proc = subprocess.Popen(cmd, cwd=self.source, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self.url

    def _wait_ready(self):
        start = time.monotonic()
        while True:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                raise ServerError(f"server exited with code {code} during startup; see {self.log_path}")
            try:
                if requests.get(self.url + "/health", timeout=1).status_code == 200:
                    self.startup_seconds = time.monotonic() - start
                    return
            except requests.RequestException:
                pass
            if time.monotonic() - start > self.startup_timeout:
                raise ServerError(f"server not healthy after {self.startup_timeout:.0f} s; see {self.log_path}")
            time.sleep(1)

    def __exit__(self, *exc):
        if self._proc is None:
            return False
        pgid = self._proc.pid
        for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + grace
            while _group_alive(pgid) and time.monotonic() < deadline:
                self._proc.poll()
                time.sleep(0.2)
            if not _group_alive(pgid):
                break
        self._proc.poll()
        self._log.close()
        shutil.rmtree(self._scratch, ignore_errors=True)
        self._proc = None
        if _group_alive(pgid) and exc[0] is None:
            raise ServerError(f"processes in group {pgid} survived SIGKILL")
        return False
