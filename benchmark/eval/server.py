"""Server lifecycle: launch ``mstar serve`` for a run, or attach to a running URL.

A managed server gets its own port, ZMQ socket prefix and upload dir, so two
evals (or an eval and a dev server) on one host never share IPC paths. Its
log goes to a file next to the results, readiness is ``GET /health`` == 200,
and teardown signals the whole process group (API server, conductor and
workers) with SIGINT, then SIGTERM, then SIGKILL.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.eval.schema import redact_url, scrub_argv, scrub_env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


class ServerError(RuntimeError):
    pass


class AttachedServer:
    """A server someone else started; only checked for health."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def __enter__(self) -> AttachedServer:
        if not is_healthy(self.url, timeout=10):
            raise ServerError(f"no healthy server at {redact_url(self.url)}/health")
        return self

    def __exit__(self, *exc) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"managed": False, "url": redact_url(self.url)}


class ManagedServer:
    """``mstar serve <model>`` (or any command) for the duration of a ``with`` block."""

    def __init__(
        self,
        model: str,
        config: str | None = None,
        port: int | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        log_path: str | Path | None = None,
        startup_timeout_s: float = 1800.0,
        command: list[str] | None = None,
        python: str | None = None,
    ):
        self.model = model
        self.python = python or sys.executable
        self.config = config
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.extra_args = list(extra_args or [])
        self.env = dict(env or {})
        self.log_path = Path(log_path) if log_path else None
        self.startup_timeout_s = startup_timeout_s
        self._command = command
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._scratch: tempfile.TemporaryDirectory | None = None
        self.startup_s: float | None = None

    def command(self) -> list[str]:
        if self._command is not None:
            return [a.replace("{port}", str(self.port)) for a in self._command]
        assert self._scratch is not None
        cmd = [self.python, "-m", "mstar.cli.main", "serve", self.model, "--host", "127.0.0.1",
               "--port", str(self.port),
               "--socket-path-prefix", os.path.join(self._scratch.name, "sock") + "/",
               "--upload-dir", os.path.join(self._scratch.name, "uploads") + "/"]
        if self.config:
            cmd += ["--config", self.config]
        return cmd + self.extra_args

    def __enter__(self) -> ManagedServer:
        self.start()
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="mstar-eval-")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "ab")  # noqa: SIM115 - closed in stop()
        self._t0 = time.monotonic()
        self._proc = subprocess.Popen(
            self.command(),
            stdin=subprocess.DEVNULL,
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env={**os.environ, **self.env},
            start_new_session=True,
        )

    def wait_ready(self) -> None:
        assert self._proc is not None
        deadline = self._t0 + self.startup_timeout_s
        poll = 0.05
        while time.monotonic() < deadline:
            code = self._proc.poll()
            if code is not None:
                raise ServerError(f"server exited with code {code} before it was healthy{self._log_hint()}")
            if is_healthy(self.url):
                self.startup_s = time.monotonic() - self._t0
                return
            time.sleep(poll)
            poll = min(1.0, poll * 2)
        raise ServerError(f"server not healthy after {self.startup_timeout_s:.0f} s{self._log_hint()}")

    def _log_hint(self) -> str:
        return f" (log: {self.log_path})" if self.log_path else ""

    def stop(self, grace_s: float = 30.0) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            for sig, wait in ((signal.SIGINT, grace_s), (signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if proc is not None:
            # Workers can outlive the API server they were spawned by; sweep the group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def info(self) -> dict[str, Any]:
        return {
            "managed": True,
            "model": self.model,
            "config": scrub_argv([self.config])[0] if self.config else None,
            "extra_args": scrub_argv(self.extra_args),
            "command": scrub_argv(self._command) if self._command else None,
            "env": scrub_env(self.env),
            "startup_s": self.startup_s,
            "url": redact_url(self.url),
        }
