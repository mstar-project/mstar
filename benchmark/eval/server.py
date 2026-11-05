"""Server lifecycle: launch ``mstar serve`` for a run, or attach to a running URL.

A managed server gets its own port, ZMQ socket prefix and upload dir, so two
evals (or an eval and a dev server) on one host never share IPC paths. Its
log goes to a file next to the results, readiness is ``GET /health`` == 200,
and teardown signals the whole process group (API server, conductor and
workers) with SIGINT, then SIGTERM, then SIGKILL.
"""

from __future__ import annotations

import functools
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

from benchmark.eval.schema import describe_args, describe_command, git_info, redact_url, scrub_argv, scrub_env

_TREE_PROBE = """
import importlib.metadata as md, importlib.util, json, pathlib
spec = importlib.util.find_spec("mstar")
root = pathlib.Path(spec.origin).resolve().parents[1] if spec and spec.origin else None
def v(d):
    try:
        return md.version(d)
    except md.PackageNotFoundError:
        return None
print(json.dumps({"root": str(root) if root else None,
                  "versions": {d: v(d) for d in ("mstar-ai", "torch", "transformers")}}))
"""

SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
_received: list[int] = []   # every signal the handler saw, for the way out of exit_on_signals
_resend: queue.SimpleQueue[int] = queue.SimpleQueue()
_resender: threading.Thread | None = None


@contextmanager
def exit_on_signals():
    """From here on SIGTERM and SIGHUP raise ``SystemExit(128 + signum)`` and SIGINT raises ``KeyboardInterrupt``,
    so ``with server:`` blocks unwind and stop their servers. SIGTERM or SIGHUP inherited ignored (nohup) stays
    ignored.

    The exception is raised wherever the main thread is. Inside an import that can be Python called from C++
    (pybind11 then aborts the process, and the server's session keeps running) or an import-lock weakref
    callback; there, and in any finalizer, Python drops the exception and the run goes on. So the handler
    doesn't raise inside an import, ``sys.unraisablehook`` catches a dropped one, and either way the signal
    is sent again 50 ms later, until it lands where it can unwind. A block that ends normally after a signal
    still exits as that signal, whatever became of those. That includes one swallowed outright (an ``except
    BaseException`` that doesn't re-raise), which nothing sends again: the run goes on to its end, then exits as
    that signal.
    """
    global _resender
    _received.clear()
    for s in SIGNALS:
        # SIGINT always: a background launch ignores it, stop() starts with it, and a server must not inherit that.
        if s == signal.SIGINT or signal.getsignal(s) != signal.SIG_IGN:
            signal.signal(s, _exit_on_signal)
    if _resender is None:
        _resender = threading.Thread(target=_resend_loop, name="exit-on-signal", daemon=True)
        _resender.start()
    hook = sys.unraisablehook
    if not (isinstance(hook, functools.partial) and hook.func is _unraisable):
        sys.unraisablehook = functools.partial(_unraisable, hook)
    try:
        yield
    except SystemExit as e:
        if e.code or not _received:
            raise
    if _received:
        raise _exit_for(_received[0])


def _exit_for(signum: int) -> BaseException:
    exc = KeyboardInterrupt() if signum == signal.SIGINT else SystemExit(128 + signum)
    exc.signum = signum  # type: ignore[attr-defined]  # for _unraisable, if a finalizer drops it
    return exc


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    _received.append(signum)
    f = frame
    while f is not None:
        if f.f_code.co_filename.startswith("<frozen importlib"):
            _resend.put(signum)
            return
        f = f.f_back
    raise _exit_for(signum)


def _unraisable(hook: Callable[[Any], None], unraisable: Any) -> None:
    signum = getattr(unraisable.exc_value, "signum", None)
    if signum is not None:
        _resend.put(signum)
    hook(unraisable)


def _resend_loop() -> None:
    # A thread started up front: the handler can't start one, it may have interrupted threading's own locks.
    while True:
        signum = _resend.get()
        time.sleep(0.05)
        os.kill(os.getpid(), signum)


@contextmanager
def signals_deferred():
    """Hold SIGTERM, SIGHUP and SIGINT until the block is done, then deliver the first one to its own handler."""
    pending: list[int] = []
    old = {s: signal.getsignal(s) for s in SIGNALS}
    try:
        for s, h in old.items():
            if h != signal.SIG_IGN:   # an ignored signal stays ignored, and can't hide a real one in pending
                signal.signal(s, lambda n, f: pending.append(n))
        yield
    finally:
        stop: BaseException | None = None
        back = [(s, h) for s, h in old.items() if h is not None]   # None: set outside Python, nothing to put back
        while back:   # signal.signal runs a pending handler before it installs, so a raise means "not done yet"
            try:
                while back:
                    signal.signal(*back[-1])
                    back.pop()
            except (KeyboardInterrupt, SystemExit) as e:
                stop = stop or e
        if stop is not None:
            raise stop
        if pending:
            signal.raise_signal(pending[0])


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def python_tree(python: str, env: dict[str, str] | None = None,
                cwd: str | Path | None = None) -> dict[str, Any]:
    """The checkout and package versions ``python`` imports M* from, without importing it.

    With ``--serve-python`` the server runs another venv, often another checkout, so the
    eval's own ``meta.git`` and ``meta.versions`` do not describe what was served. Run with
    the server's cwd and env: ``-c`` puts the cwd first on ``sys.path`` as ``-m`` does.
    """
    try:
        out = subprocess.run([python, "-c", _TREE_PROBE], capture_output=True, text=True,
                             env=env, cwd=cwd, timeout=60, check=True)
        probe = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    root = probe.get("root")
    in_checkout = bool(root) and (Path(root) / "pyproject.toml").is_file()
    return {"git": git_info(root) if in_checkout else {}, "versions": probe.get("versions") or {}}


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
        cwd: str | Path | None = None,
    ):
        self.model = model
        self.python = python or sys.executable
        self._own_python = python is not None
        self.config = config
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.extra_args = list(extra_args or [])
        self.env = dict(env or {})
        self.log_path = Path(log_path) if log_path else None
        self.startup_timeout_s = startup_timeout_s
        self._command = command
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._scratch: tempfile.TemporaryDirectory | None = None
        self._tree: dict[str, Any] | None = None
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

    def environment(self) -> dict[str, str]:
        env = {**os.environ, **self.env}
        if self._own_python and self._command is None:
            python = self.python
            if not os.path.dirname(python):
                # a bare name (`python3.12`) runs from PATH, where Popen finds it, not from the cwd
                python = shutil.which(python, path=env.get("PATH"))
            if python:
                # Tools the server finds on PATH (FlashInfer's JIT needs ninja) come from its own
                # venv, as when that venv is activated; the abspath keeps a venv's bin/python
                # symlink unresolved.
                # A relative path runs from the server's cwd, as Popen resolves it there.
                bin_dir = os.path.dirname(os.path.abspath(os.path.join(self.cwd or "", python)))
                env["PATH"] = os.pathsep.join([bin_dir, env["PATH"]]) if env.get("PATH") else bin_dir
        return env

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
            env=self.environment(),
            cwd=self.cwd,
            start_new_session=True,
        )

    @property
    def pid(self) -> int | None:
        """The launched process; it leads its own session and process group (API server, conductor, workers)."""
        return self._proc.pid if self._proc is not None else None

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

    @signals_deferred()   # a second signal must not cut the escalation or the group sweep short
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
        if self._tree is None:
            self._tree = python_tree(self.python, self.environment(), self.cwd) if self._command is None else {}
        return {
            "managed": True,
            "model": self.model,
            "config": scrub_argv([self.config])[0] if self.config else None,
            "extra_args": describe_args(self.extra_args),
            "command": describe_command(self._command),
            "env": scrub_env(self.env),
            "startup_s": self.startup_s,
            "url": redact_url(self.url),
            # the tree and versions that were served; meta.git and meta.versions are the eval's
            "git": self._tree.get("git", {}),
            "versions": self._tree.get("versions", {}),
        }
