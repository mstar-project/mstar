"""Server lifecycle: launch ``mstar serve`` for a run, or attach to a running URL.

A managed server gets its own port, ZMQ socket prefix and upload dir, so two
evals (or an eval and a dev server) on one host never share IPC paths. Its
log goes to a file next to the results, readiness is ``GET /health`` == 200,
and teardown signals the whole process group (API server, conductor and
workers) with SIGINT, then SIGTERM, then SIGKILL.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn

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
RETRY_S = 0.05
# Past this a held signal raises anyway (outside ``signals_deferred``): a rare abort beats a run that can't be
# stopped. The server's teardown still follows, so a supervisor's SIGKILL grace has to cover both.
HOLD_LIMIT_S = 20.0
_received: int | None = None   # the first signal, for the way out of exit_on_signals
_held: int | None = None   # a signal that couldn't raise where it landed, until it does
_held_since = 0.0
_deferred = 0   # open ``signals_deferred`` blocks


@contextmanager
def exit_on_signals():
    """From here on SIGTERM and SIGHUP raise ``SystemExit(128 + signum)`` and SIGINT raises ``KeyboardInterrupt``,
    so ``with server:`` blocks unwind and stop their servers. SIGTERM or SIGHUP inherited ignored (nohup) stays
    ignored.

    The exception is raised wherever the main thread is, and some places can't carry it. Inside an import that
    can be Python called from C++ (pybind11 then aborts the process, and the server's session keeps running) or
    an import-lock weakref callback; in a finalizer Python drops it; in an event loop callback aiohttp's parser
    swallows it, or it stops the callback half done and leaves a task that never wakes. So the handler holds the
    signal inside an import, ``sys.unraisablehook`` holds a dropped one, and under an event loop it is raised
    between callbacks. A held signal is tried again every 50 ms, and raises in place after ``HOLD_LIMIT_S``
    unless ``signals_deferred`` holds it (a server's teardown runs to its end).

    A block that ends without having unwound, whatever became of the exception, still exits as the first signal.
    """
    global _received, _held
    _received = _held = None
    for s in SIGNALS:
        # SIGINT always: a background launch ignores it, stop() starts with it, and a server must not inherit that.
        if s == signal.SIGINT or signal.getsignal(s) != signal.SIG_IGN:
            signal.signal(s, _exit_on_signal)
    signal.signal(signal.SIGALRM, _retry_held)
    hook = sys.unraisablehook
    if not (isinstance(hook, functools.partial) and hook.func is _unraisable):
        sys.unraisablehook = functools.partial(_unraisable, hook)
    try:
        yield
    except (Exception, SystemExit) as e:
        if _received is None or (isinstance(e, SystemExit) and e.code):
            raise
        if isinstance(e, Exception):   # it may be the signal's own, turned into this; still worth seeing
            traceback.print_exception(e)
    if _received is not None:
        _raise(_received)


def _in_import(frame: FrameType | None) -> bool:
    while frame is not None:
        if frame.f_code.co_filename.startswith("<frozen importlib"):
            return True
        frame = frame.f_back
    return False


def _hold(signum: int) -> None:
    global _held, _held_since
    if _held is None:
        _held, _held_since = signum, time.monotonic()
    if not _deferred:
        signal.setitimer(signal.ITIMER_REAL, RETRY_S)


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    global _received
    if _received is None:
        _received = signum
    loop = asyncio._get_running_loop()
    expired = _held is not None and time.monotonic() - _held_since > HOLD_LIMIT_S
    if _deferred or (not expired and (loop is not None or _in_import(frame))):
        _hold(signum)
        if loop is not None:
            loop.call_soon_threadsafe(_raise_held)
        return
    _raise(signum)


def _raise_held() -> None:
    if _held is not None and not _deferred:
        _raise(_held)


def _raise(signum: int) -> NoReturn:
    global _held
    _held = None
    signal.setitimer(signal.ITIMER_REAL, 0)
    exc = KeyboardInterrupt() if signum == signal.SIGINT else SystemExit(128 + signum)
    exc.signum = signum  # type: ignore[attr-defined]  # for _unraisable, if a finalizer drops it
    raise exc


def _retry_held(signum: int, frame: FrameType | None) -> None:  # noqa: ARG001
    if _held is not None:
        _exit_on_signal(_held, frame)


def _unraisable(hook: Callable[[Any], None], unraisable: Any) -> None:
    try:
        hook(unraisable)
    finally:   # after the hook: a retry during it would be dropped too
        signum = getattr(unraisable.exc_value, "signum", None)
        if signum is not None:
            _hold(signum)


@contextmanager
def signals_deferred():
    """Under ``exit_on_signals``, hold SIGTERM, SIGHUP and SIGINT until the block is done, then deliver the first."""
    global _deferred
    _deferred += 1
    try:
        yield
    finally:
        _deferred -= 1
        if not _deferred and _held is not None:
            signal.raise_signal(_held)


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

    @signals_deferred()   # under exit_on_signals, a second signal can't cut the escalation or group sweep short
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
