"""Start one M* API server in its own process group and tear the whole group down."""
import contextlib
import errno
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

HOST = "127.0.0.1"  # where every server binds
BIND_FAILED = 3  # the entrypoint's exit code when uvicorn never came up
HELD_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

# Runs in the server's process before exec. SIGINT, the entrypoint's graceful
# stop, may be inherited ignored (a background job of a script); restore it. On
# Linux, ask for SIGKILL when the runner dies, so a runner killed outright does
# not orphan the server; its conductor and workers already exit with their parent.
# The kernel sends it when the launching thread exits, so argv[1] is 0 (none)
# unless the runner's main thread launched the server.
LAUNCHER = """
import ctypes, os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
if sys.platform == "linux" and sys.argv[1] != "0":
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != int(sys.argv[1]):
        sys.exit(1)
os.execvp(sys.argv[2], sys.argv[2:])
"""

class ServerError(RuntimeError):
    pass


class PortTakenError(ServerError):
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def port_in_use(port: int) -> bool:
    """Whether a server could not bind ``port`` on HOST now. Binds with
    SO_REUSEADDR, as uvicorn does, so the kernel decides exactly what the
    server would meet (an IPv6-only listener does not count, a dual-stack one
    does) and the probe never keeps the server from binding."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return True
            raise
        return False


def pick_port(tries: int = 10) -> int:
    """A port from free_port() that port_in_use() also reads as free. A process
    forked while free_port()'s socket was open holds a copy of it until it
    execs, so a fresh port can briefly read as busy; pick another then."""
    for _ in range(tries):
        port = free_port()
        if not port_in_use(port):
            return port
    raise ServerError(f"no free port found in {tries} tries")


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def listening_sockets(port: int) -> set[str] | None:
    """Inodes of the sockets listening on ``port``, any address; None without /proc."""
    inodes, readable = set(), False
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        readable = True
        for row in rows:
            fields = row.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    return inodes if readable else None


def group_pids(pgid: int) -> list[int] | None:
    """Live processes in group ``pgid``; None without /proc."""
    if not Path("/proc/self").exists():
        return None
    pids = []
    for entry in Path("/proc").iterdir():
        try:
            if entry.name.isdigit() and os.getpgid(int(entry.name)) == pgid:
                pids.append(int(entry.name))
        except OSError:
            continue
    return sorted(pids)


def group_sockets(pgid: int) -> set[str]:
    """Inodes of every socket open in process group ``pgid``."""
    inodes = set()
    for pid in group_pids(pgid) or []:
        try:
            fds = list(Path(f"/proc/{pid}/fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(fd)
            except OSError:  # closed since the listing; the others still count
                continue
            if link.startswith("socket:["):
                inodes.add(link[8:-1])
    return inodes


def _survivors(pgid: int) -> list[int]:
    if not _group_alive(pgid):
        return []
    return group_pids(pgid) or [pgid]


def _scratch_parent() -> str:
    """A temp dir short enough for the server's AF_UNIX socket paths (107 bytes)."""
    tmp = tempfile.gettempdir()
    return tmp if len(tmp) <= 40 or not os.path.isdir("/tmp") else "/tmp"


class Server:
    """``with Server(...) as url:`` serves ``config`` from ``source``. The server
    and its children share a process group that is torn down on exit or
    interrupt; exit raises ServerError if any member survives.

    Without ``port``, a free port is picked at launch and, if another process
    takes it while the model loads, the server is restarted once on a fresh one,
    unless part of the first group survived SIGKILL. A given ``port`` is never
    shared: if another process serves it, entering fails.
    """

    def __init__(self, source: Path, config: Path, log_path: Path, *, port: int | None = None,
                 env: dict[str, str] | None = None, startup_timeout: float = 1200,
                 tensor_comm_protocol: str = "SHM", command: list[str] | None = None,
                 stop_grace: float = 30, settle: float = 10, kill_grace: float = 10):
        self.source, self.config, self.log_path = source, config, log_path
        self.tensor_comm_protocol = tensor_comm_protocol
        self.command = command  # default: the M* API server for ``config``
        self.requested_port = port
        self.port = port
        self.env = env or {}
        self.startup_timeout = startup_timeout
        self.stop_grace, self.settle, self.kill_grace = stop_grace, settle, kill_grace
        self.startup_seconds: float | None = None
        self.exit_code: int | None = None  # set if the server exits before teardown
        self.restarts = 0
        self.survivors: list[int] = []  # pids still alive after SIGKILL
        self._next_port: int | None = None
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._scratch: Path | None = None

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def choose_port(self) -> int:
        """Pick the port for the next launch now. Call it before starting a
        thread that forks (the VRAM sampler), so no fork copies the picking socket."""
        if self._next_port is None:
            self._next_port = self.requested_port or pick_port()
        return self._next_port

    def __enter__(self) -> str:
        while True:
            self.port, self._next_port = self.choose_port(), None
            if port_in_use(self.port):
                if self.requested_port:
                    raise ServerError(f"port {self.port} is already in use")
                self.port = pick_port()
            try:
                self._launch()
                self._wait_ready()
                return self.url
            except PortTakenError as exc:
                if self._teardown():
                    raise ServerError(f"{exc}; not restarting, since {self._survived()}") from None
                if self.requested_port:
                    raise ServerError(f"{exc}; choose another --port, or omit it to use a free one") from None
                if self.restarts:
                    raise
                self.restarts += 1
                self.exit_code = None
            except BaseException:
                self._teardown()
                raise

    def _launch(self) -> None:
        """Start the server. Interrupts wait until it is recorded, so whatever
        exists when one lands is torn down."""
        with _held_signals():
            self._scratch = Path(tempfile.mkdtemp(prefix="mstar-e2e-", dir=_scratch_parent()))
            pythonpath = os.pathsep.join(filter(None, [str(self.source), os.environ.get("PYTHONPATH")]))
            env = {**os.environ, "PYTHONPATH": pythonpath, **self.env}
            cmd = self.command or [
                sys.executable, "-u", "-m", "mstar.api_server.entrypoint", "--config", str(self.config),
                "--host", HOST, "--port", str(self.port),
                "--socket-path-prefix", str(self._scratch / "ipc"), "--upload-dir", str(self._scratch / "upload"),
                "--tensor-comm-protocol", self.tensor_comm_protocol]
            cmd = [arg.replace("{port}", str(self.port)) for arg in cmd]
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "a" if self.restarts else "w")
            parent = os.getpid() if threading.current_thread() is threading.main_thread() else 0
            self._proc = subprocess.Popen([sys.executable, "-c", LAUNCHER, str(parent), *cmd], cwd=self.source,
                                          env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                          start_new_session=True)

    def poll(self) -> int | None:
        """The server's exit code if it has exited on its own, else None."""
        if self._proc is not None and self._proc.poll() is not None and self.exit_code is None:
            self.exit_code = self._proc.returncode
        return self.exit_code

    def _listening(self) -> bool | None:
        """Whether this server's group listens on the port; None without /proc.

        Reads the listeners before our sockets, so a listener seen in the first
        read is still open in the second. It never concludes that another
        process holds the port: snapshots of other processes go stale, and
        the server's own failed bind is the one conclusive sign."""
        listening = listening_sockets(self.port)
        if listening is None:
            return None
        return bool(listening & group_sockets(self._proc.pid))

    def _wait_ready(self):
        start = time.monotonic()
        while True:
            code = self.poll()
            if code is not None:
                log = self.log_path.read_text(errors="replace").lower()
                if code == BIND_FAILED or "address already in use" in log:
                    raise PortTakenError(f"port {self.port} was taken by another process while the server loaded")
                raise ServerError(f"server exited with code {code} during startup; see {self.log_path}")
            if self._listening() is not False:  # ours, or no /proc to tell
                try:
                    healthy = requests.get(self.url + "/health", timeout=1).status_code == 200
                except requests.RequestException:
                    healthy = False
                # Listening both before and after the answer, and still running:
                # only one process can listen on the port, so the answer was ours.
                if healthy and self.poll() is None and self._listening() is not False:
                    self.startup_seconds = time.monotonic() - start
                    return
            if time.monotonic() - start > self.startup_timeout:
                raise ServerError(f"server not healthy after {self.startup_timeout:.0f} s; see {self.log_path}")
            time.sleep(1)

    def __exit__(self, *exc):
        if self._teardown() and exc[0] is None:
            raise ServerError(self._survived())
        return False

    def _survived(self) -> str:
        return f"pid(s) {', '.join(map(str, self.survivors))} of the server's group survived SIGKILL"

    def _teardown(self) -> list[int]:
        """Stop the server group if one was started and release the log and
        scratch dir, whichever exist; return the pids that survived SIGKILL.

        SIGINT/SIGTERM/SIGHUP arriving meanwhile are held, skip straight to
        SIGKILL, and are delivered once the group is gone; any other exception
        still sends SIGKILL before propagating.
        """
        self.survivors = []
        try:
            with _held_signals() as pending:
                try:
                    if self._proc is not None:
                        self._stop_group(pending)
                finally:
                    self._release()
        except BaseException:
            if self._proc is not None:  # interrupted before the signals were held
                _signal_group(self._proc.pid, signal.SIGKILL)
            self._release()
            raise
        return self.survivors

    def _release(self) -> None:
        self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def _stop_group(self, pending: list) -> None:
        pgid = self._proc.pid
        try:
            self._stop(pgid, pending)
        except BaseException:
            _signal_group(pgid, signal.SIGKILL)
            raise
        finally:
            self._proc.poll()
            self.survivors = _survivors(pgid)

    def _stop(self, pgid: int, pending: list) -> None:
        """SIGINT to the API server, its graceful stop: it shuts its conductor
        down, which stops the workers. The group gets SIGKILL if that has not
        finished within ``stop_grace``, ``settle`` after the API server exits,
        or as soon as another interrupt arrives. An exit seen by the one check
        before SIGINT is the server's own, so it is recorded."""
        if self.poll() is None:
            try:
                os.kill(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline, server_gone = time.monotonic() + self.stop_grace, None
        while _group_alive(pgid) and not pending and time.monotonic() < deadline:
            if server_gone is None and self._proc.poll() is not None:
                server_gone = time.monotonic()
            if server_gone is not None and time.monotonic() - server_gone > self.settle:
                break
            time.sleep(0.2)
        if not _group_alive(pgid):
            return
        _signal_group(pgid, signal.SIGKILL)
        deadline = time.monotonic() + self.kill_grace
        while _group_alive(pgid) and time.monotonic() < deadline:
            self._proc.poll()
            time.sleep(0.2)


@contextlib.contextmanager
def _held_signals():
    """Record the signals in HELD_SIGNALS instead of acting on them; yields the
    list they are recorded in, and delivers them on exit. Inherited ignores
    stay (nohup). Python runs handlers only in the main thread, so elsewhere
    nothing is held."""
    pending, handlers = [], {}
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in HELD_SIGNALS:
                current = signal.getsignal(sig)
                if current is not None and current != signal.SIG_IGN:
                    handlers[sig] = current
                    signal.signal(sig, lambda signum, frame: pending.append(signum))
        yield pending
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for sig in dict.fromkeys(pending):
            signal.raise_signal(sig)
