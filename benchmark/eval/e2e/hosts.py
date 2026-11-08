"""Where ``compare`` runs: GPU slots on this machine or on SSH hosts, and the queue they pull models from.

A slot is one GPU that serves one model at a time (both arms, back to back). Slots pull the next model from one
shared queue, longest first, so an idle slot always takes whatever is still queued; a flagged model's ABBA rerun
goes to the front. A slot whose host stops answering is retired and its model requeued for the others.

``--hosts user@a,user@b:1`` names SSH targets (``:N`` picks GPU N, default 0; ``:0-7`` GPUs 0 to 7; repeat a host
for more GPUs). A model that needs N GPUs (``gpus`` in cases.yaml) runs on N slots of one host at once. Hosts
need key-based ``ssh`` and ``rsync``, an M* Python environment (``--host-python``), and the Hugging Face Hub (with
a login for gated repos) or a warm cache. Machines that must be started first (cloud VMs) are a launcher's job, not
``compare``'s: implement ``HostProvider`` and pass ``--host-provider module:factory``; ``factory(args)`` returns
the provider, whose ``hosts()`` is read once before the run and whose ``release()`` is called once after it.
"""

from __future__ import annotations

import importlib
import shlex
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-o", "ConnectTimeout=30"]
MAX_ATTEMPTS = 2


class HostLostError(RuntimeError):
    """The host stopped answering (ssh failed), as opposed to a job that failed on it."""


class Interrupted(KeyboardInterrupt):
    """Ctrl-C during ``run_slots``; ``stopped`` says whether every slot stopped its job before ``run_slots`` gave up
    waiting (a job left running stops itself when its lease runs out)."""

    def __init__(self, stopped: bool):
        super().__init__(stopped)
        self.stopped = stopped


class Transport(Protocol):
    name: str

    def sh(self, script: str, timeout: float | None = None) -> subprocess.CompletedProcess: ...
    def push(self, local: Path, remote: str) -> None: ...
    def pull(self, remote: str, local: Path, exclude: tuple[str, ...] = ()) -> None: ...


class LocalTransport:
    """This machine: paths are shared, so pushing and pulling a path onto itself is a no-op."""

    name = "local"

    def sh(self, script: str, timeout: float | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout, check=False)

    def push(self, local: Path, remote: str) -> None:
        if Path(remote).resolve() != local.resolve():
            Path(remote).parent.mkdir(parents=True, exist_ok=True)
            _rsync([f"{local}/" if local.is_dir() else str(local), remote])

    def pull(self, remote: str, local: Path, exclude: tuple[str, ...] = ()) -> None:
        if Path(remote).resolve() != local.resolve():
            local.mkdir(parents=True, exist_ok=True)
            _rsync([*[f"--exclude={e}" for e in exclude], f"{remote}/", f"{local}/"])


class SSHTransport:
    def __init__(self, target: str):
        self.name = target

    def sh(self, script: str, timeout: float | None = None) -> subprocess.CompletedProcess:
        try:
            r = subprocess.run(["ssh", *SSH_OPTS, self.name, f"bash -c {shlex.quote(script)}"], capture_output=True,
                               text=True, timeout=timeout, stdin=subprocess.DEVNULL, check=False)
        except subprocess.TimeoutExpired as e:
            raise HostLostError(f"{self.name}: ssh timed out") from e
        if r.returncode == 255:   # ssh's own failure, not the remote command's
            raise HostLostError(f"{self.name}: {r.stderr.strip()[-300:]}")
        return r

    def push(self, local: Path, remote: str) -> None:
        r = self.sh(f"mkdir -p {shlex.quote(str(Path(remote).parent))}", timeout=120)
        if r.returncode:
            raise RuntimeError(f"{self.name}: cannot create {Path(remote).parent}: {r.stderr.strip()[-200:]}")
        _rsync(["-e", shlex.join(["ssh", *SSH_OPTS]), f"{local}/" if local.is_dir() else str(local),
                f"{self.name}:{remote}"], host=self.name)

    def pull(self, remote: str, local: Path, exclude: tuple[str, ...] = ()) -> None:
        local.mkdir(parents=True, exist_ok=True)
        _rsync(["-e", shlex.join(["ssh", *SSH_OPTS]), *[f"--exclude={e}" for e in exclude], f"{self.name}:{remote}/",
                f"{local}/"], host=self.name)


def _rsync(args: list[str], host: str = "local") -> None:
    """Callers create the destination's parent: ``--mkpath`` needs rsync 3.2.3, and RHEL 8 ships 3.1.3."""
    r = subprocess.run(["rsync", "-a", *args], capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)
    if r.returncode in (12, 30, 35, 255):   # connection-level failures
        raise HostLostError(f"{host}: rsync: {r.stderr.strip()[-300:]}")
    if r.returncode != 0:
        raise RuntimeError(f"rsync {' '.join(args[-2:])}: {r.stderr.strip()[-300:]}")


@dataclass
class Slot:
    transport: Transport
    gpu: int | None
    host_dir: str = ""           # absolute work directory on the host, set by compare's setup
    gpu_name: str = ""
    ready: bool = False
    devices: list[int] = field(default_factory=list)   # every GPU of the job it runs (its own first); set by run_slots

    @property
    def name(self) -> str:
        return f"{self.transport.name}:{self.gpu if self.gpu is not None else 'all'}"


class GpuBook:
    """Which GPUs of each host are free. A model that needs N GPUs runs on N GPU slots of one host at once: the slot
    that takes it borrows N - 1 idle siblings, which take no work of their own until it is done."""

    def __init__(self, slots: list[Slot]):
        self.lock = threading.Lock()
        self.free: dict[str, set[int]] = {}
        self.size: dict[str, int] = {}
        for s in slots:
            if s.gpu is not None:
                self.free.setdefault(s.transport.name, set()).add(s.gpu)
                self.size[s.transport.name] = self.size.get(s.transport.name, 0) + 1

    def largest(self) -> int:
        return max(self.size.values(), default=1)

    def claim(self, slot: Slot, need: int) -> bool | None:
        """Reserve the slot's own GPU and ``need - 1`` free siblings (lowest index first) into ``slot.devices``.
        True: reserved; None: not now (the slot's GPU is lent to a sibling's job, or too few are free yet);
        False: never on this host (it has fewer than ``need`` GPUs)."""
        if slot.gpu is None:   # a slot that is the whole machine
            if need > 1:
                return False
            slot.devices = []
            return True
        with self.lock:
            host = slot.transport.name
            free = self.free.get(host, set())
            if need > self.size.get(host, 0):
                return False
            if slot.gpu not in free or len(free) < need:
                return None
            slot.devices = [slot.gpu] + sorted(free - {slot.gpu})[:need - 1]
            free.difference_update(slot.devices)
            return True

    def give(self, slot: Slot) -> None:
        with self.lock:
            if slot.transport.name in self.free:
                self.free[slot.transport.name].update(slot.devices)
            slot.devices = []

    def retire(self, host: str) -> None:
        with self.lock:
            self.free.pop(host, None)
            self.size.pop(host, None)


def parse_hosts(spec: str) -> list[tuple[str, int]]:
    """``user@a,user@b:1,user@c:0-3`` -> [("user@a", 0), ("user@b", 1), ("user@c", 0), ..., ("user@c", 3)]."""
    out = []
    for term in (t.strip() for t in spec.split(",")):
        if not term:
            continue
        host, sep, gpu = term.rpartition(":")
        lo, dash, hi = gpu.partition("-")
        if sep and gpu.isdigit():
            out.append((host, int(gpu)))
        elif sep and dash and lo.isdigit() and hi.isdigit() and int(lo) <= int(hi):
            out += [(host, g) for g in range(int(lo), int(hi) + 1)]
        else:
            out.append((term, 0))
    if len(set(out)) != len(out):
        raise ValueError(f"--hosts names the same GPU twice: {spec}")
    return out


def local_gpus() -> list[int]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


class HostProvider:
    """Supplies SSH targets; the default is the fixed ``--hosts`` list."""

    def __init__(self, spec: str = ""):
        self.spec = spec

    def hosts(self) -> str:
        """Hosts in ``--hosts`` syntax."""
        return self.spec

    def release(self) -> None:
        """Called once when the run ends, whatever its outcome."""


def load_provider(ref: str, args) -> HostProvider:
    module, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"--host-provider wants module:factory, got {ref!r}")
    return getattr(importlib.import_module(module), attr)(args)


class WorkQueue:
    """Models waiting for a slot: retries and reruns first, then the rest longest first."""

    def __init__(self, names: list[str], cost: dict[str, float]):
        self.cv = threading.Condition()
        self.items = sorted(names, key=lambda n: (-cost.get(n, 0.0), names.index(n)))
        self.running: set[str] = set()
        self.attempts: dict[str, int] = {}
        self.stopped = False

    def take(self, live_slots: Callable[[], int], claim: Callable[[str], bool | None] | None = None) -> str | None:
        """The next model, waiting while other slots run jobs that may queue a rerun; None when all is done.

        ``claim(name)`` reserves what the model needs (its GPUs) under the queue's lock: True takes it, False skips
        it for a later item (it never fits this slot), None waits (it fits once running jobs give GPUs back), so a
        multi-GPU model at the front is not starved by smaller ones behind it."""
        with self.cv:
            while True:
                if self.stopped:
                    return None
                waiting = False
                for name in list(self.items):
                    ok = True if claim is None else claim(name)
                    if ok:
                        self.items.remove(name)
                        self.running.add(name)
                        return name
                    if ok is None:
                        waiting = True
                        break
                if not self.running or live_slots() == 0:
                    return None   # nothing runs that could free GPUs or queue a rerun: what is left never fits here
                self.cv.wait(timeout=5 if not waiting else 1)

    def notify(self) -> None:
        with self.cv:
            self.cv.notify_all()

    def finish(self, name: str, rerun: str | None = None) -> None:
        with self.cv:
            self.running.discard(name)
            if rerun:
                self.items.insert(0, rerun)
            self.cv.notify_all()

    def give_back(self, name: str) -> None:
        """Back to the front untried (the slot that took it may not run it)."""
        with self.cv:
            self.running.discard(name)
            self.items.insert(0, name)
            self.cv.notify_all()

    def requeue(self, name: str) -> bool:
        """Back to the front after a lost host; False once it has had MAX_ATTEMPTS."""
        with self.cv:
            self.running.discard(name)
            self.attempts[name] = self.attempts.get(name, 0) + 1
            ok = self.attempts[name] < MAX_ATTEMPTS
            if ok:
                self.items.insert(0, name)
            self.cv.notify_all()
            return ok

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.items.clear()
            self.cv.notify_all()

    def pending(self) -> list[str]:
        with self.cv:
            return list(self.items)


def run_slots(slots: list[Slot], queue: WorkQueue, run_one: Callable[[Slot, str], str | None],
              on_lost: Callable[[Slot, str, Exception], None] = lambda s, n, e: None,
              on_error: Callable[[Slot, str, Exception], None] = lambda s, n, e: None,
              on_given_up: Callable[[str], None] = lambda n: None,
              stop: threading.Event | None = None,
              need: Callable[[str], int] = lambda n: 1) -> dict[str, list[str]]:
    """One thread per slot pulling from ``queue``. ``run_one(slot, model)`` returns the name of a rerun to queue
    (or None) and raises HostLostError when the slot's host is gone; any other exception is that model's error and the
    slot carries on. ``need(model)`` is how many GPUs of one host the model takes (``slot.devices`` during its job).
    Returns {slot name: models it finished}."""
    live = {s.name for s in slots}
    lock = threading.Lock()
    finished: dict[str, list[str]] = {s.name: [] for s in slots}
    book = GpuBook(slots)

    def n_live() -> int:
        with lock:
            return len(live)

    all_done = threading.Event()
    running = [len(slots)]

    def worker(slot: Slot) -> None:
        try:
            work(slot)
        finally:
            with lock:
                running[0] -= 1
                if running[0] == 0:
                    all_done.set()

    dead_hosts: set[str] = set()

    def work(slot: Slot) -> None:
        while True:
            name = queue.take(n_live, lambda n: book.claim(slot, need(n)))
            if name is None:
                break
            if slot.transport.name in dead_hosts:   # another GPU of this host lost it: leave the model to others
                book.give(slot)
                queue.give_back(name)
                return
            try:
                rerun = run_one(slot, name)
            except HostLostError as e:
                with lock:   # every GPU slot of that host retires, so a retry lands on another host
                    dead_hosts.add(slot.transport.name)
                    live.difference_update(s.name for s in slots if s.transport.name == slot.transport.name)
                book.retire(slot.transport.name)
                on_lost(slot, name, e)
                if not queue.requeue(name):
                    on_given_up(name)
                return
            except Exception as e:  # noqa: BLE001 - one model's failure must not stop the slot
                book.give(slot)
                queue.finish(name)
                on_error(slot, name, e)
                continue
            book.give(slot)   # before finish(), whose notify wakes slots waiting for these GPUs
            finished[slot.name].append(name)
            queue.finish(name, rerun)

    if not slots:
        all_done.set()
    for s in slots:
        threading.Thread(target=worker, args=(s,), name=s.name, daemon=True).start()
    # Waiting on an Event, not Thread.join: a signal that interrupts join() can leave the thread marked as
    # stopped while it still runs, and every later join() then returns at once (CPython bpo-45274 handling).
    try:
        while not all_done.wait(timeout=1):
            pass
    except BaseException as e:   # Ctrl-C, or SIGTERM / SIGHUP turned into SystemExit by the CLI
        queue.stop()
        if stop is not None:
            stop.set()   # run_one sees it, stops its job on the host and returns
        try:
            stopped = all_done.wait(timeout=300)
        except KeyboardInterrupt:   # a second Ctrl-C: stop waiting
            stopped = False
        if isinstance(e, KeyboardInterrupt):
            raise Interrupted(stopped) from None
        raise
    for name in queue.pending():   # every slot was lost before these could run
        on_given_up(name)
    return finished
