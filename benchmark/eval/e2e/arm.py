"""One model's base and head arms on one GPU, back to back, then the base-vs-head facts.

python -m benchmark.eval.e2e.arm JOB.json      (started by ``compare`` on the GPU's host, local or over SSH)

Each arm's server is ``mstar serve`` launched by ``ManagedServer`` from the arm's own checkout (``PYTHONPATH`` and
working directory, plus that tree's ``rust/`` build), with the eval client of the tree running ``compare``. Server
sessions alternate the arm order (ABBA with two sessions). Layout under ``job["out"]``::

  status.json                     phase, arm, case (polled by compare)
  <arm>/s<k>/startup.json         healthy, startup_s, peak VRAM, which checkout the server imported
  <arm>/s<k>/server.log
  <arm>/s<k>/server-timeout<N>.log  the server of the cases with ``server_timeout_s: N``, started after the others
  <arm>/s<k>/<case>/              mstar-eval/1 results per concurrency, saved outputs, client log, and
                                  stacks.txt when the case hung (ran past the model's case_timeout_s)
  <arm>/s<k>/stress/run<r>.json   ``--stress`` instead of the cases: one record per run (stress.run_session)
  <arm>/s<k>/<case>/quality.json  perceptual scores of the saved outputs (quality.py), after every server stopped
  quality.log
  compare.json                    checks.compare_model
  done.json                       written last: done | skipped | error | interrupted
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from benchmark.eval.e2e import checks, stress
from benchmark.eval.server import ManagedServer, ServerError, is_healthy

CASE_TIMEOUT_S = 3600
QUALITY_TIMEOUT_S = 1800
IDLE_MB = 2000
# Each arm's first server start may compile from cold caches (whichever arm goes first pays the shared compiles,
# and a head that changed kernels compiles those even second), so it gets this many times startup_timeout_s.
COLD_START_FACTOR = 2.0

job: dict = {}
job_path = Path(".")
out = Path(".")
stop_reason = ""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def _publish(name: str, data: dict, where: Path | None = None) -> None:
    """Written whole or not at all (a reader never sees a truncated file), on one line."""
    where = where or out
    tmp = where / f".{name}.tmp"
    tmp.write_text(json.dumps(data))
    os.replace(tmp, where / name)


def status(**kw) -> None:
    _publish("status.json", {"t": now(), **kw})


def done(state: str, **kw) -> None:
    _publish("done.json", {"status": state, "t": now(), **kw})


def job_gpus() -> list[int]:
    """The job's GPUs: every one a multi-GPU model is served on, else the slot's own."""
    return list(job.get("gpus") or ([job["gpu"]] if job.get("gpu") is not None else []))


def gpu_mem_mb() -> list[int]:
    """Used memory of each of the job's GPUs (all GPUs for a whole-machine slot); [-1] when unreadable."""
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if job_gpus():
        cmd += ["-i", ",".join(map(str, job_gpus()))]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return [int(x) for x in r.stdout.split()] or [-1]
    except (OSError, ValueError, subprocess.SubprocessError):
        return [-1]


class VramSampler(threading.Thread):
    """Peak of the job's GPU memory, summed over its GPUs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_ev = threading.Event()

    def run(self):
        while not self.stop_ev.wait(2.0):
            self.peak = max(self.peak, sum(m for m in gpu_mem_mb() if m > 0))


def wait_gpu_idle(limit_s: float = 180) -> int:
    """Waits until every GPU of the job is under IDLE_MB (or ``limit_s``); returns the busiest one's MB."""
    t0, mem = time.time(), max(gpu_mem_mb())
    while mem > IDLE_MB and time.time() - t0 < limit_s:
        time.sleep(5)
        mem = max(gpu_mem_mb())
    return mem


def base_env() -> dict[str, str]:
    # As if the environment were activated: JIT builds (FlashInfer's ninja, Triton) look for tools on PATH.
    env = {"PYTHONUNBUFFERED": "1", "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    if job_gpus():
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job_gpus()))
    return env


HEALTH_PATTERNS = {   # server log lines that mean a worker crashed or a transfer failed
    "segfault": re.compile(r"SIGSEGV|Segmentation fault|core dumped|Fatal Python error"),
    "mooncake read failed": re.compile(r"Mooncake read failed|mooncake.*(read|transfer).*fail", re.I),
    "transfer timeout": re.compile(r"transfer.*timed? ?out|timed out waiting for (tensor|transfer)", re.I),
    "worker died": re.compile(r"\bworker\b.{0,80}\bexited with\b|\bworker\S* (died|crashed)", re.I),
}
SHM = Path("/dev/shm")
SHM_SETTLE_S = 10.0
MIN_AGE_S = 5.0   # a server process younger than this at the case's start may be a short-lived helper


def _group_pids(sid: int, min_age_s: float = 0.0) -> set[int]:
    """Live processes of the server's session (``ManagedServer`` starts it as a session leader); a worker that
    died and was not reaped yet (a zombie) does not count, nor do helpers such as compile workers
    (``stress.helper_process``)."""
    r = subprocess.run(["ps", "-o", "pid=,stat=,etimes=", "-g", str(sid)], capture_output=True, text=True,
                       check=False)
    return {int(f[0]) for f in (ln.split() for ln in r.stdout.splitlines()) if len(f) == 3 and f[0].isdigit()
            and not f[1].startswith("Z") and f[2].isdigit() and int(f[2]) >= min_age_s
            and not stress.helper_process(int(f[0]))}


def _namespaces(server: ManagedServer) -> set[str]:
    """This server's deployment namespace in its tensor file names (``stress.server_namespaces``)."""
    return stress.server_namespaces(server)


def _shm_owner(name: str) -> int | None:
    """The pid in mstar's per-process names (``mstar_arena_<entity>_<pid>_<token>``, ``mstar_kv_<uid>_<pid>_<hash>``);
    None for a tensor file (``mstar_<deployment>_<entity>_<uuid>``), which goes once its request is done."""
    if not name.startswith(("mstar_arena_", "mstar_kv_")):
        return None
    parts = name.split(".", maxsplit=1)[0].rsplit("_", 2)
    return int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else -1


def _alive(pid: int) -> bool:
    """Running, or a zombie no longer is."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _shm_names() -> set[str]:
    try:
        return {p.name for p in SHM.iterdir() if p.name.startswith("mstar_")}
    except OSError:
        return set()


def _new_shm(before: set[str], ours: set[int], namespaces: set[str]) -> set[str]:
    """mstar /dev/shm entries made since ``before`` that this server should have given back: its tensor files, and
    segments of one of its processes that is gone. The host's other servers (other GPU slots) make entries too:
    theirs carry another deployment token or owner pid, and a live process's segments are its working memory."""
    out = set()
    for n in _shm_names() - before:
        pid = _shm_owner(n)
        if pid is None:
            if stress.namespace_file(n, namespaces):
                out.add(n)
        elif pid in ours and not _alive(pid):
            out.add(n)
    return out


def health_before(server: ManagedServer, log_path: Path) -> dict:
    """What a worker-to-worker case is checked against afterwards: the server's processes, mstar's /dev/shm
    entries, and where its log ends."""
    return {"pids": sorted(_group_pids(server.pid, MIN_AGE_S)), "all_pids": sorted(_group_pids(server.pid)),
            "shm": sorted(_shm_names()), "ns": sorted(_namespaces(server)),
            "log_at": log_path.stat().st_size if log_path.exists() else 0}


def health_after(before: dict, server: ManagedServer, log_path: Path) -> list[str]:
    """Findings since ``before``: server processes that exited, /dev/shm entries still there after SHM_SETTLE_S,
    and crash or transfer-failure lines in the server log."""
    found = []
    now = _group_pids(server.pid)
    gone = set(before["pids"]) - now
    if gone:
        found.append(f"{len(gone)} server process(es) exited")
    ours, ns, t0 = set(before["all_pids"]) | now, set(before["ns"]), time.monotonic()
    left = _new_shm(set(before["shm"]), ours, ns)
    while left and time.monotonic() - t0 < SHM_SETTLE_S:
        time.sleep(1)
        left &= _new_shm(set(before["shm"]), ours, ns)   # an entry counts only if it stayed the whole wait
    if left:
        found.append(f"/dev/shm +{len(left)} entries")
    if gone:   # arena segments are named by namespace, not pid: only an exit says one may be orphaned
        orphans = stress.unmapped_arena_segments(str(SHM), ns, _group_pids(server.pid))
        if orphans:
            found.append(f"/dev/shm {len(orphans)} arena segment(s) no live server process maps")
    try:
        with open(log_path, "rb") as f:
            f.seek(before["log_at"])
            new = f.read().decode("utf-8", "replace")
    except OSError:
        new = ""
    found += [f"log: {name}" for name, pat in HEALTH_PATTERNS.items() if pat.search(new)]
    return found


def dump_stacks(pgid: int, path: Path) -> None:
    """Every Python thread's stack in the server's process group (``py-spy dump``), for a case that hung; without
    py-spy, the process list and why there are no stacks. Threads only: a hang at an ``await`` shows an idle event
    loop. Retries with ``sudo -n`` only when the job asks (``compare --stacks-sudo``): on a shared cluster a refused
    ``sudo`` is logged, and often mailed to the admins."""
    r = subprocess.run(["ps", "-o", "pid=,args=", "-g", str(pgid)], capture_output=True, text=True, check=False)
    procs = [ln.strip().split(None, 1) for ln in r.stdout.splitlines() if ln.strip()]
    spy = shutil.which("py-spy", path=base_env()["PATH"])
    lines = [f"# process group {pgid}: {len(procs)} processes; py-spy: {spy or 'not installed (pip install py-spy)'}"]
    if spy and not job.get("stacks_sudo"):
        lines[0] += " (no sudo retry; compare --stacks-sudo adds one where ptrace is restricted)"
    for pid, *args in procs:
        lines.append(f"\n## pid {pid}: {(args or [''])[0][:300]}")
        if not spy:
            continue
        out = ""
        cmds = [[spy, "dump", "--pid", pid]] + ([["sudo", "-n", spy, "dump", "--pid", pid]]
                                                 if job.get("stacks_sudo") else [])
        for cmd in cmds:
            try:
                d = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            except (OSError, subprocess.SubprocessError) as e:
                out = f"{cmd[0]}: {e}"
                continue
            out = d.stdout if d.returncode == 0 else (d.stdout + d.stderr).strip()[-500:]
            if d.returncode == 0:
                break
        lines.append(out)
    path.write_text("\n".join(lines) + "\n")


def server_env(arm: dict) -> dict[str, str]:
    path = [arm["src"]] + ([arm["rust_site"]] if arm.get("rust_site") else [])
    return {**base_env(), **job["model"].get("env", {}), "PYTHONPATH": ":".join(path)}


def client_env() -> dict[str, str]:
    return {**os.environ, **base_env(), "PYTHONPATH": job["evalfw"]}


@contextmanager
def host_lock(name: str):
    """Serializes builds that GPU slots on one host share (a ``rust/`` tree's extension or server binary)."""
    path = Path(job["host_dir"]) / "locks" / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def pip_install(python: str) -> list[str] | None:
    """``pip install`` into ``python``'s environment: its own pip, else ``uv pip`` (a uv venv has no pip)."""
    r = subprocess.run([python, "-m", "pip", "--version"], capture_output=True, text=True, check=False)
    if r.returncode == 0:
        return [python, "-m", "pip", "install"]
    uv = shutil.which("uv", path=f"{Path(python).parent}{os.pathsep}{base_env()['PATH']}")
    return [uv, "pip", "install", "--python", python] if uv else None


def build_rust(arm: dict) -> str | None:
    """``mstar_rust`` for the arm's ``rust/`` tree into a shared site dir; returns the problem, or None."""
    tree = arm.get("rust_tree")
    if not tree:
        return "no rust/ tree"
    site = Path(job["host_dir"]) / "rust" / tree / "site"
    with host_lock(f"rust-{tree}"):
        if not (site / ".complete").exists():
            shutil.rmtree(site, ignore_errors=True)
            pip = pip_install(job["python"])
            if pip is None:
                return "mstar_rust not built: the M* environment has no pip and no uv is on PATH"
            r = subprocess.run([*pip, "-q", "--no-deps", "--target", str(site), f"{arm['src']}/rust"],
                               capture_output=True, text=True, check=False)
            if r.returncode != 0:
                return f"mstar_rust build failed: {(r.stdout + r.stderr).strip()[-300:]}"
            (site / ".complete").touch()
    arm["rust_site"] = str(site)
    return None


def build_rust_server(arm: dict) -> str | None:
    tree = arm.get("rust_tree")
    binary = Path(job["host_dir"]) / "rust" / str(tree) / "mstar-server"
    with host_lock(f"rust-server-{tree}"):
        if not binary.exists():
            if not shutil.which("cargo"):
                return "the Rust frontend needs cargo on the host"
            target = binary.parent / "target-server"
            r = subprocess.run(["cargo", "build", "-q", "--release"], cwd=f"{arm['src']}/rust/server",
                               env={**os.environ, "PYO3_PYTHON": job["python"], "CARGO_TARGET_DIR": str(target)},
                               capture_output=True, text=True, check=False)
            if r.returncode != 0:
                return f"Rust frontend build failed: {r.stderr.strip()[-300:]}"
            shutil.copy(target / "release" / "mstar-server", binary)
    arm["rust_server"] = str(binary)
    return None


RDMA_SYSFS = Path("/sys/class/infiniband")
#: Starts the requirement check's one result line. An import can print to stdout too, before or after it
#: (qwen_tts prints transformers' docstring warnings), so only this line is read.
REQUIRES_RESULT = "mstar-e2e-requires: "


def requirements_problem(arm: dict) -> str:
    """What the arm's environment lacks of the model's ``requires``: a module to import, ``module:function`` to
    call with no arguments (the server's own loader, when it loads a package without importing it), or
    ``rdma_nic``: an RDMA device on the host."""
    mods = [m for m in job["model"].get("requires", []) if m != "rdma_nic"]
    nic = "" if "rdma_nic" not in job["model"].get("requires", []) or (
        RDMA_SYSFS.is_dir() and any(RDMA_SYSFS.iterdir())) else "rdma_nic: no RDMA device"
    if not mods:
        return nic
    code = ("import importlib, sys\nbad = []\nfor m in sys.argv[1:]:\n    mod, _, fn = m.partition(':')\n"
            "    try:\n        x = importlib.import_module(mod)\n        if fn:\n            getattr(x, fn)()\n"
            "    except Exception as e:\n        bad.append(f'{m}: {type(e).__name__}')\n"
            f"print({REQUIRES_RESULT!r} + '; '.join(bad), flush=True)")
    r = subprocess.run([job["python"], "-c", code, *mods], env={**os.environ, **server_env(arm)}, cwd=arm["src"],
                       capture_output=True, text=True, timeout=600, check=False)
    results = [line for line in r.stdout.splitlines() if line.startswith(REQUIRES_RESULT)]
    if results:
        bad = results[-1].removeprefix(REQUIRES_RESULT).strip()
    else:   # the check died before its result line, e.g. an import that crashed the interpreter
        bad = (r.stderr.strip() or r.stdout.strip())[-300:] or f"requirement check exited {r.returncode}"
    return "; ".join(x for x in (bad, nic) if x)


def case_cmds(case: dict, url: str, label: str, d: Path) -> list[list[str]]:
    py, data = job["python"], Path(job["data"])
    if case["driver"] in ("rp", "invariants"):
        return [[py, "-m", "benchmark.eval.e2e.request_path", "--url", url, "--suite", case["kind"],
                 "--inputs", str(data / "rp"), "--out", str(d / "rp.json"), *case["args"]]]
    suite = case["kind"] if case["driver"] == "suite" else "e2e"
    # e2e: one point per call, so warmup counts waves at that point's concurrency. Speech suites: one call for
    # every point, so their judges (Whisper, WavLM) load once, with the largest point's waves as warmup.
    points = [[c] for c in case["conc"]] if suite == "e2e" else [case["conc"]]
    cmds = []
    for conc in points:
        cmd = [py, "-m", "benchmark.eval", "run", suite, "--url", url, "--concurrency", ",".join(map(str, conc)),
               "--repeats", str(case["repeats"]), "--warmup", str(case["warmup"] * max(conc)), "--label", label,
               "--out", str(d), "--save-audio", "last"]
        if case.get("data"):
            cmd += ["--data", str(data / job["datasets"][case["data"]])]
        if case["driver"] == "client":
            cmd += ["--kind", case["kind"], "--media", str(data / "media")]
        if case.get("n"):
            cmd += ["--limit", str(case["n"])]
        cmds.append(cmd + case["args"])
    return cmds


def serve_args(arm: dict) -> list[str]:
    return [a.replace("{rust_server_bin}", arm.get("rust_server") or "") for a in job["model"].get("serve_args", [])]


def timeout_server(arm: dict, d: Path, timeout_s: float, scratch: str) -> ManagedServer:
    """The model's config and transport with a per-request ``--timeout``, which ``mstar serve`` does not take: the
    API entrypoint directly, with ``mstar serve``'s SHM default (the entrypoint's own default is RDMA)."""
    model = job["model"]
    cmd = [job["python"], "-m", "mstar.api_server.entrypoint", "--config", str(Path(arm["src"]) / model["config"]),
           "--host", "127.0.0.1", "--port", "{port}", "--socket-path-prefix", f"{scratch}/sock/",
           "--upload-dir", f"{scratch}/uploads/", "--tensor-comm-protocol", "SHM", *serve_args(arm),
           "--timeout", f"{timeout_s:g}"]
    return ManagedServer(model["serve"], command=cmd, env=server_env(arm),
                         log_path=d / f"server-timeout{timeout_s:g}.log", startup_timeout_s=model["startup_timeout_s"],
                         python=job["python"], cwd=arm["src"])


def run_cases(server: ManagedServer, todo: list[dict], arm_name: str, session: int, cases: dict) -> bool:
    """Each case's client commands against ``server``, results into ``cases``; False once the server hung or died
    (the remaining cases are skipped)."""
    model = job["model"]
    d = out / arm_name / f"s{session}"
    for case in todo:
        cd = d / case["id"]
        cd.mkdir(parents=True, exist_ok=True)
        status(phase="case", arm=arm_name, session=session, case=case["id"])
        before = health_before(server, server.log_path) if model.get("health") else None
        t0, rc, hung = time.monotonic(), 0, False
        cap = float(model.get("case_timeout_s") or CASE_TIMEOUT_S)
        with open(cd / "client.log", "w") as f:
            for cmd in case_cmds(case, server.url, f"{model['name']}-{case['id']}-{arm_name}", cd):
                left = cap - (time.monotonic() - t0)
                try:
                    if left <= 0:
                        raise subprocess.TimeoutExpired(cmd, cap)
                    r = subprocess.run(cmd, env=client_env(), cwd=job["evalfw"], stdout=f,
                                       stderr=subprocess.STDOUT, timeout=left, check=False)
                    rc = r.returncode or rc
                except subprocess.TimeoutExpired:
                    rc, hung = "timeout", True
                    log(f"{arm_name} s{session}: {case['id']} still running after {cap:.0f} s: saving stacks")
                    dump_stacks(server.pid, cd / "stacks.txt")
                    break
        alive = not hung and is_healthy(server.url, timeout=10)
        cases[case["id"]] = {"rc": rc, "elapsed_s": round(time.monotonic() - t0, 1), "server_alive_after": alive}
        if hung:
            cases[case["id"]].update(hung=True, cap_s=cap)
        if before is not None:
            cases[case["id"]]["health"] = health_after(before, server, server.log_path)
        log(f"{arm_name} s{session}: {case['id']} rc={rc} {cases[case['id']]['elapsed_s']} s alive={alive}")
        if hung:
            log(f"{arm_name} s{session}: {case['id']} hung; skipping its remaining cases")
            return False
        if not alive:
            log(f"{arm_name} s{session}: server died during {case['id']}; skipping its remaining cases")
            return False
    return True


def run_arm(arm_name: str, session: int) -> dict:
    model, arm = job["model"], job["arms"][arm_name]
    d = out / arm_name / f"s{session}"
    shutil.rmtree(d, ignore_errors=True)   # a rerun must not compare against a previous attempt's outputs
    d.mkdir(parents=True)
    status(phase="startup", arm=arm_name, session=session)
    info: dict = {"gpu_mem_before_mb": wait_gpu_idle()}
    sampler = VramSampler()
    sampler.start()
    config = str(Path(arm["src"]) / model["config"]) if model.get("config") else None
    timeout = model["startup_timeout_s"] * (COLD_START_FACTOR if session == 0 else 1.0)
    info["startup_timeout_s"] = timeout
    server = ManagedServer(model["serve"], config, extra_args=serve_args(arm), env=server_env(arm),
                           log_path=d / "server.log", startup_timeout_s=timeout,
                           python=job["python"], cwd=arm["src"])
    cases: dict = {}
    current, scratch = server, []
    try:
        server.start()
        (out / "server.pgid").write_text(str(server.pid))   # compare's last-resort kill reaches it
        server.wait_ready()
        info.update(healthy=True, startup_s=round(server.startup_s, 1))
        prov = subprocess.run([job["python"], "-c", "import mstar; print(mstar.__file__)"],
                              env={**os.environ, **server_env(arm)}, cwd=arm["src"], capture_output=True,
                              text=True, timeout=120, check=False)
        imported = os.path.realpath(prov.stdout.strip() or "/nonexistent")
        info["mstar_from_arm_src"] = imported.startswith(os.path.realpath(arm["src"]) + os.sep)
        log(f"{arm_name} s{session}: healthy after {info['startup_s']} s")
        if job.get("stress") and info["mstar_from_arm_src"]:
            status(phase="stress", arm=arm_name, session=session)
            stress.run_session(server.url, server.pid, arm_name, session, d, job["stress"], status, log, _publish,
                               stress.server_namespaces(server))
        mine = model["cases"] if info["mstar_from_arm_src"] and not job.get("stress") else []
        going = run_cases(server, [c for c in mine if not c.get("server_timeout_s")], arm_name, session, cases)
        for to in sorted({c["server_timeout_s"] for c in mine if c.get("server_timeout_s")}) if going else []:
            group = [c for c in mine if c.get("server_timeout_s") == to]
            with signals_deferred():
                current.stop()
            wait_gpu_idle()
            scratch.append(tempfile.mkdtemp(prefix="mstar-eval-to-"))
            current = timeout_server(arm, d, to, scratch[-1])
            status(phase="startup", arm=arm_name, session=session, timeout_s=to)
            try:
                current.start()
                (out / "server.pgid").write_text(str(current.pid))
                current.wait_ready()
            except ServerError as e:
                log(f"{arm_name} s{session}: the --timeout {to:g} server did not start: {e}")
                for c in group:   # base lacking it reports base broken, head lacking it fails (report._row)
                    cases[c["id"]] = {"rc": "no server", "elapsed_s": 0.0, "server_alive_after": False,
                                      "health": [f"--timeout {to:g} server did not start"]}
                continue
            if not run_cases(current, group, arm_name, session, cases):
                break
    except ServerError as e:
        info.update(healthy=False, error=str(e)[-500:])
        log(f"{arm_name} s{session}: did not start: {e}")
        if job.get("stress"):
            stress.lost_runs(stress.session_runs(job["stress"], session), arm_name, session, d, ["error"],
                             "not run: the server did not start", _publish, no_server=True)
    finally:   # also when interrupted; a signal must not cut the process-group sweep short
        with signals_deferred():
            current.stop()
            server.stop()
            (out / "server.pgid").unlink(missing_ok=True)
            for x in scratch:
                shutil.rmtree(x, ignore_errors=True)
            sampler.stop_ev.set()
            info["peak_vram_gb"] = round(sampler.peak / 1024, 1) if sampler.peak > 0 else None
            info["cases"] = cases
            _publish("startup.json", info, d)
    return info


SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


def _on_term(signum, frame) -> None:  # noqa: ARG001
    raise SystemExit(128 + signum)   # unwinds to run_arm's finally, which stops the server


@contextmanager
def signals_deferred():
    pending: list[int] = []
    old = {s: signal.signal(s, lambda n, f: pending.append(n)) for s in SIGNALS}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        if pending:
            raise SystemExit(128 + pending[0])


def watch_lease(lease_s: float) -> None:
    """Stop (servers included) once ``compare`` has not renewed ``out/lease`` for ``lease_s``: it was killed, or
    its host lost the connection and requeued this model elsewhere."""
    global stop_reason
    path = out / "lease"
    while True:
        time.sleep(min(30.0, lease_s / 4))
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            age = float("inf")
        if age > lease_s:
            stop_reason = f"lease not renewed for {lease_s:.0f} s (compare stopped polling)"
            log(stop_reason)
            os.kill(os.getpid(), signal.SIGTERM)
            return


def score_quality() -> None:
    """The judges in their own process on the job's GPU, with no server running: their load and GPU time stay out
    of the timed passes, and their memory is freed before ``compare`` starts the next model on this GPU. A judge
    that fails leaves its scores out (the report says not scored), not the job."""
    status(phase="quality")
    with open(out / "quality.log", "w") as f:
        try:
            r = subprocess.run([job["python"], "-m", "benchmark.eval.e2e.quality", str(job_path)], env=client_env(),
                               cwd=job["evalfw"], stdout=f, stderr=subprocess.STDOUT, timeout=QUALITY_TIMEOUT_S,
                               check=False)
            log(f"quality judges rc={r.returncode}")
        except subprocess.TimeoutExpired:
            log(f"quality judges timed out after {QUALITY_TIMEOUT_S} s")


def _broken(problems: dict[str, str | None]) -> dict:
    """``{"broken": arm}`` when exactly one arm failed to build: the report fails a head that cannot build."""
    bad = [a for a, p in problems.items() if p]
    return {"broken": bad[0]} if len(bad) == 1 else {}


def run() -> None:
    model = job["model"]
    arms = job["arms"]
    log(f"{model['name']}: order {job['order']}, {job['sessions']} session(s)")
    built: dict = {}   # one build per tree: arms that share a tree always share its result
    for arm in arms.values():
        tree = arm.get("rust_tree")
        if tree not in built:
            built[tree] = (build_rust(arm), arm.get("rust_site"))
        arm["rust_site"] = built[tree][1]
    problems = {a: built[arm.get("rust_tree")][0] for a, arm in arms.items()}
    if any(problems.values()) and arms["base"].get("rust_tree") != arms["head"].get("rust_tree"):
        bad = [f"{a}: {p}" for a, p in problems.items() if p]
        return done("error", reason="base and head have different rust/ trees and one did not build: " + "; ".join(bad),
                    **_broken(problems))
    if any("{rust_server_bin}" in a for a in model.get("serve_args", [])):
        if not shutil.which("cargo"):
            return done("skipped", reason="the Rust frontend needs cargo on the host")
        problems = {a: build_rust_server(arm) for a, arm in arms.items()}
        if any(problems.values()):
            bad = [f"{a}: {p}" for a, p in problems.items() if p]
            return done("error", reason="; ".join(bad), **_broken(problems))
    missing = {a: requirements_problem(arm) for a, arm in arms.items()}
    if all(missing.values()):
        return done("skipped", reason=f"environment lacks: {missing['base']}")
    for s in range(job["sessions"]):
        for arm_name in job["order"] if s % 2 == 0 else list(reversed(job["order"])):
            info = run_arm(arm_name, s)
            if info.get("healthy") and not info.get("mstar_from_arm_src"):
                return done("error", reason=f"the {arm_name} server did not import mstar from its own checkout")
    if any(c.get("quality") for c in model["cases"]) and not job.get("stress"):   # stress runs no cases
        score_quality()
    status(phase="compare")
    if not job.get("stress"):
        checks.compare_model(out, model["cases"], job["sessions"])
    done("done", rust={a: p for a, p in problems.items() if p})
    status(phase="done")


def main(argv: list[str] | None = None) -> None:
    global job, job_path, out
    job_path = Path((argv or sys.argv[1:])[0]).resolve()
    job = json.loads(job_path.read_text())
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "done.json").unlink(missing_ok=True)
    (out / "pid").write_text(str(os.getpid()))
    for sig in SIGNALS:
        signal.signal(sig, _on_term)
    if job.get("lease_s"):
        threading.Thread(target=watch_lease, args=(float(job["lease_s"]),), daemon=True).start()
    t0 = time.time()
    try:
        run()
    except SystemExit as e:
        done("interrupted", reason=stop_reason or f"signal {int(e.code or 0) - 128}")
        raise
    except Exception as e:  # noqa: BLE001 - always leave a done.json behind
        done("error", reason=f"{type(e).__name__}: {str(e)[:400]}")
        raise
    log(f"{model_name()} finished in {time.time() - t0:.0f} s")


def model_name() -> str:
    return job.get("model", {}).get("name", "?")


if __name__ == "__main__":
    main()
