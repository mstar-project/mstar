"""``python -m benchmark.eval compare``: serve base and head on the same GPUs, judge every case, report.

Both trees are exported from this checkout; each model runs on one GPU slot (``hosts``) with both arms back to
back, arm order alternating from model to model. A perf row flagged with one server session per arm is rerun with
two (ABBA) before it is reported. The client side (suites, checks, this module) always comes from the checkout
running ``compare``, so ``main`` can be measured before it has the eval framework.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from benchmark.eval.e2e import calibrate, hosts, media, report, stress, trees
from benchmark.eval.e2e import manifest as mf
from benchmark.eval.e2e.report import RERUN_TAG
from mstar.utils.cache_home import user_cache_dir

TOOL_ROOT = Path(__file__).resolve().parents[3]
POLL_S = 15
LEASE_S = 600           # a job whose lease compare has not renewed this long stops itself (arm.watch_lease)
GPU_TYPES = ("gb200", "b200", "h200", "h100", "a100", "l40s", "l40", "l4", "a10g")
FETCH_EXCLUDE = ("audio/",)   # saved outputs and audio stay on the host; compare.json already holds their facts


def default_work_dir() -> Path:
    """``$XDG_CACHE_HOME/mstar-eval``, else ``~/.cache/mstar-eval``."""
    return user_cache_dir() / "mstar-eval"


def gpu_type(name: str) -> str:
    """``NVIDIA H200`` -> ``h200``; whole model names only, so an L40 is not an L4."""
    n = name.lower()
    return next((t for t in GPU_TYPES if re.search(rf"(?<![a-z0-9]){t}(?![a-z0-9])", n)),
                re.sub(r"[^a-z0-9]+", "-", n).strip("-") or "unknown")


def add_args(sub) -> None:
    p = sub.add_parser("compare", allow_abbrev=False, help="base vs head on the same GPUs: correctness and perf")
    p.add_argument("--base", default="main", help="branch, PR number, commit or local ref such as HEAD (default main)")
    p.add_argument("--head", help="branch, PR number (548 or #548), commit or local ref such as HEAD")
    p.add_argument("--models", default="all", help="all, names, tag:<tag>, -name / -tag:<tag>; comma-separated")
    p.add_argument("--rp", action="store_true", help="also run the request-path and served-invariant cases")
    p.add_argument("--sessions", type=int, default=1, help="server sessions per arm, order alternated (2 = ABBA)")
    p.add_argument("--repeats", type=int, default=None, help="measured passes per point (default: cases.yaml)")
    p.add_argument("--no-rerun", action="store_true", help="report a flagged perf row without the 2-session rerun")
    p.add_argument("--gpus", default=None, help="local GPU indices to use (default: all)")
    p.add_argument("--stacks-sudo", action="store_true",
                   help="retry a hung case's py-spy dump with sudo -n where ptrace is restricted (off by default: "
                        "a refused sudo is logged, and often mailed, on shared clusters)")
    p.add_argument("--hosts", default=None,
                   help="SSH targets instead of local GPUs: user@a,user@b:1 (host:gpu), user@c:0-7 (GPUs 0 to 7)")
    p.add_argument("--host-provider", default=None, metavar="MODULE:FACTORY",
                   help="a launcher that supplies --hosts (see benchmark/eval/e2e/hosts.py)")
    p.add_argument("--python", default=sys.executable, help="Python of the M* environment on this machine")
    p.add_argument("--host-python", default="python3", help="Python of the M* environment on the SSH hosts")
    p.add_argument("--host-dir", default="mstar-eval", help="work directory on the SSH hosts (relative to home)")
    p.add_argument("--remote", default=None, help="git remote for PR heads (default: the mstar-project/mstar one)")
    p.add_argument("--work-dir", default=None,
                   help="exported trees, rust/ builds, local jobs' outputs and the inputs on this machine "
                        "(default $XDG_CACHE_HOME/mstar-eval, else ~/.cache/mstar-eval)")
    p.add_argument("--data-dir", default=None, help="inputs, built on first use (default <work dir>/e2e-data)")
    p.add_argument("--cases", default=str(mf.DEFAULT), help="the manifest")
    p.add_argument("--calibration", default=str(calibrate.DEFAULT), help="A/A noise floors and swings")
    p.add_argument("--out", default=None, help="run directory (default eval-results/compare/<time>-<head>)")
    p.add_argument("--chart-url", default=None, help="image URL for the markdown (default: the local chart)")
    p.add_argument("--render", default=None, metavar="RUN_DIR", help="only re-render RUN_DIR's report")
    p.add_argument("--list", action="store_true", help="list models, tags and cases")
    p.add_argument("--dry-run", action="store_true", help="resolve refs and slots, print the plan, run nothing")
    s = p.add_argument_group("stress", "--stress replaces the cases with a pinned-id workload of cancels, drops, "
                             "retries, reuse and overlaps, judged on failed runs (benchmark/eval/e2e/stress.py)")
    s.add_argument("--stress", action="store_true", help="run the stress workload on the models that define one")
    s.add_argument("--stress-runs", type=int, default=stress.DEFAULTS["runs"], help="runs per arm (default 20)")
    s.add_argument("--stress-seconds", type=float, default=stress.DEFAULTS["seconds"],
                   help="seconds per run (default 60)")
    s.add_argument("--stress-clients", type=int, default=stress.DEFAULTS["clients"],
                   help="concurrent clients, one pinned request_id each (default 32)")
    s.add_argument("--stress-patterns", default=",".join(stress.PATTERNS),
                   help=f"comma-separated subset of {','.join(stress.PATTERNS)}")
    s.add_argument("--stress-hang-s", type=float, default=stress.DEFAULTS["hang_s"],
                   help="no progress this long is a hang (default 30)")
    c = sub.add_parser("calibrate", allow_abbrev=False, help="fold A/A compare runs into the noise floors")
    c.add_argument("runs", nargs="+", help="run directories of compare --base X --head X")
    c.add_argument("--cases", default=str(mf.DEFAULT))
    c.add_argument("--calibration", default=str(calibrate.DEFAULT))


def work_dir(args) -> Path:
    return Path(args.work_dir).expanduser() if args.work_dir else default_work_dir()


def data_dir(args) -> Path:
    return Path(args.data_dir).expanduser() if args.data_dir else work_dir(args) / "e2e-data"


def run_models(man: mf.Manifest, run: dict) -> list[mf.Model]:
    return man.select(",".join(run["models"]), include_optional=run.get("include_optional", False))


def render(run_dir: Path, man: mf.Manifest, calibration: str, chart_url: str | None = None) -> tuple[Path, int]:
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("stress"):
        md, verdicts = stress.write(run_dir, run["models"], chart_url)
        return md, 1 if any(v.flagged for v in verdicts) else 2 if any(v.unfinished for v in verdicts) else 0
    md, rows = report.write(run_dir, run_models(man, run), chart_url, calibrate.load(calibration))
    bare = sorted({r.model for r in rows if r.no_baseline})
    if bare:
        print(f"no A/A noise floor on {run.get('gpu')} for: {', '.join(bare)}. Calibrate with python -m "
              f"benchmark.eval compare --base main --head main --models {','.join(bare)}, then python -m "
              "benchmark.eval calibrate <that run dir>", file=sys.stderr)
    return md, 1 if any(r.flagged for r in rows) else 2 if any(r.unfinished for r in rows) else 0


def cmd_calibrate(args) -> int:
    man = mf.load(args.cases)
    for d in map(Path, args.runs):
        run = json.loads((d / "run.json").read_text())
        excluded: dict[str, int] = {}
        n = calibrate.add_run(d, run_models(man, run), args.calibration, excluded)
        left_out = ", ".join(f"{name} {k}" for name, k in sorted(excluded.items())) or "none"
        print(f"{d}: {n} entries added to {args.calibration}; failed or errored rows left out of the floors: "
              f"{left_out}")
    return 0


def public_command(args) -> str:
    """The command for the report: no host names, paths or provider modules."""
    parts = ["python -m benchmark.eval compare", f"--base {shlex.quote(args.base)}", f"--head {shlex.quote(args.head)}"]
    if args.models != "all":
        parts.append(f"--models {shlex.quote(args.models)}")
    for flag in ("rp", "no_rerun"):
        if getattr(args, flag):
            parts.append("--" + flag.replace("_", "-"))
    if args.sessions != 1:
        parts.append(f"--sessions {args.sessions}")
    if args.repeats:
        parts.append(f"--repeats {args.repeats}")
    if getattr(args, "stress", False):
        parts += ["--stress", f"--stress-runs {args.stress_runs}", f"--stress-seconds {args.stress_seconds:g}",
                  f"--stress-clients {args.stress_clients}"]
        if args.stress_patterns != ",".join(stress.PATTERNS):
            parts.append(f"--stress-patterns {shlex.quote(args.stress_patterns)}")
        if args.stress_hang_s != stress.DEFAULTS["hang_s"]:
            parts.append(f"--stress-hang-s {args.stress_hang_s:g}")
    if args.hosts or args.host_provider:
        parts.append("--hosts <ssh hosts>")
    return " ".join(parts)


def stress_config(args) -> dict | None:
    """The job's ``stress`` block (model kwargs are added per model), or None without ``--stress``."""
    if not getattr(args, "stress", False):
        return None
    patterns = [p.strip() for p in args.stress_patterns.split(",") if p.strip()]
    bad = sorted(set(patterns) - set(stress.PATTERNS))
    if bad or not patterns:
        raise SystemExit(f"--stress-patterns: unknown {', '.join(bad) or 'empty list'} "
                         f"(known: {', '.join(stress.PATTERNS)})")
    if args.stress_runs < 1 or args.stress_clients < 1 or args.stress_seconds <= 0 or args.stress_hang_s <= 0:
        raise SystemExit("--stress-runs and --stress-clients must be at least 1, the seconds positive")
    return {**stress.DEFAULTS, "runs": args.stress_runs, "seconds": args.stress_seconds,
            "clients": args.stress_clients, "hang_s": args.stress_hang_s, "patterns": patterns}


def tree_id(root: Path) -> str:
    """Content key of an inputs directory (paths and bytes): runs with different inputs never share a copy on a
    host, and the same inputs are copied once."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(root)}\0".encode())
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:12]


def evalfw_id(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted([root / "benchmark" / "__init__.py", *(root / "benchmark" / "eval").rglob("*")]):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(root)).encode() + f.read_bytes())
    return h.hexdigest()[:12]


def parse_poll(text: str) -> tuple[bool, str, dict | None]:
    """(job alive, status.json, done.json when it holds a whole JSON object) from the poll's ``A:``, ``S:`` and
    ``D:`` lines (both files are one line). An unreadable done.json counts as none, so a job that died writing it
    still ends up DEAD."""
    fields = {ln[:2]: ln[2:].strip() for ln in text.splitlines() if ln[:2] in ("A:", "S:", "D:")}
    try:
        done = json.loads(fields["D:"]) if fields.get("D:") else None
    except ValueError:
        done = None
    return fields.get("A:") == "1", fields.get("S:", ""), done if isinstance(done, dict) else None


def is_local(slot: hosts.Slot) -> bool:
    return isinstance(slot.transport, hosts.LocalTransport)


def slot_python(args, slot: hosts.Slot) -> str:
    return args.python if is_local(slot) else args.host_python


def probe(args, slot: hosts.Slot) -> str:
    """The slot's GPU name, after checking its Python can run the client."""
    sel = f"-i {slot.gpu}" if slot.gpu is not None else ""
    py = slot_python(args, slot)
    r = slot.transport.sh(f"nvidia-smi --query-gpu=name --format=csv,noheader {sel} | head -1; "
                          f"{shlex.quote(py)} -c 'import aiohttp, numpy, torch' && echo PY_OK", timeout=120)
    lines = r.stdout.strip().splitlines()
    if "PY_OK" not in lines:
        raise RuntimeError(f"{py} cannot import aiohttp, numpy and torch ({r.stderr.strip()[-200:]})")
    slot.gpu_name = lines[0] if lines[0] != "PY_OK" else ""
    return slot.gpu_name


class Runner:
    """State of one compare run, shared by the slot threads."""

    def __init__(self, args, man: mf.Manifest, models: list[mf.Model], base: trees.Ref, head: trees.Ref,
                 run_dir: Path, slots: list[hosts.Slot]):
        self.args, self.man, self.run_dir, self.slots = args, man, run_dir, slots
        self.models = {m.name: m for m in models}
        self.order = [m.name for m in models]
        self.base, self.head = base, head
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.state: dict = {"models": {n: {"status": "queued"} for n in self.order}, "slots": {}}
        self.rerun_cases: dict[str, list[str]] = {}
        self.host_locks: dict[str, threading.Lock] = {}
        self.attempt: dict[str, int] = {}
        # Remote namespace: unique per invocation, whatever --out is called (two runs may share a host).
        self.run_key = f"{run_dir.name}-{uuid.uuid4().hex[:8]}"
        self.cache = work_dir(args) / "compare"
        self.data = data_dir(args)
        self.evalfw = evalfw_id(TOOL_ROOT)
        self.data_id = tree_id(self.data)

    def redact(self, text: str, slot: hosts.Slot | None = None) -> str:
        """Host names and the slot's work directory out of anything that can reach the report."""
        if slot is not None and slot.host_dir:
            text = text.replace(slot.host_dir, "<work dir>")
        names = {n for s in self.slots for n in (s.transport.name, s.transport.name.rpartition("@")[2])}
        for name in sorted(names - {"local", ""}, key=len, reverse=True):
            text = text.replace(name, "<host>")
        return text

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        with self.lock, open(self.run_dir / "events.log", "a") as f:
            f.write(line + "\n")

    def set(self, name: str, **kw) -> None:
        with self.lock:
            self.state["models"].setdefault(name, {}).update(kw)
            (self.run_dir / "state.json").write_text(json.dumps(self.state, indent=1))

    # --- slots -------------------------------------------------------------------------------------------------
    def local(self, slot: hosts.Slot) -> bool:
        return is_local(slot)

    def python(self, slot: hosts.Slot) -> str:
        return slot_python(self.args, slot)

    def prepare(self, slot: hosts.Slot) -> None:
        """Work directory, client tree, inputs and both source trees on the slot's host (once per host)."""
        t = slot.transport
        with self.lock:
            host_lock = self.host_locks.setdefault(t.name, threading.Lock())
        with host_lock:
            if slot.ready:
                return
            if self.local(slot):
                slot.host_dir = str(self.cache)
            else:
                same = next((s for s in self.slots if s.transport.name == t.name and s.ready), None)
                if same:
                    slot.host_dir, slot.ready = same.host_dir, True
                    return
                d = self.args.host_dir
                r = t.sh(f"mkdir -p {shlex.quote(d)} && cd {shlex.quote(d)} && pwd", timeout=120)
                if r.returncode:
                    raise RuntimeError(f"{t.name}: cannot create {d}: {r.stderr.strip()[-200:]}")
                slot.host_dir = r.stdout.strip().splitlines()[-1]
                self.log(f"{t.name}: copying the client tree, inputs and source trees")
                for rel in ("benchmark/__init__.py", "benchmark/eval"):
                    t.push(TOOL_ROOT / rel, f"{slot.host_dir}/evalfw/{self.evalfw}/{rel}")
                t.push(self.data, f"{slot.host_dir}/data/{self.data_id}")
                for ref in (self.base, self.head):
                    t.push(self.cache / "src" / ref.sha, f"{slot.host_dir}/src/{ref.sha}")
            slot.ready = True

    def paths(self, slot: hosts.Slot) -> dict[str, str]:
        if self.local(slot):
            return {"evalfw": str(TOOL_ROOT), "data": str(self.data), "src": str(self.cache / "src")}
        return {"evalfw": f"{slot.host_dir}/evalfw/{self.evalfw}", "data": f"{slot.host_dir}/data/{self.data_id}",
                "src": f"{slot.host_dir}/src"}

    # --- one model ---------------------------------------------------------------------------------------------
    def model_for(self, name: str) -> mf.Model:
        m = self.models[name.split(RERUN_TAG, maxsplit=1)[0]]
        if name in self.rerun_cases:
            m = copy.copy(m)
            m.cases = [c for c in m.cases if c.id in self.rerun_cases[name]]
        if self.args.repeats:
            m = copy.copy(m)
            m.cases = [dataclasses.replace(c, repeats=self.args.repeats) for c in m.cases]
        return m

    def job(self, slot: hosts.Slot, name: str, out: str) -> dict:
        m, p = self.model_for(name), self.paths(slot)
        attempts = self.state["models"].get(name, {}).get("attempts", 0)
        position = self.order.index(name.split(RERUN_TAG, maxsplit=1)[0]) + attempts
        arm = lambda ref: {"sha": ref.sha, "label": ref.label, "src": f"{p['src']}/{ref.sha}",  # noqa: E731
                           "rust_tree": ref.rust_tree}
        gpus = slot.devices or ([slot.gpu] if slot.gpu is not None else [])
        st = stress_config(self.args)
        if st is not None:
            st = {**st, **{k: v for k, v in m.stress.items() if k != "model_kwargs"},
                  "model_kwargs": m.stress.get("model_kwargs", {})}
        sessions = (2 if name in self.rerun_cases else self.args.sessions) if st is None \
            else -(-st["runs"] // stress.PER_SERVER)
        return {"run_id": self.run_key, "model": m.to_dict(), "out": out, "gpu": slot.gpu, "gpus": gpus,
                "stress": st, "sessions": sessions,
                "order": ["base", "head"] if position % 2 == 0 else ["head", "base"],
                "python": self.python(slot), "evalfw": p["evalfw"], "data": p["data"], "host_dir": slot.host_dir,
                "datasets": self.man.datasets, "arms": {"base": arm(self.base), "head": arm(self.head)},
                "lease_s": LEASE_S, "stacks_sudo": bool(getattr(self.args, "stacks_sudo", False))}

    def run_one(self, slot: hosts.Slot, name: str) -> str | None:
        self.prepare(slot)
        t, local = slot.transport, self.local(slot)
        dst = self.run_dir / "results" / name
        with self.lock:   # every attempt its own directory and lease: an abandoned one cannot touch this one
            self.attempt[name] = self.attempt.get(name, 0) + 1
            n = self.attempt[name]
        out = (str(self.run_dir / "attempts" / f"{name}.{n}") if local
               else f"{slot.host_dir}/runs/{self.run_key}/{name}.{n}")
        job = self.job(slot, name, out)
        shutil.rmtree(dst, ignore_errors=True)   # nothing from a lost attempt may survive into this one
        jf = self.run_dir / "jobs" / f"{name}.json"
        jf.parent.mkdir(exist_ok=True)
        jf.write_text(json.dumps(job, indent=1))
        q = shlex.quote
        t.sh(f"rm -rf {q(out)} && mkdir -p {q(out)} && touch {q(out)}/lease", timeout=120)
        t.push(jf, f"{out}/job.json")
        # `cd x || exit` then a plain background command: `cd x && cmd &` would background a subshell that holds
        # the caller's stdout (so ssh waits for the job) and whose pid is not the job's.
        r = t.sh(f"cd {q(job['evalfw'])} || exit 1; nohup setsid env PYTHONPATH={q(job['evalfw'])} "
                 f"{q(job['python'])} -m benchmark.eval.e2e.arm {q(out)}/job.json > {q(out)}/arm.log 2>&1 "
                 "< /dev/null & echo $!", timeout=120)
        pid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if not pid.isdigit():
            raise RuntimeError(f"{slot.name}: could not start the job: {r.stderr.strip()[-300:]}")
        t0 = time.time()
        self.set(name, status="running", slot=slot.name, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        on = f" on GPUs {','.join(map(str, job['gpus']))}" if len(job["gpus"]) > 1 else ""
        self.log(f"{slot.name}: {name} started{on} (order {' then '.join(job['order'])})")
        # SIGTERM lets the job stop its server (up to 50 s); after two minutes, its session and the server's go.
        # A zombie (a container whose init does not reap) answers kill -0, so liveness reads the process state.
        is_alive = f'case "$(ps -o stat= -p {pid} 2>/dev/null)" in ""|Z*) false;; *) true;; esac'
        kill = (f"kill -TERM {pid} 2>/dev/null; for i in $(seq 120); do {is_alive} || break; "
                f"sleep 1; done; kill -KILL -{pid} 2>/dev/null; g=$(cat {q(out)}/server.pgid 2>/dev/null) && "
                "kill -KILL -- -$g 2>/dev/null; true")
        done, dead = None, 0
        while done is None:
            if self.stop.wait(POLL_S):
                t.sh(kill, timeout=180)
                done = {"status": "interrupted", "reason": "compare was stopped"}
                break
            try:
                r = t.sh(f"touch {q(out)}/lease; {is_alive} && echo A:1 || echo A:0; "
                         f"echo \"S:$(cat {q(out)}/status.json 2>/dev/null)\"; "
                         f"echo \"D:$(cat {q(out)}/done.json 2>/dev/null)\"", timeout=60)
            except hosts.HostLostError:
                try:   # best effort; otherwise the job stops itself when its lease runs out
                    t.sh(kill, timeout=180)
                except Exception:  # noqa: BLE001
                    pass
                raise
            alive, progress, done = parse_poll(r.stdout)
            if done is None and not alive:
                dead += 1
                if dead >= 2:   # killed outright (OOM, SIGKILL): its server's session may still hold the GPU
                    t.sh(kill, timeout=180)
                    done = {"status": "error", "reason": "the job exited without a result (see arm.log)"}
            elif done is None:
                dead = 0
                self.set(name, progress=progress[:200])
        part = dst.with_name(dst.name + ".partial")   # published only once the copy is complete
        shutil.rmtree(part, ignore_errors=True)
        t.pull(out, part, FETCH_EXCLUDE)
        # The report reads done.json: write compare's own verdict when the job left none (killed, stopped), and
        # keep host names and host paths out of the reason.
        done = {**done, "reason": self.redact(done.get("reason", ""), slot)}
        (part / "done.json").write_text(json.dumps(done))
        shutil.rmtree(dst, ignore_errors=True)
        part.rename(dst)
        elapsed = time.time() - t0
        self.set(name, status=done.get("status"), reason=done.get("reason", ""),
                 elapsed_s=round(elapsed, 1), gpu_s=round(elapsed * max(1, len(job["gpus"])), 1))
        self.log(f"{slot.name}: {name} {done.get('status')} in {elapsed / 60:.1f} min"
                 + (f" ({done['reason']})" if done.get("reason") else ""))
        return self.maybe_rerun(name) if done.get("status") == "done" else None

    def maybe_rerun(self, name: str) -> str | None:
        """A perf regression with one server session per arm may be the process's speed level (some servers settle
        at one of two speeds per process): queue those cases with two sessions, ABBA, before reporting them."""
        if (RERUN_TAG in name or self.args.sessions > 1 or self.args.no_rerun or self.stop.is_set()
                or getattr(self.args, "stress", False)):
            return None
        rows = report.build_rows(self.run_dir, [self.model_for(name)], calibrate.load(self.args.calibration))
        flagged = report.flagged_perf_cases(rows)
        if not flagged:
            return None
        rname = name + RERUN_TAG
        with self.lock:
            self.rerun_cases[rname] = flagged
            self.state["models"][rname] = {"status": "queued", "cases": flagged}
        self.log(f"{name}: flagged ({', '.join(flagged)}); rerunning those cases with 2 sessions (ABBA)")
        return rname


def make_provider(args) -> hosts.HostProvider | None:
    if args.host_provider:
        return hosts.load_provider(args.host_provider, args)
    return hosts.HostProvider(args.hosts) if args.hosts else None


def make_slots(args, provider: hosts.HostProvider | None) -> list[hosts.Slot]:
    if provider is not None:
        targets = hosts.parse_hosts(provider.hosts())
        if not targets:
            raise SystemExit("no hosts to run on")
        return [hosts.Slot(hosts.SSHTransport(h), g) for h, g in targets]
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else hosts.local_gpus()
    if not gpus:
        raise SystemExit("no local GPU found (nvidia-smi); pass --hosts to run on other machines")
    local = hosts.LocalTransport()
    return [hosts.Slot(local, g) for g in gpus]


def cmd_compare(args) -> int:
    man = mf.load(args.cases)
    if args.list:
        for m in man.models.values():
            cases = ", ".join(c.id + ("*" if c.optional else "") + (f" [{'+'.join(c.quality)}]" if c.quality else "")
                              for c in m.cases)
            gpus = f" ({m.gpus} GPUs)" if m.gpus > 1 else ""
            print(f"{m.name + ('+' if m.optional else ''):26} {m.family:7} tags: {','.join(m.tags):50} cases: {cases}"
                  f"{gpus}")
        print("* request path or served invariants (--rp); + optional model: name it or use its own tag")
        return 0
    if args.render:
        md, code = render(Path(args.render), man, args.calibration, args.chart_url)
        print(md)
        return code
    if not args.head:
        raise SystemExit("compare needs --head (or --render / --list)")
    if args.out and (Path(args.out) / "run.json").exists():
        raise SystemExit(f"{args.out} already holds a run; pass a new --out (reruns and results would mix)")
    models = man.select(args.models, include_optional=args.rp)
    st = stress_config(args)
    if st is not None:
        named = {t.strip() for t in args.models.split(",")} & set(man.models)
        without = sorted(m.name for m in models if not m.stress and m.name in named)
        if without:
            raise SystemExit(f"no stress workload for {', '.join(without)} (models with one: "
                             f"{', '.join(n for n, m in man.models.items() if m.stress)})")
        models = [m for m in models if m.stress]
    if not models:
        raise SystemExit("no models selected")
    base = trees.resolve(TOOL_ROOT, args.base, args.remote)
    head = trees.resolve(TOOL_ROOT, args.head, args.remote)
    provider = make_provider(args)
    try:   # from here on the provider may hold machines: release them whatever happens
        return _run(args, man, models, base, head, make_slots(args, provider))
    finally:
        if provider is not None:
            provider.release()


def _run(args, man, models, base, head, slots) -> int:
    print(f"base {base.label} ({base.sha[:9]}), head {head.label} ({head.sha[:9]}); {len(models)} models on "
          f"{len(slots)} GPU slot(s): {', '.join(m.name for m in models)}", file=sys.stderr)
    usable = []
    for s in slots:
        try:
            print(f"  {s.name}: {probe(args, s)}", file=sys.stderr)
            usable.append(s)
        except (RuntimeError, hosts.HostLostError) as e:
            print(f"  {s.name}: not usable: {e}", file=sys.stderr)
    types = {gpu_type(s.gpu_name) for s in usable}
    if not usable:
        raise SystemExit("no usable GPU slot")
    if len(types) > 1:
        raise SystemExit(f"slots have different GPU types ({', '.join(sorted(types))}); one table never mixes them")
    if args.dry_run:
        largest = hosts.GpuBook(usable).largest()
        for m in (m for m in models if m.gpus > 1):
            fit = "fits" if m.gpus <= largest else f"does not fit (largest host: {largest} GPUs)"
            print(f"  {m.name}: {m.gpus} GPUs on one host, {fit}", file=sys.stderr)
        if getattr(args, "stress", False):
            print(stress.plan_text(stress_config(args)), file=sys.stderr)
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = re.sub(r"[^A-Za-z0-9.-]+", "_", head.label)
    run_dir = Path(args.out or f"eval-results/compare/{stamp}-{label}-{uuid.uuid4().hex[:6]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:   # reserve the directory: two invocations can never share one
        (run_dir / "run.json").open("x").close()
    except FileExistsError:
        raise SystemExit(f"{run_dir} already holds a run; pass a new --out") from None
    data = data_dir(args)
    if not (data / "e2e-data.json").exists():
        print(f"building the inputs into {data} (once)", file=sys.stderr)
        media.build(data)
    runner = Runner(args, man, models, base, head, run_dir, usable)
    for ref in (base, head):
        trees.export(TOOL_ROOT, ref.sha, runner.cache / "src" / ref.sha)
    th = man.defaults.get("thresholds", {})
    run = {"id": runner.run_key, "base": base.__dict__, "head": head.__dict__, "gpu": types.pop(),
           "gpu_name": usable[0].gpu_name, "models": [m.name for m in models], "include_optional": args.rp,
           "sessions": args.sessions, "repeats": args.repeats or man.defaults.get("repeats", 3),
           "perf_pct": th.get("perf_pct", 5), "noise_sigma": th.get("noise_sigma", 2),
           "rerun_flagged": not args.no_rerun, "stress": stress_config(args),
           "gpus_per_model": {m.name: m.gpus for m in models if m.gpus > 1},
           "slots": [s.name for s in usable], "command": public_command(args),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    largest = hosts.GpuBook(usable).largest()
    too_big = {m.name: m.gpus for m in models if m.gpus > largest}
    for name, n in too_big.items():
        runner.set(name, status="not run", reason=f"not run: needs {n} GPUs on one host, the largest has {largest}")
        runner.log(f"{name}: needs {n} GPUs on one host; the largest host here has {largest}")
    queue = hosts.WorkQueue([n for n in runner.order if n not in too_big],
                            {m.name: m.est_min * m.gpus for m in models})

    def lost(slot, name, e):
        runner.log(f"{slot.name}: host lost during {name} ({e}); requeued for the other slots")

    def error(slot, name, e):
        runner.set(name, status="error", reason=runner.redact(f"{type(e).__name__}: {str(e)[:300]}"))
        runner.log(f"{slot.name}: {name} error: {e}")

    def given_up(name):
        runner.set(name, status="not run", reason="not run: no GPU slot left")

    try:
        hosts.run_slots(usable, queue, runner.run_one, lost, error, given_up, stop=runner.stop,
                        need=lambda n: runner.models[n.split(RERUN_TAG, maxsplit=1)[0]].gpus)
    finally:
        st = runner.state["models"]
        run["gpu_hours"] = round(sum(s.get("gpu_s", 0) for s in st.values()) / 3600, 2)
        run["model_status"] = {n: {k: v for k, v in s.items() if k in ("status", "reason")} for n, s in st.items()}
        (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    md, code = render(run_dir, man, args.calibration, args.chart_url)
    print(f"report: {md}\nGPU-hours: {run['gpu_hours']}", file=sys.stderr)
    print(md.read_text())
    return code

