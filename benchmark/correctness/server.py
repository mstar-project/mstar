"""Start and stop servers for the correctness runner, and record what code served the run.

The server imports the checkout under test (--worktree, first on PYTHONPATH) and uses its configs.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RUNNER_ROOT = Path(__file__).resolve().parents[2]
SERVER_SH = RUNNER_ROOT / ".claude/skills/benchmarking/scripts/server.sh"
NODE_HEALTH = Path("/work1/stephaniewang/atindra/amd/jobs/node_health.sh")


def resolve_config(worktree: Path, config: str) -> Path:
    """Find a config by name, file name or path in the worktree."""
    p = Path(config)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p.resolve()
    name = p.name if p.suffix in (".yaml", ".yml") else f"{p.name}.yaml"
    cand = worktree / "configs" / name
    if not cand.is_file():
        raise FileNotFoundError(f"config {config!r} not found as a path or as {cand}")
    return cand


def _parse_value(text: str):
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted key=value overrides (values parsed as YAML), e.g.
    resources.attn.flashinfer_backend=fa2."""
    for ov in overrides:
        key, sep, value = ov.partition("=")
        if not sep:
            raise ValueError(f"override {ov!r} is not key=value")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {ov!r}: {part} is not a mapping")
        node[parts[-1]] = _parse_value(value)
    return cfg


def write_effective_config(src: Path, overrides: list[str], dest: Path) -> dict:
    cfg = yaml.safe_load(src.read_text()) or {}
    cfg = apply_overrides(cfg, overrides)
    dest.write_text(
        f"# effective config: {src}" + (f" with overrides {overrides}" if overrides else "") + "\n"
        + yaml.safe_dump(cfg, sort_keys=False)
    )
    return cfg


def config_gpu_count(cfg: dict) -> int:
    """Number of GPUs the config's ranks map to."""
    ranks = {r for g in cfg.get("node_groups", []) for r in g.get("ranks", [])}
    devices = cfg.get("rank_devices") or {}
    return len({devices.get(r, r) for r in ranks}) or 1


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def private_dirs(tag: str, run_key: str) -> tuple[Path, Path]:
    """Socket and upload dirs for one server instance under /tmp/$USER.

    mstar serve defaults both to per-user paths, so two servers on one node read each other's
    control messages even on different ports. Keep the key short, socket paths max out around 107
    characters.

    """
    base = Path("/tmp") / os.environ.get("USER", "user")
    return base / f"{tag}-{run_key}", base / f"{tag}-{run_key}-up"


@dataclass
class ServerSpec:
    name: str
    worktree: Path
    model: str
    config_path: Path
    gpus: str
    port: int
    log_path: Path
    socket_dir: Path
    upload_dir: Path
    timeout_s: int = 2400
    extra_env: dict = field(default_factory=dict)

    def fresh_dirs(self) -> None:
        for d in (self.socket_dir, self.upload_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, mode=0o700)

    def remove_dirs(self) -> None:
        for d in (self.socket_dir, self.upload_dir):
            shutil.rmtree(d, ignore_errors=True)

    def env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(self.worktree)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH")
                                                                     else []))
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("SERVER_STATE_DIR", f"/tmp/{os.environ.get('USER', 'user')}/correctness-servers")
        env.update(self.extra_env)
        return env

    def command(self) -> list[str]:
        return ["python", "-m", "mstar.cli.main", "serve", self.model, "--config", str(self.config_path),
                "--gpus", self.gpus, "--port", str(self.port), "--socket-path-prefix", f"{self.socket_dir}/",
                "--upload-dir", str(self.upload_dir)]


# Log lines that mean the server won't come up, and how many to allow. A kernel that fails to JIT is
# retried for every graph bucket (19 failed builds in 25 minutes once), so stop waiting early.
FATAL_STARTUP_PATTERNS = {"Ninja build failed": 3}


def _fatal_in_log(log: Path) -> str | None:
    try:
        text = log.read_text(errors="replace")
    except FileNotFoundError:
        return None
    for pattern, limit in FATAL_STARTUP_PATTERNS.items():
        if text.count(pattern) >= limit:
            module = re.search(r"cached_ops/([\w.+-]+)", text)
            error = re.search(r"error: [^\n]{0,200}", text)
            return (f"{text.count(pattern)}x {pattern!r} in {log}"
                    + (f"; module {module.group(1)}" if module else "")
                    + (f"; first {error.group(0)}" if error else ""))
    return None


def start(spec: ServerSpec, poll_s: float = 10.0) -> float:
    """Start the server and wait for /health. Returns seconds to ready. Gives up early if the log
    shows a fatal startup error."""
    if not port_is_free(spec.port):
        raise RuntimeError(f"port {spec.port} is busy")
    spec.fresh_dirs()
    t0 = time.monotonic()
    cmd = [str(SERVER_SH), "start", spec.name, str(spec.port), "--timeout", str(spec.timeout_s),
           "--log", str(spec.log_path), "--", *spec.command()]
    proc = subprocess.Popen(cmd, cwd=spec.worktree, env=spec.env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    while True:
        try:
            out, _ = proc.communicate(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            if fatal := _fatal_in_log(spec.log_path):
                proc.kill()
                proc.communicate()
                raise RuntimeError(f"server startup aborted: {fatal}") from None
    if proc.returncode != 0:
        raise RuntimeError(f"server start failed (rc {proc.returncode}):\n{out[-4000:]}")
    return time.monotonic() - t0


def stop(spec: ServerSpec) -> str:
    proc = subprocess.run([str(SERVER_SH), "stop", spec.name], cwd=spec.worktree, env=spec.env(),
                          capture_output=True, text=True, check=False)
    out = (proc.stdout + proc.stderr).strip()
    spec.remove_dirs()
    if not port_is_free(spec.port):
        raise RuntimeError(f"port {spec.port} still busy after stop: {out}")
    return out


_VERSIONS_PY = r"""
import importlib, json, os, socket, sys
out = {"python": sys.version.split()[0], "host": socket.gethostname()}
import mstar
out["mstar_file"] = mstar.__file__
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        out["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        props = torch.cuda.get_device_properties(0)
        out["arch"] = getattr(props, "gcnArchName", None)
except Exception as e:
    out["torch_error"] = repr(e)
for mod in ("transformers", "diffusers", "aiter", "flashinfer", "triton", "numpy", "safetensors"):
    try:
        m = importlib.import_module(mod)
        out[mod] = getattr(m, "__version__", "present")
    except Exception as e:
        out[mod] = f"unavailable ({type(e).__name__})"
try:
    from mstar.platforms import current_platform
    out["platform"] = current_platform.name if hasattr(current_platform, "name") else str(current_platform)
except Exception as e:
    out["platform"] = f"unavailable ({type(e).__name__})"
out["env"] = {k: v for k, v in os.environ.items() if k.startswith(("HIP_", "CUDA_VISIBLE", "HSA_", "MSTAR_",
              "PYTORCH_", "TORCHINDUCTOR", "MIOPEN_", "AITER_", "HF_HUB_OFFLINE", "SLURM_JOB_ID"))}
print(json.dumps(out))
"""


def _git(worktree: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else f"(git {' '.join(args)} failed: {p.stderr.strip()[:200]})"


def collect_versions(spec: ServerSpec) -> dict:
    """Git state of the worktree and package versions as the server sees them."""
    info = {
        "worktree": str(spec.worktree),
        "git_head": _git(spec.worktree, "rev-parse", "HEAD"),
        "git_branch": _git(spec.worktree, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_describe": _git(spec.worktree, "log", "-1", "--format=%h %s"),
        "git_dirty": [ln for ln in _git(spec.worktree, "status", "--porcelain").splitlines() if ln][:50],
        "runner_head": _git(RUNNER_ROOT, "rev-parse", "HEAD"),
        "command": shlex.join(spec.command()),
    }
    p = subprocess.run(["python", "-c", _VERSIONS_PY], cwd=spec.worktree, env=spec.env(), capture_output=True,
                       text=True, timeout=600, check=False)
    try:
        info["stack"] = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        info["stack"] = {"error": (p.stderr or p.stdout)[-2000:]}
    mfile = info["stack"].get("mstar_file", "")
    info["serves_worktree"] = bool(mfile) and Path(mfile).resolve().is_relative_to(spec.worktree.resolve())
    return info


def node_health_snapshot(out: Path) -> str:
    if not NODE_HEALTH.is_file():
        return f"{NODE_HEALTH} not found"
    p = subprocess.run([str(NODE_HEALTH), "snapshot", str(out)], capture_output=True, text=True, timeout=600,
                       check=False)
    return f"rc={p.returncode} {(p.stdout + p.stderr).strip()[-300:]}"
