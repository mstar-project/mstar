"""The results file every suite writes, and the run metadata it carries.

One file is one measured point: a suite, a server, a dataset slice and a
load setting (mode, concurrency or rate), repeated ``repeats`` times. A sweep
writes one file per point. Layout (``schema: "mstar-eval/1"``)::

    {
      "schema": "mstar-eval/1",
      "suite": "tts", "label": "kokoro", "created_at": "2026-10-16T07:00:00Z",
      "config": {...},       # everything that decides the numbers (dataset, n, load, judges, seed)
      "meta": {...},         # git commit, GPU, driver, versions, MSTAR_* env, server, argv
      "repeats": [{"index": 0, "wall_s": 12.3, "metrics": {...}, "requests": [...]}, ...],
      "summary": {"<metric>": {"mean": .., "std": .., "min": .., "max": .., "n": 3}, ...}
    }

``metrics`` are one number per repeat (a p50 over the repeat's requests, a
corpus WER, ...). ``summary`` is mean +- sample std of each metric over the
repeats. ``requests`` keeps the per-request rows so any other statistic can be
recomputed later without rerunning.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from benchmark.eval.stats import summarize

SCHEMA = "mstar-eval/1"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


@dataclass
class RepeatResult:
    index: int
    wall_s: float
    metrics: dict[str, float | None]
    requests: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    suite: str
    label: str
    config: dict[str, Any]
    meta: dict[str, Any]
    repeats: list[RepeatResult]
    summary: dict[str, dict] = field(default_factory=dict)
    created_at: str = ""
    schema: str = SCHEMA

    def finalize(self) -> RunResult:
        self.summary = summarize_repeats(self.repeats)
        if not self.created_at:
            self.created_at = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {"schema": d.pop("schema"), **d}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False, default=_json_default) + "\n")
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunResult:
        problems = validate(d)
        if problems:
            raise ValueError("not an mstar-eval result: " + "; ".join(problems))
        return cls(
            suite=d["suite"],
            label=d["label"],
            config=d["config"],
            meta=d["meta"],
            repeats=[RepeatResult(**r) for r in d["repeats"]],
            summary=d.get("summary", {}),
            created_at=d.get("created_at", ""),
            schema=d["schema"],
        )

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def summarize_repeats(repeats: list[RepeatResult]) -> dict[str, dict]:
    names: list[str] = []
    for r in repeats:
        names += [k for k in r.metrics if k not in names]
    return {k: summarize(r.metrics.get(k) for r in repeats) for k in names}


def validate(d: Any) -> list[str]:
    """Problems with a decoded results file (empty when it is valid)."""
    if not isinstance(d, dict):
        return ["top level is not an object"]
    problems = []
    if d.get("schema") != SCHEMA:
        problems.append(f"schema is {d.get('schema')!r}, expected {SCHEMA!r}")
    for key, typ in (("suite", str), ("label", str), ("config", dict), ("meta", dict), ("repeats", list)):
        if not isinstance(d.get(key), typ):
            problems.append(f"{key!r} missing or not a {typ.__name__}")
    for i, r in enumerate(d.get("repeats") or []):
        if not isinstance(r, dict) or not isinstance(r.get("metrics"), dict) or "index" not in r:
            problems.append(f"repeats[{i}] lacks index/metrics")
    return problems


def redact_url(url: str | None) -> str | None:
    """Keep loopback URLs; replace any other host so results never carry an address."""
    if not url:
        return url
    parts = urlsplit(url)
    if (parts.hostname or "") in _LOCAL_HOSTS:
        return url
    netloc = "remote-host" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _run(cmd: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo: str | Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], repo)
    if commit is None:
        return {}
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo),
        "dirty": bool(status),
        "describe": _run(["git", "log", "-1", "--format=%cs %s"], repo),
    }


def gpu_info() -> list[dict[str, str]]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
        "--format=csv,noheader",
    ])
    if not out:
        return []
    keys = ("name", "driver", "memory", "max_sm_clock")
    return [dict(zip(keys, (p.strip() for p in line.split(",")), strict=False)) for line in out.splitlines()]


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def served_models(url: str) -> list[str] | None:
    """Model ids the server reports on ``/v1/models`` (its own identity, not the request's)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=10) as r:
            return [m.get("id") for m in json.loads(r.read()).get("data", [])]
    except (OSError, ValueError, AttributeError):
        return None


def collect_metadata(repo: str | Path | None = None, server: dict[str, Any] | None = None) -> dict[str, Any]:
    """Where and how the numbers were produced. No hostnames, IPs or home paths."""
    repo = repo or Path(__file__).resolve().parents[2]
    torch_v = cuda_v = None
    if "torch" in sys.modules or _version("torch"):
        try:
            import torch

            torch_v, cuda_v = torch.__version__, torch.version.cuda
        except ImportError:
            pass
    home = os.path.expanduser("~")
    argv = [a.replace(home, "~") for a in sys.argv]
    return {
        "git": git_info(repo),
        "host": {
            "gpus": gpu_info(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(terse=True),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch_v,
            "cuda": cuda_v,
            "mstar-ai": _version("mstar-ai"),
            "transformers": _version("transformers"),
        },
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("MSTAR_")},
        "server": server or {},
        "argv": argv,
    }
