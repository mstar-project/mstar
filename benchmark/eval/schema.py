"""The results file every suite writes, and the run metadata it carries.

One file is one measured point: a suite, a server, a dataset slice and a
load setting (mode, concurrency or rate), repeated ``repeats`` times. A sweep
writes one file per point. Layout (``schema: "mstar-eval/1"``)::

    {
      "schema": "mstar-eval/1",
      "suite": "tts", "label": "kokoro", "created_at": "2026-10-16T07:00:00Z",
      "config": {...},       # everything that decides the numbers (dataset, n, load, judges, seed)
      "meta": {...},         # git commit, GPU, driver, versions, MSTAR_* env, server, CLI flags
                             # (meta.git and meta.versions describe the eval's checkout;
                             # meta.server.git and meta.server.versions what was served)
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
import ipaddress
import json
import math
import os
import platform
import re
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
        # The one place everything passes before it is written: no addresses, secrets or home paths.
        d["config"], d["meta"], d["label"] = scrub(d["config"]), scrub(d["meta"]), _scrub(d["label"])
        for rep in d["repeats"]:
            rep["requests"] = [_scrub_row(row) for row in rep["requests"]]
        d["repeats"], d["summary"] = finite(d["repeats"]), finite(d["summary"])
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
        if "host" in d["meta"] and "machine" not in d["meta"]:  # early mstar-eval/1 files
            d["meta"]["machine"] = d["meta"].pop("host")
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


def _scrub_row(row: dict[str, Any]) -> dict[str, Any]:
    """A request row as written: string fields (transcripts, errors) scrubbed, ids kept (they
    are validated plain names that rescoring matches). Fields the scrubber changed are listed
    in ``scrubbed`` so nothing downstream mistakes them for what the model said."""
    out, changed = {}, []
    for k, v in row.items():
        if isinstance(v, str) and k != "id":
            clean = _scrub(v)
            if clean != v:
                changed.append(k)
            v = clean
        out[k] = v
    if changed:
        out["scrubbed"] = sorted(set(changed) | set(row.get("scrubbed") or []))
    return out


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
    """Loopback hosts are kept, any other host is replaced; credentials and queries are always dropped."""
    if not url:
        return url
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "<url>"
    host = parts.hostname or ""
    if host not in _LOCAL_HOSTS:  # a remote URL keeps only its scheme and port
        return urlunsplit((parts.scheme, "remote-host" + (f":{port}" if port else ""), "", "", ""))
    path = _HOME.sub("~", parts.path.replace(os.path.expanduser("~"), "~"))
    path = _IPV6.sub(_ipv6, _IPV4.sub(lambda m: m.group(0) if m.group(0) in _LOCAL_HOSTS else "<ip>", path))
    return urlunsplit((parts.scheme, host + (f":{port}" if port else ""), path, "", ""))


_SECRET = re.compile(
    r"TOKEN|SECRET|PASSW|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|AUTH|CREDENTIAL|COOKIE|SESSION", re.IGNORECASE
)
# Keys whose values name a machine; their values are kept only when loopback.
_HOSTLIKE = re.compile(r"HOST|ADDR|ENDPOINT|URL|URI|DOMAIN|PEER|(^|_)IP($|_)", re.IGNORECASE)
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"]+")
# Bounded by anything that cannot continue an address, so "label_10.0.0.1" still matches.
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6 = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])")
_HOME = re.compile(r"/(?:home|Users)/[^/\s'\"]+")
_URL_FLAGS = {"--url"}
_PAIR_FLAGS = {"--serve-env", "--extra"}
_CMD_FLAGS = {"--serve-cmd"}
_HOST_FLAGS = {"--host", "--hostname", "--server", "--endpoint", "--address", "--addr"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _ipv6(m: re.Match) -> str:
    text = m.group(0)
    try:  # clock times ("07:02:51") match the pattern but are not addresses
        addr = ipaddress.IPv6Address(text)
    except ValueError:
        return text
    return text if addr.is_loopback else "<ip>"


def _scrub(text: str) -> str:
    """Home paths (anyone's) to ``~``; non-loopback URLs, IPv4 and IPv6 literals redacted."""
    text = _HOME.sub("~", text.replace(os.path.expanduser("~"), "~"))
    text = _URL.sub(lambda m: redact_url(m.group(0)) or "", text)
    text = _IPV4.sub(lambda m: m.group(0) if m.group(0) in _LOCAL_HOSTS else "<ip>", text)
    return _IPV6.sub(_ipv6, text)


def _value(key: str, value: str) -> str:
    """A ``key=value`` setting's value as a results file may keep it."""
    if _SECRET.search(key):
        return "<redacted>"
    if _HOSTLIKE.search(key):
        parts = [v for v in re.split(r"[,\s]+", value.strip()) if v]
        if len(parts) == 1 and "://" in parts[0]:
            return redact_url(parts[0]) or "<host>"  # loopback kept; credentials and queries dropped
        if all(_is_local(v) and "://" not in v for v in parts):
            return value
        return "<host>"
    return _scrub(value)


def _is_local(value: str) -> bool:
    """A loopback host, ``host:port`` or URL."""
    if "://" in value:
        try:
            return (urlsplit(value).hostname or "") in _LOCAL_HOSTS
        except ValueError:
            return False
    host, _, port = value.rpartition(":")
    return value in _LOCAL_HOSTS or (port.isdigit() and host.strip("[]") in _LOCAL_HOSTS)


def scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Environment for a results file: secret values dropped, non-loopback hosts redacted."""
    return {k: _value(k, str(v)) for k, v in sorted(env.items())}


def scrub(obj: Any, host: bool = False) -> Any:
    """Recursively scrub a config or metadata tree. Values under secret-looking keys are
    dropped; everything under a host-like key (including lists and nested objects) is
    treated as a host."""
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            name = k if isinstance(k, str) else ""
            if host and isinstance(k, str):  # under a host-like key, mapping keys can be hosts too
                k = _value("HOST", k)
                k = k if k not in out else f"{k}{i}"
            out[k] = "<redacted>" if _SECRET.search(name) else scrub(v, host or bool(_HOSTLIKE.search(name)))
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub(v, host) for v in obj]
    if isinstance(obj, str):
        return _value("HOST", obj) if host else _scrub(obj)
    return obj


def finite(obj: Any) -> Any:
    """JSON-safe numbers: NaN and infinities become null (a results file must always save)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def _fingerprint(tokens: list[str]) -> str:
    import hashlib
    import shlex

    return hashlib.sha256(shlex.join(tokens).encode()).hexdigest()[:12]


def describe_command(cmd: list[str] | str | None) -> str | None:
    """A custom server command for a results file: a fingerprint only, never its text
    (shell wrappers, inline exports, env options and engine flags can hold secrets and
    hostnames). Two runs with the same fingerprint launched the same command."""
    import shlex

    if not cmd:
        return None
    return f"<command sha256:{_fingerprint(shlex.split(cmd) if isinstance(cmd, str) else list(cmd))}>"


def describe_args(args: list[str] | None) -> dict[str, Any] | None:
    """Arguments forwarded to ``mstar serve``: the flag names and a fingerprint of the whole
    list, never the values (``--ho=<host>`` is an accepted abbreviation of ``--host``)."""
    if not args:
        return None
    return {"flags": sorted({a.split("=", 1)[0] for a in args if a.startswith("-")}), "sha256": _fingerprint(args)}


def _pair(kv: str) -> str:
    k, sep, v = kv.partition("=")
    if not sep:
        return _scrub(kv)
    try:
        nested = json.loads(v)
    except ValueError:
        nested = None
    if isinstance(nested, (dict, list)):  # --extra 'k={"api_key": ...}': scrub the structure, not the text
        return f"{k}=<redacted>" if _SECRET.search(k) else f"{k}={json.dumps(scrub(nested, bool(_HOSTLIKE.search(k))))}"
    return f"{k}{sep}{_value(k, v)}"


def _host(value: str) -> str:
    return value if value in _LOCAL_HOSTS else "remote-host"


def scrub_argv(argv: list[str]) -> list[str]:
    """A command line for a results file: ``--serve-cmd`` values become a fingerprint; values
    of secret-looking flags and ``KEY=VALUE`` assignments, non-loopback hosts and URLs, IP
    literals and home paths are removed."""
    out: list[str] = []
    for i, arg in enumerate(argv):
        prev = argv[i - 1] if i else ""
        flag, eq, val = arg.partition("=")
        is_flag = flag.startswith("-")
        if prev in _CMD_FLAGS:
            out.append(describe_command(arg))
            continue
        if is_flag and flag in _CMD_FLAGS and eq:
            out.append(f"{flag}={describe_command(val)}")
            continue
        if prev in _PAIR_FLAGS:
            arg = _pair(arg)
        elif is_flag and flag in _PAIR_FLAGS and eq:
            arg = f"{flag}={_pair(val)}"
        elif prev.startswith("-") and "=" not in prev and _SECRET.search(prev):
            arg = "<redacted>"
        elif is_flag and eq and _SECRET.search(flag):
            arg = f"{flag}=<redacted>"
        elif prev in _URL_FLAGS:
            arg = redact_url(arg) or arg
        elif is_flag and flag in _URL_FLAGS and eq:
            arg = f"{flag}={redact_url(val)}"
        elif prev in _HOST_FLAGS:
            arg = _host(arg)
        elif is_flag and flag in _HOST_FLAGS and eq:
            arg = f"{flag}={_host(val)}"
        elif not is_flag and _ASSIGN.match(arg):
            arg = _pair(arg)
        out.append(_scrub(arg))
    return out


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


def cli_shape(argv: list[str]) -> dict[str, Any]:
    """The command and the names of the flags given, never their values: everything that
    decides the numbers is in the structured (and scrubbed) ``config``, and raw values
    (forwarded server args, inline commands, URLs) are where secrets and hosts hide."""
    words = [a for a in argv[1:3] if not a.startswith("-")]
    command = words[:2] if words[:1] == ["run"] else words[:1]
    flags = sorted({a.split("=", 1)[0] for a in argv[1:] if a.startswith("--")})
    return {"command": " ".join(command), "flags": flags}


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
    return {
        "git": git_info(repo),
        "machine": {
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
        "env": scrub_env({k: v for k, v in os.environ.items() if k.startswith("MSTAR_")}),
        "server": server or {},
        "cli": cli_shape(sys.argv),
    }
