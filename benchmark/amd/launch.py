"""Start and stop the server of one benchmark arm.

    python -m benchmark.amd.launch start --arm vllm-aiter --row qwen3_5_0.8b --gpus 4 --port 8201 --run-dir DIR
    python -m benchmark.amd.launch stop --run-dir DIR

``start`` writes DIR/server.log, DIR/versions.json, DIR/startup.json and, once the server answers
/health and a first request, DIR/ready. Servers run through the benchmarking skill's server.sh in
their own session, so ``stop`` takes the whole process group down. mstar runs from the venv with
the worktree first on PYTHONPATH; the other engines run in their apptainer image via ctr.sh with
a clean environment.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from benchmark.amd import catalog
from benchmark.amd.catalog import Arm, Row

REPO = Path(__file__).resolve().parents[2]
SERVER_SH = REPO / ".claude/skills/benchmarking/scripts/server.sh"
CTR_SH = Path(catalog.CONTAINERS) / "scripts/ctr.sh"
IMAGES_JSON = Path(catalog.CONTAINERS) / "versions.json"
USER = os.environ.get("USER", "user")
LOCK_DIR = Path("/tmp") / USER / "wsj-gpu-locks"
# Host variables a container launch keeps; ctr.sh drops everything else anyway.
CTR_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "TERM", "SLURM_JOB_ID", "HF_HOME", "HF_TOKEN")
# Prefixes ctr.sh forwards into the container. Anything else has to go through CTR_ENV.
CTR_FORWARDED = ("VLLM_", "SGLANG_", "AITER_", "HSA_", "NCCL_", "RCCL_", "MIOPEN_", "TORCH_", "TORCHINDUCTOR_",
                 "PYTORCH_", "HF_", "DIFFUSION_", "OMNI_", "HIP_")
# ctr.sh's own settings: read by ctr.sh on the host, not passed into the container.
CTR_OWN = ("CTR_PWD", "CTR_BIND")
# A GPU is busy when this much VRAM is in use (idle: ~0.3 GB). amd-smi lists every process that opened a GPU
# under all of them (ROCr opens every render node unless ROCR_VISIBLE_DEVICES limits it), with the process's
# total memory rather than its share of that GPU, so process lists are recorded but don't decide.
VRAM_BUSY_MB = 1024
# Server log lines that say which attention backend an engine picked (vLLM, SGLang, sglang-omni).
BACKEND_PATTERNS = (re.compile(r"Using (\w+) backend \(selected"), re.compile(r"Overriding with (\w+) out of"),
                    re.compile(r"'attention_backend': '(\w+)'"), re.compile(r"Using (\w+/?\w*) GDN prefill kernel"),
                    re.compile(r"\battention_backend=(\w+)"))
# Server log lines naming the fused-MoE launch-config files an engine loaded (mstar's tuned tables, vLLM's
# configs), and vLLM's note that it found none and runs its default config.
MOE_TUNED_PATTERNS = (re.compile(r"fused MoE: tuned Triton configs from (\S+\.json)"),
                      re.compile(r"Using configuration from (\S+\.json) for MoE layer"))
MOE_DEFAULT_PATTERN = re.compile(r"Using default MoE config\..*?Config file not found at (.+)")
# mstar's fused all-reduce + norm under TP: the backend it set up, or why it fell back to RCCL and a separate norm.
ALLREDUCE_FUSED = re.compile(r"all-reduce fusion workspace: (.+?(?:up to \d+ tokens x \d+|$))", re.M)
ALLREDUCE_FALLBACK = re.compile(r"((?:AITER fused all-reduce \+ RMSNorm (?:failed its startup check|unavailable"
                                r"|doesn't cover)|AITER custom all-reduce failed to build|FlashInfer all-reduce "
                                r"fusion unavailable|no FlashInfer all-reduce fusion backend works here"
                                r"|TP ranks lost sync building the fusion workspace).*)$", re.M)
TTS_VOICE = {"kokoro": "af_heart", "orpheus": "tara", "qwen3_tts": "vivian", "qwen3_tts_1p7b": "vivian"}


def gpu_list(text: str) -> list[int]:
    gpus = [int(x) for x in text.split(",") if x.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f"bad GPU list {text!r}")
    return gpus


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run(cmd: list[str], timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **kw)


def amd_smi_json(*args: str) -> object | None:
    try:
        p = _run(["amd-smi", *args, "--json"], timeout=180)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    out = p.stdout
    starts = [k for k in (out.find("{"), out.find("[")) if k >= 0]
    if p.returncode != 0 or not starts:
        return None
    try:
        return json.loads(out[min(starts):])
    except ValueError:
        return None


def _entries(doc) -> list[dict]:
    if isinstance(doc, dict):
        return doc.get("gpu_data", [doc])
    return doc if isinstance(doc, list) else []


@functools.cache
def smi_index_map() -> dict[int, int]:
    """amd-smi's number for each HIP device. HIP counts GPUs in KFD topology order and amd-smi by PCI address,
    which differ on some nodes, so every amd-smi lookup about a HIP device goes through this. Identity if
    amd-smi doesn't give the KFD node."""
    entries = [e for e in _entries(amd_smi_json("list")) if isinstance(e.get("gpu"), int)]
    if not entries or any(not isinstance(e.get("node_id"), int) for e in entries):
        return {}
    return {hip: e["gpu"] for hip, e in enumerate(sorted(entries, key=lambda e: e["node_id"]))}


def smi_gpu(hip: int) -> int:
    return smi_index_map().get(hip, hip)


def gpu_occupancy(gpus: list[int]) -> dict[int, dict]:
    """VRAM in use and the processes amd-smi lists, per HIP device. Empty if amd-smi can't tell."""
    out: dict[int, dict] = {}
    want = {smi_gpu(g): g for g in gpus}
    for e in _entries(amd_smi_json("metric", "--mem-usage")):
        g = want.get(e.get("gpu"))
        if g is not None:
            used = (e.get("mem_usage") or {}).get("used_vram")
            used = used.get("value") if isinstance(used, dict) else used
            out.setdefault(g, {})["vram_used_mb"] = float(used) if isinstance(used, (int, float)) else None
    for e in _entries(amd_smi_json("process")):
        g = want.get(e.get("gpu"))
        if g is not None:
            procs = []
            for item in e.get("process_list") or []:
                info = item.get("process_info") if isinstance(item, dict) else None
                if isinstance(info, dict):
                    procs.append(f"{info.get('name')}:{info.get('pid')}")
            out.setdefault(g, {})["processes"] = procs
    return out


def _short_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


@dataclass
class Launch:
    arm: Arm
    row: Row
    gpus: list[int]
    port: int
    run_dir: Path
    worktree: Path = REPO
    numa_node: int | None = None
    lane: str = ""
    startup_timeout: int = 3600
    force: bool = False

    def __post_init__(self):
        self.run_dir = Path(self.run_dir).resolve()
        self.worktree = Path(self.worktree).resolve()
        if len(self.gpus) != self.row.gpus:
            raise ValueError(f"{self.row.name} needs {self.row.gpus} GPU(s), got {self.gpus}")
        if self.row.family not in self.arm.families:
            raise ValueError(f"arm {self.arm.name} does not serve {self.row.family} rows")
        key = _short_key(str(self.run_dir))
        # Unix socket paths stop at ~107 characters, so the IPC prefix goes under /tmp, not the run dir.
        self.ipc_dir = Path("/tmp") / USER / f"wsj-{key}"
        self.upload_dir = Path("/tmp") / USER / f"wsj-{key}-up"
        self.state_dir = self.run_dir / ".server"
        self.name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{self.arm.name}-{self.row.name}")

    # --- what runs ---------------------------------------------------------------------------
    @property
    def startup_wait(self) -> int:
        """Seconds to wait for /health: the launch's limit, or longer where the arm or the model needs it."""
        return max(self.startup_timeout, self.arm.startup_timeout or 0, self.row.startup_timeout or 0)

    @property
    def server_gpus(self) -> list[int]:
        return self.gpus[: self.arm.server_gpus] if self.arm.server_gpus else self.gpus

    @property
    def client_gpu(self) -> int:
        rest = self.gpus[len(self.server_gpus):]
        return rest[0] if rest else self.gpus[0]

    @property
    def gpu_text(self) -> str:
        return ",".join(str(g) for g in self.server_gpus)

    def mstar_config(self) -> Path:
        cfg = self.worktree / "configs" / f"{self.row.name}.yaml"
        if not cfg.is_file():
            raise FileNotFoundError(cfg)
        return cfg

    def command(self) -> list[str]:
        args, _ = catalog.server_args(self.arm, self.row)
        e = self.arm.engine
        if e == "mstar":
            cfg = self.mstar_config()
            model = (yaml.safe_load(cfg.read_text()) or {}).get("model", self.row.name)
            cmd = [sys.executable, "-m", "mstar.cli.main", "serve", model, "--config", str(cfg),
                   "--gpus", self.gpu_text, "--port", str(self.port),
                   "--socket-path-prefix", f"{self.ipc_dir}/", "--upload-dir", str(self.upload_dir), *args]
        elif e in ("vllm", "vllm-omni"):
            cmd = [str(CTR_SH), self.arm.image, "vllm", "serve", self.row.hf_id, "--port", str(self.port), *args]
        elif e == "sglang":
            cmd = [str(CTR_SH), self.arm.image, "python3", "-m", "sglang.launch_server", "--model-path",
                   self.row.hf_id, "--host", "127.0.0.1", "--port", str(self.port), *args]
        elif e == "sglang-diffusion":
            cmd = [str(CTR_SH), self.arm.image, "sglang", "serve", "--model-path", self.row.hf_id, "--host",
                   "127.0.0.1", "--port", str(self.port), *args]
        elif e == "sglang-omni":
            cmd = [str(CTR_SH), self.arm.image, "python3", "-m", "sglang_omni.cli", "serve", "--model-path",
                   self.row.hf_id, "--host", "127.0.0.1", "--port", str(self.port),
                   "--endpoints.base_path", str(self.ipc_dir), *args]
        else:
            raise ValueError(f"engine {e!r} has no server")
        if self.numa_node is not None:
            cmd = ["numactl", f"--cpunodebind={self.numa_node}", f"--membind={self.numa_node}", *cmd]
        return cmd

    def environment(self) -> dict[str, str]:
        _, arm_env = catalog.server_args(self.arm, self.row)
        if self.arm.engine == "mstar":
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(self.worktree)] + ([env["PYTHONPATH"]]
                                                                        if env.get("PYTHONPATH") else []))
            env.update({"HIP_VISIBLE_DEVICES": self.gpu_text, "CUDA_VISIBLE_DEVICES": self.gpu_text})
            env.update(arm_env)
        else:
            env = {k: os.environ[k] for k in CTR_KEEP if k in os.environ}
            env["GPUS"] = self.gpu_text
            if self.lane:
                # ctr.sh's own scratch dir (caches, AITER JIT copy) is per job and image; lanes get one each.
                job = os.environ.get("SLURM_JOB_ID", "nojob")
                sif = Path(image_record(self.arm.image).get("sif") or self.arm.image).stem
                env["CTR_SCRATCH"] = f"/tmp/{USER}/ctr-{job}-{sif}-{self.lane}"
            extra = [f"CUDA_VISIBLE_DEVICES={self.gpu_text}"]
            for k, v in arm_env.items():
                if k.startswith(CTR_FORWARDED) or k in CTR_OWN:
                    env[k] = v
                else:
                    extra.append(f"{k}={v}")
            env["CTR_ENV"] = " ".join(extra)
        env.setdefault("HF_HOME", "/work1/stephaniewang/atindra/hf_cache")
        env["HF_HUB_OFFLINE"] = "1"
        env["SERVER_STATE_DIR"] = str(self.state_dir)
        return env

    # --- guards ------------------------------------------------------------------------------
    def _lock_path(self, gpu: int) -> Path:
        return LOCK_DIR / f"gpu{gpu}"

    def _take_locks(self) -> None:
        """One lock file per GPU naming the run that holds it. A lock whose owner is gone is stale."""
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        taken = []
        try:
            for g in self.gpus:
                p = self._lock_path(g)
                for _ in range(2):
                    try:
                        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    except FileExistsError:
                        try:
                            owner = json.loads(p.read_text())
                        except (OSError, ValueError):
                            owner = {}
                        if owner.get("pid") and _pid_alive(int(owner["pid"])):
                            raise RuntimeError(f"GPU {g} is held by {owner.get('name')} "
                                               f"({owner.get('run_dir')})") from None
                        p.unlink(missing_ok=True)
                        continue
                    with os.fdopen(fd, "w") as f:
                        json.dump({"pid": os.getpid(), "name": self.name, "run_dir": str(self.run_dir)}, f)
                    taken.append(p)
                    break
                else:
                    raise RuntimeError(f"could not lock GPU {g}")
        except Exception:
            for p in taken:
                p.unlink(missing_ok=True)
            raise

    def _point_locks_at(self, pid: int) -> None:
        for g in self.gpus:
            self._lock_path(g).write_text(json.dumps({"pid": pid, "name": self.name, "run_dir": str(self.run_dir)}))

    def _release_locks(self) -> None:
        for g in self.gpus:
            p = self._lock_path(g)
            try:
                if json.loads(p.read_text()).get("run_dir") == str(self.run_dir):
                    p.unlink()
            except (OSError, ValueError):
                pass

    def check_free(self, settle_s: float = 120) -> dict:
        """Refuse a busy port or a GPU with memory in use. VRAM of a server that just exited takes a few
        seconds to come back, so it is waited on for up to ``settle_s``."""
        if not port_is_free(self.port):
            raise RuntimeError(f"port {self.port} is busy")
        deadline = time.monotonic() + settle_s
        while True:
            occ = gpu_occupancy(self.gpus)
            busy = {g: o for g, o in occ.items() if (o.get("vram_used_mb") or 0) > VRAM_BUSY_MB}
            if not busy or self.force:
                return occ
            if time.monotonic() > deadline:
                raise RuntimeError(f"GPU(s) in use: {busy}")
            time.sleep(5)

    # --- lifecycle ---------------------------------------------------------------------------
    def server_pid(self) -> int | None:
        try:
            return int((self.state_dir / f"{self.name}.pid").read_text().strip())
        except (OSError, ValueError):
            return None

    def start(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)
        for marker in ("ready", "stopped"):
            (self.run_dir / marker).unlink(missing_ok=True)
        self._take_locks()
        try:
            occ = self.check_free()
            for d in (self.ipc_dir, self.upload_dir):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, mode=0o700)
            cmd, env = self.command(), self.environment()
            versions = collect_versions(self, cmd, env)
            versions["gpu_occupancy_before"] = occ
            (self.run_dir / "versions.json").write_text(json.dumps(versions, indent=1, default=str))
            t0 = time.monotonic()
            p = subprocess.run([str(SERVER_SH), "start", self.name, str(self.port), "--timeout",
                                str(self.startup_wait), "--log", str(self.run_dir / "server.log"), "--", *cmd],
                               cwd=self.worktree if self.arm.engine == "mstar" else self.run_dir, env=env,
                               capture_output=True, text=True, check=False)
            ready_s = time.monotonic() - t0
            (self.run_dir / "server_sh_start.txt").write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                raise RuntimeError(f"{self.name} did not come up (rc {p.returncode}): {(p.stdout + p.stderr)[-3000:]}")
            pid = self.server_pid()
            if pid:
                self._point_locks_at(pid)
            first_s, first_err = first_request(self)
            startup = {"ready_s": round(ready_s, 1), "first_request_s": first_s, "first_request_error": first_err,
                       "backends": backends_in_log(self.run_dir / "server.log"),
                       "moe_configs": moe_configs_in_log(self.run_dir / "server.log"),
                       "allreduce": allreduce_in_log(self.run_dir / "server.log"),
                       "server_pid": pid, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "ipc_dir": str(self.ipc_dir), "upload_dir": str(self.upload_dir)}
            (self.run_dir / "startup.json").write_text(json.dumps(startup, indent=1))
            if first_err:
                raise RuntimeError(f"{self.name}: first request failed: {first_err}")
            (self.run_dir / "ready").write_text(json.dumps(startup))
            return startup
        except BaseException:
            try:
                self.stop()
            finally:
                self._release_locks()
            raise

    def stop(self) -> dict:
        pid = self.server_pid()
        sid = None
        if pid and _pid_alive(pid):
            try:
                sid = os.getsid(pid)
            except OSError:
                sid = None
        out = ""
        if (self.state_dir / f"{self.name}.pid").exists():
            p = subprocess.run([str(SERVER_SH), "stop", self.name], env={**os.environ,
                               "SERVER_STATE_DIR": str(self.state_dir)}, capture_output=True, text=True, check=False)
            out = p.stdout + p.stderr
        left = session_processes(sid) if sid else []
        if left:
            for lp in left:
                try:
                    os.kill(lp, 9)
                except OSError:
                    pass
            time.sleep(2)
            left = session_processes(sid)
        for d in (self.ipc_dir, self.upload_dir):
            shutil.rmtree(d, ignore_errors=True)
        self._release_locks()
        info = {"stopped": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "server_sh": out[-2000:], "leftover_pids": left,
                "port_free": port_is_free(self.port)}
        if self.run_dir.is_dir():
            (self.run_dir / "stopped").write_text(json.dumps(info, indent=1))
        return info


def backends_in_log(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    found = []
    for pat in BACKEND_PATTERNS:
        for m in pat.finditer(text):
            if m.group(1) not in found:
                found.append(m.group(1))
    return found[:6]


def moe_configs_in_log(path: Path) -> dict:
    """Fused-MoE config files the server loaded ("tuned") and the ones it looked for in vain ("default")."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {"tuned": [], "default": []}
    tuned = {Path(m).name for pat in MOE_TUNED_PATTERNS for m in pat.findall(text)}
    default = {Path(p.strip()).name for m in MOE_DEFAULT_PATTERN.findall(text) for p in m.split(", ")}
    return {"tuned": sorted(tuned), "default": sorted(default)}


def allreduce_in_log(path: Path) -> dict | None:
    """Whether the server fused all-reduce with the norm (and with what), or the line saying why not. None if
    the log says neither, as without TP."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    fallback = ALLREDUCE_FALLBACK.search(text)
    if fallback:
        return {"fused": False, "line": fallback.group(1).strip()[:400]}
    fused = ALLREDUCE_FUSED.search(text)
    return {"fused": True, "backend": fused.group(1).strip()} if fused else None


def session_processes(sid: int) -> list[int]:
    p = _run(["ps", "-e", "-o", "pid=,sid="])
    me = os.getpid()
    return [int(a) for a, b in (ln.split() for ln in p.stdout.splitlines() if ln.strip())
            if int(b) == sid and int(a) != me]


def _post_json(url: str, body: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def first_request(launch: Launch, timeout: float = 3600) -> tuple[float | None, str | None]:
    """One small request right after /health. Its latency holds the first-request JIT and
    compile work, kept apart from the timed cells."""
    base = f"http://127.0.0.1:{launch.port}"
    row = launch.row
    if row.family == "text":
        url, body = f"{base}/v1/chat/completions", {
            "model": row.hf_id, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8,
            "temperature": 0.0}
    elif row.family == "tts" and launch.arm.engine in ("mstar", "vllm-omni", "sglang-omni"):
        url, body = f"{base}/v1/audio/speech", {
            "model": row.name if launch.arm.engine == "mstar" else row.hf_id, "input": "Hello there, this is a test.",
            "response_format": "pcm", "voice": TTS_VOICE.get(row.runner_model)}
        if row.runner_model.startswith("qwen3_tts"):
            body["language"] = "English"
    elif launch.arm.engine == "vllm" and row.family == "tts":
        # A plain LM behind a TTS client (Orpheus): one short completion.
        url, body = f"{base}/v1/completions", {"model": row.hf_id, "prompt": "Hello", "max_tokens": 8}
    else:
        return None, None
    t0 = time.monotonic()
    try:
        status, data = _post_json(url, body, timeout)
    except Exception as e:  # recorded; the caller decides
        return round(time.monotonic() - t0, 2), f"{type(e).__name__}: {e}"
    if status != 200 or not data:
        return round(time.monotonic() - t0, 2), f"HTTP {status}, {len(data)} bytes"
    return round(time.monotonic() - t0, 2), None


# --- versions -------------------------------------------------------------------------------

_MSTAR_VERSIONS = r"""
import importlib.metadata as md, json, sys
out = {"python": sys.version.split()[0]}
import torch
out["torch"] = torch.__version__
out["torch_hip"] = getattr(torch.version, "hip", None)
import mstar
out["mstar_file"] = mstar.__file__
for d in ("triton", "triton-rocm", "amd-aiter", "amd-flashinfer", "flashinfer-python", "transformers", "mstar-ai",
          "mstar_rust"):
    try:
        out[d] = md.version(d)
    except Exception:
        pass
print(json.dumps(out))
"""

_CTR_VERSIONS = r"""
import importlib.metadata as md, json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_hip"] = getattr(torch.version, "hip", None)
except Exception as e:
    out["torch_error"] = repr(e)
for d in ("vllm", "vllm-omni", "vllm_omni", "sglang", "sglang-omni", "sglang_omni", "amd-aiter", "triton",
          "transformers", "flash_attn", "diffusers"):
    try:
        out[d] = md.version(d)
    except Exception:
        pass
print(json.dumps(out))
"""

_ENV_RECORD = ("HIP_", "CUDA_VISIBLE", "ROCR_", "HSA_", "MSTAR_", "PYTORCH_", "TORCH", "MIOPEN_", "AITER_", "VLLM_",
               "SGLANG_", "NCCL_", "GLOO_", "HF_HUB_OFFLINE", "CTR_", "GPUS", "SLURM_JOB_ID", "PYTHONPATH", "TRITON_")


def _last_json_line(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {"error": text[-1500:]}


def image_record(name: str) -> dict:
    try:
        images = json.loads(IMAGES_JSON.read_text()).get("images", [])
    except (OSError, ValueError):
        return {"short_name": name, "error": f"cannot read {IMAGES_JSON}"}
    for im in images:
        if im.get("short_name") == name:
            rec = {k: im.get(k) for k in ("short_name", "sif", "source", "rocm", "torch", "torch_hip",
                                          "engine_version", "aiter", "triton", "transformers")}
            sif = Path(catalog.CONTAINERS) / im["sif"]
            if sif.is_file():
                st = sif.stat()
                rec["sif_bytes"], rec["sif_mtime"] = st.st_size, time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                                time.localtime(st.st_mtime))
            return rec
    return {"short_name": name, "error": "not in versions.json"}


def host_record(gpus: list[int]) -> dict:
    rec = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "partition": os.environ.get("SLURM_JOB_PARTITION")}
    try:
        rec["amdgpu_module"] = Path("/sys/module/amdgpu/version").read_text().strip()
    except OSError:
        rec["amdgpu_module"] = None
    try:
        rec["amd_smi_version"] = _run(["amd-smi", "version"], timeout=60).stdout.strip()[:300]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        rec["amd_smi_version"] = None
    names = {}
    want = {smi_gpu(g): g for g in gpus}
    for e in _entries(amd_smi_json("static", "--asic")):
        if e.get("gpu") in want:
            asic = e.get("asic") or {}
            names[want[e["gpu"]]] = {"market_name": asic.get("market_name"), "amd_smi_gpu": e["gpu"],
                                     "arch": asic.get("target_graphics_version"), "device_id": asic.get("device_id")}
    rec["gpus"] = names
    return rec


def collect_versions(launch: Launch, cmd: list[str], env: dict) -> dict:
    rec = {"arm": launch.arm.name, "engine": launch.arm.engine, "row": launch.row.name, "hf_id": launch.row.hf_id,
           "gpus": launch.gpus, "port": launch.port, "recipe": launch.arm.recipe, "differs": launch.arm.differs,
           "command": shlex.join(cmd), "env": {k: v for k, v in sorted(env.items()) if k.startswith(_ENV_RECORD)},
           "numa_node": launch.numa_node, "collected": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "host": host_record(launch.gpus)}
    if launch.arm.engine == "mstar":
        git = lambda *a: _run(["git", "-C", str(launch.worktree), *a]).stdout.strip()  # noqa: E731
        rec["git"] = {"worktree": str(launch.worktree), "head": git("rev-parse", "HEAD"),
                      "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "subject": git("log", "-1", "--format=%s"),
                      "dirty": [ln for ln in git("status", "--porcelain").splitlines() if ln][:40]}
        rec["config"] = str(launch.mstar_config())
        p = _run([sys.executable, "-c", _MSTAR_VERSIONS], timeout=600, env=env, cwd=launch.worktree)
        rec["stack"] = _last_json_line(p.stdout + p.stderr)
        mfile = rec["stack"].get("mstar_file", "")
        rec["serves_worktree"] = bool(mfile) and Path(mfile).resolve().is_relative_to(launch.worktree)
        if not rec["serves_worktree"]:
            raise RuntimeError(f"the server would import {mfile or 'no mstar'}, not {launch.worktree}")
    else:
        rec["image"] = image_record(launch.arm.image)
        cenv = {k: v for k, v in env.items() if k in CTR_KEEP or k in ("GPUS", "CTR_SCRATCH")}
        p = _run([str(CTR_SH), launch.arm.image, "python3", "-I", "-c", _CTR_VERSIONS], timeout=900,
                 env={**cenv, "CTR_AITER_PRECOPY": "0"})
        rec["stack"] = _last_json_line(p.stdout + p.stderr)
    return rec


# --- CLI ------------------------------------------------------------------------------------

def _launch_from(args) -> Launch:
    return Launch(arm=catalog.ARMS[args.arm], row=catalog.ROWS[args.row], gpus=gpu_list(args.gpus), port=args.port,
                  run_dir=Path(args.run_dir), worktree=Path(args.worktree), numa_node=args.numa_node,
                  lane=args.lane, startup_timeout=args.startup_timeout, force=args.force)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start")
    st.add_argument("--arm", required=True, choices=sorted(catalog.ARMS))
    st.add_argument("--row", required=True, choices=sorted(catalog.ROWS))
    st.add_argument("--gpus", required=True)
    st.add_argument("--port", type=int, required=True)
    st.add_argument("--run-dir", required=True)
    st.add_argument("--worktree", default=str(REPO))
    st.add_argument("--numa-node", type=int, default=None)
    st.add_argument("--lane", default="")
    st.add_argument("--startup-timeout", type=int, default=3600)
    st.add_argument("--force", action="store_true", help="start even if amd-smi shows the GPUs in use")
    sp = sub.add_parser("stop")
    sp.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "start":
        launch = _launch_from(args)
        (Path(args.run_dir) / "launch.json").parent.mkdir(parents=True, exist_ok=True)
        (Path(args.run_dir) / "launch.json").write_text(json.dumps(vars(args), indent=1))
        print(json.dumps(launch.start(), indent=1))
        return 0
    saved = json.loads((Path(args.run_dir) / "launch.json").read_text())
    launch = _launch_from(argparse.Namespace(**saved))
    print(json.dumps(launch.stop(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
