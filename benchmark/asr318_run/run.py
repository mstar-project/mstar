"""PR #318 (Whisper large-v3-turbo / large-v3, Qwen3-ASR 1.7B / 0.6B realtime) on latest main:
correctness gate + benchmark against vLLM and sglang-omni, one Nebius H200:8 box.

    python benchmark/asr318_run/run.py <phase> --out ~/results/pr318 [--dry-run]

Phases (each resumable and idempotent: stage markers under <out>/stages, finished cells skipped;
re-running a phase redoes only what is missing or failed):
  data             (setup) LibriSpeech test-clean -> 200-utterance set, the 605 s long-form file,
                   a ~20 min file for the Qwen3-ASR stress case
  smoke            every engine x model starts (in parallel, one GPU each, nothing timed) and
                   answers one transcription; qwen3_asr_realtime also one realtime session.
                   Exit 3 if any engine failed (SKIP_ENGINES=sgl_omni,... excludes engines)
  correctness_pre  untimed, all 8 GPUs: GPUs 0-3 one mstar server per model -> collect the
                   same-code control `mstar_b` (200 greedy transcripts + long-form, word
                   timestamps, language detection) -> stress (c=128, c=65, bad kwargs, 400-word
                   prompt, detection at T=1, 8/16 x 20-min files for Qwen3-ASR); GPUs 4-7:
                   real-checkpoint pytest test/asr/ then the reference oracle (oracle.py ref)
  round1..round3   timed, GPU 0 only, one server at a time, engine order rotated per round:
                   start -> smoke -> (round 1: collect `mstar_a` / `vllm` / `sgl_omni`; round 2:
                   `mstar_c`) -> 5 warmup waves at c=32 -> c=1 (200 req), 8 (200), 32 (400),
                   64 (400), 1 warmup wave before each later point -> long-form 605 s (mstar
                   Whisper also long_form=parallel) -> qwen3_asr_realtime: realtime sessions 1 and 8
                   (rounds 1-2) -> stop. Competitors use their tuned config from round 2 on
  gate             no server up: teacher-force every collected transcript through the fp32
                   reference (oracle.py tf, GPUs 0-3), facts, gate.py per model, gallery.
                   Exit 1 on FAIL (then round 1's numbers are discarded and nothing more is timed)
  tune             vLLM / sglang-omni knobs (2 + 1), short sweep, keep a knob only if it beats
                   the round-1 default by >= 5% (geomean RTFx at c=8, c=32)
  profile          mstar turbo and Qwen3-ASR with MSTAR_PHASE_TIMING=100 at c=32 (untimed)
  summarize        summary.md, the gate re-run with every control, the gallery
  negctl           last and alone: plant a bug (Whisper decoder MLP-input norm x1.3), serve turbo,
                   collect, restore the file (verified by hash), gate it -> must FAIL
  fake             local only: synthetic results to validate summarize and gate
  cleanup          stop every server the harness recorded (after a crash), show GPU memory

Correctness criteria (fixed before the run; full list in gate.py): the oracle is
the reference implementation (transformers Whisper generate; qwen-asr SDK for Qwen3-ASR), fp32,
TF32 off. mstar_a must agree with it, token by token under teacher forcing, as well as the other
engines and the reference's own bf16 output do (agreement >= 0.95 and >= worst yardstick - 0.02;
confident disagreements <= the most any yardstick/control shows; first-disagreement gap/rank
within the yardsticks'; end-of-text where the oracle ends it), WER <= reference bf16 + max(0.15 pt,
1.5 x the reference's bf16-vs-fp32 spread), WER at c=64 within the same tolerance of c=1, and
every request of every timed mstar cell non-empty, untruncated and HTTP 200.

Benchmark validity: a cell counts only with 0 errors and 0 bad transcripts (per-request content
check); RTFx = audio seconds / wall seconds, latency p50/p95 per request; realtime: partial and
final latency. Rounds 2-3 pool for claims; round 1 pools too where the engine config didn't change.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from mstar.cli.main import DEFAULT_CONFIGS  # noqa: E402  (the `mstar serve` default per model)

HOME = Path.home()
DATA = Path(os.environ.get("ASR_DATA", HOME / "asr318_data"))
ENVS = Path(os.environ.get("ASR_ENVS", HOME / "asr318_envs"))
MSTAR_BIN = REPO / ".venv" / "bin"
ORACLE_PY = ENVS / "oracle" / "bin" / "python"
SERVER_SH = HERE / "server.sh"

MODELS = {
    "turbo": {"key": "whisper_large_v3_turbo", "hf": "openai/whisper-large-v3-turbo", "family": "whisper",
              "engines": ["mstar", "vllm", "sgl_omni"]},
    "large": {"key": "whisper_large", "hf": "openai/whisper-large-v3", "family": "whisper",
              "engines": ["mstar", "vllm", "sgl_omni"]},
    "qasr": {"key": "qwen3_asr", "hf": "Qwen/Qwen3-ASR-1.7B", "family": "qwen",
             "engines": ["mstar", "vllm", "sgl_omni"]},
    "qasr_rt": {"key": "qwen3_asr_realtime", "hf": "Qwen/Qwen3-ASR-0.6B", "family": "qwen",
                "engines": ["mstar", "vllm"],
                # vLLM registers /v1/realtime only for a SupportsRealtime class; the checkpoint's
                # default architecture is the plain transcription one (the route answers 403)
                "vllm_args": ["--hf-overrides", '{"architectures": ["Qwen3ASRRealtimeGeneration"]}']},
}
MODEL_ORDER = ["turbo", "large", "qasr", "qasr_rt"]
CONCS = [1, 8, 32, 64]
REQS = {1: 200, 8: 200, 32: 400, 64: 400}
WARM_FIRST_WAVES, WARM_CONC = 5, 32
TIMED_GPU = 0
START_TIMEOUT = 1200
REQ_TIMEOUT = 900.0
RT_SESSIONS = {1: 16, 8: 48}       # realtime: sessions -> utterances played at real time
NEGCTL_FILE = REPO / "mstar/model/whisper/components/decoder.py"
NEGCTL_OLD = ("        hidden_states = self.final_layer_norm(hidden_states)\n"
              "        hidden_states = residual + self.fc2(F.gelu(self.fc1(hidden_states)))")
NEGCTL_NEW = ("        hidden_states = self.final_layer_norm(hidden_states) * 1.3  # NEGCTL planted bug\n"
              "        hidden_states = residual + self.fc2(F.gelu(self.fc1(hidden_states)))")
# Tuning candidates (documented knobs; at most 2 per competitor)
TUNING = {
    "vllm": [["--async-scheduling"], ["--max-num-batched-tokens", "16384"]],
    "sgl_omni": {"whisper": [["--asr.engine.cuda_graph_max_bs_prefill", "256"]],
                 "qwen": []},   # async decode (default on) is sglang-omni's documented best; nothing else to try
}
TUNE_MODELS = {"whisper": "turbo", "qwen": "qasr"}
# Planning numbers for the dry-run estimate (minutes / RTFx); re-estimated from results after round 1
EST_START = {"mstar": 3.0, "vllm": 2.5, "sgl_omni": 3.0}
EST_RTFX = {"mstar": {1: 90, 8: 300, 32: 450, 64: 450}, "vllm": {1: 130, 8: 550, 32: 1000, 64: 1100},
            "sgl_omni": {1: 110, 8: 450, 32: 900, 64: 1000}}

LOCK = threading.Lock()
FAILURES: list[str] = []
NGPU = 8   # set in main(): --ngpu, else ASR318_NGPU, else `nvidia-smi -L`, else 8


def detect_ngpu() -> int:
    if os.environ.get("ASR318_NGPU"):
        return int(os.environ["ASR318_NGPU"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=False).stdout
        n = sum(1 for ln in out.splitlines() if ln.startswith("GPU "))
        return n or 8
    except Exception:  # noqa: BLE001 - no nvidia-smi (local dry run)
        return 8


def gpu_lanes(n_jobs: int) -> list[list[int]]:
    """Job indices per GPU (one job at a time per GPU), round-robin over NGPU GPUs."""
    lanes: list[list[int]] = [[] for _ in range(min(NGPU, max(1, n_jobs)))]
    for i in range(n_jobs):
        lanes[i % len(lanes)].append(i)
    return lanes


# ----------------------------------------------------------------------------- utilities

def log(msg: str) -> None:
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fail(msg: str) -> None:
    with LOCK:
        FAILURES.append(msg)
    log(f"FAILED: {msg}")


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def rjson(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def marker(out: Path, key: str) -> Path:
    return out / "stages" / f"{key.replace('/', '__')}.done"


def done(out: Path, key: str) -> bool:
    return marker(out, key).exists()


def mark(out: Path, key: str, info: dict | None = None) -> None:
    wjson(marker(out, key), {"t": time.time(), **(info or {})})


def skipped_engines() -> set[str]:
    return {e.strip() for e in os.environ.get("SKIP_ENGINES", "").split(",") if e.strip()}


def engines_of(m: str) -> list[str]:
    return [e for e in MODELS[m]["engines"] if e not in skipped_engines()]


def req_model(m: str, e: str) -> str:
    return MODELS[m]["key"] if e == "mstar" else MODELS[m]["hf"]


def gpu_mem() -> dict[int, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=False).stdout
        return {int(a): int(b) for a, b in (ln.split(",") for ln in out.strip().splitlines())}
    except Exception:  # noqa: BLE001
        return {}


def wait_gpus_free(gpus: list[int], limit_mb: int = 2048, timeout: float = 300) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        mem = gpu_mem()
        if not mem or all(mem.get(g, 0) < limit_mb for g in gpus):
            return True
        time.sleep(5)
    log(f"GPUs {gpus} still hold memory: {gpu_mem()}")
    return False


def env_versions() -> dict:
    try:
        return _env_versions()
    except Exception as e:  # noqa: BLE001 - provenance must never stop a phase
        return {"error": repr(e)}


def _env_versions() -> dict:
    out = {}
    for name, py in (("mstar", MSTAR_BIN / "python"), ("vllm", ENVS / "vllm/bin/python"),
                     ("sgl_omni", ENVS / "sgl_omni/bin/python"), ("oracle", ORACLE_PY)):
        if not Path(py).exists():
            out[name] = "missing"
            continue
        r = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True, check=False)
        keep = ("vllm", "sglang", "torch", "flashinfer", "transformers", "qwen-asr", "triton", "mstar")
        out[name] = [ln for ln in r.stdout.splitlines() if ln.lower().startswith(keep)]
    prov = HERE / "PROVENANCE.txt"
    out["tree"] = prov.read_text() if prov.exists() else "unknown"
    out["nvidia"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.max.sm",
                                    "--format=csv,noheader"],
                                   capture_output=True, text=True, check=False).stdout.strip() \
        if shutil.which("nvidia-smi") else "no nvidia-smi"
    out["host"] = os.uname().nodename
    out["cpus"] = os.cpu_count()
    return out


# ----------------------------------------------------------------------------- data

class Utt:
    __slots__ = ("uid", "path", "duration", "reference", "audio")

    def __init__(self, uid, path, duration, reference, audio):
        self.uid, self.path, self.duration, self.reference, self.audio = uid, path, duration, reference, audio


_SETS: dict[str, list[Utt]] = {}


def load(name: str) -> list[Utt]:
    if name not in _SETS:
        from benchmark.asr_bench import load_set

        _SETS[name] = [Utt(u.uid, u.path, u.duration, u.reference, u.audio) for u in load_set(DATA / name)]
    return _SETS[name]


def asset_clips() -> dict[str, Path]:
    clips = {p.stem: p for p in sorted((HERE / "assets").glob("*.wav"))}
    for u in load("test_clean_200")[:8]:
        clips[u.uid] = u.path
    return clips


_NORM = None


def normalize(s: str) -> str:
    global _NORM
    if _NORM is None:
        from benchmark.asr_bench import make_normalizer

        _NORM = make_normalizer("whisper")
    return _NORM(s or "")


def wer(refs: list[str], hyps: list[str]) -> float | None:
    import jiwer

    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return None
    return float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))


def utt_wer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0
    return float(jiwer.wer(r, h)) if h else 1.0


def judge(rec: dict, ref: str) -> str | None:
    """Per-request content check: why the transcript is bad, or None."""
    if rec.get("error") or rec.get("status") != 200:
        return f"error: {rec.get('error') or rec.get('status')}"
    text = rec.get("text") or ""
    rw, hw = len(normalize(ref).split()), len(normalize(text).split())
    if rw and not hw:
        return "empty"
    if rw >= 5 and hw < 0.5 * rw:
        return f"truncated ({hw}/{rw} words)"
    if rw >= 5 and utt_wer(ref, text) > 0.5:
        return f"garbage (utt WER {utt_wer(ref, text):.2f})"
    return None


def phase_data(a) -> int:
    """Build the inputs once (called from sky setup). Tries openslr mirrors for the tarball."""
    import urllib.request
    import wave

    import numpy as np

    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tarball = raw / "test-clean.tar.gz"
    if not (DATA / "manifest.json").exists():
        if not tarball.exists() or tarball.stat().st_size < 300_000_000:
            for url in ("https://www.openslr.org/resources/12/test-clean.tar.gz",
                        "https://us.openslr.org/resources/12/test-clean.tar.gz",
                        "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
                        "https://openslr.elda.org/resources/12/test-clean.tar.gz"):
                try:
                    log(f"downloading {url}")
                    urllib.request.urlretrieve(url, tarball.with_suffix(".part"))  # noqa: S310
                    tarball.with_suffix(".part").replace(tarball)
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"  failed: {e!r}")
        subprocess.run([str(MSTAR_BIN / "python"), "-m", "benchmark.asr_data", "--out", str(DATA),
                        "--raw-dir", str(raw)], cwd=REPO, check=True)
    # ~20 min file for the Qwen3-ASR long-file stress (the 605 s file twice, trimmed to 1190 s)
    sdir = DATA / "stress20"
    if not (sdir / "references.tsv").exists():
        lf = load("long_form")[0]
        with wave.open(str(lf.path)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        both = np.concatenate([pcm, np.zeros(8000, np.int16), pcm])[: 1190 * 16000]
        sdir.mkdir(parents=True, exist_ok=True)
        with wave.open(str(sdir / "stress20.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(both.tobytes())
        # the reference covers both copies; the tail cut makes WER here approximate (informational only)
        (sdir / "references.tsv").write_text(f"stress20\t{lf.reference} {lf.reference}\n")
    short = load("test_clean_200")
    log(f"data OK: {len(short)} utterances, {sum(u.duration for u in short):.0f} s; long-form "
        f"{load('long_form')[0].duration:.0f} s; stress {load('stress20')[0].duration:.0f} s")
    return 0


# ----------------------------------------------------------------------------- servers

class Server:
    def __init__(self, out: Path, m: str, e: str, gpu: int, port: int, tag: str, timed: bool = True,
                 extra: list[str] | None = None, env_extra: dict | None = None, cap: int | None = None):
        """``cap``: the most concurrent requests the caller drives. An mstar config capping
        max_concurrent_requests below it is served from a raised copy, Whisper's excepted (``bench_config``)."""
        self.out, self.m, self.e, self.gpu, self.port = out, m, e, gpu, port
        self.name = f"{m}_{e}_{tag}"
        self.timed, self.extra, self.env_extra = timed, list(extra or []), dict(env_extra or {})
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = out / "servers" / f"{self.name}.log"
        self.variant = None
        self.started = False
        # the mstar server's max_concurrent_requests (None: uncapped, or not mstar), for the results
        self.serve_args, self.max_concurrent = (self.bench_config(cap) if e == "mstar"
                                                else (self.extra, None))

    def bench_config(self, cap: int | None) -> tuple[list[str], int | None]:
        """``self.extra`` with the served config raised to ``cap`` concurrent requests, and the cap served.
        Past the config's cap, requests queue in the conductor, so a cell above it would time the queue,
        not the engine (Qwen3-ASR ships at 12). The KV page count stays: benchmark utterances are short.
        Whisper is served as configured: its cap is what its decoder caches hold, and the model refuses
        a config raised past it."""
        args = list(self.extra)
        at, prefix = None, ""  # where the config path sits in args: `--config path` or `--config=path`
        for i, arg in enumerate(args):
            if arg == "--config" and i + 1 < len(args):
                at, prefix = i + 1, ""
            elif arg.startswith("--config="):
                at, prefix = i, "--config="
        src = (REPO / "configs" / DEFAULT_CONFIGS[MODELS[self.m]["key"]] if at is None
               else REPO / args[at].removeprefix(prefix))
        text = src.read_text()
        found = re.search(r"^max_concurrent_requests:[ \t]*(\d+)", text, re.M)
        served = int(found.group(1)) if found else None
        if cap is None or served is None or served >= cap or MODELS[self.m]["family"] == "whisper":
            return args, served
        path = self.out / "servers" / f"{self.name}.config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# benchmark copy of {src}: max_concurrent_requests {served} -> {cap}\n"
                        + text[:found.start(1)] + str(cap) + text[found.end(1):])
        if at is None:
            args += ["--config", str(path)]
        else:
            args[at] = prefix + str(path)
        return args, cap

    def variants(self) -> list[tuple[str, list[str], dict]]:
        m, e = MODELS[self.m], self.e
        env = {"CUDA_VISIBLE_DEVICES": str(self.gpu), "HF_HUB_OFFLINE": "1", "PYTHONFAULTHANDLER": "1"}
        ts = f"{int(time.time())}_{os.getpid()}"
        if os.environ.get("ASR318_SELFTEST") == "1":  # CPU plumbing test: selftest_server.py for every engine
            return [("selftest", [sys.executable, str(HERE / "selftest_server.py"), "--port", str(self.port),
                                  "--data", str(DATA)], env)]
        if e == "mstar":
            env.update({"MSTAR_SHM_ARENA": "1",
                        "PATH": f"{MSTAR_BIN}:{os.environ.get('PATH', '')}"})
            env.pop("CUDA_VISIBLE_DEVICES")  # --gpus sets it
            cmd = [str(MSTAR_BIN / "mstar"), "serve", m["key"], "--gpus", str(self.gpu), "--port", str(self.port),
                   "--socket-path-prefix", f"/tmp/mstar_{self.name}_{ts}/",
                   "--upload-dir", f"/tmp/mstar_up_{self.name}_{ts}/"] + self.serve_args
            strict = {**env, "MSTAR_REQUIRE_CUDA_GRAPHS": "1"}
            return [("graphs_required", cmd, strict), ("graphs_not_required", cmd, env)]
        if e == "vllm":
            vb = ENVS / "vllm" / "bin"
            env.update({"PATH": f"{vb}:{os.environ.get('PATH', '')}", "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "200",
                        "VLLM_MAX_AUDIO_DECODE_DURATION_S": "1300"})
            base = [str(vb / "vllm"), "serve", m["hf"], "--port", str(self.port), "--dtype", "bfloat16",
                    "--max-num-seqs", "64", "--gpu-memory-utilization", "0.6"] + m.get("vllm_args", [])
            matched = ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"]
            return [("matched", base + matched + self.extra, env), ("minimal", base + self.extra, env)]
        sb = ENVS / "sgl_omni" / "bin"
        env.update({"PATH": f"{sb}:{os.environ.get('PATH', '')}"})
        if Path("/usr/local/cuda-13.2/bin/nvcc").exists():  # SGLang's DeepGEMM JIT needs nvcc >= 12.9
            env["CUDA_HOME"] = "/usr/local/cuda-13.2"
        base = [str(sb / "sgl-omni"), "serve", "--model-path", m["hf"], "--model-name", m["hf"],
                "--port", str(self.port)]
        return [("mem0.5", base + ["--mem-fraction-static", "0.5"] + self.extra, env),
                ("minimal", base + self.extra, env)]

    def config_id(self) -> str:
        cid = "default" if not self.extra else "tuned:" + " ".join(self.extra)
        return cid if self.max_concurrent is None else f"{cid}, max_concurrent_requests={self.max_concurrent}"

    def start(self) -> bool:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop()  # a same-named leftover from a crashed earlier attempt
        if not wait_gpus_free([self.gpu]):
            fail(f"{self.name}: GPU {self.gpu} not free")
            return False
        for vname, cmd, env in self.variants():
            full_env = {**os.environ, **env, **self.env_extra,
                        "SERVER_STATE_DIR": str(self.out / "servers" / "state")}
            log(f"start {self.name} [{vname}] on GPU {self.gpu}:{self.port}: {' '.join(cmd)}")
            t0 = time.time()
            r = subprocess.run(["bash", str(SERVER_SH), "start", self.name, str(self.port), "--timeout",
                                str(START_TIMEOUT), "--log", str(self.log_path), "--"] + cmd,
                               env=full_env, cwd=REPO, capture_output=True, text=True, check=False)
            if r.returncode == 0:
                self.variant, self.started = vname, True
                self.start_s = time.time() - t0
                log(f"{self.name} ready in {self.start_s:.0f}s [{vname}]")
                with contextlib.suppress(OSError):
                    shutil.copy(self.log_path, self.log_path.with_suffix(f".{vname}.startlog"))
                return True
            log(f"{self.name} [{vname}] failed to start: {r.stderr[-1500:]}")
            with contextlib.suppress(OSError):
                shutil.copy(self.log_path, self.log_path.with_suffix(f".{vname}.failed.log"))
            self.stop()
        fail(f"{self.name}: no variant started (see {self.log_path})")
        return False

    def stop(self) -> None:
        env = {**os.environ, "SERVER_STATE_DIR": str(self.out / "servers" / "state")}
        subprocess.run(["bash", str(SERVER_SH), "stop", self.name], env=env, capture_output=True, text=True,
                       check=False)
        self.started = False

    def healthy(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def captured_ok(self) -> dict:
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            text = ""
        return {"failed_capture_lines": text.count("Failed to capture"), "variant": self.variant}


@contextlib.contextmanager
def serving(srv: Server):
    try:
        yield srv.start()
    finally:
        srv.stop()


# ----------------------------------------------------------------------------- client

async def _post(session, url: str, model: str, audio: bytes, filename: str, fields: dict) -> dict:
    import aiohttp

    form = aiohttp.FormData()
    form.add_field("model", model)
    for k, v in fields.items():
        if isinstance(v, list):
            for x in v:
                form.add_field(f"{k}[]", str(x))
        else:
            form.add_field(k, str(v))
    form.add_field("file", audio, filename=filename, content_type="audio/wav")
    t0 = time.perf_counter()
    try:
        async with session.post(f"{url}/v1/audio/transcriptions", data=form) as resp:
            body = await resp.read()
            dt = time.perf_counter() - t0
            if resp.status != 200:
                return {"status": resp.status, "seconds": dt, "error": body[:300].decode(errors="replace")}
            try:
                js = json.loads(body)
            except ValueError:
                js = {"text": body.decode(errors="replace")}
            return {"status": 200, "seconds": dt, "text": (js.get("text") or "").strip(), "json": js}
    except Exception as e:  # noqa: BLE001
        return {"status": None, "seconds": time.perf_counter() - t0, "error": repr(e)[:300]}


def run_requests(url: str, model: str, items: list[Utt], conc: int, fields: dict,
                 timeout: float = REQ_TIMEOUT, keep_json: bool = False) -> tuple[list[dict], float]:
    async def main():
        import aiohttp

        sem = asyncio.Semaphore(conc)
        recs: list[dict | None] = [None] * len(items)

        async def one(i, s):
            async with sem:
                r = await _post(s, url, model, items[i].audio, items[i].path.name, fields)
                r["uid"] = items[i].uid
                if not keep_json:
                    r.pop("json", None)
                recs[i] = r

        conn = aiohttp.TCPConnector(limit=max(64, conc + 8))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), connector=conn) as s:
            t0 = time.perf_counter()
            await asyncio.gather(*(one(i, s) for i in range(len(items))))
            return recs, time.perf_counter() - t0

    return asyncio.run(main())


def json_fields(**kw) -> dict:
    return {"language": "en", "response_format": "json", "temperature": 0, **kw}


def repeat(items: list[Utt], n: int) -> list[Utt]:
    return (items * (n // len(items) + 1))[:n]


_REF_BAD: dict[str, dict[str, str]] = {}


def reference_bad(out: Path, m: str) -> dict[str, str]:
    """uid -> verdict for utterances where the reference implementation's own transcript (oracle.py ref,
    bf16 or fp32) already fails the content check: proper names and short clips the check can't judge
    (e.g. 'BROTHER MAC ARDLE BROTHER KEOGH' -> 'Brother McArdle. Brother Keeof.', utt WER 0.60 for
    transformers, vLLM, sglang-omni and mstar alike). A cell's content verdict on such an utterance is
    recorded but does not invalidate the cell; errors and empty replies always do."""
    if m not in _REF_BAD:
        rec = rjson(out / "oracle" / m / "ref.json", {})
        refs = {u.uid: u.reference for u in load("test_clean_200")}
        bad: dict[str, str] = {}
        for prec in ("bf16", "fp32"):
            for uid, text in (rec.get("short", {}).get(prec) or {}).items():
                if uid in refs:
                    why = judge({"status": 200, "text": text}, refs[uid])
                    if why:
                        bad.setdefault(uid, f"reference {prec}: {why}")
        if not rec.get("short"):
            log(f"WARNING: no reference transcripts for {m}; content checks are uncalibrated")
        _REF_BAD[m] = bad
    return _REF_BAD[m]


def rejudge_cells(out: Path, r: int) -> None:
    """Re-check this round's finished cells that were INVALID only through content verdicts the
    reference implementation shares (see reference_bad), so a calibration fix needn't re-time them.
    A cell that becomes valid is rewritten and gets a new results.jsonl row (summarize keeps the last)."""
    for path in sorted((out / "cells").glob(f"*/*/r{r}/*.json")):
        res = rjson(path, {})
        if res.get("valid") or res.get("errors") or "records" not in res:
            continue
        m = path.parts[-4]
        ref_bad = reference_bad(out, m)
        refs = {u.uid: u.reference for u in load("test_clean_200")}
        if path.stem.startswith("long"):
            continue  # long-form references aren't in the short set; leave those cells as judged
        bad, ref_also_bad = [], []
        for rec in res["records"]:
            why = rec.get("bad")
            if why and why.startswith(("garbage", "truncated")) and rec["uid"] in ref_bad:
                ref_also_bad.append((rec["uid"], why, ref_bad[rec["uid"]]))
            elif why:
                bad.append((rec["uid"], why))
        if bad:
            continue
        res.update({"bad": 0, "bad_examples": [], "ref_also_bad": sorted(set(ref_also_bad))[:5], "valid": True,
                    "rejudged": "content verdicts shared by the reference implementation"})
        wjson(path, res)
        row = {k: v for k, v in res.items() if k != "records"}
        row.update({"kind": "short", "ts": time.time()})
        append_jsonl(out / "results.jsonl", row)
        log(f"rejudged {path.relative_to(out)}: valid ({len(ref_also_bad)} verdicts shared by the reference)")
    # a server whose every cell is now valid is done for this round (what run_engine_round marks itself)
    for cdir in sorted((out / "cells").glob(f"*/*/r{r}")):
        m, e = cdir.parts[-3], cdir.parts[-2]
        key = f"round{r}/{m}/{e}"
        files = list(cdir.glob("*.json"))
        have = {f.stem for f in files}
        need = {f"c{c}" for c in CONCS} | {"long_seq"}
        if not done(out, key) and need <= have and all(rjson(f, {}).get("valid") for f in files):
            mark(out, key)
            log(f"{key}: all cells valid after rejudging; marked done")


def cell(srv: Server, m: str, e: str, items: list[Utt], conc: int, fields: dict) -> dict:
    l0 = os.getloadavg()[0]
    recs, wall = run_requests(srv.url, req_model(m, e), items, conc, fields)
    l1 = os.getloadavg()[0]
    refs = {u.uid: u.reference for u in items}
    durs = {u.uid: u.duration for u in items}
    ref_bad = reference_bad(srv.out, m)
    bad, ref_also_bad = [], []
    for r in recs:
        why = judge(r, refs[r["uid"]])
        r["bad"] = why
        if why and why.startswith(("garbage", "truncated")) and r["uid"] in ref_bad:
            ref_also_bad.append((r["uid"], why, ref_bad[r["uid"]]))
        elif why:
            bad.append((r["uid"], why))
    ok = [r for r in recs if not r.get("error") and r.get("status") == 200]
    lat = sorted(r["seconds"] for r in ok)
    audio = sum(durs[r["uid"]] for r in ok)
    w = wer([refs[r["uid"]] for r in ok], [r.get("text", "") for r in ok]) if ok else None
    errors = sum(1 for r in recs if r.get("error") or r.get("status") != 200)
    return {"n": len(recs), "conc": conc, "wall_s": wall, "audio_s": audio, "rtfx": audio / wall if wall else 0,
            "p50_ms": 1000 * statistics.median(lat) if lat else None,
            "p95_ms": 1000 * lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))] if lat else None,
            "wer": w, "errors": errors, "bad": len(bad), "bad_examples": bad[:5],
            "ref_also_bad": sorted(set(ref_also_bad))[:5],
            "valid": errors == 0 and not bad, "load_before": l0, "load_after": l1, "records": recs}


def smoke_request(srv: Server, m: str, e: str) -> tuple[bool, str]:
    u = load("test_clean_200")[0]
    recs, _ = run_requests(srv.url, req_model(m, e), [u], 1, json_fields(), timeout=600)
    why = judge(recs[0], u.reference)
    return why is None, f"{recs[0].get('text', '')[:80]!r} {why or ''}"


# ----------------------------------------------------------------------------- realtime

def realtime_sessions(srv: Server, m: str, e: str, sessions: int, n_utts: int, speed: float = 1.0) -> dict:
    from benchmark.asr_realtime_bench import run_session

    items = load("test_clean_200")[:n_utts]
    dialect = "openai" if e == "mstar" else "vllm"
    ws = srv.url.replace("http://", "ws://")

    async def main():
        import aiohttp

        sem = asyncio.Semaphore(sessions)

        async def one(u, s):
            async with sem:
                try:
                    return await asyncio.wait_for(run_session(s, ws, u, speed, "en", None, dialect, req_model(m, e)),
                                                  timeout=u.duration / speed + 120)
                except Exception as ex:  # noqa: BLE001
                    return {"uid": u.uid, "text": "", "error": repr(ex)[:300], "partial_latencies": [],
                            "delta_latencies": [], "final_latency": None}

        conn = aiohttp.TCPConnector(limit=max(64, sessions + 8))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), connector=conn) as s:
            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(u, s) for u in items))
            return res, time.perf_counter() - t0

    res, wall = asyncio.run(main())

    def pct(v, q):
        v = sorted(v)
        return 1000 * v[min(len(v) - 1, round(q * (len(v) - 1)))] if v else None

    parts = [x for r in res for x in r.get("partial_latencies", [])]
    deltas = [x for r in res for x in r.get("delta_latencies", [])]
    finals = [r["final_latency"] for r in res if r.get("final_latency") is not None]
    refs = {u.uid: u.reference for u in items}
    okr = [r for r in res if r.get("text")]
    return {"sessions": sessions, "n": len(res), "wall_s": wall, "errors": sum(1 for r in res if r.get("error")),
            "empty": sum(1 for r in res if not r.get("text")),
            "wer": wer([refs[r["uid"]] for r in okr], [r["text"] for r in okr]) if okr else None,
            "partial_p50_ms": pct(parts, 0.5), "partial_p95_ms": pct(parts, 0.95), "num_partials": len(parts),
            "delta_p50_ms": pct(deltas, 0.5), "delta_p95_ms": pct(deltas, 0.95), "num_deltas": len(deltas),
            "final_p50_ms": pct(finals, 0.5), "final_p95_ms": pct(finals, 0.95),
            "texts": {r["uid"]: r.get("text", "") for r in res},
            "errors_examples": [r.get("error") for r in res if r.get("error")][:3]}


# ----------------------------------------------------------------------------- collect (parity answers + extras)

def collect(out: Path, srv: Server, m: str, e: str, tag: str, extras: bool = True) -> None:
    fam = MODELS[m]["family"]
    cdir = out / "collect" / m
    path = cdir / f"{tag}.json"
    if not path.exists():
        items = load("test_clean_200")
        recs, wall = run_requests(srv.url, req_model(m, e), items, 1, json_fields())
        rows = [{"uid": r["uid"], "text": r.get("text", ""), "error": r.get("error"), "seconds": r["seconds"]}
                for r in recs]
        wjson(path, rows)
        w = wer([u.reference for u in items], [r["text"] for r in rows])
        log(f"collect {m}/{tag}: 200 transcripts in {wall:.0f}s, WER {100 * (w or 0):.2f}%, "
            f"{sum(1 for r in rows if r['error'])} errors")
    xpath = cdir / f"{tag}_extras.json"
    if not extras or xpath.exists():
        return
    x: dict = {"engine": e}
    model = req_model(m, e)
    try:
        clips = asset_clips()
        dets = {}
        for name, p in clips.items():
            u = Utt(name, p, 0, "", p.read_bytes())
            r = run_requests(srv.url, model, [u], 1, {"response_format": "verbose_json", "temperature": 0},
                             keep_json=True, timeout=300)[0][0]
            if r.get("status") != 200:  # an engine without verbose_json: plain json, language unknown
                r = run_requests(srv.url, model, [u], 1, {"response_format": "json", "temperature": 0},
                                 keep_json=True, timeout=300)[0][0]
            js = r.get("json") or {}
            dets[name] = {"status": r.get("status"), "language": js.get("language"), "text": r.get("text", ""),
                          "error": r.get("error")}
        x["detect"] = dets
        lf = load("long_form")[0]
        modes = [("seq", {})] + ([("par", {"long_form": "parallel"})] if e == "mstar" and fam == "whisper" else [])
        x["long_form"] = {}
        for mode, extra in modes:
            r = run_requests(srv.url, model, [lf], 1, {"language": "en", "response_format": "verbose_json",
                                                       "temperature": 0, **extra}, keep_json=True)[0][0]
            if r.get("status") != 200:
                r = run_requests(srv.url, model, [lf], 1, json_fields(**extra), keep_json=True)[0][0]
            js = r.get("json") or {}
            x["long_form"][mode] = {"status": r.get("status"), "text": r.get("text", ""), "seconds": r["seconds"],
                                    "segments": [{k: s.get(k) for k in ("start", "end", "text")}
                                                 for s in (js.get("segments") or [])],
                                    "duration": lf.duration, "error": r.get("error")}
        if fam == "whisper":
            words = {}
            for u in load("test_clean_200")[:8]:
                r = run_requests(srv.url, model, [u], 1, {"language": "en", "response_format": "verbose_json",
                                                          "temperature": 0, "timestamp_granularities": ["word"]},
                                 keep_json=True, timeout=300)[0][0]
                js = r.get("json") or {}
                words[u.uid] = {"status": r.get("status"), "words": js.get("words") or [], "text": r.get("text", ""),
                                "error": r.get("error")}
            x["words"] = words
        if m == "qasr_rt":
            x["realtime"] = realtime_sessions(srv, m, e, 1, 3)
    except Exception as ex:  # noqa: BLE001 - kept aside so a rerun collects them again
        x["error"] = traceback.format_exc()[-2000:]
        fail(f"collect extras {m}/{tag}: {ex!r}")
        wjson(xpath.with_suffix(".failed"), x)
        return
    wjson(xpath, x)
    log(f"collect {m}/{tag}: extras written")


# ----------------------------------------------------------------------------- stress (mstar only, untimed)

def stress(out: Path, srv: Server, m: str) -> None:
    path = out / "stress" / f"{m}.json"
    s = rjson(path, {})
    fam = MODELS[m]["family"]
    model = req_model(m, "mstar")
    items = load("test_clean_200")

    def load_case(conc, n):
        c = cell(srv, m, "mstar", repeat(items, n), conc, json_fields())
        c.pop("records")
        return c

    steps = [("c128", lambda: load_case(128, 400)), ("c65", lambda: load_case(65, 130))]

    def bad_kwargs():
        r = run_requests(srv.url, model, items[:1], 1, json_fields(max_output_tokens="abc"), timeout=120)[0][0]
        time.sleep(2)
        ok_after = smoke_request(srv, m, "mstar")[0]
        return {"status": r.get("status"), "seconds": r["seconds"], "error": (r.get("error") or "")[:200],
                "healthy_after": srv.healthy() and ok_after}

    def long_prompt():
        words = " ".join(load("long_form")[0].reference.split()[:400])
        r = run_requests(srv.url, model, items[:1], 1, json_fields(prompt=words), timeout=300)[0][0]
        time.sleep(2)
        ok_after = smoke_request(srv, m, "mstar")[0]
        return {"status": r.get("status"), "seconds": r["seconds"], "text": (r.get("text") or "")[:120],
                "error": (r.get("error") or "")[:200], "healthy_after": srv.healthy() and ok_after}

    def temp1():
        u = items[1]
        recs, _ = run_requests(srv.url, model, [u] * 10, 4, {"response_format": "verbose_json", "temperature": 1},
                               keep_json=True, timeout=300)
        langs = [((r.get("json") or {}).get("language") or "").lower() for r in recs]
        return {"n": len(recs), "en": sum(lang in ("en", "english") for lang in langs), "languages": langs}

    steps += [("bad_kwargs", bad_kwargs)] + ([("long_prompt", long_prompt)] if fam == "whisper" else [])
    steps += [("temp1", temp1)]
    if m == "qasr":
        big = load("stress20")[0]

        def files(n):
            def go():
                recs, wall = run_requests(srv.url, model, [big] * n, n, json_fields(), timeout=600)
                return {"n": n, "completed": sum(1 for r in recs if r.get("status") == 200 and r.get("text")),
                        "wall_s": wall,
                        "wer": [None if not r.get("text") else round(utt_wer(big.reference, r["text"]), 4)
                                for r in recs],
                        "errors": [r.get("error") for r in recs if r.get("error")][:3]}
            return go

        steps += [("qwen_20min_8", files(8)), ("qwen_20min_16", files(16))]
    for name, fn in steps:
        if name in s and "error" not in s[name]:
            continue
        if name == "qwen_20min_16" and NGPU < 8:
            s[name] = {"skipped": f"time budget on a {NGPU}-GPU box (informational item)"}
            continue
        if name == "qwen_20min_16" and s.get("qwen_20min_8", {}).get("completed") != 8:
            s[name] = {"skipped": "8 concurrent did not all complete"}
            continue
        if not srv.healthy():
            s[name] = {"error": "server not healthy before this step"}
            fail(f"stress {m}/{name}: server unhealthy")
            break
        try:
            t0 = time.time()
            s[name] = fn()
            log(f"stress {m}/{name} ({time.time() - t0:.0f}s): {json.dumps(s[name], default=str)[:300]}")
        except Exception as ex:  # noqa: BLE001
            s[name] = {"error": repr(ex)[:300]}
            fail(f"stress {m}/{name}: {ex!r}")
        wjson(path, s)
    wjson(path, s)


# ----------------------------------------------------------------------------- phases: smoke, correctness_pre

def guarded(fn, *args) -> None:
    """Run one job; a crash is recorded as a failure and the next job still runs."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        fail(f"{getattr(fn, '__name__', fn)}{args} crashed: {e!r}\n{traceback.format_exc()[-1500:]}")


def parallel(jobs: list, workers: int):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn) for fn in jobs]
        for f in futs:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                fail(f"job crashed: {e!r}\n{traceback.format_exc()[-1500:]}")


def phase_smoke(a) -> int:
    out = a.out
    pairs = [(m, e) for m in MODEL_ORDER for e in engines_of(m)]
    if NGPU < 4:
        # whisper_large shares turbo's class, adapter and engine paths; its first start is round 1's
        pairs = [(m, e) for m, e in pairs if m != "large"]
    if a.dry_run:
        waves = math.ceil(len(pairs) / NGPU)
        print(f"smoke: {len(pairs)} servers in {waves} wave(s) of {NGPU} (one GPU each, untimed): {pairs}")
        print(f"  estimate ~{3 + 3.5 * waves:.0f} min (first starts are cold)")
        return 0
    wjson(out / "provenance.json", env_versions())
    results: dict = rjson(out / "smoke.json", {})

    def job(m, e, gpu):
        key = f"{m}/{e}"
        if results.get(key, {}).get("ok"):
            return
        srv = Server(out, m, e, gpu, 8100 + gpu, "smoke")
        with serving(srv) as up:
            if not up:
                results[key] = {"ok": False, "why": "start failed"}
                return
            ok, detail = smoke_request(srv, m, e)
            info = {"ok": ok, "detail": detail, "start_s": getattr(srv, "start_s", None), **srv.captured_ok()}
            if ok and m == "qasr_rt":
                rt = realtime_sessions(srv, m, e, 1, 1, speed=2.0)
                info["realtime"] = {k: rt[k] for k in ("errors", "empty", "wer", "num_partials", "num_deltas")}
                info["ok"] = rt["errors"] == 0 and rt["empty"] == 0
            results[key] = info
        log(f"smoke {key}: {'OK' if results[key]['ok'] else 'FAILED'} {results[key]}")

    for wave in range(0, len(pairs), NGPU):
        chunk = pairs[wave:wave + NGPU]
        parallel([lambda m=m, e=e, g=i: job(m, e, g) for i, (m, e) in enumerate(chunk)], NGPU)
        wjson(out / "smoke.json", results)
    bad = [k for k, v in results.items() if not v.get("ok")]
    print("\n== smoke: " + ("all engines OK" if not bad else f"FAILED: {bad}"))
    for k, v in results.items():
        print(f"  {k}: {'OK' if v.get('ok') else 'FAIL'} {v}")
    if not bad:
        mark(out, "smoke")
    return 3 if bad else 0


def pytest_job(out: Path, m: str, gpu: int) -> None:
    logp = out / "pytest" / f"{m}.log"
    # a finished session (pass or fail) is not rerun; its verdict goes to the gate's pytest line
    if logp.exists() and re.search(r"^=+ .*\b(passed|failed)\b.* in [\d.]+s", logp.read_text(), re.M):
        return
    hf, fam = MODELS[m]["hf"], MODELS[m]["family"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "MSTAR_ASR_DATA": str(DATA),
           "PATH": f"{MSTAR_BIN}:{os.environ.get('PATH', '')}", "PYTHONFAULTHANDLER": "1"}
    if fam == "whisper":
        env["MSTAR_WHISPER_REPO"] = hf
        args = ["test/asr/test_whisper_parity.py"]
    else:
        env["MSTAR_QWEN_ASR_REPO"] = hf
        # the SDK half needs qwen-asr's transformers 4.57.6; oracle.py runs the SDK in its own venv instead
        args = ["test/asr/test_qwen3_asr_parity.py", "-k", "not reference_sdk"]
    logp.parent.mkdir(parents=True, exist_ok=True)
    with logp.open("w") as f:
        r = subprocess.run([str(MSTAR_BIN / "python"), "-m", "pytest", "-v", "-rA", "-p", "no:cacheprovider",
                            "--basetemp", f"/tmp/pt318_{m}"] + args, cwd=REPO, env=env, stdout=f,
                           stderr=subprocess.STDOUT, timeout=1800, check=False)
    log(f"pytest {m}: exit {r.returncode}: {logp.read_text().strip().splitlines()[-1:]}")
    if r.returncode == 1:
        # tests ran and some failed: a finding about the PR, reported by the gate's pytest line
        # (it stays FAIL there); the benchmark goes on. Anything else (crash, collection error,
        # timeout) is a broken run.
        log(f"FINDING: pytest {m}: tests failed (see {logp}); gate reports it")
    elif r.returncode:
        fail(f"pytest {m} exit {r.returncode}")


def oracle_job(out: Path, m: str, gpu: int, cmd: str) -> None:
    hf, fam = MODELS[m]["hf"], MODELS[m]["family"]
    logp = out / "oracle" / m / f"{cmd}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "NVIDIA_TF32_OVERRIDE": "0",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": ""}
    with logp.open("a") as f:
        r = subprocess.run([str(ORACLE_PY), "-I", str(HERE / "oracle.py"), cmd, "--model", m, "--hf", hf,
                            "--family", fam, "--data", str(DATA), "--out", str(out), "--device", "cuda:0",
                            "--assets", str(HERE / "assets")], env=env, stdout=f, stderr=subprocess.STDOUT,
                           cwd="/tmp", timeout=3600, check=False)
    log(f"oracle {cmd} {m}: exit {r.returncode}")
    if r.returncode:
        fail(f"oracle {cmd} {m} exit {r.returncode} (see {logp})")


def phase_correctness_pre(a) -> int:
    out = a.out
    if a.dry_run:
        print(f"correctness_pre (untimed, {NGPU} GPU(s)):")
        if NGPU >= 8:
            for i, m in enumerate(MODEL_ORDER):
                print(f"  GPU {i}: mstar {MODELS[m]['key']} -> collect mstar_b (200 utts + extras) -> stress")
                print(f"  GPU {4 + i}: pytest test/asr ({MODELS[m]['hf']}) -> oracle.py ref (bf16 + fp32 + extras)")
            print("  estimate ~20 min (8 + 16 concurrent 20-min Qwen3-ASR files are the long pole, 600 s cap)")
        elif NGPU >= 2:
            print("  GPU 0, one model after another: mstar -> collect mstar_b + extras -> stress (16 x 20-min dropped)")
            print("  GPU 1, alongside: pytest test/asr per model -> oracle.py ref per model")
            print("  estimate ~28 min")
        else:
            print("  GPU 0, one step after another: pytest test/asr per model -> oracle.py ref per model")
            print("  (the same-code control mstar_b is collected on round 2's fresh server, and stress runs on")
            print("   round 1's mstar server after its timed cells: no extra server starts on a 1-GPU box)")
            print("  estimate ~28 min")
        return 0

    def server_side(m, i, gpu):
        if done(out, f"cpre/{m}/server"):
            return
        if "mstar" not in engines_of(m):
            return
        srv = Server(out, m, "mstar", gpu, 8200 + i, "cpre")
        with serving(srv) as up:
            if not up:
                return
            collect(out, srv, m, "mstar", "mstar_b")
            stress(out, srv, m)
        st = rjson(out / "stress" / f"{m}.json", {})
        if ((out / "collect" / m / "mstar_b_extras.json").exists() and st
                and not any(isinstance(v, dict) and "error" in v for v in st.values())):
            mark(out, f"cpre/{m}/server")

    def ref_side(m, gpu):
        if not done(out, f"cpre/{m}/pytest"):
            pytest_job(out, m, gpu)
            mark(out, f"cpre/{m}/pytest")
        rec = rjson(out / "oracle" / m / "ref.json", {})
        if not ({"bf16", "fp32"} <= set(rec.get("short", {})) and "extras" in rec):
            oracle_job(out, m, gpu, "ref")

    if not wait_gpus_free(list(range(NGPU)), timeout=600):
        print("GPUs hold memory; run `cleanup` first")
        return 2
    if NGPU >= 8:
        jobs = [lambda m=m, i=i: server_side(m, i, i) for i, m in enumerate(MODEL_ORDER)]
        jobs += [lambda m=m, i=i: ref_side(m, 4 + i) for i, m in enumerate(MODEL_ORDER)]
        parallel(jobs, 8)
    elif NGPU >= 2:
        # one lane per GPU, each lane one job at a time: servers on GPU 0, reference work on GPU 1
        lane0 = lambda: [guarded(server_side, m, i, 0) for i, m in enumerate(MODEL_ORDER)]  # noqa: E731
        lane1 = lambda: [guarded(ref_side, m, 1) for m in MODEL_ORDER]  # noqa: E731
        parallel([lane0, lane1], 2)
    else:
        for m in MODEL_ORDER:
            guarded(ref_side, m, 0)
    return 3 if FAILURES else 0


# ----------------------------------------------------------------------------- timed rounds

def tuned_extra(out: Path, m: str, e: str) -> list[str]:
    t = rjson(out / "tuned.json", {})
    return list(t.get(e, {}).get(MODELS[m]["family"], []) or [])


def run_engine_round(out: Path, m: str, e: str, r: int) -> None:
    key = f"round{r}/{m}/{e}"
    if done(out, key):
        log(f"{key}: done, skipping")
        return
    extra = tuned_extra(out, m, e) if r > 1 else []
    srv = Server(out, m, e, TIMED_GPU, 8100, f"r{r}", extra=extra, cap=max(CONCS))
    cdir = out / "cells" / m / e / f"r{r}"
    ok_all = True
    with serving(srv) as up:
        if not up:
            append_jsonl(out / "results.jsonl", {"model": m, "engine": e, "round": r, "kind": "start_failed",
                                                 "valid": False, "config": srv.config_id(), "ts": time.time()})
            return
        ok, detail = smoke_request(srv, m, e)
        log(f"{key} smoke: {'OK' if ok else 'FAILED'} {detail}")
        if not ok:
            fail(f"{key}: smoke failed {detail}")
            return
        if r == 1:
            collect(out, srv, m, e, "mstar_a" if e == "mstar" else e)
        if r == 2 and e == "mstar" and NGPU == 1:
            # on a 1-GPU box correctness_pre starts no mstar server: round 2's is the control's process
            collect(out, srv, m, e, "mstar_b")
        elif r == 2 and e == "mstar" and NGPU >= 8:
            collect(out, srv, m, e, "mstar_c", extras=False)  # a third process, where GPUs are spare
        meta = {"model": m, "engine": e, "round": r, "config": srv.config_id(), "variant": srv.variant,
                "start_s": getattr(srv, "start_s", None), "server": srv.name,
                "max_concurrent_requests": srv.max_concurrent}
        items = load("test_clean_200")
        first = True
        for c in CONCS:
            path = cdir / f"c{c}.json"
            if path.exists() and rjson(path, {}).get("valid"):
                continue
            if first:
                run_requests(srv.url, req_model(m, e), repeat(items, WARM_FIRST_WAVES * WARM_CONC), WARM_CONC,
                             json_fields())
                first = False
            else:
                run_requests(srv.url, req_model(m, e), repeat(items, c), c, json_fields())
            res = cell(srv, m, e, repeat(items, REQS[c]), c, json_fields())
            res.update(meta, **srv.captured_ok())
            wjson(path, res)
            row = {k: v for k, v in res.items() if k != "records"}
            row.update({"kind": "short", "ts": time.time()})
            append_jsonl(out / "results.jsonl", row)
            log(f"{key} c={c}: RTFx {res['rtfx']:.0f} p50 {res['p50_ms'] or 0:.0f} ms WER "
                f"{100 * (res['wer'] or 0):.2f}% errors {res['errors']} bad {res['bad']} "
                f"{'' if res['valid'] else 'INVALID'}")
            if not res["valid"]:
                ok_all = False
                fail(f"{key} c={c}: {res['errors']} errors, {res['bad']} bad: {res['bad_examples'][:2]}")
            if not srv.healthy():
                fail(f"{key}: server unhealthy after c={c}")
                return
        lf = load("long_form")
        modes = [("long_seq", {})] + ([("long_par", {"long_form": "parallel"})]
                                      if e == "mstar" and MODELS[m]["family"] == "whisper" else [])
        for kind, extra_f in modes:
            path = cdir / f"{kind}.json"
            if path.exists() and rjson(path, {}).get("valid"):
                continue
            res = cell(srv, m, e, lf, 1, json_fields(**extra_f))
            res.update(meta, **srv.captured_ok())
            wjson(path, res)
            row = {k: v for k, v in res.items() if k != "records"}
            row.update({"kind": kind, "ts": time.time()})
            append_jsonl(out / "results.jsonl", row)
            log(f"{key} {kind}: RTFx {res['rtfx']:.0f} WER {100 * (res['wer'] or 0):.2f}% errors {res['errors']}")
            if not res["valid"]:
                # long-form content is judged against the whole file's reference; a few dropped
                # seam words don't make it "bad", the gate's WER bound judges it
                log(f"{key} {kind}: flagged ({res['bad_examples']})")
        if m == "qasr_rt" and r <= 2:
            for sessions, n in RT_SESSIONS.items():
                path = cdir / f"rt_s{sessions}.json"
                if path.exists():
                    continue
                res = realtime_sessions(srv, m, e, sessions, n)
                res.update(meta)
                wjson(path, res)
                row = {k: v for k, v in res.items() if k != "texts"}
                row.update({"kind": "realtime", "valid": res["errors"] == 0 and res["empty"] == 0, "ts": time.time()})
                append_jsonl(out / "results.jsonl", row)
                log(f"{key} realtime s={sessions}: partial p50 {res['partial_p50_ms']} delta p50 {res['delta_p50_ms']} "
                    f"final p50 {res['final_p50_ms']} WER {res['wer']} errors {res['errors']}")
        if NGPU == 1 and r == 1 and e == "mstar":
            # untimed, after every timed cell of this server (it may wedge it: last thing before the stop)
            stress(out, srv, m)
    if ok_all:
        mark(out, key)


def phase_round(a, r: int) -> int:
    out = a.out
    if r > 1 and not a.dry_run and not a.force and not (out / "gate" / "PASS").exists():
        print("refusing: the correctness gate has not passed (run `gate`); --force to override")
        return 2
    if not a.dry_run:
        rejudge_cells(out, r)
    order = []
    for m in MODEL_ORDER:
        engs = engines_of(m)
        k = (r - 1) % len(engs)
        order += [(m, e) for e in engs[k:] + engs[:k]]
    if a.dry_run:
        print(f"round{r} (timed, GPU {TIMED_GPU}, one server at a time): {order}")
        est = 0.0
        for m, e in order:
            mins = EST_START[e] + 1.5  # start + collect/smoke
            for c in CONCS:
                mins += (REQS[c] * 9.1) / EST_RTFX[e][c] / 60
            mins += (WARM_FIRST_WAVES * WARM_CONC * 9.1) / EST_RTFX[e][32] / 60
            lf_passes = 2 if e == "mstar" and MODELS[m]["family"] == "whisper" else 1
            mins += 605 / (110 if e == "mstar" else 900) / 60 * lf_passes
            if m == "qasr_rt" and r <= 2:
                mins += sum(n * 9.1 / s for s, n in RT_SESSIONS.items()) / 60 + 0.5
            if r == 1:
                mins += 1.5  # collect: 200 utts at c=1 + extras
            if e == "mstar" and NGPU == 1 and r in (1, 2):
                mins += 3.0 if r == 1 else 1.5  # stress after round 1's cells / control collect in round 2
            elif e == "mstar" and NGPU >= 8 and r == 2:
                mins += 0.7  # mstar_c
            est += mins
        print(f"  estimate ~{est:.0f} min")
        return 0
    if not wait_gpus_free(list(range(NGPU)), timeout=600):
        print("refusing to time with GPU memory held by something else")
        return 2
    for m, e in order:
        try:
            run_engine_round(out, m, e, r)
        except Exception as ex:  # noqa: BLE001
            fail(f"round{r}/{m}/{e} crashed: {ex!r}\n{traceback.format_exc()[-1500:]}")
    if not FAILURES:
        mark(out, f"round{r}")
    return 3 if FAILURES else 0


# ----------------------------------------------------------------------------- facts, gate, gallery

def match_words(served: list[dict], ref: list[dict]) -> list[float]:
    """|start(served) - start(ref)| for words matched by normalized text in order."""
    def norm(w):
        return "".join(ch for ch in (w or "").lower() if ch.isalnum())

    a = [norm(w.get("word")) for w in served]
    b = [norm(w.get("word")) for w in ref]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    diffs = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            sw, rw = served[blk.a + k], ref[blk.b + k]
            if a[blk.a + k] and sw.get("start") is not None and rw.get("start") is not None:
                diffs.append(abs(float(sw["start"]) - float(rw["start"])))
    return diffs


def _lang(x: str | None) -> str:
    codes = {"english": "en", "chinese": "zh", "german": "de", "french": "fr", "spanish": "es", "japanese": "ja",
             "korean": "ko", "russian": "ru", "italian": "it", "portuguese": "pt", "cantonese": "yue", "dutch": "nl"}
    s = (x or "").strip().lower()
    return codes.get(s, s)


def facts(out: Path, m: str) -> dict:
    fam = MODELS[m]["family"]
    items = load("test_clean_200")
    refs = {u.uid: u.reference for u in items}
    f: dict = {"model": m}
    cdir = out / "collect" / m
    f["wer"] = {}
    for p in sorted(cdir.glob("*.json")):
        if p.stem.endswith("_extras"):
            continue
        rows = rjson(p, [])
        by = {r["uid"]: r.get("text", "") for r in rows}
        f["wer"][p.stem] = wer([refs[u] for u in refs], [by.get(u, "") for u in refs])
    # content of every timed mstar cell
    cells = sorted((out / "cells" / m / "mstar").glob("r*/c*.json"))
    tot = {"cells": 0, "requests": 0, "bad": 0, "errors": 0, "examples": []}
    for p in cells:
        c = rjson(p, {})
        tot["cells"] += 1
        tot["requests"] += c.get("n", 0)
        tot["bad"] += c.get("bad", 0)
        tot["errors"] += c.get("errors", 0)
        tot["examples"] += [f"{p.parent.name}/{p.stem}: {x}" for x in c.get("bad_examples", [])][:3]
    f["content"] = tot
    for r in (1, 2, 3):
        c1, c64 = (rjson(out / "cells" / m / "mstar" / f"r{r}" / f"c{c}.json", {}) for c in (1, 64))
        if c1.get("wer") is not None and c64.get("wer") is not None:
            f["wer_cells"] = {"round": r, "c1": c1["wer"], "c64": c64["wer"]}
            break
    f["stress"] = rjson(out / "stress" / f"{m}.json", {})
    for k in ("c128", "c65"):
        if isinstance(f["stress"].get(k), dict):
            f["stress"][k].pop("records", None)
    ext = rjson(cdir / "mstar_a_extras.json") or rjson(cdir / "mstar_b_extras.json") or {}
    ref = rjson(out / "oracle" / m / "ref.json", {})
    rx = ref.get("extras", {})
    lf = load("long_form")[0]
    f["long_form"] = {}
    for mode, x in (ext.get("long_form") or {}).items():
        segs = x.get("segments") or []
        mono = all((s.get("start") or 0) <= (s.get("end") or 0) for s in segs) and all(
            (segs[i]["start"] or 0) >= (segs[i - 1]["start"] or 0) - 1e-3 for i in range(1, len(segs)))
        f["long_form"][f"mstar_{mode}"] = {
            "status": x.get("status"), "wer": wer([lf.reference], [x.get("text", "")]) if x.get("text") else None,
            "segments": len(segs), "segments_ok": bool(segs) and mono and (segs[-1].get("end") or 0) <= lf.duration + 1,
            "seconds": x.get("seconds")}
    rlf = rx.get("long_form") or {}
    f["long_form"]["ref"] = {"wer": wer([lf.reference], [rlf["text"]]) if rlf.get("text") else None,
                             "segments": len(rlf.get("segments") or []), "error": rlf.get("error")}
    det, rdet = ext.get("detect") or {}, rx.get("detect") or {}
    rows = []
    for clip, d in det.items():
        if clip in rdet and isinstance(rdet, dict) and not isinstance(rdet.get(clip), dict):
            rows.append({"clip": clip, "mstar": _lang(d.get("language")), "ref": _lang(rdet[clip])})
    f["detect"] = {"n": len(rows), "agree": sum(r["mstar"] == r["ref"] for r in rows),
                   "zh_ok": any(r["clip"] == "asr_zh" and r["mstar"] == r["ref"] == "zh" for r in rows), "rows": rows}
    if fam == "whisper":
        diffs = []
        for uid, x in (ext.get("words") or {}).items():
            rw = (rx.get("words") or {}).get(uid)
            if isinstance(rw, list) and x.get("words"):
                diffs += match_words(x["words"], rw)
        f["words"] = {"matched": len(diffs), "median_abs_start": statistics.median(diffs) if diffs else None,
                      "within_0.1": sum(d <= 0.1 for d in diffs) / len(diffs) if diffs else 0,
                      "within_0.5": sum(d <= 0.5 for d in diffs) / len(diffs) if diffs else 0}
    if m == "qasr_rt":
        rts = [rjson(p, {}) for p in sorted((out / "cells" / m / "mstar").glob("r*/rt_s*.json"))]
        xr = ext.get("realtime")
        if xr:
            rts.append(xr)
        texts = {}
        for i, rt in enumerate(rts):
            for uid, t in (rt.get("texts") or {}).items():
                texts[f"{i}:{uid}"] = (t, refs.get(uid, ""))
        f["realtime"] = {"sessions": sum(rt.get("n", 0) for rt in rts),
                         "errors": sum(rt.get("errors", 0) for rt in rts),
                         "empty": sum(rt.get("empty", 0) for rt in rts),
                         "wer": wer([r for _, r in texts.values()], [t for t, _ in texts.values()]) if texts else None}
    return f


def prepare_ref_tags(out: Path, m: str) -> None:
    ref = rjson(out / "oracle" / m / "ref.json", {})
    for name in ("bf16", "fp32"):
        texts = (ref.get("short") or {}).get(name)
        if texts:
            wjson(out / "collect" / m / f"ref_{name}.json",
                  [{"uid": u, "text": t, "error": None} for u, t in texts.items()])


def tf_stale(out: Path, m: str) -> bool:
    rep = out / "oracle" / m / "report.json"
    if not rep.exists():
        return True
    newest = max((p.stat().st_mtime for p in (out / "collect" / m).glob("*.json") if not p.stem.endswith("_extras")),
                 default=0)
    return rep.stat().st_mtime < newest


def run_gate(out: Path, m: str, tag: str = "mstar_a", skip: str = "", log_name: str | None = None) -> bool:
    fam = MODELS[m]["family"]
    fp = out / "gate" / f"facts_{m}.json"
    wjson(fp, facts(out, m))
    cmd = [sys.executable, str(HERE / "gate.py"), "--model", m, "--family", fam,
           "--pytest-log", str(out / "pytest" / f"{m}.log"), "--report", str(out / "oracle" / m / "report.json"),
           "--facts", str(fp), "--tag", tag,
           "--yardsticks", ",".join([e for e in ("vllm", "sgl_omni") if e in MODELS[m]["engines"]] + ["ref_bf16"])]
    if skip:
        cmd += ["--skip", skip]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out / "gate").mkdir(parents=True, exist_ok=True)
    (out / "gate" / (log_name or f"{m}.txt")).write_text(r.stdout + r.stderr)
    print(r.stdout + r.stderr, flush=True)
    return r.returncode == 0


def rejudge_stress(out: Path, m: str) -> None:
    """stress.load cases judged before the content check was calibrated (reference_bad): when every bad
    transcript is listed and each is a content verdict the reference implementation shares, the case
    has no bad transcripts. Stress keeps only bad_examples, so a case with unlisted bad ones stays."""
    path = out / "stress" / f"{m}.json"
    st = rjson(path, {})
    ref_bad, changed = reference_bad(out, m), False
    for name in ("c128", "c65"):
        x = st.get(name)
        if not isinstance(x, dict) or not x.get("bad") or x.get("errors") or x["bad"] > len(x.get("bad_examples", [])):
            continue
        ex = x["bad_examples"]
        if all(why.startswith(("garbage", "truncated")) and uid in ref_bad for uid, why in ex):
            x.update({"bad": 0, "valid": True, "bad_examples": [],
                      "ref_also_bad": sorted({(uid, why, ref_bad[uid]) for uid, why in ex}),
                      "rejudged": "content verdicts shared by the reference implementation"})
            changed = True
            log(f"rejudged stress/{m} {name}: {len(ex)} verdicts shared by the reference")
    if changed:
        wjson(path, st)


def phase_gate(a, tf: bool = True) -> int:
    out = a.out
    if a.dry_run:
        print("gate (no server up): oracle.py tf per model on GPUs 0-3 in parallel (fp32 teacher forcing of "
              "mstar_a, mstar_b, mstar_c, vllm, sgl_omni, ref_bf16, ref_fp32), facts, gate.py, gallery")
        print(f"  {NGPU} GPU lane(s); estimate ~{10 if NGPU >= 4 else 12 if NGPU >= 2 else 18} min "
              "(later re-runs only teacher-force runs not yet in the report)")
        return 0
    for m in MODEL_ORDER:
        prepare_ref_tags(out, m)
        if not a.fake:
            rejudge_stress(out, m)
    if tf and not a.fake:
        todo = [m for m in MODEL_ORDER if tf_stale(out, m)]
        lanes = gpu_lanes(len(todo))
        if not wait_gpus_free(list(range(len(lanes))), timeout=600):
            print("GPUs busy; the oracle and a server never share a GPU")
            return 2
        parallel([lambda g=g, idx=idx: [guarded(oracle_job, out, todo[k], g, "tf") for k in idx]
                  for g, idx in enumerate(lanes)], max(1, len(lanes)))
    verdicts = {m: run_gate(out, m) for m in MODEL_ORDER}
    try:
        build_gallery(out)
    except Exception as ex:  # noqa: BLE001 - the gallery never decides the verdict, but say it broke
        fail(f"gallery: {ex!r}\n{traceback.format_exc()[-1500:]}")
    summary = "\n".join(f"GATE {m}: {'PASS' if v else 'FAIL'}" for m, v in verdicts.items())
    (out / "gate" / "summary.txt").write_text(summary + "\n")
    for p in ("PASS", "FAIL"):
        with contextlib.suppress(OSError):
            (out / "gate" / p).unlink()
    passed = all(verdicts.values())
    (out / "gate" / ("PASS" if passed else "FAIL")).write_text(summary + "\n")
    print("\n" + summary + f"\nGATE OVERALL {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def fmt_segments(segs: list[dict]) -> str:
    return "\n".join(f"[{(s.get('start') or 0):7.2f} - {(s.get('end') or 0):7.2f}] {(s.get('text') or '').strip()}"
                     for s in segs)


def fmt_words(words: list[dict]) -> str:
    return " ".join(f"{w.get('word', '').strip()}[{float(w.get('start') or 0):.2f}-{float(w.get('end') or 0):.2f}]"
                    for w in words)


def build_gallery(out: Path) -> None:
    gdir = out / "gallery"
    gdir.mkdir(parents=True, exist_ok=True)
    index = ["<!doctype html><meta charset='utf-8'><title>PR #318 ASR gallery</title>",
             "<style>body{font:14px system-ui;margin:2rem auto;max-width:1100px}td,th{padding:2px 8px}</style>",
             "<h1>PR #318: Whisper large-v3-turbo / large-v3, Qwen3-ASR 1.7B / 0.6B realtime</h1>",
             "<p>Each page: input clips (play them), mstar's transcript next to the reference "
             "implementation's (transformers Whisper / qwen-asr SDK) and the other engines'. Includes "
             "language detection (no language given), the 605 s long-form file with segment timestamps, "
             "word timestamps, and realtime transcripts.</p><ul>"]
    for m in MODEL_ORDER:
        src = out / "gallery_src" / m
        (src / "parity").mkdir(parents=True, exist_ok=True)
        items = load("test_clean_200")
        cdir = out / "collect" / m
        tags = {p.stem: rjson(p, []) for p in sorted(cdir.glob("*.json"))
                if not p.stem.endswith("_extras") and not p.stem.startswith("ref_")}
        extras = {p.stem[:-7]: rjson(p, {}) for p in sorted(cdir.glob("*_extras.json"))}
        ref = rjson(out / "oracle" / m / "ref.json", {})
        rshort = (ref.get("short") or {}).get("bf16") or {}
        rx = ref.get("extras") or {}
        a_by = {r["uid"]: r.get("text", "") for r in tags.get("mstar_a", tags.get("mstar_b", []))}
        diff = [u for u in items if u.uid in rshort and normalize(a_by.get(u.uid, "")) != normalize(rshort[u.uid])]
        picks = [items[0]] + diff[:2]
        for extra_pick in (items[50], items[100]):
            if len(picks) < 3 and extra_pick not in picks:
                picks.append(extra_pick)
        prompts, texts, oracle = [], {}, {}

        def add(desc, audio, per_tag: dict, oracle_text, prompts=prompts, texts=texts, oracle=oracle):
            i = len(prompts)
            prompts.append({"text": desc, "audio": str(audio)})
            for t, v in per_tag.items():
                if v is not None:
                    texts.setdefault(t, []).append({"prompt": i, "text": v})
            if oracle_text is not None:
                oracle[str(i)] = oracle_text

        for u in picks:
            mark_ = " (mstar differs from the reference)" if u in diff else ""
            add(f"test-clean {u.uid}, language=en{mark_}. LibriSpeech reference: {u.reference}", u.path,
                {t: next((r.get("text") for r in rows if r.get("uid") == u.uid), None) for t, rows in tags.items()},
                rshort.get(u.uid))
        for clip, p in sorted((n, q) for n, q in asset_clips().items() if not n[0].isdigit()):
            add(f"{clip}: no language given (detection). Shows '[language] transcript'.", p,
                {t: (f"[{(x.get('detect') or {}).get(clip, {}).get('language')}] "
                     f"{(x.get('detect') or {}).get(clip, {}).get('text', '')}") for t, x in extras.items()},
                f"[{(rx.get('detect') or {}).get(clip)}] {(rx.get('detect_text') or {}).get(clip, '')}")
        lf = load("long_form")[0]
        per = {}
        for t, x in extras.items():
            for mode, v in (x.get("long_form") or {}).items():
                per[f"{t} ({mode})"] = (fmt_segments(v["segments"]) if v.get("segments") else v.get("text", "")) \
                    or f"(no output: {v.get('error')})"
        rlf = rx.get("long_form") or {}
        add(f"Long-form: {lf.uid}, {lf.duration:.0f} s, language=en, verbose_json segments (sequential; "
            f"mstar Whisper also long_form=parallel). Reference: {lf.reference[:300]}...", lf.path, per,
            fmt_segments(rlf["segments"]) if rlf.get("segments") else rlf.get("text"))
        if MODELS[m]["family"] == "whisper":
            u = items[0]
            add(f"Word timestamps: {u.uid} (verbose_json, timestamp_granularities=[word]); "
                "reference = transformers pipeline return_timestamps='word' (DTW on alignment heads)", u.path,
                {t: fmt_words(((x.get("words") or {}).get(u.uid) or {}).get("words") or []) or None
                 for t, x in extras.items()},
                fmt_words((rx.get("words") or {}).get(u.uid) or [])
                if isinstance((rx.get("words") or {}).get(u.uid), list) else None)
        if m == "qasr_rt":
            rt_texts = {}
            for e in ("mstar", "vllm"):
                p = out / "cells" / m / e / "r1" / "rt_s1.json"
                rt = rjson(p, {})
                for uid, t in (rt.get("texts") or {}).items():
                    rt_texts.setdefault(uid, {})[f"{e} realtime"] = t
            for uid in list(rt_texts)[:2]:
                u = next(x for x in items if x.uid == uid)
                add(f"Realtime /v1/realtime, {uid} played at real time in 100 ms frames; final transcript. "
                    "Oracle column = the SDK's offline transcript.", u.path, rt_texts[uid], rshort.get(uid))
        (src / "prompts.jsonl").write_text("\n".join(json.dumps(p) for p in prompts) + "\n")
        for t, rows in texts.items():
            wjson(src / "parity" / f"{t}.json", rows)
        wjson(src / "parity" / "report.json", {"oracle": oracle})
        subprocess.run([sys.executable, str(HERE / "gallery.py"), "--prompts", str(src / "prompts.jsonl"),
                        "--parity", str(src / "parity"), "--out", str(gdir / m),
                        "--title", f"PR #318 {MODELS[m]['key']} ({MODELS[m]['hf']}): transcripts vs reference"],
                       check=True)
        verdict = (out / "gate" / f"{m}.txt")
        v = verdict.read_text().strip().splitlines()[-1] if verdict.exists() else ""
        index.append(f"<li><a href='{m}/gallery.html'>{MODELS[m]['key']}</a> ({MODELS[m]['hf']}) - {v}</li>")
    index.append("</ul><p>Gate details: ../gate/*.txt. Numbers: ../summary.md.</p>")
    (gdir / "index.html").write_text("\n".join(index))
    log(f"gallery: {gdir / 'index.html'}")


# ----------------------------------------------------------------------------- tune, profile

def geomean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def short_sweep(srv: Server, m: str, e: str) -> dict:
    items = load("test_clean_200")
    run_requests(srv.url, req_model(m, e), repeat(items, 2 * WARM_CONC), WARM_CONC, json_fields())
    res = {}
    for c in (8, 32):
        x = cell(srv, m, e, repeat(items, REQS[c]), c, json_fields())
        res[c] = {"rtfx": x["rtfx"], "valid": x["valid"], "wer": x["wer"], "errors": x["errors"], "bad": x["bad"]}
    return res


def phase_tune(a) -> int:
    out = a.out
    if not a.dry_run and not a.force and not (out / "gate" / "PASS").exists():
        print("refusing: the correctness gate has not passed (run `gate`); --force to override")
        return 2
    cands = []
    for e in ("vllm", "sgl_omni"):
        if e in skipped_engines():
            continue
        for fam, m in TUNE_MODELS.items():
            knobs = TUNING[e] if e == "vllm" else TUNING[e][fam]
            cands += [(e, fam, m, k) for k in knobs]
    if a.dry_run:
        print(f"tune: {len(cands)} candidate servers on GPU {TIMED_GPU} (short sweep c=8, c=32 after 2 warm waves), "
              f"baseline = round-1 default cells; keep a knob only at >= 1.05x geomean RTFx:")
        for c in cands:
            print(f"  {c}")
        print(f"  estimate ~{len(cands) * 3:.0f} min")
        return 0
    state = rjson(out / "tune.json", {})
    for e, fam, m, knob in cands:
        key = f"{e}|{fam}|{' '.join(knob)}"
        if key in state:
            continue
        srv = Server(out, m, e, TIMED_GPU, 8100, "tune" + str(len(state)), extra=knob)
        with serving(srv) as up:
            state[key] = {"started": up, "sweep": short_sweep(srv, m, e) if up else None, "variant": srv.variant}
        wjson(out / "tune.json", state)
        log(f"tune {key}: {state[key]}")
    tuned = {}
    for e in ("vllm", "sgl_omni"):
        for fam, m in TUNE_MODELS.items():
            base = [rjson(out / "cells" / m / e / "r1" / f"c{c}.json", {}).get("rtfx") for c in (8, 32)]
            b = geomean(base)
            best, best_k = b * 1.05, None
            for key, v in state.items():
                ee, ff, k = key.split("|")
                if ee == e and ff == fam and v.get("sweep") and all(x["valid"] for x in v["sweep"].values()):
                    g = geomean([v["sweep"][c]["rtfx"] for c in ("8", "32")] if "8" in v["sweep"]
                                else [v["sweep"][c]["rtfx"] for c in (8, 32)])
                    if g > best:
                        best, best_k = g, k
            tuned.setdefault(e, {})[fam] = best_k.split() if best_k else []
            log(f"tune {e}/{fam}: baseline geomean {b:.0f}, choice {best_k or 'default'}")
    wjson(out / "tuned.json", tuned)
    mark(out, "tune")
    return 0


def phase_profile(a) -> int:
    out = a.out
    targets = [m for m in ("turbo", "qasr") if "mstar" in engines_of(m)]
    if a.dry_run:
        print(f"profile: MSTAR_PHASE_TIMING=100 on mstar {targets}, c=32 x 400 (untimed), server log kept; ~8 min")
        return 0
    for m in targets:
        if done(out, f"profile/{m}"):
            continue
        srv = Server(out, m, "mstar", TIMED_GPU, 8100, "profile", env_extra={"MSTAR_PHASE_TIMING": "100"},
                     cap=max(CONCS))
        with serving(srv) as up:
            if not up:
                continue
            items = load("test_clean_200")
            run_requests(srv.url, req_model(m, "mstar"), repeat(items, 64), 32, json_fields())
            res = cell(srv, m, "mstar", repeat(items, 400), 32, json_fields())
            res.pop("records")
            res["max_concurrent_requests"] = srv.max_concurrent
            wjson(out / "profile" / f"{m}.json", res)
        shutil.copy(srv.log_path, out / "profile" / f"{m}_server.log")
        mark(out, f"profile/{m}")
    return 0


# ----------------------------------------------------------------------------- summarize

def phase_summarize(a) -> int:
    out = a.out
    if a.dry_run:
        print("summarize: summary.md from results.jsonl (mean +- sd over valid rounds sharing a config), "
              "gate re-run with every control, gallery; ~2 min")
        return 0
    rows = [json.loads(ln) for ln in (out / "results.jsonl").read_text().splitlines() if ln.strip()] \
        if (out / "results.jsonl").exists() else []
    # keep the last row per (model, engine, round, kind, conc/sessions)
    last = {}
    for r in rows:
        last[(r.get("model"), r.get("engine"), r.get("round"), r.get("kind"), r.get("conc") or r.get("sessions"))] = r
    rows = list(last.values())
    prov = rjson(out / "provenance.json", {})
    lines = ["# PR #318 ASR benchmark (merged onto latest main)", "",
             f"Tree: {str(prov.get('tree', '')).strip()}", f"GPUs: {prov.get('nvidia', '')}", "",
             "Protocol: one H200 per system, one server at a time on GPU 0, fresh server per round, engine order "
             "rotated; 200 LibriSpeech test-clean utterances (1818.5 s) at c=1/8 (200 requests) and c=32/64 "
             "(400 requests), closed loop, language=en, temperature 0, json; 5 warmup waves at c=32 then 1 wave "
             "per point; RTFx = audio s / wall s. A cell counts only with 0 errors and 0 bad transcripts "
             "(per-request content check). mean +- sd over valid rounds with the same engine config. "
             f"mstar's timed servers raise a config's max_concurrent_requests to {max(CONCS)}, except "
             "Whisper's, which is what its decoder caches hold (the config column says what each served).", ""]

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-", None
        mu = statistics.mean(vals)
        return (f"{mu:.0f} +- {statistics.stdev(vals):.0f} (n={len(vals)})" if len(vals) > 1 else f"{mu:.0f} (n=1)"), mu

    gate_txt = (out / "gate" / "summary.txt").read_text() if (out / "gate" / "summary.txt").exists() else "gate not run"
    lines += ["## Correctness", "", "```", gate_txt.strip(), "```", ""]
    for m in MODEL_ORDER:
        rep = rjson(out / "oracle" / m / "report.json", {})
        fct = rjson(out / "gate" / f"facts_{m}.json", {})
        if rep.get("served"):
            lines += [f"### {MODELS[m]['key']}: teacher-forced through the fp32 reference "
                      f"({rep.get('transformers')})", "",
                      "| run | argmax agreement | fully argmax | confident disagreements | ends at oracle EOS "
                      "| max gap | WER |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for tag, rr in rep["served"].items():
                v = list(rr.values())
                mean = sum(x["argmax_agreement"] for x in v) / len(v)
                gaps = [x["logit_gap_at_first"] for x in v if x.get("logit_gap_at_first") is not None]
                w = (fct.get("wer") or {}).get(tag)
                lines.append(f"| {tag} | {mean:.4f} | {sum(x['first_disagreement'] is None for x in v)}/{len(v)} | "
                             f"{sum(x.get('confident_disagreements') or 0 for x in v)} | "
                             f"{sum(bool(x.get('eos_ok')) for x in v)}/{len(v)} | {max(gaps, default=0):.2f} | "
                             f"{'-' if w is None else f'{100 * w:.2f}%'} |")
            lines.append("")
    lines += ["## Throughput (RTFx, higher is better) and latency", ""]
    for m in MODEL_ORDER:
        lines += [f"### {MODELS[m]['key']} ({MODELS[m]['hf']})", "",
                  "| engine | config | " + " | ".join(f"RTFx c={c}" for c in CONCS)
                  + " | p50 ms c=1/8/32/64 | WER c=1 | long-form RTFx / WER |",
                  "|---|---|" + "---:|" * len(CONCS) + "---|---:|---|"]
        means = {}
        for e in MODELS[m]["engines"]:
            er = [r for r in rows if r.get("model") == m and r.get("engine") == e]
            for cfg in sorted({r.get("config") for r in er if r.get("kind") in ("short", "long_seq", "long_par")},
                              key=str):
                rr = [r for r in er if r.get("config") == cfg and r.get("valid")]
                cells_, p50s = [], []
                for c in CONCS:
                    s, mu = ms([r["rtfx"] for r in rr if r.get("kind") == "short" and r.get("conc") == c])
                    cells_.append(s)
                    means[(e, cfg, c)] = mu
                    p = [r.get("p50_ms") for r in rr if r.get("kind") == "short" and r.get("conc") == c]
                    p50s.append(f"{statistics.median(p):.0f}" if p else "-")
                w1 = [r.get("wer") for r in rr
                      if r.get("kind") == "short" and r.get("conc") == 1 and r.get("wer") is not None]
                lf = [r for r in er if r.get("config") == cfg and r.get("kind") in ("long_seq", "long_par")]
                lfs = "; ".join(f"{k[5:]} {statistics.mean([x['rtfx'] for x in lf if x['kind'] == k]):.0f} / "
                                f"{100 * statistics.mean([x['wer'] or 0 for x in lf if x['kind'] == k]):.2f}%"
                                for k in sorted({x["kind"] for x in lf}))
                lines.append(f"| {e} | {cfg} | " + " | ".join(cells_) + f" | {'/'.join(p50s)} | "
                             f"{f'{100 * statistics.mean(w1):.2f}%' if w1 else '-'} | {lfs or '-'} |")
        # mstar vs each engine at each concurrency (best config of each)
        for e in MODELS[m]["engines"]:
            if e == "mstar":
                continue
            rel = []
            for c in CONCS:
                ms_ = max([v for (ee, _, cc), v in means.items() if ee == "mstar" and cc == c and v] or [0])
                ot = max([v for (ee, _, cc), v in means.items() if ee == e and cc == c and v] or [0])
                rel.append(f"c={c} {100 * (ms_ / ot - 1):+.0f}%" if ms_ and ot else f"c={c} -")
            lines.append(f"\nmstar vs {e}: " + ", ".join(rel))
        lines.append("")
    rt = [r for r in rows if r.get("kind") == "realtime"]
    if rt:
        lines += ["## Realtime (qwen3_asr_realtime, 100 ms frames at real time)", "",
                  "| engine | round | sessions | partial p50/p95 ms | delta p50/p95 ms | final p50/p95 ms "
                  "| WER | errors |",
                  "|---|---:|---:|---|---|---|---:|---:|"]
        for r in sorted(rt, key=lambda r: (r["engine"], r["round"], r["sessions"])):
            def f2(a_, b_):
                return f"{a_:.0f}/{b_:.0f}" if a_ is not None and b_ is not None else "-"
            wer_s = "-" if r.get("wer") is None else f"{100 * r['wer']:.2f}%"
            lines.append(f"| {r['engine']} | {r['round']} | {r['sessions']} | "
                         f"{f2(r.get('partial_p50_ms'), r.get('partial_p95_ms'))} | "
                         f"{f2(r.get('delta_p50_ms'), r.get('delta_p95_ms'))} | "
                         f"{f2(r.get('final_p50_ms'), r.get('final_p95_ms'))} | "
                         f"{wer_s} | {r.get('errors')} |")
        lines.append("")
    refs = {m: rjson(out / "oracle" / m / "ref.json", {}).get("timing", {}).get("bf16") for m in MODEL_ORDER}
    lines += ["## Reference implementation, offline (not a server: batch 16, bf16, greedy)", ""]
    for m, t in refs.items():
        if t:
            lines.append(f"- {MODELS[m]['hf']}: {t['rtfx_offline_batch16']:.0f} RTFx "
                         f"({t['seconds']:.1f} s for {t['audio_s']:.0f} s)")
    excluded = [r for r in rows if not r.get("valid")]
    lines += ["", f"## Excluded cells ({len(excluded)})", ""]
    for r in excluded:
        lines.append(f"- {r.get('model')}/{r.get('engine')} r{r.get('round')} {r.get('kind')} c={r.get('conc')}: "
                     f"errors {r.get('errors')}, bad {r.get('bad')} {r.get('bad_examples', '')}")
    lines += ["", "## Tuning", "", "```", json.dumps(rjson(out / "tune.json", {}), indent=1)[:4000], "```",
              f"chosen: {rjson(out / 'tuned.json', {})}", "", "## Stress (mstar)", ""]
    for m in MODEL_ORDER:
        s = rjson(out / "stress" / f"{m}.json", {})
        for k, v in s.items():
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk not in ("records",)}
            lines.append(f"- {m} {k}: {json.dumps(v, default=str)[:400]}")
    neg = rjson(out / "negctl" / "result.json")
    if neg:
        lines += ["", "## Negative control", "",
                  f"planted: {neg.get('planted')}; caught by the gate: {neg.get('caught')}"]
    lines += ["", "## Versions", "", "```", json.dumps(prov, indent=1)[:6000], "```"]
    secs = sum(r.get("wall_s", 0) for r in rows)
    lines += ["", f"Timed request wall time across cells: {secs / 3600:.2f} h over {len(rows)} rows."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # the gate again, now with every control collected; never fails summarize
    if not a.fake and (out / "oracle").exists():
        with contextlib.suppress(Exception):
            phase_gate(a)
    return 0


# ----------------------------------------------------------------------------- negative control

def phase_negctl(a) -> int:
    out = a.out
    if a.dry_run:
        print(f"negctl: patch {NEGCTL_FILE.relative_to(REPO)} (decoder MLP-input norm x1.3), serve turbo on GPU 0, "
              "collect 200 transcripts as `negctl`, restore (sha256 checked), tf, gate --tag negctl -> must FAIL; "
              "~10 min")
        return 0
    src = NEGCTL_FILE.read_text()
    backup = out / "negctl" / "decoder.py.orig"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if "NEGCTL planted bug" in src:  # a crashed earlier run left the patch in place
        if not backup.exists():
            print("decoder.py carries the planted bug and there is no backup: refusing")
            return 2
        NEGCTL_FILE.write_text(backup.read_text())
        src = NEGCTL_FILE.read_text()
    if src.count(NEGCTL_OLD) != 1:
        print("negctl: patch anchor not found exactly once; refusing")
        return 2
    sha = hashlib.sha256(src.encode()).hexdigest()
    backup.write_text(src)
    if not (out / "collect" / "turbo" / "negctl.json").exists():
        try:
            for p in NEGCTL_FILE.parent.rglob("__pycache__"):
                shutil.rmtree(p, ignore_errors=True)
            NEGCTL_FILE.write_text(src.replace(NEGCTL_OLD, NEGCTL_NEW))
            srv = Server(out, "turbo", "mstar", TIMED_GPU, 8100, "negctl", timed=False)
            with serving(srv) as up:
                if up:
                    collect(out, srv, "turbo", "mstar", "negctl", extras=False)
        finally:
            NEGCTL_FILE.write_text(src)
    restored = hashlib.sha256(NEGCTL_FILE.read_text().encode()).hexdigest() == sha
    if not restored:
        print("negctl: RESTORE FAILED; decoder.py differs from its backup")
        return 2
    if not (out / "collect" / "turbo" / "negctl.json").exists():
        print("negctl: no transcripts collected")
        return 3
    oracle_job(out, "turbo", 0, "tf")
    passed = run_gate(out, "turbo", tag="negctl", log_name="negctl_turbo.txt",
                      skip="pytest,content,stress,long_form,detect,words,wer.under_load,realtime")
    res = {"planted": "Whisper decoder layers: final_layer_norm(h) * 1.3 before the MLP", "caught": not passed,
           "restored": restored}
    wjson(out / "negctl" / "result.json", res)
    verdict = "CAUGHT (gate FAIL, as it must)" if not passed else "NOT caught: the checks are too weak (finding)"
    print(f"NEGCTL: planted bug {verdict}")
    return 0


# ----------------------------------------------------------------------------- fake data (local validation)

def phase_fake(a) -> int:
    """Synthetic results, collect files, oracle reports and facts inputs, so summarize and gate
    can be exercised without a GPU. Writes tiny wavs into a fake DATA dir."""
    import random
    import wave

    out = a.out
    rnd = random.Random(0)
    words = "the quick brown fox jumps over the lazy dog while seven bright stars watch quietly".split()
    for name, n, dur in (("test_clean_200", 200, 1.0), ("long_form", 1, 2.0), ("stress20", 1, 2.0)):
        d = DATA / name
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(n):
            uid = f"{1000 + i}-0-{i:04d}" if name == "test_clean_200" else name
            with wave.open(str(d / f"{uid}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\0\0" * int(16000 * dur))
            lines.append(f"{uid}\t{' '.join(rnd.choice(words) for _ in range(rnd.randint(5, 14))).upper()}")
        (d / "references.tsv").write_text("\n".join(lines) + "\n")
    items = load("test_clean_200")
    lf = load("long_form")[0]
    for m in MODEL_ORDER:
        fam = MODELS[m]["family"]
        for tag in ("mstar_a", "mstar_b", "mstar_c", *[e for e in MODELS[m]["engines"] if e != "mstar"]):
            wjson(out / "collect" / m / f"{tag}.json",
                  [{"uid": u.uid, "text": u.reference.lower(), "error": None} for u in items])
        ext = {"detect": {"asr_en": {"status": 200, "language": "en", "text": "hello"},
                          "asr_zh": {"status": 200, "language": "zh", "text": "ni hao"}},
               "long_form": {"seq": {"status": 200, "text": lf.reference, "duration": lf.duration,
                                     "segments": [{"start": 0.0, "end": 1.0, "text": "a"},
                                                  {"start": 1.0, "end": 2.0, "text": "b"}]}}}
        if fam == "whisper":
            ext["long_form"]["par"] = ext["long_form"]["seq"]
            ext["words"] = {items[0].uid: {"status": 200,
                                           "words": [{"word": w, "start": 0.1 * i, "end": 0.1 * i + 0.1}
                                                     for i, w in enumerate(items[0].reference.split())]}}
        if m == "qasr_rt":
            ext["realtime"] = {"n": 3, "errors": 0, "empty": 0, "texts": {u.uid: u.reference for u in items[:3]}}
        wjson(out / "collect" / m / "mstar_a_extras.json", ext)
        ref = {"short": {"bf16": {u.uid: u.reference for u in items}, "fp32": {u.uid: u.reference for u in items}},
               "timing": {"bf16": {"seconds": 10.0, "audio_s": 200.0, "rtfx_offline_batch16": 20.0}},
               "extras": {"detect": {"asr_en": "en", "asr_zh": "zh"},
                          "detect_text": {"asr_en": "hello", "asr_zh": "ni hao"},
                          "long_form": {"text": lf.reference, "segments": [{"start": 0.0, "end": 2.0, "text": "x"}]},
                          "words": {items[0].uid: [{"word": w, "start": 0.1 * i + 0.02, "end": 0.1 * i + 0.1}
                                                   for i, w in enumerate(items[0].reference.split())]}}}
        wjson(out / "oracle" / m / "ref.json", ref)
        served = {}
        for tag in ("mstar_a", "mstar_b", "mstar_c", "ref_bf16", "ref_fp32",
                    *[e for e in MODELS[m]["engines"] if e != "mstar"]):
            served[tag] = {str(i): {"uid": u.uid, "tokens": 20, "argmax_agreement": 1.0 if tag == "ref_fp32" else 0.995,
                                    "first_disagreement": None if tag == "ref_fp32" or i % 10 else 7,
                                    "logit_gap_at_first": None if tag == "ref_fp32" or i % 10 else 0.2,
                                    "rank_at_first": None if tag == "ref_fp32" or i % 10 else 1,
                                    "confident_positions": 15, "confident_agreement": 1.0, "confident_disagreements": 0,
                                    "eos_ok": True} for i, u in enumerate(items)}
        wjson(out / "oracle" / m / "report.json",
              {"served": served, "uids": [u.uid for u in items], "transformers": "fake"})
        (out / "pytest").mkdir(parents=True, exist_ok=True)
        (out / "pytest" / f"{m}.log").write_text("====== 3 passed, 1 deselected in 42.00s ======\n")
        st = {"c128": {"n": 400, "bad": 0, "errors": 0}, "c65": {"n": 130, "bad": 0, "errors": 0},
              "bad_kwargs": {"status": 400, "seconds": 0.1, "healthy_after": True},
              "temp1": {"n": 10, "en": 10}}
        if fam == "whisper":
            st["long_prompt"] = {"status": 200, "healthy_after": True}
        if m == "qasr":
            st["qwen_20min_8"] = {"n": 8, "completed": 8}
            st["qwen_20min_16"] = {"n": 16, "completed": 9}
        wjson(out / "stress" / f"{m}.json", st)
        for r in (1, 2):
            for e in MODELS[m]["engines"]:
                for c in CONCS:
                    rtfx = EST_RTFX[e][c] * (1 + 0.03 * rnd.random())
                    row = {"model": m, "engine": e, "round": r, "kind": "short", "conc": c, "rtfx": rtfx,
                           "p50_ms": 1000 * 9 / EST_RTFX[e][1], "p95_ms": 2000.0, "wer": 0.013, "errors": 0, "bad": 0,
                           "valid": not (e == "sgl_omni" and c == 64 and r == 2), "config": "default", "wall_s": 5.0,
                           "n": REQS[c], "bad_examples": []}
                    append_jsonl(out / "results.jsonl", row)
                    wjson(out / "cells" / m / e / f"r{r}" / f"c{c}.json", {**row, "records": []})
                append_jsonl(out / "results.jsonl", {"model": m, "engine": e, "round": r, "kind": "long_seq", "conc": 1,
                                                     "rtfx": 100.0, "wer": 0.02, "valid": True, "config": "default",
                                                     "wall_s": 6.0})
                if m == "qasr_rt":
                    for s in RT_SESSIONS:
                        rt = {"model": m, "engine": e, "round": r, "kind": "realtime", "sessions": s, "n": 3,
                              "errors": 0, "empty": 0, "wer": 0.01, "partial_p50_ms": 50.0, "partial_p95_ms": 90.0,
                              "delta_p50_ms": 40.0, "delta_p95_ms": 70.0, "final_p50_ms": 60.0, "final_p95_ms": 80.0,
                              "valid": True, "texts": {u.uid: u.reference for u in items[:3]}}
                        append_jsonl(out / "results.jsonl", {k: v for k, v in rt.items() if k != "texts"})
                        wjson(out / "cells" / m / e / f"r{r}" / f"rt_s{s}.json", rt)
    wjson(out / "provenance.json", {"tree": "fake", "nvidia": "fake"})
    print(f"fake data written to {out} (DATA={DATA})")
    return 0


# ----------------------------------------------------------------------------- main

def phase_cleanup(a) -> int:
    """Stop every server this harness recorded (after a crash) and show GPU memory."""
    state = a.out / "servers" / "state"
    names = sorted(p.stem for p in state.glob("*.pid")) if state.exists() else []
    if a.dry_run:
        print(f"cleanup: would stop {names or 'nothing'}")
        return 0
    env = {**os.environ, "SERVER_STATE_DIR": str(state)}
    for n in names:
        r = subprocess.run(["bash", str(SERVER_SH), "stop", n], env=env, capture_output=True, text=True, check=False)
        print(f"stop {n}: {(r.stdout + r.stderr).strip()[-300:]}")
    print(gpu_mem())
    return 0


PHASES = ["cleanup", "data", "smoke", "correctness_pre", "round1", "gate", "tune", "round2", "round3", "profile",
          "summarize", "negctl", "fake"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=PHASES + ["all"])
    ap.add_argument("--out", type=Path, default=HOME / "results" / "pr318")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="time rounds 2-3 even without a gate PASS")
    ap.add_argument("--fake", action="store_true", help="gate/summarize on fake data (no oracle run)")
    ap.add_argument("--ngpu", type=int, help="GPU count (default: nvidia-smi -L); lays out the untimed work")
    a = ap.parse_args()
    global NGPU
    NGPU = a.ngpu or detect_ngpu()
    print(f"GPUs: {NGPU}", flush=True)
    a.out = a.out.expanduser()
    if not a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
    phases = ["smoke", "correctness_pre", "round1", "gate", "tune", "round2", "round3", "profile", "summarize"] \
        if a.phase == "all" else [a.phase]
    rc = 0
    for ph in phases:
        print(f"\n== {ph} {'(dry run)' if a.dry_run else ''}", flush=True)
        t0 = time.time()
        fn = {"data": phase_data, "smoke": phase_smoke, "correctness_pre": phase_correctness_pre,
              "round1": lambda a_: phase_round(a_, 1), "round2": lambda a_: phase_round(a_, 2),
              "round3": lambda a_: phase_round(a_, 3), "gate": lambda a_: phase_gate(a_, tf=not a_.fake),
              "tune": phase_tune, "profile": phase_profile, "summarize": phase_summarize, "negctl": phase_negctl,
              "fake": phase_fake, "cleanup": phase_cleanup}[ph]
        if a.dry_run and ph in ("data", "fake"):
            print(f"{ph}: " + (fn.__doc__ or "").strip().splitlines()[0])
            continue
        rc = fn(a) or 0
        if not a.dry_run:
            append_jsonl(a.out / "phases.jsonl", {"phase": ph, "rc": rc, "seconds": time.time() - t0,
                                                  "failures": FAILURES[:20], "ts": time.time()})
            print(f"== {ph} finished rc={rc} in {(time.time() - t0) / 60:.1f} min; failures: {len(FAILURES)}")
            for f_ in FAILURES:
                print(f"   - {f_.splitlines()[0]}")
        if rc and a.phase == "all" and ph in ("smoke", "gate"):
            break
    return rc


if __name__ == "__main__":
    sys.exit(main())
