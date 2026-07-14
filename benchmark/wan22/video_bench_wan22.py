""" Wan2.2-TI2V-5B video benchmark (t2v / i2v), engine-aware.

The measurement engine: client-side wall, closed-loop concurrency, mp4-byte sanity,
a per-phase breakdown, polled peak VRAM.

The phase breakdown is mstar-only, taken from the server's own request profiler
(launch it with ``--log-stats-file``); the baselines report only e2e, throughput and
peak VRAM. **Read the header of benchmark/wan22/reproduce.sh before comparing across
engines** — the rules that make a cross-system number valid live there.

    python -m benchmark.wan22.video_bench_wan22 --engine ours --port 8100 \
        --size 832x480 --frames 33 --steps 20 --concurrency 1 \
        --log-stats-file $SCRATCH/wan22_stats.txt --out-json row.json --label mstar
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from benchmark.wan22.profile_log import RequestProfile, parse_profiles

# ---------------------------------------------------------------------------
# Pinned generation config — the SAME on every engine, accelerations OFF.
# Values mirror mstar/model/wan22/config.py; the per-baseline flags are in the
# {vllm_omni,sglang_omni}_instructions.md docs.
# ---------------------------------------------------------------------------
PINNED_GUIDANCE_SCALE = 5.0     # Wan22Config.guidance_scale (diffusers WanPipeline default)
PINNED_NEGATIVE_PROMPT = ""     # Wan22Config.default_negative_prompt (empty; pinned on all systems)
PINNED_FLOW_SHIFT = 5.0         # Wan22Config.flow_shift (re-read from the checkpoint scheduler)
PINNED_FPS = 24                 # Wan22Config.video_fps (TI2V-5B model-card frame rate)
PINNED_SEED = 0

# The HF model id the baselines are served under (vllm-omni validates the request's
# `model` field against it; SGLang is launched with --model-path). M* uses "wan22".
WAN22_HF_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _baseline_model_id(cfg: "CellConfig") -> str:
    """Resolve the model id a baseline expects (M*'s "wan22" default → the HF id)."""
    return WAN22_HF_ID if cfg.model in ("wan22", "") else cfg.model

# The first 10 VBench subject_consistency prompts, embedded so the benchmark needs no
# network. --prompts-file overrides with a full VBench cache.
VBENCH_PROMPTS = [
    "a person swimming in ocean",
    "a person giving a presentation to a room full of colleagues",
    "a person washing the dishes",
    "a person eating a burger",
    "a person walking in the snowstorm",
    "a person drinking coffee in a cafe",
    "a person playing guitar",
    "a bicycle leaning against a tree",
    "a bicycle gliding through a snowy field",
    "a bicycle slowing down to stop",
]


# ---------------------------------------------------------------------------
# Small stats helpers (percentiles without numpy).
# ---------------------------------------------------------------------------
def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stats(values: list[float]) -> dict[str, float | None]:
    """mean / p50 / p95 / min / max of a sample (None-safe, all in the input unit)."""
    vals = [v for v in values if v is not None]
    return {
        "mean": _mean(vals),
        "p50": _pct(vals, 50),
        "p95": _pct(vals, 95),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


# ---------------------------------------------------------------------------
# Peak-VRAM poller — nvidia-smi background sampler.
# ---------------------------------------------------------------------------
class VramPoller:
    """Poll ``nvidia-smi memory.used`` on one GPU in a thread, keeping the peak.

    Whole-GPU, so only meaningful on a preflighted card. A spike shorter than
    ``interval_s`` can be missed; the interval and sample count go in the schema.
    """

    def __init__(self, gpu_index: int, interval_s: float = 0.05):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 "-i", str(self.gpu_index)],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return int(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 — a failed sample must not kill the bench
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            mib = self._poll_once()
            if mib is not None:
                self.peak_mib = max(self.peak_mib, mib)
                self.samples += 1
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.peak_mib = 0
        self.samples = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak_mib


# ---------------------------------------------------------------------------
# Per-engine video generation clients. Each returns (elapsed_s, mp4_bytes).
# ---------------------------------------------------------------------------
@dataclass
class GenResult:
    ok: bool
    e2e_s: float
    mp4_bytes: int = 0
    error: str = ""


def _gen_ours(cfg: "CellConfig", prompt: str) -> GenResult:
    """M* (this repo): POST /v1/videos/generations (JSON); mp4 in data[0].b64_json.

    guidance_scale / num_inference_steps / negative_prompt / flow_shift are not
    explicit protocol fields; they ride the request's extra_body (top-level JSON).
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": cfg.negative_prompt,
        "size": cfg.size,                       # "WxH" — Wan22Adapter splits to width/height
        "seed": cfg.seed,
        "guidance_scale": cfg.guidance_scale,
        "num_inference_steps": cfg.steps,
        "num_frames": cfg.frames,
        "fps": cfg.fps,
        "flow_shift": cfg.flow_shift,
    }
    if cfg.image_data_uri:
        payload["image"] = cfg.image_data_uri
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://{cfg.host}:{cfg.port}/v1/videos/generations",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            out = json.load(r)
        dt = time.perf_counter() - t0
        mp4 = base64.b64decode(out["data"][0]["b64_json"])
        return GenResult(ok=True, e2e_s=dt, mp4_bytes=len(mp4))
    except Exception as e:  # noqa: BLE001
        return GenResult(ok=False, e2e_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {str(e)[:160]}")


def _multipart(fields: dict) -> tuple[bytes, str]:
    """Encode fields as multipart/form-data — vllm-omni's /v1/videos rejects a JSON body."""
    boundary = f"----wan22bench{uuid.uuid4().hex}"
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _poll_and_download(base: str, vid: str, cfg: "CellConfig", t0: float) -> GenResult:
    """Poll ``GET {base}/{vid}`` until completed, then download ``/content``.

    The poll interval biases the measured e2e upward by half of it, so it is kept
    small (``cfg.poll_interval_s``) and recorded in the row's ``timing_boundary``.
    """
    deadline = t0 + cfg.timeout_s
    while time.perf_counter() < deadline:
        with urllib.request.urlopen(f"{base}/{vid}", timeout=60) as r:
            info = json.load(r)
        status = info.get("status")
        if status == "completed":
            break
        if status in ("failed", "error"):
            return GenResult(ok=False, e2e_s=time.perf_counter() - t0,
                             error=f"job {status}: {str(info.get('error'))[:120]}")
        time.sleep(cfg.poll_interval_s)
    else:
        return GenResult(ok=False, e2e_s=time.perf_counter() - t0, error="poll timeout")
    with urllib.request.urlopen(f"{base}/{vid}/content", timeout=120) as r:
        mp4 = r.read()
    return GenResult(ok=True, e2e_s=time.perf_counter() - t0, mp4_bytes=len(mp4))


def _gen_vllm(cfg: "CellConfig", prompt: str) -> GenResult:
    """vllm-omni baseline: multipart POST /v1/videos → poll → GET /content.

    An async job API, and multipart: ``/v1/videos/sync`` does not exist on the
    cu12-viable release (405), and a JSON body is rejected. fps and flow_shift are
    pinned because vllm-omni's Wan2.2 defaults (16, 12.0) differ from ours; serve it
    with --enforce-eager.
    """
    base = f"http://{cfg.host}:{cfg.port}/v1/videos"
    fields = {
        "model": _baseline_model_id(cfg),
        "prompt": prompt,
        "negative_prompt": cfg.negative_prompt,
        "size": cfg.size,
        "num_frames": cfg.frames,
        "fps": cfg.fps,
        "num_inference_steps": cfg.steps,
        "guidance_scale": cfg.guidance_scale,
        "flow_shift": cfg.flow_shift,
        "seed": cfg.seed,
    }
    body, content_type = _multipart(fields)
    req = urllib.request.Request(base, data=body, headers={"Content-Type": content_type})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            job = json.load(r)
        vid = job.get("id") or job.get("video_id")
        return _poll_and_download(base, vid, cfg, t0)
    except Exception as e:  # noqa: BLE001
        return GenResult(ok=False, e2e_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {str(e)[:160]}")


_ENGINES = {"ours": _gen_ours, "vllm": _gen_vllm}

# SGLang-Diffusion has no client here: it never produced usable numbers (blocked
# on an upstream packaging skew, and TI2V-5B is 720p-only there). run_cell refuses
# an --engine sglang cell with this message rather than pretend to measure it.
_SGLANG_UNSUPPORTED = (
    "--engine sglang is not measurable by this harness: SGLang-Diffusion is blocked "
    "on an upstream packaging skew (its cu12x wheels ship a broken deep_gemm)."
)

# What the client's measured wall actually spans, per engine. Stamped into every row
# so a cross-system delta is never read without its measurement boundary.
TIMING_BOUNDARY = {
    "ours": "POST /v1/videos/generations → response (mp4 inline, base64); synchronous, no polling",
    "vllm": "POST /v1/videos (multipart) → poll GET /v1/videos/{{id}} every {poll}ms → GET /content; "
            "includes submit + poll quantisation (+{halfpoll}ms expected bias) + download",
}


def _timing_boundary(cfg: "CellConfig") -> str:
    poll_ms = cfg.poll_interval_s * 1000
    return TIMING_BOUNDARY[cfg.engine].format(poll=f"{poll_ms:g}", halfpoll=f"{poll_ms / 2:g}")


# ---------------------------------------------------------------------------
# Cell config (one benchmark cell = one row).
# ---------------------------------------------------------------------------
@dataclass
class CellConfig:
    engine: str
    host: str
    port: int
    model: str
    size: str          # "WxH"
    frames: int
    steps: int
    concurrency: int
    guidance_scale: float = PINNED_GUIDANCE_SCALE
    negative_prompt: str = PINNED_NEGATIVE_PROMPT
    flow_shift: float = PINNED_FLOW_SHIFT
    fps: int = PINNED_FPS
    seed: int = PINNED_SEED
    num_prompts: int = 10
    warmup: int = 1
    timeout_s: float = 3600.0
    image_path: str = ""        # i2v conditioning frame (vllm: file path)
    image_data_uri: str = ""    # i2v conditioning frame (ours: base64 data URI)
    log_stats_file: str = ""    # server --log-stats-file (ours: phase breakdown source)
    gpu_index: int = 0
    vram_interval_s: float = 0.05
    # 20ms, not 250ms: the poll interval biases the baselines' e2e upward by half of it.
    poll_interval_s: float = 0.02
    # Interpreter of the SERVER's venv. Left empty, the server-side fields stay null —
    # never backfilled from this client, whose torch did not run the work.
    server_python: str = ""
    label: str = "wan22"


# ---------------------------------------------------------------------------
# Phase extraction from a parsed request profile.
# ---------------------------------------------------------------------------
def _phases_from_profile(p: RequestProfile) -> dict:
    """Map one request profile onto named generation phases (all ms).

    ``residual_tail_ms`` is a catch-all (mp4 encode + transfer + scheduling), not a
    clean postprocess. The trustworthy check is ``compute_span_ms`` vs ``node_sum``,
    which agree to under 0.12% at concurrency 1. Above concurrency 1 the per-phase
    columns are batch-shared, not per-request; only e2e and throughput hold there.
    """
    text_ms = p.node_ms("text_encoder")
    vae_enc_ms = p.node_ms("vae_encoder")
    denoise_ms = p.node_ms("dit")
    vae_dec_ms = p.node_ms("vae_decoder")
    step_mean_ms = p.denoise_step_mean_ms()
    node_sum = sum(v for v in (text_ms, vae_enc_ms, denoise_ms, vae_dec_ms) if v is not None)
    e2e_server_ms = p.total_ms

    # Tokenize/preprocess span (client bytes → engine inputs), if stamped.
    preprocess_ms = p.timeline_seg("recv -> preprocess done")
    # Independent server-side compute span (should ≈ node_sum): the conductor
    # ingest→done gap for non-streaming video (or ingest→first-chunk if streamed).
    compute_span_ms = p.timeline_seg("conductor ingest -> conductor done")
    if compute_span_ms is None:
        compute_span_ms = p.timeline_seg("conductor ingest -> first chunk")

    residual_ms = (e2e_server_ms - node_sum) if e2e_server_ms is not None else None
    # residual_tail = everything not a compute node and not tokenize: mp4 encode +
    # transfer + emit + scheduling, PLUS pre-ingest queue wait at concurrency>1.
    residual_tail_ms = None
    if residual_ms is not None and preprocess_ms is not None:
        residual_tail_ms = max(residual_ms - preprocess_ms, 0.0)

    return {
        "text_encode_ms": text_ms,
        "vae_encode_ms": vae_enc_ms,
        "denoise_ms": denoise_ms,
        "denoise_step_mean_ms": step_mean_ms,
        "vae_decode_ms": vae_dec_ms,
        "preprocess_ms": preprocess_ms,
        "residual_tail_ms": residual_tail_ms,
        "e2e_server_ms": e2e_server_ms,
        "phase_sum_ms": node_sum,
        "compute_span_ms": compute_span_ms,
        "residual_ms": residual_ms,
        "residual_frac": (residual_ms / e2e_server_ms) if (residual_ms is not None and e2e_server_ms) else None,
        # node-sum vs the server's own compute-span timestamp (the real invariant).
        "compute_span_vs_nodesum_frac": (
            abs(compute_span_ms - node_sum) / compute_span_ms
            if (compute_span_ms and node_sum) else None
        ),
    }


def _read_slice(path: str, offset: int) -> str:
    """Read a file from a byte offset to EOF (empty if missing/short)."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        return f.read()


def _file_size(path: str) -> int:
    return os.path.getsize(path) if path and os.path.exists(path) else 0


# ---------------------------------------------------------------------------
# Run one cell: warmup, measured closed-loop, phase parse, schema row.
# ---------------------------------------------------------------------------
def _require_supported_task(cfg: CellConfig) -> None:
    """Refuse an i2v cell on an engine whose client cannot send the image.

    Only ``_gen_ours`` sends a conditioning frame. The baseline clients send a
    text-only request, so accepting ``--image`` for them would issue a **t2v**
    generation and then stamp the row ``"task": "i2v"`` — a silently mislabelled
    measurement, which is worse than no measurement. Fail loudly instead.
    """
    if cfg.image_path and cfg.engine != "ours":
        raise ValueError(
            f"--image is not supported for --engine {cfg.engine}: this harness's "
            f"{cfg.engine} client sends no conditioning frame, so the cell would "
            "silently be a t2v run labelled i2v. Run i2v against --engine ours, or "
            "extend the client to upload the image before benchmarking i2v here."
        )


def run_cell(cfg: CellConfig, prompts: list[str], gpu_info: dict,
             server_info: dict | None = None) -> dict:
    _require_supported_task(cfg)
    if cfg.engine == "sglang":
        raise ValueError(_SGLANG_UNSUPPORTED)
    gen = _ENGINES[cfg.engine]
    server_info = server_info or probe_server_versions(cfg.server_python)

    # -- Warmup, excluded from every metric: the profiler offset is taken after it.
    if cfg.warmup > 0:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            list(ex.map(lambda i: gen(cfg, prompts[i % len(prompts)]), range(cfg.warmup)))
        # Settle so a late-flushed warmup profile lands before we take the offset.
        if cfg.log_stats_file:
            time.sleep(1.5)

    # Everything appended past here belongs to the measured requests only.
    log_offset = _file_size(cfg.log_stats_file)

    poller = VramPoller(cfg.gpu_index, cfg.vram_interval_s)
    poller.start()

    n = min(cfg.num_prompts, len(prompts))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        results: list[GenResult] = list(ex.map(lambda i: gen(cfg, prompts[i]), range(n)))
    makespan = time.perf_counter() - t0

    peak_vram = poller.stop()

    # -- Let the server flush the last profile(s) before slicing the log.
    if cfg.log_stats_file:
        time.sleep(1.5)

    oks = [r for r in results if r.ok]
    e2e_client = [r.e2e_s for r in oks]
    mp4_bytes = [r.mp4_bytes for r in oks]
    first_err = next((r.error for r in results if not r.ok), "")

    # -- Phase breakdown (ours only): parse the measured slice of the profile log.
    per_phase: dict[str, list[float]] = {}
    profiles: list[RequestProfile] = []
    phase_note = ""
    if cfg.engine == "ours" and cfg.log_stats_file:
        profiles = parse_profiles(_read_slice(cfg.log_stats_file, log_offset))
        if len(profiles) != len(oks):
            phase_note = f"profile/response count mismatch ({len(profiles)} profiles vs {len(oks)} ok responses)"
        for p in profiles:
            ph = _phases_from_profile(p)
            for k, v in ph.items():
                if v is not None:
                    per_phase.setdefault(k, []).append(v)

    # -- Sanity: an "ours" cell loads a 5B model (>20 GiB); a near-idle peak means
    #    the VRAM poller's --gpu-index does not match the server's GPU.
    if cfg.engine == "ours" and oks and 0 <= peak_vram < 5000:
        phase_note = (phase_note + "; " if phase_note else "") + \
            f"peak VRAM {peak_vram} MiB implausibly low for a 5B run — --gpu-index {cfg.gpu_index} " \
            "likely does not match the server's GPU"

    # -- Aggregate the per-request phase samples to mean/p50/p95 (ms → also s).
    def agg_ms(key: str) -> dict:
        return _stats(per_phase.get(key, []))

    row = {
        # ---- identity / config ----
        "label": cfg.label,
        "system": {"ours": "mstar", "vllm": "vllm-omni"}[cfg.engine],
        "engine": cfg.engine,
        "model": cfg.model if cfg.engine == "ours" else _baseline_model_id(cfg),
        "endpoint": {"ours": "/v1/videos/generations", "vllm": "/v1/videos"}[cfg.engine],
        # Only the mstar client sends the conditioning image; a baseline i2v request
        # is refused rather than run as t2v and mislabelled (_require_supported_task).
        "task": "i2v" if cfg.image_data_uri else "t2v",
        "size_wxh": cfg.size,
        "width": int(cfg.size.lower().split("x")[0]),
        "height": int(cfg.size.lower().split("x")[1]),
        "num_frames": cfg.frames,
        "num_inference_steps": cfg.steps,
        "guidance_scale": cfg.guidance_scale,
        "negative_prompt": cfg.negative_prompt,
        "flow_shift": cfg.flow_shift,   # a no-op on mstar, which reads config.flow_shift
        "scheduler": "unipc",
        "dtype_dit": "bf16",   # mstar: bf16 weights with fp32 islands. VAE/text-encoder
                               # precision is not pinnable over the wire and differs per stack.
        "seed": cfg.seed,
        "fps": cfg.fps,
        "concurrency": cfg.concurrency,
        "num_prompts": n,
        "num_warmup": cfg.warmup,
        # Asserted, not measured: there is no server read-back, so a forgotten
        # --enforce-eager would still stamp True. Verify from the serve logs.
        "accelerations_off_asserted": True,
        # What the measured wall actually spans on THIS engine's transport — never
        # compare an e2e across engines without reading this (see TIMING_BOUNDARY).
        "timing_boundary": _timing_boundary(cfg),
        "poll_interval_s": cfg.poll_interval_s if cfg.engine == "vllm" else None,
        # ---- environment ----
        "gpu_name": gpu_info.get("name"),
        "gpu_count": gpu_info.get("count"),
        "compute_cap": gpu_info.get("compute_cap"),
        "driver_version": gpu_info.get("driver_version"),
        # The CLIENT's stack (this process). NOT the server's — the baselines run in
        # their own venvs. Named so the two can never be confused again.
        "client_torch_version": gpu_info.get("client_torch_version"),
        "client_cuda_version": gpu_info.get("client_cuda_version"),
        # The SERVER's stack, read from its own interpreter (--server-python). None
        # when not supplied: unknown is recorded as unknown, never as the client's.
        "server_torch_version": server_info.get("server_torch_version"),
        "server_cuda_version": server_info.get("server_cuda_version"),
        "server_cudnn_version": server_info.get("server_cudnn_version"),
        # ---- outcome ----
        "ok": len(oks),
        "total": len(results),
        "first_error": first_err,
        # ---- latency / throughput (seconds unless noted) ----
        "e2e_client_s": _stats(e2e_client),
        "makespan_s": makespan,
        "throughput_req_s": (len(oks) / makespan) if makespan > 0 else None,
        "peak_vram_mib": peak_vram,
        # Whole-GPU, and only meaningful on a preflighted card. A client cannot read a
        # server's torch-scoped peak, so that field stays null here.
        "peak_vram_metric": "nvidia-smi:whole-gpu",
        "peak_vram_mib_torch": None,
        "vram_sample_interval_s": cfg.vram_interval_s,
        "vram_samples": poller.samples,
        "mp4_bytes_mean": _mean([float(b) for b in mp4_bytes]),
        # ---- per-phase breakdown (ours only; ms) ----
        "text_encode_ms": agg_ms("text_encode_ms"),
        "vae_encode_ms": agg_ms("vae_encode_ms"),
        "denoise_total_ms": agg_ms("denoise_ms"),
        "denoise_step_mean_ms": agg_ms("denoise_step_mean_ms"),  # per-request mean, from the profiler
        "vae_decode_ms": agg_ms("vae_decode_ms"),
        "preprocess_ms": agg_ms("preprocess_ms"),
        "residual_tail_ms": agg_ms("residual_tail_ms"),  # non-node remainder; incl. queue wait at conc>1
        "e2e_server_ms": agg_ms("e2e_server_ms"),
        # ---- invariant: phase-sum vs server e2e ----
        "phase_sum_ms": agg_ms("phase_sum_ms"),
        "compute_span_ms": agg_ms("compute_span_ms"),  # server timeline compute span (independent)
        "residual_ms": agg_ms("residual_ms"),
        "residual_frac_mean": _mean(per_phase.get("residual_frac", [])),
        # node-sum vs server compute-span agreement (the strong invariant): ~0 when trustworthy.
        "compute_span_vs_nodesum_frac_mean": _mean(per_phase.get("compute_span_vs_nodesum_frac", [])),
        "profiles_parsed": len(profiles),
        "note": phase_note,
    }
    return row


# ---------------------------------------------------------------------------
# Environment probe (torch/cuda/driver/gpu) — recorded in every row.
# ---------------------------------------------------------------------------
def probe_gpu(gpu_index: int) -> dict:
    """Hardware, plus this CLIENT's torch/cuda (field names say ``client_``).

    A baseline runs its own torch in its own venv, so use ``probe_server_versions``
    for the stack that actually did the compute.
    """
    info: dict = {"count": None, "name": None, "compute_cap": None,
                  "driver_version": None, "client_torch_version": None,
                  "client_cuda_version": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader", "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=5, check=True,
        )
        name, cap, drv = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
        info.update(name=name, compute_cap=cap, driver_version=drv)
        cnt = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5, check=True)
        info["count"] = len(cnt.stdout.strip().splitlines())
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        info["client_torch_version"] = torch.__version__
        info["client_cuda_version"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return info


# Printed by the server's interpreter, so the answer is the SERVER's stack.
_VERSION_PROBE = (
    "import json,torch;"
    "print(json.dumps({'server_torch_version':torch.__version__,"
    "'server_cuda_version':torch.version.cuda,"
    "'server_cudnn_version':torch.backends.cudnn.version()}))"
)


def probe_server_versions(server_python: str) -> dict:
    """Read torch/CUDA/cuDNN from the SERVER's venv by running its interpreter.

    The server can't be introspected over HTTP, so the probe runs its python.
    Without ``--server-python`` the fields stay None — never backfilled from the
    client, whose torch did not do the work.
    """
    empty = {"server_torch_version": None, "server_cuda_version": None,
             "server_cudnn_version": None}
    if not server_python:
        return empty
    try:
        out = subprocess.run([server_python, "-c", _VERSION_PROBE],
                             capture_output=True, text=True, timeout=60, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 — a failed probe must not kill the bench
        print(f"[wan22-bench] WARNING: server version probe failed ({e}); "
              f"server_* fields stay null", flush=True)
        return empty


# ---------------------------------------------------------------------------
# Human-readable one-line-per-phase summary.
# ---------------------------------------------------------------------------
def print_row(row: dict) -> None:
    def g(d, k):
        v = d.get(k) if isinstance(d, dict) else None
        return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
    e2e = row["e2e_client_s"]
    print(f"=== {row['label']}  {row['system']}  {row['task']}  {row['size_wxh']}x{row['num_frames']}f  "
          f"steps={row['num_inference_steps']}  conc={row['concurrency']}  "
          f"ok={row['ok']}/{row['total']} ===", flush=True)
    if row["throughput_req_s"]:
        print(f"  e2e mean {g(e2e,'mean')}s p50 {g(e2e,'p50')}s p95 {g(e2e,'p95')}s  "
              f"thrpt {row['throughput_req_s']:.4f} req/s  "
              f"peakVRAM {row['peak_vram_mib']} MiB [{row['peak_vram_metric']}]", flush=True)
    else:
        print("  (no successful requests)", flush=True)
    if row["engine"] == "ours" and row["profiles_parsed"]:
        star = "*" if row["concurrency"] > 1 else ""
        print(f"  phases(ms): text {g(row['text_encode_ms'],'mean')}  "
              f"denoise {g(row['denoise_total_ms'],'mean')} (step {g(row['denoise_step_mean_ms'],'mean')}{star})  "
              f"vae {g(row['vae_decode_ms'],'mean')}  tail {g(row['residual_tail_ms'],'mean')}  "
              f"[node-sum vs compute-span Δ {100 * (row.get('compute_span_vs_nodesum_frac_mean') or 0):.2f}%]",
              flush=True)
        if star:
            print("  * denoise-step mean is inflated by interleaving at conc>1", flush=True)
    if row["note"]:
        print(f"  note: {row['note']}", flush=True)
    if row["first_error"]:
        print(f"  first_error: {row['first_error']}", flush=True)


# ---------------------------------------------------------------------------
# Schema writers.
# ---------------------------------------------------------------------------
# Flat CSV columns (scalar projection of the nested row); runs append to this.
CSV_COLUMNS = [
    "label", "system", "task", "size_wxh", "num_frames", "num_inference_steps", "guidance_scale",
    "concurrency", "num_prompts", "accelerations_off_asserted", "gpu_name", "compute_cap",
    "client_torch_version", "server_torch_version", "server_cudnn_version",
    "poll_interval_s", "ok", "total", "throughput_req_s",
    "peak_vram_mib", "peak_vram_metric", "peak_vram_mib_torch",
    "e2e_client_mean_s", "e2e_client_p50_s", "e2e_client_p95_s",
    "text_encode_mean_ms", "denoise_total_mean_ms", "denoise_step_mean_ms", "vae_decode_mean_ms",
    "preprocess_mean_ms", "residual_tail_mean_ms",
    "e2e_server_mean_ms", "phase_sum_mean_ms", "residual_mean_ms", "residual_frac_mean",
    "compute_span_vs_nodesum_frac_mean",
]


def row_to_csv_dict(row: dict) -> dict:
    def m(d, k):
        return d.get(k) if isinstance(d, dict) else None
    return {
        "label": row["label"], "system": row["system"], "task": row["task"],
        "size_wxh": row["size_wxh"], "num_frames": row["num_frames"],
        "num_inference_steps": row["num_inference_steps"], "guidance_scale": row["guidance_scale"],
        "concurrency": row["concurrency"], "num_prompts": row["num_prompts"],
        "accelerations_off_asserted": row["accelerations_off_asserted"], "gpu_name": row["gpu_name"],
        "compute_cap": row["compute_cap"],
        "client_torch_version": row["client_torch_version"],
        "server_torch_version": row["server_torch_version"],
        "server_cudnn_version": row["server_cudnn_version"],
        "poll_interval_s": row["poll_interval_s"],
        "ok": row["ok"], "total": row["total"],
        "throughput_req_s": row["throughput_req_s"],
        "peak_vram_mib": row["peak_vram_mib"],
        "peak_vram_metric": row["peak_vram_metric"],
        "peak_vram_mib_torch": row["peak_vram_mib_torch"],
        "e2e_client_mean_s": m(row["e2e_client_s"], "mean"),
        "e2e_client_p50_s": m(row["e2e_client_s"], "p50"),
        "e2e_client_p95_s": m(row["e2e_client_s"], "p95"),
        "text_encode_mean_ms": m(row["text_encode_ms"], "mean"),
        "denoise_total_mean_ms": m(row["denoise_total_ms"], "mean"),
        "denoise_step_mean_ms": m(row["denoise_step_mean_ms"], "mean"),
        "vae_decode_mean_ms": m(row["vae_decode_ms"], "mean"),
        "preprocess_mean_ms": m(row["preprocess_ms"], "mean"),
        "residual_tail_mean_ms": m(row["residual_tail_ms"], "mean"),
        "e2e_server_mean_ms": m(row["e2e_server_ms"], "mean"),
        "phase_sum_mean_ms": m(row["phase_sum_ms"], "mean"),
        "residual_mean_ms": m(row["residual_ms"], "mean"),
        "residual_frac_mean": row["residual_frac_mean"],
        "compute_span_vs_nodesum_frac_mean": row.get("compute_span_vs_nodesum_frac_mean"),
    }


def write_rows(rows: list[dict], json_path: str | None, csv_path: str | None) -> None:
    if json_path:
        # Append into a JSON array file (create/extend), so repeated cells accrete.
        existing: list = []
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    existing = json.load(f)
            except Exception:  # noqa: BLE001
                existing = []
        existing.extend(rows)
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"wrote {len(rows)} row(s) -> {json_path} (total {len(existing)})", flush=True)
    if csv_path:
        import csv
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(row_to_csv_dict(r))
        print(f"appended {len(rows)} row(s) -> {csv_path}", flush=True)


def load_prompts(path: str | None, n: int) -> list[str]:
    if path:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines[:n] if n else lines
    return VBENCH_PROMPTS[:n]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["ours", "vllm", "sglang"], required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="wan22")
    ap.add_argument("--size", default="832x480", help="WxH (832x480 == the 480x832 HxW grid cell)")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=PINNED_GUIDANCE_SCALE)
    ap.add_argument("--negative", default=PINNED_NEGATIVE_PROMPT)
    ap.add_argument("--flow-shift", type=float, default=PINNED_FLOW_SHIFT)
    ap.add_argument("--fps", type=int, default=PINNED_FPS)
    ap.add_argument("--seed", type=int, default=PINNED_SEED)
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--image", default="", help="i2v conditioning frame (path)")
    ap.add_argument("--log-stats-file", default="", help="server --log-stats-file path (ours: phase source)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--vram-interval", type=float, default=0.05)
    ap.add_argument("--poll-interval", type=float, default=0.02,
                    help="poll tick for the async baselines (s). Biases their e2e upward by "
                         "half this; keep it small (default 20ms)")
    ap.add_argument("--server-python", default="",
                    help="interpreter of the SERVER's venv (e.g. /path/vllm-venv/bin/python) — "
                         "stamps the SERVER's torch/cuDNN into the row instead of the client's")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--label", default="wan22")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    image_data_uri = ""
    if args.image and args.engine == "ours":
        with open(args.image, "rb") as f:
            image_data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    cfg = CellConfig(
        engine=args.engine, host=args.host, port=args.port, model=args.model,
        size=args.size, frames=args.frames, steps=args.steps, concurrency=args.concurrency,
        guidance_scale=args.guidance, negative_prompt=args.negative, flow_shift=args.flow_shift,
        fps=args.fps, seed=args.seed, num_prompts=args.num_prompts, warmup=args.warmup,
        timeout_s=args.timeout, image_path=args.image,
        image_data_uri=image_data_uri, log_stats_file=args.log_stats_file,
        gpu_index=args.gpu_index,
        vram_interval_s=args.vram_interval, poll_interval_s=args.poll_interval,
        server_python=args.server_python, label=args.label,
    )
    prompts = load_prompts(args.prompts_file, args.num_prompts)
    gpu_info = probe_gpu(args.gpu_index)
    row = run_cell(cfg, prompts, gpu_info, probe_server_versions(cfg.server_python))
    print_row(row)
    write_rows([row], args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
