"""Wan VAE streaming-decode microbenchmark, one variant per process.

Every streaming variant replays the same latent stream: one first latent frame
(untimed), then ``--warmup`` chunks (untimed; absorbs torch.compile), then
``--chunks`` steady chunks of 3 latent frames timed with CUDA events around
decode + 8-bit conversion, synchronized per chunk. The stream is replayed
``--repeats`` times from a cold cache; samples from all repeats are pooled.

Variants:
  ref_fp32            diffusers, eager, fp32, TF32 off, per-frame calls: exact reference
  ref_fp32_tf32       control: ref_fp32 with TF32 allowed
  diffusers_bf16      stock diffusers, eager, bf16, per-frame
  diffusers_bf16_compile  + torch.compile of the decoder
  mstar_step{0,1,2,3,5}[_eager]  M*'s Cosmos3VAEDecoderARSubmodule with this change's
                      switches turned on one at a time (STEPS below): 0 is the
                      stateless context re-decode, 5 all switches on
  mstar_warm[_eager]  the node at its config defaults (warm cache, channels-last,
                      one latent per decoder call, cudnn.benchmark off)
  mstar_warm_bench    the same with vae_decode_cudnn_benchmark on (opt-in)
  mstar_warm_lpc{N}   mstar_warm with windowed_decode_latents_per_call=N

The prototype-harness variants (px_*, channels_last_*), the ablation suites and
the analysis scripts are on the perf/wan-vae-decode-results branch.

  python bench_wan_decoder.py --variant diffusers_bf16 --latents latents_wan21.pt --out res/
"""

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from common import (
    FRAMES_PER_CHUNK,
    WAN21_REPO,
    WAN22_REPO,
    StreamingDecoder,
    latent_stats,
    load_vae,
    mstar_path,
    to_uint8,
)

QUALITY_CHUNKS = 10  # PSNR over the first 10 steady chunks
REPO = WAN21_REPO    # --vae wan22 switches every variant to the Wan2.2 VAE M* serves


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float) -> float:
    mse = torch.mean((a.double() - b.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)


def env_info():
    info = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
    }
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gib"] = round(p.total_memory / 2**30, 1)
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
            info["nvidia_smi"] = q.stdout.strip()
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True, check=True)
            info["gpu_apps_at_start"] = apps.stdout.strip().splitlines()
        except Exception:
            pass
    try:
        info["mstar_commit"] = subprocess.run(
            ["git", "-C", mstar_path(), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    return info


class Timer:
    def __init__(self, device):
        self.cuda = device.startswith("cuda")

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.e.record()
            self.e.synchronize()
            self.ms = self.s.elapsed_time(self.e)
        else:
            self.ms = (time.perf_counter() - self.t0) * 1e3


def reset_peak(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak(device):
    if not device.startswith("cuda"):
        return {}
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


# --------------------------------------------------------------------------- variants

def build_streaming(variant, device):
    """(StreamingDecoder, dtype) for a streaming variant, or None if not one."""
    if variant in ("ref_fp32", "ref_fp32_tf32"):
        tf32 = variant == "ref_fp32_tf32"
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        vae = load_vae(REPO, torch.float32, device)
        sd = StreamingDecoder(vae, per_frame=True, compile_decoder=False)
        return sd, torch.float32
    m = re.fullmatch(r"diffusers_(bf16|fp16)(_compile)?", variant)
    if m:
        dtype = torch.bfloat16 if m[1] == "bf16" else torch.float16
        vae = load_vae(REPO, dtype, device)
        sd = StreamingDecoder(vae, per_frame=True, compile_decoder=bool(m[2]))
        return sd, dtype
    return None


def precision_flags():
    """TF32 / precision state as the run actually saw it."""
    out = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    for name, get in (("cudnn_conv_fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision),
                      ("cuda_matmul_fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision)):
        try:
            out[name] = get()
        except Exception as exc:  # API absent or mixed-API error
            out[name] = f"n/a ({type(exc).__name__})"
    return out


def run_streaming(args, raw, ref):
    built = build_streaming(args.variant, args.device)
    if built is None:
        sys.exit(f"unknown variant {args.variant}")
    sd, dtype = built
    z = raw.to(args.device, dtype)
    n_need = 1 + 3 * (args.warmup + args.chunks)
    assert z.shape[2] >= n_need, f"latent stream has {z.shape[2]} frames, need {n_need}"
    lat, quality_f, quality_u8 = [], [], []
    out_ref = [] if args.save_ref else None
    mem = {}
    t_compile = None
    with torch.inference_mode():
        for rep in range(args.repeats):
            sd.reset()
            sd.first(z[:, :, :1])
            for c in range(args.warmup + args.chunks):
                zc = z[:, :, 1 + 3 * c: 4 + 3 * c]
                if rep == 0 and c == args.warmup:
                    reset_peak(args.device)
                with Timer(args.device) as t:
                    out = sd.chunk(zc)
                    u8 = to_uint8(out)
                if rep == 0 and c == 0:
                    t_compile = t.ms
                if c < args.warmup:
                    continue
                lat.append(t.ms)
                k = c - args.warmup
                if rep == 0 and k < QUALITY_CHUNKS:
                    if out_ref is not None:
                        out_ref.append(out.float().cpu())
                    if ref is not None:
                        quality_f.append(psnr(out.float().cpu(), ref["float"][k], 2.0))
                        quality_u8.append(psnr(u8.float().cpu(), to_uint8(ref["float"][k]).float(), 255.0))
            if rep == 0:
                mem = peak(args.device)
    if out_ref is not None:
        torch.save({"float": out_ref}, args.save_ref)
    return lat, quality_f, quality_u8, mem, t_compile


# The in-tree ladder: Cosmos3Config switches per step, each on top of the one
# before. Conv-internal padding (step 4 of the prototype ladder) has no switch of
# its own: ChannelsLastCausalConv3d comes with vae_decode_channels_last (step 2).
STEPS = {
    0: dict(windowed_decode_warm_cache=False, vae_decode_channels_last=False,
            windowed_decode_latents_per_call=1, vae_decode_cudnn_benchmark=False),
    1: dict(windowed_decode_warm_cache=True, vae_decode_channels_last=False,
            windowed_decode_latents_per_call=1, vae_decode_cudnn_benchmark=False),
    2: dict(windowed_decode_warm_cache=True, vae_decode_channels_last=True,
            windowed_decode_latents_per_call=1, vae_decode_cudnn_benchmark=False),
    3: dict(windowed_decode_warm_cache=True, vae_decode_channels_last=True,
            windowed_decode_latents_per_call=0, vae_decode_cudnn_benchmark=False),
    5: dict(windowed_decode_warm_cache=True, vae_decode_channels_last=True,
            windowed_decode_latents_per_call=0, vae_decode_cudnn_benchmark=True),
}


def node_overrides(variant: str) -> tuple[dict, bool] | None:
    """(config overrides, compiled) for an in-tree node variant, or None."""
    m = re.fullmatch(r"mstar_(warm|warm_bench|step(\d))(?:_lpc(\d+))?(_eager)?", variant)
    if not m:
        return None
    if m[2] is not None:
        over = dict(STEPS[int(m[2])])
    else:  # the node's config defaults, or with cudnn.benchmark on
        over = {"vae_decode_cudnn_benchmark": True} if m[1] == "warm_bench" else {}
    if m[3] is not None:
        over["windowed_decode_latents_per_call"] = int(m[3])
    return over, not m[4]


def mstar_warm_node(raw, device, compiled, n_steady, overrides=None):
    """``Cosmos3VAEDecoderARSubmodule`` at its config defaults (or with
    ``overrides``), and ``(start, chunk)`` closures streaming ``raw``
    through it as one request of ``1 + n_steady`` windows."""
    sys.path.insert(0, mstar_path())
    os.environ["COSMOS3_COMPILE_VAE"] = "1" if compiled else "0"
    from types import SimpleNamespace

    from mstar.model.cosmos3 import constants as C
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderARSubmodule

    vae = load_vae(REPO, torch.float32, device)
    cfg = Cosmos3Config()
    for k, val in (overrides or {}).items():
        setattr(cfg, k, val)
    sub = Cosmos3VAEDecoderARSubmodule(vae=vae, config=cfg)
    mean, std = latent_stats(vae, "cpu")
    norm = ((raw - mean) / std).to(device)
    md = {
        "num_windows": 1 + n_steady,
        "overlap_latent_units": 0,
        "num_frames": 1 + n_steady * FRAMES_PER_CHUNK,
        "stream_video": True,
    }
    infos = SimpleNamespace(
        request_ids=["bench"],
        per_request_info={"bench": SimpleNamespace(step_metadata=md)},
    )

    def start():
        sub.cleanup_request("bench")
        return sub.forward(C.VIDEO_DECODE_AR_WALK, infos, norm[:, :, :1])["video_output"][0]

    def chunk(c):
        return sub.forward(C.VIDEO_DECODE_AR_WALK, infos, norm[:, :, 1 + 3 * c: 4 + 3 * c])["video_output"][0]

    return sub, start, chunk


def run_mstar_warm(args, raw, ref, compiled, overrides):
    """In-tree streaming decode: ``Cosmos3VAEDecoderARSubmodule`` with the
    serving defaults this change lands (warm conv cache, channels-last, one
    latent per decoder call) or a ladder step's switches.
    ``--dump-u8``: the first replay's uint8 frames (the whole stream), written
    outside the timed region."""
    n_steady = args.warmup + args.chunks
    sub, start, chunk = mstar_warm_node(raw, args.device, compiled, n_steady, overrides=overrides)
    lat, quality_u8 = [], []
    mem, t_compile = {}, None
    dump = None
    with torch.inference_mode():
        for rep in range(args.repeats):
            first = start()
            if rep == 0 and args.dump_u8:
                h, w = first.shape[3:]
                dump = np.lib.format.open_memmap(args.dump_u8, "w+", np.uint8,
                                                 (1 + n_steady * FRAMES_PER_CHUNK, h, w, 3))
                dump[0] = first[0, :, 0].permute(1, 2, 0).cpu().numpy()
            for c in range(n_steady):
                if rep == 0 and c == args.warmup:
                    reset_peak(args.device)
                with Timer(args.device) as t:
                    pixels = chunk(c)
                if rep == 0 and c == 0:
                    t_compile = t.ms
                if dump is not None and rep == 0:
                    a = 1 + c * FRAMES_PER_CHUNK
                    dump[a: a + FRAMES_PER_CHUNK] = pixels[0].permute(1, 2, 3, 0).cpu().numpy()
                if c < args.warmup:
                    continue
                lat.append(t.ms)
                k = c - args.warmup
                if rep == 0 and k < QUALITY_CHUNKS and ref is not None:
                    quality_u8.append(psnr(pixels.float().cpu(), to_uint8(ref["float"][k]).float(), 255.0))
            if rep == 0:
                mem = peak(args.device)
    if dump is not None:
        dump.flush()
    cfg = sub.config
    mem["decode_dtype"] = str(sub._decode_dtype())
    mem["steady_decode_latents"] = 3 if cfg.windowed_decode_warm_cache else 3 + cfg.windowed_decode_context_latents
    mem["node_config"] = {k: getattr(cfg, k) for k in STEPS[0]}
    return lat, [], quality_u8, mem, t_compile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--ref", help="ref_fp32 outputs (.pt) for PSNR")
    ap.add_argument("--save-ref", help="(ref_fp32 only) write first steady chunks' float outputs here")
    ap.add_argument("--out", required=True, help="results dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=75)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--dump-u8", help="(in-tree node variants) write the first replay's uint8 frames here (.npy)")
    args = ap.parse_args()

    global REPO
    REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    torch._dynamo.config.cache_size_limit = 64
    if not args.device.startswith("cuda"):  # CPU smoke runs: the Wan22 node reads free VRAM
        torch.cuda.mem_get_info = lambda *a, **k: (1 << 40, 1 << 40)
    raw = torch.load(args.latents)["raw"]
    ref = torch.load(args.ref) if args.ref and Path(args.ref).exists() else None
    v = args.variant
    t0 = time.time()
    if node := node_overrides(v):
        res = run_mstar_warm(args, raw, ref, compiled=node[1], overrides=node[0])
    else:
        res = run_streaming(args, raw, ref)
    lat, qf, qu8, extra, t_first = res

    med = statistics.median(lat)
    finite_u8 = [q for q in qu8 if math.isfinite(q)] or [float("inf")]
    row = {
        "variant": v,
        "n_samples": len(lat),
        "median_ms": round(med, 2),
        "p90_ms": round(pct(lat, 0.9), 2),
        "p10_ms": round(pct(lat, 0.1), 2),
        "mean_ms": round(statistics.fmean(lat), 2),
        "decoder_fps": round(FRAMES_PER_CHUNK / (med / 1e3), 1),
        "psnr_float_db_vs_fp32": round(statistics.fmean(qf), 2) if qf else None,
        "psnr_float_db_vs_fp32_peak1": round(statistics.fmean(qf) - 20 * math.log10(2), 2) if qf else None,
        "psnr_u8_db_vs_fp32": round(statistics.fmean(finite_u8), 2) if qu8 else None,
        "first_timed_call_ms": round(t_first, 1) if t_first else None,
        "wall_s": round(time.time() - t0, 1),
        **extra,
        "precision_flags": precision_flags(),
        "env": env_info(),
        "latents": str(args.latents),
        "vae": REPO,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{v}.json").write_text(json.dumps(row, indent=2))
    (out / f"{v}.latencies.json").write_text(json.dumps(lat))
    print(json.dumps({k: row[k] for k in row if k not in ("env",)}))


if __name__ == "__main__":
    main()
