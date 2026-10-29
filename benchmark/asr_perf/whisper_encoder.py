"""Device time of the Whisper encoder's captured forward, per batch size, and where it goes.

Builds the native encoder (``mstar/model/whisper/components/encoder.py``) from the cached checkpoint as
``WhisperModel`` does, then for each batch size compiles ``forward`` the way the engine compiles a capture config
(``compile_kwargs`` at the compile level, under the platform's inductor overrides), captures one graph, and times
replays with events. ``--profile`` records replays under torch.profiler and groups the kernels by family.
Parity: every batch row's output against the same encoder run eagerly in float32 on that row.

The harness imports mstar from ``PYTHONPATH``, so one copy measures any tree:

  PYTHONPATH=<tree>:<pyext> python benchmark/asr_perf/whisper_encoder.py --batch-sizes 1,4,8,16 --profile \
      --out enc.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

FAMILIES = (
    ("attention", re.compile(r"fmha|flash|attn|aotriton|attention", re.I)),
    ("gemm", re.compile(r"Cijk_|gemm|hipblaslt|wvSplt|_mm_|matmul|addmm", re.I)),
    ("conv", re.compile(r"miopen|conv|igemm|im2col|winograd|naive_", re.I)),
    ("triton", re.compile(r"^triton_")),
    ("copy", re.compile(r"copy|memcpy|memset|fill", re.I)),
)


def family(name: str) -> str:
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def load_encoder(model: str, device: str, dtype: torch.dtype):
    from huggingface_hub import snapshot_download

    from mstar.model.loader.iterators import iter_safetensors_shards
    from mstar.model.whisper.components.encoder import WhisperEncoderModel
    from mstar.model.whisper.config import WhisperModelConfig

    local = snapshot_download(model, local_files_only=True)
    config = WhisperModelConfig.from_pretrained(local)
    with torch.device("meta"):
        encoder = WhisperEncoderModel(config)
    encoder = encoder.to(dtype).to_empty(device=device)
    prefix = "model.encoder."
    encoder.load_weights((k.removeprefix(prefix), v)
                         for k, v in iter_safetensors_shards(local, device=device, prefix=prefix))
    return encoder.eval(), config


def compiled(fn, bs: int, compile_: bool):
    if not compile_:
        return fn
    from mstar.engine.accelerator_graph_runner import compile_kwargs
    from mstar.platforms.compile import with_inductor_overrides

    # the encoder's capture config counts a window as one token (input_seq_len=1)
    return with_inductor_overrides(torch.compile(fn, **compile_kwargs(None)), bs, bs)


def capture(fn, static_in: torch.Tensor):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(static_in)
    return g, out


def time_replays(g, reps: int) -> list[float]:
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)


def profile_replays(g, reps: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    per_kernel = defaultdict(lambda: [0.0, 0])
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            per_kernel[evt.name][0] += evt.device_time / 1000 / reps
            per_kernel[evt.name][1] += 1
    fams = defaultdict(lambda: [0.0, 0])
    for name, (ms, n) in per_kernel.items():
        fams[family(name)][0] += ms
        fams[family(name)][1] += n // reps
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:15]
    return {
        "kernels_per_replay": sum(n for _, n in fams.values()),
        "families_ms": {f: round(ms, 4) for f, (ms, _) in sorted(fams.items(), key=lambda kv: -kv[1][0])},
        "families_n": {f: n for f, (_, n) in fams.items()},
        "top": [{"ms": round(ms, 4), "n": n // reps, "name": name[:140]} for name, (ms, n) in top],
        "kernels": [{"ms": round(ms, 4), "n": n // reps, "family": family(name), "name": name[:200]}
                    for name, (ms, n) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--batch-sizes", default="1,4,8,16")
    ap.add_argument("--no-compile", action="store_true", help="capture the eager forward (ENCODER_COMPILE off)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--parity-rows", type=int, default=2, help="rows per batch checked against float32 eager")
    ap.add_argument("--out")
    a = ap.parse_args()

    import mstar

    torch.manual_seed(0)
    dtype = torch.bfloat16
    encoder, config = load_encoder(a.model, "cuda", dtype)
    ref_encoder = None
    if a.parity_rows:
        ref_encoder, _ = load_encoder(a.model, "cuda", torch.float32)
    frames = encoder.expected_num_frames
    result = {"tree": str(Path(mstar.__file__).parent.parent), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "compile": not a.no_compile, "cells": []}
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}), flush=True)
    for bs in map(int, a.batch_sizes.split(",")):
        # log-mel-like features: the encoder's input is clamped log10 power scaled to about [-1, 1.5]
        feats = (torch.randn(bs, config.num_mel_bins, frames, device="cuda") * 0.4 + 0.2).to(dtype)
        t0 = time.time()
        fn = compiled(encoder.forward, bs, not a.no_compile)
        with torch.inference_mode():
            g, out = capture(fn, feats)
        build_s = time.time() - t0
        times = time_replays(g, a.reps)
        cell = {"bs": bs, "build_s": round(build_s, 1), "ms_p50": round(times[len(times) // 2], 4),
                "ms_min": round(times[0], 4), "ms_per_window": round(times[len(times) // 2] / bs, 4)}
        if a.parity_rows:
            errs = []
            with torch.inference_mode():
                for i in range(min(bs, a.parity_rows)):
                    ref = ref_encoder(feats[i:i + 1].float())[0]
                    d = (out[i].float() - ref).abs()
                    errs.append({"max_abs": round(d.max().item(), 4),
                                 "rel_l2": round((d.norm() / ref.norm()).item(), 6)})
            cell["parity"] = errs
        if a.profile:
            cell["profile"] = profile_replays(g, 5)
        print(json.dumps({k: v for k, v in cell.items() if k != "profile"} | (
            {"profile": {k: v for k, v in cell["profile"].items() if k != "kernels"}} if "profile" in cell else {})),
            flush=True)
        result["cells"].append(cell)
        del g, out
        torch.cuda.empty_cache()
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
