"""Where a LingBot-World chunk's time goes, piece by piece, to compare two checkouts.

Runs one 400x976 rollout (8 chunks) through the dit node by hand, as
``run_served_numerics.py`` does, with a mark on the GPU stream at the start and end
of each piece of work: the resources' admit, plan and commit, the condition
encode, the DiT passes, the decode, and whatever a checkout adds around them
(found by name, so the script runs on any checkout of the model). Three rollouts
in one process:

  1. marks only: per chunk, the GPU time between consecutive marks, charged to
     the innermost piece open at the time. The model runs on one stream, so the
     pieces add up to the chunk, including time the GPU waits for the host.
  2. ``torch.profiler`` over chunks 2-6 (``--trace``): every GPU kernel and copy,
     charged to the piece that launched it, and the GPU's idle time before each.
  3. marks only again, on a warm process.

The checkout measured is the one ``mstar`` imports from, so one copy of this
script measures any checkout (its ``run_parity.py`` is used too):

    PYTHONPATH=<checkout> python test/lingbot_world/profile_chunk.py --out a.json --trace a.trace.json.gz
    python test/lingbot_world/profile_chunk.py --out a.json --analyse a.trace.json.gz   # no GPU
    python test/lingbot_world/profile_chunk.py --compare 'base_*.json' 'new_*.json'
"""

from __future__ import annotations

import argparse
import bisect
import collections
import functools
import glob
import gzip
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

import mstar  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(mstar.__file__)))
sys.path.insert(0, os.path.join(REPO, "test", "lingbot_world"))

from huggingface_hub import hf_hub_download  # noqa: E402
from run_parity import DEVICE, bind_resources, drive, open_request  # noqa: E402

import mstar.model.lingbot_world.submodules as subm  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import ROLLOUT_LOOP_NAME, ROLLOUT_WALK  # noqa: E402

PROMPT = "A quiet suburban street on a clear afternoon, the camera driving slowly forward."

# ---------------------------------------------------------------- marks
MARKS: list[tuple[str, torch.cuda.Event]] = []
RECORD_EVENTS = True


def mark(label: str) -> None:
    if RECORD_EVENTS:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        MARKS.append((label, e))


def wrapped(name: str, fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        mark("+" + name)
        with torch.profiler.record_function("phase:" + name):
            out = fn(*a, **k)
        mark("-" + name)
        return out
    return inner


def instrument(sub, runner) -> list[str]:
    done = []
    for name in ("encode_condition_chunk", "cache_from_pool", "cache_to_pool", "plucker_embedding"):
        if hasattr(subm, name):
            setattr(subm, name, wrapped(name, getattr(subm, name)))
            done.append(name)
    for name in ("_decode", "_decoder_cache", "_encoder_stream", "_keep_encoder_stream", "_keep_settled_output",
                 "_seed_request"):
        if hasattr(sub, name):
            setattr(sub, name, wrapped(name, getattr(sub, name)))
            done.append(name)
    sub.dit.forward = wrapped("dit_pass", sub.dit.forward)
    sub.dit.chunk_conditioning = wrapped("chunk_conditioning", sub.dit.chunk_conditioning)
    sub.forward = wrapped("forward", sub.forward)
    sub.prepare_inputs = wrapped("prepare_inputs", sub.prepare_inputs)
    sub.preprocess = wrapped("preprocess", sub.preprocess)
    sub.declare_step = wrapped("declare_step", sub.declare_step)
    for name in ("admit", "plan", "commit"):
        setattr(runner, name, wrapped("runner." + name, getattr(runner, name)))
    return done + ["dit_pass", "chunk_conditioning", "forward", "prepare_inputs", "preprocess", "declare_step",
                   "runner.admit", "runner.plan", "runner.commit"]


def segments(marks) -> dict[str, float]:
    """Milliseconds between consecutive marks, each charged to the innermost
    piece of work open at the time (a piece's own code between its children is
    charged to the piece)."""
    out: dict[str, float] = collections.defaultdict(float)
    stack: list[str] = []
    for (la, ea), (_lb, eb) in zip(marks, marks[1:], strict=False):
        if la.startswith("+"):
            stack.append(la[1:])
        else:
            stack.pop()
        out[stack[-1] if stack else "outside"] += ea.elapsed_time(eb)
    return dict(out)


# ---------------------------------------------------------------- rollout
def trajectory(num_frames: int, yaw_per_frame: float, step: float) -> np.ndarray:
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    for t in range(num_frames):
        a = yaw_per_frame * t
        poses[t, :3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        poses[t, :3, 3] = [step * t * math.sin(a), 0.0, step * t * math.cos(a)]
    return poses


def make_session(rid, seed, tensors, text_embeds, cs):
    lat_f = int(tensors["camera_poses"][0].shape[0])
    md = {"is_prefill": False, "guidance_scale": 1.0, "num_latent_frames": lat_f, "num_chunks": lat_f // cs}
    info = CurrentForwardPassInfo(request_id=f"prof-{rid}", graph_walk=ROLLOUT_WALK, fwd_index=0, random_seed=seed,
                                  max_tokens=0, step_metadata=md, rid_handle=rid)
    info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = 0  # the engine counts the loop; the DiT reads its chunk there
    inputs = {"text_embeds_pos": [text_embeds], "image_inputs": [tensors["image_inputs"][0].to(DEVICE)],
              "camera_poses": tensors["camera_poses"], "intrinsics": tensors["intrinsics"], "clock": []}
    return info, inputs, lat_f // cs


def run_rollout(sub, runner, rid, tensors, text_embeds, cs, on_chunk=None):
    info, inputs, n = make_session(rid, 42, tensors, text_embeds, cs)
    open_request(sub, runner, rid, inputs["image_inputs"][0], info.step_metadata["num_latent_frames"])
    walls, latents = [], []
    for k in range(n):
        if on_chunk:
            on_chunk(k, "before")
        torch.cuda.synchronize()
        MARKS.clear()
        mark("+chunk")
        t0 = time.perf_counter()
        out = drive(sub, ROLLOUT_WALK, info, inputs, runner)
        mark("-chunk")
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        seg = segments(MARKS) if RECORD_EVENTS else {}
        if on_chunk:
            on_chunk(k, "after", seg)
        latents.append(sub.request_state(rid)["chunk_latents"].cpu())
        info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = k + 1
        inputs["clock"] = out["clock"]
    runner.remove_request(rid)
    sub.cleanup_request(rid)
    return walls, latents


# ---------------------------------------------------------------- trace analysis
def analyse_trace(path: str) -> dict:
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        events = json.load(f)["traceEvents"]
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu = [e for e in events if e.get("cat") in gpu_cats and e.get("ph") == "X"]
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"
                and "correlation" in e.get("args", {})}
    ranges = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"
              and e["name"].startswith("phase:")]
    ranges.sort(key=lambda e: (e["ts"], -e["dur"]))

    def innermost(ts: float, tid) -> str:
        best = None
        for r in ranges:
            if r["ts"] > ts:
                break
            if r["tid"] == tid and r["ts"] <= ts <= r["ts"] + r["dur"]:
                if best is None or r["dur"] <= best["dur"]:
                    best = r
        return best["name"][6:] if best else "outside"

    by_phase: dict[str, float] = collections.defaultdict(float)
    by_phase_count: dict[str, int] = collections.defaultdict(int)
    by_kernel: dict[str, list] = {}
    by_phase_kernel: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(float))
    for g in gpu:
        launch = launches.get(g["args"].get("correlation"))
        phase = innermost(launch["ts"], launch["tid"]) if launch else "unknown"
        dur = g["dur"] / 1e3
        by_phase[phase] += dur
        by_phase_count[phase] += 1
        name = g["name"][:120]
        entry = by_kernel.setdefault(name, [0, 0.0])
        entry[0] += 1
        entry[1] += dur
        by_phase_kernel[phase][name] += dur
    # GPU idle inside each top-level forward piece: span of the piece's GPU work minus busy time
    busy = sorted((g["ts"], g["ts"] + g["dur"]) for g in gpu)
    merged = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    span = (merged[-1][1] - merged[0][0]) / 1e3 if merged else 0.0
    busy_ms = sum(e - s for s, e in merged) / 1e3
    gaps = sorted(((merged[i + 1][0] - merged[i][1]) / 1e3, merged[i][1]) for i in range(len(merged) - 1))[::-1]
    # what was launched right after each of the largest gaps
    gpu_sorted = sorted(gpu, key=lambda g: g["ts"])
    starts = [g["ts"] for g in gpu_sorted]
    big_gaps = []
    for ms, end_ts in gaps[:40]:
        i = bisect.bisect_right(starts, end_ts)
        if i < len(gpu_sorted):
            g = gpu_sorted[i]
            launch = launches.get(g["args"].get("correlation"))
            big_gaps.append({"gap_ms": ms, "next": g["name"][:80],
                             "next_phase": innermost(launch["ts"], launch["tid"]) if launch else "unknown"})
    gap_by_phase: dict[str, float] = collections.defaultdict(float)
    for ms, end_ts in gaps:
        i = bisect.bisect_right(starts, end_ts)
        if i < len(gpu_sorted):
            g = gpu_sorted[i]
            launch = launches.get(g["args"].get("correlation"))
            gap_by_phase[innermost(launch["ts"], launch["tid"]) if launch else "unknown"] += ms
    runtime = collections.Counter()
    runtime_ms = collections.defaultdict(float)
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X":
            runtime[e["name"]] += 1
            runtime_ms[e["name"]] += e["dur"] / 1e3
    return {
        "gpu_span_ms": span, "gpu_busy_ms": busy_ms, "gpu_idle_ms": span - busy_ms,
        "gpu_ms_by_phase": dict(by_phase), "gpu_ops_by_phase": dict(by_phase_count),
        "idle_before_phase_ms": dict(gap_by_phase),
        "top_kernels": sorted(([k, v[0], v[1]] for k, v in by_kernel.items()), key=lambda x: -x[2])[:60],
        "kernels_by_phase": {p: sorted(d.items(), key=lambda x: -x[1])[:25] for p, d in by_phase_kernel.items()},
        "largest_gaps": big_gaps,
        "runtime_calls": {k: [runtime[k], runtime_ms[k]] for k in runtime},
    }


# ---------------------------------------------------------------- comparison
CHUNKS = range(1, 8)


def _mean_segments(runs: list[dict]) -> dict[str, float]:
    acc = collections.defaultdict(list)
    for r in runs:
        for key in ("segments_ms_pass1", "segments_ms_pass3"):
            for k in CHUNKS:
                for name, ms in r[key][k].items():
                    acc[name].append(ms)
    n = 2 * len(CHUNKS) * len(runs)
    return {name: sum(v) / n for name, v in acc.items()}


def compare(a_glob: str, b_glob: str) -> None:
    """Print per-piece means of two sets of runs (``--out`` files, traces analysed)."""
    sides = {}
    for label, pattern in (("A", a_glob), ("B", b_glob)):
        sides[label] = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
        if not sides[label]:
            raise SystemExit(f"no files match {pattern}")
    print(f"A = {a_glob} ({len(sides['A'])} runs), B = {b_glob} ({len(sides['B'])} runs)")
    print("\nwall ms, chunks 1-7 summed (first rollout / warm third rollout):")
    for label, runs in sides.items():
        for r in runs:
            p1, p3 = (sum(r[key][k] for k in CHUNKS) for key in ("wall_ms_pass1", "wall_ms_pass3"))
            print(f"  {label}  {p1:8.1f}  {p3:8.1f}")
    seg = {label: _mean_segments(runs) for label, runs in sides.items()}
    names = sorted(set(seg["A"]) | set(seg["B"]), key=lambda n: -max(seg["A"].get(n, 0), seg["B"].get(n, 0)))
    print("\nGPU-timeline ms per chunk by piece (mean of chunks 1-7, both marked rollouts):")
    print(f"  {'piece':28s} {'A':>9s} {'B':>9s} {'B-A':>8s}")
    for n in names:
        a, b = seg["A"].get(n, 0.0), seg["B"].get(n, 0.0)
        print(f"  {n:28s} {a:9.3f} {b:9.3f} {b - a:+8.3f}")
    print(f"  {'total':28s} {sum(seg['A'].values()):9.3f} {sum(seg['B'].values()):9.3f} "
          f"{sum(seg['B'].values()) - sum(seg['A'].values()):+8.3f}")
    traces = {label: [r["trace"] for r in runs if r.get("trace")] for label, runs in sides.items()}
    if not all(traces.values()):
        return
    chunks = 5
    print("\nprofiled chunks 2-6, per chunk: GPU work launched by each piece (ms, ops) and GPU idle before it (ms):")
    print(f"  {'piece':28s} {'A ms':>9s} {'B ms':>9s} {'A ops':>7s} {'B ops':>7s} {'A idle':>7s} {'B idle':>7s}")
    phases = set().union(*[set(t["gpu_ms_by_phase"]) | set(t["idle_before_phase_ms"])
                           for ts in traces.values() for t in ts])

    def mean(label, field, p):
        return statistics.mean(t[field].get(p, 0) for t in traces[label]) / chunks

    for p in sorted(phases, key=lambda p: -max(mean("A", "gpu_ms_by_phase", p), mean("B", "gpu_ms_by_phase", p))):
        print(f"  {p:28s} {mean('A', 'gpu_ms_by_phase', p):9.3f} {mean('B', 'gpu_ms_by_phase', p):9.3f} "
              f"{mean('A', 'gpu_ops_by_phase', p):7.0f} {mean('B', 'gpu_ops_by_phase', p):7.0f} "
              f"{mean('A', 'idle_before_phase_ms', p):7.3f} {mean('B', 'idle_before_phase_ms', p):7.3f}")



def main():
    global RECORD_EVENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A_GLOB", "B_GLOB"), help="compare two sets of --out files")
    ap.add_argument("--trace")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--trace-window", default="2,6", help="first,last chunk of the profiled rollout to trace")
    ap.add_argument("--analyse", help="analyse an exported trace (no GPU) and merge it into --out")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.analyse:
        res = json.load(open(args.out)) if os.path.exists(args.out) else {}
        res["trace"] = analyse_trace(args.analyse)
        json.dump(res, open(args.out, "w"), indent=1)
        return

    image_path = hf_hub_download("nvidia/Cosmos3-Edge", "assets/edge_action_id_av_0_output.png")
    model = LingBotWorldModel()
    cfg = model.config
    cs = cfg.chunk_size
    num_frames = (args.chunks * cs - 1) * 4 + 1
    image = model.load_image(image_path, "cpu").data
    tensors = model.process_prompt(
        PROMPT, ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
        camera_poses=trajectory(num_frames, 0.002, 0.02), intrinsics=[480.0, 480.0, 416.0, 240.0],
        num_frames=num_frames).new_input_tensors

    text_sub = model.get_submodule("text_encoder", device="cuda").to(DEVICE)
    info = CurrentForwardPassInfo(request_id="text", graph_walk="encode_text", fwd_index=0, random_seed=0,
                                  max_tokens=0, step_metadata={"guidance_scale": 1.0}, rid_handle=0)
    text_embeds = drive(text_sub, "encode_text", info, {"text_inputs": tensors["text_inputs"]})["text_embeds_pos"][0]
    text_embeds = text_embeds.to(DEVICE)
    del text_sub, model._submodule_cache["text_encoder"]
    torch.cuda.empty_cache()

    sub = model.get_submodule("dit", device="cuda").to(DEVICE)
    sub.keep_chunk_outputs = True
    runner = bind_resources(model, sub, {})
    wrapped_names = instrument(sub, runner)

    # pass 1: marks only
    per_chunk = []
    walls1, lat1 = run_rollout(sub, runner, 1, tensors, text_embeds, cs,
                               on_chunk=lambda k, when, seg=None: per_chunk.append(seg) if when == "after" else None)

    # pass 2: profiler over chunks 2-6 (marks off, so the trace has no event records)
    RECORD_EVENTS = False
    result_trace = None
    walls2 = []
    if args.trace:
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA])

        first, last = (int(x) for x in args.trace_window.split(","))

        def on_chunk(k, when, seg=None):
            if k == first and when == "before":
                prof.__enter__()
            if k == last and when == "after":
                torch.cuda.synchronize()
                prof.__exit__(None, None, None)

        walls2, lat2 = run_rollout(sub, runner, 2, tensors, text_embeds, cs, on_chunk=on_chunk)
        prof.export_chrome_trace(args.trace)
        result_trace = {"same_latents_as_pass1": all(torch.equal(a, b) for a, b in zip(lat1, lat2, strict=True))}

    # pass 3: marks only again, on a warm process
    RECORD_EVENTS = True
    per_chunk3 = []
    walls3, lat3 = run_rollout(sub, runner, 3, tensors, text_embeds, cs,
                               on_chunk=lambda k, when, seg=None: per_chunk3.append(seg) if when == "after" else None)

    res = {"wall_ms_pass3": walls3, "segments_ms_pass3": per_chunk3,
           "pass3_same_latents": all(torch.equal(a, b) for a, b in zip(lat1, lat3, strict=True)),
           "repo": REPO, "wrapped": wrapped_names, "wall_ms_pass1": walls1, "segments_ms_pass1": per_chunk,
           "wall_ms_pass2_profiled": walls2, "pass2": result_trace,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wall ms pass1:", " ".join(f"{w:.1f}" for w in walls1))
    if walls2:
        print("wall ms pass2 (profiled):", " ".join(f"{w:.1f}" for w in walls2))
    print("wall ms pass3:", " ".join(f"{w:.1f}" for w in walls3))


if __name__ == "__main__":
    main()
