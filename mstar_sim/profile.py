"""Measure the simulator's ops on a real GPU and write a profile DB.

    python -m mstar_sim.profile --trace sim_trace.jsonl [more.jsonl ...] --out h100.json

Workflow: run the simulator with ``cost.start_trace(path)`` (every distinct op
and kernel it charges is appended to ``path``), copy the trace to a box with the
target GPU, run this, and point the simulator at the result with
``cost.configure(gpu, profile_db="h100.json")``. An existing ``--out`` DB is
extended: keys already in it are not measured again.

Each distinct aten op is rebuilt from its trace spec (float tensors uniform in
[0, 1), integer and bool tensors zero so every index is valid, the traced
scalars as they were) and timed with CUDA events: ``--warmup`` calls, then
``--inner`` back-to-back calls captured in one CUDA graph, replayed ``--reps``
times; the DB keeps the median per call, minus the GPU spec's per-node graph gap
(the simulator adds its own launch gap back). An op that can't be captured (it
syncs or allocates on the host) is timed eagerly instead, minus the eager gap;
for kernels shorter than the host's launch rate (~5 us) that over-estimates.
Inputs stay L2-resident across the repeated calls, as activations mostly are.

DB format (JSON)::

    {"format": "mstar_sim.profile/1", "gpu": "<torch.cuda.get_device_name()>",
     "torch": "...", "cuda": "...",
     "entries": {"<key>": {"seconds": <measured kernel time>,
                           "roofline": <cost model's estimate for the same op on this GPU>,
                           "spec": <the trace spec: {"op", "args", "kwargs"}>}}}

The key is :func:`mstar_sim.cost.spec_key` of the spec. ``roofline`` lets the
simulator scale a measured family to shapes it has not seen (see
:mod:`mstar_sim.cost`).

TODO: fused-kernel keys (``kernel:<name>``, the FlashInfer/Triton stand-ins) are
skipped. Their trace records only FLOPs and bytes, not the call's shapes, so the
real kernel (e.g. ``flashinfer.norm.rmsnorm`` or a ``BatchPrefillWithPagedKVCacheWrapper``
plan/run) can't be rebuilt from it. Measuring them needs the stand-ins to trace
their argument shapes; then this tool can call the real library when it imports.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics

import torch

from mstar_sim import cost, gpus


def read_trace(paths: list[str]) -> dict[str, dict]:
    """The distinct specs of one or more trace files, by key."""
    specs: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    specs.setdefault(entry["key"], entry["spec"])
    return specs


def build(x, device: torch.device):
    """Rebuild an argument encoded by :func:`mstar_sim.cost.encode` on ``device``."""
    if isinstance(x, list):
        return [build(i, device) for i in x]
    if not isinstance(x, dict):
        return x
    if "tensor" in x:
        dtype = getattr(torch, x["dtype"])
        if dtype.is_floating_point:
            make = torch.bfloat16 if dtype.itemsize == 1 else dtype  # no rand for fp8
            return torch.rand(x["tensor"], dtype=make, device=device).to(dtype)
        if dtype.is_complex:
            return torch.randn(x["tensor"], dtype=dtype, device=device)
        return torch.zeros(x["tensor"], dtype=dtype, device=device)
    if "torch" in x:
        return getattr(torch, x["torch"])
    if "device" in x:  # every traced op ran on the simulated GPU
        return device
    raise ValueError(f"can't rebuild a {x.get('opaque', x)!r} argument")


def op(name: str):
    """The OpOverload named like ``str(torch.ops.aten.mm.default)``: "aten.mm.default"."""
    namespace, packet, overload = name.split(".")
    return getattr(getattr(getattr(torch.ops, namespace), packet), overload)


def time_op(fn, gpu: gpus.GPU, *, warmup: int, reps: int, inner: int) -> float:
    """Median device seconds of one ``fn()`` call, launch gap excluded."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        run, gap, fixed = graph.replay, gpu.graph_node_s, gpu.graph_launch_s
    except RuntimeError:
        torch.cuda.synchronize()

        def run():
            for _ in range(inner):
                fn()

        gap, fixed = gpu.launch_s, 0.0
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append((start.elapsed_time(end) / 1e3 - fixed) / inner)
    return max(statistics.median(times) - gap, 0.0)


def write_db(path: str, gpu: gpus.GPU, entries: dict) -> None:
    data = {
        "format": cost.FORMAT, "gpu": gpu.name, "torch": torch.__version__, "cuda": torch.version.cuda,
        "entries": entries,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mstar_sim.profile", description=__doc__.split("\n")[0])
    parser.add_argument("--trace", nargs="+", required=True, help="trace files written by cost.start_trace")
    parser.add_argument("--out", required=True, help="profile DB to write (extended if it exists)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--inner", type=int, default=10, help="calls per timed graph replay")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("mstar_sim.profile measures kernels on a real CUDA GPU; none is visible")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gpu = gpus.lookup(torch.cuda.get_device_name(device))  # add a new GPU to mstar_sim/gpus.py first
    cost.configure(gpu.name)
    entries: dict = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        if existing.get("gpu") != gpu.name:
            raise SystemExit(f"{args.out} was measured on {existing.get('gpu')!r}, this is {gpu.name!r}")
        entries = existing["entries"]
    specs = read_trace(args.trace)
    skipped: collections.Counter = collections.Counter()
    todo = [(key, spec) for key, spec in specs.items() if key not in entries]
    for n, (key, spec) in enumerate(todo, 1):
        if "kernel" in spec:
            skipped["fused kernel (TODO)"] += 1
            continue
        try:
            func = op(spec["op"])
            call_args = build(spec["args"], device)
            call_kwargs = {k: build(v, device) for k, v in spec["kwargs"].items()}
            out = func(*call_args, **call_kwargs)
            seconds = time_op(
                lambda func=func, a=call_args, k=call_kwargs: func(*a, **k), gpu,
                warmup=args.warmup, reps=args.reps, inner=args.inner,
            )
            roofline = cost.roofline_seconds(func, call_args, call_kwargs, out)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            skipped[f"{spec['op']}: {type(e).__name__}"] += 1
            continue
        entries[key] = {"seconds": seconds, "roofline": roofline, "spec": spec}
        if n % 100 == 0:
            write_db(args.out, gpu, entries)
            print(f"{n}/{len(todo)} measured", flush=True)
    write_db(args.out, gpu, entries)
    print(f"{args.out}: {len(entries)} entries on {gpu.name}; {len(specs) - len(todo)} were already measured")
    for reason, count in skipped.most_common():
        print(f"  skipped {count}: {reason}")


if __name__ == "__main__":
    main()
