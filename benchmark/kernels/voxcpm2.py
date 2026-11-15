"""Paired B1 complete-patch CUDA Graph timing, standard O2 versus the native solve.

Run with CUTLASS_ROOT and a local checkpoint; see installation.rst. Optional
--fixtures holds the recorded inputs-b1.pt list. This measures
the solve, not serving throughput or audio quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def replay_ms(graph, replays):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


@torch.inference_mode()
def main():
    from mstar.model.voxcpm2.components.local import cfm_t_span
    from mstar.model.voxcpm2.config import VoxCPM2Config
    from mstar.model.voxcpm2.voxcpm2_model import load_generator
    from mstar.utils.voxcpm2_solve import load_native

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model = load_generator(VoxCPM2Config.from_pretrained(args.checkpoint), str(args.checkpoint),
                           "cuda", torch.bfloat16).to(dtype=torch.bfloat16)
    assert model.patch_solver is not None, "Native specialization must be enabled"
    solver = model.patch_solver
    binary = Path(load_native().__file__)
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "binary": str(binary), "checkpoint": str(args.checkpoint), "batches": {}}
    if args.fixtures:
        xs = tuple(x.cuda() for x in torch.load(args.fixtures / "inputs-b1.pt",
                                               weights_only=True)[0])
    else:
        rng = torch.Generator(device="cuda").manual_seed(20261109)
        xs = (*(torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)
                for shape in ((1, 2048), (1, 2048), (1, 4, 64), (1, 64, 4))),
              torch.full((1,), 2.0, device="cuda"), cfm_t_span(10, "cuda", torch.bfloat16))
    native, yn = capture(partial(solver, *xs))
    model.patch_solver = None
    compiled = torch.compile(model.predict_patch, fullgraph=True, mode="max-autotune-no-cudagraphs")
    baseline, yb = capture(partial(compiled, *xs))
    model.patch_solver = solver
    assert torch.isfinite(yn).all() and torch.isfinite(yb).all()
    samples = {"standard_O2": [], "native": []}
    graphs = [("standard_O2", baseline), ("native", native)]
    for round_ in range(args.rounds):
        for name, graph in graphs[::1 if round_ % 2 == 0 else -1]:
            samples[name].append(replay_ms(graph, args.replays))
    report["batches"]["1"] = {
        name: {"median_ms": statistics.median(values), "samples_ms": values}
        for name, values in samples.items()
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"batch": 1, **report["batches"]["1"]}), flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
