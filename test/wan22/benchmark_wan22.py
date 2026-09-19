#!/usr/bin/env python3
"""Drive the wan22 video benchmark over its grid.

Sweeps the grid and calls ``run_cell`` in ``benchmark/wan22/video_bench_wan22.py``
for each cell, writing one config-complete row per cell to JSON and CSV. The
measurement itself lives there, not here.

Grid: 480x832 x 33 frames, steps {20, 50}, concurrency {1, 2}, 10 VBench prompts,
one warmup per cell excluded from every metric.

    PYTHONPATH=. python test/wan22/benchmark_wan22.py --engine ours --port 8100 \
        --log-stats-file $SCRATCH/wan22_stats.txt \
        --out-json $SCRATCH/wan22.json --label mstar

Baselines need --server-python (their venv), or the row records this client's torch as
the server's. Read the run order at the top of benchmark/wan22/reproduce.sh before
comparing engines; baseline setup is in benchmark/{vllm_omni,sglang_omni}_instructions.md.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `benchmark.wan22...` importable when run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from _env import get_host_port  # noqa: E402  (sibling; shares .env with the other wan22 scripts)

from benchmark.wan22.video_bench_wan22 import (  # noqa: E402
    CellConfig,
    load_prompts,
    print_row,
    probe_gpu,
    probe_server_versions,
    run_cell,
    write_rows,
)

# The grid lives here, not in the engine: the engine stays a general video-bench
# tool and this driver owns which cells get swept.
DEFAULT_SIZE = "832x480"        # WxH == the 480x832 (HxW) grid cell
DEFAULT_FRAMES = 33
DEFAULT_STEPS = [20, 50]
DEFAULT_CONCURRENCY = [1, 2]
DEFAULT_NUM_PROMPTS = 10


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env_host, env_port = get_host_port()  # from test/wan22/.env if present, else 127.0.0.1:8100
    ap.add_argument("--engine", choices=["ours", "vllm", "sglang"], default="ours")
    ap.add_argument("--host", default=env_host)
    ap.add_argument("--port", type=int, default=env_port)
    ap.add_argument("--model", default="wan22")
    ap.add_argument("--size", default=DEFAULT_SIZE)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="steps to sweep (default 20 50); pass one value for a single cell")
    ap.add_argument("--concurrency", type=int, nargs="*", default=None,
                    help="concurrency to sweep (default 1 2)")
    ap.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompts-file", default=None, help="one prompt per line (else embedded VBench 10)")
    ap.add_argument("--image", default="", help="i2v conditioning frame (switches the grid to i2v)")
    ap.add_argument("--log-stats-file", default="", help="server --log-stats-file (ours: phase source)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--vram-interval", type=float, default=0.05)
    ap.add_argument("--poll-interval", type=float, default=0.02,
                    help="poll tick for the async baselines (s); biases their e2e upward by half this")
    ap.add_argument("--server-python", default="",
                    help="interpreter of the SERVER's venv — stamps the SERVER's torch/cuDNN "
                         "into every row instead of this client's")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--label", default="mstar")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    steps_grid = args.steps if args.steps else DEFAULT_STEPS
    conc_grid = args.concurrency if args.concurrency else DEFAULT_CONCURRENCY
    prompts = load_prompts(args.prompts_file, args.num_prompts)

    image_data_uri = ""
    if args.image and args.engine == "ours":
        import base64
        with open(args.image, "rb") as f:
            image_data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    gpu_info = probe_gpu(args.gpu_index)
    # Probed once, from the SERVER's interpreter — not this client's (the baselines
    # run in their own venvs, and a client-stamped torch on a baseline row is a false
    # provenance record). Null when --server-python is not given.
    server_info = probe_server_versions(args.server_python)
    print(f"[wan22-bench] engine={args.engine} grid: size={args.size} frames={args.frames} "
          f"steps={steps_grid} concurrency={conc_grid} prompts={len(prompts)} "
          f"gpu={gpu_info.get('name')} cc={gpu_info.get('compute_cap')} "
          f"server_torch={server_info.get('server_torch_version') or '<unprobed>'} "
          f"client_torch={gpu_info.get('client_torch_version')}",
          flush=True)

    rows = []
    for steps in steps_grid:
        for conc in conc_grid:
            cfg = CellConfig(
                engine=args.engine, host=args.host, port=args.port, model=args.model,
                size=args.size, frames=args.frames, steps=steps, concurrency=conc,
                num_prompts=args.num_prompts, warmup=args.warmup, timeout_s=args.timeout,
                image_path=args.image, image_data_uri=image_data_uri,
                log_stats_file=args.log_stats_file,
                gpu_index=args.gpu_index, vram_interval_s=args.vram_interval,
                poll_interval_s=args.poll_interval, server_python=args.server_python,
                label=args.label,
            )
            row = run_cell(cfg, prompts, gpu_info, server_info)
            print_row(row)
            rows.append(row)
            # Persist incrementally so a mid-sweep crash keeps completed cells.
            write_rows([row], args.out_json, args.out_csv)

    print(f"\n[wan22-bench] done: {len(rows)} cells "
          f"({sum(r['ok'] for r in rows)}/{sum(r['total'] for r in rows)} requests ok)", flush=True)


if __name__ == "__main__":
    main()
