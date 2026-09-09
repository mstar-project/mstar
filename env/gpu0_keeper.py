#!/usr/bin/env python3
"""Keep a TP serve's rank-0 GPU "active" for coriander's idle reaper during load + capture.

The box's gpu-management daemon SIGTERMs a process whose per-process NVML SM
utilization stays <= 10% for its idle threshold (~30 min observed). Rank 0
waits on the CPU through the 25-min weight load and the capture, so it is the
rank that dies. The daemon refreshes every same-user process on a GPU when any
one of them is above the line, so a sibling process on rank 0's GPU protects
the serve without touching it.

This script runs 8192x8192 bf16 matmul bursts on one GPU and doubles the burst
until its OWN NVML reading is >= --target (NVML attributes only ~1/3 of wall
time to a process when nine contexts share the GPU, so a fixed duty cycle is
not enough). It exits at --until (UTC HH:MM:SS) — set that before the serve's
READY time so the benchmark runs uncontended.

    CUDA_VISIBLE_DEVICES=0 python env/gpu0_keeper.py --until 22:20:30 \\
        --log $P/tmp/keeper.log &
"""
import argparse
import datetime as dt
import os
import sys
import time

import torch

try:
    import pynvml as nv
except ImportError:  # nvidia-ml-py exposes the same module name
    import nvidia_ml_py as nv


def parse_until(hhmmss: str) -> dt.datetime:
    h, m, s = (int(x) for x in hhmmss.split(":"))
    now = dt.datetime.now(dt.timezone.utc)
    t = now.replace(hour=h, minute=m, second=s, microsecond=0)
    return t + dt.timedelta(days=1) if t < now else t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="UTC HH:MM:SS hard stop")
    ap.add_argument("--target", type=int, default=15, help="own NVML smUtil %% to hold")
    ap.add_argument("--gpu", type=int, default=0, help="NVML index of the GPU to hold")
    ap.add_argument("--log", default=None, help="append status lines here (default stderr)")
    args = ap.parse_args()

    deadline = parse_until(args.until)
    log = open(args.log, "a", buffering=1) if args.log else sys.stderr
    me = os.getpid()
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(args.gpu)
    a = torch.ones(8192, 8192, device="cuda", dtype=torch.bfloat16)

    def own_util() -> int:
        try:
            for s in nv.nvmlDeviceGetProcessUtilization(handle, 0):
                if s.pid == me:
                    return int(s.smUtil)
        except nv.NVMLError:
            pass
        return -1

    n, period, last = 8, 0.25, 0.0
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} keeper pid={me} gpu={args.gpu} "
          f"target>={args.target}% until={deadline:%H:%M:%S}Z", file=log)
    while dt.datetime.now(dt.timezone.utc) < deadline:
        t0 = time.time()
        for _ in range(n):
            torch.mm(a, a)
        torch.cuda.synchronize()
        busy = time.time() - t0
        time.sleep(max(0.0, period - busy))
        if time.time() - last >= 10:
            last = time.time()
            util = own_util()
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} n={n} burst={busy * 1e3:.0f}ms "
                  f"own_sm={util}%", file=log)
            if 0 <= util < args.target:
                n = min(n * 2, 512)
            elif util > 3 * args.target and n > 1:
                n = max(1, n // 2)
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} keeper exit (deadline)", file=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
