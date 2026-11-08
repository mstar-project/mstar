#!/usr/bin/env python3
"""Load every GPU for a minute and fail if one is thermally throttling or lags the others.

usage: gpu_health.py [--seconds 60]      (runs on the box; needs any python with torch)

Only the GPUs this process can see are loaded and judged, numbered as torch numbers them.

A TP/SP instance runs at the pace of its slowest rank, so one badly cooled GPU slows every
engine on the box, by a different amount for each, and invalidates absolute and relative
numbers alike. Seen on a RunPod 4x H200: GPU 0 at 86 C and 585-1185 MHz under load while
the other three sat at ~35 C and 1980 MHz.
"""
import argparse
import subprocess
import sys
import threading
import time

THERMAL = 0x20 | 0x40  # SW / HW thermal slowdown in clocks_throttle_reasons.active


def burn(i: int, stop: threading.Event) -> None:
    import torch

    torch.cuda.set_device(i)
    a = torch.randn(8192, 8192, device=f"cuda:{i}", dtype=torch.bfloat16)
    b = a.clone()
    while not stop.is_set():
        for _ in range(50):
            a @ b
        torch.cuda.synchronize(i)


def visible_uuids() -> list[str]:
    """The GPUs this process can see (CUDA_VISIBLE_DEVICES, a Slurm or shared box), in torch's order,
    by uuid: nvidia-smi lists every GPU on the host under its physical index."""
    import torch

    return [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]


def sample(uuids: list[str]) -> list[tuple[int, int, int, int]]:
    """(torch index, temperature, SM clock, throttle reasons) of each GPU the burn loaded."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,temperature.gpu,clocks.sm,clocks_throttle_reasons.active",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return by_torch_index(out, uuids)


def by_torch_index(out: str, uuids: list[str]) -> list[tuple[int, int, int, int]]:
    rows = {}
    for line in out.strip().splitlines():
        uuid, *vals = (v.strip() for v in line.split(","))
        rows[uuid.removeprefix("GPU-").lower()] = tuple(int(v, 0) for v in vals)
    missing = [u for u in uuids if u.lower() not in rows]
    if missing:
        raise SystemExit(f"gpu_health.py: nvidia-smi does not list the visible GPU(s) {missing}")
    return [(i, *rows[u.lower()]) for i, u in enumerate(uuids)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    stop = threading.Event()
    uuids = visible_uuids()
    threads = [threading.Thread(target=burn, args=(i, stop)) for i in range(len(uuids))]
    for t in threads:
        t.start()
    time.sleep(args.seconds / 2)  # let temperatures settle before judging
    rows = []
    for _ in range(5):
        rows.append(sample(uuids))
        time.sleep(args.seconds / 10)
    stop.set()
    for t in threads:
        t.join()

    bad = []
    last = rows[-1]
    for idx, temp, clock, reasons in last:
        print(f"GPU {idx}: {temp} C, {clock} MHz, throttle 0x{reasons:x}")
    for snap in rows:
        for idx, _, _, reasons in snap:
            if reasons & THERMAL:
                bad.append(f"GPU {idx} thermal throttle (0x{reasons:x})")
    temps = [r[1] for r in last]
    clocks = [r[2] for r in last]
    for idx, temp, clock, _ in last:
        if temp - min(temps) > 20:
            bad.append(f"GPU {idx} runs {temp - min(temps)} C hotter than the coolest")
        if clock < 0.75 * max(clocks):
            bad.append(f"GPU {idx} clock {clock} MHz is under 75% of the fastest ({max(clocks)} MHz)")
    if bad:
        print("UNHEALTHY: " + "; ".join(sorted(set(bad))))
        return 1
    print("healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
