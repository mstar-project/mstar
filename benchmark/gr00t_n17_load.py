#!/usr/bin/env python3
"""Closed-loop load on a running GR00T N1.7 server, M* or NVIDIA's own: the recorded DROID requests,
sent as each system's client sends them, at each concurrency in turn. Per level: latency median,
p95 and mean, throughput, and failed requests. Requests are encoded before the clock starts.

    --system mstar   `mstar serve gr00t_n17` over HTTP: PNG frames and kwargs, as
                     `test/gr00t_n17/serve_check.py`'s `observation` builds them. Run in M*'s venv.
    --system nvidia  NVIDIA's policy server (Isaac-GR00T's `gr00t/eval/run_gr00t_server.py`) over
                     ZeroMQ, through its own `PolicyClient.get_action` on the recorded `raw_obs`.
                     Run in Isaac-GR00T's venv (it imports `gr00t`).

Each level runs 3 warmup waves (3c requests, discarded), then max(10, 5c) measured requests; a failed
request is counted, never dropped. Each in-flight request has its own client. `--text-rows` keeps
only the recorded requests whose prompt is that many text rows (one DT bucket).

The requests are `test/gr00t_n17/record_dumps.py`'s, in `$GR00T_N17_ORACLE_DIR/dump`. The PR's
numbers (one GH200, three boots of each system, interleaved, one at a time on the GPU):

    mstar serve gr00t_n17 --port 18620
    python benchmark/gr00t_n17_load.py --system mstar --port 18620 --out /tmp/gr00t_load/mstar

    cd ~/ref/Isaac-GR00T && .venv/bin/python gr00t/eval/run_gr00t_server.py \\
        --model-path <GR00T-N1.7-3B snapshot> --embodiment-tag oxe_droid_relative_eef_relative_joint --port 18630
    cd ~/ref/Isaac-GR00T && .venv/bin/python <mstar>/benchmark/gr00t_n17_load.py --system nvidia --port 18630 \\
        --out /tmp/gr00t_load/nvidia

and the same two with `--text-rows 21` for one prompt length.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("GR00T_N17_ORACLE_DIR", Path.home() / "gr00t-runs"))


def requests(dump: Path, text_rows: int | None) -> list[dict]:
    dumps = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(dump.glob("req_*.pt"))]
    if not dumps:
        raise SystemExit(f"no recorded requests in {dump}")
    if text_rows is None:
        return dumps
    # the image token from the backbone's config, so the NVIDIA arm needs no M* import
    from huggingface_hub.constants import HF_HUB_CACHE

    cfg = next(iter(glob.glob(f"{HF_HUB_CACHE}/models--nvidia--Cosmos-Reason2-2B/snapshots/*/config.json")))
    image_token = json.loads(Path(cfg).read_text())["image_token_id"]
    kept = [d for d in dumps if int((d["collated"]["inputs"]["input_ids"][0] != image_token).sum()) == text_rows]
    if not kept:
        raise SystemExit(f"no recorded request has {text_rows} text rows")
    return kept


def mstar_client(port: int, dumps: list[dict]):
    sys.path.insert(0, str(REPO / "test/gr00t_n17"))
    from serve_check import observation, send

    from mstar.client import MStarClient

    url = f"http://127.0.0.1:{port}"
    if not MStarClient(url).health():
        raise SystemExit(f"no M* server at {url}")

    def call(client, obs):
        return send(client, obs, seed=7)

    return [observation(d) for d in dumps], lambda: MStarClient(url), call


def nvidia_client(port: int, dumps: list[dict]):
    from gr00t.policy.server_client import PolicyClient

    if not PolicyClient(host="127.0.0.1", port=port, timeout_ms=60000).ping():
        raise SystemExit(f"no NVIDIA policy server on port {port}")

    def call(client, obs):
        action, _ = client.get_action(obs)
        if isinstance(action, dict) and "error" in action:
            raise RuntimeError(action["error"])
        return action

    return [d["raw_obs"] for d in dumps], lambda: PolicyClient(host="127.0.0.1", port=port, timeout_ms=600000), call


def level(obs: list, new_client, call, c: int, n: int) -> dict:
    """n requests, c in flight at a time."""
    lat, failed, lock, nxt = [], [], threading.Lock(), iter(range(n))

    def worker():
        client = new_client()
        while True:
            with lock:
                i = next(nxt, None)
            if i is None:
                return
            t = time.perf_counter()
            try:
                call(client, obs[i % len(obs)])
            except Exception as exc:  # counted, not dropped: a failed request is a finding
                with lock:
                    failed.append(f"{type(exc).__name__}: {exc}"[:200])
                continue
            with lock:
                lat.append((time.perf_counter() - t) * 1e3)

    threads = [threading.Thread(target=worker) for _ in range(c)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    lat.sort()

    def pct(q):
        return lat[min(len(lat) - 1, int(q * (len(lat) - 1)))] if lat else None

    return {"concurrency": c, "requests": n, "completed": len(lat), "failed": len(failed),
            "failures": failed[:5], "median_ms": statistics.median(lat) if lat else None,
            "p95_ms": pct(0.95), "mean_ms": statistics.mean(lat) if lat else None,
            "throughput_rps": len(lat) / wall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["mstar", "nvidia"], required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dump", type=Path, default=RUNS / "dump", help="the recorded requests (default: %(default)s)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="concurrencies, in order")
    ap.add_argument("--min-requests", type=int, default=10, help="measured requests per level: max(this, 5c)")
    ap.add_argument("--warmup-waves", type=int, default=3)
    ap.add_argument("--text-rows", type=int, help="only the recorded requests with this many text rows")
    ap.add_argument("--out", type=Path, required=True, help="folder for load_<system>[_<rows>].json")
    ap.add_argument("--label", default="", help="written into the JSON, e.g. 'mixed boot 2'")
    args = ap.parse_args()

    dumps = requests(args.dump, args.text_rows)
    build = mstar_client if args.system == "mstar" else nvidia_client
    obs, new_client, call = build(args.port, dumps)
    results = []
    for c in args.levels:
        warm = level(obs, new_client, call, c, args.warmup_waves * c)
        r = level(obs, new_client, call, c, max(args.min_requests, 5 * c))
        r["warmup_failed"] = warm["failed"]
        results.append(r)
        print(f"{args.system} c={c}  median {r['median_ms'] or float('nan'):7.1f} ms  "
              f"p95 {r['p95_ms'] or float('nan'):7.1f} ms  {r['throughput_rps']:6.2f} req/s  "
              f"failed {r['failed']}/{r['requests']} (warmup {warm['failed']})", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"load_{args.system}{'_' + str(args.text_rows) if args.text_rows else ''}.json"
    out.write_text(json.dumps({"system": args.system, "label": args.label, "requests_pool": len(obs),
                               "text_rows": args.text_rows, "levels": results}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
