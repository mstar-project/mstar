"""What a persistent step kernel pays to synchronise, and what one XCD can stream (ROCm).

    python -m benchmark.kernels.probes.sync --out DIR [--quick]

Builds ``sync_probe.cu`` through ``JitExtension`` and measures on this
process's GPU:

- placement: each workgroup's XCD and CU, to check that workgroup ``i`` runs
  on XCD ``i % xcds`` (the round-robin dispatch the XCD-local cases assume);
- one handoff (``pingpong``): a producer releases a flag, after an optional
  payload, and a consumer acquires it and checks the payload, between two
  workgroups on one XCD or on two, by scope: ``agent`` (the HIP memory model),
  ``xcd`` and ``xcd_load`` (only the shared L2 of one XCD; across XCDs these
  are the negative control of the payload check);
- barriers: the whole grid, or each XCD on its own, by algorithm (one counter,
  flags gathered by a leader, flags polled by every member) and scope;
- streaming reads by reader set: every XCD reading its slice (aggregate), one
  XCD alone, and every XCD reading the whole buffer (what a loop that splits
  rows across XCDs does with its weights), at buffer sizes that live in L2,
  in the memory-side cache and in HBM.

Times come from the device's constant-rate wall clock inside the kernels
(handoffs, barriers) or from events around a launch (streams). Results go to
``DIR/probe.json`` and a table to ``DIR/probe.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from mstar.kernels.hip.build import JitExtension

HERE = Path(__file__).resolve().parent
SCOPES = {"agent": 0, "xcd": 1, "xcd_load": 2}
ALGOS = {"counter": 0, "gather": 1, "all": 2}
MiB = 1 << 20


def extension() -> JitExtension:
    return JitExtension("_mstar_sync_probe", [HERE / "sync_probe.cu"])


def ops():
    extension().load()
    return torch.ops.mstar_sync_probe


def decode_hw_id(hw: int) -> dict[str, int]:
    """The fields of gfx9's HW_REG_HW_ID that place a wave."""
    return {"wave": hw & 0xF, "simd": (hw >> 4) & 0x3, "cu": (hw >> 8) & 0xF, "sh": (hw >> 12) & 0x1,
            "se": (hw >> 13) & 0x7}


def placement(op, grid: int, xcds: int) -> dict[str, Any]:
    out = torch.zeros(2 * grid, dtype=torch.int32, device="cuda")
    op.placement(grid, out)
    torch.cuda.synchronize()
    pairs = out.view(grid, 2).tolist()
    round_robin = sum(1 for i, (xcc, _) in enumerate(pairs) if xcc == i % xcds)
    cus: dict[int, set] = {}
    for xcc, hw in pairs:
        f = decode_hw_id(hw & 0xFFFFFFFF)
        cus.setdefault(xcc, set()).add((f["se"], f["sh"], f["cu"]))
    return {"grid": grid, "round_robin": round_robin, "xcds_seen": sorted(cus),
            "cus_per_xcd": {str(k): len(v) for k, v in sorted(cus.items())},
            "first_16": [{"wg": i, "xcc": xcc, **decode_hw_id(hw & 0xFFFFFFFF)} for i, (xcc, hw) in
                         enumerate(pairs[:16])]}


def pingpong(op, khz: int, a: int, b: int, scope: str, words: int, rounds: int) -> dict[str, Any]:
    flags = torch.zeros(128, dtype=torch.int32, device="cuda")
    payload = torch.zeros(max(2 * words, 2), dtype=torch.int32, device="cuda")
    ctl = torch.zeros(4, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int64, device="cuda")
    op.pingpong(rounds, a, b, SCOPES[scope], words, flags, payload, ctl, out)
    torch.cuda.synchronize()
    t0, t1, errors, timeouts, xa, xb, ha, hb = out.tolist()
    us = (t1 - t0) / (khz / 1e3) / (2 * rounds) if timeouts == 0 else None
    return {"a": a, "b": b, "scope": scope, "payload_bytes": 4 * words, "rounds": rounds, "us_per_hop": us,
            "payload_errors": errors, "timeouts": timeouts, "xcc_a": xa, "xcc_b": xb,
            "cu_a": decode_hw_id(ha & 0xFFFFFFFF), "cu_b": decode_hw_id(hb & 0xFFFFFFFF)}


def barrier(op, khz: int, grid: int, algo: str, scope: str, groups: int, iters: int,
            graph: bool = False) -> dict[str, Any]:
    state = torch.zeros(groups * 4096, dtype=torch.int32, device="cuda")
    ctl = torch.zeros(4, dtype=torch.int32, device="cuda")
    out = torch.zeros(4 + 2 * groups, dtype=torch.int64, device="cuda")

    def launch():
        op.barrier(iters, grid, ALGOS[algo], SCOPES[scope], groups, state, ctl, out)

    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch()
        state.zero_(), ctl.zero_(), out.zero_()
        g.replay()
    else:
        launch()
    torch.cuda.synchronize()
    vals = out.tolist()
    timeouts, mismatches = vals[0], vals[1]
    spans = [(vals[4 + 2 * g + 1] - vals[4 + 2 * g]) / (khz / 1e3) / iters for g in range(groups)]
    return {"grid": grid, "members": grid // groups, "groups": groups, "algo": algo, "scope": scope,
            "iters": iters, "graph": graph, "us_per_barrier": max(spans) if timeouts == 0 else None,
            "us_per_barrier_min_group": min(spans) if timeouts == 0 else None, "timeouts": timeouts,
            "xcc_mismatches": mismatches}


def stream(op, buf: torch.Tensor, broadcast: bool, active_xcds: int, wgs_per_xcd: int, passes: int,
           repeats: int = 5) -> dict[str, Any]:
    grid = 8 * wgs_per_xcd
    sink = torch.zeros(grid + 1, dtype=torch.int32, device="cuda")
    op.stream_read(buf, passes, int(broadcast), active_xcds, wgs_per_xcd, sink)  # warm the caches
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        op.stream_read(buf, passes, int(broadcast), active_xcds, wgs_per_xcd, sink)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    us = statistics.median(times)
    size = buf.numel() * buf.element_size()
    moved = passes * size * (active_xcds if broadcast else 1)
    return {"buffer_mib": size / MiB, "broadcast": broadcast, "active_xcds": active_xcds,
            "wgs_per_xcd": wgs_per_xcd, "passes": passes, "us": us, "gbps": moved / us / 1e3,
            "gbps_per_xcd": moved / us / 1e3 / active_xcds, "runs_us": times,
            "xcc_mismatches": int(sink[grid].item())}


def run(quick: bool = False) -> dict[str, Any]:
    from mstar.platforms import current_platform

    op = ops()
    khz = int(op.wall_clock_khz())
    facts = current_platform.device_facts()
    xcds = facts.xcds or 8
    cus = facts.compute_units or torch.cuda.get_device_properties(0).multi_processor_count
    result: dict[str, Any] = {"device": torch.cuda.get_device_name(0), "arch": facts.arch, "cus": cus,
                              "xcds": xcds, "wall_clock_khz": khz}
    result["placement"] = placement(op, 8 * cus // xcds * 4, xcds)

    rounds = 500 if quick else 2000
    hops = []
    pairs = [(0, xcds), (0, 2 * xcds), (0, 1), (0, xcds // 2)]
    for a, b in pairs:
        for scope in SCOPES:
            for words in (0, 1024, 16384):
                hops.append(pingpong(op, khz, a, b, scope, words, rounds))
    result["pingpong"] = hops

    per_xcd = cus // xcds
    bars = []
    for grid in (cus, 2 * cus):
        for algo in ALGOS:
            bars.append(barrier(op, khz, grid, algo, "agent", 1, 200 if quick else 1000))
    for grid in (xcds * 8, cus, 2 * cus):
        for algo in ALGOS:
            for scope in SCOPES:
                bars.append(barrier(op, khz, grid, algo, scope, xcds, 500 if quick else 2000))
    bars.append(barrier(op, khz, cus, "gather", "xcd", xcds, 2000, graph=True))
    bars.append(barrier(op, khz, cus, "gather", "agent", 1, 1000, graph=True))
    result["barrier"] = bars

    streams = []
    for mib, cases in (
        (2, [(True, 1, per_xcd * 4, 400), (True, xcds, per_xcd * 4, 400)]),
        (32, [(True, 1, per_xcd * 4, 40), (True, xcds, per_xcd * 4, 40), (False, xcds, per_xcd * 4, 40)]),
        (160, [(False, xcds, per_xcd * 4, 8), (False, xcds, per_xcd, 8), (True, 1, per_xcd * 4, 4),
               (True, 1, per_xcd, 4), (True, 2, per_xcd * 4, 4), (True, 4, per_xcd * 4, 4),
               (True, xcds, per_xcd * 4, 4), (True, xcds, per_xcd, 4)]),
        (2048, [(False, xcds, per_xcd * 4, 2), (False, 1, per_xcd * 4, 1)]),
    ):
        buf = torch.randint(0, 1 << 30, (mib * MiB // 4,), dtype=torch.int32, device="cuda")
        for broadcast, active, wgs, passes in cases:
            streams.append(stream(op, buf, broadcast, active, wgs, passes))
        del buf
        torch.cuda.empty_cache()
    result["stream"] = streams
    return result


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def table(result: dict[str, Any]) -> str:
    lines = [f"# sync probe: {result['device']} ({result['arch']}), {result['cus']} CUs, {result['xcds']} XCDs",
             "", f"placement: {result['placement']['round_robin']}/{result['placement']['grid']} workgroups on "
             f"XCD = id % {result['xcds']}; CUs seen per XCD {result['placement']['cus_per_xcd']}", "",
             "## one handoff (us per hop), payload errors / timeouts", "",
             "| pair | xcc | scope | payload B | us/hop | errors | timeouts |", "|---|---|---|---|---|---|---|"]
    for h in result["pingpong"]:
        lines.append(f"| {h['a']}-{h['b']} | {h['xcc_a']}-{h['xcc_b']} | {h['scope']} | {h['payload_bytes']} | "
                     f"{_fmt(h['us_per_hop'])} | {h['payload_errors']} | {h['timeouts']} |")
    lines += ["", "## barriers (us per barrier, slowest group)", "",
              "| grid | groups | members | algo | scope | graph | us | timeouts | xcc mismatches |",
              "|---|---|---|---|---|---|---|---|---|"]
    for b in result["barrier"]:
        lines.append(f"| {b['grid']} | {b['groups']} | {b['members']} | {b['algo']} | {b['scope']} | "
                     f"{b['graph']} | {_fmt(b['us_per_barrier'])} | {b['timeouts']} | {b['xcc_mismatches']} |")
    lines += ["", "## streaming reads", "",
              "| MiB | readers | XCDs | WGs/XCD | GB/s | GB/s per XCD | us |", "|---|---|---|---|---|---|---|"]
    for s in result["stream"]:
        readers = "each XCD all" if s["broadcast"] else "split"
        lines.append(f"| {s['buffer_mib']:.0f} | {readers} | {s['active_xcds']} | {s['wgs_per_xcd']} | "
                     f"{s['gbps']:.0f} | {s['gbps_per_xcd']:.0f} | {s['us']:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="fewer rounds and iterations")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    result = run(args.quick)
    (args.out / "probe.json").write_text(json.dumps(result, indent=1))
    (args.out / "probe.md").write_text(table(result))
    print(table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
