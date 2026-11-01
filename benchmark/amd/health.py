"""Node health around a benchmark table, on top of $WORK/amd/jobs/node_health.sh.

``check`` gates the table: a FAIL aborts it. When the run owns only some of the node's GPUs, an
occupied GPU outside that set is reported, not failed. Around each arm: ``snapshot`` before and
after, ``monitor`` at 1 Hz during the timed window, then ``delta`` and ``summarize`` go into the
arm's health.md. Thermal throttling on the arm's GPUs fails the arm; power throttling is reported.

    python -m benchmark.amd.health check --out DIR --expect-gpus 8 --own-gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

NODE_HEALTH = Path(os.environ.get("NODE_HEALTH", "/work1/stephaniewang/atindra/amd/jobs/node_health.sh"))
ROW_RE = re.compile(r"^(NODE|GPU\d+)\s+(PASS|WARN|FAIL|INFO)\s+(\S+)\s*(.*)$")
# Checks about other people's work on a GPU, not about its health.
OCCUPANCY_CHECKS = {"occupancy", "hotspot_temp", "hbm_temp", "throttle_status", "violation_status"}
# An occupancy row whose GPU holds no more than this only lists processes that opened every GPU (ROCr does that
# unless ROCR_VISIBLE_DEVICES limits it); it is not a tenant. Momentary activity at idle memory is not either.
IDLE_VRAM_MB = 1024
OCCUPANCY_RE = re.compile(r"vram_used=(\d+)/\d+ MB")


def _gpu_of(scope: str) -> int | None:
    return int(scope[3:]) if scope.startswith("GPU") else None


def smi_gpus(gpus: list[int] | None) -> list[int] | None:
    """amd-smi's numbers for HIP devices; node_health.sh rows and monitor samples use amd-smi's."""
    if gpus is None:
        return None
    from benchmark.amd.launch import smi_gpu
    return [smi_gpu(g) for g in gpus]


def _idle_occupancy(r: dict) -> bool:
    m = OCCUPANCY_RE.search(r["detail"])
    return r["check"] == "occupancy" and bool(m) and int(m.group(1)) <= IDLE_VRAM_MB


@dataclass
class Verdict:
    status: str = "PASS"
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, why: str) -> None:
        self.status = "FAIL"
        self.reasons.append(why)

    def warn(self, why: str) -> None:
        if self.status == "PASS":
            self.status = "WARN"
        self.reasons.append(why)

    def as_dict(self) -> dict:
        return {"status": self.status, "reasons": self.reasons, "notes": self.notes}


def interpret_check(rows: list[dict], own_gpus: list[int] | None) -> Verdict:
    """Verdict of node_health.sh check rows for a run that owns ``own_gpus`` (amd-smi numbers; None: the whole
    node)."""
    v = Verdict()
    for r in rows:
        gpu = _gpu_of(r["scope"])
        mine = gpu is None or own_gpus is None or gpu in own_gpus
        what = f"{r['scope']} {r['check']}: {r['detail']}"
        if r["status"] == "FAIL" and _idle_occupancy(r):
            v.notes.append("no memory or work on it, " + what)
        elif r["status"] == "FAIL":
            if mine:
                v.fail(what)
            elif r["check"] in OCCUPANCY_CHECKS:
                v.notes.append("not ours, " + what)
            else:
                v.warn("not ours, " + what)
        elif r["status"] == "WARN":
            if mine or r["check"] not in OCCUPANCY_CHECKS:
                v.warn(what)
            else:
                v.notes.append("not ours, " + what)
    return v


def parse_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            rows.append({"scope": m.group(1), "status": m.group(2), "check": m.group(3), "detail": m.group(4)})
    return rows


def _run(args: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run([str(NODE_HEALTH), *args], capture_output=True, text=True, timeout=timeout, check=False)


def check(out_dir: Path, expect_gpus: int | None, own_gpus: list[int] | None, name: str = "check") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["check", "--json", str(out_dir / f"{name}.raw.json")]
    if expect_gpus:
        args += ["--expect-gpus", str(expect_gpus)]
    p = _run(args)
    (out_dir / f"{name}.txt").write_text(p.stdout + p.stderr)
    try:
        rows = json.loads((out_dir / f"{name}.raw.json").read_text())["rows"]
    except (OSError, ValueError, KeyError):
        rows = parse_rows(p.stdout)
    smi = smi_gpus(own_gpus)
    v = interpret_check(rows, smi)
    if not rows:
        v.fail(f"node_health.sh check gave no rows (rc {p.returncode})")
    res = {"verdict": v.as_dict(), "own_gpus": own_gpus, "own_gpus_amd_smi": smi, "expect_gpus": expect_gpus,
           "rc": p.returncode, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


def snapshot(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = _run(["snapshot", str(path)])
    return (p.stdout + p.stderr).strip()[-500:]


class Monitor:
    """node_health.sh monitor in its own session, sampling sysfs once a second (cheapest backend)."""

    def __init__(self, csv: Path, interval_s: float = 1.0):
        self.csv = csv
        self.interval_s = interval_s
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "NODE_HEALTH_MONITOR_BACKEND": "sysfs"}
        self.proc = subprocess.Popen([str(NODE_HEALTH), "monitor", str(self.csv), str(self.interval_s)], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(5)
        self.proc = None


def arm_verdict(delta_rows: list[dict], summary_text: str, own_gpus: list[int]) -> Verdict:
    """Thermal throttling or new uncorrectable ECC errors on our GPUs fail the arm; power is reported."""
    v = Verdict()
    for r in delta_rows:
        gpu = _gpu_of(r["scope"])
        if gpu is not None and gpu not in own_gpus:
            continue
        what = f"{r['scope']} {r['check']}: {r['detail']}"
        hard = ("thermal_residency", "ecc_delta", "perf_level_changed", "same_host")
        if r["check"] in hard and r["status"] == "FAIL":
            v.fail(what)
        elif r["check"] == "thermal_residency" and r["status"] == "WARN":
            v.warn(what)
        elif r["check"] in ("power_residency", "energy"):
            v.notes.append(what)
    for line in summary_text.splitlines():
        parts = line.split("|")
        cols = line.split()
        if len(parts) < 4 or not cols or not cols[0].isdigit() or int(cols[0]) not in own_gpus:
            continue
        verdict = parts[-1].strip()
        if "HOT" in verdict or "TVIOL" in verdict:
            v.fail(f"GPU{cols[0]} monitor: {verdict}")
        elif verdict != "ok":
            v.notes.append(f"GPU{cols[0]} monitor: {verdict}")
    return v


@dataclass
class ArmHealth:
    """Snapshot before, monitor during, snapshot + delta + summary after one arm. ``own_gpus`` are amd-smi
    numbers (smi_gpus)."""

    out_dir: Path
    own_gpus: list[int]
    monitor: Monitor | None = None

    def begin(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        snapshot(self.out_dir / "snapshot_before.txt")
        self.monitor = Monitor(self.out_dir / "monitor.csv")
        self.monitor.start()

    def end(self) -> dict:
        if self.monitor:
            self.monitor.stop()
        snapshot(self.out_dir / "snapshot_after.txt")
        d = _run(["delta", str(self.out_dir / "snapshot_before.txt.json"),
                  str(self.out_dir / "snapshot_after.txt.json")])
        (self.out_dir / "delta.txt").write_text(d.stdout + d.stderr)
        s = _run(["summarize", str(self.out_dir / "monitor.csv")]) if (self.out_dir / "monitor.csv").is_file() else None
        summary = (s.stdout + s.stderr) if s else "no monitor samples"
        (self.out_dir / "summary.txt").write_text(summary)
        v = arm_verdict(parse_rows(d.stdout), summary, self.own_gpus)
        res = {"verdict": v.as_dict(), "own_gpus": self.own_gpus}
        (self.out_dir / "health.json").write_text(json.dumps(res, indent=1))
        write_health_md(self.out_dir.parent / "health.md", res, d.stdout, summary)
        return res


def write_health_md(path: Path, res: dict, delta_text: str, summary_text: str) -> None:
    v = res["verdict"]
    lines = [f"# Health: {v['status']}", "", f"GPUs (amd-smi numbers): {res['own_gpus']}", ""]
    if v["reasons"]:
        lines += ["Reasons:", *[f"- {r}" for r in v["reasons"]], ""]
    if v["notes"]:
        lines += ["Notes:", *[f"- {r}" for r in v["notes"]], ""]
    lines += ["## delta (snapshots before and after)", "```", delta_text.strip(), "```", "",
              "## monitor summary (1 Hz, busy samples)", "```", summary_text.strip(), "```", ""]
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--expect-gpus", type=int, default=None)
    c.add_argument("--own-gpus", default=None, help="comma list; default: the whole node")
    args = ap.parse_args(argv)
    own = [int(x) for x in args.own_gpus.split(",")] if args.own_gpus else None
    res = check(Path(args.out), args.expect_gpus, own)
    print(json.dumps(res["verdict"], indent=1))
    return 1 if res["verdict"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
