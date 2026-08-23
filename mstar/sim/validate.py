"""Compare a simulated run against the measured run it is supposed to predict.

Four gates, tightest first, following the discipline that a loose end-to-end
match can hide two errors that cancel:

``V1 semantics``  Did the simulator execute the same *work*? Per (node, walk)
                  step counts, simulated vs measured. A mismatch here means
                  the scheduling model is wrong, and no timing comparison
                  below it is meaningful.
``V2 step``       Does the stepdb reproduce the measured per-step GPU time it
                  was built from? This is a self-consistency check on the
                  cost table, not an independent one — it catches bucketing
                  and aggregation mistakes, not model error.
``V3 e2e``        TTFT / ITL / E2E / throughput distributions.
``V4 ranking``    Across several deployments, does the simulator order them
                  the way the measurements do? This is the gate that matters
                  for the actual product question — choosing a placement —
                  and a simulator can pass it while failing V3's absolute
                  numbers.

Distributions are compared as distributions. Per-request pairing is not
attempted: DP replica choice, arrival jitter, and batch composition make any
individual request's fate incomparable between two runs even with identical
inputs.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from mstar.profile.export import read_profiles_json
from mstar.sim.metrics import SimReport
from mstar.sim.steplog import read_step_log


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    rows: list[tuple] = field(default_factory=list)

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        out = [f"[{mark}] {self.name}", f"       {self.detail}"]
        for row in self.rows:
            out.append("       " + "  ".join(str(c) for c in row))
        return "\n".join(out)


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _rel_err(sim: float, real: float) -> float:
    if real == 0:
        return 0.0 if sim == 0 else float("inf")
    return abs(sim - real) / real


def gate_v1_semantics(
    sim_steps: dict[tuple[str, str], int],
    measured_step_log: list[dict],
    tolerance: float = 0.10,
    absolute_slack: int = 2,
) -> GateResult:
    """Simulated vs measured step counts per (node, walk).

    Judged on relative error, with ``absolute_slack`` steps of grace: on a
    node that runs a handful of times (a prefill in a short run), one step of
    difference is 12% and means nothing, while on a decode loop running
    thousands of steps the same percentage is a real divergence.
    """
    measured: dict[tuple[str, str], int] = {}
    for rec in measured_step_log:
        key = (rec["node"], rec["graph_walk"])
        measured[key] = measured.get(key, 0) + 1

    rows = [("node/walk", "sim", "real", "rel_err")]
    worst = 0.0
    worst_key = ""
    for key in sorted(set(sim_steps) | set(measured)):
        s = sim_steps.get(key, 0)
        m = measured.get(key, 0)
        err = _rel_err(s, m)
        err = err if err != float("inf") else 1.0
        rows.append((f"{key[0]}/{key[1]}", s, m, f"{err * 100:.1f}%"))
        if abs(s - m) <= absolute_slack:
            continue
        if err > worst:
            worst, worst_key = err, f"{key[0]}/{key[1]}"

    return GateResult(
        name="V1 semantics — per-(node,walk) step counts",
        passed=worst <= tolerance,
        detail=(
            f"worst relative error {worst * 100:.1f}%"
            + (f" ({worst_key})" if worst_key else "")
            + f" (tolerance {tolerance * 100:.0f}%, "
            f"ignoring differences of <={absolute_slack} steps)"
        ),
        rows=rows,
    )


def gate_v2_step_costs(
    db, model: str, measured_step_log: list[dict], tolerance: float = 0.05,
    min_observations: int = 5,
) -> GateResult:
    """Stepdb lookups vs the measured step times, per bucket.

    Only buckets with at least ``min_observations`` samples are judged. A
    bucket seen once has no distribution to compare against — its single
    sample may be the warmup replay — and scoring the table against it says
    more about sampling luck than about the table.
    """
    from mstar.sim.harvest import bucket_kv
    from mstar.sim.stepdb import StepKey

    by_key: dict[tuple, list[float]] = {}
    for rec in measured_step_log:
        if rec.get("gpu_s") is None or rec.get("mode") is None:
            continue
        if (rec.get("num_sub_batches") or 1) > 1:
            continue
        k = StepKey(
            model=model, node=rec["node"], graph_walk=rec["graph_walk"],
            padded_bs=rec["padded_bs"], padded_num_tokens=rec["padded_num_tokens"],
            tp_size=rec.get("tp_size", 1), sp_size=rec.get("sp_size", 1),
            requires_cfg=bool(rec.get("requires_cfg", False)), mode=rec["mode"],
        )
        by_key.setdefault((k, bucket_kv(rec.get("kv_len_total"))), []).append(rec["gpu_s"])

    rows = [("node/walk", "bs", "kv", "sim_ms", "real_ms", "rel_err", "n")]
    worst = 0.0
    judged = skipped = 0
    for (key, kv), obs in sorted(
        by_key.items(), key=lambda kv2: (kv2[0][0].node, kv2[0][0].padded_bs)
    )[:40]:
        cost = db.lookup(key, kv)
        real = _median(obs)
        err = _rel_err(cost.gpu_s, real)
        err = err if err != float("inf") else 1.0
        thin = len(obs) < min_observations
        if thin:
            skipped += 1
        else:
            judged += 1
            worst = max(worst, err)
        rows.append((
            f"{key.node}/{key.graph_walk}", key.padded_bs, kv,
            f"{cost.gpu_s * 1e3:.3f}", f"{real * 1e3:.3f}",
            f"{err * 100:.1f}%", f"{len(obs)}{' (thin)' if thin else ''}",
        ))

    return GateResult(
        name="V2 step costs — stepdb vs measured GPU time",
        passed=judged > 0 and worst <= tolerance,
        detail=(
            f"worst relative error {worst * 100:.1f}% over {judged} buckets "
            f"with >={min_observations} samples ({skipped} thin buckets not "
            f"judged); tolerance {tolerance * 100:.0f}%; self-consistency check"
        ),
        rows=rows,
    )


def gate_v3_e2e(
    sim: SimReport, profiles: list[dict], tolerance: float = 0.25,
) -> GateResult:
    """TTFT / E2E distributions, simulated vs measured."""
    ttft, e2e = [], []
    for prof in profiles:
        rel = prof.get("timing_rel_ms") or {}
        if "first_chunk_ms" in rel:
            ttft.append(rel["first_chunk_ms"])
        if "finish_ms" in rel:
            e2e.append(rel["finish_ms"])

    rows = [("metric", "sim_ms", "real_ms", "rel_err")]
    worst = 0.0
    for name, sim_d, real_vals in (
        ("TTFT p50", sim.ttft_ms, ttft),
        ("E2E p50", sim.e2e_ms, e2e),
    ):
        if not sim_d or not real_vals:
            rows.append((name, "-", "-", "no data"))
            continue
        real = _median(real_vals)
        s = sim_d["p50"]
        err = _rel_err(s, real)
        worst = max(worst, err if err != float("inf") else 1.0)
        rows.append((name, f"{s:.1f}", f"{real:.1f}", f"{err * 100:.1f}%"))

    return GateResult(
        name="V3 end-to-end — latency distributions",
        passed=worst <= tolerance and worst > 0,
        detail=f"worst relative error {worst * 100:.1f}% (tolerance {tolerance * 100:.0f}%)",
        rows=rows,
    )


def gate_v4_ranking(pairs: list[tuple[str, float, float]]) -> GateResult:
    """Do simulated and measured orderings agree across deployments?

    ``pairs`` is (label, simulated_metric, measured_metric). Ranking is
    scored by counting concordant/discordant pairs (Kendall's tau), because
    the product question is "which config is faster", not "by how much".
    """
    n = len(pairs)
    if n < 2:
        return GateResult(
            name="V4 ranking — placement ordering",
            passed=False,
            detail=f"need at least 2 deployments to rank, got {n}",
        )
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ds = pairs[i][1] - pairs[j][1]
            dm = pairs[i][2] - pairs[j][2]
            if ds * dm > 0:
                concordant += 1
            elif ds * dm < 0:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else 0.0
    rows = [("deployment", "sim", "real")] + [
        (label, f"{s:.2f}", f"{m:.2f}") for label, s, m in pairs
    ]
    return GateResult(
        name="V4 ranking — placement ordering",
        passed=tau == 1.0,
        detail=f"Kendall tau {tau:+.2f} ({concordant} concordant, {discordant} discordant)",
        rows=rows,
    )


def load_measured(profile_glob: str, step_glob: str) -> tuple[list[dict], list[dict]]:
    import glob
    import os

    def expand(p: str) -> list[str]:
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.jsonl")))
        return sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]

    profiles: list[dict] = []
    for f in expand(profile_glob):
        profiles.extend(read_profiles_json(f))
    steps: list[dict] = []
    for f in expand(step_glob):
        steps.extend(read_step_log(f))
    return profiles, steps
