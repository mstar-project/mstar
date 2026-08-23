"""``mstar validate`` — run the simulator against a measured run and score it."""

from __future__ import annotations

import argparse
import json

from mstar.sim.calibration import load_timing
from mstar.sim.deployment import load_deployment
from mstar.sim.des import Simulator
from mstar.sim.metrics import summarize
from mstar.sim.stepdb import StepDB
from mstar.sim.validate import (
    gate_v1_semantics,
    gate_v2_step_costs,
    gate_v3_e2e,
    load_measured,
)
from mstar.sim.workload import WorkloadSpec, drive


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mstar validate",
        description=(
            "Simulate the same workload that was measured, then score the "
            "simulation against it: step counts (V1), step costs (V2), and "
            "end-to-end latency (V3)."
        ),
    )
    p.add_argument("--config", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--profiles", required=True,
                   help="request-profile JSONL from the measured run")
    p.add_argument("--steps", required=True,
                   help="step-log files/dir from the measured run")
    p.add_argument("--timing", default=None, help="calibration JSON")
    p.add_argument("--model", default=None)
    p.add_argument("--gpu", default=None)

    p.add_argument("--requests", type=int, required=True,
                   help="how many requests the measured run sent")
    p.add_argument("--mode", default="closed_loop",
                   choices=["online", "closed_loop", "offline"])
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--output-tokens", type=int, required=True,
                   help="max_tokens the measured run pinned")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    profiles, steps = load_measured(args.profiles, args.steps)
    if not steps:
        print(f"no step records found at {args.steps} — was the measured run "
              f"started with MSTAR_STEP_LOG set and --log-stats on?")
        return 1

    dep = load_deployment(args.config, model_key=args.model)
    timing, calibrated = load_timing(args.timing)
    db = StepDB(args.db, gpu_name=args.gpu)
    try:
        spec = WorkloadSpec(
            num_requests=args.requests, mode=args.mode, rate=args.rate,
            concurrency=args.concurrency, prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens, seed=args.seed,
        )
        sim = Simulator(dep, db, timing=timing, seed=args.seed)
        drive(sim, spec)
        report = summarize(sim, num_submitted=spec.num_requests)

        sim_steps = dict(sim.step_counts_by_key)

        gates = [
            gate_v1_semantics(sim_steps, steps),
            gate_v2_step_costs(db, dep.model_key, steps),
            gate_v3_e2e(report, profiles),
        ]

        print(f"measured: {len(profiles)} requests, {len(steps)} steps")
        print(f"simulated: {report.num_completed} requests, {report.steps} steps")
        if not calibrated:
            print("timing:   built-in placeholders — V3 will be biased")
        print()
        for g in gates:
            print(g.render())
            print()

        if args.json:
            with open(args.json, "w") as fh:
                json.dump({
                    "config": args.config,
                    "calibrated": calibrated,
                    "report": report.to_dict(),
                    "gates": [
                        {"name": g.name, "passed": g.passed, "detail": g.detail}
                        for g in gates
                    ],
                }, fh, indent=2)
            print(f"wrote {args.json}")

        return 0 if all(g.passed for g in gates) else 3
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
