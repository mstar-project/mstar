"""``mstar harvest`` — turn a profiled run's step logs into a stepdb."""

from __future__ import annotations

import argparse

from mstar.sim.harvest import DEFAULT_KV_BUCKET, harvest_paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mstar harvest",
        description=(
            "Aggregate per-step traces (written by a server run with "
            "MSTAR_STEP_LOG set and --log-stats on) into the measured step "
            "cost table the simulator prices from."
        ),
    )
    p.add_argument(
        "logs", nargs="+",
        help="step-log files, globs, or directories (one log per worker)",
    )
    p.add_argument("--db", required=True, help="stepdb to create or extend")
    p.add_argument("--model", required=True, help="model key to file these rows under")
    p.add_argument(
        "--kv-bucket", type=int, default=DEFAULT_KV_BUCKET,
        help=f"KV-length bucket width in tokens (default {DEFAULT_KV_BUCKET}); "
             "wider buckets aggregate more observations per row",
    )
    p.add_argument(
        "--gpu", default=None,
        help="GPU name to file rows under (default: this host's device 0)",
    )
    args = p.parse_args(argv)

    report = harvest_paths(
        args.logs, db_path=args.db, model=args.model,
        kv_bucket=args.kv_bucket, gpu_name=args.gpu,
    )
    print(report.summary())
    if report.rows_written == 0:
        print(
            "\nNo rows written. The usual cause is a run without profiling: "
            "the step log only carries GPU times when the server ran with "
            "--log-stats (which is what enables the CUDA event pair)."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
