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


def _input_spec(args):
    """Turn the CLI's request description into the spec models read."""
    import json as _json

    from mstar.sim.request_inputs import InputSpec

    h, _, w = args.image_size.partition("x")
    return InputSpec(
        input_modalities=[
            m.strip() for m in args.input_modalities.split(",") if m.strip()
        ],
        output_modalities=[
            m.strip() for m in args.output_modalities.split(",") if m.strip()
        ],
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        image_size=(int(h), int(w or h)),
        num_images=args.num_images,
        video_frames=args.video_frames,
        audio_samples=int(args.audio_seconds * 16000),
        model_kwargs=_json.loads(args.model_kwargs) if args.model_kwargs else {},
    )


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
    p.add_argument(
        "--profiles", required=True, help="request-profile JSONL from the measured run"
    )
    p.add_argument(
        "--steps", required=True, help="step-log files/dir from the measured run"
    )
    p.add_argument("--timing", default=None, help="calibration JSON")
    p.add_argument("--model", default=None)
    p.add_argument("--gpu", default=None)

    p.add_argument(
        "--requests",
        type=int,
        required=True,
        help="how many requests the measured run sent",
    )
    p.add_argument(
        "--mode", default="closed_loop", choices=["online", "closed_loop", "offline"]
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument(
        "--output-tokens",
        type=int,
        required=True,
        help="max_tokens the measured run pinned",
    )
    p.add_argument(
        "--input-modalities",
        default="text",
        help="comma-separated request inputs (text,image,audio,video,state). "
        "Models branch on these: asking a TTS model for text output "
        "simulates a different request than asking it for audio.",
    )
    p.add_argument(
        "--output-modalities",
        default="text",
        help="comma-separated request outputs (text,image,audio,video,action)",
    )
    p.add_argument(
        "--image-size",
        default="1024x1024",
        help="HxW for image inputs and generated images",
    )
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--video-frames", type=int, default=16)
    p.add_argument("--audio-seconds", type=float, default=5.0)
    p.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON dict passed to the model's transition functions",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    profiles, steps = load_measured(args.profiles, args.steps)
    if not steps:
        print(
            f"no step records found at {args.steps} — was the measured run "
            f"started with MSTAR_STEP_LOG set and --log-stats on?"
        )
        return 1

    dep = load_deployment(args.config, model_key=args.model)
    timing, calibrated = load_timing(args.timing)
    db = StepDB(args.db, gpu_name=args.gpu)
    try:
        spec = WorkloadSpec(
            num_requests=args.requests,
            mode=args.mode,
            rate=args.rate,
            concurrency=args.concurrency,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            seed=args.seed,
            inputs=_input_spec(args),
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
                json.dump(
                    {
                        "config": args.config,
                        "calibrated": calibrated,
                        "report": report.to_dict(),
                        "gates": [
                            {"name": g.name, "passed": g.passed, "detail": g.detail}
                            for g in gates
                        ],
                    },
                    fh,
                    indent=2,
                )
            print(f"wrote {args.json}")

        return 0 if all(g.passed for g in gates) else 3
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
