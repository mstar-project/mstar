"""``mstar simulate`` — run a workload against a deployment without GPUs."""

from __future__ import annotations

import argparse
import json
import logging

from mstar.sim.calibration import load_timing
from mstar.sim.deployment import load_deployment
from mstar.sim.des import Simulator
from mstar.sim.metrics import summarize
from mstar.sim.stepdb import StepDB
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
        prog="mstar simulate",
        description=(
            "Predict end-to-end serving performance for a deployment config "
            "using measured per-step costs. No GPUs and no weights required."
        ),
    )
    p.add_argument("--config", required=True, help="deployment YAML (node_groups)")
    p.add_argument("--db", required=True, help="stepdb with measured step costs")
    p.add_argument("--model", default=None, help="model key (default: from the YAML)")
    p.add_argument("--gpu", default=None, help="price for this GPU (default: local)")
    p.add_argument(
        "--timing",
        default=None,
        help="calibration JSON from 'mstar calibrate' (default: built-in "
        "placeholders, which the report flags)",
    )

    p.add_argument("--requests", type=int, default=32)
    p.add_argument(
        "--mode", default="online", choices=["online", "closed_loop", "offline"]
    )
    p.add_argument("--rate", type=float, default=4.0, help="req/s for online mode")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="fractional spread on prompt/output lengths",
    )
    p.add_argument("--seed", type=int, default=0)

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
    p.add_argument("--json", default=None, help="write the report as JSON here")
    p.add_argument(
        "--describe", action="store_true", help="print the resolved placement and exit"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dep = load_deployment(args.config, model_key=args.model)
    if args.describe:
        print(dep.describe())
        return 0

    timing, calibrated = load_timing(args.timing)

    db = StepDB(args.db, gpu_name=args.gpu)
    try:
        if db.count() == 0:
            print(
                f"stepdb {args.db} is empty — run 'mstar harvest' on a "
                f"profiled run first"
            )
            return 1

        spec = WorkloadSpec(
            num_requests=args.requests,
            mode=args.mode,
            rate=args.rate,
            concurrency=args.concurrency,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            length_jitter=args.jitter,
            seed=args.seed,
            inputs=_input_spec(args),
        )
        sim = Simulator(dep, db, timing=timing, seed=args.seed)
        drive(sim, spec)
        report = summarize(sim, num_submitted=spec.num_requests)

        print(f"deployment: {dep.model_key} on ranks {dep.ranks} ({args.config})")
        print(f"workload:   {spec.describe()}")
        if not calibrated:
            print(
                "timing:     built-in placeholders "
                "(run 'mstar calibrate' for measured overheads)"
            )
        print(report.render())

        if args.json:
            with open(args.json, "w") as fh:
                json.dump(
                    {
                        "config": args.config,
                        "model": dep.model_key,
                        "ranks": dep.ranks,
                        "workload": vars(spec),
                        "calibrated": calibrated,
                        "report": report.to_dict(),
                    },
                    fh,
                    indent=2,
                )
            print(f"\nwrote {args.json}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
