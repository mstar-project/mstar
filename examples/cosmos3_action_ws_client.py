"""Real-time action loop against ``/generate/ws`` (openpi-style client).

Streams robot observations to a Cosmos3 policy served by M* and receives
action chunks back over one persistent WebSocket, measuring the loop rate.
Mirrors openpi's ``WebsocketClientPolicy`` shape: one msgpack message per
observation, one reply per action chunk, optional pipelining so the next
observation is in flight while the current chunk executes.

    python examples/cosmos3_action_ws_client.py --host localhost --port 8000 \\
        --image path/to/observation.jpg --prompt "pick up the mug" \\
        --domain droid_lerobot --action-dim 10 --chunk 32 --iters 20

Each reply's ``action`` payload is float32 ``[chunk, action_dim_padded]``;
the first ``--action-dim`` columns are the embodiment's actions.
"""

from __future__ import annotations

import argparse
import statistics
import time

import msgpack
import numpy as np
import websockets.sync.client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--image", required=True, help="observation frame (jpg/png)")
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--domain", default="droid_lerobot")
    ap.add_argument("--action-dim", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None, help="denoise steps (default: the server's policy default)")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--size", default=None, help="observation size WxH (default: the server's 480p tier)")
    ap.add_argument("--fps", type=float, default=15.0, help="control rate the chunk is consumed at (for the budget)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pipeline", type=int, default=1, help="observations in flight (1 = strict request/response)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="leading chunks excluded from the rates (the first request of a shape pays JIT/capture)")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        obs_bytes = f.read()
    model_kwargs = {
        "action_mode": "policy", "domain_name": args.domain, "raw_action_dim": args.action_dim,
        "action_chunk_size": args.chunk, "num_frames": args.chunk + 1,
    }
    if args.steps is not None:
        model_kwargs["num_inference_steps"] = args.steps
    if args.guidance is not None:
        model_kwargs["guidance_scale"] = args.guidance
    if args.size:
        model_kwargs["size"] = args.size

    def observation(i: int) -> bytes:
        return msgpack.packb({
            "text": args.prompt,
            "files": [{"name": f"obs_{i}.{args.image.rsplit('.', 1)[-1]}", "data": obs_bytes}],
            "input_modalities": ["image", "text"],
            "output_modalities": ["action"],
            "model_kwargs": model_kwargs,
            "request_id": f"obs-{i}",
        }, use_bin_type=True)

    uri = f"ws://{args.host}:{args.port}/generate/ws"
    latencies: list[float] = []
    sent: dict[str, float] = {}
    finished_at: list[float] = []
    with websockets.sync.client.connect(uri, max_size=None) as ws:
        t_start = time.perf_counter()
        next_i = 0
        done = 0
        # Prime the pipeline.
        while next_i < min(args.pipeline, args.iters):
            sent[f"obs-{next_i}"] = time.perf_counter()
            ws.send(observation(next_i))
            next_i += 1
        while done < args.iters:
            reply = msgpack.unpackb(ws.recv(), raw=False)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            if reply.get("modality") == "action":
                actions = np.frombuffer(reply["data"], dtype=np.float32).reshape(args.chunk, -1)[:, :args.action_dim]
                latencies.append(time.perf_counter() - sent[reply["request_id"]])
                print(f"{reply['request_id']}: actions {actions.shape} first={actions[0, :3]} "
                      f"latency {latencies[-1] * 1000:.0f} ms")
            if reply.get("finish"):
                done += 1
                finished_at.append(time.perf_counter())
                if next_i < args.iters:
                    sent[f"obs-{next_i}"] = time.perf_counter()
                    ws.send(observation(next_i))
                    next_i += 1
        wall = time.perf_counter() - t_start
    warm = min(args.warmup, len(latencies) - 1) if len(latencies) > 1 else 0
    steady = latencies[warm:]
    steady_wall = finished_at[-1] - (finished_at[warm - 1] if warm else t_start)
    n = len(steady)
    med = statistics.median(steady)
    budget = args.chunk / args.fps
    warm_ms = ", ".join(f"{x * 1000:.0f} ms" for x in latencies[:warm])
    print(f"\n{args.iters} chunks in {wall:.2f}s (first {warm} excluded as warmup: {warm_ms}); "
          f"steady state {n} chunks in {steady_wall:.2f}s: "
          f"{n / steady_wall:.2f} chunks/s, {n * args.chunk / steady_wall:.1f} actions/s; "
          f"latency median {med * 1000:.0f} ms, p95 {sorted(steady)[int(0.95 * (n - 1))] * 1000:.0f} ms; "
          f"budget for {args.chunk} actions at {args.fps:g} Hz = {budget:.2f}s -> RTF {budget / med:.2f}")


if __name__ == "__main__":
    main()
