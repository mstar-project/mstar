"""Streaming rollout client: windowed Cosmos3 video over ``/generate/ws``.

Sends one windowed video request (``window_mode`` kv or chained, ``stream_video``
on) and writes every window's mp4 to disk the moment it arrives, printing the
time to the first frame chunk and the cadence of the following windows. The
same request over ``POST /generate`` yields the chunks as NDJSON lines; the
WebSocket keeps the connection for follow-up requests (e.g. the next prompt of
a session) without a new handshake.

    python examples/cosmos3_stream_video_ws_client.py --port 8000 \\
        --prompt "a drone flies over a coastal town at dawn" \\
        --frames 241 --mode kv --window-frames 29 --out /tmp/rollout
"""

from __future__ import annotations

import argparse
import json
import time

import msgpack
import websockets.sync.client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default=None, help="optional i2v conditioning frame")
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--frames", type=int, default=241)
    ap.add_argument("--mode", choices=["kv", "chained"], default="kv")
    ap.add_argument("--window-frames", type=int, default=29)
    ap.add_argument("--context-frames", type=int, default=None,
                    help="kv: committed context to keep (server default 61)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="/tmp/cosmos3_rollout", help="prefix for <out>_<window>.mp4")
    args = ap.parse_args()

    model_kwargs = {
        "size": args.size, "num_frames": args.frames, "window_mode": args.mode,
        "window_frames": args.window_frames, "stream_video": True,
    }
    for key, value in (("context_frames", args.context_frames), ("num_inference_steps", args.steps),
                       ("guidance_scale", args.guidance), ("seed", args.seed)):
        if value is not None:
            model_kwargs[key] = value
    message = {
        "text": args.prompt, "output_modalities": ["video"], "model_kwargs": model_kwargs,
        "request_id": "rollout-0",
    }
    if args.image:
        with open(args.image, "rb") as f:
            message["files"] = [{"name": args.image.rsplit("/", 1)[-1], "data": f.read()}]
        message["input_modalities"] = ["image", "text"]

    uri = f"ws://{args.host}:{args.port}/generate/ws"
    with websockets.sync.client.connect(uri, max_size=None) as ws:
        t0 = time.perf_counter()
        ws.send(msgpack.packb(message, use_bin_type=True))
        arrivals: list[float] = []
        while True:
            reply = msgpack.unpackb(ws.recv(), raw=False)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            if reply.get("modality") == "video":
                arrivals.append(time.perf_counter() - t0)
                path = f"{args.out}_{len(arrivals) - 1}.mp4"
                with open(path, "wb") as f:
                    f.write(reply["data"])
                gap = "" if len(arrivals) == 1 else f" (+{arrivals[-1] - arrivals[-2]:.2f}s)"
                print(f"window {len(arrivals) - 1}: {len(reply['data']) // 1024} KB "
                      f"at {arrivals[-1]:.2f}s{gap} -> {path}")
            elif reply.get("modality") == "error":
                raise RuntimeError(reply["data"])
            if reply.get("finish"):
                break
    if arrivals:
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:], strict=False)]
        cadence = f", median window gap {sorted(gaps)[len(gaps) // 2]:.2f}s" if gaps else ""
        print(json.dumps({
            "windows": len(arrivals), "ttff_s": round(arrivals[0], 3),
            "total_s": round(arrivals[-1], 3), "frames": args.frames,
            "frames_per_s": round(args.frames / arrivals[-1], 2),
        }) + cadence)


if __name__ == "__main__":
    main()
