"""Real-time action loop against ``/generate/ws`` (openpi-style client).

Streams robot observations to a Cosmos3 DROID policy served by M* and receives
action chunks back over one persistent WebSocket, measuring the loop rate.
Mirrors openpi's ``WebsocketClientPolicy`` shape: one msgpack message per
observation, one reply per action chunk, optional pipelining so the next
observation is in flight while the current chunk executes.

A request carries what NVIDIA's RoboLab server (cosmos-framework
``action_policy_server_robolab.py``) feeds the released DROID policies: the
wrist frame above the two exterior frames at half size, resized to 640x540 and
reflection-padded into the 736x544 bucket; the task followed by the view,
duration and resolution text, as a JSON caption for Edge; and the robot's
current 7 joint positions and gripper position as ``robot_state``. The chunk,
fps, size, embodiment and (Edge) guidance interval are the deployment's policy
defaults (``configs/cosmos3_droid.yaml``, ``configs/cosmos3_edge_droid.yaml``).

    python examples/cosmos3_action_ws_client.py --host localhost --port 8000 \\
        --wrist wrist.png --exterior-1 ext1.png --exterior-2 ext2.png \\
        --joints=-0.07,0.26,0.02,-2.21,0.02,2.48,0.79 --gripper 0.0 \\
        --prompt "put the marker in the pot" --iters 20
    # Edge: add --json-prompt

Each reply's ``action`` payload is float32 ``[32, action_dim_padded]``; the
first 8 columns are the 7 joint positions and the gripper position.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time

import msgpack
import numpy as np
import websockets.sync.client
from PIL import Image

# The policies' chunk, fps and size, which the prompt text states
CHUNK = 32
FPS = 15.0
COMPOSITE_HW = (540, 640)
BUCKET_HW = (544, 736)  # the 480p bucket closest to 540x640
VIEW_TEXT = (
    "This video contains concatenated views from multiple camera perspectives. "
    "The top row is from the wrist-mounted camera. The bottom row contains two horizontally "
    "concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
)


def resize_bilinear(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """uint8 HWC resize like torch's ``F.interpolate(mode="bilinear", align_corners=False)``,
    truncated back to uint8 as the reference server does."""
    def taps(n_in, n_out):
        src = np.maximum((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0.0)
        i0 = np.minimum(src.astype(np.int64), n_in - 1)
        return i0, np.minimum(i0 + 1, n_in - 1), (src - i0).astype(np.float32)

    y0, y1, fy = taps(img.shape[0], h)
    x0, x1, fx = taps(img.shape[1], w)
    a = img.astype(np.float32)
    fx = fx[None, :, None]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bottom = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    fy = fy[:, None, None]
    return (top * (1 - fy) + bottom * fy).astype(np.uint8)


def compose_observation(wrist: np.ndarray, ext1: np.ndarray, ext2: np.ndarray) -> np.ndarray:
    """The policy's observation image: wrist on top, both exteriors at half size below,
    resized to 640x540 and reflection-padded on the right and bottom to 736x544."""
    half = (wrist.shape[0] // 2, wrist.shape[1] // 2)
    bottom = np.concatenate([resize_bilinear(ext1, *half), resize_bilinear(ext2, *half)], axis=1)
    img = resize_bilinear(np.concatenate([wrist, bottom], axis=0), *COMPOSITE_HW)
    pad = ((0, BUCKET_HW[0] - img.shape[0]), (0, BUCKET_HW[1] - img.shape[1]), (0, 0))
    return np.pad(img, pad, mode="reflect")


def policy_prompt(task: str, json_prompt: bool) -> str:
    """The task with the view, duration and resolution metadata the policies were trained on."""
    task = task.strip()
    if not task:
        return ""
    frames = CHUNK + 1
    if json_prompt:
        return json.dumps({
            "cinematography": {"framing": VIEW_TEXT},
            "actions": [{"time": f"0:00-0:{round(frames / FPS):02d}",
                         "description": task if task.endswith((".", "!", "?")) else task + "."}],
            "duration": f"{int(frames / FPS)}s",
            "fps": FPS,
            "resolution": {"H": BUCKET_HW[0], "W": BUCKET_HW[1]},
            "aspect_ratio": "4,3",
        })
    text = task + (" " if task.endswith(".") else ". ") + VIEW_TEXT
    return (f"{text} The video is {float(int(frames / FPS)):.1f} seconds long and is of {FPS:.0f} FPS. "
            f"This video is of {BUCKET_HW[0]}x{BUCKET_HW[1]} resolution.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--wrist", required=True, help="wrist camera frame (jpg/png)")
    ap.add_argument("--exterior-1", required=True, help="exterior camera 1 frame")
    ap.add_argument("--exterior-2", required=True, help="exterior camera 2 frame")
    ap.add_argument("--joints", required=True, help="7 comma-separated joint positions (rad)")
    ap.add_argument("--gripper", type=float, required=True, help="gripper position in [0, 1]")
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--json-prompt", action="store_true", help="JSON caption, as the Edge policy is served")
    ap.add_argument("--steps", type=int, default=None, help="denoise steps (default: the server's policy default)")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--guidance-interval", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="guide only at timesteps in [LO, HI] (default: the deployment's)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pipeline", type=int, default=1, help="observations in flight (1 = strict request/response)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="leading chunks excluded from the rates (the first request of a shape pays JIT/capture)")
    args = ap.parse_args()

    joints = [float(v) for v in args.joints.split(",")]
    if len(joints) != 7:
        raise SystemExit("--joints takes 7 values")
    views = [np.asarray(Image.open(p).convert("RGB")) for p in (args.wrist, args.exterior_1, args.exterior_2)]
    buf = io.BytesIO()
    Image.fromarray(compose_observation(*views)).save(buf, format="PNG")
    obs_bytes = buf.getvalue()
    model_kwargs = {"action_mode": "policy", "robot_state": joints + [args.gripper]}
    if args.steps is not None:
        model_kwargs["num_inference_steps"] = args.steps
    if args.guidance is not None:
        model_kwargs["guidance_scale"] = args.guidance
    if args.guidance_interval is not None:
        model_kwargs["guidance_interval"] = args.guidance_interval
    prompt = policy_prompt(args.prompt, args.json_prompt)

    def observation(i: int) -> bytes:
        return msgpack.packb({
            "text": prompt,
            "files": [{"name": f"obs_{i}.png", "data": obs_bytes}],
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
                actions = np.frombuffer(reply["data"], dtype=np.float32).reshape(CHUNK, -1)[:, :8]
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
    budget = CHUNK / FPS
    warm_ms = ", ".join(f"{x * 1000:.0f} ms" for x in latencies[:warm])
    print(f"\n{args.iters} chunks in {wall:.2f}s (first {warm} excluded as warmup: {warm_ms}); "
          f"steady state {n} chunks in {steady_wall:.2f}s: "
          f"{n / steady_wall:.2f} chunks/s, {n * CHUNK / steady_wall:.1f} actions/s; "
          f"latency median {med * 1000:.0f} ms, p95 {sorted(steady)[int(0.95 * (n - 1))] * 1000:.0f} ms; "
          f"budget for {CHUNK} actions at {FPS:g} Hz = {budget:.2f}s -> RTF {budget / med:.2f}")


if __name__ == "__main__":
    main()
