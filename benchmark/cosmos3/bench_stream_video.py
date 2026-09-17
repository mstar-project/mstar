"""Streaming rollout latency for Cosmos3 windowed video (M* only — no baseline
engine streams frames; their whole-clip time is video_bench.py's number).

Drives the native ``/generate`` route with a windowed request and
``stream_video`` on, timing every window chunk as it arrives:

  TTFF       time to the first decoded frame chunk (window 0 denoised + decoded)
  chunk gap  median time between consecutive window chunks (steady-state
             generation cadence; the decoder partition overlaps the loop)
  frames/s   frames delivered / wall time, whole request
  total      wall time to the last chunk (compare with the non-windowed clip)

``--mode none`` runs the same frame count as one plain (non-windowed) request
through the same route, so TTFF == total there — the reference the streaming
modes are judged against. Chunk frame counts come from the mp4s (PyAV) when it
is installed, else from the window schedule.

  python bench_stream_video.py --port 8100 --mode kv --frames 241 --size 832x480 --steps 20
  python bench_stream_video.py --port 8100 --mode chained --frames 241
  python bench_stream_video.py --port 8100 --mode none --frames 241
"""
import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--mode", choices=["kv", "chained", "none"], default="kv")
ap.add_argument("--size", default="832x480")
ap.add_argument("--frames", type=int, default=241)
ap.add_argument("--window-frames", type=int, default=29)
ap.add_argument("--overlap-frames", type=int, default=None, help="chained only (default: server's)")
ap.add_argument("--context-frames", type=int, default=None, help="kv only (default: server's)")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--gs", type=float, default=6.0)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--image", default="", help="i2v conditioning frame (jpg/png); t2v when empty")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--save", default="", help="write the received chunks as <save>_<i>.mp4")
args = ap.parse_args()

PROMPT = "A robot arm is cleaning a plate in the kitchen, smooth natural motion."


def _count_frames(mp4: bytes) -> int | None:
    try:
        import av  # noqa: PLC0415
    except ImportError:
        return None
    with av.open(io.BytesIO(mp4)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def _multipart(fields: dict[str, str], files: list[tuple[str, str, bytes]]):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, filename, data in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def run_once():
    mk = {
        "size": args.size, "num_frames": args.frames, "num_inference_steps": args.steps,
        "guidance_scale": args.gs, "fps": args.fps, "seed": args.seed,
    }
    if args.mode != "none":
        mk.update({"window_mode": args.mode, "window_frames": args.window_frames, "stream_video": True})
        if args.overlap_frames is not None:
            mk["overlap_frames"] = args.overlap_frames
        if args.context_frames is not None:
            mk["context_frames"] = args.context_frames
    fields = {"text": PROMPT, "output_modalities": "video", "streaming": "true", "model_kwargs": json.dumps(mk)}
    files = []
    if args.image:
        with open(args.image, "rb") as f:
            files.append(("files", args.image.rsplit("/", 1)[-1], f.read()))
        fields["input_modalities"] = "image,text"
    body, ctype = _multipart(fields, files)
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/generate", data=body, headers={"Content-Type": ctype},
    )
    t0 = time.perf_counter()
    arrivals, sizes, frames = [], [], []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("modality") == "error":
                raise RuntimeError(base64.b64decode(msg["data"]).decode(errors="replace"))
            if msg.get("modality") != "video":
                continue
            data = base64.b64decode(msg["data"])
            arrivals.append(time.perf_counter() - t0)
            sizes.append(len(data))
            frames.append(_count_frames(data))
            if args.save:
                with open(f"{args.save}_{len(sizes) - 1}.mp4", "wb") as f:
                    f.write(data)
    if not arrivals:
        raise RuntimeError("no video chunk received")
    if any(n is None for n in frames):
        frames = [args.frames] if len(frames) == 1 else None
    return arrivals, sizes, frames


print(f"=== stream mode={args.mode} size={args.size} frames={args.frames} window={args.window_frames} "
      f"steps={args.steps} gs={args.gs} seed={args.seed} {'i2v' if args.image else 't2v'} ===", flush=True)
for _ in range(args.warmup):
    run_once()
ttff, totals, gaps, fps_out = [], [], [], []
for _ in range(args.rounds):
    arrivals, sizes, frames = run_once()
    ttff.append(arrivals[0])
    totals.append(arrivals[-1])
    if len(arrivals) > 1:
        gaps.append(statistics.median(b - a for a, b in zip(arrivals, arrivals[1:], strict=False)))
    delivered = sum(frames) if frames else args.frames
    fps_out.append(delivered / arrivals[-1])
    print(f"  chunks={len(arrivals)} frames={delivered} TTFF={arrivals[0]:.2f}s total={arrivals[-1]:.2f}s "
          f"mp4={sum(sizes) // 1024}KB", flush=True)
med = statistics.median
print(f"  TTFF median {med(ttff):.2f}s | chunk gap median {med(gaps):.2f}s | "
      f"frames/s {med(fps_out):.2f} | total median {med(totals):.2f}s  (n={args.rounds})"
      if gaps else
      f"  TTFF=total median {med(ttff):.2f}s | frames/s {med(fps_out):.2f}  (n={args.rounds})", flush=True)
print("DONE", flush=True)
