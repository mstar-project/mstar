"""Check a running MiniMax-H3 server end to end, and time it.

Requests through ``POST /generate`` (saved as ``.mp4`` + ``.wav``), checking: PSNR / SNR
against ``oracle.py``'s recording (H.264 bounds the frames); identical repeats; a
malformed request is a 400 and the server serves after it; ``/v1/videos/generations``
returns h264 + stereo AAC; with ``--rejections``, each request ``process_prompt`` must
refuse is a fast 400.

Exits nonzero on any failed check.

    PYTHONPATH=<tree> python test/minimax_h3/serve_check.py --url http://127.0.0.1:8100 \
        --out /tmp/h3_serve --height 256 --width 448 --steps 6 --seed 0 [--oracle o/oracle.pt]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np


def decode_mp4(data: bytes) -> np.ndarray:
    import av

    with av.open(io.BytesIO(data)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def write_wav(path: Path, pcm: bytes, rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def generate(client, args, seed: int, **extra):
    """One request; with ``--image`` / ``--last-image`` an FL2VA one (the keyframes as image
    inputs, ``keyframe_anchors`` saying which end each one is)."""
    images, anchors = [], []
    for path, anchor in ((args.image, "first"), (args.last_image, "last")):
        if path is not None:
            images.append(str(path))
            anchors.append(anchor)
    if images:
        extra = {"images": images, "input_modalities": ("image", "text"), "keyframe_anchors": anchors, **extra}
    t0 = time.perf_counter()
    res = client.generate(
        text=args.prompt, output_modalities=("video", "audio"), height=args.height, width=args.width,
        num_frames=args.num_frames, num_inference_steps=args.steps, seed=seed, **extra,
    )
    elapsed = time.perf_counter() - t0
    video = next(c["bytes"] for c in res.raw if c["modality"] == "video")
    audio = next(c for c in res.raw if c["modality"] == "audio")
    return video, audio, elapsed


def rejections(client, args) -> dict:
    """Each request the 400 seam must refuse, sent once: its status and seconds."""
    import requests

    keyframe = {"images": [str(args.image)], "input_modalities": ("image", "text")} if args.image else None
    cases = {
        "vision_token": dict(text=args.prompt + " <|image_pad|>"),
        "too_long": dict(text="word " * 270_000),  # past the text encoder's 262,144 positions
        "too_many_rows": dict(text=args.prompt, height=1536, width=2688),
    }
    if keyframe:
        cases.update({
            "vision_token_keyframe": dict(text="<|vision_start|>" + args.prompt, **keyframe),
            "huge_canvas_keyframe": dict(text=args.prompt, height=40000, width=40000, **keyframe),
            "half_size_keyframe": dict(text=args.prompt, height=768, **keyframe),
        })
    out = {}
    for name, kwargs in cases.items():
        kwargs = {"output_modalities": ("video", "audio"), "num_frames": args.num_frames,
                  "num_inference_steps": args.steps, **kwargs}
        t0 = time.perf_counter()
        try:
            client.generate(**kwargs)
            status = 200
        except requests.HTTPError as e:
            status = e.response.status_code
        out[name] = {"status": status, "seconds": round(time.perf_counter() - t0, 3)}
        print(f"rejection {name}: {out[name]}", flush=True)
    return out


def openai_route(args) -> dict:
    """One request through ``/v1/videos/generations``: the mp4's streams and the audio's."""
    import base64

    import av
    import requests

    body = {"model": "minimax_h3", "prompt": args.prompt, "size": f"{args.width}x{args.height}",
            "num_frames": args.num_frames, "seed": args.seed, "num_inference_steps": args.steps}
    t0 = time.perf_counter()
    res = requests.post(f"{args.url}/v1/videos/generations", json=body, timeout=3600)
    res.raise_for_status()
    mp4 = base64.b64decode(res.json()["data"][0]["b64_json"])
    elapsed = time.perf_counter() - t0
    (args.out / "openai.mp4").write_bytes(mp4)
    with av.open(io.BytesIO(mp4)) as container:
        names = [s.codec_context.name for s in container.streams]
        audio = next((s for s in container.streams if s.type == "audio"), None)
        info = {"streams": names, "seconds": elapsed,
                "audio_channels": audio.codec_context.channels if audio else 0,
                "audio_rate": audio.codec_context.sample_rate if audio else 0,
                "duration_s": float(container.duration or 0) / 1e6}
    print(f"openai route: {info}", flush=True)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--num-frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=2, help="identical requests (determinism, timing)")
    p.add_argument("--oracle", type=Path)
    p.add_argument("--min-psnr", type=float, default=30.0)
    p.add_argument("--skip-checks", action="store_true", help="timing only")
    p.add_argument("--rejections", action="store_true", help="also send each request the 400 seam must refuse")
    p.add_argument("--image", type=Path, default=None, help="FL2VA first keyframe")
    p.add_argument("--last-image", type=Path, default=None, help="FL2VA last keyframe")
    args = p.parse_args()

    import torch

    from mstar.client import MStarClient

    if args.prompt is None:
        sys.path.insert(0, str(Path(__file__).parent))
        from oracle import DEFAULT_PROMPT

        args.prompt = DEFAULT_PROMPT
    args.out.mkdir(parents=True, exist_ok=True)
    client = MStarClient(args.url, timeout=3600)
    report: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "requests": []}
    failures = []
    outputs = []
    for i in range(args.repeat):
        video, audio, elapsed = generate(client, args, args.seed)
        meta = audio.get("metadata") or {}
        rate, channels = int(meta.get("sample_rate", 32000)), int(meta.get("num_channels", 2))
        (args.out / f"r{i}.mp4").write_bytes(video)
        write_wav(args.out / f"r{i}.wav", audio["bytes"], rate, channels)
        frames = decode_mp4(video)
        pcm = np.frombuffer(audio["bytes"], dtype=np.int16).reshape(-1, channels)
        outputs.append((frames, pcm))
        report["requests"].append({"seconds": elapsed, "frames": list(frames.shape), "audio_samples": len(pcm)})
        print(f"request {i}: {elapsed:.2f} s, frames {frames.shape}, audio {pcm.shape}", flush=True)
    if not args.skip_checks and len(outputs) > 1:
        same = all(np.array_equal(outputs[0][0], f) and np.array_equal(outputs[0][1], a) for f, a in outputs[1:])
        report["repeat_identical"] = bool(same)
        if not same:
            failures.append("repeat_identical")
    if args.oracle and not args.skip_checks:
        oracle = torch.load(args.oracle, weights_only=False)
        ref = oracle["videos"]
        ref = (ref[0] if ref.ndim == 5 else ref).permute(0, 2, 3, 1).numpy()  # [F, H, W, 3] in [0, 1]
        ref_u8 = np.clip(np.round(ref * 255), 0, 255).astype(np.uint8)
        frames, pcm = outputs[0]
        report["video_psnr_db"] = psnr(frames, ref_u8) if frames.shape == ref_u8.shape else None
        ref_audio = oracle["audio"][0].numpy().T  # [samples, 2]
        ours = pcm.astype(np.float64) / 32767.0
        n = min(len(ours), len(ref_audio))
        noise = np.mean((ours[:n] - ref_audio[:n]) ** 2)
        report["audio_snr_db"] = float("inf") if noise == 0 else 10 * math.log10(np.mean(ref_audio[:n] ** 2) / noise)
        report["audio_len"] = [len(ours), len(ref_audio)]
        if report["video_psnr_db"] is None or report["video_psnr_db"] < args.min_psnr:
            failures.append("video_psnr")
    if not args.skip_checks:
        import requests

        try:
            client.generate(text=args.prompt, output_modalities=("video", "audio"), height=100, width=448,
                            num_frames=args.num_frames, num_inference_steps=args.steps)
            report["malformed"] = "accepted"
            failures.append("malformed_accepted")
        except requests.HTTPError as e:
            report["malformed"] = e.response.status_code
            if e.response.status_code != 400:
                failures.append("malformed_status")
        _, _, after = generate(client, args, args.seed + 1)
        report["after_malformed_seconds"] = after
        if args.rejections:
            report["rejections"] = rejections(client, args)
            failures += [f"rejection_{name}" for name, r in report["rejections"].items()
                         if r["status"] != 400 or r["seconds"] > 10]
        report["openai_route"] = openai_route(args)
        if report["openai_route"]["streams"] != ["h264", "aac"] or report["openai_route"]["audio_channels"] != 2:
            failures.append("openai_route")
    report["failures"] = failures
    text = json.dumps(report, indent=1, default=str)
    print(text)
    (args.out / "report.json").write_text(text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
