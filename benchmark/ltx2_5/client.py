#!/usr/bin/env python3
"""One LTX-2.5 text-to-audio+video request against a running server: saves the video,
the audio (WAV) and the two muxed into one mp4 for a person to watch.

    python benchmark/ltx2_5/client.py --port 8123 --out /tmp/ltx --prompt "..." --seed 0
"""
import argparse
import time
import wave
from pathlib import Path

from mstar.client.client import MStarClient

DEFAULT_PROMPT = (
    "A golden retriever runs along a beach at sunset, waves crashing behind it. "
    "The dog barks twice, and seagulls cry overhead. Warm cinematic lighting, "
    "handheld camera following the dog."
)


def mux(video_path: Path, wav_path: Path, out_path: Path) -> None:
    """Copy the server's H.264 stream and add the audio as AAC, with an ffmpeg binary
    (``FFMPEG`` or imageio-ffmpeg's)."""
    import os
    import subprocess

    ffmpeg = os.environ.get("FFMPEG")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(wav_path),
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--num-frames", type=int, default=121)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = MStarClient(f"http://127.0.0.1:{args.port}", timeout=1800)
    t0 = time.perf_counter()
    res = client.generate(
        text=args.prompt, output_modalities=("video", "audio"), seed=args.seed,
        height=args.height, width=args.width, num_frames=args.num_frames,
    )
    elapsed = time.perf_counter() - t0
    videos = [c["bytes"] for c in res.raw if c.get("modality") == "video"]
    (out / "video.mp4").write_bytes(videos[0])
    (audio,) = [c for c in res.raw if c.get("modality") == "audio"]
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(int(audio["metadata"].get("num_channels", 2)))
        w.setsampwidth(2)
        w.setframerate(int(audio["metadata"].get("sample_rate", 48000)))
        w.writeframes(audio["bytes"])
    with wave.open(str(out / "audio.wav")) as w:
        audio_s, rate, channels = w.getnframes() / w.getframerate(), w.getframerate(), w.getnchannels()
    mux(out / "video.mp4", out / "audio.wav", out / "av.mp4")
    print(f"latency {elapsed:.2f}s  video {len(videos[0])} bytes  audio {audio_s:.2f}s @ {rate} Hz x{channels}"
          f"  -> {out / 'av.mp4'}")


if __name__ == "__main__":
    main()
