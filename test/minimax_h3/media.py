"""Write MiniMax-H3 frames and audio as files a person can play: an H.264 mp4 with the
audio muxed in as AAC, and the audio alone as a 16-bit WAV (AAC-in-MP4 does not play
everywhere).

    python test/minimax_h3/media.py --frames x.frames.pt --audio x.audio.pt --out clip   # clip.mp4, clip.wav
    python test/minimax_h3/media.py --oracle o/oracle.pt --out ref                      # the reference's own
"""

from __future__ import annotations

import argparse
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

FPS, SAMPLE_RATE = 24, 32000


def write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """``audio`` ``[channels, samples]`` in [-1, 1]."""
    pcm = (np.clip(audio, -1, 1) * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.T.tobytes())


def write_mp4(path: Path, frames: np.ndarray, audio: np.ndarray | None, rate: int = SAMPLE_RATE) -> None:
    """``frames`` uint8 ``[F, H, W, 3]``; ``audio`` ``[2, samples]`` in [-1, 1] or None."""
    import av

    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=Fraction(FPS))
        video.height, video.width, video.pix_fmt = frames.shape[1], frames.shape[2], "yuv420p"
        video.options = {"crf": "16", "preset": "medium"}
        sound = None
        if audio is not None:
            sound = container.add_stream("aac", rate=rate)
            sound.layout = "stereo"
        for frame in frames:
            for packet in video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if sound is not None:
            samples = np.clip(audio, -1, 1).astype(np.float32)
            chunk = 1024
            for start in range(0, samples.shape[1], chunk):
                piece = np.ascontiguousarray(samples[:, start:start + chunk])
                frame = av.AudioFrame.from_ndarray(piece, format="fltp", layout="stereo")
                frame.sample_rate = rate
                frame.pts = start
                for packet in sound.encode(frame):
                    container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, help="uint8 [F, H, W, 3] tensor (torch.save)")
    p.add_argument("--audio", type=Path, help="[2, samples] tensor in [-1, 1] (torch.save)")
    p.add_argument("--oracle", type=Path, help="an oracle.py recording: its videos and audio")
    p.add_argument("--out", type=Path, required=True, help="path stem: writes <out>.mp4 and <out>.wav")
    args = p.parse_args()
    if args.oracle:
        rec = torch.load(args.oracle, weights_only=False)
        video = rec["videos"]
        video = (video[0] if video.ndim == 5 else video).permute(0, 2, 3, 1)
        frames = (video * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        audio = rec["audio"][0].float().numpy()
    else:
        frames = torch.load(args.frames, weights_only=False).numpy()
        audio = torch.load(args.audio, weights_only=False).float().numpy() if args.audio else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(args.out.with_suffix(".mp4"), frames, audio)
    if audio is not None:
        write_wav(args.out.with_suffix(".wav"), audio)
    print(f"wrote {args.out.with_suffix('.mp4')}", flush=True)


if __name__ == "__main__":
    main()
