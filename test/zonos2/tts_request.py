#!/usr/bin/env python3
"""Zonos2 TTS client: send text, stream back PCM, save a 44.1 kHz WAV.

Usage:
    python test/zonos2/tts_request.py
    python test/zonos2/tts_request.py --text "Good morning!" --output speech.wav
    python test/zonos2/tts_request.py --url http://127.0.0.1:20002/generate
    python test/zonos2/tts_request.py --ref-audio voice.wav   # clone that voice
"""
import argparse
import base64
import json
import os
import struct
import sys

import requests

SAMPLE_RATE = 44100  # DAC 44.1 kHz
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


def write_wav(pcm: bytes, path: str) -> None:
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH
    block_align = NUM_CHANNELS * SAMPLE_WIDTH
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, NUM_CHANNELS, SAMPLE_RATE,
                            byte_rate, block_align, SAMPLE_WIDTH * 8))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm)))
        f.write(pcm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:20002/generate")
    ap.add_argument("--text", default="And now for something completely different.")
    ap.add_argument("--output", default="zonos2_out.wav")
    # Generous safety cap only — the model stops at its natural EOS well before
    # this. Raise it if you feed very long text; it is not a target length.
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="generous upper bound on frames; natural EOS stops first")
    ap.add_argument("--ref-audio", default=None,
                    help="reference clip to clone the voice from; needs a "
                         "speaker-conditioned checkpoint")
    ap.add_argument("--accurate-mode", action="store_true",
                    help="favour a closer clone over expressiveness")
    ap.add_argument("--noisy-background", action="store_true",
                    help="tell the model the reference clip has background noise")
    args = ap.parse_args()

    data = {
        "text": args.text,
        "output_modalities": "audio",
        # ignore_eos=False => natural EOS (not forced-length). max_output_tokens
        # is only a generous cap.
        "model_kwargs": json.dumps({
            "max_output_tokens": args.max_tokens,
            "ignore_eos": False,
            "accurate_mode": args.accurate_mode,
            "clean_speaker_background": not args.noisy_background,
        }),
    }
    files = None
    if args.ref_audio:
        # The server infers the modality from the filename extension, so the
        # clip arrives as tensors["audio_inputs"] in process_prompt.
        data["input_modalities"] = "audio,text"
        with open(args.ref_audio, "rb") as f:
            files = {"files": (os.path.basename(args.ref_audio), f.read(),
                               "application/octet-stream")}

    pcm = b""
    chunks = 0
    with requests.post(args.url, data=data, files=files, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("modality") != "audio":
                continue
            data = msg.get("data", "")
            if data:
                pcm += base64.b64decode(data)
                chunks += 1

    if not pcm:
        print("No audio received.", file=sys.stderr)
        return 1
    write_wav(pcm, args.output)
    audio_s = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
    frames = len(pcm) // (SAMPLE_WIDTH * 512)  # 512 audio samples per frame
    capped = " (HIT CAP — raise --max-tokens)" if frames >= args.max_tokens else ""
    print(f"Received {chunks} chunks, {len(pcm)} bytes, "
          f"{audio_s:.2f}s (~{frames} frames){capped} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
