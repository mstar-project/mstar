#!/usr/bin/env python3
"""Text-to-speech request against a Ming-flash-omni-2.0 server (thinker + talker).

Streams both modalities: the thinker's text goes to stdout while the talker's
audio chunks are appended and written out as a WAV. The server emits headerless
int16 PCM; the sample rate comes from the model (Ming's talker AudioVAE runs at
44.1 kHz) and is echoed per chunk in the stream metadata, which this script uses
when present.

Needs the full omni deploy (configs/ming_flash_omni.yaml) — the thinker-only
config has no Talker node.

Usage:
    python test/ming_flash_omni/t2s_request.py --text "Hello there!" --output speech.wav
"""

import argparse
import base64
import json
import struct
import sys

import requests
from _env import get_server_url

DEFAULT_SAMPLE_RATE = 44100
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


def write_wav(pcm: bytes, path: str, sample_rate: int) -> None:
    """Wrap raw int16 PCM in a WAV container."""
    byte_rate = sample_rate * NUM_CHANNELS * SAMPLE_WIDTH
    block_align = NUM_CHANNELS * SAMPLE_WIDTH
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", NUM_CHANNELS))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", SAMPLE_WIDTH * 8))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm)))
        f.write(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ming-flash-omni speech request")
    parser.add_argument("--text", default="Hello, how are you doing today?")
    parser.add_argument("--output", default="ming_speech.wav")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    data = {
        "text": args.text,
        "output_modalities": "text,audio",
        "model_kwargs": json.dumps({"max_output_tokens": args.max_tokens}),
    }

    pcm = bytearray()
    sample_rate = DEFAULT_SAMPLE_RATE
    with requests.post(get_server_url(), data=data, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            modality = msg.get("modality")
            blob = base64.b64decode(msg.get("data", ""))
            if modality == "text":
                sys.stdout.write(blob.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            elif modality == "audio":
                metadata = msg.get("metadata") or {}
                sample_rate = int(metadata.get("sample_rate", sample_rate))
                pcm += blob

    sys.stdout.write("\n")
    if not pcm:
        raise SystemExit(
            "no audio returned — is the server running the full omni config "
            "(configs/ming_flash_omni.yaml, which declares the Talker node)?"
        )
    write_wav(bytes(pcm), args.output, sample_rate)
    print(
        f"wrote {args.output} "
        f"({len(pcm) // SAMPLE_WIDTH} samples @ {sample_rate} Hz)"
    )


if __name__ == "__main__":
    main()
