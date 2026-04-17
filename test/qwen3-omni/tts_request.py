#!/usr/bin/env python3
"""
Orpheus TTS client: sends a text prompt, streams back PCM audio, and saves as a WAV file.

Usage:
    python test/orpheus/tts_request.py
    python test/orpheus/tts_request.py --text "Good morning!" --voice zoe --output speech.wav
"""

import argparse
import base64
import io
import json
import struct
import sys

import requests

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes


def write_wav(pcm_data: bytes, path: str):
    """Write raw int16 PCM bytes to a WAV file."""
    data_size = len(pcm_data)
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH
    block_align = NUM_CHANNELS * SAMPLE_WIDTH

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # PCM format
        f.write(struct.pack("<H", NUM_CHANNELS))
        f.write(struct.pack("<I", SAMPLE_RATE))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", SAMPLE_WIDTH * 8))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_data)


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS client")
    parser.add_argument("--text", default="Hello, how are you doing today?", help="Text to synthesize")
    parser.add_argument("--voice", default="ethan", help="Voice name")
    parser.add_argument("--output", default="output.wav", help="Output WAV file path")
    parser.add_argument("--port", type=int, default=20001, help="Port number to connect to (localhost only)")
    args = parser.parse_args()

    # Always construct the URL with localhost and user-specified port
    args.url = f"http://127.0.0.1:{args.port}/generate"

    print(f"Text:  {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Sending request to {args.url} ...")

    pcm_buffer = io.BytesIO()
    chunk_count = 0

    try:
        with requests.post(
            args.url,
            data={
                "text": args.text,
                "output_modalities": "audio",
                "model_kwargs": json.dumps({"voice": args.voice}),
            },
            stream=True,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if msg.get("modality") == "text":
                    data_b64 = msg.get("data", "")
                    decoded = base64.b64decode(data_b64)

                    sys.stdout.write(decoded.decode("utf-8", errors="replace"))
                    sys.stdout.flush()

                if msg.get("modality") != "audio":
                    continue

                data_b64 = msg.get("data", "")
                if not data_b64:
                    continue

                decoded = base64.b64decode(data_b64)
                if len(decoded) == 0:
                    continue

                pcm_buffer.write(decoded)
                chunk_count += 1
                sys.stdout.write(f"\rReceived {chunk_count} audio chunks ({pcm_buffer.tell()} bytes)")
                sys.stdout.flush()
    except BaseException as e:
        print("Exception: ", e)

    pcm_data = pcm_buffer.getvalue()
    print()

    if len(pcm_data) == 0:
        print("No audio data received.")
        return

    duration = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH)
    write_wav(pcm_data, args.output)
    print(f"Saved {duration:.2f}s of audio to {args.output}")


if __name__ == "__main__":
    main()
