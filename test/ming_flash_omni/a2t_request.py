#!/usr/bin/env python3
"""Audio-to-text request against a Ming-flash-omni-2.0 server.

Exercises the audio encoder -> thinker path: the uploaded clip is decoded by the
data worker (keyed as ``audio_inputs``), turned into mel features + lengths by
``MingFlashOmniModel.process_prompt``, encoded by the ``audio_encoder`` node and
spliced into the thinker prefill.

Usage:
    python test/ming_flash_omni/a2t_request.py --audio test/qwen3-omni/audio.wav
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests
from _env import get_server_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Ming-flash-omni audio understanding")
    parser.add_argument(
        "--audio",
        default="test/qwen3-omni/audio.wav",
        help="Path to the input clip (wav/flac/mp3 — decoded server-side)",
    )
    parser.add_argument(
        "--text",
        default="Listen to this audio and describe what you hear.",
        help="Instruction accompanying the clip",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Decode budget")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        files = [("files", (audio_path.name, f, "application/octet-stream"))]
        data = {
            "text": args.text,
            "output_modalities": "text",
            "model_kwargs": json.dumps({"max_output_tokens": args.max_tokens}),
        }
        with requests.post(get_server_url(), data=data, files=files, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("modality") == "text":
                    decoded = base64.b64decode(msg.get("data", ""))
                    sys.stdout.write(decoded.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
