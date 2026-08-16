#!/usr/bin/env python3
"""Text-to-text request against a Ming-flash-omni-2.0 server (thinker only).

Streams the decoded text back as it is generated. Works against both the full
omni deploy (configs/ming_flash_omni.yaml) and the thinker-only one.

Usage:
    python test/ming_flash_omni/t2t_request.py
    python test/ming_flash_omni/t2t_request.py --text "Who are you?" --max-tokens 128
"""

import argparse
import base64
import json
import sys

import requests
from _env import get_server_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Ming-flash-omni text request")
    parser.add_argument(
        "--text",
        default="What is the 7th value after the decimal point in pi?",
        help="Prompt text",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Decode budget")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 = greedy (deterministic)",
    )
    args = parser.parse_args()

    data = {
        "text": args.text,
        "output_modalities": "text",
        "model_kwargs": json.dumps({
            "max_output_tokens": args.max_tokens,
            "temperature": args.temperature,
        }),
    }

    with requests.post(get_server_url(), data=data, stream=True) as resp:
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
