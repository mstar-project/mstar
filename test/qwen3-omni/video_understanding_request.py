#!/usr/bin/env python3

import base64
import json
import sys
from pathlib import Path

import requests

from _env import get_server_url


URL = get_server_url()
IMAGE_PATH = "test/qwen3-omni/video.webm"


def main():
    image_path = Path(IMAGE_PATH)

    with open(image_path, "rb") as f:
        files = [
            ("files", (image_path.name, f, "application/octet-stream")),
        ]

        data = {
            "text": "Please describe this video in detail",
            # "model_kwargs": json.dumps({
            #     "think_mode": True,
            # })
        }

        with requests.post(URL, data=data, files=files, stream=True) as resp:
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


if __name__ == "__main__":
    main()
