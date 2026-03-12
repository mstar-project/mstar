#!/usr/bin/env python3

import requests
import base64
import json
import sys
from pathlib import Path

URL = "http://0.0.0.0:8000/generate"
IMAGE_PATH = "test/bagel/bagel.png"


def main():
    image_path = Path(IMAGE_PATH)

    with open(image_path, "rb") as f:
        files = [
            ("files", (image_path.name, f, "application/octet-stream")),
        ]

        data = {
            "text": "Generate a cat eating the item in this image.",
            "output_modalities": "image",
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

                decoded = base64.b64decode(msg.get("data", ""))
                if msg.get("modality") == "text":
                    sys.stdout.write(decoded.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                

                elif msg.get("modality") == "image":
                    with open("output_edit.png", "wb") as f:
                        f.write(decoded)
                    print("\nSaved image to output_edit.png")


if __name__ == "__main__":
    main()