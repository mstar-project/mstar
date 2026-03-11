#!/usr/bin/env python3

import requests
import base64
import json
import sys


URL = "http://0.0.0.0:8000/generate"


def main():
    with requests.post(
        URL,
        data={
            # "text": "Generate an image of a cute cat",
            "text": "Generate an image of a cute cat who has discovered the meaning of life",
            "output_modalities": "image",
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

            modality = msg.get("modality")
            data_b64 = msg.get("data", "")

            if not data_b64:
                continue

            decoded = base64.b64decode(data_b64)

            if modality == "text":
                sys.stdout.write(decoded.decode("utf-8", errors="replace"))
                sys.stdout.flush()

            elif modality == "image":
                with open("output.png", "wb") as f:
                    f.write(decoded)
                print("\nSaved image to output.png")


if __name__ == "__main__":
    main()