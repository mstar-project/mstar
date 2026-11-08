#!/usr/bin/env python3

import base64
import json
import sys

import requests
from _env import get_server_url

URL = get_server_url()


def main():
    with requests.post(
        URL,
        data={
            "text": "A cat in a suit and tie",
            "output_modalities": "image",
            "model_kwargs": json.dumps({
                "think_mode": True,
            }),
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
                # raw bytes: a character can span two chunks
                sys.stdout.buffer.write(decoded)
                sys.stdout.buffer.flush()

            elif modality == "image":
                with open("output.png", "wb") as f:
                    f.write(decoded)
                print("\nSaved image to output.png")


if __name__ == "__main__":
    main()
