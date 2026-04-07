#!/usr/bin/env python3

import base64
import json
import sys

import requests

from _env import get_server_url

URL = get_server_url()


def main():
    data = {
        "text": "What is the 7th value after the decimal point in pi?",
        "model_kwargs": json.dumps({
            "think_mode": True,
            # "max_output_tokens": 100,
        }),
    }
    with requests.post(
        URL,
        data=data,
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


if __name__ == "__main__":
    main()
