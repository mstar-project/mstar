#!/usr/bin/env python3

import base64
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

URL = "http://0.0.0.0:8000/generate"

# 5 image paths (can be the same image if you just want to test batching)
IMAGE_PATHS = [
    "test/bagel/bagel.png",
    "test/bagel/bagel.png",
    "test/bagel/bagel.png",
    "test/bagel/bagel.png",
    "test/bagel/bagel.png"
]

# Barrier to synchronize all threads
barrier = threading.Barrier(len(IMAGE_PATHS))

# Lock for thread-safe printing
print_lock = threading.Lock()


def make_request(image_path: str):
    image_path = Path(image_path)

    with open(image_path, "rb") as f:
        files = [
            ("files", (image_path.name, f, "application/octet-stream")),
        ]

        data = {
            "text": "Please describe this image in detail",
            # "model_kwargs": json.dumps({
            #     "think_mode": True,
            # })
        }

        # Wait for all threads to be ready
        barrier.wait()

        try:
            with requests.post(URL, data=data, files=files, stream=True) as resp:
                resp.raise_for_status()

                buffer = []

                for line in resp.iter_lines():
                    if not line:
                        continue

                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("modality") == "text":
                        decoded = base64.b64decode(msg.get("data", ""))
                        text = decoded.decode("utf-8", errors="replace")
                        buffer.append(text)

                # Thread-safe print
                with print_lock:
                    print(f"\n=== Response for {image_path} ===")
                    buffer = "".join(buffer)
                    print(f"[{image_path.name}] {buffer}", end="")
                    print(f"\n=== End of {image_path} ===\n")

        except Exception as e:
            with print_lock:
                print(f"[{image_path}] ERROR: {e}")


def main():
    with ThreadPoolExecutor(max_workers=len(IMAGE_PATHS)) as executor:
        futures = [executor.submit(make_request, p) for p in IMAGE_PATHS]

        for future in as_completed(futures):
            future.result()  # raise any exceptions


if __name__ == "__main__":
    main()