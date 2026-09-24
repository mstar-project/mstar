"""Check native HTTP serving with create_smoke_checkpoint's random tiny model."""

import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18937")
    args = parser.parse_args()

    def run(budget):
        response = requests.post(args.url.rstrip("/") + "/generate", data={
            "text": "Hello from the Command A+ smoke test", "streaming": "false",
            "output_modalities": "text", "model_kwargs": json.dumps({
                "max_output_tokens": budget, "temperature": 0,
                "repetition_penalty": 1, "ignore_eos": True,
            }),
        }, timeout=70)
        response.raise_for_status()
        chunks = response.json()["outputs"].get("text", [])
        if len(chunks) != budget:
            raise AssertionError((budget, response.json()))
        output = b"".join(base64.b64decode(chunk["data"]) for chunk in chunks)
        print(f"budget={budget}, chunks={len(chunks)}, text={output.decode('utf-8')!r}")

    run(1)
    run(6)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, [3, 7]))
    print("Sequential/concurrent HTTP requests and exact output budgets passed")


if __name__ == "__main__":
    main()
