#!/usr/bin/env python3

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from _env import get_server_url

URL = get_server_url()

prompts = [
    "What is the 7th value after the decimal point in pi?",
    "Explain what a black hole is in one paragraph.",
    "Write a haiku about recursion.",
    "What is 3 * 4?",
    "Summarize the theory of relativity briefly.",
    "What is the capital of France?",
    "Explain gradient descent in simple terms. Please respond in English.",
    "What is the Fibonacci sequence?",
    "Give a fun fact about octopuses.",
]

print_lock = threading.Lock()
start_barrier = threading.Barrier(len(prompts))  # ensure simultaneous start


def make_request(prompt: str, idx: int) -> str:
    data = {
        "text": prompt,
        "model_kwargs": json.dumps({
            "think_mode": True,
            # "max_output_tokens": 400,
        }),
    }

    output_chunks = []

    # Synchronize all threads to start at once
    start_barrier.wait()

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
                output_chunks.append(decoded.decode("utf-8", errors="replace"))

    result = "".join(output_chunks)

    # Thread-safe print
    with print_lock:
        print(f"PROMPT={prompt}\nRES={result}\n" + ("-" * 60))

    return result


def main():
    results = []

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(make_request, prompt, i)
            for i, prompt in enumerate(prompts)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # # Optional: final summary print
    # print("\n===== ALL RESPONSES COLLECTED =====")
    # for i, res in enumerate(results):
    #     print(f"--- {i} ---\n{res}\n")


if __name__ == "__main__":
    main()
