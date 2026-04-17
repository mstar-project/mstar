#!/usr/bin/env python3
"""Pi0.5 client: posts a robotics observation (3 RGB images + task prompt +
robot state) and decodes the resulting action trajectory.

Usage::

    # Default: 3 random 224x224 images, prompt "pick up the block",
    # zero state vector. Just exercises the server end-to-end.
    python test/pi05/action_request.py

    # Real images on disk + a state vector
    python test/pi05/action_request.py \
        --base-image base.png --left-image left.png --right-image right.png \
        --text "pick up the red block" \
        --state "0.1,0.2,-0.3,..." \
        --port 20002

The server returns a single ``action`` chunk whose ``data`` field is the
raw float32 bytes of a ``[chunk_size, action_dim]`` tensor (default
``[50, 32]`` for Pi0.5). We decode it back to a numpy array and print
some summary statistics.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys

import numpy as np
import requests
from PIL import Image

ACTION_HORIZON = 50
ACTION_DIM = 32


def _make_random_image(seed: int) -> bytes:
    """Generate a deterministic 224x224 PNG for smoke testing."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _load_image_bytes(path: str | None, fallback_seed: int) -> tuple[str, bytes]:
    if path is None:
        return f"random_{fallback_seed}.png", _make_random_image(fallback_seed)
    with open(path, "rb") as f:
        return path.rsplit("/", 1)[-1], f.read()


def _parse_state(state_arg: str | None) -> list[float]:
    """Parse the --state argument as a comma-separated float list and pad
    with zeros up to ``ACTION_DIM``."""
    if not state_arg:
        return [0.0] * ACTION_DIM
    values = [float(x) for x in state_arg.split(",") if x.strip()]
    if len(values) > ACTION_DIM:
        raise ValueError(
            f"--state has {len(values)} values, max is action_dim={ACTION_DIM}"
        )
    return values + [0.0] * (ACTION_DIM - len(values))


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 inference client")
    parser.add_argument("--base-image", type=str, default=None,
                        help="Path to base camera image (defaults to a random 224x224 PNG).")
    parser.add_argument("--left-image", type=str, default=None,
                        help="Path to left wrist camera image.")
    parser.add_argument("--right-image", type=str, default=None,
                        help="Path to right wrist camera image.")
    parser.add_argument("--text", default="pick up the block",
                        help="Task description.")
    parser.add_argument("--state", default=None,
                        help="Comma-separated robot state values (padded to action_dim with zeros).")
    parser.add_argument("--port", type=int, default=20002,
                        help="Port number to connect to (localhost only).")
    parser.add_argument("--streaming", action="store_true",
                        help="Use the NDJSON streaming endpoint instead of blocking.")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/generate"
    print(f"text:  {args.text!r}")
    print(f"state: {_parse_state(args.state)[:8]}{'...' if ACTION_DIM > 8 else ''}")
    print(f"POST   {url}")

    # Build the multipart form: 3 images + text + model_kwargs (state).
    # The server's _detect_modality routes anything ending in .png/.jpg to
    # the "image" modality bucket.
    img_files = [
        ("files", _load_image_bytes(args.base_image, fallback_seed=0)),
        ("files", _load_image_bytes(args.left_image, fallback_seed=1)),
        ("files", _load_image_bytes(args.right_image, fallback_seed=2)),
    ]
    files = [(field, (name, blob, "image/png")) for field, (name, blob) in img_files]

    data = {
        "text": args.text,
        "input_modalities": "image,text",
        "output_modalities": "action",
        "streaming": "true" if args.streaming else "false",
        "model_kwargs": json.dumps({"robot_state": _parse_state(args.state)}),
    }

    if args.streaming:
        actions = _post_streaming(url, data, files)
    else:
        actions = _post_blocking(url, data, files)

    if actions is None:
        print("No action data received from server.")
        sys.exit(1)

    print()
    print(f"action trajectory shape: {actions.shape}")
    print(f"  abs max:  {float(np.abs(actions).max()):.4f}")
    print(f"  mean abs: {float(np.abs(actions).mean()):.4f}")
    print(f"  first step (8 dims): {actions[0, :8].tolist()}")
    print(f"  last step  (8 dims): {actions[-1, :8].tolist()}")


def _post_blocking(url, data, files) -> np.ndarray | None:
    resp = requests.post(url, data=data, files=files)
    if not resp.ok:
        # FastAPI puts the server-side exception message in the response body's
        # ``detail`` field via ``HTTPException(detail=str(e))``; print it so we
        # can see what went wrong without having to grep server logs.
        print(f"server returned {resp.status_code}:", file=sys.stderr)
        try:
            print(json.dumps(resp.json(), indent=2), file=sys.stderr)
        except Exception:
            print(resp.text, file=sys.stderr)
        resp.raise_for_status()
    payload = resp.json()
    chunks = payload.get("outputs", {}).get("action", [])
    if not chunks:
        return None
    raw = base64.b64decode(chunks[0]["data"])
    return _decode_action_bytes(raw)


def _post_streaming(url, data, files) -> np.ndarray | None:
    last: np.ndarray | None = None
    with requests.post(url, data=data, files=files, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("modality") != "action":
                continue
            data_b64 = msg.get("data", "")
            if not data_b64:
                continue
            last = _decode_action_bytes(base64.b64decode(data_b64))
    return last


def _decode_action_bytes(raw: bytes) -> np.ndarray:
    expected = ACTION_HORIZON * ACTION_DIM * 4  # float32
    if len(raw) != expected:
        print(
            f"warning: action payload size {len(raw)} != expected {expected};"
            " server may be using a different chunk_size/action_dim",
            file=sys.stderr,
        )
    arr = np.frombuffer(raw, dtype=np.float32)
    n_steps = arr.size // ACTION_DIM
    return arr.reshape(n_steps, ACTION_DIM)


if __name__ == "__main__":
    main()
