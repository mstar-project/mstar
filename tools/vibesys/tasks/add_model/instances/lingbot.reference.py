"""Reference oracle + candidate I/O for the lingbot-video-dense-1.3b port.

lingbot-video is a text-to-video DiT (LingBotVideoPipeline: Qwen3VL text encoder
-> LingBotVideoTransformer3DModel flow DiT -> AutoencoderKLWan VAE,
FlowUniPCMultistepScheduler). Only the final mp4 is observable over
``/v1/videos/generations``, so there is a single generative walk.

- smoke mode (default): the checker calls ``to_request`` + ``from_response`` and
  asserts a valid, non-empty mp4. No weights, no oracle.
- strict mode: additionally runs the official pipeline as the oracle and
  compares decoded frames. Loading the custom pipeline needs the official
  ``lingbot_video`` package + diffusers 0.37.x; wire it in ``load``/``oracle``
  once a reference clip is captured.
"""

from __future__ import annotations

import base64
from typing import Any

# The eval image bind-mounts weights at /model; local runs fall back to the HF id.
MODEL_DIR = "/model"
HF_ID = "robbyant/lingbot-video-dense-1.3b"
SERVED = "lingbot"

# Fixed, small generation settings: short clip + low steps so a round evaluates
# quickly and deterministically.
PROMPT = "A robot arm neatly stacking wooden blocks on a table, steady camera."
NEGATIVE = ""
HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 17
STEPS = 8
GUIDANCE = 3.0
FPS = 24.0
SEED = 0

_PIPE: Any = None


def sample_input(walk: str) -> dict:
    return {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE,
        "fps": FPS,
        "seed": SEED,
    }


def to_request(walk: str, s: dict) -> tuple[str, str, dict]:
    """POST the text-to-video request to the candidate mstar server."""
    return "POST", "/v1/videos/generations", {
        "model": SERVED,
        "prompt": s["prompt"],
        "negative_prompt": s["negative_prompt"],
        "size": f"{s['width']}x{s['height']}",
        "num_frames": s["num_frames"],
        "num_inference_steps": s["num_inference_steps"],
        "guidance_scale": s["guidance_scale"],
        "fps": s["fps"],
        "seed": s["seed"],
        "response_format": "b64_json",
    }


def from_response(walk: str, response) -> Any:
    """Return the candidate mp4 bytes (smoke). Empty bytes signals invalid output."""
    payload = response.json()
    data = payload.get("data") or []
    if not data or not data[0].get("b64_json"):
        return b""
    blob = base64.b64decode(data[0]["b64_json"])
    # ISO-BMFF / mp4 sanity: the 'ftyp' box appears in the first bytes.
    if b"ftyp" not in blob[:64]:
        return b""
    return blob


# --- strict-mode oracle (wire up once capturing a reference clip) -----------
def load() -> None:
    global _PIPE
    if _PIPE is not None:
        return
    import os

    import torch
    from diffusers import DiffusionPipeline

    src = MODEL_DIR if os.path.isdir(MODEL_DIR) else HF_ID
    _PIPE = DiffusionPipeline.from_pretrained(
        src, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")


def oracle(walk: str, s: dict) -> Any:
    import numpy as np
    import torch

    load()
    out = _PIPE(
        prompt=s["prompt"],
        negative_prompt=s["negative_prompt"] or None,
        height=s["height"],
        width=s["width"],
        num_frames=s["num_frames"],
        num_inference_steps=s["num_inference_steps"],
        guidance_scale=s["guidance_scale"],
        generator=torch.Generator(device="cuda").manual_seed(s["seed"]),
    )
    return np.asarray(out.frames[0])  # (T, H, W, C)
