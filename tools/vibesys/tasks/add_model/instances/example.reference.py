"""Example oracle + client I/O (a TEMPLATE — copy to <model>.reference.py).

The ONE model-specific eval file. The modality checker/benchmark call these
hooks. It defines a single client request/response and the reference oracle;
it must NOT reuse the candidate's mstar code (the oracle is the independent
upstream model). Real <model>.reference.py files are gitignored.

This example targets text->video (mp4 over /v1/videos/generations). For
diffusion/flow models the strict gate feeds ONE shared initial latent to both
the candidate (via `init_latents` in the request) and the oracle (via the
pipeline `latents=` kwarg), so the comparison measures compute, not the noise
draw.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

MODEL_DIR = "/model"
HF_ID = "org/my-model"
SERVED = "mymodel"
# Small, fixed, deterministic probe.
PROMPT, SEED, H, W, NF, STEPS, GUID = "a fixed probe prompt", 0, 480, 832, 9, 8, 3.0
_PIPE: Any = None


def sample_input() -> dict:
    return {"prompt": PROMPT, "seed": SEED, "height": H, "width": W,
            "num_frames": NF, "num_inference_steps": STEPS, "guidance_scale": GUID}


def _b64_npy(arr) -> str:
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def initial_latent(s: dict) -> Any:
    """Shared starting noise, shape [B, C, T_lat, H/f, W/f]. Deterministic (seed)."""
    import torch
    load()
    g = torch.Generator(device="cuda").manual_seed(s["seed"])
    ncl = _PIPE.transformer.config.in_channels
    lat = _PIPE.prepare_latents(1, ncl, s["height"], s["width"], s["num_frames"],
                                torch.bfloat16, torch.device("cuda"), g)
    return lat.detach().to(torch.float32).cpu().numpy()


def to_request(s: dict, latents: Any = None) -> tuple[str, str, dict]:
    body = {
        "model": SERVED, "prompt": s["prompt"], "size": f"{s['width']}x{s['height']}",
        "num_frames": s["num_frames"], "num_inference_steps": s["num_inference_steps"],
        "guidance_scale": s["guidance_scale"], "seed": s["seed"], "response_format": "b64_json",
    }
    if latents is not None:
        body["init_latents"] = _b64_npy(latents)
    return "POST", "/v1/videos/generations", body


def _decode_mp4(blob: bytes):
    import av
    import numpy as np
    container = av.open(io.BytesIO(blob))
    return np.stack([np.asarray(f.to_image()) for f in container.decode(video=0)])


def from_response(response) -> Any:
    data = (response.json().get("data") or [{}])[0]
    if not data.get("b64_json"):
        return None
    blob = base64.b64decode(data["b64_json"])
    if b"ftyp" not in blob[:64]:
        return None
    return _decode_mp4(blob)


def load() -> None:
    global _PIPE
    if _PIPE is not None:
        return
    import torch
    from diffusers import DiffusionPipeline  # or the upstream package's pipeline
    src = MODEL_DIR if os.path.isdir(MODEL_DIR) else HF_ID
    _PIPE = DiffusionPipeline.from_pretrained(src, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")


def oracle(s: dict, latents: Any = None) -> Any:
    import numpy as np
    import torch
    load()
    kw: dict = {}
    if latents is not None:
        kw["latents"] = torch.from_numpy(np.asarray(latents, dtype=np.float32)).to("cuda", torch.bfloat16)
    else:
        kw["generator"] = torch.Generator(device="cuda").manual_seed(s["seed"])
    out = _PIPE(prompt=s["prompt"], height=s["height"], width=s["width"], num_frames=s["num_frames"],
                num_inference_steps=s["num_inference_steps"], guidance_scale=s["guidance_scale"], **kw)
    return np.asarray(out.frames[0])
