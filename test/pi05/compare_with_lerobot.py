#!/usr/bin/env python3
"""Compare a running mminf Pi0.5 server's actions against the lerobot
reference implementation under deterministic inputs.

What it does
------------
1. Generates 3 deterministic 224x224 random PNGs (so the server's data_worker
   path runs end-to-end), a fixed prompt, and a fixed robot state.
2. POSTs them to the running mminf server with a pinned ``request_id`` so the
   conductor's per-request RNG seed is reproducible (md5 hash, see
   ``mminf/conductor/conductor.py::_req_id_to_seed``).
3. Recomputes the exact noise tensor the server's Pi05 submodule will sample
   (same seed, same shape, same device).
4. Loads ``lerobot.policies.pi05.PI05Policy.from_pretrained("lerobot/pi05_base")``
   in this process and runs ``sample_actions(...)`` with:
     * the same images (decoded from the same PNGs and normalized identically),
     * the same tokens (same prompt + state, same PaliGemma tokenizer),
     * the same noise (recomputed via the seed above).
5. Compares server's [50, 32] action chunk to lerobot's [1, 50, 32] reference
   and prints diagnostic stats. Returns nonzero on large divergence.

Usage::

    # Make sure the server is running first:
    bash test/pi05/launch_server_pi05.sh keisuke 0
    # Then in another shell:
    python test/pi05/compare_with_lerobot.py
    python test/pi05/compare_with_lerobot.py --port 20002 --tolerance 1e-3

This script does NOT need to live next to the server: ``--host`` / ``--port``
are configurable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from _env import get_server_url
from PIL import Image

# These constants must match Pi0.5's defaults / lerobot/pi05_base.
ACTION_HORIZON = 50
ACTION_DIM = 32
NUM_FLOW_STEPS = 10
HF_REPO = "lerobot/pi05_base"
TOKENIZER_REPO = "google/paligemma-3b-pt-224"
DEFAULT_REQUEST_ID = "pi05_compare_fixed_seed_v1"


# ----------------------------------------------------------------------------
# Determinism helpers — must mirror server-side logic exactly.
# ----------------------------------------------------------------------------

def server_seed_for(request_id: str) -> int:
    """Reproduce ``mminf.conductor.conductor._req_id_to_seed`` exactly."""
    import hashlib

    digest = hashlib.md5(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def reproduce_server_noise(request_id: str, device: torch.device) -> torch.Tensor:
    """Reproduce the noise tensor that ``Pi05LLMSubmodule._preprocess_action_gen``
    will sample on iteration 0 for this request.

    Server code (mminf/model/pi05/submodules.py)::

        generator = torch.Generator(device=device).manual_seed(info.random_seed)
        noisy_actions = torch.randn(action_horizon, action_dim, device=device, generator=generator)
    """
    seed = server_seed_for(request_id)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        ACTION_HORIZON, ACTION_DIM, device=device, generator=generator
    )
    return noise


# ----------------------------------------------------------------------------
# Build deterministic inputs and POST them to the server.
# ----------------------------------------------------------------------------

def build_deterministic_images(seed: int = 12345) -> list[bytes]:
    """Three deterministic 224x224 RGB PNG payloads.

    Saving as PNG (lossless) and loading them back via PIL/torchvision is
    bit-exact at the uint8 level, so the client and the server agree on the
    pre-normalization pixel values.
    """
    rng = np.random.default_rng(seed)
    images = []
    for cam_idx in range(3):
        arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        # Mark each camera with a small constant offset on one channel so we
        # can visually tell them apart in case of debugging.
        arr[:, :, cam_idx] = np.clip(arr[:, :, cam_idx].astype(int) + 8, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        images.append(buf.getvalue())
    return images


def post_to_server(
    *,
    request_id: str,
    text: str,
    state: list[float],
    image_bytes: list[bytes],
) -> np.ndarray:
    """POST a deterministic Pi0.5 request and return the [50, 32] action array."""
    url = get_server_url()
    files = [
        ("files", (f"camera_{i}.png", blob, "image/png"))
        for i, blob in enumerate(image_bytes)
    ]
    data = {
        "text": text,
        "input_modalities": "image,text",
        "output_modalities": "action",
        "streaming": "false",
        "model_kwargs": json.dumps({"robot_state": state}),
        "request_id": request_id,
    }
    print(f"POST  {url}  request_id={request_id}")
    # Generous timeout: the very first request after a server restart pays
    # the full torch.compile cost on both vit_encoder (SigLIP) and the LLM
    # language_model. SigLIP compile alone is often 1–3 minutes.
    resp = requests.post(url, data=data, files=files, timeout=600)
    if not resp.ok:
        print(f"server returned {resp.status_code}:", file=sys.stderr)
        try:
            print(json.dumps(resp.json(), indent=2), file=sys.stderr)
        except Exception:
            print(resp.text, file=sys.stderr)
        resp.raise_for_status()
    payload = resp.json()
    chunks = payload.get("outputs", {}).get("action", [])
    if not chunks:
        raise RuntimeError("server returned no action chunk")
    raw = base64.b64decode(chunks[0]["data"])
    expected = ACTION_HORIZON * ACTION_DIM * 4
    if len(raw) != expected:
        raise RuntimeError(
            f"action payload size {len(raw)} != expected {expected}"
        )
    arr = np.frombuffer(raw, dtype=np.float32).reshape(ACTION_HORIZON, ACTION_DIM)
    return arr.copy()  # detach from the immutable buffer


# ----------------------------------------------------------------------------
# lerobot reference forward.
# ----------------------------------------------------------------------------

def decode_pngs_to_minus1_to_plus1(image_bytes: list[bytes], device: torch.device) -> list[torch.Tensor]:
    """Decode each PNG to a [1, 3, 224, 224] float32 tensor in [-1, 1].

    This MUST match the server's pipeline exactly:
        data_worker.py: torchvision.io.decode_image -> uint8 CHW -> /255 -> float32 [0, 1]
        Pi05ViTEncoderSubmodule._preprocess_one: detects float in [0, 1] -> *2 - 1 -> [-1, 1]
        (224x224 already, so the letterbox resize is a no-op)
    """
    out = []
    for blob in image_bytes:
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)  # HWC uint8
        chw = torch.from_numpy(arr).permute(2, 0, 1).to(device)  # uint8 CHW
        f01 = chw.float() / 255.0
        norm = f01 * 2.0 - 1.0
        out.append(norm.unsqueeze(0))  # [1, 3, 224, 224]
    return out


def build_tokens(
    *, prompt: str, robot_state: list[float], device: torch.device, max_lang_tokens: int = 200
) -> torch.Tensor:
    """Reproduce ``Pi05Model.process_prompt`` exactly: build the openpi prompt
    template, tokenize via the same PaliGemma fast tokenizer."""
    from transformers import AutoTokenizer

    from mminf.model.pi05.components.flow_matching import discretize_state

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, use_fast=True)
    state_t = torch.tensor(robot_state, dtype=torch.float32)
    bins = discretize_state(state_t, num_bins=256).tolist()
    state_str = " ".join(str(b) for b in bins)
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    # Pi05Tokenizer.encode_prompt does .strip().lower() on top.
    full_prompt = full_prompt.strip().lower()
    ids = tokenizer(full_prompt, add_special_tokens=True).input_ids
    ids = ids[:max_lang_tokens]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]


def _stub_broken_lerobot_subpackages():
    """Work around lerobot's broken ``groot`` subpackage in some installs.

    On some lerobot versions, importing ``lerobot.policies`` eagerly imports
    ``lerobot.policies.groot.groot_n1``, which contains a malformed
    ``@dataclass`` declaration that crashes at module-import time. We don't
    need groot at all here — we only want PI05 — so we pre-register fake
    stub modules in ``sys.modules`` before the real ones get a chance to
    load. The stubs satisfy the broken import chain so that
    ``lerobot.policies.__init__`` can finish executing and we can grab
    ``PI05Policy``.
    """
    import sys
    import types

    if "lerobot.policies.groot.groot_n1" in sys.modules:
        return  # already stubbed (or real module already loaded successfully)

    def _make_stub(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # mark as package so submodule imports succeed
        return m

    pkg = _make_stub("lerobot.policies.groot")
    g_n1 = _make_stub("lerobot.policies.groot.groot_n1")
    g_n1.GR00TN15 = type("GR00TN15", (), {})  # fake class for the import line
    cfg = _make_stub("lerobot.policies.groot.configuration_groot")
    cfg.GrootConfig = type("GrootConfig", (), {})
    modg = _make_stub("lerobot.policies.groot.modeling_groot")
    modg.GrootPolicy = type("GrootPolicy", (), {})

    sys.modules["lerobot.policies.groot"] = pkg
    sys.modules["lerobot.policies.groot.groot_n1"] = g_n1
    sys.modules["lerobot.policies.groot.configuration_groot"] = cfg
    sys.modules["lerobot.policies.groot.modeling_groot"] = modg


def run_lerobot_reference(
    *,
    image_bytes: list[bytes],
    prompt: str,
    state: list[float],
    noise: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Run lerobot's reference Pi0.5 forward with the EXACT same noise/inputs."""
    _stub_broken_lerobot_subpackages()
    from lerobot.policies.pi05 import PI05Policy

    print("Loading lerobot reference (this may download ~14 GB on first run) ...")
    policy = PI05Policy.from_pretrained(HF_REPO).to(device).eval()
    ref_model = policy.model
    ref_config = policy.config

    images = decode_pngs_to_minus1_to_plus1(image_bytes, device)
    img_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in images]

    tokens = build_tokens(
        prompt=prompt,
        robot_state=state,
        device=device,
        max_lang_tokens=200,
    )
    masks = torch.ones(tokens.shape, dtype=torch.bool, device=device)

    print(
        f"  prompt token count = {tokens.shape[1]}, image count = {len(images)}, "
        f"noise shape = {tuple(noise.shape)}"
    )

    with torch.no_grad():
        ref_actions = ref_model.sample_actions(
            images=[i.to(torch.float32) for i in images],
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            noise=noise.unsqueeze(0).to(torch.float32),  # [1, 50, 32]
            num_steps=ref_config.num_inference_steps,
        )

    info = {
        "num_inference_steps": ref_config.num_inference_steps,
        "chunk_size": ref_config.chunk_size,
        "max_action_dim": ref_config.max_action_dim,
        "tokens_shape": tuple(tokens.shape),
        "noise_first8": noise[0, :8].cpu().tolist(),
    }
    return ref_actions[0].detach().cpu().numpy(), info


# ----------------------------------------------------------------------------
# Comparison + diagnostics.
# ----------------------------------------------------------------------------

def report(server_actions: np.ndarray, ref_actions: np.ndarray, tolerance: float) -> int:
    """Print diagnostic stats and return shell exit code (0 = within tolerance)."""
    assert server_actions.shape == ref_actions.shape, (
        f"shape mismatch: server={server_actions.shape}  ref={ref_actions.shape}"
    )
    diff = server_actions - ref_actions
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    rel = abs_diff / (np.abs(ref_actions) + 1e-6)
    mean_rel = float(rel.mean())
    max_rel = float(rel.max())

    print()
    print("=" * 70)
    print(f"action shape:        {server_actions.shape}")
    print(f"max abs delta:       {max_abs:.4e}  (tolerance: {tolerance:.0e})")
    print(f"mean abs delta:      {mean_abs:.4e}")
    print(f"max rel error:       {max_rel:.4e}")
    print(f"mean rel error:      {mean_rel:.4e}")
    print()
    print("First step (8 dims):")
    print(f"  server: {server_actions[0, :8].tolist()}")
    print(f"  ref:    {ref_actions[0, :8].tolist()}")
    print(f"  diff:   {diff[0, :8].tolist()}")
    print()
    print("Last step (8 dims):")
    print(f"  server: {server_actions[-1, :8].tolist()}")
    print(f"  ref:    {ref_actions[-1, :8].tolist()}")
    print(f"  diff:   {diff[-1, :8].tolist()}")
    print()

    # Per-dim worst-case
    per_dim_max = abs_diff.max(axis=0)
    worst_dims = np.argsort(per_dim_max)[::-1][:5]
    print("Top-5 worst action dims:")
    for d in worst_dims:
        print(f"  dim {int(d):2d}: max abs delta = {per_dim_max[d]:.4e}")
    print("=" * 70)

    if max_abs <= tolerance:
        print("PASS  (within tolerance)")
        return 0
    print("FAIL  (max abs delta exceeds tolerance — see diagnostics above)")
    return 1


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-id",
        default=DEFAULT_REQUEST_ID,
        help="Pinned request id (md5-hashed into the conductor seed).",
    )
    parser.add_argument("--text", default="pick up the block")
    parser.add_argument(
        "--state",
        default=",".join(["0.0"] * 8),
        help="Comma-separated initial robot state values (padded to action_dim with zeros).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Max-abs-delta threshold for PASS/FAIL. Default 0.2 accounts for "
        "bf16 precision loss in the server path (FlashInfer paged KV cache + "
        "bf16 autocast). The in-process fp32 integration test achieves 4.6e-4.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run lerobot reference + noise on."
    )
    parser.add_argument(
        "--save-images-to",
        type=Path,
        default=None,
        help="If set, dump the deterministic PNGs here for inspection.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ----- 1. Build deterministic inputs -----
    image_bytes = build_deterministic_images()
    if args.save_images_to is not None:
        args.save_images_to.mkdir(parents=True, exist_ok=True)
        for i, blob in enumerate(image_bytes):
            (args.save_images_to / f"camera_{i}.png").write_bytes(blob)
        print(f"saved deterministic images to {args.save_images_to}")

    state_values = [float(x) for x in args.state.split(",") if x.strip()]
    state_values += [0.0] * (ACTION_DIM - len(state_values))
    state_values = state_values[:ACTION_DIM]

    # ----- 2. POST to server -----
    server_actions = post_to_server(
        request_id=args.request_id,
        text=args.text,
        state=state_values,
        image_bytes=image_bytes,
    )
    print(f"server returned action shape: {server_actions.shape}")

    # ----- 3. Reproduce server noise -----
    noise = reproduce_server_noise(args.request_id, device=device)
    print(
        f"reproduced noise: shape={tuple(noise.shape)}  seed={server_seed_for(args.request_id)}  "
        f"first8={noise[0, :8].cpu().tolist()}"
    )

    # ----- 4. Run lerobot reference -----
    ref_actions, info = run_lerobot_reference(
        image_bytes=image_bytes,
        prompt=args.text,
        state=state_values,
        noise=noise,
        device=device,
    )
    print(f"lerobot reference info: {info}")

    # ----- 5. Compare -----
    return report(server_actions, ref_actions, tolerance=args.tolerance)


if __name__ == "__main__":
    sys.exit(main())
