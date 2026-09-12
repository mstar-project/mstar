"""Collect real-checkpoint evidence for the QwenVL PR-0 acceptance gates.

Run ``checkpoint`` on the same GPU that will serve the model, then start
``mstar serve qwenvl`` and run ``server`` against it. Results are emitted as
JSON so they can be attached to the PR without treating CPU tests as system
acceptance.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _write_evidence(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output is not None:
        output.write_text(rendered + "\n")


def checkpoint_evidence(args: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import snapshot_download

    from mstar.model.qwenvl.qwenvl_model import QwenVLModel

    if not torch.cuda.is_available():
        raise RuntimeError("PR-0 checkpoint acceptance requires a CUDA GPU.")
    local_dir = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    model = QwenVLModel(local_dir)
    config = model.config
    model.get_submodule("LLM", device=args.device, autocast_dtype=torch.bfloat16)
    model.get_submodule("vision_encoder", device=args.device, autocast_dtype=torch.bfloat16)
    torch.cuda.synchronize(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    allocated = torch.cuda.max_memory_allocated(args.device)
    reserved = torch.cuda.max_memory_reserved(args.device)
    total = properties.total_memory
    _write_evidence(
        {
            "gates": ["P0-G1", "P0-G2", "P0-G7"],
            "model": args.model,
            "revision": args.revision,
            "snapshot": local_dir,
            "device": properties.name,
            "config": {
                "model_type": config.model_type,
                "hidden_size": config.text_config.hidden_size,
                "num_hidden_layers": config.text_config.num_hidden_layers,
                "num_experts": config.text_config.num_experts,
                "spatial_merge_size": config.vision_config.spatial_merge_size,
            },
            "load": {
                "llm": "complete",
                "vision_encoder": "complete",
            },
            "memory_gib": {
                "peak_allocated": allocated / 2**30,
                "peak_reserved": reserved / 2**30,
                "device_total": total / 2**30,
                "reserved_headroom": (total - reserved) / 2**30,
            },
        },
        args.output,
    )


def _checkerboard_png() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (96, 64))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (255, 255, 255) if (x // 16 + y // 16) % 2 else (0, 0, 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def position_evidence(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeModel,
        Qwen3VLMoeTextRotaryEmbedding,
    )

    from mstar.model.qwenvl.components import compute_mrope_cos_sin
    from mstar.model.qwenvl.qwenvl_model import QwenVLModel

    if not torch.cuda.is_available():
        raise RuntimeError("PR-0 position acceptance requires a CUDA GPU.")
    local_dir = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"],
    )
    model = QwenVLModel(local_dir)
    image = Image.open(io.BytesIO(_checkerboard_png())).convert("RGB")
    image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    processed = model.process_prompt(
        "Describe the dominant colors and pattern in this image in one sentence.",
        ["image", "text"],
        ["text"],
        {"image_inputs": [image_tensor]},
    )
    input_ids = processed["text_inputs"][0]
    grid = processed["image_grid_thw"][0]
    actual_positions = processed["position_ids"][0]
    expected_positions, _ = Qwen3VLMoeModel.get_rope_index(
        SimpleNamespace(config=model.config),
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=grid,
    )
    expected_positions = expected_positions[:, 0]
    torch.testing.assert_close(actual_positions, expected_positions, atol=0, rtol=0)

    device = torch.device(args.device)
    actual_cos, actual_sin = compute_mrope_cos_sin(
        actual_positions.to(device),
        head_dim=model.config.text_config.head_dim,
        rope_theta=model.config.text_config.rope_theta,
        mrope_section=tuple(model.config.text_config.rope_scaling["mrope_section"]),
        dtype=torch.bfloat16,
    )
    reference_rope = Qwen3VLMoeTextRotaryEmbedding(model.config.text_config, device=device)
    expected_cos, expected_sin = reference_rope(
        torch.empty(1, device=device, dtype=torch.bfloat16),
        expected_positions[:, None, :].to(device),
    )
    expected_cos, expected_sin = expected_cos[0], expected_sin[0]
    torch.testing.assert_close(actual_cos, expected_cos, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(actual_sin, expected_sin, atol=args.atol, rtol=args.rtol)
    _write_evidence(
        {
            "gate": "P0-G3",
            "model": args.model,
            "revision": args.revision,
            "device": torch.cuda.get_device_name(device),
            "image_grid_thw": grid.tolist(),
            "position_ids_exact": True,
            "cos_max_abs_error": (actual_cos - expected_cos).abs().max().item(),
            "sin_max_abs_error": (actual_sin - expected_sin).abs().max().item(),
            "atol": args.atol,
            "rtol": args.rtol,
        },
        args.output,
    )


def server_evidence(args: argparse.Namespace) -> None:
    from mstar import MStarClient
    from mstar.client import TextChunk

    client = MStarClient(args.url, timeout=args.timeout)
    if not client.health():
        raise RuntimeError(f"QwenVL server is not healthy at {args.url}.")
    chunks: list[str] = []
    events = client.chat(
        "Describe the dominant colors and pattern in this image in one sentence.",
        images=[("qwenvl-pr0-checkerboard.png", _checkerboard_png())],
        stream=True,
        temperature=0.0,
        max_output_tokens=args.max_output_tokens,
    )
    for event in events:
        if isinstance(event, TextChunk):
            chunks.append(event.text)
    text = "".join(chunks)
    if not text.strip():
        raise RuntimeError("QwenVL server returned no decoded text.")
    if "<|" in text:
        raise RuntimeError(f"QwenVL server leaked a special token: {text!r}")
    _write_evidence(
        {
            "gate": "P0-G6",
            "url": args.url,
            "prompt": "Describe the dominant colors and pattern in this image in one sentence.",
            "temperature": 0.0,
            "max_output_tokens": args.max_output_tokens,
            "stream_chunk_count": len(chunks),
            "decoded_text": text,
            "human_coherence_review_required": True,
        },
        args.output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint", help="Validate real config/load and record peak memory")
    checkpoint.add_argument("--model", default=DEFAULT_MODEL)
    checkpoint.add_argument("--revision", required=True, help="Immutable Hub commit SHA")
    checkpoint.add_argument("--cache-dir")
    checkpoint.add_argument("--device", default="cuda:0")
    checkpoint.add_argument("--output", type=Path)
    checkpoint.set_defaults(run=checkpoint_evidence)

    positions = subparsers.add_parser("positions", help="Compare real processor positions and rotary tensors")
    positions.add_argument("--model", default=DEFAULT_MODEL)
    positions.add_argument("--revision", required=True, help="Immutable Hub commit SHA")
    positions.add_argument("--cache-dir")
    positions.add_argument("--device", default="cuda:0")
    positions.add_argument("--atol", type=float, default=2e-3)
    positions.add_argument("--rtol", type=float, default=2e-3)
    positions.add_argument("--output", type=Path)
    positions.set_defaults(run=position_evidence)

    server = subparsers.add_parser("server", help="Run a deterministic image-chat streaming smoke")
    server.add_argument("--url", default="http://localhost:8000")
    server.add_argument("--timeout", type=float, default=600.0)
    server.add_argument("--max-output-tokens", type=int, default=64)
    server.add_argument("--output", type=Path)
    server.set_defaults(run=server_evidence)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.run(parsed)
