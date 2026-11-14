"""Bounded DROID checkpoint gate: all shard headers, strict tower loads, and one GPU expert pass.

Run with the pinned Praxis reference on PYTHONPATH. ``--mode index`` checks the
public config and index without the large shard files. The default ``full``
mode reads the local shards and needs a CUDA GPU. It does not enable serving.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from praxis.action_expert import build_trunk_rope_masks
from praxis.checkpoint import expected_dtype, load_config, load_state_dict_from_dir, strict_load
from praxis.model import PraxisPolicy
from safetensors import safe_open

from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.native_towers import build_backbone, build_expert
from mstar.model.praxis1.weight_loader import _local_key, load_component


def _modules(root: Path):
    config = Praxis1Config.from_checkpoint(root)
    published = load_config(root)
    with torch.device("meta"):
        native_backbone = build_backbone(config)
        native_expert = build_expert(config)
        reference = PraxisPolicy.from_config(published)
    components = (
        ("backbone.encoder.", native_backbone.encoder, reference.backbone.encoder),
        ("backbone.decoder.", native_backbone.decoder, reference.backbone.decoder),
        ("action_expert.", native_expert, reference.action_expert),
    )
    return config, published, native_backbone, native_expert, reference, components


def _check_index(root: Path, components) -> dict[str, str]:
    raw = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = raw["weight_map"]
    assert isinstance(weight_map, dict) and weight_map, "empty or invalid weight map"
    seen = set()
    for prefix, native, reference in components:
        native_keys = native.state_dict()
        reference_keys = reference.state_dict()
        indexed = {key[len(prefix):] for key in weight_map if key.startswith(prefix)}
        assert indexed == set(reference_keys), (
            prefix, "reference key mismatch", sorted(indexed - set(reference_keys))[:5],
            sorted(set(reference_keys) - indexed)[:5],
        )
        translated = {_local_key(key) for key in indexed}
        assert len(translated) == len(indexed), (prefix, "key translation collision")
        assert translated == set(native_keys), (
            prefix, "native key mismatch", sorted(translated - set(native_keys))[:5],
            sorted(set(native_keys) - translated)[:5],
        )
        seen.update(prefix + key for key in indexed)
    assert seen == set(weight_map), ("unhandled checkpoint keys", sorted(set(weight_map) - seen)[:5])
    return weight_map


def _check_headers(root: Path, published: dict, components, weight_map: dict[str, str]) -> None:
    expected = {}
    for prefix, native, reference in components:
        native_state = native.state_dict()
        reference_state = reference.state_dict()
        for name in reference_state:
            key = prefix + name
            native_shape = tuple(native_state[_local_key(name)].shape)
            reference_shape = tuple(reference_state[name].shape)
            assert native_shape == reference_shape, (key, native_shape, reference_shape)
            expected[key] = (native_shape, expected_dtype(published, key))
    dtype_names = {torch.bfloat16: "BF16", torch.float32: "F32"}
    for filename in sorted(set(weight_map.values())):
        shard_path = root / filename
        assert shard_path.is_file(), f"missing shard {shard_path}"
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            indexed = {key for key, file in weight_map.items() if file == filename}
            assert set(shard.keys()) == indexed, (filename, "index differs from shard")
            for key in indexed:
                tensor = shard.get_slice(key)
                shape, dtype = expected[key]
                assert tuple(tensor.get_shape()) == shape, (key, tensor.get_shape(), shape)
                assert tensor.get_dtype() == dtype_names[dtype], (key, tensor.get_dtype(), dtype)


def _load_native(root: Path, config: Praxis1Config, expert: torch.nn.Module) -> None:
    # Load one tower at a time so a 54 GB checkpoint does not occupy memory twice.
    for prefix, component in (
        ("backbone.encoder.", "encoder"),
        ("backbone.decoder.", "decoder"),
        ("action_expert.", None),
    ):
        if component is None:
            native = expert
        else:
            with torch.device("meta"):
                backbone = build_backbone(config)
            native = getattr(backbone, component)
        started = time.perf_counter()
        load_component(native, root, prefix, assign=True)
        assert all(not value.is_meta for value in native.state_dict().values()), (prefix, "unloaded meta tensor")
        print(json.dumps({
            "phase": "strict_load", "component": prefix,
            "seconds": round(time.perf_counter() - started, 1),
        }), flush=True)
        if component is not None:
            del native, backbone
            gc.collect()


def _expert_parity(root: Path, config: Praxis1Config, native: torch.nn.Module, reference: torch.nn.Module) -> dict:
    state = load_state_dict_from_dir(root, prefix="action_expert.", device="cpu")
    strict_load(reference, state, assign=True)
    del state
    reference.attention = "sdpa"
    reference.flex_kernel_options = None
    native = native.eval().requires_grad_(False).to("cuda")
    reference = reference.eval().requires_grad_(False).to("cuda")
    torch.cuda.reset_peak_memory_stats()

    # One cold chunk with the published expert's widths. Two video tokens per
    # frame keep the gate short while still exercising every attention layer.
    chunk = config.latents_per_chunk
    tpf, globals_ = 2, 5
    n_noised = chunk
    generator = torch.Generator(device="cuda").manual_seed(81)
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    args = dict(
        noised_action_latents=rand(1, chunk, config.action_patch_dim),
        action_timesteps=torch.full((1,), 0.5, device="cuda", dtype=torch.bfloat16),
        trunk_features=[rand(1, chunk * tpf, config.trunk_dim) for _ in range(reference.num_cross_layers)],
        global_tokens=rand(1, globals_, config.trunk_dim),
        proprio_latents=rand(1, chunk, config.proprio_patch_dim),
        kv_rope_ids=torch.arange(globals_ + chunk * tpf, device="cuda", dtype=torch.long)[None],
        chunk_size=chunk,
        num_clean_frames=0,
        num_noised_frames=n_noised,
        tokens_per_frame=tpf,
        window_size=chunk,
    )
    cross, self_mask = build_trunk_rope_masks(
        num_action_tokens=chunk, num_future_per_chunk=reference.num_future_per_chunk,
        chunk_size=chunk, num_clean_frames=0, num_noised_frames=n_noised,
        tokens_per_frame=tpf, window_size=chunk, num_global_tokens=globals_,
        device=torch.device("cuda"), include_noised=not reference.clean_only_conditioning,
        block=False,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        expected = reference.forward_trunk_rope(**args)
        actual = native.forward_trunk_rope(**args, cross_mask=cross, self_mask=self_mask)
    torch.cuda.synchronize()
    assert expected.isfinite().all() and actual.isfinite().all(), "non-finite expert output"
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)
    return {
        "shape": list(actual.shape), "max_abs": float((actual - expected).abs().max()),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("index", "header", "full"), default="full")
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    started = time.perf_counter()
    config, published, backbone, expert, reference, components = _modules(root)
    weight_map = _check_index(root, components)
    print(json.dumps({"phase": "index", "keys": len(weight_map), "shards": len(set(weight_map.values()))}), flush=True)
    if args.mode == "index":
        return
    _check_headers(root, published, components, weight_map)
    print(json.dumps({"phase": "headers", "checked": len(weight_map)}), flush=True)
    if args.mode == "header":
        return
    assert torch.cuda.is_available(), "full mode needs CUDA"
    _load_native(root, config, expert)
    result = _expert_parity(root, config, expert, reference.action_expert)
    print(json.dumps({
        "phase": "expert_parity", **result,
        "total_seconds": round(time.perf_counter() - started, 1),
    }), flush=True)
    del backbone


if __name__ == "__main__":
    main()
