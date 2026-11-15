"""Check published YAM shards and compare one real-weight backbone pass.

The header and strict-load phases cover encoder, decoder and action expert.
The GPU phase compares one expert pass, then encoder taps and decoder velocity
with the pinned reference under the same eager SDPA mask. The default 1x2
latent grid bounds attention memory; this is a component gate, not YAM serving.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from praxis1_checkpoint_gate import _check_headers, _check_index, _expert_parity, _load_native, _modules
from praxis1_real_backbone_probe import _compare, _inputs, _run_native, _run_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mstar-revision", required=True)
    parser.add_argument("--reference-revision", default="68f4d7a254b4bf41951d2e97f44b07f6cac2bf69")
    parser.add_argument("--grid-height", type=int, default=1)
    parser.add_argument("--grid-width", type=int, default=2)
    parser.add_argument("--mode", choices=("headers", "full"), default="full")
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    assert root.is_dir() and (root / "config.json").is_file(), "missing local YAM checkpoint"
    assert args.grid_height > 0 and args.grid_width > 0
    started = time.perf_counter()
    config, published, backbone, expert, reference, components = _modules(root)
    assert config.embodiment == "bimanual", f"expected YAM bimanual config, got {config.embodiment!r}"
    print(json.dumps({"phase": "source", "checkpoint": str(root),
                      "mstar_revision": args.mstar_revision,
                      "reference_revision": args.reference_revision,
                      "embodiment": config.embodiment,
                      "grid": [args.grid_height, args.grid_width]}), flush=True)
    weight_map = _check_index(root, components)
    _check_headers(root, published, components, weight_map)
    print(json.dumps({"phase": "headers", "keys": len(weight_map),
                      "shards": len(set(weight_map.values()))}), flush=True)
    if args.mode == "headers":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("YAM full checkpoint parity needs a CUDA GPU")
    _load_native(root, config, expert)
    print(json.dumps({"phase": "strict_load_complete", "seconds":
                      round(time.perf_counter() - started, 1)}), flush=True)
    result = _expert_parity(root, config, expert, reference.action_expert)
    print(json.dumps({"phase": "expert_parity", **result}), flush=True)
    del backbone, expert, reference, components
    gc.collect()
    torch.cuda.empty_cache()
    inputs = _inputs(config, args.grid_height, args.grid_width)
    expected_velocity, expected_tap = _run_reference(root, inputs)
    # Report the native tower's peak separately from the reference and
    # expert passes, whose allocations have already been released.
    torch.cuda.reset_peak_memory_stats()
    actual_velocity, actual_tap = _run_native(root, config, inputs)
    _compare("yam_velocity", actual_velocity, expected_velocity)
    _compare("yam_last_tap", actual_tap, expected_tap)
    print(json.dumps({"phase": "complete", "seconds":
                      round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
