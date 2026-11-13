"""Write a bf16 checkpoint's linears as block fp8, in the layout DeepSeek, Qwen and GLM ship and vLLM and
SGLang read (``weight`` e4m3 plus ``weight_scale_inv`` per 128 x 128 block, ``quant_method: fp8``).

For a model with no official fp8 checkpoint, and so a pre-quantized load can be checked against quantizing
the same weights at load: the two give the same tensors when the shard boundaries fall on blocks.

    python -m mstar.quantization.convert SRC DST --include 'thinker.model.layers.*.mlp.experts.*' \\
        --include 'thinker.model.layers.*.self_attn.*_proj'

``--include`` / ``--exclude`` patterns are ``fnmatch`` over checkpoint linear names (``*`` crosses dots); only
2-D ``.weight`` tensors included and not excluded are converted. Shards are rewritten one at a time; every
other file is copied.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from mstar.utils.fp8_block import quantize_fp8_block_weight


def convert_checkpoint(
    src: str | Path, dst: str | Path, include: Sequence[str], exclude: Sequence[str] = (),
    block: tuple[int, int] = (128, 128),
) -> list[str]:
    """Convert the linears ``include`` matches and ``exclude`` doesn't; returns their names."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    converted: list[str] = []
    kept: list[str] = []
    weight_map: dict[str, str] = {}
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {src}")
    for shard in shards:
        tensors = load_file(str(shard))
        out: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            base = name.removesuffix(".weight")
            if (name.endswith(".weight") and tensor.dim() == 2 and tensor.dtype in (torch.bfloat16, torch.float16)
                    and any(fnmatch.fnmatchcase(base, pattern) for pattern in include)
                    and not any(fnmatch.fnmatchcase(base, pattern) for pattern in exclude)):
                q, scale_inv = quantize_fp8_block_weight(tensor, block)
                out[name], out[f"{base}.weight_scale_inv"] = q, scale_inv
                converted.append(base)
            else:
                out[name] = tensor
                if name.endswith(".weight") and tensor.dim() == 2:
                    kept.append(base)
        save_file(out, str(dst / shard.name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(out, shard.name))
    for path in src.iterdir():
        if path.suffix == ".safetensors" or path.name in ("model.safetensors.index.json", "config.json"):
            continue
        if path.is_file():
            shutil.copy2(path, dst / path.name)
    if (src / "model.safetensors.index.json").is_file():
        index = json.loads((src / "model.safetensors.index.json").read_text())
        index["weight_map"] = weight_map
        (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    config = json.loads((src / "config.json").read_text()) if (src / "config.json").is_file() else {}
    # the linears left bf16, under both names readers use (vLLM's `ignored_layers`, HF's `modules_to_not_convert`)
    config["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": list(block),
        "modules_to_not_convert": sorted(kept), "ignored_layers": sorted(kept),
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    return converted


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--include", action="append", required=True, help="fnmatch over linear names")
    parser.add_argument("--exclude", action="append", default=[], help="fnmatch over linear names")
    args = parser.parse_args(argv)
    names = convert_checkpoint(args.src, args.dst, args.include, args.exclude)
    print(f"converted {len(names)} linears to block fp8 under {args.dst}")


if __name__ == "__main__":
    main()
