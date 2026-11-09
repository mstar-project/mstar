"""Write a small Kimi K3.1 Pro checkpoint for GPU parity tests: the release's first layers and
its head, MXFP4 experts dequantized to bf16.

The output is a checkpoint in the release's own names and layout (``language_model.model.*``,
``config.json`` with 1-based layer lists), so ``kimi_k31`` loads it through its normal path with
``strict_config: false``. Kept: layers ``0 .. --layers - 1`` (the default 4 holds KDA + the dense
MLP, two KDA + MoE layers and the first MLA + DSA layer), embeddings, the final norm, the output
attention-residual read and the LM head; and the first ``--experts`` routed experts of each MoE
layer, with the router's rows and correction bias cut to match (the full 896 are ~59 GB a layer
in bf16; 128 keep three MoE layers at ~25 GB). The MXFP4 decode is compressed-tensors'
mxfp4-pack-quantized layout: low nibble first, E2M1 codes, an E8M0 scale per 32 along K (as
merceod's #257 reads it); bf16 holds it exactly.

    python -m mstar.model.kimi_k3.truncate --src <release snapshot> --dst <dir> [--layers 4] [--experts 128]
        [--keep-mxfp4]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_EXPERT = re.compile(r"^(.*\.experts)\.(\d+)\.(w[123])\.(weight_packed|weight_scale)$")


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``uint8 [N, K/2]``, ``uint8 [N, K/32]`` -> bf16 ``[N, K]``."""
    n, half = packed.shape
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(n, 2 * half).long()
    vals = E2M1[codes].view(n, -1, 32) * torch.exp2(scale.float() - 127.0).unsqueeze(-1)
    return vals.view(n, 2 * half).to(torch.bfloat16)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=128)
    # keep the kept experts MXFP4 as the release stores them (and its quantization_config), for the
    # quantization API to load as stored, instead of decoding them to bf16
    ap.add_argument("--keep-mxfp4", action="store_true")
    args = ap.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    def kept(name: str) -> bool:
        if not name.startswith("language_model."):
            return False
        m = _LAYER.match(name)
        if m and int(m.group(1)) >= args.layers:
            return False
        e = _EXPERT.match(name)
        return e is None or int(e.group(2)) < args.experts

    by_file = defaultdict(list)
    for name, shard in weight_map.items():
        if kept(name):
            by_file[shard].append(name)

    out_map, n_out = {}, 0
    for shard, names in sorted(by_file.items()):
        tensors, packed = {}, {}
        with safe_open(str(src / shard), framework="pt") as f:
            for name in names:
                t = f.get_tensor(name)
                e = _EXPERT.match(name)
                if e and not args.keep_mxfp4:
                    base = f"{e.group(1)}.{e.group(2)}.{e.group(3)}"
                    packed.setdefault(base, {})[e.group(4)] = t
                    continue
                if name.endswith(".block_sparse_moe.gate.weight"):
                    t = t[: args.experts].contiguous()
                elif name.endswith(".block_sparse_moe.gate.e_score_correction_bias"):
                    t = t[: args.experts].contiguous()
                tensors[name] = t
        for base, parts in packed.items():
            tensors[f"{base}.weight"] = dequant_mxfp4(parts["weight_packed"], parts["weight_scale"])
        n_out += 1
        out_name = f"model-{n_out:05d}.safetensors"
        save_file(tensors, str(dst / out_name), metadata={"format": "pt"})
        out_map.update(dict.fromkeys(tensors, out_name))
        print(f"{shard}: {len(tensors)} tensors -> {out_name}", flush=True)

    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    n = args.layers
    linear = text["linear_attn_config"]
    linear["kda_layers"] = [i for i in linear["kda_layers"] if i <= n]
    linear["full_attn_layers"] = [i for i in linear["full_attn_layers"] if i <= n]
    text["dsa_config"]["dsa_layers"] = list(linear["full_attn_layers"])
    text["num_hidden_layers"] = n
    text["num_experts"] = args.experts
    if not args.keep_mxfp4:
        text.pop("quantization_config", None)
        config.pop("quantization_config", None)
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": out_map}, indent=1))
    for extra in ("generation_config.json",):
        if (src / extra).is_file():
            shutil.copy(src / extra, dst / extra)
    print(f"wrote {len(out_map)} tensors, {n} layers, {args.experts} experts a MoE layer to {dst}")


if __name__ == "__main__":
    main()
