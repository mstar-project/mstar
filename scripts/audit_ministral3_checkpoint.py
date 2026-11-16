"""Audit a local safetensors header JSON; never downloads or loads weight values.

Example from the repository root:
  python -m scripts.audit_ministral3_checkpoint \
    --metadata test/modular/fixtures/ministral3 \
    --header test/modular/fixtures/ministral3/model.safetensors.header.json \
    --tokenizer-eos-token-id 2
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mstar.model.ministral3.checkpoint import audit_checkpoint
from mstar.model.ministral3.components.language_model import Ministral3DenseLM
from mstar.model.ministral3.config import Ministral3TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--tokenizer-eos-token-id", required=True, type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON with every weight mapping and excluded vision key")
    args = parser.parse_args()
    config = Ministral3TextConfig.from_hf(args.metadata, tokenizer_eos_token_id=args.tokenizer_eos_token_id)
    with torch.device("meta"):
        model = Ministral3DenseLM(config, attn_key="attn", kv_key="kv_cache", pos_key="rope")
    quantization = json.loads((args.metadata / "config.json").read_text())["quantization_config"]
    audit = audit_checkpoint(json.loads(args.header.read_text()), model, quantization)
    if args.output:
        args.output.write_text(json.dumps(asdict(audit), indent=2) + "\n")
    print(json.dumps({
        "text_weights": len(audit.weights),
        "fp8_weights": sum(row.dtype == "F8_E4M3" for row in audit.weights),
        "weight_scales": sum(row.weight_scale is not None for row in audit.weights),
        "activation_scales": sum(row.activation_scale is not None for row in audit.weights),
        "native_parameters": len(audit.native_parameters),
        "excluded_vision_tensors": len(audit.excluded_vision),
        "aliases": audit.aliases,
    }, indent=2))


if __name__ == "__main__":
    main()
