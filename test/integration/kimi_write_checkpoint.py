"""Write a random-weight Kimi checkpoint.

``--variant reduced`` (default, unchanged): the small checkpoint the
configs/synthetic/kimi_k2_7_*.yaml deployments load, bf16, no quantization.
Delegates to ``kimi_reference.write_checkpoint``.

``--variant k27_code``: a random-weight, real-dimension, INT4 pack-quantized
checkpoint in the exact on-disk format of moonshotai/Kimi-K2.7-Code (sharded
safetensors + index + config.json + tokenizer files), for exercising the real
Marlin W4A16 load path without the real 600 GB checkpoint. Streams one
decoder layer at a time so a full 61-layer/64-expert write never holds more
than one shard in memory. Output tokens are meaningless either way — this is
plumbing-only.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kimi_reference import write_checkpoint  # noqa: E402

from mstar.model.components.quantization import CompressedTensorsQuantConfig  # noqa: E402
from mstar.model.kimi_k2_7.components.language_model import is_moe_layer  # noqa: E402
from mstar.model.kimi_k2_7.config import KimiK2Config  # noqa: E402

# Reference checkpoint this variant mirrors: config.json, model.safetensors.index.json
# (tensor names only — no weight data), and the tokenizer/template files.
K27_REF = Path("/shared/home/garv901-55613a/kimik27/k27_ref")
TOKENIZER_FILES = (
    "tiktoken.model",
    "tokenizer_config.json",
    "tokenization_kimi.py",
    "tool_declaration_ts.py",
    "chat_template.jinja",
    "generation_config.json",
)

# Kimi-K2.7-Code's compressed-tensors format: INT4, group_size=32, symmetric.
# See k27_ref/config.json's text_config.quantization_config.
QUANT = CompressedTensorsQuantConfig(num_bits=4, group_size=32, symmetric=True)


def _bf16_randn(*shape, std=0.02):
    return (torch.randn(*shape) * std).to(torch.bfloat16)


def _ones_bf16(n):
    return torch.ones(n, dtype=torch.bfloat16)


def _quantize_random(out_features, in_features):
    """A random, well-formed INT4 pack-quantized weight: any int32 bit
    pattern is a valid packed nibble sequence (offset-binary unpack always
    lands in a finite range), so the packed values need no relation to the
    scales — only the shapes matter."""
    info = torch.iinfo(torch.int32)
    packed = torch.randint(
        info.min, info.max + 1,
        (out_features, in_features // QUANT.pack_factor), dtype=torch.int32,
    )
    scale = (0.005 + 0.01 * torch.rand(out_features, in_features // QUANT.group_size)).to(
        torch.bfloat16
    )
    shape = torch.tensor([out_features, in_features], dtype=torch.int64)
    return packed, scale, shape


def _attention_tensors(cfg, prefix):
    h = cfg.num_attention_heads
    return {
        prefix + "self_attn.q_a_proj.weight": _bf16_randn(cfg.q_lora_rank, cfg.hidden_size),
        prefix + "self_attn.q_a_layernorm.weight": _ones_bf16(cfg.q_lora_rank),
        prefix + "self_attn.q_b_proj.weight": _bf16_randn(h * cfg.qk_head_dim, cfg.q_lora_rank),
        prefix + "self_attn.kv_a_proj_with_mqa.weight": _bf16_randn(
            cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size
        ),
        prefix + "self_attn.kv_a_layernorm.weight": _ones_bf16(cfg.kv_lora_rank),
        prefix + "self_attn.kv_b_proj.weight": _bf16_randn(
            h * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank
        ),
        prefix + "self_attn.o_proj.weight": _bf16_randn(cfg.hidden_size, h * cfg.v_head_dim),
        prefix + "input_layernorm.weight": _ones_bf16(cfg.hidden_size),
        prefix + "post_attention_layernorm.weight": _ones_bf16(cfg.hidden_size),
    }


def _dense_mlp_tensors(cfg, prefix):
    return {
        prefix + "mlp.gate_proj.weight": _bf16_randn(cfg.intermediate_size, cfg.hidden_size),
        prefix + "mlp.up_proj.weight": _bf16_randn(cfg.intermediate_size, cfg.hidden_size),
        prefix + "mlp.down_proj.weight": _bf16_randn(cfg.hidden_size, cfg.intermediate_size),
    }


def _moe_mlp_tensors(cfg, prefix):
    shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    tensors = {
        prefix + "mlp.gate.weight": _bf16_randn(cfg.n_routed_experts, cfg.hidden_size),
        prefix + "mlp.gate.e_score_correction_bias": (
            torch.randn(cfg.n_routed_experts) * 0.02
        ).to(torch.float32),
        prefix + "mlp.shared_experts.gate_proj.weight": _bf16_randn(
            shared_inter, cfg.hidden_size
        ),
        prefix + "mlp.shared_experts.up_proj.weight": _bf16_randn(shared_inter, cfg.hidden_size),
        prefix + "mlp.shared_experts.down_proj.weight": _bf16_randn(
            cfg.hidden_size, shared_inter
        ),
    }
    for e in range(cfg.n_routed_experts):
        ep = prefix + f"mlp.experts.{e}."
        for proj, out_in in (
            ("gate_proj", (cfg.moe_intermediate_size, cfg.hidden_size)),
            ("up_proj", (cfg.moe_intermediate_size, cfg.hidden_size)),
            ("down_proj", (cfg.hidden_size, cfg.moe_intermediate_size)),
        ):
            packed, scale, shape = _quantize_random(*out_in)
            tensors[ep + proj + ".weight_packed"] = packed
            tensors[ep + proj + ".weight_scale"] = scale
            tensors[ep + proj + ".weight_shape"] = shape
    return tensors


def _layer_tensors(cfg, layer_idx):
    prefix = f"language_model.model.layers.{layer_idx}."
    tensors = _attention_tensors(cfg, prefix)
    if is_moe_layer(cfg, layer_idx):
        tensors.update(_moe_mlp_tensors(cfg, prefix))
    else:
        tensors.update(_dense_mlp_tensors(cfg, prefix))
    return tensors


def write_k27_code_checkpoint(out, cfg, shard_layers, seed):
    """Stream one decoder layer at a time into layer-count-bounded shards, so
    the full checkpoint (~130 GB at 61 layers/64 experts) is never
    materialized. Returns (tensor_count, total_size_bytes)."""
    torch.manual_seed(seed)
    num_layers = cfg.num_hidden_layers
    num_shards = math.ceil(num_layers / shard_layers)
    weight_map = {}
    total_size = 0
    tensor_count = 0

    for shard_idx in range(num_shards):
        first = shard_idx * shard_layers
        last = min(first + shard_layers, num_layers)
        shard_tensors = {}
        if shard_idx == 0:
            shard_tensors["language_model.model.embed_tokens.weight"] = _bf16_randn(
                cfg.vocab_size, cfg.hidden_size
            )
        for layer_idx in range(first, last):
            shard_tensors.update(_layer_tensors(cfg, layer_idx))
        if shard_idx == num_shards - 1:
            shard_tensors["language_model.model.norm.weight"] = _ones_bf16(cfg.hidden_size)
            shard_tensors["language_model.lm_head.weight"] = _bf16_randn(
                cfg.vocab_size, cfg.hidden_size
            )

        shard_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard_tensors, str(out / shard_name), metadata={"format": "pt"})
        for key, tensor in shard_tensors.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()
            tensor_count += 1

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index))
    return tensor_count, total_size


def write_k27_code_config(out, cfg, ref_config_path):
    """A copy of the reference config.json with only the layer/expert counts
    overridden; ``quantization_config``, ``vision_config``, etc. pass through
    unchanged."""
    with open(ref_config_path) as f:
        raw = json.load(f)
    raw["text_config"]["num_hidden_layers"] = cfg.num_hidden_layers
    raw["text_config"]["n_routed_experts"] = cfg.n_routed_experts
    (out / "config.json").write_text(json.dumps(raw, indent=2))


def copy_tokenizer_files(tokenizer_src, out):
    for name in TOKENIZER_FILES:
        src_file = Path(tokenizer_src) / name
        if src_file.is_file():
            shutil.copy2(src_file, out / name)
        else:
            print(f"kimi_write_checkpoint: {src_file} not found, skipping", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=("reduced", "k27_code"), default="reduced")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--shard-layers", type=int, default=1)
    parser.add_argument("--tokenizer-src", default=str(K27_REF))
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--moe-intermediate", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.variant == "reduced":
        write_checkpoint(out, KimiK2Config.reduced(), seed=args.seed)
        print(out / "model.safetensors")
        return

    cfg = KimiK2Config.k27_code()
    if args.layers is not None:
        cfg.num_hidden_layers = args.layers
    if args.experts is not None:
        cfg.n_routed_experts = args.experts
    if args.hidden is not None:
        cfg.hidden_size = args.hidden
    if args.moe_intermediate is not None:
        cfg.moe_intermediate_size = args.moe_intermediate
    if args.vocab is not None:
        cfg.vocab_size = args.vocab

    tensor_count, total_size = write_k27_code_checkpoint(
        out, cfg, args.shard_layers, args.seed
    )
    write_k27_code_config(out, cfg, Path(args.tokenizer_src) / "config.json")
    copy_tokenizer_files(args.tokenizer_src, out)

    print(out)
    print(f"tensors: {tensor_count}")
    print(f"total_size_bytes: {total_size}")


if __name__ == "__main__":
    main()
