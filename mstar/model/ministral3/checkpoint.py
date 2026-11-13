"""Released HF checkpoint schema, header audit, and strict BF16 vision loading.

The header audit reads no tensor values. Vision loading copies BF16 values
through the shared strict loader; text FP8 loading lives in fp8.py.
The alternate consolidated.safetensors format is deliberately not accepted.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from mstar.model.loader.base import LLAMA_STACKED_PARAMS, CheckpointRules, apply_stacked, load_weights_into
from mstar.model.ministral3.components.language_model import Ministral3DenseLM, Ministral3LMConfig
from mstar.model.ministral3.components.vision import Ministral3Vision, PixtralConfig

HF_FILENAME = "model.safetensors"
TEXT_RULES = CheckpointRules(prefix="language_model.model.", stacked_params=tuple(LLAMA_STACKED_PARAMS))
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


@dataclass(frozen=True)
class WeightMapping:
    source: str
    target: str
    shard: str | int | None
    shape: tuple[int, ...]
    dtype: str
    weight_scale: str | None = None
    activation_scale: str | None = None


@dataclass(frozen=True)
class CheckpointAudit:
    weights: tuple[WeightMapping, ...]
    excluded_vision: tuple[str, ...]
    native_parameters: tuple[str, ...]
    # No separate LM head tensor is present in this tied checkpoint.
    aliases: dict[str, str]


def expected_text_weights(config: Ministral3LMConfig) -> dict[str, tuple[tuple[int, ...], str]]:
    h, f = config.hidden_size, config.intermediate_size
    q, kv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
    expected = {"embed_tokens.weight": ((config.vocab_size, h), "BF16"), "norm.weight": ((h,), "BF16")}
    per_layer = {
        "input_layernorm.weight": ((h,), "BF16"),
        "post_attention_layernorm.weight": ((h,), "BF16"),
        "self_attn.q_proj.weight": ((q, h), "F8_E4M3"),
        "self_attn.k_proj.weight": ((kv, h), "F8_E4M3"),
        "self_attn.v_proj.weight": ((kv, h), "F8_E4M3"),
        "self_attn.o_proj.weight": ((h, q), "F8_E4M3"),
        "mlp.gate_proj.weight": ((f, h), "F8_E4M3"),
        "mlp.up_proj.weight": ((f, h), "F8_E4M3"),
        "mlp.down_proj.weight": ((h, f), "F8_E4M3"),
    }
    for i in range(config.num_hidden_layers):
        expected.update({f"layers.{i}.{name}": spec for name, spec in per_layer.items()})
    return {TEXT_RULES.prefix + name: spec for name, spec in expected.items()}


def validate_quantization(quantization: Mapping[str, Any]) -> None:
    """Require the released static scalar FP8 format without a fallback cast."""
    if (
        quantization.get("quant_method") != "fp8"
        or quantization.get("activation_scheme") != "static"
        or "weight_block_size" not in quantization
        or quantization["weight_block_size"] is not None
        or quantization.get("dequantize", False)
    ):
        raise ValueError("Expected static per-tensor FP8 with weight_block_size=null and no dequantization")


def audit_checkpoint(
    header: Mapping[str, Any], model: Ministral3DenseLM, quantization: Mapping[str, Any],
) -> CheckpointAudit:
    """Require the known tied, static per-tensor FP8 layout; fail on any drift.

    ``model`` must be an unsharded meta decoder. Meta tensors check projection
    shapes and fused-shard routing without reading weight values. This does not
    validate FP8 storage, scale loading, or quantized execution in native layers.
    """
    if any(not parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Checkpoint audit requires a meta-only native decoder")
    if tuple(model.embed_tokens.weight.shape) != (model.config.vocab_size, model.config.hidden_size):
        raise ValueError("Checkpoint audit requires the full, unsharded embedding shape")
    if not model.config.tie_word_embeddings or model.lm_head.weight is not model.embed_tokens.weight:
        raise ValueError("The released checkpoint requires tied embedding/LM-head parameters")
    validate_quantization(quantization)

    expected = expected_text_weights(model.config)
    stacked = list(TEXT_RULES.stacked_params)
    mappings = []
    consumed = set()

    def check(key: str, shape: tuple[int, ...], dtype: str):
        if key not in header:
            raise ValueError(f"Missing checkpoint tensor: {key}")
        entry = header[key]
        if entry.get("shape") != list(shape) or entry.get("dtype") != dtype:
            raise ValueError(f"{key}: expected {dtype} shape {list(shape)}, got {entry}")
        consumed.add(key)

    for source, (shape, dtype) in expected.items():
        check(source, shape, dtype)
        weight_scale = activation_scale = None
        if dtype == "F8_E4M3":
            stem = source.removesuffix(".weight")
            weight_scale, activation_scale = stem + ".weight_scale_inv", stem + ".activation_scale"
            check(weight_scale, (), "BF16")
            check(activation_scale, (), "BF16")
        target, shard = apply_stacked(TEXT_RULES.native(source), stacked)
        mappings.append(WeightMapping(source, target, shard, shape, dtype, weight_scale, activation_scale))

    excluded = tuple(sorted(key for key in header if key.startswith(VISION_PREFIXES)))
    unknown = set(header) - consumed - set(excluded) - {"__metadata__"}
    if unknown:
        raise ValueError(f"Unclassified checkpoint tensors: {sorted(unknown)}")

    # Exercise the actual native shard loaders, not just a parallel naming table.
    # All tensors and parameters are meta: there is no weight storage or copy of
    # real data, and scalar scales are NOT silently cast into native parameters.
    weights = ((row.source, torch.empty(row.shape, device="meta", dtype=torch.bfloat16)) for row in mappings)
    loaded = load_weights_into(
        model, weights, name_remapper=TEXT_RULES.native, stacked_params=stacked, strict=True,
    )
    return CheckpointAudit(
        tuple(mappings), excluded, tuple(sorted(loaded)), {"lm_head.weight": "embed_tokens.weight"},
    )


def text_tensor_schema(config: Ministral3LMConfig) -> dict[str, tuple[tuple[int, ...], str]]:
    """All text tensors, including each quantized projection's two scales."""
    expected = expected_text_weights(config)
    for name, (_, dtype) in list(expected.items()):
        if dtype == "F8_E4M3":
            stem = name.removesuffix(".weight")
            expected[stem + ".weight_scale_inv"] = ((), "BF16")
            expected[stem + ".activation_scale"] = ((), "BF16")
    return expected


def checkpoint_node_keys(
    path: str | Path, text_config: Ministral3LMConfig, vision_config: PixtralConfig,
) -> tuple[set[str], set[str]]:
    """Validate the whole HF file before selecting a node's payloads for I/O.

    Reading only known keys without this check would silently hide unexpected
    parameters. safe_open's slices expose header shapes/dtypes without reading
    tensor payloads. Every text and vision entry must be present and classified.
    """
    text = text_tensor_schema(text_config)
    with torch.device("meta"):
        vision = {name: (tuple(p.shape), "BF16")
                  for name, p in Ministral3Vision(vision_config).named_parameters()}
    expected = text | vision
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        if missing := expected.keys() - keys:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        if unknown := keys - expected.keys():
            raise ValueError(f"Unclassified checkpoint tensors: {sorted(unknown)}")
        for name, (shape, dtype) in expected.items():
            entry = checkpoint.get_slice(name)
            if tuple(entry.get_shape()) != shape or entry.get_dtype() != dtype:
                raise ValueError(f"{name}: expected {dtype} shape {shape}, "
                                 f"got {entry.get_dtype()} {entry.get_shape()}")
    return set(text), set(vision)


def load_vision_weights(
    model: Ministral3Vision, weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load the tower and projector with exact BF16 values and full coverage."""
    params = dict(model.named_parameters())
    if any(p.is_meta or p.dtype != torch.bfloat16 for p in params.values()):
        raise ValueError("Materialize vision parameters in BF16 before loading weights")
    seen = set()

    def checked():
        for name, tensor in weights:
            if name in seen:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            seen.add(name)
            if name not in params:
                raise ValueError(f"Unclassified vision tensor: {name}")
            if tensor.is_meta or tensor.dtype != torch.bfloat16 or tensor.shape != params[name].shape:
                raise ValueError(f"{name}: expected BF16 shape {tuple(params[name].shape)}")
            yield name, tensor

    return load_weights_into(model, checked(), strict=True)
