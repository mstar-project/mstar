"""Build and strictly load the released scalar/static FP8 text representation.

The native decoder is first built on meta, so replacing its dense projection
slots allocates no discarded weight storage. Each FP8 projection keeps its own
scales; embeddings and norms remain BF16. This builder is unsharded. It does not
resolve downloads, allocate engine resources, or implement a BF16 fallback.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mstar.model.components.scalar_fp8_linear import FP8_DTYPE, ScalarFp8Linear, ScalarFp8MergedLinear
from mstar.model.loader.base import CheckpointRules, load_weights_into
from mstar.model.ministral3.checkpoint import VISION_PREFIXES, text_tensor_schema, validate_quantization
from mstar.model.ministral3.components.language_model import Ministral3DenseLM, Ministral3LMConfig


def _projection_name(name: str) -> str:
    for projection in ("q", "k", "v"):
        name = name.replace(f".self_attn.{projection}_proj.",
                            f".self_attn.qkv_proj.projections.{projection}.")
    for projection in ("gate", "up"):
        name = name.replace(f".mlp.{projection}_proj.", f".mlp.gate_up_proj.projections.{projection}.")
    return name


FP8_TEXT_RULES = CheckpointRules(prefix="language_model.model.", remap=_projection_name)


@dataclass(frozen=True)
class TextWeightLoad:
    loaded: frozenset[str]
    excluded_vision: tuple[str, ...]


def build_fp8_text_model(
    config: Ministral3LMConfig,
    quantization: Mapping[str, Any],
    *,
    device: str | torch.device,
    attn_key: str,
    kv_key: str,
    pos_key: str,
) -> Ministral3DenseLM:
    """Allocate the unsharded text stack in its checkpoint storage representation.

    The returned parameters are uninitialized; ``load_fp8_text_weights`` must
    succeed before execution. ``device='meta'`` supports name/shape inspection.
    """
    validate_quantization(quantization)
    if not config.tie_word_embeddings:
        raise ValueError("The released Ministral 3 checkpoint requires tied embeddings")
    h, f = config.hidden_size, config.intermediate_size
    q, kv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
    with torch.device("meta"):
        model = Ministral3DenseLM(config, attn_key=attn_key, kv_key=kv_key, pos_key=pos_key)
        for layer in model.layers:
            layer.self_attn.qkv_proj = ScalarFp8MergedLinear(h, {"q": q, "k": kv, "v": kv})
            layer.self_attn.o_proj = ScalarFp8Linear(q, h)
            layer.mlp.gate_up_proj = ScalarFp8MergedLinear(h, {"gate": f, "up": f})
            layer.mlp.down_proj = ScalarFp8Linear(f, h)
        model.to(dtype=torch.bfloat16)
    # Materialize the shared embedding once: _apply otherwise visits its alias
    # in lm_head too and allocates a second vocabulary-sized tensor.
    for name, child in model.named_children():
        if name != "lm_head":
            child.to_empty(device=device)
    model.lm_head.weight = model.embed_tokens.weight
    return model.eval()


def load_fp8_text_weights(
    model: Ministral3DenseLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    quantization: Mapping[str, Any],
) -> TextWeightLoad:
    """Validate the released tensor schema and dispatch through the shared loader.

    Full multimodal streams are accepted: vision keys are explicitly reported
    for a separate loader. Unknown, duplicate, missing, or malformed text tensors
    fail. The source is consumed once, without buffering checkpoint tensors.
    """
    validate_quantization(quantization)
    if model.lm_head.weight is not model.embed_tokens.weight or not model.config.tie_word_embeddings:
        raise ValueError("The released Ministral 3 checkpoint requires tied embeddings")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("Materialize the FP8 text model before loading weights")
    for layer in model.layers:
        if not isinstance(layer.self_attn.qkv_proj, ScalarFp8MergedLinear):
            raise TypeError("Use build_fp8_text_model before loading the FP8 checkpoint")
    expected = text_tensor_schema(model.config)
    excluded, seen = [], set()

    def checked():
        for name, tensor in weights:
            if name in seen:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            seen.add(name)
            if name.startswith(VISION_PREFIXES):
                excluded.append(name)
                continue
            if name not in expected:
                raise ValueError(f"Unclassified checkpoint tensor: {name}")
            shape, dtype = expected[name]
            wanted_dtype = FP8_DTYPE if dtype == "F8_E4M3" else torch.bfloat16
            if tuple(tensor.shape) != shape or tensor.dtype != wanted_dtype:
                raise ValueError(f"{name}: expected {dtype} shape {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
            if tensor.is_meta:
                raise ValueError(f"Checkpoint tensor has no values: {name}")
            yield name, tensor

    loaded = load_weights_into(model, checked(), name_remapper=FP8_TEXT_RULES.native, strict=True)
    return TextWeightLoad(frozenset(loaded), tuple(sorted(excluded)))
