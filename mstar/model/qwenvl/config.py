"""Official Transformers config loading for the supported Qwen3-VL checkpoint."""

from __future__ import annotations

from typing import Any


def validate_qwenvl_config(config: Any) -> None:
    """Reject architecture variants that the M* QwenVL graph does not implement."""
    if config.model_type != "qwen3_vl_moe":
        raise ValueError(f"QwenVL only supports Qwen3-VL MoE checkpoints; got model_type={config.model_type!r}.")

    text = config.text_config
    if text.head_dim <= 0 or text.head_dim % 2:
        raise ValueError(f"QwenVL head_dim must be a positive even number, got {text.head_dim}.")
    if text.num_attention_heads % text.num_key_value_heads:
        raise ValueError(
            "QwenVL num_attention_heads must be divisible by num_key_value_heads; "
            f"got {text.num_attention_heads} and {text.num_key_value_heads}."
        )
    if text.num_experts <= 0 or not 0 < text.num_experts_per_tok <= text.num_experts:
        raise ValueError(
            "QwenVL requires a valid sparse-MoE expert configuration; "
            f"got top-{text.num_experts_per_tok} of {text.num_experts}."
        )
    if text.decoder_sparse_step != 1 or text.mlp_only_layers:
        raise ValueError("QwenVL requires a sparse MoE block on every decoder layer.")

    rope_scaling = text.rope_scaling or {}
    section = tuple(rope_scaling.get("mrope_section", ()))
    if len(section) != 3 or sum(section) != text.head_dim // 2:
        raise ValueError(
            "QwenVL mrope_section must have three entries covering head_dim / 2; "
            f"got {section!r} for head_dim={text.head_dim}."
        )
    if not rope_scaling.get("mrope_interleaved"):
        raise ValueError("Qwen3-VL requires rope_scaling.mrope_interleaved=true.")

    vision = config.vision_config
    if vision.out_hidden_size != text.hidden_size:
        raise ValueError(
            "QwenVL vision output width must match text hidden size; "
            f"got {vision.out_hidden_size} and {text.hidden_size}."
        )
    if len(vision.deepstack_visual_indexes) != 3:
        raise ValueError(
            f"Qwen3-VL requires three DeepStack vision features; got indexes={vision.deepstack_visual_indexes!r}."
        )
    if config.tie_word_embeddings:
        raise ValueError("QwenVL supports the published checkpoint's untied output head only.")


def load_qwenvl_config(path: str):
    """Load the official HF config and validate the M* execution contract."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    validate_qwenvl_config(config)
    return config
