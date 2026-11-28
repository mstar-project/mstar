"""Fields of a Hugging Face ``config.json`` that more than one model reads."""
from __future__ import annotations


def hf_weight_dtype(hf: dict) -> str:
    """The weights' dtype a ``config.json`` names: ``dtype``, which transformers 5
    writes and keeps over ``torch_dtype`` when both are given, else ``torch_dtype``,
    which older checkpoints carry, else bfloat16. Two keys that disagree raise:
    which one the weights are in can't be told."""
    dtype, torch_dtype = hf.get("dtype"), hf.get("torch_dtype")
    if dtype and torch_dtype and str(dtype) != str(torch_dtype):
        raise ValueError(f"config.json names dtype {dtype!r} and torch_dtype {torch_dtype!r}")
    return str(dtype or torch_dtype or "bfloat16")
