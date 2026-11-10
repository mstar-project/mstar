"""Qwen-Image-2.1's configuration, read from the checkpoint's component configs."""

from __future__ import annotations

from dataclasses import dataclass


def _fields(cls, cfg: dict) -> dict:
    names = cls.__dataclass_fields__
    return {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names}


@dataclass(frozen=True)
class QwenImage21TransformerConfig:
    """``transformer/config.json`` (``QwenImage21Transformer2DModel``)."""

    patch_size: int = 1
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> "QwenImage21TransformerConfig":
        out = _fields(cls, cfg)
        if out.get("out_channels") is None:
            out["out_channels"] = out.get("in_channels", cls.in_channels)
        return cls(**out)
