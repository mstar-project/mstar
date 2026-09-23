"""The DSpark draft's configuration (``ckpt/Kimi-K3-DSpark/config.json``, architecture K3DSparkModel)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class YarnParams:
    factor: float = 32.0
    original_max_position_embeddings: int = 32768
    rope_theta: float = 50000.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0


@dataclass(frozen=True)
class DSparkConfig:
    hidden_size: int = 7168
    intermediate_size: int = 14336
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    target_hidden_size: int = 7168
    target_layer_ids: tuple[int, ...] = (2, 23, 47, 71, 89)
    mask_token_id: int = 163837
    markov_rank: int = 256
    rope: YarnParams = field(default_factory=YarnParams)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def context_width(self) -> int:
        return self.target_hidden_size * len(self.target_layer_ids)

    @classmethod
    def from_dir(cls, path: str | Path) -> "DSparkConfig":
        raw = json.loads((Path(path) / "config.json").read_text())
        if raw.get("architectures") != ["K3DSparkModel"]:
            raise ValueError(f"not a K3DSparkModel checkpoint: {raw.get('architectures')}")
        if raw.get("mla_use_nope") or raw.get("mla_use_output_gate"):
            raise ValueError("this draft is built for the rope MLA without output gate")
        rp = raw.get("rope_parameters") or {}
        if rp.get("rope_type", "yarn") != "yarn":
            raise ValueError(f"expected yarn rope parameters, got {rp}")
        rope = YarnParams(
            factor=float(rp.get("factor", 32.0)),
            original_max_position_embeddings=int(rp.get("original_max_position_embeddings", 32768)),
            rope_theta=float(rp.get("rope_theta", raw.get("rope_theta", 50000.0))),
            beta_fast=float(rp.get("beta_fast", 32)), beta_slow=float(rp.get("beta_slow", 1)),
            mscale=float(rp.get("mscale", 1.0)), mscale_all_dim=float(rp.get("mscale_all_dim", 1.0)),
        )
        return cls(
            hidden_size=raw["hidden_size"], intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"], num_attention_heads=raw["num_attention_heads"],
            q_lora_rank=raw["q_lora_rank"], kv_lora_rank=raw["kv_lora_rank"],
            qk_nope_head_dim=raw["qk_nope_head_dim"], qk_rope_head_dim=raw["qk_rope_head_dim"],
            v_head_dim=raw["v_head_dim"], vocab_size=raw["vocab_size"], rms_norm_eps=raw["rms_norm_eps"],
            target_hidden_size=raw["target_hidden_size"], target_layer_ids=tuple(raw["target_layer_ids"]),
            mask_token_id=raw["mask_token_id"], markov_rank=raw["markov_rank"], rope=rope,
        )
