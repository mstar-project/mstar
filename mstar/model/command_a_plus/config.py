import json
from dataclasses import dataclass
from pathlib import Path

# Resource names shared by the text components and the LLM node.
KV_CACHE = "kv_cache"
LOCAL_ATTN = "local_attn"
GLOBAL_ATTN = "global_attn"
ROPE = "rope"
SAMPLER = "sampler"


@dataclass
class CommandAPlusTextConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    # MoE
    intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int

    layer_types: list[str]
    sliding_window: int
    max_position_embeddings: int
    rope_theta: float
    layer_norm_eps: float
    logit_scale: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


    @property
    def shared_intermediate_size(self) -> int:
        """Width of the shared MLP, derived from the expert dimensions."""
        return self.intermediate_size * self.num_shared_experts


    @classmethod
    def from_dict(cls, data: dict) -> "CommandAPlusTextConfig":
        """Parse the text config; shapes are checked when weights load, so only the math is pinned here."""
        if not isinstance(data, dict):
            raise ValueError("text_config must be a JSON object")
        for name, supported in {
            "model_type": "cohere2_moe",
            "hidden_act": "silu",
            "expert_selection_fn": "sigmoid",
            "norm_topk_prob": True,
            "shared_expert_combination_strategy": "average",
            "use_parallel_block": True,
            "tie_word_embeddings": True,
            "attention_bias": False,
            "use_qk_norm": False,
            "position_embedding_type": "rope_gptj",
            "rotary_pct": 1.0,
            "first_k_dense_replace": 0,
            "rms_norm_eps": None,
        }.items():
            if data.get(name) != supported:
                raise ValueError(f"{name} must be {supported!r}; got {data.get(name)!r}")
        rope = data.get("rope_parameters") or {}
        theta = data["rope_theta"]
        if (rope.get("rope_type", "default"), rope.get("rope_theta", theta)) != ("default", theta):
            raise ValueError("rope_parameters must be default RoPE with the same rope_theta")
        if not set(data["layer_types"]) <= {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unsupported layer_types: {sorted(set(data['layer_types']))}")
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass
class CommandAPlusConfig:
    """Text-only view of the outer Cohere2 Vision checkpoint configuration."""

    text_config: CommandAPlusTextConfig

    @classmethod
    def from_dict(cls, data: dict) -> "CommandAPlusConfig":
        if not isinstance(data, dict):
            raise ValueError("Checkpoint config must be a JSON object")
        if data.get("model_type") != "cohere2_vision":
            raise ValueError("Outer model_type must be 'cohere2_vision'")
        if data.get("tie_word_embeddings") is not True:
            raise ValueError("Outer tie_word_embeddings must be True")
        if "text_config" not in data:
            raise ValueError("Missing required text_config")
        return cls(text_config=CommandAPlusTextConfig.from_dict(data["text_config"]))

    @classmethod
    def from_json(cls, path: str | Path) -> "CommandAPlusConfig":
        """Load a local config.json without resolving or downloading weights."""
        with Path(path).open(encoding="utf-8") as config_file:
            return cls.from_dict(json.load(config_file))
