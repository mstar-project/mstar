import json
import math
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
        """Parse the text config and reject unsupported checkpoint mathematics."""
        if not isinstance(data, dict):
            raise ValueError("text_config must be a JSON object")

        expected = {
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
        }
        for name, supported in expected.items():
            if name not in data:
                raise ValueError(f"Missing required text_config field: {name}")
            value = data[name]
            if isinstance(supported, bool):
                matches = value is supported
            elif type(supported) is int:
                matches = type(value) is int and value == supported
            elif type(supported) is float:
                matches = type(value) in (int, float) and value == supported
            else:
                matches = value == supported
            if not matches:
                raise ValueError(f"{name} must be {supported!r}; got {value!r}")

        # This checkpoint carries both legacy and nested RoPE settings.
        # Accept the legacy-only form, but never silently ignore a conflict.
        if "rope_parameters" in data:
            rope = data["rope_parameters"]
            if not isinstance(rope, dict):
                raise ValueError("rope_parameters must be a JSON object")
            if rope.get("rope_type") != "default":
                raise ValueError("rope_parameters.rope_type must be 'default'")
            if "rope_theta" in rope:
                theta = rope["rope_theta"]
                if type(theta) not in (int, float) or theta != data["rope_theta"]:
                    raise ValueError(
                        "rope_parameters.rope_theta must match rope_theta"
                    )

        return cls(
            hidden_size=data["hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            vocab_size=data["vocab_size"],
            intermediate_size=data["intermediate_size"],
            num_experts=data["num_experts"],
            num_experts_per_tok=data["num_experts_per_tok"],
            num_shared_experts=data["num_shared_experts"],
            layer_types=data["layer_types"],
            sliding_window=data["sliding_window"],
            max_position_embeddings=data["max_position_embeddings"],
            rope_theta=data["rope_theta"],
            layer_norm_eps=data["layer_norm_eps"],
            logit_scale=data["logit_scale"],
            bos_token_id=data["bos_token_id"],
            eos_token_id=data["eos_token_id"],
            pad_token_id=data["pad_token_id"]
        )

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "num_shared_experts",
            "sliding_window",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")

        # Validate that the number of attention heads is divisible by the number of key-value heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # reject num_experts_per_tok greater than num_experts
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) cannot be greater than "
                f"num_experts ({self.num_experts})"
            )

        # even head dim
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be an even integer")

        for name in ("rope_theta", "layer_norm_eps", "logit_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number; got {value!r}")

        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError(
                    f"{name} must be an integer in [0, {self.vocab_size}); got {value!r}"
                )

        if not isinstance(self.layer_types, list) or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must be a list with one entry per hidden layer")
        for layer_idx, layer_type in enumerate(self.layer_types):
            expected = "full_attention" if layer_idx % 4 == 3 else "sliding_attention"
            if layer_type != expected:
                raise ValueError(
                    f"layer_types[{layer_idx}] must be {expected!r}; got {layer_type!r}"
                )


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
