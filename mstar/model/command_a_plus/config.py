import json
from dataclasses import dataclass
from pathlib import Path

# Resource names shared by the text components and the LLM node.
KV_CACHE = "kv_cache"
LOCAL_ATTN = "local_attn"
GLOBAL_ATTN = "global_attn"
ROPE = "rope"
SAMPLER = "sampler"

PREFILL_WALKS = ("prefill", "prefill_image")


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


@dataclass(frozen=True)
class CommandAPlusVisionConfig:
    """The SigLIP tower, pixel shuffle and SwiGLU projector that feed image tiles to the LLM."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    downsample_factor: int
    alignment_intermediate_size: int
    image_token_id: int

    @property
    def tokens_per_tile(self) -> int:
        return (self.image_size // self.patch_size // self.downsample_factor) ** 2

    @classmethod
    def from_dict(cls, data: dict) -> "CommandAPlusVisionConfig":
        """``data`` is the whole checkpoint config; the tower's own fields sit in ``vision_config``."""
        vision = data["vision_config"]
        # Shapes are checked when weights load; these would load fine and compute the wrong thing.
        for source, name, supported in (
            (vision, "hidden_act", "gelu_pytorch_tanh"),
            (data, "alignment_activation_fn", "swiglu"),
            (data, "vision_feature_layer", -1),
            (data, "vision_feature_select_strategy", "full"),
        ):
            if source.get(name) != supported:
                raise ValueError(f"{name} must be {supported!r}; got {source.get(name)!r}")
        return cls(
            **{name: vision[name] for name in (
                "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "image_size", "patch_size", "layer_norm_eps",
            )},
            downsample_factor=data["downsample_factor"],
            alignment_intermediate_size=data["alignment_intermediate_size"],
            image_token_id=data["image_token_id"],
        )


@dataclass
class CommandAPlusConfig:
    """The outer Cohere2 Vision checkpoint configuration."""

    text_config: CommandAPlusTextConfig
    vision_config: CommandAPlusVisionConfig

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
        return cls(
            text_config=CommandAPlusTextConfig.from_dict(data["text_config"]),
            vision_config=CommandAPlusVisionConfig.from_dict(data),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CommandAPlusConfig":
        """Load a local config.json without resolving or downloading weights."""
        with Path(path).open(encoding="utf-8") as config_file:
            return cls.from_dict(json.load(config_file))
