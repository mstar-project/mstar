"""Text architecture and node configuration, without loading checkpoint weights."""

import json
from dataclasses import dataclass
from pathlib import Path

from mstar.engine.resources.position.config import YaRNConfig
from mstar.model.ministral3.components.language_model import Ministral3LMConfig

LLM_NODE = "LLM"
ATTN = "attn"


@dataclass(frozen=True, kw_only=True)
class Ministral3TextConfig(Ministral3LMConfig):
    """Architecture plus the text node's context bound and explicit stop IDs.

    No tokenizer defaults are guessed: ``from_hf`` combines the tokenizer's EOS
    with stop IDs declared in the model/generation configuration.
    """

    stop_token_ids: tuple[int, ...]
    max_position_embeddings: int = 262144

    def __post_init__(self):
        super().__post_init__()
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive")
        if not self.stop_token_ids or any(
            type(token) is not int or not 0 <= token < self.vocab_size
            for token in self.stop_token_ids
        ):
            raise ValueError("stop_token_ids must contain valid vocabulary IDs")

    @classmethod
    def from_hf(cls, directory: str | Path, *, tokenizer_eos_token_id: int) -> "Ministral3TextConfig":
        """Read the released multimodal checkpoint's text configuration only.

        This does not interpret FP8 scales or load any weights. Required geometry
        comes from the file, never from the 3B constructor defaults.
        """
        directory = Path(directory)
        raw = json.loads((directory / "config.json").read_text())
        text = raw["text_config"]
        if raw.get("model_type") != "mistral3" or text.get("model_type") != "ministral3":
            raise ValueError("Expected a mistral3 checkpoint with ministral3 text_config")
        if text.get("hidden_act") != "silu" or text.get("sliding_window") is not None:
            raise ValueError("Ministral 3 text requires SwiGLU and full-context attention")
        if any(text.get(key, False) for key in ("attention_bias", "mlp_bias", "qk_norm")):
            raise ValueError("Ministral 3 text does not implement attention/MLP biases or QK norms")
        if text.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("Ministral 3 text requires full-head rotary embeddings")
        rope = text["rope_parameters"]
        if rope.get("rope_type") != "yarn" or rope.get("type", "yarn") != "yarn":
            raise ValueError("Ministral 3 text requires YaRN")
        yarn = YaRNConfig(
            **{key: rope[key] for key in (
                "factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim",
            )},
            **{key: rope[key] for key in ("attention_factor", "truncate") if key in rope},
        )
        generation_path = directory / "generation_config.json"
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        # Chat EOS and generation EOS need not be the same token. Preserve both.
        if type(tokenizer_eos_token_id) is not int:
            raise ValueError("Ministral 3 requires a tokenizer EOS token ID")
        stops = {tokenizer_eos_token_id}
        for source in (raw, text, generation):
            eos = source.get("eos_token_id")
            if eos is not None:
                ids = eos if isinstance(eos, list) else [eos]
                if any(type(token) is not int for token in ids):
                    raise ValueError("Ministral 3 EOS token IDs must be integers")
                stops.update(ids)
        return cls(
            **{key: text[key] for key in (
                "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "tie_word_embeddings",
                "max_position_embeddings",
            )},
            rope_theta=rope["rope_theta"], yarn=yarn,
            llama_4_scaling_beta=rope["llama_4_scaling_beta"],
            stop_token_ids=tuple(sorted(stops)),
        )
