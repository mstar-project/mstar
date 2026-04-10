"""Configuration for the Pi0.5 vision-language-action model."""

from dataclasses import dataclass, field


@dataclass
class Pi05Config:
    """Pi0.5 model configuration.

    Pi0.5 combines a SigLIP vision encoder with two Gemma transformer experts:
    PaliGemma (Gemma-2B backbone) processes the prefix (image + text + state
    tokens) and writes a KV cache; an action expert (Gemma) reads that frozen
    cache and runs a 10-step Euler flow-matching loop with adaRMS timestep
    conditioning to produce a 50-step robot action trajectory.

    Both experts share KV-cache dimensions (num_kv_heads, head_dim) so that the
    action expert can attend to the cache written by PaliGemma.
    """

    # ----- SigLIP vision encoder (So400m/14) -----
    vit_hidden_size: int = 1152
    vit_num_layers: int = 27
    vit_num_heads: int = 16
    vit_intermediate_size: int = 4304
    vit_patch_size: int = 14
    vit_image_size: int = 224
    tokens_per_image: int = 256  # (224 / 14) ** 2
    num_cameras: int = 3  # base, left wrist, right wrist (typical)

    # ----- Shared transformer dimensions (PaliGemma + action expert) -----
    # Shared between the two experts so a single KV cache works for both.
    hidden_size: int = 2048
    num_layers: int = 18
    num_qo_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # ----- PaliGemma expert (Gemma-2B) -----
    pali_intermediate_size: int = 16384
    vocab_size: int = 257152
    pad_token_id: int = 0

    # ----- Action expert -----
    # Defaults to gemma_2b dimensions; gemma_300m users can override
    # action_intermediate_size to 4096.
    action_intermediate_size: int = 16384

    # ----- Flow matching -----
    num_flow_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 32
    state_dim: int = 32  # robot proprioceptive state dimension (padded to action_dim)
    state_token_bins: int = 256  # number of discretization bins for robot state
    state_token_offset: int = 0  # vocab offset where state-bin tokens start

    # ----- Tokenization / sequence limits -----
    max_lang_tokens: int = 200
    max_position_embeddings: int = 2048

    # ----- adaRMS conditioning -----
    # Sinusoidal timestep embedding range (matches openpi).
    timestep_min_period: float = 4e-3
    timestep_max_period: float = 4.0

    # ----- Default per-request sampling parameters -----
    # Pi0.5 has no stochastic sampling; included for API parity.
    default_action_dtype: str = "float32"
    extra: dict = field(default_factory=dict)


def load_pi05_config(hf_config: dict | None = None) -> Pi05Config:
    """Build a Pi05Config, optionally overlaying values from an HF config dict."""
    cfg = Pi05Config()
    if not hf_config:
        return cfg

    # Map HF config keys onto Pi05Config fields where they exist.
    overlay = {
        "hidden_size": hf_config.get("hidden_size"),
        "num_layers": hf_config.get("num_hidden_layers"),
        "num_qo_heads": hf_config.get("num_attention_heads"),
        "num_kv_heads": hf_config.get("num_key_value_heads"),
        "head_dim": hf_config.get("head_dim"),
        "pali_intermediate_size": hf_config.get("intermediate_size"),
        "vocab_size": hf_config.get("vocab_size"),
        "rms_norm_eps": hf_config.get("rms_norm_eps"),
        "rope_theta": hf_config.get("rope_theta"),
        "max_position_embeddings": hf_config.get("max_position_embeddings"),
    }
    for key, value in overlay.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg
