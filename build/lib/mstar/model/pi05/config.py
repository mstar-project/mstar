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

    # ----- Shared attention dimensions (PaliGemma + action expert) -----
    # Both experts share num_kv_heads and head_dim so the action expert can
    # read PaliGemma's KV cache; only the per-expert hidden_size and MLP
    # dimensions differ.
    num_layers: int = 18
    num_qo_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # ----- PaliGemma expert (Gemma-2B by default) -----
    hidden_size: int = 2048  # paligemma hidden / "width"
    pali_intermediate_size: int = 16384
    vocab_size: int = 257152
    pad_token_id: int = 0

    # ----- Action expert (Gemma-300m by default to match lerobot/pi05_base) -----
    # The production Pi0.5 release uses gemma_300m (1024 hidden, 4096 mlp).
    # gemma_2b dimensions are also valid; override these to use that variant.
    action_hidden_size: int = 1024
    action_intermediate_size: int = 4096

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


# HF config keys whose names differ from the corresponding Pi05Config fields.
# Any key NOT listed here is auto-mapped if the HF key matches a Pi05Config
# field name exactly. This keeps the mapping maintainable: only the few
# exceptions need to be enumerated, rather than every field.
_HF_KEY_RENAMES: dict[str, str] = {
    "num_hidden_layers": "num_layers",
    "num_attention_heads": "num_qo_heads",
    "num_key_value_heads": "num_kv_heads",
    "intermediate_size": "pali_intermediate_size",
}

# Pi05Config field names — computed once from the dataclass.
_PI05_FIELDS: set[str] = {f.name for f in Pi05Config.__dataclass_fields__.values()}


def load_pi05_config(hf_config: dict | None = None) -> Pi05Config:
    """Build a Pi05Config, optionally overlaying values from an HF config dict.

    Auto-maps any HF key that matches a Pi05Config field name. For the few
    HF keys whose names differ (e.g. ``num_hidden_layers`` -> ``num_layers``),
    an explicit rename dict is used. Unrecognised keys are silently ignored.
    """
    cfg = Pi05Config()
    if not hf_config:
        return cfg

    for hf_key, value in hf_config.items():
        if not isinstance(value, (int, float, str, bool)):
            continue  # skip nested dicts / lists
        field_name = _HF_KEY_RENAMES.get(hf_key, hf_key)
        if field_name in _PI05_FIELDS:
            setattr(cfg, field_name, value)
    return cfg
