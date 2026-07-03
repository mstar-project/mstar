"""V-JEPA 2 configuration dataclasses.

Flat config matching HuggingFace `VJEPA2Config` plus an optional action-conditioned
predictor block. A single `VJepa2Config.from_hf_config(dict)` handles every
open V-JEPA 2 checkpoint (vitl/h/g at 256, vitg at 384, and the AC variant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VJepa2ACPredictorConfig:
    """Configuration for the action-conditioned predictor (upstream V-JEPA 2-AC).

    Mirrors ``vjepa2/src/models/ac_predictor.py`` defaults.
    """

    img_size: tuple[int, int] = (256, 256)
    patch_size: int = 16
    num_frames: int = 64
    tubelet_size: int = 2
    embed_dim: int = 1408
    predictor_embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    layer_norm_eps: float = 1e-6
    is_frame_causal: bool = True
    use_rope: bool = True
    action_embed_dim: int = 7
    use_extrinsics: bool = False


@dataclass
class VJepa2Config:
    """Top-level V-JEPA 2 config (shared across the HF and upstream ports).

    Field names match HF ``VJEPA2Config`` so a plain JSON round-trip works.
    """

    # Encoder
    patch_size: int = 16
    crop_size: int = 256
    frames_per_clip: int = 64
    tubelet_size: int = 2
    hidden_size: int = 1024
    in_chans: int = 3
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    drop_path_rate: float = 0.0
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_probs_dropout_prob: float = 0.0
    hidden_act: str = "gelu"

    # Masked predictor
    pred_hidden_size: int = 384
    pred_num_attention_heads: int = 12
    pred_num_hidden_layers: int = 12
    pred_num_mask_tokens: int = 10
    pred_zero_init_mask_tokens: bool = True
    pred_mlp_ratio: float = 4.0

    # Which predictor to instantiate.  "masked" = HF VJEPA2Predictor;
    # "ac" = upstream VisionTransformerPredictorAC.
    predictor_kind: str = "masked"
    ac_predictor: VJepa2ACPredictorConfig | None = None

    # Rollout.  ``max_rollout_horizon`` caps the Loop's
    # ``max_iters`` at graph-build time; per-request ``rollout_horizon`` is
    # enforced by early-exit (``check_stop``) from inside the
    # rollout submodule.
    max_rollout_horizon: int = 16
    # AnticipativeWrapper-parity rollout geometry.  Per-request tuning is a
    # later extension — for now these are set once from config and baked
    # into the cached ``rollout_predictor`` submodule.
    rollout_num_output_frames: int = 2
    rollout_frames_per_second: int = 4
    rollout_anticipation_seconds: float = 1.0

    # MPC.  Cost function used by VJepa2MPCScorerSubmodule to
    # score K candidate predicted latents against the goal latent.  "l1"
    # matches upstream vjepa2/notebooks/utils/mpc_utils.py::l1 (the CEM
    # objective).  Fixed at deploy time rather than per-request so that
    # reproducibility / parity tests are deterministic across clients.
    mpc_cost_fn: str = "l1"

    @classmethod
    def from_hf_config(cls, config_dict: dict[str, Any]) -> "VJepa2Config":
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_keys}
        return cls(**filtered)

    @property
    def grid_size(self) -> int:
        return self.crop_size // self.patch_size

    @property
    def grid_depth(self) -> int:
        return self.frames_per_clip // self.tubelet_size

    @property
    def num_patches(self) -> int:
        return self.grid_depth * self.grid_size * self.grid_size
