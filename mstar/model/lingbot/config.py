from dataclasses import dataclass

from mstar.model.lingbot.default_negative_prompt import DEFAULT_NEGATIVE_PROMPT


@dataclass
class LingBotConfig:
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    hidden_size: int = 2048
    num_attention_heads: int = 16
    depth: int = 24
    intermediate_size: int = 6144
    text_dim: int = 2560
    freq_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 256.0
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    axes_lens: tuple[int, int, int] = (8192, 1024, 1024)
    qkv_bias: bool = False
    out_bias: bool = True
    patch_embed_bias: bool = True
    timestep_mlp_bias: bool = True
    num_experts: int = 0
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    n_shared_experts: int | None = None
    score_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int | None = None
    topk_group: int | None = None
    routed_scaling_factor: float = 1.0

    token_length: int = 37698
    hidden_state_skip_layer: int = 0
    vae_z_dim: int = 16
    vae_scale_factor_spatial: int = 8
    vae_scale_factor_temporal: int = 4

    default_height: int = 480
    default_width: int = 480
    default_num_frames: int = 81
    default_num_inference_steps: int = 40
    default_guidance_scale: float = 6.0
    default_shift: float = 3.0
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    video_fps: int = 24
    max_denoise_steps: int = 100

    @property
    def spatial_alignment(self) -> tuple[int, int]:
        return (
            self.vae_scale_factor_spatial * self.patch_size[1],
            self.vae_scale_factor_spatial * self.patch_size[2],
        )
