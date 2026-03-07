from dataclasses import dataclass
from typing import Any


@dataclass
class BagelAutoEncoderConfig:
    resolution: int = 256
    in_channels: int = 3
    downsample: int = 8
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BagelAutoEncoderConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class BagelViTConfig:
    # ViT
    hidden_size=768
    intermediate_size=3072
    num_hidden_layers=12
    num_attention_heads=12
    num_channels=3
    image_size=224
    patch_size=16
    hidden_act="gelu_pytorch_tanh"
    layer_norm_eps=1e-6
    attention_dropout=0.0

    def __post_init__(self):
        self.rope = False
        self.num_hidden_layers -= 1

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BagelModelConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class BagelModelConfig:
    vae_config: BagelAutoEncoderConfig
    vit_config: BagelViTConfig

    latent_patch_size: int = 2
    max_latent_size: int = 32
    num_timesteps: int = 50
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.5
    think_mode: bool = False
    vocab_size=151936
    hidden_size=4096
    intermediate_size=22016
    num_hidden_layers=32
    num_attention_heads=32
    num_key_value_heads=32
    hidden_act="silu"
    max_position_embeddings=32768
    initializer_range=0.02
    rms_norm_eps=1e-6
    use_cache=True
    rope_theta=10000.0
    rope_scaling=None
    use_sliding_window=False
    sliding_window=4096
    max_window_layers=28
    attention_dropout=0.0
    is_causal=True
    freeze_und=False
    connector_act="gelu_pytorch_tanh"
    vit_max_num_patch_per_side=70
    pad_token_id=1

    @classmethod
    def from_dict(
        cls, vae_config, vit_config,
        config_dict: dict[str, Any]
    ) -> "BagelModelConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(
            vae_config=vae_config,
            vit_config=vit_config,
            **filtered_dict
        )

    def __post_init__(self):
        self.latent_downsample = self.vae_config.downsample * self.latent_patch_size
        self.patch_latent_dim = self.latent_patch_size ** 2 \
            * self.vae_config.z_channels
        self.qk_norm = True
        self.tie_word_embeddings = False