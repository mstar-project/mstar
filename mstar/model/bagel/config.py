from dataclasses import dataclass
from typing import Any


@dataclass
class BagelAutoEncoderConfig:
    resolution: int = 256
    in_channels: int = 3
    downsample: int = 8
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BagelAutoEncoderConfig":
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class BagelViTConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    # added in post init
    rope: bool = False

    def __post_init__(self) -> None:
        self.rope = False
        self.num_hidden_layers -= 1

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BagelViTConfig":
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


@dataclass
class BagelModelConfig:
    vae_config: BagelAutoEncoderConfig
    vit_config: BagelViTConfig

    latent_patch_size: int = 2
    max_latent_size: int = 64
    num_timesteps: int = 50
    timestep_shift: float = 3.0
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple[float, float] = (0.4, 1.0)
    cfg_renorm_type: str = "global"
    cfg_renorm_min: float = 0.0
    think_mode: bool = False

    # Sampling defaults (per-request overridable via model_kwargs)
    temperature: float = 0.6  # 0 = greedy (argmax), >0 = sampling
    top_k: int = 0            # 0 = disabled
    top_p: float = 1.0        # 1.0 = disabled
    repetition_penalty: float = 1.05
    ignore_eos: bool = False  # benchmark parity: decode to max_tokens regardless of EOS

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32

    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True

    rope_theta: float = 10000.0
    rope_scaling: Any | None = None

    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    is_causal: bool = True

    freeze_und: bool = False
    connector_act: str = "gelu_pytorch_tanh"
    vit_max_num_patch_per_side: int = 70
    pad_token_id: int = 1

    # computed fields
    latent_downsample: int | None = None
    patch_latent_dim: int | None = None
    qk_norm: bool = True
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(
        cls,
        vae_config: BagelAutoEncoderConfig,
        vit_config: BagelViTConfig,
        config_dict: dict[str, Any],
    ) -> "BagelModelConfig":
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(
            vae_config=vae_config,
            vit_config=vit_config,
            **filtered_dict,
        )

    def __post_init__(self) -> None:
        self.max_latent_size = 64
        self.timestep_shift = 3.0
        self.latent_downsample = self.vae_config.downsample * self.latent_patch_size
        self.patch_latent_dim = (
            self.latent_patch_size ** 2 * self.vae_config.z_channels
        )
        self.qk_norm = True
        self.tie_word_embeddings = False
        self.cfg_text_scale = 4.0
        self.cfg_img_scale = 1.0
        self.cfg_interval = (0.4, 1.0)


def load_bagel_config(config_hf: dict) -> BagelModelConfig:
    # --- Sub configs ---
    vae_config = BagelAutoEncoderConfig.from_dict(config_hf.get("vae_config", {}))
    vit_config = BagelViTConfig.from_dict(config_hf.get("vit_config", {}))

    # --- Flatten LLM config ---
    llm_config = config_hf.get("llm_config", {})

    # --- Collect remaining top-level fields ---
    excluded_keys = {"vae_config", "vit_config", "llm_config"}

    top_level_fields = {
        k: v for k, v in config_hf.items() if k not in excluded_keys
    }

    # Merge (llm_config takes precedence if duplicate keys exist)
    model_config_dict = {**top_level_fields, **llm_config}

    # --- Instantiate model config ---
    model_config = BagelModelConfig.from_dict(
        vae_config=vae_config,
        vit_config=vit_config,
        config_dict=model_config_dict,
    )

    return model_config
