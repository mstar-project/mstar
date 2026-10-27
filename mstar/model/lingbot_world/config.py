"""Configuration for LingBot-World-V2 causal-fast (camera-controlled world model).

The architecture values are facts of the ``robbyant/lingbot-world-v2-*-causal-fast``
checkpoints (the reference repo's ``wan/configs/wan_i2v_*.py``), hardcoded so that
constructing the model never touches the network. The DiT is a Wan2.1-layout I2V
transformer (16-channel Wan2.1 VAE latents + 4 mask + 16 image-latent channels in)
with per-block Plucker camera injection, run block-causally over chunks of latent
frames against a rolling self-attention KV cache with attention-sink frames.

The sampling knobs are the reference's ``generate.py`` / ``run_fast.sh`` defaults:
4 distilled steps per chunk at fixed indices into a 1000-step shifted schedule,
re-noising between steps, then one t=0 pass over the clean chunk to commit its K/V.
"""

from dataclasses import dataclass

LINGBOT_VARIANT_1_3B = "1.3b_causal_fast"

LINGBOT_HF_REPOS = {
    LINGBOT_VARIANT_1_3B: "robbyant/lingbot-world-v2-1.3b-causal-fast",
}

# The 1.3B repo ships the DiT only; the reference takes T5, the VAE and the
# tokenizer from the 14B repo (``--assets_dir``). These are diffusers-format
# copies, which mstar already loads for other models; the parity harness
# checks they equal the LingBot assets.
WAN21_VAE_DIFFUSERS_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
UMT5_DIFFUSERS_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


@dataclass
class LingBotWorldConfig:
    variant: str = LINGBOT_VARIANT_1_3B

    # DiT (reference wan_i2v_1_3B.py)
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    freq_dim: int = 256
    text_dim: int = 4096
    in_dim: int = 36
    out_dim: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)
    eps: float = 1e-6
    text_len: int = 512
    # Plucker rays: origin + direction per pixel, folded 8x8 into each latent cell.
    control_dim: int = 6
    # RoPE tables cover this many latent frames; the rollout cannot run past it.
    rope_max_frames: int = 1024

    # Wan2.1 VAE
    vae_z_dim: int = 16
    vae_stride: tuple[int, int, int] = (4, 8, 8)

    # Causal-fast sampling (reference generate.py / run_fast.sh)
    chunk_size: int = 4  # latent frames per chunk
    local_attn_size: int = 18  # latent frames of self-attention K/V kept, sink included
    sink_size: int = 6  # leading latent frames never evicted
    timesteps_index: tuple[int, ...] = (0, 250, 500, 750)
    num_train_timesteps: int = 1000
    sample_shift: float = 5.0

    # Request geometry. The output aspect follows the conditioning image and the
    # area is height*width, as in the reference; intrinsics are given at 480x832.
    default_height: int = 480
    default_width: int = 832
    max_area: int = 720 * 1280  # largest height*width a request may ask for
    intrinsics_height: int = 480
    intrinsics_width: int = 832
    video_fps: int = 16
    max_chunks: int = 255  # rollout ceiling, inside rope_max_frames

    # The reference decodes in fp32. bf16 is faster on cuDNN >= 9.16 but is not
    # the reference's numerics.
    vae_decode_fp32: bool = True
    vae_decode_channels_last: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


def lingbot_world_1_3b() -> LingBotWorldConfig:
    return LingBotWorldConfig(variant=LINGBOT_VARIANT_1_3B)


LINGBOT_CONFIGS = {
    LINGBOT_VARIANT_1_3B: lingbot_world_1_3b,
}
