"""Configuration for Wan2.2-TI2V-5B (dense video DiT + UMT5-XXL + Wan2.2-VAE).

The architecture values are facts of the ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``
checkpoint, hardcoded so that constructing the model never touches the network.
``Wan22Model._refresh_checkpoint_defaults`` re-reads ``flow_shift`` from the
checkpoint at load time and hard-fails if the solver order or train-timestep
count drift from what the inline UniPC port implements.
"""

from dataclasses import dataclass

# The only variant implemented. ``Wan22Model.__init__`` rejects any other; the
# A14B variants are a MoE dual-DiT and a separate piece of work.
WAN22_VARIANT_TI2V_5B = "ti2v_5b"


@dataclass
class Wan22Config:
    """Wan2.2-TI2V-5B model configuration.

    A single dense video DiT driven by UMT5-XXL embeddings, denoising Wan2.2-VAE
    latents with UniPC. T2V and I2V share the one transformer: I2V injects the
    encoded first frame into the latent grid and zeroes its per-token timestep,
    rather than concatenating on the channel axis.
    """

    variant: str = WAN22_VARIANT_TI2V_5B

    # DiT (transformer/config.json)
    num_attention_heads: int = 24
    attention_head_dim: int = 128  # hidden size = 24 * 128 = 3072
    num_layers: int = 30
    ffn_dim: int = 14336
    in_channels: int = 48
    out_channels: int = 48
    patch_size: tuple[int, int, int] = (1, 2, 2)  # (temporal, height, width)
    text_dim: int = 4096
    freq_dim: int = 256
    rope_max_seq_len: int = 1024
    qk_norm: str = "rms_norm_across_heads"
    cross_attn_norm: bool = True
    eps: float = 1e-6

    # UMT5-XXL: tokenizer truncation and the padded embedding length.
    text_max_seq_len: int = 512

    # Wan2.2-VAE (vae/config.json)
    vae_z_dim: int = 48
    vae_scale_factor_spatial: int = 16  # pixels per latent cell (H and W)
    vae_scale_factor_temporal: int = 4  # frames per latent frame

    # UniPC. The solver order and train-timestep count are NOT knobs: the inline
    # port hardcodes their math, and a checkpoint that drifts from them hard-fails.
    flow_shift: float = 5.0

    # Generation defaults, each overridable per request via model_kwargs. The
    # sampling knobs match the diffusers WanPipeline defaults; the size defaults to
    # the checkpoint's native 720P tier (1280x704) rather than diffusers' 832x480,
    # so a no-size request produces full-resolution output.
    guidance_scale: float = 5.0
    default_num_inference_steps: int = 50
    default_height: int = 704  # must be a multiple of spatial_alignment
    default_width: int = 1280  # must be a multiple of spatial_alignment
    default_num_frames: int = 81  # must be 4k+1
    default_negative_prompt: str = ""
    video_fps: int = 24

    # Ceiling on the denoise loop; the per-request step count is clamped to it.
    max_denoise_steps: int = 100

    # torch.compile the inner DiT region (patchify -> blocks -> head). The UniPC
    # solver stays eager — its CPU-resident sigma trips Inductor (see
    # Wan22DitSubmodule). Default ON in serving; the reference-equivalence gates
    # build with this False so the eager path stays the bit-exact reference.
    compile_dit: bool = True

    # VAE-decode tiling policy: "auto" decodes untiled when free VRAM comfortably
    # exceeds the estimated untiled peak for the output size and tiles otherwise;
    # "tiled"/"untiled" force a path. WAN22_VAE_DECODE_TILING overrides at runtime.
    vae_decode_tiling: str = "auto"

    @property
    def hidden_size(self) -> int:
        """DiT hidden width (num_attention_heads * attention_head_dim)."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def spatial_alignment(self) -> tuple[int, int]:
        """(height, width) pixel multiples a request's size must satisfy — 32 each.

        The VAE downsamples a pixel dimension by 16 and the DiT then patchifies by
        2, so only exact multiples of the product survive both. An unaligned size
        has no clean failure inside the model, so it is rejected at the request
        seam (``Wan22Model.process_prompt``).
        """
        return (
            self.vae_scale_factor_spatial * self.patch_size[1],
            self.vae_scale_factor_spatial * self.patch_size[2],
        )
