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

from mstar.engine.state import KeptWindow

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

    # True: the reference's numerics, with which a rollout is bit-identical to
    # the reference's: both DiT attentions on FlashAttention-2 varlen (needs
    # flash-attn), an eager DiT, and the fp32 contiguous eager VAE encode and
    # decode. False: the engine's ragged attention resource (its backend is the
    # deployment's; AUTO is FlashInfer's prefill, the cutlass FMHA on
    # Blackwell), each DiT block compiled (Inductor's deterministic mode, eager
    # rounding; the attentions and the cache writes stay eager), the fp32
    # encode channels-last (187 -> 140 ms per chunk on a B300 at 464x832 for
    # the first ~8 chunks, after which the black-frame encode is cached; the
    # latents within 0.0023 of the contiguous encode's), and the decode
    # channels-last, compiled and bf16 where cuDNN >= 9.16 has fast bf16
    # conv3d (291 -> 77 ms per 16-frame chunk, frames 51 dB from the fp32
    # decode of the same latents; fp32 channels-last compiled, the pick under
    # an older cuDNN, 135 ms at 71 dB). Nothing in the rollout reads the
    # decoded frames, so the decode's error stays per frame. The compiles are
    # per output size; the default size is compiled at load.
    reference_numerics: bool = False
    # Compile the blocks at the -O level's options instead of Inductor's
    # deterministic defaults: 35.4 -> 37.1 fps served at 464x832 on a B300
    # (-O2), but each server process picks its own kernels, so one seed gives
    # a different rollout per process. No effect under reference_numerics.
    dit_compile_autotune: bool = False
    # Add each chunk's x0 latents, float frames and I2V condition to its outputs (``CHUNK_OUTPUTS``), for the parity
    # harnesses.
    keep_chunk_outputs: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    def frame_tokens(self, lat_h: int, lat_w: int) -> int:
        """A latent frame's tokens at a ``lat_h`` by ``lat_w`` latent: one per DiT patch."""
        return lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

    @property
    def max_frame_tokens(self) -> int:
        """The most tokens a frame has at any aspect: ``max_area`` holds that many DiT patches of pixels."""
        return self.max_area // (self.vae_stride[1] * self.vae_stride[2] * self.patch_size[1] * self.patch_size[2])

    def kept_window(self, frame_tokens: int) -> KeptWindow:
        """What a chunk's queries attend before its own frames, at ``frame_tokens`` a frame: the sink, then the latest
        frames up to ``local_attn_size``. Without a window, every frame of the longest rollout."""
        if self.local_attn_size == -1:
            return KeptWindow(0, window=self.max_chunks * self.chunk_size * frame_tokens)
        return KeptWindow(self.sink_size * frame_tokens,
                          window=(self.local_attn_size - self.sink_size - self.chunk_size) * frame_tokens)

    def window_pages(self, frame_tokens: int, page_size: int) -> int:
        """The most pages of ``page_size`` a rollout at ``frame_tokens`` a frame holds as the pool releases them: its
        sink in whole pages, the window, less than a page past it, and the chunk a step writes. This is
        ``RetentionPolicy.max_pages`` of the kept window, which the model can't import."""
        kept = self.kept_window(frame_tokens)
        sink = -(-kept.sink // page_size) * page_size
        return -(-(sink + kept.window + page_size - 1 + self.chunk_size * frame_tokens) // page_size)

    def max_rollout_tokens(self) -> int:
        """The self-attention entries of the longest rollout at the largest area."""
        return self.max_chunks * self.chunk_size * self.max_frame_tokens


def lingbot_world_1_3b() -> LingBotWorldConfig:
    return LingBotWorldConfig(variant=LINGBOT_VARIANT_1_3B)


LINGBOT_CONFIGS = {
    LINGBOT_VARIANT_1_3B: lingbot_world_1_3b,
}
