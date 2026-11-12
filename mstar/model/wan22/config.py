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


# MagCache tables (``step_cache``): the residual magnitude ratio per step of a dense run, by name. "default" is
# TI2V-5B at 1280x704x81, 50 UniPC steps, CFG 5: the mean of ``{method: magcache_calibrate}`` over 9 VBench prompts
# on an H200 (the prompts' tables differ by under 0.01 per step up to step 44, 0.05 at step 48).
MAG_RATIOS: dict[str, tuple[float, ...]] = {"default": (
    1.0, 0.992, 0.99471, 0.99518, 0.99368, 0.9951, 0.99502, 0.99488, 0.99479, 0.99471, 0.99341, 0.99581, 0.99314,
    0.99347, 0.99443, 0.99429, 0.99342, 0.99339, 0.99263, 0.99164, 0.99186, 0.99173, 0.99149, 0.99054, 0.98872,
    0.9899, 0.9893, 0.98968, 0.98803, 0.98681, 0.98472, 0.98486, 0.98457, 0.9828, 0.98185, 0.98046, 0.9772, 0.97628,
    0.97319, 0.97305, 0.96804, 0.96475, 0.96176, 0.95681, 0.95313, 0.94797, 0.94417, 0.94284, 0.94411, 0.97743,
)}

# The DiT's self-attention: the array over its tokens, and the read that names its resource.
DIT_TOKENS, SELF_ATTN = "dit_tokens", "dit_self_attn"


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

    # The denoise loop's max_iters; the conductor refuses a num_inference_steps past it.
    max_denoise_steps: int = 100

    # Ceiling on a request's height x width x num_frames (about 3.6x the default
    # 1280x704x81 clip); DiT memory grows with it. A larger request is a 400.
    max_video_pixels: int = 256 * 2**20

    # torch.compile the inner DiT region (patchify -> blocks -> head). The UniPC
    # solver stays eager — its CPU-resident sigma trips Inductor. Default ON in serving; the reference-equivalence gates
    # build with this False to hold the eager path against the reference.
    compile_dit: bool = True
    # "WxHxF" request shapes (width x height x num_frames) the compiled DiT
    # serves; None is the default shape (1280x704x81). Every other shape runs
    # the transformer eager. The compile traces one graph per input shape and
    # requests choose theirs, so an open set would let each new shape cost a
    # compile on the shared worker. A listed shape compiles at most twice (CFG
    # on and off).
    compile_dit_sizes: tuple[str, ...] | None = None
    # Requests at one latent grid sharing a dit forward. Each row count traces its
    # own compiled graph, on top of the shapes above.
    dit_max_batch_size: int = 1

    # VAE-decode tiling policy: "auto" decodes untiled when free VRAM comfortably
    # exceeds the estimated untiled peak for the output size and tiles otherwise;
    # "tiled"/"untiled" force a path. WAN22_VAE_DECODE_TILING overrides at runtime.
    vae_decode_tiling: str = "auto"

    # Compiled untiled decode through the shared Wan VAE decoder. Each output size costs a compile, so only
    # ``vae_decode_fast_sizes`` use it; other sizes and tiled decodes run stock ``vae.decode`` / ``tiled_decode``.
    # Compiled in Inductor's deterministic mode, so the frames do not depend on the server process, except on a
    # torch without that mode (2.9.1).
    vae_decode_fast: bool = True
    # "WxH" output sizes the fast decode compiles for; None is the default size
    # (1280x704). Requests choose their size, so an open list would let each new
    # size cost a compile on the shared worker.
    vae_decode_fast_sizes: tuple[str, ...] | None = None

    # Opt-in step caching, lossy (``components/diffusion/step_cache.py``); None runs every step. E.g.
    # ``{method: magcache, threshold: 0.06, max_skip_steps: 2, retention_ratio: 0.2}`` reuses the block residual on
    # the steps MagCache picks from the calibrated table in ``MAG_RATIOS`` (or a ``mag_ratios`` list given here);
    # ``{method: magcache_calibrate}`` logs a table from a dense run.
    step_cache: dict | None = None

    def __post_init__(self):
        # A yaml "false" is a truthy string, so a non-boolean is rejected here
        # rather than read as on.
        if not isinstance(self.vae_decode_fast, bool):
            raise ValueError(f"Wan2.2 vae_decode_fast must be true or false, got {self.vae_decode_fast!r}.")
        self.fast_decode_sizes()

    def fast_decode_sizes(self) -> frozenset[tuple[int, int]]:
        """``vae_decode_fast_sizes`` as (width, height) pixel pairs."""
        sizes = self.vae_decode_fast_sizes
        if sizes is None:
            return frozenset({(self.default_width, self.default_height)})
        if isinstance(sizes, str) or not isinstance(sizes, (list, tuple)):
            raise ValueError(f"Wan2.2 vae_decode_fast_sizes must be a list of 'WxH' sizes, got {sizes!r}.")
        align_h, align_w = self.spatial_alignment
        parsed = set()
        for size in sizes:
            try:
                width, height = (int(v) for v in str(size).lower().split("x"))
            except ValueError:
                raise ValueError(f"Wan2.2 vae_decode_fast_sizes entry must be 'WxH', got {size!r}.") from None
            if width <= 0 or height <= 0 or width % align_w or height % align_h:
                raise ValueError(
                    f"Wan2.2 vae_decode_fast_sizes entry {size!r} must be positive multiples of "
                    f"{align_w}x{align_h}."
                )
            parsed.add((width, height))
        return frozenset(parsed)

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

    def compile_dit_grids(self) -> frozenset[tuple[int, int, int]]:
        """``compile_dit_sizes`` as latent (frames, height, width) grids, the
        trailing dims of the DiT's input."""
        sizes = self.compile_dit_sizes
        if sizes is None:
            sizes = (f"{self.default_width}x{self.default_height}x{self.default_num_frames}",)
        if isinstance(sizes, str) or not isinstance(sizes, (list, tuple)):
            raise ValueError(f"Wan2.2 compile_dit_sizes must be a list of 'WxHxF' shapes, got {sizes!r}.")
        align_h, align_w = self.spatial_alignment
        temporal = self.vae_scale_factor_temporal
        grids = set()
        for size in sizes:
            try:
                width, height, frames = (int(v) for v in str(size).lower().split("x"))
            except ValueError:
                raise ValueError(f"Wan2.2 compile_dit_sizes entry must be 'WxHxF', got {size!r}.") from None
            if min(width, height, frames) <= 0 or width % align_w or height % align_h or (frames - 1) % temporal:
                raise ValueError(
                    f"Wan2.2 compile_dit_sizes entry {size!r} must be positive multiples of "
                    f"{align_w}x{align_h} with {temporal}k+1 frames."
                )
            grids.add((
                (frames - 1) // temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ))
        return frozenset(grids)
