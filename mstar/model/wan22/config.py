"""Configuration for Wan2.2: TI2V-5B (dense video DiT + UMT5-XXL + Wan2.2-VAE)
and T2V-A14B (two experts of one video DiT + UMT5-XXL + the Wan2.1 VAE).

The architecture values are facts of the ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` and
``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` checkpoints, hardcoded so that constructing
the model never touches the network: the dataclass defaults are TI2V-5B's, and
``Wan22Config.for_variant`` lays A14B's over them. ``Wan22Model._refresh_checkpoint_defaults``
re-reads ``flow_shift`` (and A14B's ``boundary_ratio``) from the checkpoint at load
time and hard-fails if the solver order or train-timestep count drift from what the
inline UniPC port implements; the DiT loader hard-fails on a transformer config
that is not the variant's.
"""

from dataclasses import dataclass

WAN22_VARIANT_TI2V_5B = "ti2v_5b"
WAN22_VARIANT_T2V_A14B = "t2v_a14b"

# What T2V-A14B changes from the dataclass defaults (TI2V-5B). The DiT values are
# transformer/config.json's (transformer_2 has the same), the VAE's are the Wan2.1
# VAE's, and boundary_ratio is model_index.json's. The sampling defaults are the
# checkpoint's model card (40 steps, guidance 4.0 on the high-noise expert and 3.0
# on the low-noise one, 1280x720 at 81 frames) with the checkpoint scheduler's
# flow_shift (3.0) and the Wan2.1 frame rate (16 fps).
_A14B = {
    "num_attention_heads": 40,
    "num_layers": 40,
    "ffn_dim": 13824,
    "in_channels": 16,
    "out_channels": 16,
    "vae_scale_factor_spatial": 8,
    "flow_shift": 3.0,
    "boundary_ratio": 0.875,
    "expand_timesteps": False,
    "guidance_scale": 4.0,
    "guidance_scale_2": 3.0,
    "default_num_inference_steps": 40,
    "default_height": 720,
    "default_width": 1280,
    "video_fps": 16,
}
_VARIANTS = {WAN22_VARIANT_TI2V_5B: {}, WAN22_VARIANT_T2V_A14B: _A14B}


# The dit's samplers: the checkpoint's UniPC, or a distilled checkpoint's few fixed steps,
# FastVideo's DMD recipe (fresh noise between steps) or LightX2V's step distill (none).
SAMPLER_UNIPC, SAMPLER_DMD, SAMPLER_STEP_DISTILL = "unipc", "dmd", "step_distill"
# Each distilled recipe's timesteps and shift: FastWan2.2-TI2V-5B's (its README's
# --dmd-denoising-steps; FastVideo's DMD_TRAINING_NOISE_SHIFT), and LightX2V's 4-step
# Wan2.2 distill (configs/wan22/wan_moe_t2v_distill.json: denoising_step_list, sample_shift).
_DISTILL_DEFAULTS = {
    SAMPLER_DMD: ((1000, 757, 522), 8.0),
    SAMPLER_STEP_DISTILL: ((1000, 750, 500, 250), 5.0),
}

# The DiT's self-attention: the array over its tokens, and the read that names its resource.
DIT_TOKENS, SELF_ATTN = "dit_tokens", "dit_self_attn"


@dataclass
class Wan22Config:
    """Wan2.2 model configuration; the defaults are TI2V-5B's (``for_variant`` builds A14B's).

    TI2V-5B: a single dense video DiT driven by UMT5-XXL embeddings, denoising
    Wan2.2-VAE latents with UniPC. T2V and I2V share the one transformer: I2V injects the
    encoded first frame into the latent grid and zeroes its per-token timestep,
    rather than concatenating on the channel axis.

    T2V-A14B: two DiTs of one architecture, the high-noise expert for the steps whose
    timestep is at least ``boundary_ratio * 1000`` and the low-noise expert after,
    each with its own guidance scale, denoising Wan2.1-VAE latents with UniPC. Text to
    video only, and one timestep per sample rather than per frame.
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
    # A timestep per latent frame (TI2V-5B, whose I2V zeroes frame 0), or one per
    # sample (A14B).
    expand_timesteps: bool = True
    # A14B: the share of the 1000 train timesteps run by the high-noise expert
    # (``transformer``); the steps below it run ``transformer_2``. None is one DiT.
    boundary_ratio: float | None = None

    # UMT5-XXL: tokenizer truncation and the padded embedding length.
    text_max_seq_len: int = 512

    # Wan2.2-VAE (vae/config.json)
    vae_scale_factor_spatial: int = 16  # pixels per latent cell (H and W)
    vae_scale_factor_temporal: int = 4  # frames per latent frame

    # UniPC. The solver order and train-timestep count are NOT knobs: the inline
    # port hardcodes their math, and a checkpoint that drifts from them hard-fails.
    flow_shift: float = 5.0

    # "dmd" (FastWan2.2-TI2V-5B) and "step_distill" (A14B with LightX2V's distill LoRAs)
    # sample a distilled checkpoint on ``dmd_timesteps`` with no CFG, one forward per
    # timestep (components/diffusion/dmd.py). A checkpoint choice, so a server setting,
    # not a request one.
    sampler: str = SAMPLER_UNIPC
    # None is the sampler's recipe (_DISTILL_DEFAULTS)
    dmd_timesteps: tuple[int, ...] | None = None
    # the shift of the sigma table the timesteps map through; None is the recipe's
    dmd_shift: float | None = None
    # LoRA adapters folded into the DiT at load (A14B: the high-noise expert), and into
    # A14B's low-noise expert: paths or {path, scale}
    lora: tuple = ()
    lora_2: tuple = ()

    # Generation defaults, each overridable per request via model_kwargs. The
    # sampling knobs match the diffusers WanPipeline defaults; the size defaults to
    # the checkpoint's native 720P tier (1280x704; A14B 1280x720) rather than diffusers'
    # 832x480, so a no-size request produces full-resolution output.
    guidance_scale: float = 5.0
    # The low-noise expert's guidance (A14B only). A request that sets
    # guidance_scale alone gets it for both experts, as in diffusers' WanPipeline.
    guidance_scale_2: float | None = None
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
    # serves; None is the default shape (TI2V-5B 1280x704x81). Every other shape runs
    # the transformer eager. The compile traces one graph per input shape and
    # requests choose theirs, so an open set would let each new shape cost a
    # compile on the shared worker. A listed shape compiles at most twice (CFG
    # on and off), for each of A14B's experts.
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

    @classmethod
    def for_variant(cls, variant: str = WAN22_VARIANT_TI2V_5B, **overrides) -> "Wan22Config":
        """The config of ``variant`` with ``overrides`` (serving knobs) laid over it."""
        if variant not in _VARIANTS:
            raise NotImplementedError(
                f"Wan2.2 variant {variant!r} is not implemented; the variants are {sorted(_VARIANTS)}."
            )
        return cls(variant=variant, **{**_VARIANTS[variant], **overrides})

    @property
    def is_moe(self) -> bool:
        """Two experts split the denoise steps at ``boundary_ratio`` (A14B)."""
        return self.boundary_ratio is not None

    @property
    def supports_i2v(self) -> bool:
        """Image conditioning, which only TI2V-5B's per-frame timestep path has."""
        return self.expand_timesteps

    def __post_init__(self):
        # A yaml "false" is a truthy string, so a non-boolean is rejected here
        # rather than read as on.
        if not isinstance(self.vae_decode_fast, bool):
            raise ValueError(f"Wan2.2 vae_decode_fast must be true or false, got {self.vae_decode_fast!r}.")
        self.fast_decode_sizes()
        samplers = (SAMPLER_UNIPC, *_DISTILL_DEFAULTS)
        if self.sampler not in samplers:
            raise ValueError(f"Wan2.2 sampler must be one of {samplers}, got {self.sampler!r}.")
        if self.lora_2 and not self.is_moe:
            raise ValueError("Wan2.2 lora_2 is for A14B's low-noise expert; this variant has one DiT.")
        if self.sampler in _DISTILL_DEFAULTS:
            default_steps, default_shift = _DISTILL_DEFAULTS[self.sampler]
            steps = default_steps if self.dmd_timesteps is None else self.dmd_timesteps
            if (
                isinstance(steps, str) or not isinstance(steps, (list, tuple)) or not steps
                or any(isinstance(t, bool) or not isinstance(t, int) or not 0 < t <= 1000 for t in steps)
                or any(a <= b for a, b in zip(steps, steps[1:], strict=False))
            ):
                raise ValueError(
                    f"Wan2.2 dmd_timesteps must be decreasing integers in (0, 1000], got {steps!r}."
                )
            self.dmd_timesteps = tuple(steps)
            shift = self.dmd_shift = default_shift if self.dmd_shift is None else self.dmd_shift
            if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not 0 < shift < float("inf"):
                raise ValueError(f"Wan2.2 dmd_shift must be a positive number, got {shift!r}.")
            # the recipe fixes both: one forward per timestep, and no CFG
            self.default_num_inference_steps = len(steps)
            self.guidance_scale = 1.0
            if self.is_moe:
                self.guidance_scale_2 = 1.0

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
        """(height, width) pixel multiples a request's size must satisfy: 32 each on
        TI2V-5B, 16 on A14B.

        The VAE downsamples a pixel dimension by 16 (8 on A14B) and the DiT then patchifies by
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
