"""Configuration for Waypoint-1.5 (autoregressive video world model).

The values here are facts of the ``Overworld/Waypoint-1.5-1B`` checkpoint's
``config.yaml``, hardcoded so that constructing the model never touches the
network. The reference implementation reads that YAML through OmegaConf with
``MODEL_CONFIG_DEFAULTS`` merged underneath; this dataclass is the merged
result, with the defaults that actually matter spelled out.

Waypoint is not a diffusion pipeline that happens to run several times. It is a
*world model*: one 1.28B DiT denoises exactly one latent frame per step, and the
KV cache IS the world state rather than an optimization over it. Two facts
follow and drive most of this file:

  * Attention geometry is per-layer heterogeneous. 18 layers attend densely over
    a 16-frame local window; 6 attend over a 128-frame window subsampled at
    stride 8. See ``global_layers`` / ``ring_frames``.
  * Every frame costs 5 forwards: 4 non-committing Euler denoise passes over
    ``scheduler_sigmas``, then 1 committing pass at sigma=0 that writes the
    settled K/V into the ring.
"""

from dataclasses import dataclass, field

# The 720P checkpoint is the one this port implements end to end. The 360P
# sibling differs only in the token grid (see ``waypoint_1_5_1b_360p``).
WAYPOINT_VARIANT_720P = "waypoint-1.5-1b-720p"
WAYPOINT_VARIANT_360P = "waypoint-1.5-1b-360p"


@dataclass
class WaypointConfig:
    """Waypoint-1.5-1B model configuration.

    Field names track the reference ``config.yaml`` keys rather than mstar's
    usual spellings. That is deliberate: the checkpoint's YAML is the ground
    truth a reviewer will diff this against, and renaming ``d_model`` to
    ``hidden_size`` buys nothing but a translation step during review.
    """

    variant: str = WAYPOINT_VARIANT_720P

    # ---- Transformer ------------------------------------------------------
    n_layers: int = 24
    n_heads: int = 32
    n_kv_heads: int = 16  # GQA: 2 query heads per kv head
    d_model: int = 2048
    mlp_ratio: int = 4
    channels: int = 32  # VAE latent channels

    # ---- Token grid -------------------------------------------------------
    # height/width are POST-patch token counts, not latent pixels: the latent
    # frame is (height*patch[0], width*patch[1]) and patchify collapses it to
    # tokens_per_frame tokens. The reference asserts tokens_per_frame == h*w.
    tokens_per_frame: int = 512
    height: int = 16
    width: int = 32
    patch: tuple[int, int] = (2, 2)

    # ---- Attention geometry -----------------------------------------------
    # Layer i is "global" iff (i - global_attn_offset % period) % period == 0.
    # With offset=-1, period=4 that is {3, 7, 11, 15, 19, 23}; the other 18 are
    # local. Local layers see local_window consecutive frames. Global layers see
    # global_window frames subsampled at stride global_pinned_dilation, i.e.
    # global_window // global_pinned_dilation == 16 retained frames spanning
    # 128 frames of history.
    local_window: int = 16
    global_window: int = 128
    global_pinned_dilation: int = 8
    global_attn_period: int = 4
    global_attn_offset: int = -1

    # ---- RoPE -------------------------------------------------------------
    # OrthoRoPE splits the head into disjoint axis slices. d_head//8 rotation
    # PAIRS go to x, d_head//8 to y, d_head//4 to t -- 8+8+16 = 32 pairs for
    # d_head=64, i.e. the whole head. x owns dims 0-15, y 16-31, t 32-63;
    # nothing is left unrotated. (The counts are pairs, not dims: the angle
    # table is d_head//2 wide and unfold(-1, 2, 2) pairs the head up.)
    rope_impl: str = "ortho"
    rope_nyquist_frac: float = 0.8
    rope_theta: float = 10000.0

    # ---- Conditioning -----------------------------------------------------
    noise_conditioning: str = "wan"  # WAN-style CondHead with a shared cond_proj
    value_residual: bool = True
    gated_attn: bool = False
    moe: bool = False
    prompt_conditioning: str | None = None  # no cross-attention in this checkpoint

    ctrl_conditioning: bool = True
    ctrl_cond_dropout: float = 0.0
    # Controller conditioning is injected on layers where i % period == 0, i.e.
    # 8 of 24 layers: {0, 3, 6, 9, 12, 15, 18, 21}.
    ctrl_conditioning_period: int = 3
    n_buttons: int = 256

    # ---- Sampling ---------------------------------------------------------
    # 5 entries -> 4 Euler steps (zip(sigmas, sigmas.diff())) -> then one
    # separate committing pass at sigma=0. Not a per-request knob: the cached
    # sigma/cond tables in the reference's inference patches are keyed on it.
    scheduler_sigmas: tuple[float, ...] = (1.0, 0.9, 0.75, 0.3, 0.0)

    # ---- Temporal ---------------------------------------------------------
    base_fps: int = 15  # fps the RoPE time axis was trained against
    inference_fps: int = 60  # raw video fps
    temporal_compression: int = 4  # raw frames per latent frame (TAEHV)
    max_frames: int = 512  # training-time rollout ceiling; not enforced here

    # ---- VAE --------------------------------------------------------------
    taehv_ae: bool = True
    ae_uri: str = "Overworld-Models/taehv1_5"
    auto_aspect_ratio: bool = True

    # ---- Port-local knobs (NOT checkpoint facts) --------------------------
    # The reference allocates global-layer ring storage as
    # ``global_window * tokens_per_frame`` tokens but can only ever address
    # ``global_window // global_pinned_dilation`` frame slots, so 7/8 of that
    # buffer is permanently unwritten and permanently masked off. Compacting it
    # is bit-exact -- unwritten blocks are absent from the BlockMask, and the
    # stable argsort that orders the visited blocks is unaffected by trailing
    # False entries -- and saves ~1.35 GiB. Set True to restore the reference's
    # allocation for an A/B parity run.
    full_global_ring: bool = False

    # torch.compile the two OUTER regions (denoise pass, cache pass), matching
    # the reference's two @torch.compile(fullgraph=True, dynamic=False) sites.
    #
    # This is a throughput knob only. It does NOT govern attention correctness:
    # the BlockMask carries a no-op mask_mod, so eager flex_attention ignores it and
    # attends to unwritten ring slots (measured: 2.7e-01 off a masked-dense
    # reference, silently). The FLEX attention resource
    # (engine/resources/attn/flex.py) pins its own torch.compile around the
    # flex_attention call for that reason. Unlike wan22, this model has no
    # eager reference-equivalence mode.
    compile_dit: bool = True

    # Guard rails the ported modules assert against, kept here so a drifting
    # checkpoint fails loudly at construction rather than silently mis-serving.
    _supported_rope_impls: tuple[str, ...] = field(
        default=("ortho",), repr=False, compare=False
    )
    _supported_noise_conditioning: tuple[str, ...] = field(
        default=("wan",), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.rope_impl not in self._supported_rope_impls:
            raise ValueError(
                f"WaypointDiT implements rope_impl in {self._supported_rope_impls}; "
                f"got {self.rope_impl!r}."
            )
        if self.noise_conditioning not in self._supported_noise_conditioning:
            raise ValueError(
                f"WaypointDiT implements noise_conditioning in "
                f"{self._supported_noise_conditioning}; got {self.noise_conditioning!r}."
            )
        if self.moe:
            raise ValueError("WaypointDiT does not implement the MoE variant.")
        if self.prompt_conditioning is not None:
            raise ValueError(
                "WaypointDiT does not implement prompt cross-attention; this "
                f"checkpoint declares prompt_conditioning={self.prompt_conditioning!r}."
            )
        if self.tokens_per_frame != self.height * self.width:
            raise ValueError(
                f"tokens_per_frame ({self.tokens_per_frame}) must equal "
                f"height*width ({self.height}*{self.width})."
            )
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for GQA."
            )
        if self.global_window % self.global_pinned_dilation:
            raise ValueError(
                f"global_window ({self.global_window}) must be divisible by "
                f"global_pinned_dilation ({self.global_pinned_dilation})."
            )

    # ---- Derived ----------------------------------------------------------

    @property
    def d_head(self) -> int:
        """Per-head width (64). OrthoRoPE rotates exactly half of it."""
        return self.d_model // self.n_heads

    @property
    def enable_gqa(self) -> bool:
        return self.n_kv_heads != self.n_heads

    @property
    def d_ctrl_in(self) -> int:
        """Controller feature width: mouse(2) + button(n_buttons) + scroll(1).

        The concat order in ``ControllerInputEmbedding.forward`` is
        ``(mouse, button, scroll)``. Getting it wrong does not raise -- the
        widths still sum to 259 -- it just produces plausible wrong video.
        """
        return self.n_buttons + 3

    @property
    def latent_height(self) -> int:
        """Latent frame height in cells, pre-patchify (32)."""
        return self.height * self.patch[0]

    @property
    def latent_width(self) -> int:
        """Latent frame width in cells, pre-patchify (64)."""
        return self.width * self.patch[1]

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """One latent frame as (channels, height, width) = (32, 32, 64)."""
        return (self.channels, self.latent_height, self.latent_width)

    @property
    def ts_mult(self) -> int:
        """RoPE time-axis stride per latent frame.

        ``base_fps // (inference_fps // temporal_compression)`` = 15 // 15 = 1
        for this checkpoint, so the RoPE clock ``t_pos`` and the ring-bucketing
        clock ``f_pos`` are numerically equal. They are still threaded through
        the model as two separate values: a checkpoint served at a different
        inference_fps would separate them, and conflating them there is a
        silent-drift bug rather than a crash.
        """
        return self.base_fps // (self.inference_fps // self.temporal_compression)

    @property
    def num_denoise_steps(self) -> int:
        """Euler steps per frame (4). The committing pass is separate."""
        return len(self.scheduler_sigmas) - 1

    @property
    def global_layers(self) -> frozenset[int]:
        """Layer indices attending over the dilated 128-frame window.

        ``{3, 7, 11, 15, 19, 23}`` -- note the offset is applied modulo the
        period first, so offset=-1 means "the last layer of each period".
        """
        period = self.global_attn_period
        off = self.global_attn_offset % period
        return frozenset(i for i in range(self.n_layers) if (i - off) % period == 0)

    @property
    def ctrl_layers(self) -> frozenset[int]:
        """Layer indices that fuse controller conditioning (8 of 24)."""
        if not self.ctrl_conditioning:
            return frozenset()
        return frozenset(
            i for i in range(self.n_layers) if i % self.ctrl_conditioning_period == 0
        )

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.global_layers

    def pinned_dilation(self, layer_idx: int) -> int:
        """Frame stride this layer retains history at (8 global, 1 local)."""
        return self.global_pinned_dilation if self.is_global_layer(layer_idx) else 1

    def ring_frames(self, layer_idx: int) -> int:
        """Number of frame slots in this layer's ring.

        Local: 16 consecutive frames. Global: 16 slots holding frames 8 apart,
        spanning 128 frames. ``full_global_ring`` restores the reference's
        8x-oversized global allocation, whose extra slots are unreachable.
        """
        if not self.is_global_layer(layer_idx):
            return self.local_window
        if self.full_global_ring:
            return self.global_window
        return self.global_window // self.global_pinned_dilation

    def ring_buckets(self, layer_idx: int) -> int:
        """Addressable ring slots. Equals ``ring_frames`` unless
        ``full_global_ring`` is set, in which case it is 8x smaller."""
        if not self.is_global_layer(layer_idx):
            return self.local_window
        return self.global_window // self.global_pinned_dilation

    def kv_capacity(self, layer_idx: int) -> int:
        """Total KV slots for this layer, in tokens.

        ``ring + one scratch frame``. The scratch frame at the tail is where an
        uncommitted (frozen) denoise pass parks its K/V so the current frame can
        attend to itself; it is permanently marked visible and is overwritten on
        every forward.
        """
        return (self.ring_frames(layer_idx) + 1) * self.tokens_per_frame


def waypoint_1_5_1b_720p() -> WaypointConfig:
    """The default checkpoint: 512 tokens/frame over a 32x64 latent grid."""
    return WaypointConfig(variant=WAYPOINT_VARIANT_720P)


def waypoint_1_5_1b_360p() -> WaypointConfig:
    """The 360P sibling. Identical weights-shape-wise except the token grid."""
    return WaypointConfig(
        variant=WAYPOINT_VARIANT_360P, tokens_per_frame=128, height=8, width=16
    )
