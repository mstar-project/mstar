"""Configuration for Waypoint-1.5 (autoregressive video world model).

The values here are facts of the published 720P and 360P checkpoint manifests,
hardcoded so that constructing the model never touches the network. The
reference reads those YAML files through OmegaConf with
``MODEL_CONFIG_DEFAULTS`` merged underneath; this dataclass is the merged result.

One 1.28B DiT denoises exactly one latent frame per step and the KV cache IS the
world state, so attention geometry is per-layer heterogeneous (18 layers over a
16-frame local window, 6 over a 128-frame window at stride 8; see
``global_layers`` / ``ring_frames``) and every frame costs 5 forwards: 4
non-committing Euler passes over ``scheduler_sigmas``, then 1 committing pass at
sigma=0 that writes the settled K/V into the ring.
"""

import math
from dataclasses import dataclass, field

WAYPOINT_VARIANT_720P = "waypoint-1.5-1b-720p"
WAYPOINT_VARIANT_360P = "waypoint-1.5-1b-360p"

WAYPOINT_VARIANT_HF_REPOS: dict[str, str] = {
    WAYPOINT_VARIANT_720P: "Overworld/Waypoint-1.5-1B",
    WAYPOINT_VARIANT_360P: "Overworld/Waypoint-1.5-1B-360P",
}

WAYPOINT_SCHEDULER_SIGMAS = (1.0, 0.9, 0.75, 0.3, 0.0)

# Geometry is the only manifest-field difference between the two deployment
# configs; the repositories still hold variant-specific weights. Startup
# validation reads this rather than a downloaded checkpoint.
WAYPOINT_VARIANT_GEOMETRY: dict[str, tuple[int, int, int]] = {
    WAYPOINT_VARIANT_720P: (512, 16, 32),
    WAYPOINT_VARIANT_360P: (128, 8, 16),
}


@dataclass
class WaypointConfig:
    """Waypoint-1.5-1B model configuration.

    Field names track the reference ``config.yaml`` keys, not mstar's usual
    spellings, so this diffs directly against the checkpoint's YAML.
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
    # Layer i is "global" iff (i - global_attn_offset % period) % period == 0:
    # {3, 7, 11, 15, 19, 23}. Local layers see local_window consecutive frames;
    # global layers see global_window frames at stride global_pinned_dilation,
    # i.e. 16 retained frames spanning 128 frames of history.
    local_window: int = 16
    global_window: int = 128
    global_pinned_dilation: int = 8
    global_attn_period: int = 4
    global_attn_offset: int = -1

    # ---- RoPE -------------------------------------------------------------
    # OrthoRoPE splits the head into disjoint axis slices: d_head//8 rotation
    # PAIRS to x, d_head//8 to y, d_head//4 to t -- 8+8+16 = 32 pairs for
    # d_head=64, so x owns dims 0-15, y 16-31, t 32-63 and nothing is left
    # unrotated. The counts are pairs, not dims.
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
    # 5 entries -> 4 Euler steps -> one separate committing pass at sigma=0.
    # Not a per-request knob: the reference's cached sigma/cond tables key on it.
    scheduler_sigmas: tuple[float, ...] = WAYPOINT_SCHEDULER_SIGMAS

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
    # ``global_window * tokens_per_frame`` tokens but can only address
    # ``global_window // global_pinned_dilation`` frame slots, so 7/8 of it is
    # permanently unwritten and masked off. Compacting it is bit-exact and saves
    # ~1.35 GiB; True restores the reference's allocation for an A/B parity run.
    full_global_ring: bool = False

    # The reference's ``NoCastModule._apply`` casts every tensor it holds to the
    # requested dtype and back, so its derived fp32 tables (``rope_angles.xy``/
    # ``inv_t``, ``denoise_step_emb.freq``) are served bf16-quantized —
    # parameters recover from ``load_state_dict``, non-persistent buffers never
    # do — and its cached sigma LUT rounds the same way through a TF32 batch-5
    # GEMM the served per-sigma GEMV avoids. True reproduces that rounding, which
    # is what the live parity gate compares against; False serves exact tables
    # and deliberately diverges from the released reference.
    reference_compat: bool = True

    # torch.compile the two OUTER regions (denoise pass, cache pass), matching
    # the reference's two @torch.compile(fullgraph=True, dynamic=False) sites.
    # Independent of CUDA graph capture, and of the masked FlexAttention
    # primitive, which stays compiled for correctness (engine/resources/attn/flex.py).
    compile_dit: bool = True

    # Attempt fixed-shape CUDA graph capture for the encoder prime, steady DiT
    # rollout, and decoder prime/rollout paths. An optimization: disabled
    # declares no buckets, and a failed capture falls back to eager submodule
    # forwards.
    cuda_graph: bool = True

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
        positive_ints = {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_model": self.d_model,
            "mlp_ratio": self.mlp_ratio,
            "channels": self.channels,
            "tokens_per_frame": self.tokens_per_frame,
            "height": self.height,
            "width": self.width,
            "local_window": self.local_window,
            "global_window": self.global_window,
            "global_pinned_dilation": self.global_pinned_dilation,
            "global_attn_period": self.global_attn_period,
            "ctrl_conditioning_period": self.ctrl_conditioning_period,
            "n_buttons": self.n_buttons,
            "base_fps": self.base_fps,
            "inference_fps": self.inference_fps,
            "temporal_compression": self.temporal_compression,
            "max_frames": self.max_frames,
        }
        invalid = [name for name, value in positive_ints.items() if type(value) is not int or value <= 0]
        if invalid:
            values = ", ".join(f"{name}={positive_ints[name]!r}" for name in invalid)
            raise ValueError(f"Waypoint positive integer fields are invalid: {values}.")
        if len(self.patch) != 2 or any(type(size) is not int or size <= 0 for size in self.patch):
            raise ValueError(f"patch must contain two positive integers; got {self.patch!r}.")
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
        if self.d_head % 8:
            raise ValueError(
                f"OrthoRoPE requires d_head ({self.d_head}) to be divisible by 8."
            )
        if self.global_window % self.global_pinned_dilation:
            raise ValueError(
                f"global_window ({self.global_window}) must be divisible by "
                f"global_pinned_dilation ({self.global_pinned_dilation})."
            )
        if self.inference_fps % self.temporal_compression:
            raise ValueError(
                f"inference_fps ({self.inference_fps}) must be divisible by "
                f"temporal_compression ({self.temporal_compression})."
            )
        latent_fps = self.inference_fps // self.temporal_compression
        if self.base_fps % latent_fps:
            raise ValueError(
                f"base_fps ({self.base_fps}) must be divisible by latent fps "
                f"({self.inference_fps}/{self.temporal_compression}={latent_fps})."
            )
        sigmas = tuple(float(value) for value in self.scheduler_sigmas)
        if len(sigmas) < 2 or not all(math.isfinite(value) for value in sigmas):
            raise ValueError(
                f"scheduler_sigmas must contain at least two finite values; got {self.scheduler_sigmas!r}."
            )
        pairs = zip(sigmas[:-1], sigmas[1:], strict=True)
        if sigmas[-1] != 0.0 or any(left <= right for left, right in pairs):
            raise ValueError(
                "scheduler_sigmas must be strictly descending and end at 0.0; "
                f"got {self.scheduler_sigmas!r}."
            )

    def validate_supported_deployment(self) -> None:
        """Reject a config that is internally valid but not a released shape.

        Separate from ``__post_init__`` because unit tests construct smaller,
        internally consistent models. Startup calls it before allocating storage.
        """
        expected_geometry = WAYPOINT_VARIANT_GEOMETRY.get(self.variant)
        if expected_geometry is None:
            raise ValueError(
                f"Unsupported Waypoint variant {self.variant!r}; expected one of "
                f"{sorted(WAYPOINT_VARIANT_GEOMETRY)}."
            )
        actual_geometry = (self.tokens_per_frame, self.height, self.width)
        if actual_geometry != expected_geometry:
            raise ValueError(
                f"Waypoint variant {self.variant!r} requires "
                f"(tokens_per_frame, height, width)={expected_geometry}; got {actual_geometry}."
            )
        checkpoint_facts = {
            "n_layers": (self.n_layers, 24),
            "n_heads": (self.n_heads, 32),
            "n_kv_heads": (self.n_kv_heads, 16),
            "d_model": (self.d_model, 2048),
            "mlp_ratio": (self.mlp_ratio, 4),
            "channels": (self.channels, 32),
            "patch": (self.patch, (2, 2)),
            "local_window": (self.local_window, 16),
            "global_window": (self.global_window, 128),
            "global_pinned_dilation": (self.global_pinned_dilation, 8),
            "global_attn_period": (self.global_attn_period, 4),
            "global_attn_offset": (self.global_attn_offset, -1),
            "rope_impl": (self.rope_impl, "ortho"),
            "rope_nyquist_frac": (self.rope_nyquist_frac, 0.8),
            "rope_theta": (self.rope_theta, 10_000.0),
            "noise_conditioning": (self.noise_conditioning, "wan"),
            "value_residual": (self.value_residual, True),
            "gated_attn": (self.gated_attn, False),
            "moe": (self.moe, False),
            "prompt_conditioning": (self.prompt_conditioning, None),
            "ctrl_conditioning": (self.ctrl_conditioning, True),
            "ctrl_cond_dropout": (self.ctrl_cond_dropout, 0.0),
            "ctrl_conditioning_period": (self.ctrl_conditioning_period, 3),
            "n_buttons": (self.n_buttons, 256),
            "scheduler_sigmas": (tuple(self.scheduler_sigmas), WAYPOINT_SCHEDULER_SIGMAS),
            "base_fps": (self.base_fps, 15),
            "inference_fps": (self.inference_fps, 60),
            "temporal_compression": (self.temporal_compression, 4),
            "max_frames": (self.max_frames, 512),
            "taehv_ae": (self.taehv_ae, True),
            "ae_uri": (self.ae_uri, "Overworld-Models/taehv1_5"),
            "auto_aspect_ratio": (self.auto_aspect_ratio, True),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in checkpoint_facts.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "Waypoint deployment config disagrees with the released checkpoint: "
                + "; ".join(mismatches)
                + "."
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
        """Controller feature width: mouse(2) + button(n_buttons) + scroll(1),
        concatenated in that order by ``ControllerInputEmbedding.forward``."""
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

        ``base_fps // (inference_fps // temporal_compression)`` = 1 here, so the
        RoPE clock ``t_pos`` and the ring-bucketing clock ``f_pos`` are equal.
        They stay two separate values through the model: another inference_fps
        separates them, and conflating them drifts silently rather than crashing.
        """
        return self.base_fps // (self.inference_fps // self.temporal_compression)

    @property
    def num_denoise_steps(self) -> int:
        """Euler steps per frame (4). The committing pass is separate."""
        return len(self.scheduler_sigmas) - 1

    @property
    def global_layers(self) -> frozenset[int]:
        """Layer indices attending over the dilated 128-frame window.

        ``{3, 7, 11, 15, 19, 23}``: the offset is taken modulo the period first,
        so offset=-1 means "the last layer of each period".
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

        ``ring + one scratch frame``. The scratch frame at the tail is where a
        frozen denoise pass parks its K/V so the current frame can attend to
        itself; permanently visible, overwritten every forward.
        """
        return (self.ring_frames(layer_idx) + 1) * self.tokens_per_frame


def waypoint_1_5_1b_720p() -> WaypointConfig:
    """The default checkpoint: 512 tokens/frame over a 32x64 latent grid."""
    return WaypointConfig(variant=WAYPOINT_VARIANT_720P)


def waypoint_1_5_1b_360p() -> WaypointConfig:
    """The 360P checkpoint: 128 tokens/frame over a 16x32 latent grid."""
    return WaypointConfig(
        variant=WAYPOINT_VARIANT_360P, tokens_per_frame=128, height=8, width=16
    )
