"""Praxis-1 config, read from a checkpoint's ``config.json``.

The checkpoint carries the whole variant: the tile canvas, the pose frame of
the actions, the attention backend and the rig. We read the fields the served
graph needs and refuse a variant we have not ported, rather than serving a
checkpoint that runs and means something else.

The rig follows ``embodiment``: ``single_arm`` is the DROID rig (exterior and
wrist views, 10-wide action rows), ``bimanual`` is the YAM rig (top and two
wrists, 20-wide rows). One policy call answers one chunk: 24 command rows at
15 Hz, carried by 6 action tokens of 4 frames each, beside 6 video latents.

Shapes worth having in one place, for the droid checkpoint: the canvas is one
224x416 tile per view stacked side by side (224x832), the VAE takes it to 48
channels at stride (4, 16, 16), and the backbone is a 3072-wide MoE with 14
encoder blocks over 128 routed experts plus a shared one, and a 6-block
decoder.

The MoE's two expert counts are the one place in this file where a field name
alone will mislead a reader. See :func:`_moe_fields`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# The rigs the published checkpoints cover. The action width belongs to the
# rig rather than to a knob: the head's output projection has it.
RIG_VIEWS: dict[str, tuple[str, ...]] = {
    "single_arm": ("exterior", "wrist"),
    "bimanual": ("top_left", "wrist_left", "wrist_right"),
}
RIG_ACTION_DIMS: dict[str, int] = {"single_arm": 10, "bimanual": 20}

# The variant this port serves. Anything else loads tensors that would mean
# something different under our graph, so it is refused at load.
SUPPORTED_LAYOUT = "chunk_causal"
SUPPORTED_CANVAS = "tiles"
SUPPORTED_VAE = "Wan2.2"
# A base checkpoint has no action heads (upstream refuses it the same way).
BASE_EMBODIMENT = "base"

# The MoE regime :mod:`mstar.model.praxis1.components.moe` implements. Each of
# these is a field the checkpoint states and the reference then accepts without
# reading, because its layer hard-codes the same choice. So a checkpoint that
# flipped one would be mis-served by the reference too, quietly, and we would
# have no oracle to catch it with. Refusing is the only honest answer.
SUPPORTED_ROUTER_ACTIVATION = "sigmoid"
SUPPORTED_ROUTER_DTYPE = "float32"
SUPPORTED_ROUTED_WEIGHT_LAYOUT = "expert_first"


# The published checkpoints' own shape, for dummy mode: the startup checks and
# the graph tests construct a model with no checkpoint to read. Serving reads
# the real config.json, so these are never what a request runs against.
RIG_DEFAULTS: dict[str, dict] = {
    "single_arm": {"canvas": (224, 832), "tile": (224, 416), "action_dim": 10},
    "bimanual": {"canvas": (256, 1248), "tile": (256, 416), "action_dim": 20},
}
# Shared by both checkpoints.
_SHARED_DEFAULTS = {
    "dim": 3072, "timestep_dim": 128, "num_registers": 4,
    "encoder_depth": 14, "decoder_depth": 6,
    "tap_blocks": (0, 2, 4, 6, 7, 9, 11, 13),
    "num_heads": 24, "num_kv_heads": 8, "head_dim": 128,
    "flex_block_size": 128, "attention_backend": "flash",
    "num_experts_with_shared": 129, "top_k_with_shared": 9,
    "num_routed_experts": 128, "top_k_routed": 8, "routed_scaling_factor": 2.5,
    # What all four published configs set, so the dummy graph runs the MoE
    # combine a served checkpoint runs, not the reference's null default.
    "dispatch_chunk_size": 16384,
    "expert_hidden": 1280, "dense_ffn_hidden": 11520,
    "expert_dim": 1024, "expert_layers": 16, "expert_heads": 16,
    "expert_head_dim": 64, "trunk_dim": 3072, "max_action_tokens": 64,
    "expert_attention": "flex_dynamic",
    "action_chunk": 24, "action_frames_per_token": 4, "latents_per_chunk": 6,
    "window_frames": 93, "n_chunks": 4, "grid_hz": 15.0,
    "latent_channels": 48, "vae_stride": (4, 16, 16), "patch_size": (1, 2, 2),
    "text_model": "Qwen3.5-2B", "text_hidden_dim": 2048, "text_hidden_layer": -3,
    "antialias": True,
}


class Praxis1ConfigError(ValueError):
    """A checkpoint this build does not serve."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise Praxis1ConfigError(f"config section {name!r} must be an object")
    return value


def _moe_fields(moe: Mapping[str, Any]) -> dict[str, Any]:
    """The MoE widths, with the routed counts held apart from the totals.

    ``backbone.moe`` states the same two quantities twice, and this is the
    trap the whole function exists for. ``num_experts`` and ``top_k`` are 129
    and 9 because they *count the shared expert*, which is always on and which
    the router therefore never picks. ``num_routed_experts`` and
    ``top_k_routed`` are 128 and 8, and they are the only pair
    :class:`~mstar.model.praxis1.components.moe.MoEFeedForward` can be built
    from, exactly as the reference builds its own layer from them.

    Handing the totals to the layer instead is the failure this guards. It
    does not raise: a 129-way router picking nine experts renormalizes nine
    gates to one and returns the right shape. Every token simply gets one
    routed expert too many, and the 129th bank, which the checkpoint never
    filled because the shared expert lives in its own tensors, is routed to as
    if it held weights. The output stays plausible and the policy is wrong.

    So the totals are deliberately *not* called ``num_experts`` and ``top_k``
    on :class:`Praxis1Config`. A caller reaching for the obvious short name
    gets an :exc:`AttributeError` instead of a number that is off by one, and
    the translation to the layer's own argument names happens once, in
    :meth:`Praxis1Config.moe_kwargs`.

    The totals are redundant, so a config may omit them, but if it states them
    they have to be the routed counts plus the one shared expert. A config
    where they differ by anything else is not an off-by-one we can absorb: it
    means one of the four fields does not mean what is assumed here.
    """
    if "num_routed_experts" not in moe or "top_k_routed" not in moe:
        raise Praxis1ConfigError(
            "backbone.moe must state num_routed_experts and top_k_routed; the "
            "shared-expert-inclusive num_experts and top_k cannot be used to "
            "build the router, since neither says how many of their count the "
            "router actually picks"
        )
    num_routed = int(moe["num_routed_experts"])
    top_k_routed = int(moe["top_k_routed"])

    if not bool(moe.get("shared_expert", True)):
        raise Praxis1ConfigError(
            "backbone.moe.shared_expert is false; this build's MoE always adds an "
            "ungated shared expert and cannot turn it off, so it would add a "
            "contribution this checkpoint was not trained with"
        )

    num_total = int(moe.get("num_experts", num_routed + 1))
    top_k_with_shared = int(moe.get("top_k", top_k_routed + 1))
    if (num_total, top_k_with_shared) != (num_routed + 1, top_k_routed + 1):
        raise Praxis1ConfigError(
            f"backbone.moe has num_experts {num_total} and top_k {top_k_with_shared}, "
            f"which are not {num_routed} routed experts and {top_k_routed} picks plus "
            "the one shared expert; the two pairs may differ only by that expert"
        )
    if not 0 < top_k_routed <= num_routed:
        raise Praxis1ConfigError(
            f"backbone.moe picks {top_k_routed} of {num_routed} routed experts"
        )

    for key, supported in (
        ("router_activation", SUPPORTED_ROUTER_ACTIVATION),
        ("router_dtype", SUPPORTED_ROUTER_DTYPE),
        ("routed_weight_layout", SUPPORTED_ROUTED_WEIGHT_LAYOUT),
    ):
        value = moe.get(key, supported)
        if value != supported:
            raise Praxis1ConfigError(
                f"backbone.moe.{key} is {value!r}; this build implements only "
                f"{supported!r} and the choice is wired into the layer rather than "
                "read from here"
            )

    # The chunked dispatch is a different combine from the unchunked one, not
    # a memory split of it, and the layer picks between the two on this field
    # being set, exactly as the reference does. So it is passed through
    # untouched rather than clamped or defaulted: every published config sets
    # 16384, and a reader that dropped it would serve those weights on the
    # combine they were not run with. See "Two dispatch regimes" in
    # :mod:`mstar.model.praxis1.components.moe`.
    #
    # The validation is the reference's ``_positive_int``. A bool is refused
    # although ``True`` is an int, because a chunk of one token is a strange
    # thing to mean by ``true``; a float or a string is refused rather than
    # truncated, because ``16384.5`` is not a chunk size anyone wrote.
    chunk = moe.get("dispatch_chunk_size")
    if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0):
        raise Praxis1ConfigError(
            f"backbone.moe.dispatch_chunk_size must be a positive int or null, got {chunk!r}"
        )

    return {
        "num_experts_with_shared": num_total,
        "top_k_with_shared": top_k_with_shared,
        "num_routed_experts": num_routed,
        "top_k_routed": top_k_routed,
        "expert_hidden": int(moe.get("expert_hidden", 0)),
        "routed_scaling_factor": float(moe.get("routed_scaling_factor", 2.5)),
        "dispatch_chunk_size": chunk,
    }


@dataclass(frozen=True)
class Praxis1Config:
    """What the served graph needs from a checkpoint."""

    embodiment: str
    views: tuple[str, ...]
    action_dim: int

    # backbone trunk
    dim: int
    timestep_dim: int
    num_registers: int
    encoder_depth: int
    decoder_depth: int
    tap_blocks: tuple[int, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    flex_block_size: int
    attention_backend: str

    # backbone MoE. There is no ``num_experts`` and no ``top_k`` here on
    # purpose: both names are ambiguous between the routed count and the count
    # that includes the always-on shared expert, and the checkpoint uses them
    # for the second. :func:`_moe_fields` says what breaks when the two are
    # swapped; :meth:`moe_kwargs` is the only place either is read.
    num_experts_with_shared: int
    top_k_with_shared: int
    num_routed_experts: int
    top_k_routed: int
    expert_hidden: int
    routed_scaling_factor: float
    # Selects the MoE combine on ``is not None``, not a memory knob: see
    # :func:`_moe_fields`.
    dispatch_chunk_size: int | None
    dense_ffn_hidden: int

    # action expert
    expert_dim: int
    expert_layers: int
    expert_heads: int
    expert_head_dim: int
    trunk_dim: int
    proprio_patch_dim: int
    action_patch_dim: int
    max_action_tokens: int
    expert_attention: str

    # chunk geometry
    action_chunk: int
    action_frames_per_token: int
    latents_per_chunk: int
    window_frames: int
    n_chunks: int
    grid_hz: float

    # canvas and VAE
    canvas_height: int
    canvas_width: int
    tile_height: int
    tile_width: int
    # Which canvas rule the checkpoint was trained with. The two crop to
    # different aspects and round in different places, so this is not a
    # quality knob: it selects the observation the weights expect.
    antialias: bool
    latent_channels: int
    vae_stride: tuple[int, int, int]
    patch_size: tuple[int, int, int]

    # text encoder
    text_model: str
    text_hidden_dim: int
    text_hidden_layer: int

    normalization: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def action_horizon(self) -> int:
        """Command rows in one answer: what a client gets per policy call."""
        return self.action_chunk

    @property
    def action_tokens(self) -> int:
        return self.action_chunk // self.action_frames_per_token

    @property
    def num_cameras(self) -> int:
        return len(self.views)

    @property
    def latent_height(self) -> int:
        return self.canvas_height // self.vae_stride[1]

    @property
    def latent_width(self) -> int:
        return self.canvas_width // self.vae_stride[2]

    def moe_kwargs(self) -> dict[str, Any]:
        """The arguments :class:`...components.moe.MoEFeedForward` is built with.

        The layer's ``num_experts`` and ``top_k`` are the *routed* counts, which
        is the reference's own naming for its own layer, and the reference feeds
        them from ``num_routed_experts`` and ``top_k_routed`` too. Those two
        namings meet here and nowhere else, so a block builder never picks
        between the config's two pairs and there is one place to read when a
        routed count looks wrong.

        Returning kwargs rather than letting a caller spell out five arguments
        is the point: a positional call is what silently puts the right number
        in the wrong slot, and it is the fifth argument,
        ``routed_scaling_factor``, that would be hardest to notice, since 2.5
        is also the layer's default. ``dispatch_chunk_size`` is the same
        trap one argument later: the layer's default is ``None``, which is
        the other combine, so a builder that spelled out five arguments and
        forgot the sixth would serve every published checkpoint on a regime
        none of them selects, and nothing would raise.
        """
        return {
            "dim": self.dim,
            "hidden": self.expert_hidden,
            "num_experts": self.num_routed_experts,
            "top_k": self.top_k_routed,
            "routed_scaling_factor": self.routed_scaling_factor,
            "dispatch_chunk_size": self.dispatch_chunk_size,
        }

    @classmethod
    def for_rig(cls, rig: str) -> "Praxis1Config":
        """The published checkpoint's shape, without reading one.

        Dummy mode only: the startup checks build a model with no checkpoint.
        A served model reads the real config.json, which is the only thing a
        request ever runs against.
        """
        if rig not in RIG_DEFAULTS:
            raise Praxis1ConfigError(f"no rig defaults for {rig!r}")
        rig_defaults = RIG_DEFAULTS[rig]
        canvas, tile = rig_defaults["canvas"], rig_defaults["tile"]
        action_dim = rig_defaults["action_dim"]
        return cls(
            embodiment=rig,
            views=RIG_VIEWS[rig],
            action_dim=action_dim,
            proprio_patch_dim=action_dim,
            action_patch_dim=action_dim * _SHARED_DEFAULTS["action_frames_per_token"],
            canvas_height=canvas[0], canvas_width=canvas[1],
            tile_height=tile[0], tile_width=tile[1],
            **_SHARED_DEFAULTS,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "Praxis1Config":
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            raise Praxis1ConfigError(f"no config.json in the checkpoint at {path}")
        return cls.from_dict(json.loads(config_path.read_text()))

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "Praxis1Config":
        embodiment = config.get("embodiment")
        if embodiment == BASE_EMBODIMENT:
            raise Praxis1ConfigError(
                "this is a base checkpoint: it has no action heads and cannot be served"
            )
        if embodiment not in RIG_VIEWS:
            raise Praxis1ConfigError(
                f"embodiment {embodiment!r} is not served; this build has "
                f"{', '.join(sorted(RIG_VIEWS))}"
            )
        embodiment = str(embodiment)

        backbone = _section(config, "backbone")
        layout = backbone.get("layout", SUPPORTED_LAYOUT)
        if layout != SUPPORTED_LAYOUT:
            raise Praxis1ConfigError(f"backbone layout {layout!r} is not served")

        encoder = _section(backbone, "encoder")
        decoder = _section(backbone, "decoder")
        attention = _section(backbone, "attention")
        moe = _section(backbone, "moe")
        expert = _section(config, "action_expert")
        chunking = _section(config, "chunking")
        video = _section(config, "video")
        text = _section(config, "text_encoder")

        canvas = video.get("canvas", SUPPORTED_CANVAS)
        if canvas != SUPPORTED_CANVAS:
            raise Praxis1ConfigError(f"video canvas {canvas!r} is not served")
        vae = video.get("vae", SUPPORTED_VAE)
        if vae != SUPPORTED_VAE:
            raise Praxis1ConfigError(f"video VAE {vae!r} is not served")

        views = tuple(video.get("views", RIG_VIEWS[embodiment]))
        if views != RIG_VIEWS[embodiment]:
            raise Praxis1ConfigError(
                f"the {embodiment} rig has the views {list(RIG_VIEWS[embodiment])}, got {list(views)}"
            )

        action_dim = int(_section(config, "action_space").get("dim", RIG_ACTION_DIMS[embodiment]))
        if action_dim != RIG_ACTION_DIMS[embodiment]:
            raise Praxis1ConfigError(
                f"the {embodiment} rig has action rows of {RIG_ACTION_DIMS[embodiment]}, "
                f"got {action_dim}"
            )

        action_chunk = int(chunking.get("action_chunk", 24))
        frames_per_token = int(chunking.get("action_frames_per_token", 4))
        if action_chunk % frames_per_token:
            raise Praxis1ConfigError(
                f"action_chunk {action_chunk} is not a whole number of tokens of "
                f"{frames_per_token} frames"
            )
        # The head's projections carry both widths, so a config that disagrees
        # with them would load and then answer rows of the wrong shape.
        patch = int(expert.get("action_patch_dim", action_dim * frames_per_token))
        proprio = int(expert.get("proprio_patch_dim", action_dim))
        if patch != action_dim * frames_per_token or proprio != action_dim:
            raise Praxis1ConfigError(
                f"action rows of {action_dim} need proprio_patch_dim {action_dim} and "
                f"action_patch_dim {action_dim * frames_per_token}, got {proprio} and {patch}"
            )

        stride = video.get("vae_stride", (4, 16, 16))
        patch_size = video.get("patch_size", (1, 2, 2))

        return cls(
            embodiment=embodiment,
            views=views,
            action_dim=action_dim,
            dim=int(backbone["dim"]),
            timestep_dim=int(backbone["timestep_dim"]),
            num_registers=int(backbone.get("num_registers", 0)),
            encoder_depth=int(encoder["depth"]),
            decoder_depth=int(decoder["depth"]),
            tap_blocks=tuple(int(b) for b in encoder.get("tap_blocks", ())),
            num_heads=int(attention["num_heads"]),
            num_kv_heads=int(attention.get("num_kv_heads", attention["num_heads"])),
            head_dim=int(attention["head_dim"]),
            flex_block_size=int(attention.get("flex_block_size", 128)),
            attention_backend=str(attention.get("backend", "flash")),
            **_moe_fields(moe),
            dense_ffn_hidden=int(backbone.get("dense_ffn_hidden", 0)),
            expert_dim=int(expert["dim"]),
            expert_layers=int(expert["num_layers"]),
            expert_heads=int(expert["num_heads"]),
            expert_head_dim=int(expert["head_dim"]),
            trunk_dim=int(expert.get("trunk_dim", backbone["dim"])),
            proprio_patch_dim=proprio,
            action_patch_dim=patch,
            max_action_tokens=int(expert.get("max_action_tokens", 64)),
            expert_attention=str(expert.get("attention", "flex_dynamic")),
            action_chunk=action_chunk,
            action_frames_per_token=frames_per_token,
            latents_per_chunk=int(chunking.get("latents_per_chunk", 6)),
            window_frames=int(chunking.get("window_frames", 93)),
            n_chunks=int(chunking.get("n_chunks", 4)),
            grid_hz=float(chunking.get("grid_hz", 15.0)),
            antialias=bool(video.get("antialias", True)),
            canvas_height=int(video["height"]),
            canvas_width=int(video["width"]),
            tile_height=int(video.get("tile_height", video["height"])),
            tile_width=int(video.get("tile_width", video["width"])),
            latent_channels=int(video.get("latent_channels", 48)),
            vae_stride=(int(stride[0]), int(stride[1]), int(stride[2])),
            patch_size=(int(patch_size[0]), int(patch_size[1]), int(patch_size[2])),
            text_model=str(text.get("model", "Qwen3.5-2B")),
            text_hidden_dim=int(text.get("hidden_dim", 2048)),
            text_hidden_layer=int(text.get("hidden_layer", -3)),
            normalization=_section(config, "normalization"),
            raw=dict(config),
        )
