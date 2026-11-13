"""Praxis-1 config, read from a checkpoint's ``config.json``.

The checkpoint carries the whole variant: the tile canvas, the pose frame of
the actions, the attention backend and the rig. We read the fields the served
graph needs and refuse a variant we have not ported, rather than serving a
checkpoint that runs and means something else.

The rig follows ``embodiment``: ``single_arm`` is the DROID rig (exterior and
wrist views, 10-wide action rows), ``bimanual`` is the YAM rig (top and two
wrists, 20-wide rows). One cold call answers 21 relative action rows at
15 Hz: the first of 6 action tokens contributes one row, and the other five
contribute four each. Later chunks answer 24 rows once episode serving lands.

Shapes worth having in one place, for the droid checkpoint: the canvas is one
224x416 tile per view stacked side by side (224x832), the VAE takes it to 48
channels at stride (4, 16, 16), and the backbone is a 3072-wide MoE with 14
encoder blocks over 129 experts and a 6-block decoder.
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
    "num_experts": 129, "num_routed_experts": 128, "top_k": 9,
    "expert_hidden": 1280, "dense_ffn_hidden": 11520,
    "expert_dim": 1024, "expert_layers": 16, "expert_heads": 16,
    "expert_head_dim": 64, "trunk_dim": 3072, "max_action_tokens": 64,
    "expert_attention": "flex_dynamic",
    "action_chunk": 24, "action_frames_per_token": 4, "latents_per_chunk": 6,
    "window_frames": 93, "n_chunks": 4, "grid_hz": 15.0,
    "latent_channels": 48, "vae_stride": (4, 16, 16), "patch_size": (1, 2, 2),
    "text_model": "Qwen3.5-2B", "text_hidden_dim": 2048, "text_hidden_layer": -3,
}


class Praxis1ConfigError(ValueError):
    """A checkpoint this build does not serve."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise Praxis1ConfigError(f"config section {name!r} must be an object")
    return value


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

    # backbone MoE
    num_experts: int
    num_routed_experts: int
    top_k: int
    expert_hidden: int
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
        """Rows in a cold openpi reply; the first token owns the anchor frame."""
        return 1 + (self.action_tokens - 1) * self.action_frames_per_token

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
            num_experts=int(moe.get("num_experts", 0)),
            num_routed_experts=int(moe.get("num_routed_experts", 0)),
            top_k=int(moe.get("top_k", 0)),
            expert_hidden=int(moe.get("expert_hidden", 0)),
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
