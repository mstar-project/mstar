import logging
import math
import pathlib
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mstar.model.waypoint.checkpoint import TAEHV_UPSTREAM_ARCHIVE

logger = logging.getLogger(__name__)

# The AE runs on a fixed grid per source resolution: a 720p frame is resized to
# 512x1024 on the way in and the decode is resized back out.
_ENCODE_SIZES = {(720, 1280): (512, 1024), (360, 640): (256, 512)}
_DECODE_SIZES = {v: k for k, v in _ENCODE_SIZES.items()}

# ``TAEHV.__init__`` reads patch_size and latent_channels off the checkpoint
# *filename*: "taehv1_5" means (2, 32), anything else falls back to (1, 16) and
# the state_dict load then fails on shape. Pinned, not globbed.
_CHECKPOINT_NAME = "taehv1_5.pth"

DECODER_HISTORY_PREFIX = "decoder_history_"


def encoded_size_for_latent(latent_height: int, latent_width: int) -> tuple[int, int]:
    encoded = (int(latent_height) * 16, int(latent_width) * 16)
    if encoded not in _DECODE_SIZES:
        raise ValueError(f"unsupported Waypoint TAEHV latent grid {latent_height}x{latent_width}")
    return encoded


def pixel_size_for_latent(latent_height: int, latent_width: int) -> tuple[int, int]:
    return _DECODE_SIZES[encoded_size_for_latent(latent_height, latent_width)]


def load_taehv(ae_uri: str, cache_dir: str | None = None) -> nn.Module:
    """Load the shared weights module from a resolved local checkpoint.

    Hub resolution belongs to ``checkpoint.resolve_taehv_checkpoint`` so only
    the required file is downloaded and missing artifacts fail before allocation.
    ``cache_dir`` remains accepted for compatibility with existing direct
    callers, but is intentionally unused here.
    """
    del cache_dir
    try:
        from taehv import TAEHV
    except ImportError as exc:
        raise RuntimeError(
            "Waypoint requires the pinned TAEHV implementation in addition to "
            "the index-safe `.[waypoint]` extra. Install it separately with "
            "`uv pip install --no-deps "
            f"'taehv @ {TAEHV_UPSTREAM_ARCHIVE}'`; use uv>=0.4.0 or pip>=24.3."
        ) from exc

    base = pathlib.Path(ae_uri)
    checkpoint = base if base.is_file() else base / _CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No TAEHV checkpoint for ae_uri={ae_uri!r}; looked for {checkpoint}."
        )
    return TAEHV(str(checkpoint)).eval()


def _block_kind(block: nn.Module) -> str:
    """Return the three stateful TAEHV block kinds without importing taehv.

    The optional dependency must stay deferred until weights are requested.
    These class names are part of the pinned upstream revision and are checked
    again by :func:`validate_taehv_architecture` at model startup.
    """
    return type(block).__name__


def validate_taehv_architecture(ae_model: nn.Module) -> None:
    """Validate the fixed architecture the captured functional path supports."""
    encoder_kinds = [_block_kind(block) for block in ae_model.encoder]
    decoder_kinds = [_block_kind(block) for block in ae_model.decoder]
    facts = {
        "patch_size": getattr(ae_model, "patch_size", None),
        "latent_channels": getattr(ae_model, "latent_channels", None),
        "t_downscale": getattr(ae_model, "t_downscale", None),
        "t_upscale": getattr(ae_model, "t_upscale", None),
        "frames_to_trim": getattr(ae_model, "frames_to_trim", None),
        "encoder_memblocks": encoder_kinds.count("MemBlock"),
        "decoder_memblocks": decoder_kinds.count("MemBlock"),
    }
    expected = {
        "patch_size": 2,
        "latent_channels": 32,
        "t_downscale": 4,
        "t_upscale": 4,
        "frames_to_trim": 3,
        "encoder_memblocks": 9,
        "decoder_memblocks": 9,
    }
    mismatches = {
        key: (facts[key], value) for key, value in expected.items()
        if facts[key] != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(
            "The installed TAEHV/checkpoint architecture is not Waypoint-1.5 "
            f"compatible: {detail}. Install the pinned TAEHV source from "
            f"{TAEHV_UPSTREAM_ARCHIVE} separately and use {_CHECKPOINT_NAME}."
        )


def _apply_encoder_sequence(model: nn.Sequential, frames: Tensor) -> Tensor:
    """Pinned TAEHV's sequential encoder with all state local to this call.

    ``frames`` is NTCHW after pixel unshuffle. The four-frame seed is a fixed
    clip, so temporal-pool queues and MemBlock histories can be represented as
    local tensor lists and no encoder state survives the prime.
    """
    sequence = list(frames.unbind(1))
    for block in model:
        kind = _block_kind(block)
        if kind == "MemBlock":
            past = torch.zeros_like(sequence[0])
            next_sequence = []
            for current in sequence:
                next_sequence.append(block(current, past))
                past = current
            sequence = next_sequence
        elif kind == "TPool":
            stride = int(block.stride)
            if len(sequence) % stride:
                raise ValueError(
                    f"encoder sequence length {len(sequence)} is not divisible by "
                    f"TPool stride {stride}"
                )
            next_sequence = []
            for start in range(0, len(sequence), stride):
                chunk = sequence[start : start + stride]
                batch, channels, height, width = chunk[0].shape
                joined = torch.cat(chunk, dim=1).view(
                    batch * stride, channels, height, width
                )
                next_sequence.append(block(joined))
            sequence = next_sequence
        elif kind == "TGrow":
            raise ValueError("TAEHV encoder unexpectedly contains TGrow")
        else:
            sequence = [block(value) for value in sequence]
    return torch.stack(sequence, dim=1)


def encode_seed_clip(
    ae_model: nn.Module,
    frames: Tensor,
    *,
    output_size: tuple[int, int],
) -> Tensor:
    """Encode one fixed four-frame RGB clip into one latent tensor.

    The operation is pure with respect to ``ae_model``: only its parameters are
    read, and every temporal history is created and consumed within this call.
    """
    expected = int(ae_model.t_downscale)
    if frames.ndim != 4 or frames.shape[0] != expected or frames.shape[-1] != 3:
        raise ValueError(
            f"expected [{expected}, H, W, 3] RGB frames, got {tuple(frames.shape)}"
        )
    rgb = frames.unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()
    rgb = F.interpolate(
        rgb[0], size=output_size, mode="bilinear", align_corners=False
    )[None]
    rgb = ae_model.preprocess_input_frames(rgb)
    latent = _apply_encoder_sequence(ae_model.encoder, rgb)
    if latent.shape[1] != 1:
        raise RuntimeError(
            f"four-frame TAEHV prime produced {latent.shape[1]} latents, expected one"
        )
    return latent.squeeze(1)


def _conv_output_size(size: int, block: nn.Conv2d, dim: int) -> int:
    kernel = block.kernel_size[dim]
    stride = block.stride[dim]
    padding = block.padding[dim]
    dilation = block.dilation[dim]
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def decoder_history_shapes(
    ae_model: nn.Module,
    *,
    batch_size: int,
    latent_height: int,
    latent_width: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """The nine fixed MemBlock input shapes for one decoder request."""
    validate_taehv_architecture(ae_model)
    height, width = int(latent_height), int(latent_width)
    channels = int(ae_model.latent_channels)
    shapes: list[tuple[int, int, int, int]] = []
    for block in ae_model.decoder:
        kind = _block_kind(block)
        if kind == "MemBlock":
            block_channels = int(block.conv[0].in_channels // 2)
            if channels != block_channels:
                raise ValueError(
                    f"decoder MemBlock expects {block_channels} channels after "
                    f"a {channels}-channel block"
                )
            shapes.append((batch_size, channels, height, width))
        elif kind == "TGrow":
            continue
        elif isinstance(block, nn.Conv2d):
            height = _conv_output_size(height, block, 0)
            width = _conv_output_size(width, block, 1)
            channels = int(block.out_channels)
        elif isinstance(block, nn.Upsample):
            scale = block.scale_factor
            scale_h, scale_w = (
                (float(scale), float(scale))
                if isinstance(scale, (int, float)) else map(float, scale)
            )
            height, width = int(height * scale_h), int(width * scale_w)
    if len(shapes) != 9:
        raise ValueError(f"Waypoint decoder needs nine histories; derived {len(shapes)}")
    return tuple(shapes)


def initial_decoder_histories(
    ae_model: nn.Module, latent: Tensor,
) -> tuple[Tensor, ...]:
    shapes = decoder_history_shapes(
        ae_model,
        batch_size=int(latent.shape[0]),
        latent_height=int(latent.shape[-2]),
        latent_width=int(latent.shape[-1]),
    )
    return tuple(latent.new_zeros(shape) for shape in shapes)


def _apply_decoder_sequence(
    model: nn.Sequential,
    latent: Tensor,
    histories: Sequence[Tensor],
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Decode one latent while explicitly threading every MemBlock history."""
    sequence = [latent]
    next_histories: list[Tensor] = []
    history_idx = 0
    for block in model:
        kind = _block_kind(block)
        if kind == "MemBlock":
            past = histories[history_idx]
            next_sequence = []
            for current in sequence:
                next_sequence.append(block(current, past))
                past = current
            next_histories.append(past)
            history_idx += 1
            sequence = next_sequence
        elif kind == "TGrow":
            stride = int(block.stride)
            next_sequence = []
            for current in sequence:
                batch = current.shape[0]
                grown = block(current)
                next_sequence.extend(grown.view(batch, stride, *grown.shape[1:]).unbind(1))
            sequence = next_sequence
        elif kind == "TPool":
            raise ValueError("TAEHV decoder unexpectedly contains TPool")
        else:
            sequence = [block(value) for value in sequence]
    if history_idx != len(histories):
        raise ValueError(
            f"decoder consumed {history_idx} histories, received {len(histories)}"
        )
    return torch.stack(sequence, dim=1), tuple(next_histories)


def decode_latent(
    ae_model: nn.Module,
    latent: Tensor,
    histories: Sequence[Tensor],
    *,
    output_size: tuple[int, int],
    initialize: bool,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Pure decoder initialization or steady-state step.

    Initialization reproduces ``ChunkedStreamingTAEHV.decode`` exactly: the
    seed latent is fed ``frames_to_trim`` extra times, those reconstructed clips
    are discarded, and a final feed advances the state through the seed frame.
    The caller intentionally emits none of those frames.
    """
    state = tuple(histories)
    feeds = int(ae_model.frames_to_trim) + 1 if initialize else 1
    decoded = None
    for _ in range(feeds):
        decoded, state = _apply_decoder_sequence(ae_model.decoder, latent, state)
    decoded = ae_model.postprocess_output_frames(decoded)
    expected_frames = int(ae_model.t_upscale)
    if decoded.shape[1] != expected_frames:
        raise ValueError(
            "TAEHV decoder returned "
            f"{decoded.shape[1]} frames for one latent; expected {expected_frames}."
        )
    decoded = F.interpolate(
        decoded[0], size=output_size, mode="bilinear", align_corners=False
    )[None]
    frames = (decoded.clamp(0, 1) * 255).round().to(torch.uint8)
    frames = frames.squeeze(0).permute(0, 2, 3, 1)[..., :3].contiguous()
    return frames, state


class ChunkedStreamingTAEHV:
    """One streaming session: this request's frames, in order.

    The order-dependent state lives here; ``ae_model`` is shared and never
    mutated, so a fresh instance is the reference's ``reset()``. Not an
    ``nn.Module``: the module tree would give ``state_dict()``, ``to()`` and the
    weight loader a second path to the same parameters.
    """

    def __init__(
        self,
        ae_model: nn.Module,
        auto_aspect_ratio: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        height: int | None = None,
        width: int | None = None,
    ):
        from taehv import StreamingTAEHV

        self.device = device
        self.dtype = dtype
        self.auto_aspect_ratio = auto_aspect_ratio
        scale = ae_model.patch_size * 2 ** sum(
            getattr(m, "stride", None) == (2, 2) for m in ae_model.encoder
        )
        # height/width are the LATENT grid; _img_size is the pixel resolution
        # encoded from and decoded back to, or None to read it per call.
        self._img_size = (
            None if height is None else _DECODE_SIZES[(height * scale, width * scale)]
        )
        # The reference places the weights here; they are shared, so the node
        # that owns them placed them already.
        self.streaming_ae_model = StreamingTAEHV(ae_model)

    def _resize(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(x[0], size=size, mode="bilinear", align_corners=False)[None]

    @torch.inference_mode()
    def encode(self, frames: Tensor) -> Tensor:
        """``[t_downscale, H, W, 3]`` in [0, 1] -> one latent ``[B, C, h, w]``.
        Exactly that many frames: fewer buffer and return None. The reference
        scales from uint8 here; this takes it pre-scaled in the same
        cast-then-divide order, so the two agree bit for bit."""
        t = self.streaming_ae_model.taehv.t_downscale
        if frames.dim() != 4 or frames.shape[0] != t or frames.shape[-1] != 3:
            raise ValueError(
                f"expected [{t}, H, W, 3] RGB frames, got {tuple(frames.shape)}."
            )
        rgb = frames.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()
        if self.auto_aspect_ratio:
            if frames.shape[1] * 16 != frames.shape[2] * 9:
                raise ValueError(f"Expected 16:9 input, got {tuple(frames.shape[1:3])}.")
            rgb = self._resize(rgb, _ENCODE_SIZES[self._img_size or tuple(frames.shape[1:3])])
        latent = self.streaming_ae_model.encode(rgb)
        assert latent is not None, (
            f"the streaming encoder buffered {t} frames without emitting a latent"
        )
        return latent.squeeze(1)

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        """One latent ``[B, C, h, w]`` -> ``[t_upscale, H, W, 3]`` uint8.

        Single-use and order-dependent: the temporal memory advances per call,
        so a latent decoded twice, out of turn or not at all shifts every frame
        after it, silently. The first call primes that memory with
        ``frames_to_trim`` extra feeds, which is why the prime walk decodes.
        """
        if latent.dim() != 4:
            raise ValueError(f"expected a [B, C, h, w] latent, got {tuple(latent.shape)}.")
        z = latent.unsqueeze(1).to(device=self.device, dtype=self.dtype)
        if self.streaming_ae_model.n_frames_decoded == 0:
            for _ in range(self.streaming_ae_model.taehv.frames_to_trim):
                self.streaming_ae_model.decode(z)
                self.streaming_ae_model.flush_decoder()
        first = self.streaming_ae_model.decode(z)
        assert first is not None, "the streaming decoder returned no frame for a latent"
        decoded = torch.cat([first, *self.streaming_ae_model.flush_decoder()], dim=1)
        if self.auto_aspect_ratio:
            decoded = self._resize(
                decoded, self._img_size or _DECODE_SIZES[tuple(decoded.shape[-2:])]
            )
        decoded = (decoded.clamp(0, 1) * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(0, 2, 3, 1)[..., :3]
