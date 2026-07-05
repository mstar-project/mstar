"""Joint-sequence packing for Cosmos3 generation (ported from the diffusers
``Cosmos3OmniPipeline``).

Pure, stateless primitives that turn a prompt + latent shape into the
transformer's per-step inputs: the 3D interleaved mRoPE position ids, the
text/vision segment layouts, and the chat-template tokenization. Shared by the
t2i pipeline and the engine submodule's input preprocessing. Reproduces the
diffusers pipeline's packed t2i inputs byte-for-byte.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# ---------------------------------------------------------------------------
# 3D interleaved mRoPE position ids (exact ports of the pipeline helpers).
# ---------------------------------------------------------------------------


def get_3d_mrope_ids_text_tokens(
    num_tokens: int, temporal_offset: int | float, use_float_positions: bool = False
) -> tuple[torch.Tensor, int | float]:
    """Text tokens: all three axes share the same increasing ids from ``temporal_offset``."""
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()  # [3, num_tokens]
    return mrope_ids, temporal_offset + num_tokens


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """Vision/sound (VAE) tokens: (t, h, w) grid ids, with optional fps modulation
    of the temporal axis (only when ``fps`` is set and ``grid_t > 1``)."""
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_temporal_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_temporal_offset


def get_3d_mrope_ids_action_tokens(
    grid_t: int,
    temporal_offset: int | float,
    action_fps: float | None,
    base_fps: float = 24.0,
    base_temporal_compression_factor: int = 4,
    start_frame_offset: int = 1,
) -> tuple[torch.Tensor, int | float]:
    """Action tokens: a frame-rate ``(T, 1, 1)`` temporal grid sharing the media
    offset with the vision band. The action stream is uncompressed in time
    (``temporal_compression_factor=1``) but its rate is expressed in the same
    base-fps units as the vision latents (``base_temporal_compression_factor``),
    so an action chunk lines up temporally with the conditioning video."""
    return get_3d_mrope_ids_vae_tokens(
        grid_t=grid_t,
        grid_h=1,
        grid_w=1,
        temporal_offset=temporal_offset,
        reset_spatial_indices=True,
        fps=action_fps,
        base_fps=base_fps,
        temporal_compression_factor=1,
        base_temporal_compression_factor=base_temporal_compression_factor,
        start_frame_offset=start_frame_offset,
    )


# ---------------------------------------------------------------------------
# Action conditioning layout (ported from vllm-omni ``action.py``). Each mode
# fixes which latent video frames and which action tokens are clean context vs
# noisy/predicted:
#   * forward_dynamics  -- action is the condition (all clean); video frame 0 is
#                          clean, the rest are predicted.
#   * inverse_dynamics  -- video is the condition (all latent frames clean);
#                          every action token is predicted.
#   * policy            -- video frame 0 is clean (the rest predicted) and every
#                          action token is predicted.
# ---------------------------------------------------------------------------

ACTION_MODE_FORWARD_DYNAMICS = "forward_dynamics"
ACTION_MODE_INVERSE_DYNAMICS = "inverse_dynamics"
ACTION_MODE_POLICY = "policy"
ACTION_MODES = (ACTION_MODE_FORWARD_DYNAMICS, ACTION_MODE_INVERSE_DYNAMICS, ACTION_MODE_POLICY)


def action_condition_frame_indexes(mode: str, action_length: int) -> list[int]:
    """Clean (conditioning) action tokens for ``mode``."""
    if mode == ACTION_MODE_FORWARD_DYNAMICS:
        return list(range(action_length))
    if mode in (ACTION_MODE_INVERSE_DYNAMICS, ACTION_MODE_POLICY):
        return []
    raise ValueError(f"Unknown Cosmos3 action mode: {mode!r}")


def vision_condition_frame_indexes(mode: str, latent_frames: int) -> list[int]:
    """Clean (conditioning) latent video frames for ``mode``."""
    if mode in (ACTION_MODE_FORWARD_DYNAMICS, ACTION_MODE_POLICY):
        return [0]
    if mode == ACTION_MODE_INVERSE_DYNAMICS:
        return list(range(latent_frames))
    raise ValueError(f"Unknown Cosmos3 action mode: {mode!r}")


def action_start_frame_offset(action_length: int, video_length: int) -> int:
    """mRoPE start-frame offset for the action band: action chunks of length
    ``num_frames - 1`` start one frame in (aligned to predicted frames 1..N);
    a full ``num_frames`` chunk starts at 0."""
    if action_length == video_length - 1:
        return 1
    if action_length == video_length:
        return 0
    raise ValueError(
        "Cosmos3 action_chunk_size must equal num_frames - 1 or num_frames; "
        f"got action_chunk_size={action_length}, num_frames={video_length}."
    )


# ---------------------------------------------------------------------------
# Prompt tokenization — ported from pipeline.tokenize_prompt. Image mode
# (num_frames == 1) and video mode differ only in the system prompt and the
# metadata sentences appended to the prompt (resolution always; duration for
# video). Both append the eos + start-of-generation special tokens.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
INVERSE_IMAGE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."
VIDEO_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
INVERSE_VIDEO_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."
DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."


def _append(base: str, addition: str) -> str:
    base = base.rstrip(".")
    return f"{base}. {addition}" if base else addition


def tokenize_prompt(
    tokenizer,
    prompt: str,
    negative_prompt: str | None,
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
    use_system_prompt: bool = True,
    add_resolution_template: bool = True,
    add_duration_template: bool = True,
) -> tuple[list[int], list[int]]:
    """Return ``(cond_input_ids, uncond_input_ids)`` for image/video generation.

    Mirrors the diffusers pipeline: apply the Qwen2 chat template with the
    mode-specific system prompt and metadata sentences (duration for video, then
    resolution), using inverse templates for the negative prompt, then append the
    eos + start-of-generation (``<|vision_start|>``) special tokens. Image mode is
    ``num_frames == 1``.
    """
    is_image = num_frames == 1
    if negative_prompt is None:
        negative_prompt = ""
    eos_id = tokenizer.eos_token_id
    sog_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

    resolution_template = IMAGE_RESOLUTION_TEMPLATE if is_image else VIDEO_RESOLUTION_TEMPLATE
    inverse_resolution_template = (
        INVERSE_IMAGE_RESOLUTION_TEMPLATE if is_image else INVERSE_VIDEO_RESOLUTION_TEMPLATE
    )

    def _apply_templates(text: str, is_negative: bool) -> str:
        if not is_image and add_duration_template:
            tmpl = INVERSE_DURATION_TEMPLATE if is_negative else DURATION_TEMPLATE
            text = _append(text, tmpl.format(duration=num_frames / fps, fps=fps))
        if add_resolution_template:
            tmpl = inverse_resolution_template if is_negative else resolution_template
            text = _append(text, tmpl.format(height=height, width=width))
        return text

    def _tokenize(text: str) -> list[int]:
        conversations = []
        if use_system_prompt:
            conversations.append(
                {"role": "system", "content": SYSTEM_PROMPT_IMAGE if is_image else SYSTEM_PROMPT_VIDEO}
            )
        conversations.append({"role": "user", "content": text})
        enc = tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=True
        )
        return list(enc["input_ids"]) + [eos_id, sog_id]

    cond = _tokenize(_apply_templates(prompt, is_negative=False))
    uncond = _tokenize(_apply_templates(negative_prompt, is_negative=True))
    return cond, uncond


# ---------------------------------------------------------------------------
# Segment builders + full t2i static-input assembly.
# ---------------------------------------------------------------------------


def build_text_segment(input_ids: list[int], config, device) -> dict[str, Any]:
    und_len = len(input_ids)
    text_mrope_ids, next_off = get_3d_mrope_ids_text_tokens(
        num_tokens=und_len, temporal_offset=0, use_float_positions=config.enable_fps_modulation
    )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
        "und_len": und_len,
        "text_mrope_ids": text_mrope_ids.to(device),
        "vision_start_temporal_offset": next_off + config.unified_3d_mrope_temporal_modality_margin,
    }


def build_vision_segment(
    latent_shape: tuple[int, int, int, int, int],
    has_image_condition: bool,
    mrope_offset: int | float,
    vision_fps: float | None,
    curr: int,
    config,
    vae_scale_factor_temporal: int,
    device,
    noisy_frames: list[int] | None = None,
) -> dict[str, Any]:
    """``latent_shape`` is the vision latent tensor shape ``[B, C, T, H, W]``.

    ``noisy_frames`` lists the latent frames that are noisy (predicted); the rest
    are clean conditioning context. When ``None`` it defaults to frame 0 clean
    if ``has_image_condition`` else all frames noisy — i.e. the t2i/t2v/i2v
    layouts. Action modes pass an explicit list (e.g. ``[]`` for
    inverse-dynamics, where the whole video is conditioning)."""
    p = config.latent_patch_size
    _, _, latent_t, latent_h, latent_w = latent_shape
    patch_h = math.ceil(latent_h / p)
    patch_w = math.ceil(latent_w / p)
    num_vision_tokens = latent_t * patch_h * patch_w

    if noisy_frames is None:
        noisy_start = 1 if has_image_condition else 0
        noisy_list = list(range(noisy_start, latent_t))
    else:
        noisy_list = sorted(noisy_frames)
    noisy_frame_indexes = torch.tensor(noisy_list, device=device, dtype=torch.long)

    frame_token_stride = patch_h * patch_w
    mse_loss_indexes: list[int] = []
    for frame_idx in noisy_list:
        frame_start = curr + frame_idx * frame_token_stride
        mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

    effective_fps = vision_fps if config.enable_fps_modulation else None
    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=mrope_offset,
        reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
        fps=effective_fps,
        base_fps=float(config.base_fps),
        temporal_compression_factor=vae_scale_factor_temporal,
    )

    return {
        "vision_token_shapes": [(latent_t, patch_h, patch_w)],
        "vision_sequence_indexes": torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
        "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
        "vision_noisy_frame_indexes": [noisy_frame_indexes],
        "vision_mrope_ids": vision_mrope_ids.to(device),
        "num_vision_tokens": num_vision_tokens,
        "num_noisy_vision_tokens": len(noisy_list) * frame_token_stride,
    }


def build_static_inputs(
    input_ids: list[int],
    latent_shape: tuple[int, int, int, int, int],
    config,
    vae_scale_factor_temporal: int,
    fps: float,
    device,
    has_image_condition: bool = False,
) -> dict[str, Any]:
    """Assemble the per-prompt static transformer inputs for image/video
    generation. ``latent_shape`` is ``[B, C, T, H, W]`` (``T == 1`` for images;
    ``T == 1 + (num_frames - 1) // temporal_factor`` for video). When
    ``has_image_condition`` is set, latent frame 0 is a clean conditioning anchor
    (image-to-video): it stays in the sequence but is excluded from the noisy /
    predicted frames. Step-varying fields (``vision_tokens``,
    ``vision_timesteps``) are spliced in per denoising step by the caller."""
    text = build_text_segment(input_ids, config, device)
    vision = build_vision_segment(
        latent_shape=latent_shape,
        has_image_condition=has_image_condition,
        mrope_offset=text["vision_start_temporal_offset"],
        vision_fps=fps,
        curr=text["und_len"],
        config=config,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        device=device,
    )
    position_ids = torch.cat([text["text_mrope_ids"], vision["vision_mrope_ids"]], dim=1)
    return {
        **text,
        **vision,
        "position_ids": position_ids,
        "sequence_length": text["und_len"] + vision["num_vision_tokens"],
    }


def build_t2i_static_inputs(
    input_ids: list[int],
    latent_shape: tuple[int, int, int, int, int],
    config,
    vae_scale_factor_temporal: int,
    fps: float,
    device,
) -> dict[str, Any]:
    """Image-mode convenience wrapper around :func:`build_static_inputs`."""
    return build_static_inputs(
        input_ids, latent_shape, config, vae_scale_factor_temporal, fps, device,
        has_image_condition=False,
    )


def build_action_static_inputs(
    input_ids: list[int],
    video_latent_shape: tuple[int, int, int, int, int],
    action_chunk_size: int,
    mode: str,
    config,
    vae_scale_factor_temporal: int,
    fps: float,
    action_fps: float,
    action_start_offset: int,
    device,
) -> dict[str, Any]:
    """Assemble the static transformer inputs for joint video+action generation.

    The generation sequence is ``[video tokens | action tokens]`` after the text
    prefix. Both media bands share the post-text temporal offset (the 15000
    margin), with the action band on its own frame-rate grid. Conditioning per
    ``mode`` decides which video frames and action tokens are clean context vs
    noisy/predicted (see :func:`vision_condition_frame_indexes` /
    :func:`action_condition_frame_indexes`)."""
    text = build_text_segment(input_ids, config, device)
    media_offset = text["vision_start_temporal_offset"]
    _, _, latent_t, _, _ = video_latent_shape

    vision_clean = set(vision_condition_frame_indexes(mode, latent_t))
    vision_noisy = [f for f in range(latent_t) if f not in vision_clean]
    vision = build_vision_segment(
        latent_shape=video_latent_shape,
        has_image_condition=False,
        mrope_offset=media_offset,
        vision_fps=fps,
        curr=text["und_len"],
        config=config,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        device=device,
        noisy_frames=vision_noisy,
    )

    curr = text["und_len"] + vision["num_vision_tokens"]
    action_clean = set(action_condition_frame_indexes(mode, action_chunk_size))
    action_noisy = [a for a in range(action_chunk_size) if a not in action_clean]
    effective_action_fps = action_fps if config.enable_fps_modulation else None
    action_mrope_ids, _ = get_3d_mrope_ids_action_tokens(
        grid_t=action_chunk_size,
        temporal_offset=media_offset,
        action_fps=effective_action_fps,
        base_fps=float(config.base_fps),
        base_temporal_compression_factor=vae_scale_factor_temporal,
        start_frame_offset=action_start_offset,
    )

    parts = [text["text_mrope_ids"], vision["vision_mrope_ids"], action_mrope_ids.to(device)]
    pos_dtype = torch.float32 if any(p.is_floating_point() for p in parts) else torch.long
    position_ids = torch.cat([p.to(pos_dtype) for p in parts], dim=1)

    return {
        **text,
        **vision,
        "action_token_shapes": [(action_chunk_size, 1, 1)],
        "action_sequence_indexes": torch.arange(curr, curr + action_chunk_size, dtype=torch.long, device=device),
        "action_noisy_frame_indexes": [torch.tensor(action_noisy, dtype=torch.long, device=device)],
        "action_mse_loss_indexes": torch.tensor(
            [curr + a for a in action_noisy], dtype=torch.long, device=device
        ),
        "action_mrope_ids": action_mrope_ids.to(device),
        "num_action_tokens": action_chunk_size,
        "num_noisy_action_tokens": len(action_noisy),
        "action_mode": mode,
        "position_ids": position_ids,
        "sequence_length": curr + action_chunk_size,
    }
