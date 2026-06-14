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


def tokenize_t2i_prompt(
    tokenizer,
    prompt: str,
    negative_prompt: str | None,
    height: int,
    width: int,
    use_system_prompt: bool = True,
    add_resolution_template: bool = True,
) -> tuple[list[int], list[int]]:
    """Image-mode convenience wrapper around :func:`tokenize_prompt`."""
    return tokenize_prompt(
        tokenizer,
        prompt,
        negative_prompt,
        num_frames=1,
        height=height,
        width=width,
        use_system_prompt=use_system_prompt,
        add_resolution_template=add_resolution_template,
    )


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
) -> dict[str, Any]:
    """``latent_shape`` is the vision latent tensor shape ``[B, C, T, H, W]``."""
    p = config.latent_patch_size
    _, _, latent_t, latent_h, latent_w = latent_shape
    patch_h = math.ceil(latent_h / p)
    patch_w = math.ceil(latent_w / p)
    num_vision_tokens = latent_t * patch_h * patch_w

    noisy_start = 1 if has_image_condition else 0
    noisy_frame_indexes = torch.arange(noisy_start, latent_t, device=device, dtype=torch.long)

    frame_token_stride = patch_h * patch_w
    mse_loss_indexes: list[int] = []
    for frame_idx in range(noisy_start, latent_t):
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
        "num_noisy_vision_tokens": (latent_t - noisy_start) * frame_token_stride,
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
