"""The Qwen3-TTS 1.7B CodePredictor's depth loop as one persistent kernel, on the GPUs it can run on."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import triton

from mstar.model.qwen3_tts.components.depth_frame import schedule
from mstar.platforms import current_platform

if TYPE_CHECKING:
    from mstar.model.qwen3_tts.components.depth_frame.frame import DepthFrame

logger = logging.getLogger(__name__)

# each per-layer weight the kernel's tiles and task lists are built for, as (rows, columns)
_SHAPES = {"qkv": (4096, 1024), "o": (1024, 2048), "gate": (3072, 1024), "down": (1024, 3072)}


def build(code_predictor, talker_table: torch.Tensor) -> DepthFrame | None:
    """A `DepthFrame` over ``code_predictor``, or None, with a warning, where the kernel can't run.

    It is built for the 1.7B CodePredictor in bfloat16 on a Hopper GPU with Triton 3.7's Gluon. Its CTAs spin on each
    other, so all of them have to be resident at once, which another process's kernels under MPS can prevent.
    """
    reason = _unsupported(code_predictor, talker_table)
    if reason is not None:
        logger.warning("depth_kernel is set, but the depth loop runs as launches: %s", reason)
        return None
    from mstar.model.qwen3_tts.components.depth_frame.frame import DepthFrame

    return DepthFrame(code_predictor, talker_table)


def _unsupported(code_predictor, talker_table: torch.Tensor) -> str | None:
    fused = code_predictor.fused
    attn = code_predictor.model.layers[0].self_attn
    fits = (
        len(fused.qkv) == 5 and code_predictor.input_tables is not None and len(code_predictor.input_tables) == 14
        and all(tuple(getattr(fused, name)[0].shape) == shape for name, shape in _SHAPES.items())
        and tuple(fused.heads.shape) == (15, 2048, 1024) and tuple(talker_table.shape) == (3072, 2048)
        and (attn.num_heads, attn.num_kv_heads, attn.head_dim) == (16, 8, 128)
    )
    if not fits:
        return "not the 1.7B CodePredictor's shapes"
    if fused.qkv[0].dtype != torch.bfloat16:
        return f"weights in {fused.qkv[0].dtype}, not bfloat16"
    if current_platform.cuda_capability() != (9, 0):
        return "not a Hopper GPU"
    if torch.cuda.get_device_properties(fused.qkv[0].device).multi_processor_count < schedule.MIN_GRID:
        return f"fewer than {schedule.MIN_GRID} SMs"
    if tuple(int(part) for part in triton.__version__.split(".")[:2]) < (3, 7):
        return f"Triton {triton.__version__}, not 3.7 or later"
    if "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
        return "under MPS"
    return None
