"""Vision encoder factory for Ming-flash-omni-2.0.

The Ming-flash-omni-2.0 vision encoder is ``Qwen3MoeVisionTransformer``
from the Ming source repo's ``qwen3_moe_vit.py`` (574 LOC). Rather than
fork the file, we resolve it dynamically from the staged Ming source dir
that ``MingFlashOmniModel.__init__`` already symlinks alongside the
snapshot (see ``_prepare_tokenizer_dir``).

The vllm-omni port (``vision_encoder.py:MingVisionEncoder``) wraps
vLLM's ``Qwen3Omni_VisionTransformer`` because vLLM ships a TP/quant-
aware re-implementation. mminf doesn't have vLLM as a dep, and the
upstream encoder runs at full quality on a single GPU (~1 GB at bf16),
so we use the reference implementation as-is. The encoder is built once
per process and lives on the rank that owns the ``vision_encoder`` graph
node (typically rank 0; see ``configs/ming_flash_omni.yaml``).

Returned encoder's ``.forward(hidden_states, grid_thw)`` matches the
upstream signature: returns a single ``(N_tokens, out_hidden_size)``
tensor when ``use_deepstack=False`` (the default for the released ckpt,
since the LLM-side DeepStack splicing isn't enabled in step 4), or a
``(hidden_states, deepstack_feature_lists)`` tuple when
``use_deepstack=True``.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import torch
from torch import nn

from mminf.model.ming_omni_flash.config import VisionEncoderConfig

logger = logging.getLogger(__name__)


def _import_ming_vit(local_dir: str | None = None) -> type[nn.Module]:
    """Resolve ``Qwen3MoeVisionTransformer`` from the staged Ming source.

    ``MingFlashOmniModel.__init__`` pushes the snapshot dir onto
    ``sys.path`` and symlinks ``qwen3_moe_vit.py`` into it (see
    ``_MING_CODE_FILES`` and ``_prepare_tokenizer_dir``). We import via
    that path so all the other dynamic imports the file performs
    (e.g. ``from configuration_bailingmm2 import ...``) keep resolving
    against the same staged tree.

    Args:
        local_dir: Optional snapshot dir to put on ``sys.path`` first.
            Callers that bypass ``MingFlashOmniModel.__init__`` (tests,
            standalone benchmarks) can pass this to avoid an
            ``ImportError`` on a fresh interpreter.
    """
    if local_dir is not None:
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        # Also push the Ming source repo (if discoverable) so the dynamic
        # imports inside qwen3_moe_vit.py resolve cross-file. The snapshot
        # is the symlink staging dir; we discover any "real" source by
        # following one of the staged symlinks back to its target.
        candidate = Path(local_dir) / "qwen3_moe_vit.py"
        if candidate.is_symlink():
            ming_root = Path(candidate).resolve().parent
            if str(ming_root) not in sys.path:
                sys.path.insert(0, str(ming_root))

    try:
        module = importlib.import_module("qwen3_moe_vit")
    except ImportError as e:
        raise ImportError(
            "Could not import qwen3_moe_vit. Ensure MingFlashOmniModel "
            "was constructed (which stages the Ming source files), or "
            "pass local_dir=<snapshot path> explicitly. See "
            "PORTING_NOTES.md 'Ming source dependency' for setup."
        ) from e

    return module.Qwen3MoeVisionTransformer


def build_vision_encoder(
    config: VisionEncoderConfig,
    use_deepstack: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    attn_implementation: str = "flash_attention_2",
    local_dir: str | None = None,
) -> nn.Module:
    """Construct the Ming vision encoder.

    Args:
        config:              VisionEncoderConfig from MingFlashOmniModelConfig.
        use_deepstack:       Whether ``.forward()`` returns the per-checkpoint
                             deepstack feature lists. Off by default — the
                             LLM-side DeepStack splice lands with step 5
                             (thinker graph walks for vision prefill).
        dtype:               Cast the encoder to this dtype after construction.
                             bf16 matches the released ckpt; fp16 also works.
        device:              Final device for the encoder weights.
        attn_implementation: Maps to ``config._attn_implementation`` on the
                             internal Qwen3VLMoeVisionConfig. ``flash_attention_2``
                             is mandatory for video performance — sdpa falls
                             into the per-segment Python loop (see qwen3_omni
                             model.py:1508-1519 for the same gotcha).
        local_dir:           Snapshot directory to add to sys.path if the Ming
                             source modules aren't already importable.

    Returns:
        An ``nn.Module`` ready to consume ``(pixel_values, grid_thw)``.
        Weight loading is the caller's job — Ming stores vision encoder
        weights under the top-level ``vision.*`` prefix in the released
        ckpt.
    """
    Qwen3MoeVisionTransformer = _import_ming_vit(local_dir=local_dir)

    # Build the internal config the Ming module expects.
    module = sys.modules["qwen3_moe_vit"]
    InternalConfig = module.Qwen3VLMoeVisionConfig
    internal_config = InternalConfig(
        depth=config.depth,
        hidden_size=config.hidden_size,
        hidden_act=config.hidden_act,
        intermediate_size=config.intermediate_size,
        num_heads=config.num_heads,
        in_channels=config.in_channels,
        patch_size=config.patch_size,
        spatial_merge_size=config.spatial_merge_size,
        temporal_patch_size=config.temporal_patch_size,
        out_hidden_size=config.out_hidden_size,
        num_position_embeddings=config.num_position_embeddings,
        deepstack_visual_indexes=list(config.deepstack_visual_indexes),
    )
    # The attention path branches on _attn_implementation. The Ming
    # source hard-codes it to "flash_attention_2" inside __init__ of
    # Qwen3VLMoeVisionAttention, but we set it on the config too for
    # the rare debug path that wants to flip to "sdpa" or "eager".
    internal_config._attn_implementation = attn_implementation

    encoder = Qwen3MoeVisionTransformer(
        internal_config,
        use_deepstack=use_deepstack,
    )
    encoder = encoder.to(dtype=dtype, device=device)
    encoder.eval()
    return encoder


__all__ = ["build_vision_encoder"]
