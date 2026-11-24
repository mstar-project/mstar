"""The Qwen3-VL prefix of LingBot-VLA 2.0.

The released VLA checkpoint stores its VLM under ``model.qwenvl_with_expert``.
Its own config.json names only the VLM family, so the tower dimensions come
from the Qwen3-VL config that the reference policy loads separately. This
module loads the prefix into M*'s shared Qwen3 vision and language towers;
the action expert and graph walk are separate stages of the port.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig
from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel
from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_VLM_PREFIX = "model.qwenvl_with_expert.qwenvl.model."
_VISION_PREFIX = _VLM_PREFIX + "visual."
_TEXT_PREFIX = _VLM_PREFIX + "language_model."


@dataclass(frozen=True)
class LingBotVLAPrefixConfig:
    vision: QwenVisionConfig
    text: Qwen3LMConfig
    mrope_section: tuple[int, int, int]

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: str | Path, qwen3_vl_dir: str | Path,
    ) -> "LingBotVLAPrefixConfig":
        """Read the VLA family marker and the separate Qwen3-VL architecture.

        ``checkpoint_dir`` is the VLA ``hf_ckpt`` directory, not its parent
        training run. ``qwen3_vl_dir`` contains the base VLM's config.json.
        Neither path downloads weights or executes remote model code.
        """
        checkpoint_dir = Path(checkpoint_dir)
        qwen3_vl_dir = Path(qwen3_vl_dir)
        vla = json.loads((checkpoint_dir / "config.json").read_text())
        if vla.get("vlm_family") != "qwen3_vl":
            raise ValueError("LingBot-VLA 2.0 prefix requires vlm_family='qwen3_vl'")

        qwen = json.loads((qwen3_vl_dir / "config.json").read_text())
        if qwen.get("model_type") != "qwen3_vl":
            raise ValueError("LingBot-VLA 2.0 requires a Qwen3-VL base config")
        text_config = qwen["text_config"]
        rope = text_config.get("rope_parameters") or text_config.get("rope_scaling") or {}
        section = tuple(rope.get("mrope_section", ()))
        if not rope.get("mrope_interleaved") or len(section) != 3:
            raise ValueError("LingBot-VLA 2.0 requires interleaved 3D MRoPE")
        text = Qwen3LMConfig.from_hf(text_config)
        vision = QwenVisionConfig.from_hf(qwen3_vl_dir)
        if not text.tie_word_embeddings:
            raise ValueError("LingBot-VLA 2.0 prefix requires the released VLM's tied output head")
        if sum(section) * 2 != text.head_dim:
            raise ValueError("Qwen3-VL MRoPE sections do not fill one attention head")
        if vision.out_hidden_size != text.hidden_size:
            raise ValueError("Qwen3-VL vision and text widths differ")
        return cls(vision=vision, text=text, mrope_section=section)


def load_prefix_weights(
    vision: QwenVisionModel | None,
    text: Qwen3DenseLM | None,
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[set[str], set[str]]:
    """Stream the released VLA's two prefix towers into shared M* modules.

    A checkpoint index is required so every prefix key is checked before any
    shard is read. Filtering in the safetensors iterator leaves the action
    expert and the training-only alignment heads unread. The vision tower's
    names match the checkpoint directly. The dense LM's loader handles
    separate Q/K/V and gate/up tensors and its tied output head.
    Either ``vision`` or ``text`` may be ``None`` to load only one tower.
    """
    checkpoint_dir = Path(checkpoint_dir)
    index = checkpoint_dir / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(f"LingBot-VLA prefix requires {index}")
    validate_prefix_index(vision, text, index)
    loaded_vision: set[str] = set()
    loaded_text: set[str] = set()
    if vision is not None:
        visual = iter_safetensors_shards(checkpoint_dir, device=device, prefix=_VISION_PREFIX)
        loaded_vision = load_weights_into(
            vision,
            ((name.removeprefix(_VISION_PREFIX), value) for name, value in visual),
            strict=True,
        )
    if text is not None:
        language = iter_safetensors_shards(checkpoint_dir, device=device, prefix=_TEXT_PREFIX)
        loaded_text = text.load_weights(
            (name.removeprefix(_TEXT_PREFIX), value) for name, value in language
        )
    return loaded_vision, loaded_text


def validate_prefix_index(
    vision: QwenVisionModel | None, text: Qwen3DenseLM | None, index_path: str | Path,
) -> None:
    """Check the published prefix's keys before reading its large shards.

    The language tower holds fused Q/K/V and gate/up parameters, while the
    checkpoint stores those projections separately. The released VLM ties
    ``lm_head`` to ``embed_tokens`` and omits the head tensor.
    """
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    checks: list[tuple[str, set[str], set[str]]] = []
    if vision is not None:
        actual_vision = {name.removeprefix(_VISION_PREFIX) for name in weight_map if name.startswith(_VISION_PREFIX)}
        expected_vision = set(dict(vision.named_parameters()))
        checks.append(("vision", actual_vision, expected_vision))

    if text is not None:
        actual_text = {name.removeprefix(_TEXT_PREFIX) for name in weight_map if name.startswith(_TEXT_PREFIX)}
        expected_text = set()
        for name in dict(text.named_parameters()):
            if name == "lm_head.weight" and text.config.tie_word_embeddings:
                continue
            if ".qkv_proj." in name:
                expected_text.update(name.replace(".qkv_proj.", f".{part}_proj.") for part in ("q", "k", "v"))
            elif ".gate_up_proj." in name:
                expected_text.update(name.replace(".gate_up_proj.", f".{part}_proj.") for part in ("gate", "up"))
            else:
                expected_text.add(name)
        checks.append(("language", actual_text, expected_text))

    for tower, actual, expected in checks:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"LingBot-VLA {tower} checkpoint keys differ: "
                f"{len(missing)} missing (e.g. {missing[:3]}), "
                f"{len(extra)} extra (e.g. {extra[:3]})"
            )


@dataclass(frozen=True)
class LingBotVLAPackedPrefix:
    """Flattened variable-length Qwen3-VL prefix ready for ``Qwen3DenseLM``.

    Attributes:
        input_embeds: ``[prefix_tokens, hidden_size]`` text, image, and optional
            learned alignment query embeddings.
        position_ids_3d: ``[3, prefix_tokens]`` MRoPE positions across T/H/W.
        deepstack_addends: One ``[prefix_tokens, hidden_size]`` tensor per
            deepstack visual index, non-zero only on image token rows.
    """

    input_embeds: torch.Tensor
    position_ids_3d: torch.Tensor
    deepstack_addends: list[torch.Tensor]


def compact_deepstack(
    tokens: int,
    hidden: int,
    image_positions: torch.Tensor,
    deepstack_features: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Scatter ViT deepstack features onto ``[tokens, hidden]`` per-layer addends."""
    if not deepstack_features:
        return []
    pos = image_positions.to(device=device, dtype=torch.long)
    addends: list[torch.Tensor] = []
    for feat in deepstack_features:
        if feat.shape[0] != pos.shape[0] or feat.shape[1] != hidden:
            raise ValueError(
                f"Deepstack feature shape {tuple(feat.shape)} does not match "
                f"{pos.shape[0]} image tokens x {hidden}"
            )
        buf = torch.zeros((tokens, hidden), dtype=dtype, device=device)
        if pos.numel() > 0:
            buf.index_copy_(0, pos, feat.to(device=device, dtype=dtype))
        addends.append(buf)
    return addends


def compact_prefix(
    token_embeds: torch.Tensor,
    position_ids_3d: torch.Tensor,
    image_mask: torch.Tensor,
    visual_embeds: torch.Tensor | None = None,
    deepstack_features: Sequence[torch.Tensor] = (),
    extra_prefix_tokens: torch.Tensor | None = None,
) -> LingBotVLAPackedPrefix:
    """Splice ViT features into ``token_embeds``, append learned prefix queries,
    and build aligned ``deepstack_addends``.
    """
    if token_embeds.ndim != 2:
        raise ValueError(f"Expected 2D token_embeds [tokens, hidden], got {tuple(token_embeds.shape)}")
    if position_ids_3d.shape != (3, token_embeds.shape[0]):
        raise ValueError(
            f"Expected position_ids_3d [3, {token_embeds.shape[0]}], got {tuple(position_ids_3d.shape)}"
        )
    if image_mask.shape != (token_embeds.shape[0],):
        raise ValueError(
            f"Expected image_mask [{token_embeds.shape[0]}], got {tuple(image_mask.shape)}"
        )

    embeds = token_embeds.clone()
    image_positions = torch.nonzero(image_mask, as_tuple=False).squeeze(-1)
    if visual_embeds is not None:
        if visual_embeds.shape != (image_positions.numel(), embeds.shape[1]):
            raise ValueError(
                f"Visual embeds shape {tuple(visual_embeds.shape)} does not match "
                f"{image_positions.numel()} placeholder token(s) x {embeds.shape[1]}"
            )
        if image_positions.numel() > 0:
            embeds.index_copy_(0, image_positions, visual_embeds.to(dtype=embeds.dtype, device=embeds.device))

    pos_3d = position_ids_3d.to(device=embeds.device, dtype=torch.long)
    if extra_prefix_tokens is not None and extra_prefix_tokens.numel() > 0:
        if extra_prefix_tokens.ndim != 2 or extra_prefix_tokens.shape[1] != embeds.shape[1]:
            raise ValueError(
                f"extra_prefix_tokens shape {tuple(extra_prefix_tokens.shape)} incompatible "
                f"with hidden width {embeds.shape[1]}"
            )
        q_count = extra_prefix_tokens.shape[0]
        next_pos = (pos_3d.max() + 1) if pos_3d.numel() > 0 else 0
        q_pos = torch.arange(next_pos, next_pos + q_count, device=embeds.device, dtype=torch.long)
        pos_3d = torch.cat([pos_3d, q_pos.unsqueeze(0).expand(3, -1)], dim=1)
        embeds = torch.cat([embeds, extra_prefix_tokens.to(dtype=embeds.dtype, device=embeds.device)], dim=0)

    addends = compact_deepstack(
        embeds.shape[0],
        embeds.shape[1],
        image_positions,
        deepstack_features,
        dtype=embeds.dtype,
        device=embeds.device,
    )
    return LingBotVLAPackedPrefix(
        input_embeds=embeds,
        position_ids_3d=pos_3d,
        deepstack_addends=addends,
    )


@dataclass(frozen=True)
class LingBotVLA2PrefixQueriesConfig:
    """Configuration for optional learned alignment query tokens appended to the VLM prefix."""

    hidden_size: int = 2560
    num_task_tokens: int = 8
    use_current_patch_loss: bool = False
    use_current_shared_task_proj: bool = False
    use_future_depth: bool = False
    use_future_video: bool = False
    share_future_depth_query: bool = False
    use_shared_future_task_proj: bool = False
    block_future_depth_to_action: bool = False
    block_suffix_to_future_video: bool = False

    def __post_init__(self) -> None:
        if self.block_future_depth_to_action or self.block_suffix_to_future_video:
            raise ValueError(
                "block_future_depth_to_action and block_suffix_to_future_video are not supported "
                "(requires sub-prefix per-segment suffix visibility masking)"
            )

    @property
    def has_current_depth(self) -> bool:
        return self.use_current_patch_loss or self.use_current_shared_task_proj

    @property
    def has_current_video(self) -> bool:
        return self.use_current_patch_loss

    @property
    def num_query_tokens(self) -> int:
        count = 0
        if self.use_current_shared_task_proj and self.has_current_depth and self.has_current_video:
            count += self.num_task_tokens
        else:
            if self.has_current_depth:
                count += self.num_task_tokens
            if self.has_current_video:
                count += self.num_task_tokens
        if (
            self.share_future_depth_query
            and self.use_shared_future_task_proj
            and self.use_future_depth
            and self.use_future_video
        ):
            count += self.num_task_tokens
        else:
            if self.use_future_depth:
                count += self.num_task_tokens
            if self.use_future_video:
                count += self.num_task_tokens
        return count


class LingBotVLA2PrefixQueries(nn.Module):
    """Learned alignment query tokens (current depth/video + future depth/video) appended
    to the end of each VLM prefix before caching KV for the action expert.
    """

    def __init__(self, config: LingBotVLA2PrefixQueriesConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        k = config.num_task_tokens

        self.depth_align_embs = (
            nn.Parameter(torch.zeros(k, h)) if config.has_current_depth else None
        )
        self.current_video_align_embs = (
            nn.Parameter(torch.zeros(k, h)) if config.has_current_video else None
        )
        self.current_shared_task_proj = (
            nn.Linear(h * 2, h)
            if (config.use_current_shared_task_proj and config.has_current_depth and config.has_current_video)
            else None
        )

        self.future_depth_align_embs = (
            nn.Parameter(torch.zeros(k, h)) if config.use_future_depth else None
        )
        self.future_video_align_embs = (
            nn.Parameter(torch.zeros(k, h)) if config.use_future_video else None
        )
        self.future_shared_task_proj = (
            nn.Linear(h * 2, h)
            if (
                config.share_future_depth_query
                and config.use_shared_future_task_proj
                and config.use_future_depth
                and config.use_future_video
            )
            else None
        )

    def forward(self) -> torch.Tensor | None:
        if self.config.num_query_tokens == 0:
            return None
        parts: list[torch.Tensor] = []
        if self.current_shared_task_proj is not None:
            assert self.depth_align_embs is not None and self.current_video_align_embs is not None
            parts.append(
                self.current_shared_task_proj(
                    torch.cat([self.depth_align_embs, self.current_video_align_embs], dim=-1)
                )
            )
        else:
            if self.depth_align_embs is not None:
                parts.append(self.depth_align_embs)
            if self.current_video_align_embs is not None:
                parts.append(self.current_video_align_embs)

        if self.future_shared_task_proj is not None:
            assert self.future_depth_align_embs is not None and self.future_video_align_embs is not None
            parts.append(
                self.future_shared_task_proj(
                    torch.cat([self.future_depth_align_embs, self.future_video_align_embs], dim=-1)
                )
            )
        else:
            if self.future_depth_align_embs is not None:
                parts.append(self.future_depth_align_embs)
            if self.future_video_align_embs is not None:
                parts.append(self.future_video_align_embs)
        return torch.cat(parts, dim=0) if parts else None


def load_prefix_query_weights(
    queries: LingBotVLA2PrefixQueries,
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[set[str], set[str]]:
    """Load learned prefix alignment query tensors from ``model.*`` checkpoint keys."""
    if queries.config.num_query_tokens == 0:
        return set(), set()

    checkpoint_dir = Path(checkpoint_dir)
    query_names = {name for name, _ in queries.named_parameters()}
    target_keys = {f"model.{name}" for name in query_names}
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if strict and index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        missing_in_index = sorted(query_names - {k.removeprefix("model.") for k in weight_map if k in target_keys})
        if missing_in_index:
            raise ValueError(
                f"LingBot-VLA prefix query checkpoint left {len(missing_in_index)} key(s) unloaded, "
                f"e.g. {missing_in_index[:5]}"
            )

    def _filtered():
        for name, tensor in iter_safetensors_shards(
            checkpoint_dir, device=device, keys=target_keys
        ):
            if name in target_keys:
                yield name[len("model.") :], tensor

    loaded = load_weights_into(queries, _filtered())
    missing = query_names - loaded
    if strict and missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"LingBot-VLA prefix query checkpoint left {len(missing)} key(s) unloaded, e.g. {sample}"
        )
    return loaded, missing
