"""The Qwen3-VL prefix of LingBot-VLA 2.0.

The released VLA checkpoint stores its VLM under ``model.qwenvl_with_expert``.
Its own config.json names only the VLM family, so the tower dimensions come
from the Qwen3-VL config that the reference policy loads separately. This
module loads the prefix into M*'s shared Qwen3 vision and language towers;
the action expert and graph walk are separate stages of the port.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

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
        rope = text_config.get("rope_scaling") or {}
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
    vision: QwenVisionModel,
    text: Qwen3DenseLM,
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
    """
    checkpoint_dir = Path(checkpoint_dir)
    index = checkpoint_dir / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(f"LingBot-VLA prefix requires {index}")
    validate_prefix_index(vision, text, index)
    visual = iter_safetensors_shards(checkpoint_dir, device=device, prefix=_VISION_PREFIX)
    loaded_vision = load_weights_into(
        vision,
        ((name.removeprefix(_VISION_PREFIX), value) for name, value in visual),
        strict=True,
    )
    language = iter_safetensors_shards(checkpoint_dir, device=device, prefix=_TEXT_PREFIX)
    loaded_text = text.load_weights(
        (name.removeprefix(_TEXT_PREFIX), value) for name, value in language
    )
    return loaded_vision, loaded_text


def validate_prefix_index(
    vision: QwenVisionModel, text: Qwen3DenseLM, index_path: str | Path,
) -> None:
    """Check the published prefix's keys before reading its large shards.

    The language tower holds fused Q/K/V and gate/up parameters, while the
    checkpoint stores those projections separately. The released VLM ties
    ``lm_head`` to ``embed_tokens`` and omits the head tensor.
    """
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    actual_vision = {name.removeprefix(_VISION_PREFIX) for name in weight_map if name.startswith(_VISION_PREFIX)}
    expected_vision = set(dict(vision.named_parameters()))

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

    for tower, actual, expected in (
        ("vision", actual_vision, expected_vision),
        ("language", actual_text, expected_text),
    ):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"LingBot-VLA {tower} checkpoint keys differ: "
                f"{len(missing)} missing (e.g. {missing[:3]}), "
                f"{len(extra)} extra (e.g. {extra[:3]})"
            )
