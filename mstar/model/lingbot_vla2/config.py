"""Configuration for LingBot-VLA 2.0 (Qwen3-VL-4B prefix + 36L MoE action expert)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mstar.model.components.lingbot_vla2_action_expert import LingBotVLAActionExpertConfig
from mstar.model.components.lingbot_vla2_prefix import (
    LingBotVLA2PrefixQueriesConfig,
    LingBotVLAPrefixConfig,
)
from mstar.model.components.qwen3_lm import Qwen3LMConfig
from mstar.model.components.qwen_vision import QwenVisionConfig

LLM_KV = "LLM_KV"
LLM_ATTN = "LLM_ATTN"
SUFFIX_ATTN = "SUFFIX_ATTN"
VISION_TOKENS = "VISION_TOKENS"
VISION_ATTN = "VISION_ATTN"
PREFILL_WALK = "prefill"
ACTION_WALK = "action_gen"
LLM_NODES = frozenset({"LLM", "action_expert"})

# Exact defaults from Qwen/Qwen3-VL-4B-Instruct/config.json so real VLA checkpoints
# that only ship their top-level VLA config.json can still resolve the prefix tower.
_QWEN3_VL_4B_DEFAULTS: dict[str, Any] = {
    "model_type": "qwen3_vl",
    "image_token_id": 151655,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": True,
    "vision_config": {
        "depth": 24,
        "hidden_size": 1024,
        "out_hidden_size": 2560,
        "num_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": [5, 11, 17],
        "in_channels": 3,
    },
    "text_config": {
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5_000_000.0,
        "mrope_section": [24, 20, 20],
        "rope_parameters": {
            "rope_theta": 5_000_000.0,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        },
        "tie_word_embeddings": True,
    },
}


def _resolve_qwen_dict(checkpoint_dir: Path, qwen3_vl_dir: str | Path | None) -> dict[str, Any]:
    if qwen3_vl_dir is not None:
        cfg_file = Path(qwen3_vl_dir) / "config.json"
        if cfg_file.exists():
            return json.loads(cfg_file.read_text())
    embedded = checkpoint_dir / "qwen3_vl" / "config.json"
    if embedded.exists():
        return json.loads(embedded.read_text())
    vla_cfg_file = checkpoint_dir / "config.json"
    if vla_cfg_file.exists():
        vla = json.loads(vla_cfg_file.read_text())
        if isinstance(vla.get("qwen3_vl_config"), dict):
            return vla["qwen3_vl_config"]
        if "vision_config" in vla and "text_config" in vla:
            return vla
    return _QWEN3_VL_4B_DEFAULTS


@dataclass
class LingBotVLA2Config:
    """Full configuration for LingBot-VLA 2.0."""

    vision: QwenVisionConfig
    text: Qwen3LMConfig
    action_expert: LingBotVLAActionExpertConfig
    prefix_queries: LingBotVLA2PrefixQueriesConfig = field(
        default_factory=LingBotVLA2PrefixQueriesConfig
    )
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    image_token_id: int = 151655
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    pad_token_id: int = 151643
    num_cameras: int = 3
    image_size: int = 256
    tokenizer_max_length: int = 72
    chunk_size: int = 50
    num_flow_steps: int = 10
    max_state_dim: int = 55
    max_action_dim: int = 55
    state_dim: int = 14
    action_dim: int = 14
    fp32_attention: bool = False
    norm_stats: dict | str | None = None
    qwen3_vl_config_path: str | None = None
    qwen3_vl_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text.num_key_value_heads != self.action_expert.num_key_value_heads:
            raise ValueError(
                f"Prefix num_key_value_heads ({self.text.num_key_value_heads}) must match "
                f"action_expert num_key_value_heads ({self.action_expert.num_key_value_heads})"
            )
        if self.text.head_dim != self.action_expert.head_dim:
            raise ValueError(
                f"Prefix head_dim ({self.text.head_dim}) must match "
                f"action_expert head_dim ({self.action_expert.head_dim})"
            )
        if self.text.num_hidden_layers != self.action_expert.num_hidden_layers:
            raise ValueError(
                f"Prefix num_hidden_layers ({self.text.num_hidden_layers}) must match "
                f"action_expert num_hidden_layers ({self.action_expert.num_hidden_layers})"
            )
        if self.action_expert.max_state_dim != self.max_state_dim:
            raise ValueError("max_state_dim must match action_expert.max_state_dim")
        if self.action_expert.max_action_dim != self.max_action_dim:
            raise ValueError("max_action_dim must match action_expert.max_action_dim")
        if self.action_expert.fp32_attention != self.fp32_attention:
            raise ValueError("fp32_attention must match action_expert.fp32_attention")
        if self.action_expert.mrope_section != self.mrope_section:
            raise ValueError("mrope_section must match action_expert.mrope_section")
        if self.state_dim > self.max_state_dim or self.action_dim > self.max_action_dim:
            raise ValueError("Active state_dim/action_dim cannot exceed max_state_dim/max_action_dim")
        rotary_from_mrope = 2 * sum(self.mrope_section)
        if rotary_from_mrope != self.text.head_dim:
            raise ValueError(
                f"2 * sum(mrope_section)={rotary_from_mrope} must equal head_dim={self.text.head_dim}"
            )
        if self.prefix_queries.hidden_size != self.text.hidden_size:
            raise ValueError(
                f"prefix_queries.hidden_size ({self.prefix_queries.hidden_size}) must match "
                f"text.hidden_size ({self.text.hidden_size})"
            )

    @property
    def tokens_per_image(self) -> int:
        patch_grid = self.image_size // self.vision.patch_size
        merged = patch_grid // self.vision.spatial_merge_size
        return merged * merged

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        qwen3_vl_dir: str | Path | None = None,
        *,
        fp32_attention: bool = False,
        norm_stats: dict | str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "LingBotVLA2Config":
        checkpoint_dir = Path(checkpoint_dir)
        vla = json.loads((checkpoint_dir / "config.json").read_text())
        if vla.get("vlm_family", "qwen3_vl") != "qwen3_vl":
            raise ValueError(f"LingBot-VLA 2.0 requires vlm_family='qwen3_vl', got {vla.get('vlm_family')!r}")

        qwen = _resolve_qwen_dict(checkpoint_dir, qwen3_vl_dir)
        if qwen.get("model_type", "qwen3_vl") != "qwen3_vl":
            raise ValueError("LingBot-VLA 2.0 requires a Qwen3-VL base config")

        if qwen3_vl_dir is not None and (Path(qwen3_vl_dir) / "config.json").exists():
            prefix_cfg = LingBotVLAPrefixConfig.from_pretrained(checkpoint_dir, qwen3_vl_dir)
            vision_cfg = prefix_cfg.vision
            text_cfg = prefix_cfg.text
            section = prefix_cfg.mrope_section
        else:
            vis = qwen["vision_config"]
            txt = qwen["text_config"]
            rope_params = txt.get("rope_parameters") or txt.get("rope_scaling") or {}
            raw_sec = tuple(
                int(x) for x in rope_params.get("mrope_section", txt.get("mrope_section", ()))
            )
            if not rope_params.get("mrope_interleaved") or len(raw_sec) != 3:
                raise ValueError("LingBot-VLA 2.0 requires interleaved 3D MRoPE")
            section = (raw_sec[0], raw_sec[1], raw_sec[2])
            vision_cfg = QwenVisionConfig(
                depth=int(vis["depth"]),
                hidden_size=int(vis["hidden_size"]),
                out_hidden_size=int(vis["out_hidden_size"]),
                num_heads=int(vis["num_heads"]),
                intermediate_size=int(vis["intermediate_size"]),
                patch_size=int(vis["patch_size"]),
                temporal_patch_size=int(vis["temporal_patch_size"]),
                spatial_merge_size=int(vis["spatial_merge_size"]),
                num_position_embeddings=int(vis["num_position_embeddings"]),
                deepstack_visual_indexes=tuple(int(x) for x in vis.get("deepstack_visual_indexes", (5, 11, 17))),
                in_channels=int(vis.get("in_channels", 3)),
            )
            text_cfg = Qwen3LMConfig(
                num_hidden_layers=int(txt["num_hidden_layers"]),
                hidden_size=int(txt["hidden_size"]),
                intermediate_size=int(txt["intermediate_size"]),
                num_attention_heads=int(txt["num_attention_heads"]),
                num_key_value_heads=int(txt["num_key_value_heads"]),
                head_dim=int(txt["head_dim"]),
                vocab_size=int(txt["vocab_size"]),
                max_position_embeddings=int(txt["max_position_embeddings"]),
                rms_norm_eps=float(txt["rms_norm_eps"]),
                rope_theta=float(rope_params.get("rope_theta", txt.get("rope_theta", 5_000_000.0))),
                tie_word_embeddings=bool(
                    txt.get("tie_word_embeddings", qwen.get("tie_word_embeddings", True))
                ),
            )
            if not text_cfg.tie_word_embeddings:
                raise ValueError("LingBot-VLA 2.0 prefix requires the released VLM's tied output head")
            if vision_cfg.out_hidden_size != text_cfg.hidden_size:
                raise ValueError("Qwen3-VL vision and text widths differ")

        action_cfg = LingBotVLAActionExpertConfig(
            num_hidden_layers=text_cfg.num_hidden_layers,
            hidden_size=int(vla.get("expert_hidden_size", 768)),
            num_attention_heads=int(vla.get("action_num_attention_heads", 32)),
            num_key_value_heads=int(vla.get("action_num_key_value_heads", text_cfg.num_key_value_heads)),
            head_dim=int(vla.get("action_head_dim", text_cfg.head_dim)),
            num_experts=int(vla.get("token_num_experts", 32)),
            top_k=int(vla.get("token_top_k", 4)),
            routed_intermediate_size=int(vla.get("token_moe_intermediate_size", 512)),
            shared_intermediate_size=int(vla.get("token_shared_intermediate_size", 704)),
            routed_scaling_factor=float(vla.get("routed_scaling_factor", 4.0)),
            max_state_dim=int(vla.get("max_state_dim", 55)),
            max_action_dim=int(vla.get("max_action_dim", 55)),
            rms_norm_eps=text_cfg.rms_norm_eps,
            rope_theta=text_cfg.rope_theta,
            mrope_section=section,
            final_norm_adanorm=bool(vla.get("final_norm_adanorm", False)),
            action_fp32=bool(vla.get("action_fp32", False)),
            fp32_attention=fp32_attention,
            kv_key=LLM_KV,
        )

        queries_cfg = LingBotVLA2PrefixQueriesConfig(
            hidden_size=text_cfg.hidden_size,
            num_task_tokens=int(vla.get("num_task_tokens", 8)),
            use_current_patch_loss=bool(vla.get("use_current_patch_loss", False)),
            use_current_shared_task_proj=bool(vla.get("use_current_shared_task_proj", False)),
            use_future_depth=bool(vla.get("use_future_depth", False)),
            use_future_video=bool(vla.get("use_future_video", False)),
            share_future_depth_query=bool(vla.get("share_future_depth_query", False)),
            use_shared_future_task_proj=bool(vla.get("use_shared_future_task_proj", False)),
            block_future_depth_to_action=bool(vla.get("block_future_depth_to_action", False)),
            block_suffix_to_future_video=bool(vla.get("block_suffix_to_future_video", False)),
        )

        if norm_stats is None:
            for candidate_name in ("norm_stats.json", "dataset_statistics.json"):
                candidate = checkpoint_dir / candidate_name
                if candidate.exists():
                    norm_stats = str(candidate)
                    break

        return cls(
            vision=vision_cfg,
            text=text_cfg,
            action_expert=action_cfg,
            prefix_queries=queries_cfg,
            mrope_section=section,
            image_token_id=int(qwen.get("image_token_id", 151655)),
            vision_start_token_id=int(qwen.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(qwen.get("vision_end_token_id", 151653)),
            pad_token_id=int(vla.get("pad_token_id", 151643)),
            num_cameras=int(vla.get("num_cameras", 3)),
            image_size=int(vla.get("img_size", vla.get("image_size", 256))),
            tokenizer_max_length=int(vla.get("tokenizer_max_length", 72)),
            chunk_size=int(vla.get("chunk_size", vla.get("action_horizon", 50))),
            num_flow_steps=int(vla.get("num_steps", vla.get("num_flow_steps", 10))),
            max_state_dim=action_cfg.max_state_dim,
            max_action_dim=action_cfg.max_action_dim,
            state_dim=int(vla.get("state_dim", 14)),
            action_dim=int(vla.get("action_dim", 14)),
            fp32_attention=fp32_attention,
            norm_stats=norm_stats,
            qwen3_vl_config_path=None if qwen3_vl_dir is None else str(qwen3_vl_dir),
            qwen3_vl_model_id=str(vla.get("tokenizer_path", "Qwen/Qwen3-VL-4B-Instruct")),
            extra=dict(extra or {}),
        )

    @classmethod
    def for_defaults(
        cls,
        *,
        fp32_attention: bool = False,
        norm_stats: dict | str | None = None,
    ) -> "LingBotVLA2Config":
        """Build the default 6B RoboTwin configuration without reading files."""
        qwen = _QWEN3_VL_4B_DEFAULTS
        vis = qwen["vision_config"]
        txt = qwen["text_config"]
        section = tuple(int(x) for x in txt["mrope_section"])
        vision_cfg = QwenVisionConfig(
            depth=int(vis["depth"]),
            hidden_size=int(vis["hidden_size"]),
            out_hidden_size=int(vis["out_hidden_size"]),
            num_heads=int(vis["num_heads"]),
            intermediate_size=int(vis["intermediate_size"]),
            patch_size=int(vis["patch_size"]),
            temporal_patch_size=int(vis["temporal_patch_size"]),
            spatial_merge_size=int(vis["spatial_merge_size"]),
            num_position_embeddings=int(vis["num_position_embeddings"]),
            deepstack_visual_indexes=tuple(int(x) for x in vis["deepstack_visual_indexes"]),
            in_channels=int(vis["in_channels"]),
        )
        text_cfg = Qwen3LMConfig(
            num_hidden_layers=int(txt["num_hidden_layers"]),
            hidden_size=int(txt["hidden_size"]),
            intermediate_size=int(txt["intermediate_size"]),
            num_attention_heads=int(txt["num_attention_heads"]),
            num_key_value_heads=int(txt["num_key_value_heads"]),
            head_dim=int(txt["head_dim"]),
            vocab_size=int(txt["vocab_size"]),
            max_position_embeddings=int(txt["max_position_embeddings"]),
            rms_norm_eps=float(txt["rms_norm_eps"]),
            rope_theta=float(txt["rope_theta"]),
            tie_word_embeddings=bool(txt["tie_word_embeddings"]),
        )
        action_cfg = LingBotVLAActionExpertConfig(
            num_hidden_layers=text_cfg.num_hidden_layers,
            rms_norm_eps=text_cfg.rms_norm_eps,
            rope_theta=text_cfg.rope_theta,
            mrope_section=(section[0], section[1], section[2]),
            fp32_attention=fp32_attention,
            kv_key=LLM_KV,
        )
        return cls(
            vision=vision_cfg,
            text=text_cfg,
            action_expert=action_cfg,
            prefix_queries=LingBotVLA2PrefixQueriesConfig(hidden_size=text_cfg.hidden_size),
            mrope_section=(section[0], section[1], section[2]),
            fp32_attention=fp32_attention,
            norm_stats=norm_stats,
        )

    @classmethod
    def for_dummy(
        cls,
        *,
        num_layers: int = 2,
        text_hidden: int = 64,
        expert_hidden: int = 32,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 32,
        mrope_section: tuple[int, int, int] = (6, 5, 5),
        vocab_size: int = 256,
        max_state_dim: int = 16,
        max_action_dim: int = 16,
        state_dim: int = 8,
        action_dim: int = 8,
        chunk_size: int = 4,
        num_flow_steps: int = 2,
        num_cameras: int = 1,
        image_size: int = 8,
        action_fp32: bool = False,
        fp32_attention: bool = False,
        with_prefix_queries: bool = False,
        norm_stats: dict | str | None = None,
    ) -> "LingBotVLA2Config":
        """Build a tiny config for CPU unit, graph, and parity tests."""
        vision = QwenVisionConfig(
            depth=2,
            hidden_size=32,
            out_hidden_size=text_hidden,
            num_heads=4,
            intermediate_size=64,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=(0, 1),
            in_channels=3,
        )
        text = Qwen3LMConfig(
            num_hidden_layers=num_layers,
            hidden_size=text_hidden,
            intermediate_size=text_hidden * 2,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            tie_word_embeddings=True,
        )
        action_expert = LingBotVLAActionExpertConfig(
            num_hidden_layers=num_layers,
            hidden_size=expert_hidden,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=4,
            top_k=2,
            routed_intermediate_size=expert_hidden,
            shared_intermediate_size=expert_hidden,
            routed_scaling_factor=2.0,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            mrope_section=mrope_section,
            final_norm_adanorm=False,
            action_fp32=action_fp32,
            fp32_attention=fp32_attention,
            kv_key=LLM_KV,
        )
        prefix_queries = LingBotVLA2PrefixQueriesConfig(
            hidden_size=text_hidden,
            num_task_tokens=2,
            use_current_patch_loss=with_prefix_queries,
            use_current_shared_task_proj=with_prefix_queries,
            use_future_depth=with_prefix_queries,
            use_future_video=with_prefix_queries,
            share_future_depth_query=with_prefix_queries,
            use_shared_future_task_proj=with_prefix_queries,
        )
        return cls(
            vision=vision,
            text=text,
            action_expert=action_expert,
            prefix_queries=prefix_queries,
            mrope_section=mrope_section,
            image_token_id=vocab_size - 3,
            vision_start_token_id=vocab_size - 2,
            vision_end_token_id=vocab_size - 1,
            pad_token_id=0,
            num_cameras=num_cameras,
            image_size=image_size,
            tokenizer_max_length=32,
            chunk_size=chunk_size,
            num_flow_steps=num_flow_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            fp32_attention=fp32_attention,
            norm_stats=norm_stats,
        )
