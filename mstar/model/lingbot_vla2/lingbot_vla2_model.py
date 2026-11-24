"""LingBotVLA2Model: RobbyAnt LingBot-VLA 2.0 vision-language-action model.

Architecture (3 nodes, 2 graph walks):
    PREFILL_WALK ("prefill"):
        vit_encoder    -> Qwen3-VL ViT with per-image Block attention & deepstack taps
        LLM            -> Qwen3DenseLM prefix prefill + optional alignment query tokens
                          writing ``LLM_KV`` (Causal read) and emitting ``prefix_done``
                          (persisted MRoPE prefix end position).
    ACTION_WALK ("action_gen"):
        action_expert  -> 36L flow-matching MoE action expert reading ``LLM_KV`` via
                          ``PrefixSuffixGroups(first_group_len=1)`` across ``num_flow_steps``
                          Euler steps in one node call, emitting ``actions``.
"""

from __future__ import annotations

import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.state import (
    Array,
    Attend,
    Block,
    Causal,
    KVEntry,
    Layout,
    Positions,
    PrefixSuffixGroups,
    Read,
    Span,
)
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    KVLabel,
    Model,
    ProcessPromptOutput,
    RobotIO,
    TensorAndMetadata,
    WalkStep,
)
from mstar.model.components.lingbot_vla2_action_expert import (
    LingBotVLAActionExpert,
    load_action_expert_weights,
)
from mstar.model.components.lingbot_vla2_prefix import (
    LingBotVLA2PrefixQueries,
    load_prefix_query_weights,
    load_prefix_weights,
)
from mstar.model.components.mrope import compute_inv_freq, layout_extent
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.qwen_vision import QwenVisionModel
from mstar.model.lingbot_vla2.config import (
    ACTION_WALK,
    LLM_ATTN,
    LLM_KV,
    LLM_NODES,
    PREFILL_WALK,
    SUFFIX_ATTN,
    VISION_ATTN,
    VISION_TOKENS,
    LingBotVLA2Config,
)
from mstar.model.lingbot_vla2.normalization import (
    LingBotVLA2NormStats,
    load_norm_stats,
    pad_vector,
    slice_vector,
)
from mstar.model.lingbot_vla2.submodules import (
    LingBotVLA2ActionSubmodule,
    LingBotVLA2LLMSubmodule,
    LingBotVLA2ViTEncoderSubmodule,
    _image_layout_from_tokens,
)
from mstar.model.media_limits import read_image
from mstar.model.multimodal import TEXT, PromptPart
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


def _flat(value: Any):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flat(item)
    else:
        yield value


def _resolve_checkpoint_subdir(root: Path) -> Path:
    """Resolve HF snapshot root or ``checkpoints/global_step_*/hf_ckpt`` layout."""
    if (root / "config.json").exists():
        return root
    candidates = sorted(root.glob("checkpoints/global_step_*/hf_ckpt"))
    for candidate in reversed(candidates):
        if (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(
        f"No config.json found in {root} or {root / 'checkpoints/global_step_*/hf_ckpt'}"
    )


class LingBotVLA2Model(Model):
    """RobbyAnt LingBot-VLA 2.0 vision-language-action model."""

    PREPROCESS_ONLY_KWARGS = ("robot_state",)
    PREPROCESS_PARALLEL = True
    PREPROCESS_PARALLEL_IMAGES = True

    PREFILL_WALK = PREFILL_WALK
    ACTION_WALK = ACTION_WALK
    DEFAULT_PROMPT = "Execute the task."

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs: Any,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self._qwen3_vl_dir: str | Path | None = kwargs.pop("qwen3_vl_dir", None)
        explicit_config: LingBotVLA2Config | None = kwargs.pop("config", None)
        self._yaml_config_overrides: dict[str, Any] = dict(kwargs)

        self._repo_dir: Path | None = None
        self._checkpoint_dir: Path | None = None

        self.config: LingBotVLA2Config = self._load_config(explicit_config)
        self.norm_stats: LingBotVLA2NormStats | None = load_norm_stats(self.config.norm_stats)
        self.tokenizer = self._load_tokenizer()

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self.vision: QwenVisionModel | None = None
        self.language_model: Qwen3DenseLM | None = None
        self.prefix_queries: LingBotVLA2PrefixQueries | None = None
        self.action_expert: LingBotVLAActionExpert | None = None

    # ------------------------------------------------------------------
    # Configuration & tokenizer loading
    # ------------------------------------------------------------------

    def _ensure_checkpoint_dir(self) -> Path:
        if self._checkpoint_dir is not None:
            return self._checkpoint_dir
        local_candidate = Path(self.model_path_hf)
        if local_candidate.is_dir():
            self._repo_dir = local_candidate
            self._checkpoint_dir = _resolve_checkpoint_subdir(local_candidate)
            return self._checkpoint_dir

        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir))
        self._repo_dir = snapshot
        self._checkpoint_dir = _resolve_checkpoint_subdir(snapshot)
        return self._checkpoint_dir

    def _load_config(self, explicit_config: LingBotVLA2Config | None) -> LingBotVLA2Config:
        if explicit_config is not None:
            cfg = explicit_config
        elif self.skip_weight_loading:
            cfg = LingBotVLA2Config.for_defaults()
        else:
            checkpoint_dir = self._ensure_checkpoint_dir()
            cfg = LingBotVLA2Config.from_pretrained(
                checkpoint_dir,
                qwen3_vl_dir=self._qwen3_vl_dir,
            )

        if self._yaml_config_overrides:
            valid = {f.name for f in fields(LingBotVLA2Config)}
            updates: dict[str, Any] = {}
            for key, value in self._yaml_config_overrides.items():
                if key in valid:
                    updates[key] = value
                else:
                    logger.warning(
                        "LingBotVLA2Model: model_kwargs key %r is not a LingBotVLA2Config field; ignored.",
                        key,
                    )
            if updates:
                cfg = replace(cfg, **updates)
        return cfg

    def _load_tokenizer(self):
        if self.skip_weight_loading:
            local = Path(self.model_path_hf)
            if local.is_dir() and (
                (local / "tokenizer.json").exists() or (local / "tokenizer_config.json").exists()
            ):
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(str(local), use_fast=True)
            return None

        from transformers import AutoTokenizer

        candidates: list[str] = [str(self._ensure_checkpoint_dir())]
        if self._qwen3_vl_dir is not None:
            candidates.append(str(self._qwen3_vl_dir))
        candidates.append(self.config.qwen3_vl_model_id)

        errors: list[str] = []
        for source in candidates:
            try:
                return AutoTokenizer.from_pretrained(
                    source, cache_dir=self.cache_dir, use_fast=True
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}: {exc}")
        raise ValueError("LingBotVLA2Model could not load a tokenizer:\n" + "\n".join(errors))

    # ------------------------------------------------------------------
    # Model ABC: state contract & execution graph
    # ------------------------------------------------------------------

    def declare_state(self) -> list[Array]:
        c = self.config
        t = c.text
        v = c.vision
        e = c.action_expert
        return [
            Array(
                LLM_KV,
                nodes=set(LLM_NODES),
                layers=frozenset(range(t.num_hidden_layers)),
                entry=KVEntry(heads=t.num_key_value_heads, head_dim=t.head_dim),
                max_len=t.max_position_embeddings,
                positions=Positions(),
                reads={
                    LLM_ATTN: Read(
                        mask=Causal(),
                        op=Attend(q_heads=t.num_attention_heads),
                        walks={PREFILL_WALK},
                    ),
                    SUFFIX_ATTN: Read(
                        mask=PrefixSuffixGroups(
                            first_group_len=1,
                            fp32_attention=c.fp32_attention,
                        ),
                        op=Attend(q_heads=e.num_attention_heads),
                        walks={ACTION_WALK},
                    ),
                },
            ),
            Array(
                VISION_TOKENS,
                nodes={"vit_encoder"},
                layers=frozenset(range(v.depth)),
                entry=KVEntry(heads=v.num_heads, head_dim=v.head_dim),
                max_len=0,
                reads={VISION_ATTN: Read(mask=Block(), op=Attend(q_heads=v.num_heads))},
            ),
        ]

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        groups = config.get("node_groups")
        if groups and not any(set(LLM_NODES) <= set(group.get("node_names", [])) for group in groups):
            raise ValueError(
                f"{config_path}: node_groups must list {' and '.join(sorted(LLM_NODES))} in one entry, "
                "so the action step follows the prefix on the worker that holds its KV cache"
            )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential(
            [
                GraphNode(
                    name="vit_encoder",
                    input_names=["image_inputs"],
                    outputs=[
                        GraphEdge(next_node="LLM", name="img_emb"),
                        GraphEdge(next_node="LLM", name="deepstack_features"),
                    ],
                ),
                GraphNode(
                    name="LLM",
                    input_names=["img_emb", "deepstack_features", "text_inputs"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION,
                            name="prefix_done",
                            persist=True,
                        ),
                    ],
                ),
            ]
        )
        action_gen = GraphNode(
            name="action_expert",
            input_names=["prefix_done", "robot_state"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="actions",
                    output_modality="action",
                ),
            ],
        )
        return {
            self.PREFILL_WALK: prefill,
            self.ACTION_WALK: action_gen,
        }

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        del partition_name, input_modalities, output_modalities, model_kwargs
        if not input_signals.get("image_inputs"):
            raise ValueError("LingBot-VLA 2.0 requires camera images (image_inputs)")
        if not input_signals.get("robot_state"):
            raise ValueError("LingBot-VLA 2.0 requires robot_state")
        return [WalkStep(self.PREFILL_WALK), WalkStep(self.ACTION_WALK)]

    # ------------------------------------------------------------------
    # Model ABC: request I/O & OpenPI interface
    # ------------------------------------------------------------------

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {LLM_KV: {"main": KVLabel()}}

    def robot_io(self) -> RobotIO:
        return RobotIO("robot_state", self.config.chunk_size, self.config.action_dim)

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"robot_state"})

    def max_input_files(self) -> dict[str, int]:
        return {"image": self.config.num_cameras}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        return self.load_decoded_image(read_image(filepath, rgb=True), device)

    def load_decoded_image(self, image: torch.Tensor, device: str) -> TensorAndMetadata:
        return TensorAndMetadata(image.to(device))

    def _robot_state(self, raw: Any) -> torch.Tensor:
        if raw is None:
            raise ValueError("LingBot-VLA 2.0 requires robot_state in request model_kwargs")
        limit = self.config.state_dim
        if isinstance(raw, torch.Tensor):
            state = raw.to(torch.float64)
        else:
            if isinstance(raw, (str, bytes)) or any(
                isinstance(v, (bool, str, bytes)) for v in _flat(raw)
            ):
                raise ValueError(
                    f"robot_state must be a 1D sequence of 1 to {limit} finite numbers; got {raw!r}"
                )
            try:
                state = torch.tensor(raw, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError):
                raise ValueError(
                    f"robot_state must be a 1D sequence of 1 to {limit} finite numbers; got {raw!r}"
                ) from None
        if state.dim() != 1 or not (1 <= state.numel() <= limit):
            raise ValueError(
                f"robot_state must have shape (1..{limit},), got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("robot_state must be finite; it holds NaN or inf")
        return state

    def _encode_text_segment(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        # Deterministic byte-level fallback for skip_weight_loading tests without tokenizer files.
        vocab = self.config.text.vocab_size
        forbidden = {
            self.config.image_token_id,
            self.config.vision_start_token_id,
            self.config.vision_end_token_id,
            self.config.pad_token_id,
        }
        tokens: list[int] = []
        for byte in text.encode("utf-8"):
            tid = byte % max(vocab, 1)
            while tid in forbidden and vocab > len(forbidden):
                tid = (tid + 1) % vocab
            tokens.append(tid)
        return tokens

    def _build_prompt_ids(
        self,
        prompt: str | None,
        num_images: int,
        prompt_parts: list[PromptPart] | None,
    ) -> torch.Tensor:
        cfg = self.config
        image_block = (
            [cfg.vision_start_token_id]
            + [cfg.image_token_id] * cfg.tokens_per_image
            + [cfg.vision_end_token_id]
        )

        ids: list[int] = []
        if prompt_parts is not None:
            image_slots = sum(1 for part in prompt_parts if part.modality == "image")
            if image_slots and image_slots != num_images:
                raise ValueError(
                    f"Prompt specifies {image_slots} image placeholders, "
                    f"but received {num_images} images"
                )
            if image_slots == 0:
                for _ in range(num_images):
                    ids.extend(image_block)
            for part in prompt_parts:
                if part.modality == "image":
                    ids.extend(image_block)
                elif part.modality == TEXT and part.text:
                    ids.extend(self._encode_text_segment(part.text))
                elif part.modality not in (TEXT, "image"):
                    raise ValueError(
                        f"Unsupported prompt part modality for LingBot-VLA 2.0: {part.modality!r}"
                    )
        else:
            for _ in range(num_images):
                ids.extend(image_block)
            text = prompt if (prompt is not None and prompt != "") else self.DEFAULT_PROMPT
            ids.extend(self._encode_text_segment(text))

        return torch.tensor(ids, dtype=torch.long)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        open_streams: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> ProcessPromptOutput:
        del input_modalities, output_modalities, open_streams
        tensor_map = tensors if tensors is not None else kwargs.get("tensors") or {}
        images = list(tensor_map.get("image_inputs") or ())
        if not images:
            raise ValueError("LingBot-VLA 2.0 requires at least one camera image")
        if len(images) > self.config.num_cameras:
            raise ValueError(
                f"Received {len(images)} camera images, max num_cameras={self.config.num_cameras}"
            )

        raw_state = self._robot_state(kwargs.get("robot_state"))
        if self.norm_stats is not None:
            norm_state = self.norm_stats.normalize_state(
                raw_state.to(torch.float32),
                self.config.max_state_dim,
            )
        else:
            norm_state = pad_vector(raw_state.to(torch.float32), self.config.max_state_dim)

        text_ids = self._build_prompt_ids(prompt, len(images), prompt_parts)
        gh = self.config.image_size // self.config.vision.patch_size
        merge = self.config.vision.spatial_merge_size
        layout_dict = _image_layout_from_tokens(
            text_ids,
            image_token_id=self.config.image_token_id,
            gh=gh,
            gw=gh,
            tokens_per_image=self.config.tokens_per_image,
        )
        pos_extent = layout_extent(layout_dict, merge, int(text_ids.numel()))
        num_queries = self.config.prefix_queries.num_query_tokens
        prefix_tokens = int(text_ids.numel()) + num_queries
        pos_len = pos_extent + num_queries

        kv_layout = Layout(
            spans=(Span("rows", self.PREFILL_WALK, prefix_tokens, pos_len, "text_inputs"),),
            scratch=1 + self.config.chunk_size,
        )
        return ProcessPromptOutput(
            new_input_tensors={
                "text_inputs": [text_ids],
                "robot_state": [norm_state],
            },
            metadata={"state_layout": {LLM_KV: {"main": kv_layout}}},
        )

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        del request_kwargs
        if modality != "action":
            raise ValueError(f"Unsupported output modality for LingBot-VLA 2.0: {modality!r}")
        actions = output.detach().cpu()
        if self.norm_stats is not None:
            actions = self.norm_stats.unnormalize_actions(
                actions, active_dim=self.config.action_dim
            )
        else:
            actions = slice_vector(actions, self.config.action_dim)
        return actions.to(torch.float32).numpy().tobytes()

    # ------------------------------------------------------------------
    # Model ABC: submodule materialization & weight loading
    # ------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group: Any = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        del tp_group, autocast_dtype
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]

        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        if node_name == "vit_encoder":
            self._init_vision(device)
            return LingBotVLA2ViTEncoderSubmodule(self.vision, self.config)
        if node_name == "LLM":
            self._init_language_model(device)
            return LingBotVLA2LLMSubmodule(self.language_model, self.prefix_queries, self.config)
        if node_name == "action_expert":
            self._init_action_expert(device)
            return LingBotVLA2ActionSubmodule(self.action_expert, self.config)
        return None

    def _init_vision(self, device: str) -> None:
        if self.vision is not None:
            return
        init_dev = "cpu" if self.skip_weight_loading else "meta"
        with torch.device(init_dev):
            vision = QwenVisionModel(
                self.config.vision,
                attn_key=VISION_ATTN,
                array=VISION_TOKENS,
            )
        if not self.skip_weight_loading:
            vision = vision.to_empty(device=device)
            load_prefix_weights(vision, None, self._ensure_checkpoint_dir(), device=device)
        elif device != "cpu":
            vision = vision.to(device=device)
        self.vision = vision.eval()

    def _init_language_model(self, device: str) -> None:
        if self.language_model is not None and self.prefix_queries is not None:
            return
        init_dev = "cpu" if self.skip_weight_loading else "meta"
        with torch.device(init_dev):
            lm = Qwen3DenseLM(
                self.config.text,
                attn_key=LLM_ATTN,
                kv_key=LLM_KV,
                array=LLM_KV,
                mrope_section=list(self.config.mrope_section),
            )
            prefix_queries = LingBotVLA2PrefixQueries(self.config.prefix_queries)

        if not self.skip_weight_loading:
            checkpoint_dir = self._ensure_checkpoint_dir()
            lm = lm.to_empty(device=device)
            prefix_queries = prefix_queries.to_empty(device=device)
            load_prefix_weights(None, lm, checkpoint_dir, device=device)
            load_prefix_query_weights(prefix_queries, checkpoint_dir, device=device)
        elif device != "cpu":
            lm = lm.to(device=device)
            prefix_queries = prefix_queries.to(device=device)

        lm.mrope_inv_freq = compute_inv_freq(
            self.config.text.head_dim,
            self.config.text.rope_theta,
            device=device,
        )
        self.language_model = lm.eval()
        self.prefix_queries = prefix_queries.eval()

    def _init_action_expert(self, device: str) -> None:
        if self.action_expert is not None:
            return
        init_dev = "cpu" if self.skip_weight_loading else "meta"
        with torch.device(init_dev):
            expert = LingBotVLAActionExpert(self.config.action_expert)

        if not self.skip_weight_loading:
            expert = expert.to_empty(device=device)
            load_action_expert_weights(expert, self._ensure_checkpoint_dir(), device=device)
        elif device != "cpu":
            expert = expert.to(device=device)

        expert.reset_non_persistent_buffers(device=device)
        self.action_expert = expert.eval()


__all__ = ["LingBotVLA2Model"]
