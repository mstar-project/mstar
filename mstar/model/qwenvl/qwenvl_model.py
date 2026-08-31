"""Qwen3-VL-30B-A3B graph: vision encoder -> Qwen3 MoE decoder."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.qwenvl.config import load_qwenvl_config
from mstar.model.qwenvl.submodules import qwen_vl_position_ids
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)


class QwenVLModel(Model):
    """Qwen3-VL-30B-A3B: image/text prefill and text decode.

    The MoE LLM node is TP-capable. The Hugging Face Qwen3-VL vision tower
    stays unsharded and emits both final merger embeddings and the three
    DeepStack feature sets injected into early decoder layers.
    """

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = self._resolve_snapshot()
        self.config = load_qwenvl_config(self.local_dir)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.local_dir, trust_remote_code=True)

    def _resolve_snapshot(self) -> str:
        path = Path(self.model_path_hf)
        if path.is_dir():
            return str(path)
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        text = self.config.text_config
        return [
            KVCacheConfig(
                num_layers=text.num_hidden_layers,
                num_kv_heads=text.num_key_value_heads,
                head_dim=text.head_dim,
                max_seq_len=text.max_position_embeddings,
                num_qo_heads=text.num_attention_heads,
                nodes=["LLM"],
            )
        ]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {"vision_encoder": EngineType.STATELESS, "LLM": EngineType.KV_CACHE}

    def get_graph_walk_graphs(self):
        return {
            "prefill": GraphNode(
                name="LLM",
                input_names=["text_inputs", "position_ids"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    )
                ],
            ),
            "prefill_vision": Sequential(
                [
                    GraphNode(
                        name="vision_encoder",
                        input_names=["pixel_values", "image_grid_thw", "text_inputs", "position_ids"],
                        outputs=[
                            GraphEdge(next_node="LLM", name="vision_embeds"),
                            GraphEdge(next_node="LLM", name="deepstack_visual_embeds"),
                            GraphEdge(next_node="LLM", name="text_inputs"),
                            GraphEdge(next_node="LLM", name="position_ids"),
                        ],
                    ),
                    GraphNode(
                        name="LLM",
                        input_names=["vision_embeds", "deepstack_visual_embeds", "text_inputs", "position_ids"],
                        outputs=[
                            GraphEdge(
                                next_node=EMIT_TO_CLIENT,
                                name="new_token",
                                output_modality="text",
                                persist=True,
                            )
                        ],
                    ),
                ]
            ),
            "decode": Loop(
                name="decode_loop",
                section=GraphNode(
                    name="LLM",
                    input_names=["text_inputs"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="new_token",
                            output_modality="text",
                        ),
                        GraphEdge(next_node="LLM", name="text_inputs"),
                    ],
                ),
                max_iters=self.get_max_output_tokens(),
                outputs=[],
            ),
        }

    def process_prompt(
        self,
        prompt,
        input_modalities,
        output_modalities,
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        tensors = tensors or {}
        if tensors.get("video_inputs"):
            raise NotImplementedError("QwenVL supports image+text inputs; video support is a later slice.")
        raw_images = tensors.get("image_inputs", [])
        images = []
        for image in raw_images:
            # data_worker provides (C, H, W) float32 in [0, 1].  The HF
            # processor expects uint8 HWC; do_rescale=True would otherwise
            # double-rescale a float image to near-black.
            if image.dtype.is_floating_point:
                image = (image * 255.0).clamp(0, 255).to(torch.uint8)
            if image.dim() == 3 and image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            images.append(image.cpu().contiguous().numpy())
        content = ([{"type": "image"}] * len(images)) + [{"type": "text", "text": prompt or ""}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_args = {"text": [text], "return_tensors": "pt"}
        if images:
            processor_args["images"] = images
        processed = self.processor(**processor_args)
        input_ids = processed["input_ids"][0]
        grids = processed.get("image_grid_thw")
        # Qwen's processor stores one ``[T, H, W]`` row per image (not a
        # batch dimension), so keep the complete matrix for both the vision
        # tower and MRoPE construction.
        grid = grids if grids is not None else None
        result: NameToTensorList = {
            "text_inputs": [input_ids],
            "position_ids": [qwen_vl_position_ids(input_ids, grid, self.config)],
        }
        if images:
            result["pixel_values"] = [processed["pixel_values"]]
            result["image_grid_thw"] = [grid]
        return result

    def get_initial_forward_pass_args(
        self,
        partition_name,
        input_modalities,
        output_modalities,
        input_signals,
        model_kwargs=None,
    ):
        has_image = "pixel_values" in input_signals
        walk = "prefill_vision" if has_image else "prefill"
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
        )
        names = (
            ["pixel_values", "image_grid_thw", "text_inputs", "position_ids"]
            if has_image
            else ["text_inputs", "position_ids"]
        )
        edges = []
        for name in names:
            edge = GraphEdge(next_node="vision_encoder" if has_image else "LLM", name=name)
            edge.tensor_info = input_signals.get(name, [])
            edges.append(edge)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=edges,
            unpersist_tensors=sum((edge.tensor_info for edge in edges), start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name,
        partition_metadata,
        persist_signals,
        incoming_connections: list[StreamingConnectionState] | None = None,
    ):
        if partition_metadata.is_prefill:
            partition_metadata.is_prefill = False
            partition_metadata.graph_walk = "decode"
        else:
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )
        edge = GraphEdge(next_node="LLM", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[edge],
            unpersist_tensors=sum(edge.tensor_info, start=[]),
            step_metadata={"is_prefill": False},
        )

    def get_sampling_config(self, node_name, model_kwargs=None):
        options = model_kwargs or {}
        return SamplingConfig(
            vocab_size=self.config.text_config.vocab_size,
            temperature=options.get("temperature", 0.0),
            top_p=options.get("top_p", 1.0),
            ignore_eos=options.get("ignore_eos", False),
        )

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality != "text":
            raise ValueError(f"Unsupported QwenVL output modality {modality!r}.")
        return self.processor.decode(output.reshape(-1)).encode()

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # LLM weights shard via ParallelAttention / ColumnParallelLinear.
        # Vision stays replicated: the HF tower is not TP-rewritten, and
        # vision_embeds is already the merged language-width tensor.
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(node_name, device, tp_group, autocast_dtype)
        return self._submodule_cache[node_name]

    def _create_submodule(self, node_name, device, tp_group, autocast_dtype):
        from mstar.model.qwenvl.weight_loader import (
            iter_qwen_vl_text_weights,
            iter_qwen_vl_vision_weights,
            load_qwen_vl_text_weights,
            load_qwen_vl_vision_weights,
            require_complete_weight_load,
        )

        if node_name == "LLM":
            from mstar.model.qwenvl.components import QwenVLForCausalLM
            from mstar.model.qwenvl.submodules import QwenVLLLMSubmodule

            with torch.device("meta"):
                model = QwenVLForCausalLM(self.config, comm_group=tp_group)
            if autocast_dtype is not None:
                model = model.to(autocast_dtype)
            model.to_empty(device=device)
            loaded = load_qwen_vl_text_weights(model, iter_qwen_vl_text_weights(self.local_dir, device))
            require_complete_weight_load(model, loaded, "text")
            model.eval()
            return QwenVLLLMSubmodule(model, self.config)
        if node_name == "vision_encoder":
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
                Qwen3VLMoeVisionModel,
            )

            from mstar.model.qwenvl.submodules import QwenVLVisionSubmodule

            with torch.device("meta"):
                vision = Qwen3VLMoeVisionModel(self.config.vision_config)
            if autocast_dtype is not None:
                vision = vision.to(autocast_dtype)
            vision.to_empty(device=device)
            loaded = load_qwen_vl_vision_weights(vision, iter_qwen_vl_vision_weights(self.local_dir, device))
            require_complete_weight_load(vision, loaded, "vision")
            vision.eval()
            return QwenVLVisionSubmodule(vision)
        return None
