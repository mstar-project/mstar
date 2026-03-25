from copy import deepcopy

import numpy as np
import torch

from mminf.communication.tensors import NameToTensorList
from mminf.engine.kv_store import KVCacheConfig
from mminf.engine.base import EngineType
from mminf.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential, TensorPointerInfo
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import CurrentForwardMetadata, Model


class DummyModel(Model):
    """
    Show-o2-inspired model that does nothing, for testing and example purposes.
    """
    def _get_text_emb(self):
        return Sequential([
            GraphNode(
                name="text_emb",
                input_ids=["text_inputs"],
                outputs=[
                    GraphEdge(next_node="concat_text", name="new_text_emb")
                ]
            ),
            GraphNode(
                name="concat_text",
                input_ids=["new_text_emb", "existing_text_emb"],
                outputs=[
                    GraphEdge(next_node="LLM", name="text_emb", persist=True)
                ]
            )
        ])

    def _get_img_emb(self):
        return Sequential([
            GraphNode(
                name="image_emb",
                input_ids=["image_inputs"],
                outputs=[
                    GraphEdge(next_node="concat_img", name="new_image_emb")
                ]
            ),
            GraphNode(
                name="concat_img",
                input_ids=["new_image_emb", "existing_image_emb"],
                outputs=[
                    GraphEdge(next_node="LLM", name="img_emb", persist=True)
                ]
             )
        ])

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "text_emb": EngineType.ENC_DEC,
            "concat_text": EngineType.ENC_DEC,
            "image_emb": EngineType.ENC_DEC,
            "concat_img": EngineType.ENC_DEC,
            "LLM": EngineType.AR,
            "flow": EngineType.FLOW,
            "VAE_dec": EngineType.ENC_DEC,
        }

    def get_graph_walk_graphs(self):
        prefill = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            GraphNode(
                name="LLM",
                input_ids=["text_emb", "img_emb"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        output_modality="text",
                        name="new_token",
                        is_new_token=True
                    )
                ]
            )
        ])
        decode = deepcopy(prefill)
        image_gen = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            Loop(
                section=Sequential([
                    GraphNode(
                        name="LLM",
                        input_ids=["text_emb", "img_emb", "latents"],
                        outputs=[
                            GraphEdge(next_node="flow", name="hidden_states")
                        ]
                    ),
                    GraphNode(
                        "flow",
                        input_ids=["hidden_states"],
                        outputs=[
                            GraphEdge(next_node="LLM", name="latents")
                        ]
                    )
                ]),
                n_iters=10,
                outputs=[
                    GraphEdge(next_node="VAE_dec", name="latents")
                ]
            ),
            GraphNode(
                name="VAE_dec",
                input_ids=["latents"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        output_modality="image",
                        name="image_output",
                        persist=True
                    )
                ]
            )
        ])

        return dict(
            prefill=prefill,
            decode=decode,
            image_gen=image_gen
        )

    def get_kv_cache_config(self) -> KVCacheConfig:
        return KVCacheConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            max_seq_len=1,
        )

    def get_initial_forward_pass_args(
        self, input_modalities, output_modalities
    ):
        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True
        )

    def get_forward_pass_args(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        prev_forward_metadata: CurrentForwardMetadata=None,
    ) -> list[GraphEdge]:
        text_inp = GraphEdge(
            next_node="text_emb",
            name="text_inputs",
        )
        img_inp = GraphEdge(
            next_node="image_emb",
            name="image_inputs",
        )
        existing_text = GraphEdge(
            next_node="concat_text",
            name="existing_text_emb",
        )
        existing_img = GraphEdge(
            next_node="concat_img",
            name="existing_image_emb",
        )

        graph_edges = [
            text_inp, img_inp, existing_text, existing_img
        ]

        if metadata.is_prefill: # first forward
            text_inp.tensor_info = persist_signals.get("text_inputs", [])
            img_inp.tensor_info = persist_signals.get("image_inputs", [])
        else:
            existing_text.tensor_info = persist_signals.get("text_emb", [])
            existing_img.tensor_info = persist_signals.get("img_emb", [])
            if prev_forward_metadata.graph_walk == "image_gen":
                img_inp.tensor_info = persist_signals.get("image_output", [])
                text_inp.tensor_info = persist_signals.get("new_token", [])

            if metadata.graph_walk == "image_gen":
                graph_edges.append(
                    GraphEdge(
                        next_node="LLM",
                        name="latents",
                        tensor_info=persist_signals.get("latents", [])
                    )
                )
        return graph_edges

    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        new_tokens: dict[str, list[int]],
    ) -> CurrentForwardMetadata:
        # dummy model doesn't actually do anything, so this function will just
        # randomly select a graph walk
        if metadata.graph_walk == "image_gen":
            metadata.request_done = True
            return
        metadata.graph_walk = str(np.random.choice(["decode", "image_gen"]))
        if metadata.graph_walk == "decode":
            metadata.output_modalities = ["text"]
        else:
            metadata.output_modalities = ["image"]
        return metadata

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        result = {}
        if prompt is not None:
            byte_data = prompt.encode("utf-8")
            result["text_inputs"] = [
                torch.tensor(list(byte_data), dtype=torch.uint8)
            ]
        return result

    def get_submodule(self, node_name: str, device="cpu") -> torch.nn.Module | None:
        return None  # dummy mode — no real computation
