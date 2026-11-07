"""Qwen3.5-MoE (``qwen3_5_moe``: Qwen3.5-35B-A3B to 397B-A17B), served text-only.

The dense model's hybrid stack with every MLP a sparse MoE block (``components/moe.py``); the delta-net and
full-attention layers, resources, walks and prompt handling are the dense model's. The checkpoint's vision tower
(``model.visual.*``) and MTP layer (``mtp.*``) are not loaded: there is no ``prefill_vision`` walk and no
``vision_encoder`` node, and a request with an image or a video is refused at intake.

A quantized checkpoint serves as stored under ``quantization: auto`` (``get_quantization_roots``): nvidia's
NVFP4 one runs its routed experts NVFP4 and its fp8-per-tensor linears fp8, Qwen's FP8 one block fp8.
"""
from __future__ import annotations

from typing import Any

from mstar.graph.base import GraphSection
from mstar.model.base import ProcessPromptOutput
from mstar.model.multimodal import TEXT
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.submodule_base import NodeSubmodule


class Qwen3_5MoeModel(Qwen3_5DenseModel):
    # the conductor builds the fused-MoE align op before spawning the workers
    prebuild_fused_moe = True

    def __init__(
        self,
        model_path_hf: str = "",
        cache_dir: str | None = None,
        prefill_token_buckets: list[int] | None = None,
        **kwargs: Any,
    ):
        # `model_kwargs.checkpoint_path`: a local snapshot (or another repo) in place of the registry's
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        super().__init__(checkpoint_path or model_path_hf, cache_dir, prefill_token_buckets, **kwargs)
        if not self.config.is_moe:
            raise ValueError(
                f"{self.model_path_hf} is a dense Qwen3.5 checkpoint (no `num_experts`); serve it as qwen3_5_<size>b"
            )
        self.vision_config = None

    def _llm_submodule(self, model: Qwen3_5ForCausalLM) -> NodeSubmodule:
        from mstar.model.qwen3_5.submodules import TextOnlyLLMSubmodule

        return TextOnlyLLMSubmodule(model, self.config, prefill_token_buckets=self.prefill_token_buckets)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        walks = super().get_graph_walk_graphs()
        del walks["prefill_vision"]
        return walks

    def process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None,
                       **kwargs) -> ProcessPromptOutput:
        media = sorted({m for m in input_modalities if m != TEXT})
        if media:
            raise ValueError(f"Qwen3.5-MoE is served text-only here; the request carries {', '.join(media)}")
        return super().process_prompt(
            prompt, input_modalities, output_modalities, tensors=tensors, prompt_parts=prompt_parts, **kwargs,
        )
