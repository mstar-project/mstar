"""Multimodal Ministral 3 graph declarations and host-side prompt processing.

The shared scaffold owns graph walks, KV layout, sampling request configs and
token-output routing. This wrapper connects text/image preparation, the separate
vision and decoder nodes, and their resource declarations.
"""

import json
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import NodeResourceSpec
from mstar.model.base import ProcessPromptOutput
from mstar.model.components.scalar_fp8_linear import require_scalar_fp8_device
from mstar.model.keys import KV_CACHE, ROPE
from mstar.model.loader import iter_safetensors_file
from mstar.model.ministral3.checkpoint import (
    HF_FILENAME,
    checkpoint_node_keys,
    load_vision_weights,
    validate_quantization,
)
from mstar.model.ministral3.components.vision import Ministral3Vision, PixtralConfig
from mstar.model.ministral3.config import ATTN, LLM_NODE
from mstar.model.ministral3.fp8 import build_fp8_text_model, load_fp8_text_weights
from mstar.model.ministral3.image import IMAGE_GRIDS, Ministral3ImageProcessor
from mstar.model.ministral3.resources import text_node_resources
from mstar.model.ministral3.submodules import Ministral3Decoder
from mstar.model.ministral3.text import Ministral3TextProcessor
from mstar.model.ministral3.vision import (
    VISION_NODE,
    VISION_OUTPUT,
    Ministral3VisionEncoder,
    vision_node_resources,
)
from mstar.model.multimodal import TEXT, MediaEncoder, PromptPart, media_layout
from mstar.model.multimodal_decoder import MediaIntake, MultimodalDecoderModel
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextOutput


class Ministral3Model(MultimodalDecoderModel):
    """A separate Pixtral node feeds LLM prefill, followed by text decode."""

    NAME = "Ministral 3"
    METADATA_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "processor_config.json")

    def __init__(self, model_path_hf: str, cache_dir: str | None = None):
        super().__init__(model_path_hf, cache_dir)
        self.text_processor = Ministral3TextProcessor(self.local_dir)
        self.config = self.text_processor.config
        self.tokenizer = self.text_processor.tokenizer
        raw = json.loads((Path(self.local_dir) / "config.json").read_text())
        self.quantization = raw.get("quantization_config", {})
        validate_quantization(self.quantization)
        self.vision_config = PixtralConfig.from_hf(raw)
        self.image_processor = Ministral3ImageProcessor(self.local_dir, self.vision_config, self.tokenizer)
        self.media = (MediaIntake(
            "image", MediaEncoder("vision", VISION_NODE, ("pixel_values",), VISION_OUTPUT),
            self.image_processor.image_id, self.image_processor.prepare,
        ),)

    def get_node_resources(self) -> list[NodeResourceSpec]:
        return [*text_node_resources(self.config, enable_min_p=bool(self.sampling_defaults.get("min_p"))),
                *vision_node_resources(self.vision_config)]

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        open_streams: frozenset[str] = frozenset(),
        **kwargs,
    ) -> ProcessPromptOutput:
        # Check every intake representation: an undeclared attachment must not
        # disappear merely because input_modalities says this is a text prompt.
        if (set(input_modalities) - {TEXT, "image"}
                or any(p.modality not in (TEXT, "image") for p in prompt_parts or ())):
            raise ValueError("Ministral 3 supports text and image inputs only")
        if prompt_parts is not None and [p.modality for p in prompt_parts] != input_modalities:
            raise ValueError("Ministral 3 prompt parts disagree with input_modalities")
        if any(values for name, values in (tensors or {}).items() if name != "image_inputs"):
            raise ValueError("Ministral 3 intake accepts only image attachment tensors")
        if open_streams:
            raise ValueError("Ministral 3 expects a complete prompt, not streamed input")
        if output_modalities != [TEXT]:
            raise ValueError("Ministral 3 supports text output only")
        return super().process_prompt(
            prompt, input_modalities, output_modalities, tensors=tensors, prompt_parts=prompt_parts, **kwargs,
        )

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        if any(token in (part.text or "") for part in parts if part.modality == TEXT
               for token in self.image_processor.special_tokens):
            raise ValueError("Image control tokens must come from attachments, not literal prompt text")
        return self.text_processor.prompt_ids(parts, allow_images=True)

    def intake(self, *args, **kwargs):
        prompt = super().intake(*args, **kwargs)
        return self.image_processor.add_row_delimiters(prompt, self.config.max_position_embeddings)

    def prompt_layout(self, prompt_in):
        layout = media_layout(prompt_in.pads)
        if prompt_in.image_grids:
            layout[IMAGE_GRIDS] = [list(grid) for grid in prompt_in.image_grids]
        return layout

    def text_output(self) -> TextOutput:
        return self.text_processor.output

    def request_kwargs(self) -> frozenset[str]:
        # The Instruct template has no enable_thinking switch.
        return super().request_kwargs() - {"enable_thinking"}

    def get_default_sharding_config(self) -> ShardingConfig:
        # Independent node placement is supported; scalar FP8 projections are unsharded.
        return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})

    def _resolve_weights(self) -> str:
        directory = Path(self.local_dir)
        if not (directory / HF_FILENAME).is_file() and not Path(self.model_path_hf).is_dir():
            from huggingface_hub import snapshot_download

            # Metadata resolution returned a Hub snapshot directory named by its
            # commit. Fetch weights from that same revision, not a moving main.
            directory = Path(snapshot_download(
                repo_id=self.model_path_hf, cache_dir=self.cache_dir,
                revision=directory.name, allow_patterns=[HF_FILENAME],
            ))
        if not (directory / HF_FILENAME).is_file():
            raise FileNotFoundError(f"Ministral 3 requires {directory / HF_FILENAME}; "
                                    "the consolidated checkpoint format is not supported")
        return str(directory)

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule:
        if node_name not in (LLM_NODE, VISION_NODE):
            raise ValueError(f"Ministral 3 has no node {node_name!r}")
        if dtype != torch.bfloat16:
            raise ValueError("Ministral 3 requires BF16 activations with native FP8 text weights")
        if node_name == LLM_NODE:
            require_scalar_fp8_device(device)
        path = Path(self._resolve_weights()) / HF_FILENAME
        text_keys, vision_keys = checkpoint_node_keys(path, self.config, self.vision_config)
        keys = text_keys if node_name == LLM_NODE else vision_keys
        # Keep source tensors on CPU: scalar validation runs once at loading,
        # and only this node's payloads are read by the shared iterator.
        weights = iter_safetensors_file(path, keys=keys, quant_methods={"fp8"})
        try:
            if node_name == LLM_NODE:
                native = build_fp8_text_model(
                    self.config, self.quantization, device=device,
                    attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE,
                )
                load_fp8_text_weights(native, weights, self.quantization)
                node = Ministral3Decoder(self.config, model=native, encoders=self.encoders)
            else:
                with torch.device("meta"):
                    native = Ministral3Vision(self.vision_config).to(dtype=torch.bfloat16)
                native.to_empty(device=device)
                load_vision_weights(native, weights)
                node = Ministral3VisionEncoder(self.vision_config, model=native)
        finally:
            weights.close()
        return node.requires_grad_(False).eval()
