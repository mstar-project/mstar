"""Phi-4-multimodal-instruct on the VLM scaffold: a SigLIP vision tower, a conformer
audio tower and Phi-4-mini, three nodes.

The LLM runs every request under the adapter its media call for, as the reference
does (``Phi4MMForCausalLM.forward``'s ``input_mode``): ``vision`` for a prompt with
an image (with or without audio), ``speech`` for audio alone, the base model for
text. The adapters are the engine LoRA pool's, read from the checkpoint's PEFT
directories, so rows on different adapters share a batch.

LongRoPE picks its table per request from the prompt's length (the position
resource's ``MAX_POSITION_KWARG``): a prompt past 4096 tokens runs on the long
table throughout, others on the short one. The reference also switches a request
whose output crosses 4096 tokens, recomputing its cache; this one stays on the
table it started on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.position.config import MAX_POSITION_KWARG, PositionConfig, PositionSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.base import KEEP_ADAPTER, KeepAdapter, ProcessPromptOutput, TensorAndMetadata
from mstar.model.components.audio_features import load_speech_file, speech_file_rate
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.multimodal_decoder import MIXED_SPLIT, MediaIntake, MultimodalDecoderModel
from mstar.model.phi4mm.audio import NARROWBAND_RATE, audio_encoder
from mstar.model.phi4mm.audio import SAMPLE_RATE as AUDIO_RATE
from mstar.model.phi4mm.components.audio import Phi4MMAudioConfig
from mstar.model.phi4mm.components.vision import Phi4MMVisionConfig
from mstar.model.phi4mm.config import (
    ATTN,
    AUDIO_PAD_ID,
    IMAGE_PAD_ID,
    KV_CACHE,
    ROPE,
    SPEECH_ADAPTER,
    VISION_ADAPTER,
    Phi4MMConfig,
    read_hf_config,
)
from mstar.model.phi4mm.vision import vision_media_encoder
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

_PAD_TEXT = {"image": "<|endoftext10|>", "audio": "<|endoftext11|>"}


def _generation_eos_ids(local_dir: str) -> set[int]:
    path = Path(local_dir) / "generation_config.json"
    ids = json.loads(path.read_text()).get("eos_token_id", []) if path.exists() else []
    return {ids} if isinstance(ids, int) else set(ids)


class Phi4MMModel(MultimodalDecoderModel):
    NAME = "Phi-4-multimodal"
    METADATA_FILES = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        # the API server reads the modality adapters to check and key them
        "vision-lora/*",
        "speech-lora/*",
    )

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        super().__init__(model_path_hf, cache_dir)
        hf = read_hf_config(self.local_dir)
        self.config = Phi4MMConfig.from_hf(hf)
        self.vision_config = Phi4MMVisionConfig.from_dir(self.local_dir)
        self.audio_config = Phi4MMAudioConfig.from_hf(hf)
        # tokenizer_config.json names the class; AutoTokenizer would read the remote-code config
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.local_dir)
        # generation_config.json's <|end|> and <|endoftext|>, and the tokenizer's eos
        eos = _generation_eos_ids(self.local_dir)
        if self.tokenizer.eos_token_id is not None:
            eos.add(self.tokenizer.eos_token_id)
        self.config.stop_token_ids = tuple(sorted(eos))
        # vision first: the walk of a prompt with both is `prefill_vision_audio`, whose
        # audio rows take the `vision` projection
        self.media = (
            MediaIntake("image", vision_media_encoder(), IMAGE_PAD_ID, self._image_intake),
            MediaIntake("audio", audio_encoder(), AUDIO_PAD_ID, self._audio_intake),
        )

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """The upload at the rate the reference's feature extractor picks for its own
        rate: 8 kHz (its narrowband front end) for one recorded between 8 and 16 kHz,
        16 kHz above; below 8 kHz it is refused, as there. The reference reaches those
        rates by keeping every n-th sample, so 44.1 kHz becomes 22.05 kHz it reads as
        16 kHz, and 11.025 or 22.05 kHz isn't resampled at all; here every rate is
        resampled properly, so such uploads' features differ from the reference's."""
        rate = speech_file_rate(filepath) or AUDIO_RATE
        if rate < NARROWBAND_RATE:
            raise ValueError(f"Phi-4-multimodal takes audio at 8 kHz or more, not {rate} Hz")
        target = NARROWBAND_RATE if rate < AUDIO_RATE else AUDIO_RATE
        waveform = load_speech_file(filepath, target, self.MAX_AUDIO_SECONDS)
        return TensorAndMetadata(data=waveform, metadata=dict(sample_rate=target, num_channels=1))

    def _image_intake(self, image: torch.Tensor, metadata: dict):
        from mstar.model.phi4mm.vision import image_intake

        return image_intake(image, metadata, self.vision_config)

    def _audio_intake(self, clip: torch.Tensor, metadata: dict):
        from mstar.model.phi4mm.audio import audio_intake

        return audio_intake(clip, metadata, self.audio_config)

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        cfg = self.config
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=cfg.num_hidden_layers,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    max_seq_len=cfg.max_position_embeddings,
                    num_qo_heads=cfg.num_attention_heads,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN, nodes={"LLM"},
                config=AttentionConfig(kv_cache=KV_CACHE, splits=(MIXED_SPLIT,)),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                # the reference rounds its cos/sin to the activations' bf16
                config=PositionConfig(
                    kv_cache=KV_CACHE, rotary_dim=cfg.rotary_dim, rope_theta=cfg.rope_theta,
                    longrope=cfg.longrope, rope_table_dtype=torch.bfloat16,
                ),
            ),
            self._sampler_spec(cfg.vocab_size),
            *self._encoder_resources(),
            self._lora_spec(),
        ]

    def _encoder_resources(self) -> list[NodeResourceSpec]:
        from mstar.model.phi4mm.vision import vision_resources

        return vision_resources(self.vision_config)

    def _lora_spec(self):
        from mstar.model.phi4mm.lora import phi4mm_lora_spec

        spec = phi4mm_lora_spec(self.config)
        # the checkpoint's own adapters; a deployment that names its own `adapters:`
        # names these too (`prompt_lora_adapters`, checked at startup)
        spec.adapters = {
            name: str(Path(self.local_dir) / f"{name}-lora" / "adapter_model.safetensors")
            for name in (VISION_ADAPTER, SPEECH_ADAPTER)
        }
        return spec

    # -----------------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------------

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The chat template's ``<|role|>content<|end|>`` turns then ``<|assistant|>``,
        each attachment one pad where it sits in its turn (the processor's
        ``<|image_N|>`` / ``<|audio_N|>`` become the same pads)."""
        messages = [
            {"role": turn.role, "content": "".join(
                part.text or "" if part.modality == TEXT else _PAD_TEXT[part.modality]
                for part in turn.parts
            )}
            for turn in chat_turns(parts)
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

    def process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs):
        out = super().process_prompt(prompt, input_modalities, output_modalities, tensors, prompt_parts, **kwargs)
        metadata = dict(out.metadata)
        # the table LongRoPE runs the request on, from the prompt's length
        metadata[MAX_POSITION_KWARG] = int(out.new_input_tensors["text_inputs"][0].numel())
        return ProcessPromptOutput(out.new_input_tensors, metadata, self._adapter_for(tensors or {}))

    def prompt_lora_adapters(self) -> frozenset[str]:
        return frozenset({VISION_ADAPTER, SPEECH_ADAPTER})

    @staticmethod
    def _adapter_for(tensors: dict) -> str | KeepAdapter:
        """The reference's ``input_mode``: vision with any image, speech with audio
        alone. A text prompt keeps the client's choice: the base model, or an adapter
        of the deployment's own."""
        if tensors.get("image_inputs"):
            return VISION_ADAPTER
        if tensors.get("audio_inputs"):
            return SPEECH_ADAPTER
        return KEEP_ADAPTER

    # -----------------------------------------------------------------------
    # Submodules
    # -----------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        if node_name == "LLM":
            return self._create_llm(device, dtype, tp_group)
        if node_name == "vision_encoder":
            from mstar.model.phi4mm.vision import create_vision_submodule

            submodule = create_vision_submodule(self._resolve_weights(), device, dtype, self.vision_config)
            logger.info("Loaded Phi-4-multimodal vision tower onto %s", device)
            return submodule
        if node_name == "audio_encoder":
            from mstar.model.phi4mm.audio import Phi4MMAudioEncoderSubmodule
            from mstar.model.phi4mm.components.audio import Phi4MMAudioTower

            tower = Phi4MMAudioTower.load(self.audio_config, self._resolve_weights(), device=device, dtype=dtype)
            logger.info("Loaded Phi-4-multimodal audio tower onto %s", device)
            return Phi4MMAudioEncoderSubmodule(tower)
        return None

    def _create_llm(self, device: str, dtype: torch.dtype, tp_group) -> NodeSubmodule:
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.phi4mm.components.language_model import Phi4MiniLM
        from mstar.model.phi4mm.submodules import Phi4MMLLMSubmodule

        llm = self._build(lambda: Phi4MiniLM(self.config, tp_group), dtype, device)
        llm.load_weights(iter_safetensors_shards(self._resolve_weights(), device=device))
        llm.requires_grad_(False).eval()
        logger.info("Loaded Phi-4-mini onto %s", device)
        return Phi4MMLLMSubmodule(llm, self.config, self.encoders)
