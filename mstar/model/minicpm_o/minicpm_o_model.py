"""MiniCPM-o 4.5: a Qwen3-8B LLM over SigLIP image slices and Whisper audio.

Graph:
    prefill_text   LLM
    prefill_image  vision_encoder -> LLM
    prefill_audio  audio_encoder -> LLM
    prefill_omni   (vision_encoder || audio_encoder) -> LLM
    decode         LLM, looped

A request runs one prefill walk over its whole rendered prompt, picked by
which attachments it carries, then decodes. The prompt keeps upstream's
placeholder tokens; the LLM overwrites their embeddings with the encoders'
outputs at positions ``process_prompt`` reads off the prompt (see
``submodules``).

Half-duplex chat only: the duplex/streaming-session machinery upstream is not
ported.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedBlockCausalAttentionSpec,
    RaggedCrossAttentionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span
from mstar.model.minicpm_o import processor
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    AUDIO_FEATURE_LENS,
    DECODE_LOOP,
    IMAGE_TGT_SIZES,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    RESAMPLER_ATTN,
    VISION_ATTN,
    MiniCPMOConfig,
)
from mstar.model.multimodal import TEXT, PromptPart, check_attachments, messages_from_parts, parts_from_modalities
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextOutput

logger = logging.getLogger(__name__)

LLM = "LLM"
VISION = "vision_encoder"
AUDIO = "audio_encoder"

# Upstream's `chat` defaults when sampling.
DEFAULT_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.02)

# walk -> the request signals each node reads
_WALK_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "prefill_text": {LLM: ("text_inputs",)},
    "prefill_image": {
        VISION: ("pixel_values",),
        LLM: ("text_inputs", "image_positions"),
    },
    "prefill_audio": {
        AUDIO: ("audio_features",),
        LLM: ("text_inputs", "audio_positions"),
    },
    "prefill_omni": {
        VISION: ("pixel_values",),
        AUDIO: ("audio_features",),
        LLM: ("text_inputs", "image_positions", "audio_positions"),
    },
}


def _resolve_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Config and tokenizer files only; workers fetch weights."""
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id, cache_dir=cache_dir,
        allow_patterns=["*.json", "*.txt", "assets/*.wav"],
    )


def _as_pil(image: torch.Tensor):
    """The data worker's ``(C, H, W)`` float [0, 1] as an RGB PIL image."""
    from PIL import Image

    if image.dtype.is_floating_point:
        image = (image * 255.0).clamp(0, 255).to(torch.uint8)
    if image.shape[0] == 4:
        image = image[:3]
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    return Image.fromarray(image.permute(1, 2, 0).cpu().contiguous().numpy(), mode="RGB")


class MiniCPMOModel(Model):
    PREPROCESS_TORCH_THREADS = 4
    # the encoders' segment sizes, which process_prompt settles
    INTERNAL_KWARGS = (IMAGE_TGT_SIZES, AUDIO_FEATURE_LENS)

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None, max_seq_len: int | None = None, **kwargs: Any,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = _resolve_metadata(model_path_hf, cache_dir)
        self.config = MiniCPMOConfig.from_hf(self.local_dir)
        limit = self.config.llm.max_position_embeddings
        self.max_seq_len = limit if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= limit:
            raise ValueError(f"max_seq_len must be in 1..{limit} (max_position_embeddings), got {max_seq_len!r}")
        # upstream's tokenizer class is Qwen2's with token-name attributes added
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.local_dir)
        self.tokenizer = tokenizer
        # upstream's mel: Whisper's, as its own audio processor configures it
        self._mel = WhisperFeatureExtractor(feature_size=self.config.audio.num_mel_bins)
        # A reply ends on <|im_end|> (the tokenizer's eos; config.json names
        # only that) or <|endoftext|>, as upstream's `chat` terminators; and
        # a spoken reply's text closes with <|tts_eos|> just before <|im_end|>,
        # which is where the text the TTS reads ends.
        self.config.stop_token_ids = tuple(
            tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|tts_eos|>")
        )
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def checkpoint_path(self) -> str | None:
        return self.local_dir

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        llm, vision, resampler, audio = (
            self.config.llm, self.config.vision, self.config.resampler, self.config.audio,
        )
        return [
            KVSpec(
                resource_key=LLM_KV, nodes={LLM},
                config=PagedKVConfig(
                    num_layers=llm.num_hidden_layers,
                    num_kv_heads=llm.num_key_value_heads,
                    head_dim=llm.head_dim,
                    max_seq_len=self.max_seq_len,
                    num_qo_heads=llm.num_attention_heads,
                ),
            ),
            AttentionSpec(resource_key=LLM_ATTN, nodes={LLM}, config=AttentionConfig(kv_cache=LLM_KV)),
            PositionSpec(
                resource_key=LLM_POS, nodes={LLM},
                config=PositionConfig(kv_cache=LLM_KV, rope_theta=llm.rope_theta),
            ),
            SamplerSpec(
                resource_key=LLM_SAMPLER, nodes={LLM},
                vocab_size=llm.vocab_size, enable_repetion_penalty=True,
            ),
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={VISION},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_attention_heads,
                    num_kv_heads=vision.num_attention_heads,
                    head_dim=vision.head_dim,
                ),
            ),
            RaggedCrossAttentionSpec(
                resource_key=RESAMPLER_ATTN, nodes={VISION},
                config=RaggedAttentionConfig(
                    num_qo_heads=resampler.num_heads,
                    num_kv_heads=resampler.num_heads,
                    head_dim=resampler.head_dim,
                ),
            ),
            RaggedBlockCausalAttentionSpec(
                resource_key=AUDIO_ATTN, nodes={AUDIO},
                block_size=audio.chunk_frames,
                config=RaggedAttentionConfig(
                    num_qo_heads=audio.encoder_attention_heads,
                    num_kv_heads=audio.encoder_attention_heads,
                    head_dim=audio.head_dim,
                ),
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs], model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        model_kwargs = model_kwargs or {}
        knobs = {**DEFAULT_SAMPLING, **{
            k: model_kwargs[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "ignore_eos")
            if k in model_kwargs
        }}
        return {LLM_SAMPLER: SamplingReqConfig(**knobs)}

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def llm(inputs: list[str]) -> GraphNode:
            return GraphNode(
                name=LLM,
                input_names=inputs,
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", persist=True, output_modality="text"),
                ],
            )

        vision = GraphNode(
            name=VISION, input_names=["pixel_values"],
            outputs=[GraphEdge(next_node=LLM, name="vision_embeds")],
        )
        audio = GraphNode(
            name=AUDIO, input_names=["audio_features"],
            outputs=[GraphEdge(next_node=LLM, name="audio_embeds")],
        )
        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=LLM, name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                ],
            ),
            # a safety bound; the per-request budget is `get_loop_caps`'
            max_iters=self.max_seq_len,
            outputs=[],
        )
        return {
            "prefill_text": llm(["text_inputs"]),
            "prefill_image": Sequential([vision, llm(["text_inputs", "vision_embeds", "image_positions"])]),
            "prefill_audio": Sequential([audio, llm(["text_inputs", "audio_embeds", "audio_positions"])]),
            "prefill_omni": Sequential([
                Parallel([vision, audio]),
                llm(["text_inputs", "vision_embeds", "image_positions", "audio_embeds", "audio_positions"]),
            ]),
            "decode": decode,
        }

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _render(
        self,
        parts: list[PromptPart],
        has_audio: bool,
        enable_thinking: bool,
        system_prompt: str | None,
    ) -> str:
        """The chat string as upstream's `chat` renders it: a message per turn,
        its items joined by newlines, with a tag per attachment."""
        tags = {"image": processor.IMAGE_TAG, "audio": processor.AUDIO_TAG}
        msgs = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "\n".join(
                item["text"] if item["type"] == TEXT else tags[item["type"]] for item in m["content"]
            )}
            for m in messages_from_parts(parts, default_system=system_prompt or None)
        ]
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            # upstream turns the speech template on whenever audio is anywhere
            # in the request
            use_tts_template=has_audio,
            enable_thinking=enable_thinking,
        )

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        unsupported = set(output_modalities) - {TEXT}
        if unsupported:
            raise ValueError(f"MiniCPM-o here has no {', '.join(sorted(unsupported))} output yet")
        tensors = tensors or {}
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT] if prompt_parts is not None else prompt,
            roles=[p.role for p in prompt_parts] if prompt_parts is not None else None,
        )
        bad = {p.modality for p in parts} - {TEXT, "image", "audio"}
        if bad:
            raise ValueError(f"MiniCPM-o takes text, image and audio inputs; got {sorted(bad)}")
        raw_images = tensors.get("image_inputs", [])
        raw_audios = tensors.get("audio_inputs", [])
        check_attachments(parts, {"image": len(raw_images), "audio": len(raw_audios)})
        if not any(p.text for p in parts if p.modality == TEXT) and not raw_images and not raw_audios:
            raise ValueError("MiniCPM-o got a request with no text and no attachments")

        audios = [a.reshape(-1).float().cpu().numpy() for a in raw_audios]
        text = self._render(
            parts, bool(audios),
            enable_thinking=bool(kwargs.get("enable_thinking", False)),
            system_prompt=kwargs.get("system_prompt"),
        )
        max_slice_nums = kwargs.get("max_slice_nums")
        max_slice_nums = processor.MAX_SLICE_NUMS if max_slice_nums is None else max_slice_nums
        p, queries = self.config.vision.patch_size, self.config.resampler.num_queries
        sliced = [processor.slice_image(_as_pil(img), max_slice_nums, p) for img in raw_images]
        text = processor.expand_tags(
            text,
            [processor.image_placeholder(i, grid, queries) for i, (_, grid) in enumerate(sliced)],
            [processor.audio_placeholder(len(a), self.config.audio) for a in audios],
        )
        input_ids, image_positions, audio_positions = processor.prompt_ids(self.tokenizer, text)
        result: NameToTensorList = {"text_inputs": [input_ids]}
        metadata: dict[str, Any] = {}

        if sliced:
            patches, tgt_sizes = zip(
                *(processor.slice_patches(s, p) for slices, _ in sliced for s in slices), strict=True,
            )
            if image_positions.numel() != len(tgt_sizes) * queries:
                raise ValueError(
                    f"image placeholders ({image_positions.numel()}) do not match "
                    f"{len(tgt_sizes)} slices x {queries} queries"
                )
            # bf16 is what the tower casts them to before its patch conv, so the cast is exact
            result["pixel_values"] = [torch.cat(patches).to(torch.bfloat16)]
            metadata[IMAGE_TGT_SIZES] = list(tgt_sizes)
            result["image_positions"] = [image_positions]

        if audios:
            mel = self._mel(
                [piece for a in audios for piece in processor.audio_pieces(a)],
                sampling_rate=processor.SAMPLE_RATE, padding="max_length",
                return_attention_mask=True, return_tensors="pt",
            )
            # a piece shorter than one pooling step (a clip under ~0.1 s, or what
            # is left of a long one past its last 30 s cut) pools to no tokens
            lens = [
                (i, n) for i, n in enumerate(mel["attention_mask"].sum(dim=1).tolist())
                if self.config.audio.pooled_tokens(n) > 0
            ]
            pieces = [mel["input_features"][i, :, :n] for i, n in lens]
            tokens = sum(self.config.audio.pooled_tokens(n) for _, n in lens)
            if audio_positions.numel() != tokens:
                # upstream sizes the placeholder off the whole clip but encodes
                # 30 s pieces, which disagree for some long clips
                raise ValueError(
                    f"audio placeholders ({audio_positions.numel()}) do not match the "
                    f"{tokens} tokens its {len(pieces)} piece(s) encode to"
                )
            if pieces:  # else the clips are all too short to encode, and the prompt is text only
                result["audio_features"] = [torch.cat(pieces, dim=-1).to(torch.bfloat16)]
                metadata[AUDIO_FEATURE_LENS] = [n for _, n in lens]
                result["audio_positions"] = [audio_positions]
        # the prefill writes every prompt row, placeholders included, then a row per decode step
        rows = int(input_ids.numel())
        layout = LabelLayout(
            spans=(Span("rows", self._prefill_walk(result), rows, rows, "text_inputs"),),
            growth=((DECODE_LOOP, 1),),
        )
        return ProcessPromptOutput(result, {**metadata, "kv_layout": {LLM_KV: {"main": layout}}})

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill samples the first of max_output_tokens, so the loop runs one fewer
        return {DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {LLM_KV: {"main": KVLabel(growth=(("decode", DECODE_LOOP),))}}

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), self.config.stop_token_ids)

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == TEXT:
            return super().postprocess(output, modality, request_kwargs)
        raise ValueError(f"Unsupported modality for MiniCPM-o: {modality!r}")

    # ------------------------------------------------------------------
    # Walks
    # ------------------------------------------------------------------

    @staticmethod
    def _prefill_walk(signals: dict[str, list[TensorPointerInfo]]) -> str:
        has_image = bool(signals.get("pixel_values"))
        has_audio = bool(signals.get("audio_features"))
        if has_image and has_audio:
            return "prefill_omni"
        if has_image:
            return "prefill_image"
        if has_audio:
            return "prefill_audio"
        return "prefill_text"

    @staticmethod
    def _walk_inputs(walk: str, signals: dict[str, list[TensorPointerInfo]]) -> list[GraphEdge]:
        edges = []
        for node, names in _WALK_INPUTS[walk].items():
            for name in names:
                edge = GraphEdge(next_node=node, name=name)
                edge.tensor_info = list(signals[name])
                edges.append(edge)
        return edges

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        walk = self._prefill_walk(input_signals)
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
        )
        inputs = self._walk_inputs(walk, input_signals)
        model_kwargs = model_kwargs or {}
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum((e.tensor_info for e in inputs), start=[]),
            step_metadata={k: model_kwargs[k] for k in self.INTERNAL_KWARGS if k in model_kwargs},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        if not metadata.is_prefill:
            # the decode loop has stopped; back here, the reply is done
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        metadata.is_prefill = False
        metadata.graph_walk = "decode"
        edge = GraphEdge(next_node=LLM, name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info))

    # ------------------------------------------------------------------
    # Submodules
    # ------------------------------------------------------------------

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(groups=[], tp_enabled_nodes={LLM}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(
                node_name, device, tp_group, autocast_dtype or torch.bfloat16,
            )
        return self._submodule_cache[node_name]

    def _weights_dir(self) -> str:
        if Path(self.model_path_hf).is_dir():
            return self.model_path_hf
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct under ``dtype`` so no fp32 copy of the weights is
        allocated; fp32-only buffers restore themselves in ``_apply``."""
        torch.set_default_dtype(dtype)
        try:
            return make().to(device)
        finally:
            torch.set_default_dtype(torch.float32)

    def _create_submodule(self, node_name: str, device: str, tp_group, dtype: torch.dtype) -> NodeSubmodule | None:
        from mstar.model.minicpm_o import submodules, weight_loader

        if node_name not in (LLM, VISION, AUDIO):
            return None
        weights = self._weights_dir()
        if node_name == VISION:
            from mstar.model.minicpm_o.components.vision import MiniCPMOVision

            tower = self._build(lambda: MiniCPMOVision(self.config.vision, self.config.resampler), dtype, device)
            weight_loader.load_vision_weights(tower, weights, device)
            return submodules.VisionEncoderSubmodule(tower.requires_grad_(False).eval(), self.config)
        if node_name == AUDIO:
            from mstar.model.minicpm_o.components.audio import MiniCPMOAudio

            tower = self._build(lambda: MiniCPMOAudio(self.config.audio), dtype, device)
            weight_loader.load_audio_weights(tower, weights, device)
            return submodules.AudioEncoderSubmodule(tower.requires_grad_(False).eval(), self.config)

        from mstar.model.components.qwen3_lm import Qwen3DenseLM

        lm = self._build(
            lambda: Qwen3DenseLM(
                self.config.llm, attn_key=LLM_ATTN, kv_key=LLM_KV, pos_key=LLM_POS, comm_group=tp_group,
            ),
            dtype, device,
        )
        weight_loader.load_llm_weights(lm, weights, device)
        logger.info("Loaded MiniCPM-o LLM onto %s", device)
        return submodules.LLMSubmodule(lm.requires_grad_(False).eval(), self.config)
