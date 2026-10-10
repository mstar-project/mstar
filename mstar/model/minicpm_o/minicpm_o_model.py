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

import numpy as np
import torch
from transformers import AutoProcessor

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
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
from mstar.engine.resources.kv.bounded import BoundedKVConfig, BoundedKVSpec
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.minicpm_o.components.tts import TTSConfig
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    RESAMPLER_ATTN,
    T2W_DIT_KV,
    T2W_STATE,
    TTS_ATTN,
    TTS_KV,
    TTS_POS,
    TTS_SAMPLER,
    VISION_ATTN,
    MiniCPMOConfig,
    TTSSampling,
)
from mstar.model.multimodal import TEXT, PromptPart, check_attachments, parts_from_modalities
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import LeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

LLM = "LLM"
VISION = "vision_encoder"
AUDIO = "audio_encoder"
TTS = "TTS"
TOKEN2WAV = "Token2Wav"

# async partitions: the main one runs the LLM and the TTS; Token2Wav vocodes
# the TTS's code stream as it arrives
MAIN = "main"
T2W_CHUNK = "t2w_chunk"

TTS_PREFILL = "tts_prefill"
TTS_DECODE = "tts_decode"
TTS_DECODE_LOOP = "tts_decode_loop"
TTS_SAMPLING = TTSSampling()

# Voices shipped in the checkpoint's assets, by the name a request picks.
# The vocoder's per-request state grows with the longest voice prompt (~0.4 GB
# a request for this 6 s clip), so the checkpoint's 11 s and 17 s system
# voices are left out.
VOICES = {"default": "HT_ref_audio.wav"}
DEFAULT_VOICE = "default"
T2W_DEFAULT_SLOTS = 9

# Upstream's `get_sys_prompt(mode="audio_assistant", language="en")`, around
# the voice's reference audio.
VOICE_PROMPT_PREFIX = "Clone the voice in the provided audio prompt."
VOICE_PROMPT_SUFFIX = (
    "Please assist users while maintaining this voice style. Please answer the user's questions "
    "seriously and in a high quality. Please chat with the user in a highly human-like and oral style. "
    "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
)

# Upstream's `chat` defaults when sampling.
DEFAULT_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.02)

# walk -> the request signals each node reads
_WALK_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "prefill_text": {LLM: ("text_inputs",)},
    "prefill_image": {
        VISION: ("pixel_values", "image_tgt_sizes"),
        LLM: ("text_inputs", "image_positions"),
    },
    "prefill_audio": {
        AUDIO: ("audio_features", "audio_feature_lens"),
        LLM: ("text_inputs", "audio_positions"),
    },
    "prefill_omni": {
        VISION: ("pixel_values", "image_tgt_sizes"),
        AUDIO: ("audio_features", "audio_feature_lens"),
        LLM: ("text_inputs", "image_positions", "audio_positions"),
    },
}


def _resolve_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Config, tokenizer and processor files only; workers fetch weights."""
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id, cache_dir=cache_dir,
        allow_patterns=["*.json", "*.py", "*.txt", "assets/*.wav"],
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


def _bound_positions(bounds: torch.Tensor) -> torch.Tensor:
    """Concatenated ``[start, end)`` ranges of ``[n, 2]`` placeholder bounds."""
    if bounds.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat([torch.arange(int(a), int(b)) for a, b in bounds.reshape(-1, 2).tolist()])


class MiniCPMOModel(Model):
    PREPROCESS_TORCH_THREADS = 4

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs: Any):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = _resolve_metadata(model_path_hf, cache_dir)
        self.config = MiniCPMOConfig.from_hf(self.local_dir)
        self.tts_config = TTSConfig.from_hf(self.config.tts_config)
        self._processor = None
        tokenizer = self.processor.tokenizer
        self.tokenizer = tokenizer
        # A reply ends on <|im_end|> (the tokenizer's eos; config.json names
        # only that) or <|endoftext|>, as upstream's `chat` terminators; and
        # a spoken reply's text closes with <|tts_eos|> just before <|im_end|>,
        # which is where the text the TTS reads ends.
        self.config.stop_token_ids = tuple(
            tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|tts_eos|>")
        )
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._voices: dict[str, np.ndarray] = {}

    @property
    def processor(self):
        """Upstream's processor: image slicing, the Whisper-style mel and the
        placeholder layout, run in the API server's preprocessing."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir, trust_remote_code=True,
            )
        return self._processor

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
                    max_seq_len=llm.max_position_embeddings,
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
                    # 30 s pieces: the largest captured bucket holds two
                    max_segments_per_request=2,
                ),
            ),
            *self._tts_resources(),
            RecurrentStateSpec(
                resource_key=T2W_STATE, nodes={TOKEN2WAV},
                config=RecurrentStateConfig(
                    num_layers=1,
                    blocks={
                        name: RecurrentBlockConfig(shape=block.shape, dtype=block.dtype)
                        for name, block in self._t2w_slot_layout().items()
                    },
                    # ~0.4 GB a slot (one speaking request) for the bundled
                    # voices; `t2w_state.max_slots` in the yaml sizes it
                    max_slots=T2W_DEFAULT_SLOTS,
                ),
            ),
            self._t2w_dit_kv(),
        ]

    def _t2w_capacity(self):
        """Token2wav's caches sized for the longest bundled voice."""
        import soundfile as sf

        from mstar.model.minicpm_o.components.token2wav import CacheCapacity

        seconds = max(
            sf.info(str(Path(self.local_dir) / "assets" / f)).duration for f in VOICES.values()
        )
        # the s3 tokenizer runs at 25 Hz; one token of slack for rounding
        return CacheCapacity(int(seconds * 25) + 1)

    def _t2w_slot_layout(self) -> dict:
        from mstar.model.minicpm_o.components.token2wav import slot_layout

        return slot_layout(self._t2w_capacity())

    def _t2w_dit_kv(self) -> BoundedKVSpec:
        from mstar.model.minicpm_o.components.token2wav import dit_retention
        from mstar.model.minicpm_o.components.token2wav_flow import (
            DIT_DEPTH,
            DIT_HEAD_DIM,
            DIT_HEADS,
            N_TIMESTEPS,
            UP_RATE,
        )

        return BoundedKVSpec(
            resource_key=T2W_DIT_KV, nodes={TOKEN2WAV},
            config=BoundedKVConfig(
                num_layers=N_TIMESTEPS * DIT_DEPTH,
                num_heads=DIT_HEADS,
                head_dim=DIT_HEAD_DIM,
                # the two guidance rows
                rows_per_request=2,
                max_source_len=UP_RATE * self._t2w_capacity().prompt_tokens,
                retention=dit_retention,
                max_slots=T2W_DEFAULT_SLOTS - 1,
                reverse_step_order=True,
            ),
        )

    def _tts_resources(self) -> list[NodeResourceSpec]:
        tts = self.tts_config
        return [
            KVSpec(
                resource_key=TTS_KV, nodes={TTS},
                config=PagedKVConfig(
                    num_layers=tts.num_hidden_layers,
                    num_kv_heads=tts.num_key_value_heads,
                    head_dim=tts.head_dim,
                    max_seq_len=tts.max_position_embeddings,
                    num_qo_heads=tts.num_attention_heads,
                ),
            ),
            AttentionSpec(resource_key=TTS_ATTN, nodes={TTS}, config=AttentionConfig(kv_cache=TTS_KV)),
            PositionSpec(
                resource_key=TTS_POS, nodes={TTS},
                # a TTS step is a few small kernels a layer: fused RoPE saves 5-10% of
                # its GPU time (on the LLM it measured slower at batch 32)
                config=PositionConfig(kv_cache=TTS_KV, rope_theta=tts.rope_theta, fused=True),
            ),
            # upstream's TTS sampling (see TTSSubmodule)
            SamplerSpec(
                resource_key=TTS_SAMPLER, nodes={TTS},
                vocab_size=tts.num_audio_tokens, enable_repetion_penalty=False,
                max_repetition_window=TTS_SAMPLING.penalty_window,
                min_tokens_stop_ids=(tts.eos_code,),
                enable_top_p_first=True,
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
        tts = TTS_SAMPLING
        return {
            LLM_SAMPLER: SamplingReqConfig(**knobs),
            TTS_SAMPLER: SamplingReqConfig(
                temperature=tts.temperature, top_p=tts.top_p, top_k=tts.top_k,
                repetition_penalty=tts.repetition_penalty, repetition_window=tts.penalty_window,
                min_tokens=tts.min_new_tokens, top_p_first=True, top_p_min_keep=tts.top_p_min_keep,
            ),
        }

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
            name=VISION, input_names=["pixel_values", "image_tgt_sizes"],
            outputs=[GraphEdge(next_node=LLM, name="vision_embeds")],
        )
        audio = GraphNode(
            name=AUDIO, input_names=["audio_features", "audio_feature_lens"],
            outputs=[GraphEdge(next_node=LLM, name="audio_embeds")],
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name=LLM,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=LLM, name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                    # per-step (input token, its hidden) of a spoken reply, for
                    # the accumulation below; nothing reads them per step
                    GraphEdge(next_node=EMPTY_DESTINATION, name="tts_ids"),
                    GraphEdge(next_node=EMPTY_DESTINATION, name="tts_hidden"),
                ],
            ),
            # a safety bound; the per-request budget is `check_stop`'s
            max_iters=self.config.llm.max_position_embeddings,
            outputs=[],
            # the reply's TTS condition, handed over whole when it ends
            accumulated_outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="tts_ids", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="tts_hidden", persist=True),
            ],
        )
        tts_outputs = [
            StreamingGraphEdge(next_node=TOKEN2WAV, name="tts_code", target_partition=TOKEN2WAV),
        ]
        tts_prefill = GraphNode(
            name=TTS, input_names=["tts_ids", "tts_hidden"],
            outputs=tts_outputs + [
                GraphEdge(next_node=EMPTY_DESTINATION, name="tts_code", persist=True),
            ],
        )
        tts_decode = Loop(
            name=TTS_DECODE_LOOP,
            section=GraphNode(
                name=TTS,
                input_names=["tts_code"],
                outputs=tts_outputs + [
                    GraphEdge(next_node=TTS, name="tts_code"),
                ],
            ),
            max_iters=TTS_SAMPLING.max_new_tokens,
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
            TTS_PREFILL: tts_prefill,
            TTS_DECODE: tts_decode,
            T2W_CHUNK: GraphNode(
                name=TOKEN2WAV, input_names=["tts_code"],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
            ),
        }

    def get_partitions(self) -> list[PartitionDefinition]:
        main = set(self.get_graph_walk_graphs()) - {T2W_CHUNK}
        return [
            PartitionDefinition(name=MAIN, graph_walks=main, initial_walk=None, producer_partitions=[]),
            PartitionDefinition(
                name=TOKEN2WAV, graph_walks={T2W_CHUNK}, initial_walk=T2W_CHUNK, producer_partitions=[MAIN],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        from mstar.model.minicpm_o.components.token2wav import HOP, LEAD_SILENCE

        return PartitionTopology(
            partitions=[MAIN, TOKEN2WAV],
            connections=[Connection(
                from_partition=MAIN, to_partition=TOKEN2WAV, edge_name="tts_code",
                # upstream's stream windows: 25 new codes, the previous window's
                # last 3 again as the encoder's look-back
                chunk_policy_factory=lambda: LeftContextChunkPolicy(chunk=HOP, left_context=LEAD_SILENCE),
            )],
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _render(
        self,
        parts: list[PromptPart],
        images: list,
        audios: list[np.ndarray],
        enable_thinking: bool,
        system_prompt: str | None,
        voice_audio: np.ndarray | None,
        speech: bool,
    ) -> tuple[str, list[int]]:
        """The chat string as upstream's `chat` renders it, and which message
        each audio sits in (the processor merges audios of one message).

        A spoken reply's system message is upstream's voice prompt around the
        voice's reference clip (``voice_audio``, then the first of ``audios``);
        without it, the request's ``system_prompt``."""
        msgs: list[dict] = []
        audio_parts: list[int] = []
        if voice_audio is not None:
            msgs.append({"role": "system", "content": "\n".join(
                [VOICE_PROMPT_PREFIX, "<audio>./</audio>", VOICE_PROMPT_SUFFIX]
            )})
            audio_parts.append(0)
        elif system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        user: list[str] = []
        for part in parts:
            if part.modality == TEXT:
                user.append(part.text or "")
            elif part.modality == "image":
                user.append("<image>./</image>")
            elif part.modality == "audio":
                user.append("<audio>./</audio>")
                audio_parts.append(len(msgs))
        msgs.append({"role": "user", "content": "\n".join(user)})
        text = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            # upstream turns the speech template on whenever audio is anywhere
            # in the request, and a spoken reply needs it
            use_tts_template=bool(audios) or speech,
            enable_thinking=enable_thinking,
        )
        return text, audio_parts

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        unsupported = set(output_modalities) - {TEXT, "audio"}
        if unsupported:
            raise ValueError(f"MiniCPM-o outputs text and audio; got {sorted(unsupported)}")
        speech = "audio" in output_modalities
        tensors = tensors or {}
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT] if prompt_parts is not None else prompt,
        )
        bad = {p.modality for p in parts} - {TEXT, "image", "audio"}
        if bad:
            raise ValueError(f"MiniCPM-o takes text, image and audio inputs; got {sorted(bad)}")
        raw_images = tensors.get("image_inputs", [])
        raw_audios = tensors.get("audio_inputs", [])
        check_attachments(parts, {"image": len(raw_images), "audio": len(raw_audios)})
        if not any(p.text for p in parts if p.modality == TEXT) and not raw_images and not raw_audios:
            raise ValueError("MiniCPM-o got a request with no text and no attachments")

        images = [_as_pil(img) for img in raw_images]
        audios = [a.reshape(-1).float().cpu().numpy() for a in raw_audios]
        # `voice_prompt=False` leaves the voice clip out of the system message
        # (the reply is still spoken in the voice): what vllm-omni's examples
        # send, so the two can be compared on the same conditioning
        voice_audio = self.voice_audio(kwargs.get("voice")) if speech else None  # validates the voice
        if not kwargs.get("voice_prompt", True):
            voice_audio = None
        if voice_audio is not None:
            audios = [voice_audio] + audios
        text, audio_parts = self._render(
            parts, images, audios,
            enable_thinking=bool(kwargs.get("enable_thinking", False)),
            system_prompt=kwargs.get("system_prompt"),
            voice_audio=voice_audio,
            speech=speech,
        )
        out = self.processor(
            [text], [images], [audios], [audio_parts],
            max_slice_nums=kwargs.get("max_slice_nums"),
            return_tensors="pt",
        )
        input_ids = out["input_ids"][0].long()
        result: NameToTensorList = {"text_inputs": [input_ids]}

        if images:
            strips = out["pixel_values"][0]
            # [3, p, n * p] strips -> [n, 3, p, p] patches; bf16 is what the
            # tower casts them to before its patch conv, so the cast is exact
            p = self.config.vision.patch_size
            patches = torch.cat([
                s.view(3, p, s.shape[-1] // p, p).permute(2, 0, 1, 3) for s in strips
            ]).to(torch.bfloat16)
            positions = _bound_positions(out["image_bound"][0])
            if positions.numel() != len(strips) * self.config.resampler.num_queries:
                raise ValueError(
                    f"image placeholders ({positions.numel()}) do not match "
                    f"{len(strips)} slices x {self.config.resampler.num_queries} queries"
                )
            result["pixel_values"] = [patches]
            result["image_tgt_sizes"] = [out["tgt_sizes"][0].int()]
            result["image_positions"] = [positions]

        if audios:
            lens = out["audio_feature_lens"][0].reshape(-1).long()
            feats = out["audio_features"]
            pieces = [feats[i, :, :n] for i, n in enumerate(lens.tolist())]
            positions = _bound_positions(out["audio_bounds"][0])
            tokens = sum(self.config.audio.pooled_tokens(n) for n in lens.tolist())
            if positions.numel() != tokens:
                # upstream sizes the placeholder off the whole clip but encodes
                # 30 s pieces, which disagree for some long clips
                raise ValueError(
                    f"audio placeholders ({positions.numel()}) do not match the "
                    f"{tokens} tokens its {len(pieces)} piece(s) encode to"
                )
            result["audio_features"] = [torch.cat(pieces, dim=-1).to(torch.bfloat16)]
            result["audio_feature_lens"] = [lens]
            result["audio_positions"] = [positions]
        return result

    def get_voices(self) -> list[str]:
        return list(VOICES)

    def get_default_voice(self) -> str:
        return DEFAULT_VOICE

    def voice_audio(self, voice: str | None) -> np.ndarray:
        """A bundled voice's 16 kHz reference clip, which the LLM hears in the
        system prompt (and token2wav clones)."""
        name = voice or DEFAULT_VOICE
        if name not in VOICES:
            raise ValueError(f"unknown MiniCPM-o voice {name!r}; choose one of {sorted(VOICES)}")
        cached = self._voices.get(name)
        if cached is None:
            import soundfile as sf

            audio, rate = sf.read(Path(self.local_dir) / "assets" / VOICES[name], dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError(f"voice {name!r} is not 16 kHz mono")
            cached = self._voices[name] = audio
        return cached

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == TEXT:
            ids = [t for t in output.reshape(-1).tolist() if t not in self.config.stop_token_ids]
            return self.tokenizer.decode(ids).encode("utf-8")
        if modality == "audio":
            pcm = (output.reshape(-1).float().clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            return pcm.cpu().numpy().tobytes()
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
        model_kwargs = model_kwargs or {}
        speech = "audio" in output_modalities
        if partition_name == TOKEN2WAV:
            # self-triggered by its stream; idle unless the reply is spoken
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities, output_modalities=output_modalities,
                graph_walk=T2W_CHUNK, is_prefill=False,
                kwargs={"voice": model_kwargs.get("voice") or DEFAULT_VOICE},
            )
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=not speech,
                step_metadata={"voice": metadata.kwargs["voice"]},
            )
        walk = self._prefill_walk(input_signals)
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
            kwargs={"audio_output": "audio" in output_modalities},
        )
        inputs = self._walk_inputs(walk, input_signals)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum((e.tensor_info for e in inputs), start=[]),
            step_metadata=self._step_metadata(metadata),
        )

    @staticmethod
    def _step_metadata(metadata: CurrentForwardConductorMetadata) -> dict:
        # every walk carries it: the LLM's decode keeps the TTS condition only
        # for a spoken reply
        return {"audio_output": metadata.kwargs["audio_output"]}

    @staticmethod
    def _carry(
        node: str, names: tuple[str, ...], persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        edges = []
        for name in names:
            edge = GraphEdge(next_node=node, name=name)
            edge.tensor_info = list(persist_signals.get(name, []))
            edges.append(edge)
        return edges

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """prefill -> decode -> (spoken reply) tts_prefill -> tts_decode -> done.

        The loops stop in their submodules' ``check_stop``; arriving here after
        one means it finished."""
        metadata = partition_metadata
        if partition_name == TOKEN2WAV:
            # The stream buffer decides when it is done (after flushing the last
            # window); nothing here predicts it from counts.
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[],
                step_metadata={"voice": metadata.kwargs["voice"]},
            )
        walk = metadata.graph_walk
        speech = metadata.kwargs["audio_output"]
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
            inputs = self._carry(LLM, ("new_token",), persist_signals)
            inputs[0].name = "text_inputs"
        elif walk == "decode" and speech:
            metadata.graph_walk = TTS_PREFILL
            inputs = self._carry(TTS, ("tts_ids", "tts_hidden"), persist_signals)
        elif walk == TTS_PREFILL:
            metadata.graph_walk = TTS_DECODE
            inputs = self._carry(TTS, ("tts_code",), persist_signals)
        else:
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum((e.tensor_info for e in inputs), start=[]),
            step_metadata=self._step_metadata(metadata),
        )

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

    def _create_token2wav(self, weights: str, device: str) -> NodeSubmodule:
        from mstar.model.minicpm_o import submodules
        from mstar.model.minicpm_o.components.token2wav import load_token2wav

        t2w = load_token2wav(str(Path(weights) / "assets" / "token2wav"), device=device)
        # the reference prepares voice prompts on the CPU; it is once per voice
        t2w.voice_encoder.cpu()
        voices = {
            name: t2w.prepare_voice(t2w.voice_encoder(torch.from_numpy(self.voice_audio(name))))
            for name in VOICES
        }
        logger.info("Loaded MiniCPM-o token2wav with voices %s onto %s", sorted(voices), device)
        return submodules.Token2WavSubmodule(t2w.requires_grad_(False), voices, self.tts_config.eos_code)

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

        if node_name not in (LLM, VISION, AUDIO, TTS, TOKEN2WAV):
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

        if node_name == TOKEN2WAV:
            return self._create_token2wav(weights, device)
        if node_name == TTS:
            from mstar.model.loader.iterators import iter_safetensors_shards
            from mstar.model.minicpm_o.components.tts import MiniCPMTTS

            tts = self._build(
                lambda: MiniCPMTTS(self.tts_config, attn_key=TTS_ATTN, kv_key=TTS_KV, pos_key=TTS_POS),
                dtype, device,
            )
            tts.load_weights(
                (name.removeprefix("tts."), tensor)
                for name, tensor in iter_safetensors_shards(weights, device=device, prefix="tts.")
            )
            return submodules.TTSSubmodule(tts.requires_grad_(False).eval(), self.config, TTS_SAMPLING)

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
