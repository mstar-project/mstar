"""MiniCPM-o 4.5: a Qwen3-8B LLM over SigLIP image slices and Whisper audio.

Graph:
    prefill_text   LLM
    prefill_image  vision_encoder -> LLM
    prefill_audio  audio_encoder -> LLM
    prefill_omni   (vision_encoder || audio_encoder) -> LLM
    decode         LLM, looped (a text reply)
    decode_spoken  LLM, looped, accumulating the TTS's input (a spoken reply)
    tts_prefill    TTS -> Token2Wav, streamed
    tts_decode     TTS, looped -> Token2Wav, streamed
    t2w_chunk      Token2Wav (its own partition) -> client audio

A request runs one prefill walk over its whole rendered prompt, picked by
which attachments it carries, then decodes; a spoken reply's TTS then turns
the decode's tokens and hiddens into codes, which Token2Wav vocodes window by
window. Text and spoken replies decode in one step as ``decode_mixed``.

The prompt keeps upstream's placeholder tokens; the LLM overwrites their
embeddings with the encoders' outputs at positions ``process_prompt`` reads
off the prompt (see ``submodules``).

Half-duplex chat only: the duplex/streaming-session machinery upstream is not
ported.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2Tokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL, PartitionDefinition
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
from mstar.engine.resources.kv.config import DEFAULT_PAGE_SIZE
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, WalkStep
from mstar.model.components.audio_features import HOP_LENGTH, LogMelSpectrogram
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    DIT_COMPUTE_DTYPE,
    HOP,
    LEAD_SILENCE,
    load_token2wav,
)
from mstar.model.components.token2wav.resources import check_token2wav_slots, token2wav_resources
from mstar.model.components.token2wav.submodule import Token2WavSubmodule
from mstar.model.minicpm_o import processor, weight_loader
from mstar.model.minicpm_o.components.audio import MiniCPMOAudio
from mstar.model.minicpm_o.components.tts import MiniCPMTTS, TTSConfig
from mstar.model.minicpm_o.components.vision import MiniCPMOVision
from mstar.model.minicpm_o.config import (
    AUDIO_ATTN,
    AUDIO_FEATURE_LENS,
    DECODE,
    DECODE_LOOP,
    DECODE_MIXED,
    DECODE_SPOKEN,
    IMAGE_TGT_SIZES,
    LLM_ATTN,
    LLM_KV,
    LLM_POS,
    LLM_SAMPLER,
    RESAMPLER_ATTN,
    SPEECH_TEXT_TOKENS,
    T2W_KV,
    T2W_STATE,
    TTS_ATTN,
    TTS_DECODE,
    TTS_DECODE_LOOP,
    TTS_KV,
    TTS_POS,
    TTS_PREFILL,
    TTS_SAMPLER,
    VISION_ATTN,
    MiniCPMOConfig,
    TTSSampling,
)
from mstar.model.minicpm_o.submodules import (
    AudioEncoderSubmodule,
    LLMSubmodule,
    TTSSubmodule,
    VisionEncoderSubmodule,
)
from mstar.model.multimodal import TEXT, PromptPart, check_attachments, messages_from_parts, parts_from_modalities
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextOutput
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

TTS_SAMPLING = TTSSampling()

# Voices shipped in the checkpoint's assets, by the name a request picks.
# The vocoder's per-request state grows with the longest voice prompt (~0.4 GB
# a request for this 6 s clip), so the checkpoint's 11 s and 17 s system
# voices are left out.
VOICES = {"default": "HT_ref_audio.wav"}
DEFAULT_VOICE = "default"
T2W_DEFAULT_SLOTS = 9
T2W_KEYS = Token2WavKeys(codes="tts_code", state=T2W_STATE, kv=T2W_KV)

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
# the sampler knobs a request may set
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "ignore_eos",
    "return_logprobs", "logprobs_mode",
)



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
    # Preprocessing is a few small ops per request, which a thread per core
    # only slows (Qwen3.5 measured its image preprocessing at 4 threads).
    PREPROCESS_TORCH_THREADS = 4
    # the encoders' segment sizes and a spoken reply's text cap, which process_prompt settles
    INTERNAL_KWARGS = (IMAGE_TGT_SIZES, AUDIO_FEATURE_LENS, SPEECH_TEXT_TOKENS)

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
        self.tts_config = TTSConfig.from_hf(self.config.tts_config)
        tokenizer = Qwen2Tokenizer.from_pretrained(self.local_dir)
        self.tokenizer = tokenizer
        # upstream's mel: Whisper's, as its own audio processor configures it
        self.log_mel = LogMelSpectrogram(num_mel_bins=self.config.audio.num_mel_bins)
        # A reply ends on <|im_end|> (the tokenizer's eos; config.json names
        # only that) or <|endoftext|>, as upstream's `chat` terminators; and
        # a spoken reply's text closes with <|tts_eos|> just before <|im_end|>,
        # which is where the text the TTS reads ends.
        self.config.stop_token_ids = tuple(
            tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|tts_eos|>")
        )
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._voices: dict[str, torch.Tensor] = {}

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """A spoken reply holds a slot of each ``t2w_*`` resource, so one count sizes
        them all: ``t2w_state`` has one more (its sink). Set apart, the smallest caps
        concurrent spoken replies and the rest strand ~0.4 GiB a slot.

        The vocoder's slots don't hold back the TTS: it takes every spoken reply the
        LLM finishes and keeps its pages, up to its positions, until its last code.
        Only ``max_concurrent_requests`` caps those replies, so ``tts_kv`` must hold
        that many full ones; a burst that overfills it stalls with no reply able to
        finish."""
        resources = config.get("resources") or {}

        check_token2wav_slots(T2W_KEYS, resources, T2W_DEFAULT_SLOTS, config_path)

        tts = resources.get(TTS_KV) or {}
        reply = -(-self.tts_config.max_position_embeddings // int(tts.get("page_size") or DEFAULT_PAGE_SIZE))
        pages = int(tts.get("max_num_pages", PagedKVConfig.max_num_pages))
        cap = config.get("max_concurrent_requests")
        # and the pool's sink page
        if cap is None or pages < int(cap) * reply + 1:
            raise ValueError(
                f"{config_path}: resources.{TTS_KV}.max_num_pages ({pages}) must hold "
                f"max_concurrent_requests ({cap}) full spoken replies of {reply} pages, plus a sink page: "
                "nothing else caps the replies the TTS holds at once, and a burst past the pool stalls"
            )

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
                vocab_size=llm.vocab_size, enable_repetition_penalty=True,
                # upstream's `chat` hands min_p to HF `generate`, which honours it
                enable_min_p=True,
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
                    # the pieces a captured block loop's plan holds (a spoken reply's
                    # voice clip and two 30 s pieces, say); a request with more
                    # runs eagerly
                    max_segments_per_request=4,
                ),
            ),
            *self._tts_resources(),
            # ~0.4 GB a slot (one speaking request) for the bundled voices;
            # `t2w_state.max_slots` in the yaml sizes it
            *token2wav_resources(T2W_KEYS, TOKEN2WAV, self._voice_seconds(), T2W_DEFAULT_SLOTS),
        ]

    def _voice_seconds(self) -> tuple[float, float]:
        """The shortest and longest bundled voice."""
        import soundfile as sf

        seconds = [sf.info(str(Path(self.local_dir) / "assets" / f)).duration for f in VOICES.values()]
        return min(seconds), max(seconds)

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
                config=PositionConfig(kv_cache=TTS_KV, rope_theta=tts.rope_theta),
            ),
            # upstream's TTS sampling (see TTSSubmodule)
            SamplerSpec(
                resource_key=TTS_SAMPLER, nodes={TTS},
                vocab_size=tts.num_audio_tokens, enable_repetition_penalty=False,
                max_repetition_window=TTS_SAMPLING.penalty_window,
                min_tokens_stop_ids=(tts.eos_code,),
                enable_top_p_first=True,
            ),
        ]

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        knobs = {**DEFAULT_SAMPLING, **{k: model_kwargs[k] for k in SAMPLING_KEYS if k in model_kwargs}}
        tts = TTS_SAMPLING
        return {
            # upstream generates from embeddings, so its penalty sees only generated tokens
            LLM_SAMPLER: SamplingReqConfig(**knobs, penalize_prompt=False),
            TTS_SAMPLER: SamplingReqConfig(
                temperature=tts.temperature, top_p=tts.top_p, top_k=tts.top_k,
                repetition_penalty=tts.repetition_penalty, repetition_window=tts.penalty_window,
                min_tokens=tts.min_new_tokens, top_p_first=True, top_p_min_keep=tts.top_p_min_keep,
            ),
        }

    def request_kwargs(self) -> frozenset[str]:
        return frozenset(
            {*SAMPLING_KEYS, "enable_thinking", "max_slice_nums", "system_prompt", "voice", "voice_prompt"}
        )

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
        def decode(spoken: bool) -> Loop:
            # A text reply's only output is its token, which the LLM keeps on the
            # device: its steps loop back. A spoken reply's also hands the TTS its
            # per-step (input token, its hidden), for the accumulation below;
            # nothing reads them per step.
            tts = ("tts_ids", "tts_hidden") if spoken else ()
            return Loop(
                name=DECODE_LOOP,
                section=GraphNode(
                    name=LLM,
                    input_names=["text_inputs"],
                    outputs=[
                        GraphEdge(next_node=LLM, name="text_inputs"),
                        GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                        *(GraphEdge(next_node=EMPTY_DESTINATION, name=name) for name in tts),
                    ],
                ),
                # a safety bound; the per-request budget is `get_loop_caps`'
                max_iters=self.max_seq_len,
                outputs=[],
                # the reply's TTS condition, handed over whole when it ends
                accumulated_outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=name, persist=True) for name in tts],
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
            DECODE: decode(spoken=False),
            DECODE_SPOKEN: decode(spoken=True),
            TTS_PREFILL: tts_prefill,
            TTS_DECODE: tts_decode,
            T2W_CHUNK: GraphNode(
                name=TOKEN2WAV, input_names=["tts_code"],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
                # Not speculated: a window's step syncs on the host (the inverse STFT), so
                # overlap hides little, and a batch picked after the step's postprocess
                # can carry that step's rows again.
                enable_async_scheduling=False,
            ),
        }

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        # text and spoken replies decode in one step; a text row still loops back
        return {LLM: {DECODE_MIXED: {DECODE, DECODE_SPOKEN}}}

    def get_partitions(self) -> list[PartitionDefinition]:
        main = set(self.get_graph_walk_graphs()) - {T2W_CHUNK}
        return [
            PartitionDefinition(name=MAIN, graph_walks=main),
            PartitionDefinition(name=TOKEN2WAV, graph_walks={T2W_CHUNK}),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=[MAIN, TOKEN2WAV],
            connections=[Connection(
                from_partition=MAIN, to_partition=TOKEN2WAV, edge_name="tts_code",
                # upstream's stream windows: 25 new codes, the previous window's
                # last 3 again as the encoder's look-back
                chunk_policy_factory=lambda: LeftContextChunkPolicy(chunk=HOP, left_context=LEAD_SILENCE),
                # one code an item, which the vocoder takes as a CPU window
                by_value=True,
            )],
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _render(
        self,
        parts: list[PromptPart],
        has_audio: bool,
        enable_thinking: bool,
        system_prompt: str | None,
        voice_audio: torch.Tensor | None,
        speech: bool,
    ) -> str:
        """The chat string as upstream's `chat` renders it: a message per turn,
        its items joined by newlines, with a tag per attachment.

        A spoken reply's system message is upstream's voice prompt around the
        voice's reference clip (``voice_audio``, the first audio the prompt
        holds; ``process_prompt`` refuses a system message of the request's
        beside it); without it, the request's ``system_prompt``."""
        tags = {"image": processor.IMAGE_TAG, "audio": processor.AUDIO_TAG}
        system = system_prompt or None
        if voice_audio is not None:
            system = "\n".join([VOICE_PROMPT_PREFIX, processor.AUDIO_TAG, VOICE_PROMPT_SUFFIX])
        msgs = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "\n".join(
                item["text"] if item["type"] == TEXT else tags[item["type"]] for item in m["content"]
            )}
            for m in messages_from_parts(parts, default_system=system)
        ]
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            # upstream turns the speech template on whenever audio is anywhere
            # in the request, and a spoken reply needs it
            use_tts_template=has_audio or speech,
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
        unsupported = set(output_modalities) - {TEXT, "audio"}
        if unsupported:
            raise ValueError(f"MiniCPM-o outputs text and audio; got {sorted(unsupported)}")
        speech = "audio" in output_modalities
        enable_thinking = kwargs.get("enable_thinking", False)
        if not isinstance(enable_thinking, bool):
            raise ValueError(f"enable_thinking must be true or false, got {enable_thinking!r}")
        max_slice_nums = kwargs.get("max_slice_nums")
        max_slice_nums = processor.MAX_SLICE_NUMS if max_slice_nums is None else max_slice_nums
        # the checkpoint's own setting is the most: each slice is up to 1024 patches
        if type(max_slice_nums) is not int or not 1 <= max_slice_nums <= processor.MAX_SLICE_NUMS:
            raise ValueError(
                f"max_slice_nums must be an integer in 1..{processor.MAX_SLICE_NUMS}, got {max_slice_nums!r}"
            )
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

        audios = [a.reshape(-1).float().cpu() for a in raw_audios]
        # `voice_prompt=False` leaves the voice clip out of the system message
        # (the reply is still spoken in the voice): what vllm-omni's examples
        # send, so the two can be compared on the same conditioning
        voice_audio = self.voice_audio(kwargs.get("voice")) if speech else None  # validates the voice
        if not kwargs.get("voice_prompt", True):
            voice_audio = None
        if voice_audio is not None:
            if kwargs.get("system_prompt") or any(p.role == "system" for p in parts):
                # upstream's voice prompt is the reply's system message, and a second
                # one would change what the voice clone was conditioned on
                raise ValueError(
                    "a spoken MiniCPM-o reply's system message is its voice prompt; "
                    "send voice_prompt=false to use your own system prompt"
                )
            audios = [voice_audio] + audios
        text = self._render(
            parts, bool(audios),
            enable_thinking=enable_thinking,
            system_prompt=kwargs.get("system_prompt"),
            voice_audio=voice_audio,
            speech=speech,
        )
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
            all_pieces = [piece for a in audios for piece in processor.audio_pieces(a)]
            # each piece padded to Whisper's 30 s window, then cut to its own frames
            mel = self.log_mel(torch.stack([self.log_mel.pad_or_trim(piece) for piece in all_pieces]))
            # a piece shorter than one pooling step (a clip under ~0.1 s, or what
            # is left of a long one past its last 30 s cut) pools to no tokens
            lens = [
                (i, n) for i, n in enumerate(math.ceil(len(piece) / HOP_LENGTH) for piece in all_pieces)
                if self.config.audio.pooled_tokens(n) > 0
            ]
            pieces = [mel[i, :, :n] for i, n in lens]
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
        if speech:
            # The TTS reads the whole reply and then speaks it, all within its
            # positions; a longer reply would run past them, so speech caps the text.
            limit = self.speech_max_text_tokens
            requested = kwargs.get("max_output_tokens")
            if requested is not None and int(requested) > limit:
                raise ValueError(f"a spoken reply holds at most {limit} text tokens; got max_output_tokens={requested}")
            metadata[SPEECH_TEXT_TOKENS] = limit
        # the LLM's prefill writes every prompt row, placeholders included, then a row per decode step
        rows = int(input_ids.numel())
        llm = LabelLayout(
            spans=(Span("rows", self._prefill_walk(result), rows, rows, "text_inputs"),),
            growth=((DECODE_LOOP, 1),),
        )
        # A spoken reply's TTS prefills its condition in one step: the reply, at most a token
        # per LLM decode step, then two more. That is a bound, laid out as two rows plus a row
        # per LLM step; then a row per code.
        tts = LabelLayout(
            spans=(Span("rows", TTS_PREFILL, 2, 2, "tts_hidden"),),
            growth=((DECODE_LOOP, 1), (TTS_DECODE_LOOP, 1)),
        ) if speech else LabelLayout()
        metadata["kv_layout"] = {LLM_KV: {"main": llm}, TTS_KV: {"main": tts}}
        return ProcessPromptOutput(result, metadata)

    def get_max_output_tokens(self, **model_kwargs) -> int:
        cap = super().get_max_output_tokens(**model_kwargs)
        return min(cap, model_kwargs.get(SPEECH_TEXT_TOKENS, cap))

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # each prefill samples the first token or code, so its loop runs one fewer
        return {
            DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1),
            TTS_DECODE_LOOP: TTS_SAMPLING.max_new_tokens - 1,
        }

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {
            LLM_KV: {"main": KVLabel(growth=((DECODE, DECODE_LOOP), (DECODE_SPOKEN, DECODE_LOOP)))},
            TTS_KV: {"main": KVLabel(growth=((TTS_DECODE, TTS_DECODE_LOOP),))},
        }

    @property
    def speech_max_text_tokens(self) -> int:
        """The longest reply the TTS can speak: its condition (the reply, then two
        tokens) and every code it may sample fit its positions."""
        return self.tts_config.max_position_embeddings - 2 - TTS_SAMPLING.max_new_tokens

    def get_voices(self) -> list[str]:
        return list(VOICES)

    def get_default_voice(self) -> str:
        return DEFAULT_VOICE

    def voice_audio(self, voice: str | None) -> torch.Tensor:
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
            cached = self._voices[name] = torch.from_numpy(audio)
        return cached

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), self.config.stop_token_ids, "</think>")

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

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """Main: one prefill walk over the whole prompt, picked by the
        attachments it carries, then decode fed the prefill's token. A spoken
        reply decodes on ``decode_spoken`` instead and then runs the TTS:
        ``tts_prefill`` over the reply's hidden states, then ``tts_decode``,
        whose codes stream to the vocoder. The prefill carries the encoders'
        segment sizes that ``process_prompt`` settled.

        Token2Wav runs a chunk per streamed window, and only for a spoken reply."""
        speech = "audio" in output_modalities
        if partition_name == TOKEN2WAV:
            if not speech:
                return []
            return [WalkStep(T2W_CHUNK, step_metadata={"voice": model_kwargs.get("voice") or DEFAULT_VOICE})]
        settled = {k: model_kwargs[k] for k in self.INTERNAL_KWARGS if k in model_kwargs}
        prefill = WalkStep(self._prefill_walk(input_signals), step_metadata={FINAL_PREFILL: True, **settled})
        if not speech:
            return [prefill, WalkStep(DECODE, bind={"text_inputs": "new_token"})]
        return [
            prefill,
            WalkStep(DECODE_SPOKEN, bind={"text_inputs": "new_token"}),
            WalkStep(TTS_PREFILL),
            WalkStep(TTS_DECODE),
        ]

    # ------------------------------------------------------------------
    # Submodules
    # ------------------------------------------------------------------

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(groups=[], tp_enabled_nodes={LLM}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
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

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
            allow_patterns=["*.safetensors", "*.json", "assets/token2wav/*.pt", "assets/token2wav/*.onnx"],
        )

    def _create_token2wav(self, weights: str, device: str) -> NodeSubmodule:
        t2w = load_token2wav(str(Path(weights) / "assets" / "token2wav"), device=device)
        # the reference prepares voice prompts on the CPU; it is once per voice
        t2w.voice_encoder.cpu()
        voices = {
            name: t2w.prepare_voice(t2w.voice_encoder(self.voice_audio(name)))
            for name in VOICES
        }
        logger.info("Loaded MiniCPM-o token2wav with voices %s onto %s", sorted(voices), device)
        return Token2WavSubmodule(
            t2w.requires_grad_(False), T2W_KEYS, voices, self.tts_config.eos_code, dit_dtype=DIT_COMPUTE_DTYPE)

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct on ``meta``, then give it uninitialized ``dtype`` storage on
        ``device`` for the loader to fill (it raises on any parameter left
        unfilled); fp32-only buffers restore themselves in ``_apply``."""
        with torch.device("meta"):
            module = make()
        return module.to(dtype).to_empty(device=device)

    def _create_submodule(self, node_name: str, device: str, tp_group, dtype: torch.dtype) -> NodeSubmodule | None:

        if node_name not in (LLM, VISION, AUDIO, TTS, TOKEN2WAV):
            return None
        weights = self._weights_dir()
        if node_name == VISION:

            tower = self._build(lambda: MiniCPMOVision(self.config.vision, self.config.resampler), dtype, device)
            weight_loader.load_vision_weights(tower, weights, device)
            return VisionEncoderSubmodule(tower.requires_grad_(False).eval(), self.config)
        if node_name == AUDIO:

            tower = self._build(lambda: MiniCPMOAudio(self.config.audio), dtype, device)
            weight_loader.load_audio_weights(tower, weights, device)
            return AudioEncoderSubmodule(tower.requires_grad_(False).eval(), self.config)

        if node_name == TOKEN2WAV:
            return self._create_token2wav(weights, device)
        if node_name == TTS:
            from mstar.model.loader.iterators import iter_safetensors_shards

            tts = self._build(
                lambda: MiniCPMTTS(self.tts_config, attn_key=TTS_ATTN, kv_key=TTS_KV, pos_key=TTS_POS),
                dtype, device,
            )
            tts.load_weights(
                (name.removeprefix("tts."), tensor)
                for name, tensor in iter_safetensors_shards(weights, device=device, prefix="tts.")
            )
            return TTSSubmodule(tts.requires_grad_(False).eval(), self.config, TTS_SAMPLING)

        from mstar.model.components.qwen3_lm import Qwen3DenseLM

        lm = self._build(
            lambda: Qwen3DenseLM(
                self.config.llm, attn_key=LLM_ATTN, kv_key=LLM_KV, pos_key=LLM_POS, comm_group=tp_group,
            ),
            dtype, device,
        )
        weight_loader.load_llm_weights(lm, weights, device)
        logger.info("Loaded MiniCPM-o LLM onto %s", device)
        return LLMSubmodule(lm.requires_grad_(False).eval(), self.config)
