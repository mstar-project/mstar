"""MiniCPM-o 4.5: a Qwen3-8B LLM over SigLIP image slices and Whisper audio,
on the shared VLM scaffold (`MultimodalDecoderModel`), with a spoken reply's
TTS and vocoder beside it.

Graph:
    prefill_text         LLM
    prefill_image        vision_encoder -> LLM
    prefill_audio        audio_encoder -> LLM
    prefill_image_audio  (vision_encoder || audio_encoder) -> LLM
    decode               LLM, looped (a text reply)
    decode_spoken        LLM, looped, accumulating the TTS's input (a spoken reply)
    tts_prefill          TTS -> Token2Wav, streamed
    tts_decode           TTS, looped -> Token2Wav, streamed
    t2w_chunk            Token2Wav (its own partition) -> client audio

A request runs one prefill walk over its whole rendered prompt, picked by the
media it holds, then decodes; a spoken reply's TTS then turns the decode's
tokens and hiddens into codes, which Token2Wav vocodes window by window. Text
prefill and both decode walks share a step as ``mixed``.

Images and audio share upstream's ``<unk>`` placeholder: each image slice and
each audio clip is one run of it, and the media layout, not the token, places
the encoders' rows. A spoken reply's voice clip is a system-turn audio
attachment like any other.

Half-duplex chat only: the duplex/streaming-session machinery upstream is not
ported.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from transformers import Qwen2Tokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import PartitionDefinition
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
from mstar.engine.resources.kv.config import DEFAULT_PAGE_SIZE
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import KVLabel, LabelLayout, ProcessPromptOutput, Span, WalkStep
from mstar.model.components.audio_features import HOP_LENGTH, LogMelSpectrogram
from mstar.model.components.sampled_token_decoder import ROW_HIDDEN, ROW_IDS
from mstar.model.keys import KV_CACHE, ROPE, VISION_ATTN
from mstar.model.minicpm_o import processor, weight_loader
from mstar.model.minicpm_o.components.audio import MiniCPMOAudio
from mstar.model.minicpm_o.components.token2wav import (
    CACHE_FAMILIES,
    DIT_COMPUTE_DTYPE,
    HOP,
    LEAD_SILENCE,
    CacheCapacity,
    load_token2wav,
    slot_layout,
)
from mstar.model.minicpm_o.components.tts import MiniCPMTTS, TTSConfig
from mstar.model.minicpm_o.components.vision import MiniCPMOVision
from mstar.model.minicpm_o.config import (
    ATTN,
    AUDIO_ATTN,
    AUDIO_ENCODER,
    DECODE_SPOKEN,
    IMAGE_ENCODER,
    MAX_SLICE_NUMS_KEY,
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
    MiniCPMOConfig,
    TTSSampling,
)
from mstar.model.minicpm_o.submodules import (
    AudioEncoderSubmodule,
    MiniCPMOLLM,
    Token2WavSubmodule,
    TTSSubmodule,
    VisionEncoderSubmodule,
)
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    PromptPart,
    decode_walk,
    media_walks,
    messages_from_parts,
    parts_from_modalities,
)
from mstar.model.multimodal_decoder import MIXED_SPLIT, MediaIntake, MultimodalDecoderModel, PreparedMedia
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import LeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

LLM = "LLM"
VISION = IMAGE_ENCODER.node
AUDIO = AUDIO_ENCODER.node
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

# Upstream's `get_sys_prompt(mode="audio_assistant", language="en")`, around
# the voice's reference audio.
VOICE_PROMPT_PREFIX = "Clone the voice in the provided audio prompt."
VOICE_PROMPT_SUFFIX = (
    "Please assist users while maintaining this voice style. Please answer the user's questions "
    "seriously and in a high quality. Please chat with the user in a highly human-like and oral style. "
    "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
)

# Upstream's `chat` defaults when sampling, in place of the checkpoint's
# generation_config.json. It generates from embeddings, so its repetition
# penalty sees only the reply.
SAMPLING_DEFAULTS = MappingProxyType(dict(
    temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.02, penalize_prompt=False,
))


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


def _max_slice_nums(request: Mapping) -> int:
    """The request's ``max_slice_nums`` (`process_prompt` checks it), or the checkpoint's."""
    return request.get(MAX_SLICE_NUMS_KEY) or processor.MAX_SLICE_NUMS


def _image_size(image: torch.Tensor) -> tuple[int, int]:
    """A ``(C, H, W)`` image's ``(width, height)``, as PIL gives it."""
    return int(image.shape[-1]), int(image.shape[-2])


class MiniCPMOModel(MultimodalDecoderModel):
    NAME = "MiniCPM-o"
    # config and tokenizer, and the voices' clips; workers fetch weights (`_resolve_weights`)
    METADATA_FILES = ("*.json", "*.txt", "assets/*.wav")
    # the media layout, and a spoken reply's text cap, which process_prompt settles
    INTERNAL_KWARGS = (MEDIA_LAYOUT, SPEECH_TEXT_TOKENS)
    sampling_defaults = SAMPLING_DEFAULTS

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None, max_seq_len: int | None = None, **kwargs: Any,
    ):
        super().__init__(model_path_hf, cache_dir)
        # not the checkpoint's generation_config.json: upstream's `chat` sets its own
        self.sampling_defaults = dict(SAMPLING_DEFAULTS)  # a plain dict: the model is pickled
        self.config = MiniCPMOConfig.from_hf(self.local_dir)
        limit = self.config.llm.max_position_embeddings
        self.max_seq_len = limit if max_seq_len is None else max_seq_len
        if type(self.max_seq_len) is not int or not 0 < self.max_seq_len <= limit:
            raise ValueError(f"max_seq_len must be in 1..{limit} (max_position_embeddings), got {max_seq_len!r}")
        self.tts_config = TTSConfig.from_hf(self.config.tts_config)
        # upstream's tokenizer class is Qwen2's with token-name attributes added
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
        # Each image slice and each audio clip is one run of `<unk>`, wrapped in
        # its sentinels by the rendered prompt (`_prompt_ids`). A `<unk>` the
        # user typed follows none of them and stays a token, as upstream's.
        tok = tokenizer.convert_tokens_to_ids
        unk = tok("<unk>")
        self.media = (
            MediaIntake(
                "image", IMAGE_ENCODER, unk, self._image_intake, runs=self._image_runs,
                opens=frozenset({tok("<image>"), tok("<slice>")}),
            ),
            MediaIntake(
                "audio", AUDIO_ENCODER, unk, self._audio_intake, runs=self._audio_runs,
                opens=frozenset({tok("<|audio_start|>")}),
            ),
        )
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

        def slots(key: str, default: int) -> int:
            return int((resources.get(key) or {}).get("max_slots", default))

        state = slots(T2W_STATE, T2W_DEFAULT_SLOTS)
        kv = {key: slots(key, T2W_DEFAULT_SLOTS - 1) for key in T2W_KV.values()}
        if any(n != state - 1 for n in kv.values()):
            raise ValueError(
                f"{config_path}: resources.{T2W_STATE}.max_slots ({state}) must be one more than "
                f"each of {', '.join(f'{k}.max_slots ({n})' for k, n in kv.items())}"
            )

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

    def _resolve_weights(self) -> str:
        """The weights and the vocoder's assets, nothing else of the snapshot."""
        if Path(self.model_path_hf).is_dir():
            return self.model_path_hf
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
            allow_patterns=["*.safetensors", "*.json", "assets/token2wav/*.pt", "assets/token2wav/*.onnx"],
        )

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        llm, vision, resampler, audio = (
            self.config.llm, self.config.vision, self.config.resampler, self.config.audio,
        )
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={LLM},
                config=PagedKVConfig(
                    num_layers=llm.num_hidden_layers,
                    num_kv_heads=llm.num_key_value_heads,
                    head_dim=llm.head_dim,
                    max_seq_len=self.max_seq_len,
                    num_qo_heads=llm.num_attention_heads,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN, nodes={LLM}, config=AttentionConfig(kv_cache=KV_CACHE, splits=(MIXED_SPLIT,)),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={LLM},
                config=PositionConfig(kv_cache=KV_CACHE, rope_theta=llm.rope_theta),
            ),
            # upstream's `chat` hands min_p to HF `generate`, which honours it
            replace(self._sampler_spec(llm.vocab_size), enable_min_p=True),
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
            *self._t2w_kv(),
        ]

    def _voice_seconds(self) -> tuple[float, float]:
        """The shortest and longest bundled voice."""
        import soundfile as sf

        seconds = [sf.info(str(Path(self.local_dir) / "assets" / f)).duration for f in VOICES.values()]
        return min(seconds), max(seconds)

    def _t2w_capacity(self):
        """Token2wav's caches sized for the longest voice."""
        # the s3 tokenizer runs at 25 Hz; one token of slack for rounding
        return CacheCapacity(int(self._voice_seconds()[1] * 25) + 1)

    def _t2w_slot_layout(self) -> dict:
        return slot_layout(self._t2w_capacity(), dit_dtype=DIT_COMPUTE_DTYPE)

    def _t2w_kv(self) -> list[BoundedKVSpec]:
        """Token2wav's attention caches, sized for every voice length allowed."""
        voice_tokens = self._t2w_capacity().prompt_tokens
        # one token of slack for rounding
        min_tokens = max(0, int(self._voice_seconds()[0] * 25) - 1)
        return [
            BoundedKVSpec(
                resource_key=T2W_KV[name], nodes={TOKEN2WAV},
                config=BoundedKVConfig(
                    num_layers=family.num_layers,
                    num_heads=family.num_heads,
                    head_dim=family.head_dim,
                    rows_per_request=family.rows,
                    max_source_len=family.rate * voice_tokens,
                    min_source_len=family.rate * min_tokens,
                    retention=family.retention,
                    max_slots=T2W_DEFAULT_SLOTS - 1,
                    reverse_step_order=family.reverse_step_order,
                    dtype=DIT_COMPUTE_DTYPE if name == "dit" else torch.float32,
                ),
            )
            for name, family in CACHE_FAMILIES.items()
        ]

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
                # a TTS step is a few small kernels a layer: fused RoPE saved 5-10% of its
                # GPU time per step (4e880c21d1: 1 x H100, TTS alone), not re-measured since
                config=PositionConfig(kv_cache=TTS_KV, rope_theta=tts.rope_theta, fused=True),
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
        tts = TTS_SAMPLING
        return {
            **super().get_request_resource_configs(input_modalities, output_modalities, model_kwargs),
            TTS_SAMPLER: SamplingReqConfig(
                temperature=tts.temperature, top_p=tts.top_p, top_k=tts.top_k,
                repetition_penalty=tts.repetition_penalty, repetition_window=tts.penalty_window,
                min_tokens=tts.min_new_tokens, top_p_first=True, top_p_min_keep=tts.top_p_min_keep,
            ),
        }

    def request_kwargs(self) -> frozenset[str]:
        return super().request_kwargs() | {MAX_SLICE_NUMS_KEY, "system_prompt", "voice", "voice_prompt"}

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # the scaffold's, its decode loop bounded by max_seq_len (a safety bound;
        # the per-request budget is `get_loop_caps`')
        walks = media_walks(self.max_seq_len, self.encoders)
        # A spoken reply's decode also hands the TTS each step's input token and
        # its hidden state, accumulated until the reply ends; nothing reads them
        # per step.
        hidden = (ROW_IDS, ROW_HIDDEN)
        walks[DECODE_SPOKEN] = decode_walk(self.max_seq_len, LLM, accumulated=hidden)
        tts_outputs = [
            StreamingGraphEdge(next_node=TOKEN2WAV, name="tts_code", target_partition=TOKEN2WAV),
        ]
        walks[TTS_PREFILL] = GraphNode(
            name=TTS, input_names=list(hidden),
            outputs=tts_outputs + [
                GraphEdge(next_node=EMPTY_DESTINATION, name="tts_code", persist=True),
            ],
        )
        walks[TTS_DECODE] = Loop(
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
        walks[T2W_CHUNK] = GraphNode(
            name=TOKEN2WAV, input_names=["tts_code"],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
            # Not speculated: a window's step syncs on the host (the inverse STFT), so
            # overlap hides little, and a batch picked after the step's postprocess
            # can carry that step's rows again.
            enable_async_scheduling=False,
        )
        return walks

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
            )],
        )

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {**super().kv_labels(), TTS_KV: {"main": KVLabel(growth=((TTS_DECODE, TTS_DECODE_LOOP),))}}

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """The scaffold's intake, after MiniCPM-o's own request knobs: a
        spoken reply's voice clip joins the prompt as a system-turn audio
        attachment. Then the TTS's KV layout and a spoken reply's text cap."""
        unsupported = set(output_modalities) - {TEXT, "audio"}
        if unsupported:
            raise ValueError(f"MiniCPM-o outputs text and audio; got {sorted(unsupported)}")
        speech = "audio" in output_modalities
        if not isinstance(kwargs.get("enable_thinking", False), bool):
            raise ValueError(f"enable_thinking must be true or false, got {kwargs['enable_thinking']!r}")
        max_slice_nums = kwargs.get(MAX_SLICE_NUMS_KEY)
        max_slice_nums = processor.MAX_SLICE_NUMS if max_slice_nums is None else max_slice_nums
        # the checkpoint's own setting is the most: each slice is up to 1024 patches
        if type(max_slice_nums) is not int or not 1 <= max_slice_nums <= processor.MAX_SLICE_NUMS:
            raise ValueError(
                f"max_slice_nums must be an integer in 1..{processor.MAX_SLICE_NUMS}, got {max_slice_nums!r}"
            )
        if speech:
            # The TTS reads the whole reply and then speaks it, all within its
            # positions; a longer reply would run past them, so speech caps the text.
            requested = kwargs.get("max_output_tokens")
            if requested is not None and int(requested) > self.speech_max_text_tokens:
                raise ValueError(
                    f"a spoken reply holds at most {self.speech_max_text_tokens} text tokens; "
                    f"got max_output_tokens={requested}"
                )
        tensors = dict(tensors or {})
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT] if prompt_parts is not None else prompt,
            [p.role for p in prompt_parts] if prompt_parts is not None else None,
        )
        if not any(p.text for p in parts if p.modality == TEXT) and not any(p.modality != TEXT for p in parts):
            raise ValueError("MiniCPM-o got a request with no text and no attachments")
        metadata = dict(kwargs.get("input_metadata") or {})

        # `voice_prompt=False` leaves the voice clip out of the system message
        # (the reply is still spoken in the voice): what vllm-omni's examples
        # send, so the two can be compared on the same conditioning
        voice = self.voice_audio(kwargs.get("voice")) if speech else None  # validates the voice
        if voice is not None and kwargs.get("voice_prompt", True):
            if kwargs.get("system_prompt") or any(p.role == "system" for p in parts):
                # upstream's voice prompt is the reply's system message, and a second
                # one would change what the voice clone was conditioned on
                raise ValueError(
                    "a spoken MiniCPM-o reply's system message is its voice prompt; "
                    "send voice_prompt=false to use your own system prompt"
                )
            parts = [
                PromptPart(TEXT, VOICE_PROMPT_PREFIX, role="system"),
                PromptPart("audio", role="system"),
                PromptPart(TEXT, VOICE_PROMPT_SUFFIX, role="system"),
                *parts,
            ]
            tensors["audio_inputs"] = [voice, *tensors.get("audio_inputs", [])]
            if metadata.get("audio_inputs"):
                metadata["audio_inputs"] = [{}, *metadata["audio_inputs"]]
        # the parts carry the text; an empty request is refused above
        out = super().process_prompt(
            "" if prompt is None else prompt, [p.modality for p in parts], output_modalities, tensors, parts,
            **{
                **kwargs, "input_metadata": metadata, MAX_SLICE_NUMS_KEY: max_slice_nums,
                # upstream turns the speech template on whenever audio is anywhere in
                # the request, and a spoken reply needs it
                "use_tts_template": speech or bool(tensors.get("audio_inputs")),
            },
        )
        # A spoken reply's TTS prefills its condition in one step: the reply, at most a token
        # per LLM decode step, then two more. That is a bound, laid out as two rows plus a row
        # per LLM step; then a row per code.
        out.metadata["kv_layout"][TTS_KV] = {"main": LabelLayout(
            spans=(Span("rows", TTS_PREFILL, 2, 2, ROW_HIDDEN),),
            growth=((self.DECODE_LOOP, 1), (TTS_DECODE_LOOP, 1)),
        ) if speech else LabelLayout()}
        if speech:
            out.metadata[SPEECH_TEXT_TOKENS] = self.speech_max_text_tokens
        return out

    def _prompt_ids(self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs) -> torch.Tensor:
        """The chat as upstream's `chat` renders it: a message per turn, its
        items joined by newlines, a tag per attachment; each tag then becomes
        upstream's placeholder text with one ``<unk>`` a run, in its sentinels,
        and the ids drop ``<|listen|>``, as upstream's do."""
        tags = {"image": processor.IMAGE_TAG, "audio": processor.AUDIO_TAG}
        messages = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "\n".join(
                item["text"] if item["type"] == TEXT else tags[item["type"]] for item in m["content"]
            )}
            for m in messages_from_parts(parts, default_system=kwargs.get("system_prompt") or None)
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            use_tts_template=kwargs.get("use_tts_template", bool(raw.get("audio"))),
            enable_thinking=kwargs.get("enable_thinking", False),
        )
        max_slice_nums = _max_slice_nums(kwargs)
        text = processor.expand_tags(
            text,
            [
                processor.image_placeholder(i, processor.slice_grid(_image_size(image), max_slice_nums))
                for i, image in enumerate(raw.get("image", []))
            ],
            [processor.audio_placeholder(self._audio_runs(clip, {}, kwargs)) for clip in raw.get("audio", [])],
        )
        listen = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        return torch.tensor([t for t in self.tokenizer.encode(text) if t != listen], dtype=torch.long)

    def _image_runs(self, image: torch.Tensor, metadata: dict, request: Mapping = {}) -> int:
        """The source image's run, and one per slice."""
        return processor.image_runs(processor.slice_grid(_image_size(image), _max_slice_nums(request)))

    def _image_intake(self, image: torch.Tensor, metadata: dict, request: Mapping = {}) -> PreparedMedia:
        """Each slice's patches, its ``num_queries`` tokens and its ``(h, w)``
        in patches, which the tower lays its segments out by."""
        p = self.config.vision.patch_size
        slices, _ = processor.slice_image(_as_pil(image), _max_slice_nums(request), p)
        patches, grids = zip(*(processor.slice_patches(s, p) for s in slices), strict=True)
        queries = self.config.resampler.num_queries
        # bf16 is what the tower casts them to before its patch conv, so the cast is exact
        return PreparedMedia({"pixel_values": torch.cat(patches).to(torch.bfloat16)}, [queries] * len(grids), grids)

    def _audio_runs(self, clip: torch.Tensor, metadata: dict, request: Mapping = {}) -> int:
        """One run a clip, none for a clip too short to pool to a token."""
        return 1 if processor.audio_tokens(clip.numel(), self.config.audio) > 0 else 0

    def _audio_intake(self, clip: torch.Tensor, metadata: dict, request: Mapping = {}) -> PreparedMedia:
        """A clip's log-mel pieces (30 s each at most), cut to their frames
        and joined; its run's tokens, and its pieces' mel lengths, which the
        encoder lays its segments out by."""
        audio = self.config.audio
        clip = clip.reshape(-1).float().cpu()
        pieces = processor.audio_pieces(clip)
        # each piece padded to Whisper's 30 s window, then cut to its own frames
        mel = self.log_mel(torch.stack([self.log_mel.pad_or_trim(piece) for piece in pieces]))
        # a piece shorter than one pooling step (what is left of a long clip past its
        # last 30 s cut) pools to no tokens
        lens = [
            (i, n) for i, n in enumerate(math.ceil(len(piece) / HOP_LENGTH) for piece in pieces)
            if audio.pooled_tokens(n) > 0
        ]
        tokens = sum(audio.pooled_tokens(n) for _, n in lens)
        if tokens != processor.audio_tokens(clip.numel(), audio):
            # upstream sizes the placeholder off the whole clip but encodes 30 s
            # pieces, which disagree for some long clips
            raise ValueError(
                f"an audio clip's placeholder ({processor.audio_tokens(clip.numel(), audio)} tokens) does not "
                f"match the {tokens} tokens its {len(lens)} piece(s) encode to"
            )
        if not tokens:  # too short to encode: no run (`_audio_runs`)
            return PreparedMedia({}, [])
        features = torch.cat([mel[i, :, :n] for i, n in lens], dim=-1).to(torch.bfloat16)
        return PreparedMedia({"audio_features": features}, [tokens], [[n for _, n in lens]])

    def get_max_output_tokens(self, **model_kwargs) -> int:
        cap = super().get_max_output_tokens(**model_kwargs)
        return min(cap, model_kwargs.get(SPEECH_TEXT_TOKENS, cap))

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the TTS prefill samples the first code, so its loop runs one fewer
        return {**super().get_loop_caps(**model_kwargs), TTS_DECODE_LOOP: TTS_SAMPLING.max_new_tokens - 1}

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

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality == TEXT:
            return super().postprocess(output, modality, request_kwargs)
        if modality == "audio":
            pcm = (output.reshape(-1).float().clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            return pcm.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for MiniCPM-o: {modality!r}")

    # ------------------------------------------------------------------
    # Walks
    # ------------------------------------------------------------------

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """Main: the scaffold's prefill and decode. A spoken reply decodes on
        ``decode_spoken`` instead and then runs the TTS: ``tts_prefill`` over
        the reply's ids and hiddens, then ``tts_decode``, whose codes stream
        to the vocoder.

        Token2Wav runs a chunk per streamed window, and only for a spoken reply."""
        speech = "audio" in output_modalities
        if partition_name == TOKEN2WAV:
            if not speech:
                return []
            return [WalkStep(T2W_CHUNK, step_metadata={"voice": model_kwargs.get("voice") or DEFAULT_VOICE})]
        prefill, decode = super().walk_plan(
            partition_name, input_modalities, output_modalities, input_signals, model_kwargs,
        )
        if not speech:
            return [prefill, decode]
        return [
            prefill,
            replace(decode, walk=DECODE_SPOKEN),
            WalkStep(TTS_PREFILL),
            WalkStep(TTS_DECODE),
        ]

    # ------------------------------------------------------------------
    # Submodules
    # ------------------------------------------------------------------

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
            t2w.requires_grad_(False), voices, self.tts_config.eos_code, dit_dtype=DIT_COMPUTE_DTYPE)

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct on ``meta``, then give it uninitialized ``dtype`` storage on
        ``device`` for the loader to fill (it raises on any parameter left
        unfilled); fp32-only buffers restore themselves in ``_apply``."""
        with torch.device("meta"):
            module = make()
        return module.to(dtype).to_empty(device=device)

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        if node_name not in (LLM, VISION, AUDIO, TTS, TOKEN2WAV):
            return None
        weights = self._resolve_weights()
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
            lambda: Qwen3DenseLM(self.config.llm, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=tp_group),
            dtype, device,
        )
        weight_loader.load_llm_weights(lm, weights, device)
        logger.info("Loaded MiniCPM-o LLM onto %s", device)
        return MiniCPMOLLM(lm.requires_grad_(False).eval(), self.config, self.encoders)
