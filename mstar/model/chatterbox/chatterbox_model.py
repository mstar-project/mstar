"""Chatterbox / Chatterbox-Turbo text-to-speech for M*.

Architecture (three nodes, two asynchronous partitions)::

    voice_encoder  reference audio -> speaker embedding (LSTM voice encoder)
                   + S3 prompt tokens (S3TokenizerV2, first 6 s / 15 s)
    T3             Llama-520M (Turbo: GPT-2-medium) over
                   [cond | text | BOS] -> S3 speech tokens, 25 Hz
    s3gen          S3 tokens -> mel (flow matching, reference-conditioned)
                   -> waveform (HiFT) -> Perth watermark -> 24 kHz PCM16

Walks and partitions::

    T3 partition:    prefill        (built-in voice)       T3
                     prefill_voice  (uploaded/preset voice) voice_encoder -> T3
                     decode         Loop over T3, one speech token per step
    S3Gen partition: s3gen_chunk / s3gen_chunk_voice, fed by the
                     ``speech_tokens`` stream from T3

Classifier-free guidance is two KV streams per request (``main`` with the
text, ``uncond`` with the text embeddings zeroed) packed into one attention
plan, so a CFG decode step is one forward over 2B rows; ``cfg_weight`` is a
per-request tensor input, so decode replays a single captured CUDA graph per
batch size and guidance mode. Turbo has no guidance.

The reference implementation (``chatterbox/tts.py``, ``chatterbox/tts_turbo.py``)
was read for every contract here; nothing of it runs at serve time.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.chatterbox.components.audio_frontend import resample, trim_silence
from mstar.model.chatterbox.config import (
    COND_LABEL,
    S3_SR,
    S3GEN_NODE,
    S3GEN_SR,
    T3_ATTN,
    T3_KV,
    T3_NODE,
    T3_POS,
    T3_SAMPLER,
    UNCOND_LABEL,
    VOICE_ENCODER_NODE,
    ChatterboxConfig,
)
from mstar.model.chatterbox.loader import resolve_snapshot
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import FixedChunkPolicy, RampChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

logger = logging.getLogger(__name__)

T3_PARTITION = "T3"
S3GEN_PARTITION = "S3Gen"

# Edge names
TEXT_INPUTS = "text_inputs"
REF_AUDIO = "ref_audio"          # 24 kHz mono waveform of the reference voice
VOICE_KEY = "voice_key"          # content hash of that waveform, for the voice caches
SPEAKER_EMB = "speaker_emb"
PROMPT_TOKENS = "prompt_tokens"
SPEECH_TOKENS = "speech_tokens"  # T3 output: streamed to S3Gen, persisted for decode
PREV_TOKEN = "prev_token"        # T3 decode input: the previous speech token
AUDIO_CHUNK = "audio_chunk"

BUILTIN_VOICE = "default"
_BUILTIN_VOICE_ALIASES = {None, "", BUILTIN_VOICE, "builtin", "built-in"}

# Per-request generation knobs that ride on the conductor metadata
_T3_KNOBS = ("cfg_weight", "exaggeration", "min_p", "max_new_tokens")
_S3GEN_KNOBS = ("n_cfm_timesteps", "watermark")

MAX_REFERENCE_SECONDS = 30.0
# Sound left after the voice encoder's silence trim; shorter clips crash its
# STFT (under 12.5 ms) or clone noise (40 ms gave WER 1.0)
MIN_REFERENCE_SECONDS = 0.5
# The trim is relative to the clip's own peak, so it keeps all of a silent one;
# a clip that never swings past -60 dBFS around its mean is refused instead
MIN_REFERENCE_PEAK = 1e-3
# Bounds S3Gen time per chunk; the reference uses 10 (Turbo 2)
MAX_CFM_TIMESTEPS = 100
# The conductor's seed is an int64; sentence chunking wraps its per-chunk seeds into it
MAX_SEED = 2**63 - 1
# The reference demo's slider maxima. Far past them T3 runs to max_new_tokens,
# and a value that overflows the sampler's fp32 row fails the whole batch.
MAX_TEMPERATURE = 5.0
MAX_REPETITION_PENALTY = 2.0
MAX_CFG_WEIGHT = 1.0
MAX_EXAGGERATION = 2.0


class ChatterboxModel(Model):
    """Model contract: prompt processing, graph, partitions, resources and the
    per-partition state machine. No GPU compute lives here."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        variant: str | None = None,
        voices_dir: str | None = None,
        watermark: bool | None = None,
        stream_chunk_tokens: int | None = None,
        stream_first_chunk_tokens: int | None = None,
        stream_context_tokens: int | None = None,
        stream_chunk_growth: float | None = None,
        stream_max_chunk_tokens: int | None = None,
        s3gen_compile: bool | None = None,
        s3gen_compile_mode: str | None = None,
        s3gen_frame_bucket: int | None = None,
        s3gen_graphs: bool | None = None,
        s3gen_graph_stages: str | None = None,
        s3gen_estimator_dtype: str | None = None,
        t3_prefill_graphs: bool | None = None,
        s3gen_max_batch_size: int | None = None,
        t3_dtype: str | None = None,
        default_language: str | None = None,
        max_new_tokens_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        # T3 runs in bf16 by default (``t3_dtype: float16`` is the other
        # option); the engine sizes the KV cache and plans attention in it too
        self._t3_dtype = _parse_dtype(t3_dtype) if t3_dtype else torch.bfloat16
        if self._t3_dtype == torch.float32:
            # the engine sizes the KV cache and plans attention in this dtype,
            # and FlashInfer has no float32 kernels; fp32 token parity is
            # covered by the CPU tests (test/chatterbox) instead
            raise ValueError(
                "t3_dtype float32 cannot be served: the paged attention runs in "
                "bfloat16/float16 only. Use bfloat16 (default) or float16."
            )
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.config = (
            ChatterboxConfig.from_variant(variant)
            if variant
            else ChatterboxConfig.from_model_path(model_path_hf)
        )
        if watermark is not None:
            self.config.generation.watermark = bool(watermark)
        if stream_chunk_tokens is not None:
            self.config.stream_chunk_tokens = int(stream_chunk_tokens)
        if stream_first_chunk_tokens is not None:
            self.config.stream_first_chunk_tokens = int(stream_first_chunk_tokens)
        if stream_context_tokens is not None:
            self.config.stream_context_tokens = int(stream_context_tokens)
        if stream_chunk_growth is not None:
            self.config.stream_chunk_growth = float(stream_chunk_growth)
        if stream_max_chunk_tokens is not None:
            self.config.stream_max_chunk_tokens = int(stream_max_chunk_tokens)
        if s3gen_compile is not None:
            self.config.s3gen_compile = bool(s3gen_compile)
        if s3gen_compile_mode is not None:
            self.config.s3gen_compile_mode = str(s3gen_compile_mode)
        if s3gen_frame_bucket is not None:
            self.config.s3gen_frame_bucket = int(s3gen_frame_bucket)
        if s3gen_graphs is not None:
            self.config.s3gen_graphs = bool(s3gen_graphs)
        if s3gen_graph_stages is not None:
            self.config.s3gen_graph_stages = str(s3gen_graph_stages)
        stages = {s.strip() for s in self.config.s3gen_graph_stages.split(",") if s.strip()}
        if unknown := stages - {"solve", "encoder", "vocoder"}:
            raise ValueError(f"unknown s3gen_graph_stages {sorted(unknown)}; choose from solve, encoder, vocoder")
        self._graph_stages = tuple(sorted(stages))
        if s3gen_estimator_dtype is not None:
            self.config.s3gen_estimator_dtype = str(s3gen_estimator_dtype)
        if t3_prefill_graphs is not None:
            self.config.t3_prefill_graphs = bool(t3_prefill_graphs)
        if default_language is not None:
            self.config.default_language = str(default_language)
        if max_new_tokens_limit is not None:
            self.config.max_new_tokens_limit = _integer(
                "max_new_tokens_limit", max_new_tokens_limit, low=1, high=self.config.t3.max_speech_tokens,
            )
        if s3gen_max_batch_size is not None:
            if int(s3gen_max_batch_size) < 1:
                raise ValueError("s3gen_max_batch_size must be at least 1")
            self.config.s3gen_max_batch_size = int(s3gen_max_batch_size)
        self._s3gen_estimator_dtype = _parse_dtype(self.config.s3gen_estimator_dtype)
        if self.config.s3gen_compile and self.config.s3gen_graphs:
            if s3gen_graphs:
                raise ValueError("s3gen_graphs and s3gen_compile are alternatives; enable one of them")
            # compile was asked for explicitly, graphs are the default: compile wins
            logger.info("s3gen_compile requested: the S3Gen CUDA graphs are off for this deployment")
            self.config.s3gen_graphs = False
        if self.config.s3gen_graphs:
            if self.config.s3gen_frame_bucket <= 0:
                # graphs are per shape: without a bucket every chunk length
                # would be a capture of its own
                self.config.s3gen_frame_bucket = 64
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self.local_dir = resolve_snapshot(model_path_hf, cache_dir)
        self.tokenizer = self._build_text_tokenizer()
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._shared: dict[str, Any] = {}

    def _build_text_tokenizer(self):
        from mstar.model.chatterbox.components.text import (
            ChatterboxTextTokenizer,
            MultilingualTextTokenizer,
            TurboTextTokenizer,
        )

        if self.config.is_turbo:
            return TurboTextTokenizer(self.local_dir)
        if self.config.is_multilingual:
            cangjie = self.config.cangjie_file
            return MultilingualTextTokenizer(
                Path(self.local_dir) / self.config.text_tokenizer_file,
                Path(self.local_dir) / cangjie if cangjie else None,
                start_token=self.config.t3.start_text_token,
                stop_token=self.config.t3.stop_text_token,
            )
        return ChatterboxTextTokenizer(
            Path(self.local_dir) / self.config.text_tokenizer_file,
            start_token=self.config.t3.start_text_token,
            stop_token=self.config.t3.stop_text_token,
        )

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a request cap the T3 KV pool cannot hold.

        Nothing preempts a stream once the pool is full: the requests wait on
        each other until they time out. So ``max_concurrent_requests`` requests
        at full text and ``max_new_tokens_limit`` speech tokens must fit, next to
        the sink page and the pages the captured decode's padding rows keep
        between replays.
        """
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning(
                "%s sets no max_concurrent_requests: a burst of long requests can fill "
                "the T3 KV cache and stall until the requests time out", config_path,
            )
            return
        kv = next(s for s in self.get_node_resources() if s.resource_key == T3_KV).config
        overrides = (config.get("resources") or {}).get(T3_KV) or {}
        page_size = int(overrides.get("page_size", kv.page_size))
        max_num_pages = int(overrides.get("max_num_pages", kv.max_num_pages))
        need = 1 + int(cap) * self._request_kv_pages(page_size) + self._padding_kv_pages()
        if need > max_num_pages:
            raise ValueError(
                f"{config_path}: {cap} concurrent requests (max_concurrent_requests) at "
                f"max_new_tokens_limit={self.config.max_new_tokens_limit or self.config.t3.max_speech_tokens} "
                f"need {need} pages of resources.{T3_KV}, which has {max_num_pages}; "
                "raise max_num_pages or lower one of the other two"
            )

    def _request_kv_pages(self, page_size: int) -> int:
        """Pages one request can hold at full text and the token limit, per
        stream (CFG keeps a conditional and an unconditional one): the prompt
        (conditioning, text, the speech BOS, twice where T3 repeats it) and a
        token per decode step, the step that runs past the stop included."""
        t3 = self.config.t3
        limit = self.config.max_new_tokens_limit or t3.max_speech_tokens
        bos = 2 if t3.duplicate_bos_in_prefill else 1
        tokens = t3.cond_len + self.config.max_text_tokens + bos + limit
        streams = 1 if self.config.is_turbo else 2
        return streams * -(-tokens // page_size)

    def _padding_kv_pages(self) -> int:
        """Pages the captured decode's padding rows keep between replays: a page
        per stream for every row but the first (a batch that drops a request
        keeps its bucket), per capture (guided and unguided) and per graph slot
        (the engine double-buffers because the KV cache pre-plans)."""
        from mstar.model.chatterbox.submodules import T3Submodule

        rows = T3Submodule.DECODE_CAPTURE_BATCH_SIZES[-1] - 1
        streams = 1 if self.config.is_turbo else 3  # guided rows hold 2 streams, unguided 1
        slots = max(2, int(os.environ.get("MSTAR_NUM_SLOTS", "2")))  # as the engine reads it
        return slots * rows * streams

    def get_node_resources(self) -> list[NodeResourceSpec]:
        t3 = self.config.t3
        bb = t3.backbone
        kv = PagedKVConfig(
            num_layers=bb.num_hidden_layers,
            num_kv_heads=bb.num_key_value_heads,
            head_dim=bb.head_dim,
            max_seq_len=t3.cond_len + self.config.max_text_tokens + t3.max_speech_tokens,
            num_qo_heads=bb.num_attention_heads,
        )
        if bb.is_gpt2:
            # learned absolute positions: the resource only counts
            position = PositionConfig(kv_cache=T3_KV)
        else:
            position = PositionConfig(
                kv_cache=T3_KV,
                rope_theta=bb.rope_theta,
                rope_scale=bb.rope_scaling["factor"],
                low_freq_factor=bb.rope_scaling["low_freq_factor"],
                high_freq_factor=bb.rope_scaling["high_freq_factor"],
                old_context_len=bb.rope_scaling["original_max_position_embeddings"],
            )
        return [
            KVSpec(resource_key=T3_KV, nodes={T3_NODE}, config=kv),
            AttentionSpec(
                resource_key=T3_ATTN, nodes={T3_NODE},
                config=AttentionConfig(kv_cache=T3_KV),
            ),
            PositionSpec(resource_key=T3_POS, nodes={T3_NODE}, config=position),
            SamplerSpec(
                resource_key=T3_SAMPLER, nodes={T3_NODE},
                vocab_size=t3.speech_vocab_size,
                enable_repetion_penalty=True,
                # the reference samples with min_p 0.05 (Turbo does not)
                enable_min_p=not self.config.is_turbo,
            ),
        ]

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        knobs = self.resolve_generation_kwargs(model_kwargs)
        labels = [COND_LABEL, UNCOND_LABEL] if knobs["cfg_weight"] > 0 else [COND_LABEL]
        return {
            T3_SAMPLER: SamplingReqConfig(
                temperature=knobs["temperature"],
                top_k=knobs["top_k"],
                top_p=knobs["top_p"],
                repetition_penalty=knobs["repetition_penalty"],
                min_p=knobs["min_p"],
                ignore_eos=knobs["ignore_eos"],
            ),
            T3_KV: KVReqConfig(needed_labels=labels),
        }

    def resolve_generation_kwargs(self, model_kwargs: dict | None) -> dict[str, Any]:
        """Every public knob, defaulted from the variant's generation config.

        Turbo has no guidance, no exaggeration and no min-p; a request that
        asks for them gets the reference behaviour (ignored) with a warning.

        Every value is checked here, so ``process_prompt`` answers a bad one
        with a 400 before the conductor or a worker sees it.
        """
        mk = dict(model_kwargs or {})
        g = self.config.generation
        do_sample = _flag("do_sample", mk.get("do_sample", True))
        temperature = _number(
            "temperature", mk.get("temperature", g.temperature), low=0.0, high=MAX_TEMPERATURE,
        )
        limit = self.config.max_new_tokens_limit or self.config.t3.max_speech_tokens
        max_new_tokens = _integer(
            "max_new_tokens",
            mk.get("max_new_tokens", mk.get("max_output_tokens", min(g.max_new_tokens, limit))),
            low=1, high=self.config.t3.max_speech_tokens,
        )
        if max_new_tokens > limit:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} is over this deployment's limit of {limit} "
                "(max_new_tokens_limit, sized with max_concurrent_requests to fit the KV cache)"
            )
        knobs = {
            "temperature": temperature if do_sample else 0.0,
            "top_p": _number("top_p", mk.get("top_p", g.top_p), low=0.0, high=1.0),
            # the sampler's top-k row is int32
            "top_k": _integer(
                "top_k", mk.get("top_k", g.top_k), low=0, high=self.config.t3.speech_vocab_size,
            ),
            "min_p": _number("min_p", mk.get("min_p", g.min_p), low=0.0, high=1.0),
            "repetition_penalty": _number(
                "repetition_penalty", mk.get("repetition_penalty", g.repetition_penalty),
                low=0.0, high=MAX_REPETITION_PENALTY, low_open=True,
            ),
            "cfg_weight": _number(
                "cfg_weight", mk.get("cfg_weight", g.cfg_weight), low=0.0, high=MAX_CFG_WEIGHT,
            ),
            "exaggeration": _number(
                "exaggeration", mk.get("exaggeration", g.exaggeration), low=0.0, high=MAX_EXAGGERATION,
            ),
            "max_new_tokens": max_new_tokens,
            "n_cfm_timesteps": _integer(
                "n_cfm_timesteps", mk.get("n_cfm_timesteps", g.n_cfm_timesteps),
                low=1, high=MAX_CFM_TIMESTEPS,
            ),
            "watermark": _flag("watermark", mk.get("watermark", g.watermark)),
            "ignore_eos": _flag("ignore_eos", mk.get("ignore_eos", False)),
        }
        # the conductor seeds the request with it; checked here, not returned
        if mk.get("seed") is not None:
            _integer("seed", mk["seed"], low=0, high=MAX_SEED)
        if self.config.is_turbo and (
            knobs["cfg_weight"] > 0 or knobs["exaggeration"] > 0 or knobs["min_p"] > 0
        ):
            logger.warning(
                "Chatterbox-Turbo ignores cfg_weight, exaggeration and min_p"
            )
            knobs.update(cfg_weight=0.0, exaggeration=0.0, min_p=0.0)
        return knobs

    def get_max_output_tokens(self, **model_kwargs: Any) -> int:
        return self.resolve_generation_kwargs(model_kwargs)["max_new_tokens"]

    # -----------------------------------------------------------------------
    # Graph
    # -----------------------------------------------------------------------

    def _t3_outputs(self, first: bool) -> list[GraphEdge]:
        edges = [
            StreamingGraphEdge(
                next_node=S3GEN_NODE, name=SPEECH_TOKENS,
                target_partition=S3GEN_PARTITION,
            ),
        ]
        if first:
            # the first speech token persists so the decode loop can pick it up
            edges.insert(0, GraphEdge(
                next_node=EMPTY_DESTINATION, name=SPEECH_TOKENS,
                conductor_new_token=True, persist=True,
            ))
        else:
            edges.insert(0, GraphEdge(next_node=T3_NODE, name=PREV_TOKEN))
        return edges

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=T3_NODE, input_names=[TEXT_INPUTS], outputs=self._t3_outputs(first=True),
        )
        prefill_voice = Sequential([
            GraphNode(
                name=VOICE_ENCODER_NODE,
                input_names=[REF_AUDIO, VOICE_KEY],
                outputs=[
                    GraphEdge(next_node=T3_NODE, name=SPEAKER_EMB),
                    GraphEdge(next_node=T3_NODE, name=PROMPT_TOKENS),
                ],
            ),
            GraphNode(
                name=T3_NODE,
                input_names=[TEXT_INPUTS, SPEAKER_EMB, PROMPT_TOKENS],
                outputs=self._t3_outputs(first=True),
            ),
        ])
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name=T3_NODE, input_names=[PREV_TOKEN], outputs=self._t3_outputs(first=False),
            ),
            max_iters=self.config.t3.max_speech_tokens,
            outputs=[],
        )
        s3gen_out = [GraphEdge(next_node=EMIT_TO_CLIENT, name=AUDIO_CHUNK, output_modality="audio")]
        s3gen_chunk = GraphNode(
            name=S3GEN_NODE, input_names=[SPEECH_TOKENS], outputs=list(s3gen_out),
        )
        s3gen_chunk_voice = GraphNode(
            name=S3GEN_NODE, input_names=[SPEECH_TOKENS, REF_AUDIO, VOICE_KEY],
            outputs=list(s3gen_out),
        )
        return {
            "prefill": prefill,
            "prefill_voice": prefill_voice,
            "decode": decode,
            "s3gen_chunk": s3gen_chunk,
            "s3gen_chunk_voice": s3gen_chunk_voice,
        }

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name=T3_PARTITION,
                graph_walks={"prefill", "prefill_voice", "decode"},
                initial_walk="prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name=S3GEN_PARTITION,
                graph_walks={"s3gen_chunk", "s3gen_chunk_voice"},
                initial_walk=None,
                producer_partitions=[T3_PARTITION],
            ),
        ]

    def _chunk_policy(self):
        """Speech tokens reach S3Gen in a small first chunk and fixed later
        chunks; ``stream_chunk_tokens=0`` hands the whole utterance over once
        T3 finishes (the reference's offline decode)."""
        if self.config.stream_chunk_tokens <= 0:
            return FixedChunkPolicy(chunk_size=self.config.t3.max_speech_tokens + 1)
        return RampChunkPolicy(
            first_chunk=self.config.stream_first_chunk_tokens,
            chunk_size=self.config.stream_chunk_tokens,
            growth=self.config.stream_chunk_growth,
            max_chunk=max(self.config.stream_max_chunk_tokens, self.config.stream_chunk_tokens),
        )

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=[T3_PARTITION, S3GEN_PARTITION],
            connections=[
                Connection(
                    from_partition=T3_PARTITION,
                    to_partition=S3GEN_PARTITION,
                    edge_name=SPEECH_TOKENS,
                    chunk_policy_factory=self._chunk_policy,
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # Prompt processing (API data worker)
    # -----------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """Decode a reference clip to 24 kHz mono float32 (the rate S3Gen's
        reference mel needs; the 16 kHz views are derived on the worker)."""
        audio = self._decode_audio(filepath).to(device)
        return TensorAndMetadata(
            data=audio, metadata=dict(sample_rate=S3GEN_SR, num_channels=1)
        )

    @staticmethod
    def _decode_audio(filepath: str) -> torch.Tensor:
        """``[T]`` float32 at 24 kHz, mono.

        libsndfile (bundled with ``soundfile``) covers WAV/FLAC/OGG/MP3 without
        any system library; torchcodec needs FFmpeg's shared libraries, which
        the nodes may not have, so it is only the fallback for other codecs.
        """
        try:
            import soundfile as sf

            data, sr = sf.read(filepath, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data).mean(dim=1)
        except Exception as exc:  # noqa: BLE001 - any decode failure falls through
            logger.debug("soundfile could not decode %s (%s); trying torchcodec", filepath, exc)
            from torchcodec.decoders import AudioDecoder

            # an undecodable upload is the client's error (400); a missing
            # decoder library above still surfaces as a server error
            try:
                decoder = AudioDecoder(filepath, sample_rate=S3GEN_SR, num_channels=1)
                return decoder.get_all_samples().data[0].float()
            except RuntimeError as decode_exc:
                raise ValueError(f"Could not decode the reference audio: {decode_exc}") from decode_exc
        if wav.numel() == 0:
            # before the resampler, which cannot reshape an empty clip
            raise ValueError("Reference audio is empty")
        if sr != S3GEN_SR:
            wav = resample(wav, sr, S3GEN_SR)
        return wav

    def _preset_voice_path(self, voice: str) -> Path:
        if self.voices_dir is None:
            raise ValueError(
                f"Unknown voice {voice!r}: no voices_dir is configured; use "
                f"voice={BUILTIN_VOICE!r} or upload reference audio (ref_audio)"
            )
        # a bare stem ("Abigail") or the file name other servers expect ("Abigail.wav")
        for ext in ("", ".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            path = self.voices_dir / f"{voice}{ext}"
            if path.is_file() and path.parent == self.voices_dir:
                return path
        available = sorted(p.stem for p in self.voices_dir.iterdir() if p.is_file())
        raise ValueError(f"Unknown voice {voice!r}; presets: {available}")

    def _prepare_reference(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.detach().to("cpu", torch.float32).reshape(-1)
        if wav.numel() == 0:
            raise ValueError("Reference audio is empty")
        if not torch.isfinite(wav).all():
            raise ValueError("Reference audio has NaN or infinite samples")
        max_len = int(MAX_REFERENCE_SECONDS * S3GEN_SR)
        if wav.numel() > max_len:
            wav = wav[:max_len]
        # the same trim the voice encoder applies before its STFT
        sound = trim_silence(resample(wav, S3GEN_SR, S3_SR), top_db=20.0).numel() / S3_SR
        if sound < MIN_REFERENCE_SECONDS:
            raise ValueError(
                f"Reference audio has {sound:.2f} s of sound once silence is trimmed; "
                f"at least {MIN_REFERENCE_SECONDS} s is needed"
            )
        if (wav - wav.mean()).abs().max() < MIN_REFERENCE_PEAK:
            raise ValueError("Reference audio is silent (it never swings past -60 dBFS)")
        if self.config.is_turbo:
            if wav.numel() < 5 * S3GEN_SR:
                raise ValueError("Chatterbox-Turbo needs a reference clip longer than 5 s")
            if self.config.normalize_reference_loudness:
                wav = _normalize_loudness(wav, S3GEN_SR, self.config.reference_target_lufs)
        return wav.contiguous()

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        if not prompt or not prompt.strip():
            raise ValueError("Chatterbox requires a non-empty text prompt")
        if any(m not in ("text", "audio") for m in input_modalities):
            raise ValueError("Chatterbox takes text plus an optional reference audio clip")
        if set(output_modalities) != {"audio"}:
            raise ValueError("Chatterbox produces audio output only")
        # a bad knob fails here as a 400; past this point it would hang the request
        self.resolve_generation_kwargs(kwargs)

        text_ids = self._tokenize(prompt, kwargs.get("language_id"))
        if text_ids.numel() > self.config.max_text_tokens:
            raise ValueError(
                f"Text is {text_ids.numel()} tokens; the limit is "
                f"{self.config.max_text_tokens}. Split it into sentences."
            )
        out: NameToTensorList = {TEXT_INPUTS: [text_ids]}

        voice = kwargs.get("voice")
        uploaded = (tensors or {}).get("audio_inputs") or []
        if len(uploaded) > 1:
            raise ValueError("Give one reference clip per request")
        if uploaded:
            wav = uploaded[0]
        elif voice in _BUILTIN_VOICE_ALIASES:
            return out
        else:
            wav = self.load_audio(str(self._preset_voice_path(str(voice))), "cpu").data
        wav = self._prepare_reference(wav)
        out[REF_AUDIO] = [wav]
        out[VOICE_KEY] = [voice_key_for(wav)]
        return out

    def _tokenize(self, prompt: str, language_id: str | None) -> torch.Tensor:
        """Text ids for the variant's front end; ``language_id`` (a request knob)
        selects the multilingual checkpoint's language token and preprocessing
        and falls back to the deployment's ``default_language``."""
        if self.config.is_multilingual:
            lang = language_id if language_id is not None else self.config.default_language
            return self.tokenizer(prompt, language_id=lang)
        if language_id is not None:
            logger.warning("language_id %r is ignored: only Chatterbox Multilingual takes a language", language_id)
        return self.tokenizer(prompt)

    # -----------------------------------------------------------------------
    # Conductor state machine
    # -----------------------------------------------------------------------

    def _step_metadata(self, metadata: CurrentForwardConductorMetadata, keys) -> dict[str, Any]:
        step = {k: metadata.kwargs[k] for k in keys}
        step["is_prefill"] = metadata.is_prefill
        return step

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        knobs = self.resolve_generation_kwargs(model_kwargs)
        has_voice = bool(input_signals.get(REF_AUDIO))
        wants_audio = "audio" in output_modalities

        if partition_name == T3_PARTITION:
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="prefill_voice" if has_voice else "prefill",
                is_prefill=True,
                kwargs={k: knobs[k] for k in _T3_KNOBS},
            )
            names = [TEXT_INPUTS] + ([REF_AUDIO, VOICE_KEY] if has_voice else [])
            targets = {TEXT_INPUTS: T3_NODE, REF_AUDIO: VOICE_ENCODER_NODE, VOICE_KEY: VOICE_ENCODER_NODE}
            inputs = []
            for name in names:
                edge = GraphEdge(next_node=targets[name], name=name)
                edge.tensor_info = input_signals.get(name, [])
                inputs.append(edge)
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                # the text is read once; the reference clip stays persisted for S3Gen
                unpersist_tensors=list(input_signals.get(TEXT_INPUTS, [])),
                request_done=not wants_audio,
                step_metadata=self._step_metadata(metadata, _T3_KNOBS),
            )

        if partition_name == S3GEN_PARTITION:
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="s3gen_chunk_voice" if has_voice else "s3gen_chunk",
                is_prefill=False,
                kwargs={k: knobs[k] for k in _S3GEN_KNOBS},
            )
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=self._s3gen_voice_inputs(input_signals) if has_voice else [],
                unpersist_tensors=[],
                request_done=not wants_audio,
                step_metadata=self._step_metadata(metadata, _S3GEN_KNOBS),
            )
        raise ValueError(f"Unknown Chatterbox partition {partition_name!r}")

    @staticmethod
    def _s3gen_voice_inputs(
        signals: dict[str, list[TensorPointerInfo]], with_tensors: bool = True,
    ) -> list[GraphEdge]:
        """The reference clip and its key for the S3Gen node. A stream consumer
        with non-stream inputs fires once those are in, so the edges accompany
        every chunk; the tensors themselves ride only with the first forward
        (``with_tensors``), later chunks carry signal-only edges and the node
        reuses the reference it conditioned on. Signal-only edges never touch
        the transport, so the extra handshake the conductor issues after the
        final chunk cannot race the request's teardown."""
        inputs = []
        for name in (REF_AUDIO, VOICE_KEY):
            edge = GraphEdge(next_node=S3GEN_NODE, name=name)
            edge.tensor_info = list(signals.get(name, [])) if with_tensors else []
            inputs.append(edge)
        return inputs

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        del incoming_connections
        if partition_name == T3_PARTITION:
            if partition_metadata.is_prefill:
                partition_metadata.is_prefill = False
                partition_metadata.graph_walk = "decode"
                edge = GraphEdge(next_node=T3_NODE, name=PREV_TOKEN)
                edge.tensor_info = persist_signals.get(SPEECH_TOKENS, [])
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[edge],
                    unpersist_tensors=list(edge.tensor_info),
                    step_metadata=self._step_metadata(partition_metadata, _T3_KNOBS),
                )
            if partition_metadata.graph_walk == "decode":
                return ForwardPassArgs(
                    full_metadata=partition_metadata, inputs=[],
                    unpersist_tensors=[], request_done=True,
                )
            raise ValueError(f"T3 in unexpected walk {partition_metadata.graph_walk!r}")

        if partition_name == S3GEN_PARTITION:
            inputs = []
            if partition_metadata.graph_walk == "s3gen_chunk_voice":
                inputs = self._s3gen_voice_inputs(persist_signals, with_tensors=False)
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=inputs,
                unpersist_tensors=[],
                step_metadata=self._step_metadata(partition_metadata, _S3GEN_KNOBS),
            )
        raise ValueError(f"Unknown Chatterbox partition {partition_name!r}")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def get_autocast_dtype(self):
        return self._t3_dtype

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.sample_rate

    def postprocess(
        self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None,
    ) -> bytes:
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported Chatterbox output modality {modality!r}")
        if output.numel() == 0:
            return b""
        pcm = output.detach().cpu()
        if pcm.is_floating_point():
            pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
        elif pcm.dtype != torch.int16:
            pcm = pcm.to(torch.int16)
        return pcm.contiguous().numpy().tobytes()

    # -----------------------------------------------------------------------
    # Submodules (worker side)
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={T3_NODE}, shard_dim={})

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        del sp_group
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        if node_name == VOICE_ENCODER_NODE:
            submodule = self._create_voice_encoder_submodule(device)
        elif node_name == T3_NODE:
            submodule = self._create_t3_submodule(device, tp_group, autocast_dtype)
        elif node_name == S3GEN_NODE:
            submodule = self._create_s3gen_submodule(device)
        else:
            raise ValueError(f"Unknown Chatterbox node {node_name!r}")
        self._submodule_cache[node_name] = submodule
        logger.info("Loaded Chatterbox submodule %s on %s", node_name, device)
        return submodule

    def _weights_path(self, name: str) -> Path:
        path = Path(self.local_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing from the checkpoint snapshot")
        return path

    def _builtin_voice(self) -> dict:
        """``conds.pt``: the voice the checkpoint ships (T3 and S3Gen halves)."""
        if "builtin_voice" not in self._shared:
            self._shared["builtin_voice"] = torch.load(
                self._weights_path(self.config.builtin_voice_file),
                map_location="cpu", weights_only=True,
            )
        return self._shared["builtin_voice"]

    def _s3_tokenizer(self, device: str):
        """One S3 tokenizer per device, shared by the voice encoder and S3Gen
        nodes when they are colocated."""
        key = f"s3_tokenizer:{device}"
        if key not in self._shared:
            from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
            from mstar.model.chatterbox.loader import iter_weights

            # built for real, not on meta: its mel filters, window and RoPE
            # tables are computed in __init__ and have no checkpoint entry
            tokenizer = S3Tokenizer(self.config.s3_tokenizer).to(device)
            tokenizer.load_weights(iter_weights(
                self._weights_path(self.config.s3gen_weights), device=device, prefix="tokenizer.",
            ))
            self._shared[key] = tokenizer.eval()
        return self._shared[key]

    def _create_voice_encoder_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.chatterbox.components.voice_encoder import VoiceEncoder
        from mstar.model.chatterbox.loader import iter_weights
        from mstar.model.chatterbox.submodules import VoiceEncoderSubmodule

        # computed mel filters -> built for real (see _s3_tokenizer)
        encoder = VoiceEncoder(self.config.voice_encoder).to(device)
        encoder.load_weights(iter_weights(
            self._weights_path(self.config.voice_encoder_weights), device=device,
        ))
        return VoiceEncoderSubmodule(
            encoder.eval(), self._s3_tokenizer(device), self.config,
        )

    def _create_t3_submodule(
        self, device: str, tp_group=None, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.chatterbox.components.t3 import T3Model
        from mstar.model.chatterbox.loader import iter_weights, materialize
        from mstar.model.chatterbox.submodules import BuiltinT3Voice, T3Submodule

        with torch.device("meta"):
            model = T3Model(self.config.t3, comm_group=tp_group)
        materialize(model, device, autocast_dtype)
        model.load_weights(iter_weights(self._weights_path(self.config.t3_weights), device=device))
        voice = self._builtin_voice()["t3"]
        builtin = BuiltinT3Voice(
            speaker_emb=voice["speaker_emb"].reshape(-1).to(device),
            prompt_tokens=voice["cond_prompt_speech_tokens"].reshape(-1).to(device),
        )
        return T3Submodule(model.eval(), self.config, builtin_voice=builtin)

    def _create_s3gen_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.chatterbox.components.s3gen import ReferenceConditioning, S3Gen
        from mstar.model.chatterbox.loader import iter_weights
        from mstar.model.chatterbox.submodules import S3GenSubmodule

        # mel basis, STFT windows, positional table and fades are computed in
        # __init__ -> built for real (see _s3_tokenizer)
        s3gen = S3Gen(self.config.s3gen).to(device)
        s3gen.load_weights(
            iter_weights(self._weights_path(self.config.s3gen_weights), device=device),
        )
        s3gen.eval()
        if self._s3gen_estimator_dtype != torch.float32:
            # after load_weights: the checkpoint is float32
            s3gen.decoder.set_estimator_dtype(self._s3gen_estimator_dtype)
        if self.config.s3gen_compile:
            # after load_weights: the compiled wrapper renames parameters
            s3gen.decoder.estimator = torch.compile(
                s3gen.decoder.estimator, dynamic=True, mode=self.config.s3gen_compile_mode,
            )
        gen = self._builtin_voice()["gen"]
        builtin = ReferenceConditioning(
            prompt_tokens=gen["prompt_token"].to(device),
            prompt_feat=gen["prompt_feat"].to(device),
            embedding=gen["embedding"].to(device),
        )
        if self.config.s3gen_graphs:
            solver = s3gen.enable_graphs(
                rows=_graph_rows(self.config.s3gen_max_batch_size), stages=self._graph_stages,
                token_bucket=max(1, self.config.s3gen_frame_bucket // self.config.s3gen.token_mel_ratio),
            )
            if solver is not None and torch.device(device).type == "cuda":
                self._warm_solve_graphs(s3gen, solver, builtin.prompt_feat.shape[1])
        return S3GenSubmodule(
            s3gen.eval(), self._s3_tokenizer(device), self.config,
            builtin_voice=builtin, watermarker=self._watermarker(device),
        )

    def _graph_warmup_frames(self, prompt_frames: int) -> list[int]:
        """Solve lengths (bucketed) the streaming chunks of the built-in voice
        produce: prompt + 2 x (left context + chunk + look-ahead) for every
        chunk size of the ramp. Other voices and whole-utterance solves are
        captured on first use."""
        cfg = self.config
        if cfg.stream_chunk_tokens <= 0:
            return []  # whole-utterance solves: one shape per utterance length
        chunks = {cfg.stream_first_chunk_tokens}
        size = cfg.stream_chunk_tokens
        cap = max(cfg.stream_max_chunk_tokens, size)
        while size and size <= cap:
            chunks.add(size)
            nxt = int(size * cfg.stream_chunk_growth) if cfg.stream_chunk_growth > 1 else cap + 1
            if nxt <= size:
                break
            size = nxt
        chunks.add(cap)
        ratio = cfg.s3gen.token_mel_ratio
        lookahead = cfg.s3gen.encoder.pre_lookahead_len
        bucket = cfg.s3gen_frame_bucket
        frames = set()
        for chunk in chunks:
            if chunk <= 0:
                continue
            total = prompt_frames + ratio * (cfg.stream_context_tokens + chunk + lookahead)
            frames.add(-(-total // bucket) * bucket)
        return sorted(frames)

    def _warm_solve_graphs(self, s3gen, solver, prompt_frames: int) -> None:
        frames = self._graph_warmup_frames(prompt_frames)
        if not frames:
            return
        longest = max(frames)
        n_mel = self.config.s3gen.output_size
        device, dtype = s3gen.device, s3gen.dtype
        gen = torch.Generator(device=device).manual_seed(0)
        example = {
            "mu": torch.randn(1, n_mel, longest, device=device, dtype=dtype, generator=gen),
            "mask": torch.ones(1, 1, longest, device=device, dtype=dtype),
            "spks": torch.randn(1, n_mel, device=device, dtype=dtype, generator=gen),
            "cond": torch.zeros(1, n_mel, longest, device=device, dtype=dtype),
            "noise": torch.randn(1, n_mel, longest, device=device, dtype=dtype, generator=gen),
        }
        captured = solver.warmup(frames, self.config.generation.n_cfm_timesteps, example)
        logger.info(
            "S3Gen: captured %d flow-solve graphs for %s frames x rows %s", captured, frames, list(solver.rows),
        )

    def _watermarker(self, device: str):
        from mstar.model.chatterbox.components.watermark import PerthWatermarker

        return PerthWatermarker.build(device) if self.config.generation.watermark else None


def _flag(name: str, value: Any) -> bool:
    """A JSON boolean; a string such as "false" is refused, not read as true."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


def _number(
    name: str, value: Any, *, low: float | None = None, high: float | None = None,
    low_open: bool = False,
) -> float:
    """A finite number in [low, high], or (low, high] when ``low_open``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    too_low = low is not None and (value <= low if low_open else value < low)
    if too_low or (high is not None and value > high):
        lo = "-inf" if low is None else low
        hi = "inf" if high is None else high
        raise ValueError(f"{name}={value} is outside {'(' if low_open else '['}{lo}, {hi}]")
    return value


def _integer(name: str, value: Any, *, low: int, high: int | None = None) -> int:
    """An integer in [low, high]; a float is taken only when it is whole (2.0, not 2.5)."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name}={value} is outside [{low}, {'inf' if high is None else high}]")
    return value


def _graph_rows(max_rows: int) -> tuple[int, ...]:
    """Captured row counts for a batch limit: the powers of two up to it, plus
    the limit itself (1, 2, 4, 8 for 8; 1, 2, 4, 8, 12 for 12)."""
    rows = [1]
    while rows[-1] * 2 <= max_rows:
        rows.append(rows[-1] * 2)
    if rows[-1] != max_rows:
        rows.append(int(max_rows))
    return tuple(rows)


def _parse_dtype(name: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    try:
        return dtypes[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown t3_dtype {name!r}; use bfloat16, float16 or float32") from None


def voice_key_for(wav: torch.Tensor) -> torch.Tensor:
    """A stable 63-bit content key for a reference waveform, so the workers'
    voice caches recognise a clip they already conditioned on."""
    digest = hashlib.blake2b(
        wav.detach().to("cpu", torch.float32).contiguous().numpy().tobytes(), digest_size=8,
    ).digest()
    return torch.tensor([int.from_bytes(digest, "little") >> 1], dtype=torch.long)


def _normalize_loudness(wav: torch.Tensor, sample_rate: int, target_lufs: float) -> torch.Tensor:
    """ITU-R BS.1770 integrated-loudness gain to ``target_lufs`` (reference
    ``ChatterboxTurboTTS.norm_loudness``); skipped, with a warning, when the
    clip is too quiet or too short to measure."""
    try:
        import pyloudnorm
    except ImportError:
        logger.warning("pyloudnorm is not installed; reference loudness is not normalised")
        return wav
    import math

    meter = pyloudnorm.Meter(sample_rate)
    loudness = meter.integrated_loudness(wav.numpy())
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    if math.isfinite(gain) and gain > 0.0:
        return wav * gain
    return wav
