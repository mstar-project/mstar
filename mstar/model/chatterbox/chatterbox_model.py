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
    KVConfig,
    KVReqConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.chatterbox.components.audio_frontend import resample
from mstar.model.chatterbox.config import (
    COND_LABEL,
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
        t3_dtype: str | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        # T3 runs in bf16 by default; ``t3_dtype: float32`` (a parity switch)
        # keeps the reference package's numerics at about half the decode speed.
        self._t3_dtype = _parse_dtype(t3_dtype) if t3_dtype else torch.bfloat16
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
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self.local_dir = resolve_snapshot(model_path_hf, cache_dir)
        self.tokenizer = self._build_text_tokenizer()
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._shared: dict[str, Any] = {}

    def _build_text_tokenizer(self):
        from mstar.model.chatterbox.components.text import (
            ChatterboxTextTokenizer,
            TurboTextTokenizer,
        )

        if self.config.is_turbo:
            return TurboTextTokenizer(self.local_dir)
        return ChatterboxTextTokenizer(
            Path(self.local_dir) / self.config.text_tokenizer_file,
            start_token=self.config.t3.start_text_token,
            stop_token=self.config.t3.stop_text_token,
        )

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        t3 = self.config.t3
        bb = t3.backbone
        kv = KVConfig(
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
                ignore_eos=knobs["ignore_eos"],
            ),
            T3_KV: KVReqConfig(needed_labels=labels),
        }

    def resolve_generation_kwargs(self, model_kwargs: dict | None) -> dict[str, Any]:
        """Every public knob, defaulted from the variant's generation config.

        Turbo has no guidance, no exaggeration and no min-p; a request that
        asks for them gets the reference behaviour (ignored) with a warning.
        """
        mk = dict(model_kwargs or {})
        g = self.config.generation
        do_sample = bool(mk.get("do_sample", True))
        knobs = {
            "temperature": float(mk.get("temperature", g.temperature)) if do_sample else 0.0,
            "top_p": float(mk.get("top_p", g.top_p)),
            "top_k": int(mk.get("top_k", g.top_k)),
            "min_p": float(mk.get("min_p", g.min_p)),
            "repetition_penalty": float(mk.get("repetition_penalty", g.repetition_penalty)),
            "cfg_weight": float(mk.get("cfg_weight", g.cfg_weight)),
            "exaggeration": float(mk.get("exaggeration", g.exaggeration)),
            "max_new_tokens": int(
                mk.get("max_new_tokens", mk.get("max_output_tokens", g.max_new_tokens))
            ),
            "n_cfm_timesteps": int(mk.get("n_cfm_timesteps", g.n_cfm_timesteps)),
            "watermark": bool(mk.get("watermark", g.watermark)),
            "ignore_eos": bool(mk.get("ignore_eos", False)),
        }
        if self.config.is_turbo and (
            knobs["cfg_weight"] > 0 or knobs["exaggeration"] > 0 or knobs["min_p"] > 0
        ):
            logger.warning(
                "Chatterbox-Turbo ignores cfg_weight, exaggeration and min_p"
            )
            knobs.update(cfg_weight=0.0, exaggeration=0.0, min_p=0.0)
        if knobs["max_new_tokens"] > self.config.t3.max_speech_tokens:
            raise ValueError(
                f"max_new_tokens {knobs['max_new_tokens']} exceeds the T3 limit "
                f"{self.config.t3.max_speech_tokens}"
            )
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

            decoder = AudioDecoder(filepath, sample_rate=S3GEN_SR, num_channels=1)
            return decoder.get_all_samples().data[0].float()
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
        max_len = int(MAX_REFERENCE_SECONDS * S3GEN_SR)
        if wav.numel() > max_len:
            wav = wav[:max_len]
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

        text_ids = self.tokenizer(prompt)
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
    def _s3gen_voice_inputs(signals: dict[str, list[TensorPointerInfo]]) -> list[GraphEdge]:
        """The reference clip and its key, handed to the S3Gen node with every
        chunk (a stream consumer only fires once its other inputs are in)."""
        inputs = []
        for name in (REF_AUDIO, VOICE_KEY):
            edge = GraphEdge(next_node=S3GEN_NODE, name=name)
            edge.tensor_info = list(signals.get(name, []))
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
                inputs = self._s3gen_voice_inputs(persist_signals)
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
            from mstar.model.chatterbox.loader import iter_weights, materialize

            with torch.device("meta"):
                tokenizer = S3Tokenizer(self.config.s3_tokenizer)
            materialize(tokenizer, device, torch.float32)
            tokenizer.load_weights(iter_weights(
                self._weights_path(self.config.s3gen_weights), device=device, prefix="tokenizer.",
            ))
            self._shared[key] = tokenizer.eval()
        return self._shared[key]

    def _create_voice_encoder_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.chatterbox.components.voice_encoder import VoiceEncoder
        from mstar.model.chatterbox.loader import iter_weights, materialize
        from mstar.model.chatterbox.submodules import VoiceEncoderSubmodule

        with torch.device("meta"):
            encoder = VoiceEncoder(self.config.voice_encoder)
        materialize(encoder, device, torch.float32)
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
        from mstar.model.chatterbox.loader import iter_weights, materialize
        from mstar.model.chatterbox.submodules import S3GenSubmodule

        with torch.device("meta"):
            s3gen = S3Gen(self.config.s3gen)
        materialize(s3gen, device, torch.float32)
        s3gen.load_weights(
            iter_weights(self._weights_path(self.config.s3gen_weights), device=device),
        )
        gen = self._builtin_voice()["gen"]
        builtin = ReferenceConditioning(
            prompt_tokens=gen["prompt_token"].to(device),
            prompt_feat=gen["prompt_feat"].to(device),
            embedding=gen["embedding"].to(device),
        )
        return S3GenSubmodule(
            s3gen.eval(), self._s3_tokenizer(device), self.config,
            builtin_voice=builtin, watermarker=self._watermarker(device),
        )

    def _watermarker(self, device: str):
        from mstar.model.chatterbox.components.watermark import PerthWatermarker

        return PerthWatermarker.build(device) if self.config.generation.watermark else None


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
