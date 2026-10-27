"""Fish Audio S2-Pro model contract and two-partition streaming topology.

S2-Pro (``fishaudio/s2-pro``) is a dual-AR TTS model: a Qwen3-4B-shaped Slow
AR predicts one semantic code per 21.5 Hz frame, a 4-layer Fast AR fills the
frame's 9 residual codebooks, and a causal DAC codec turns the 10-codebook
frames into 44.1 kHz audio. Voice cloning conditions on a reference clip's
DAC codes placed in the system turn of a Qwen3 chat prompt.

The weights are under the Fish Audio Research License (non-commercial use
only); M* downloads them at runtime and does not redistribute them.

Architecture (two asynchronous partitions):
    AR         - prompt prefill, then one 10-codebook frame per step
    RefEncoder - (voice clone) reference clip -> DAC codes, feeds the prefill
    Codec      - DAC decoder producing PCM chunks

Streaming topology:
    AR --[codec_tokens, ScheduledLeftContextChunkPolicy((4, 8, 16), 32, 1026)]--> Codec

Each codec window carries the decoder's whole receptive field as context, so
the streamed audio is sample for sample the audio of one decode of the
stream (other S2-Pro serving implementations stream with shorter overlaps).

Request state machine:
    AR:    ar_prefill | ar_prefill_clone -> ar_decode loop -> done on <|im_end|> / token limit
    Codec: waits for streamed frames -> codec_chunk -> emits audio -> waits
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
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
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, TensorAndMetadata
from mstar.model.fish_s2_pro.config import (
    ALLOWED_TEXT_SPECIAL_TOKENS,
    AR_ATTN,
    AR_KV,
    AR_POS,
    AR_SAMPLER,
    CLONE_SPEECH_TEXT,
    CLONE_SYSTEM_TEXT,
    IM_END,
    IM_START,
    PROMPT_LAYOUT_KEY,
    REF_CLIP_KEY,
    VOICE_TOKEN,
    FishS2ProConfig,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
CODEC_FILE = "codec.pth"
# Deployment ``model_kwargs`` that override codec streaming controls.
CODEC_KWARGS = ("chunk_schedule", "chunk_frames", "left_context_frames", "conv_context_frames")


def _resolve_snapshot(repo_id: str, cache_dir: str | None, allow_patterns: list[str] | None) -> str:
    local = Path(repo_id)
    if local.is_dir():
        return str(local)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, allow_patterns=allow_patterns)


def clip_digest(clip: torch.Tensor) -> str:
    """Content key of a reference clip, under which its codes are memoised."""
    samples = clip.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.blake2b(samples.numpy().tobytes(), digest_size=16).hexdigest()


logger = logging.getLogger(__name__)

LICENSE_NOTICE = (
    "Fish S2-Pro: the fishaudio/s2-pro weights are under the Fish Audio Research License "
    "(research and non-commercial use only); read it before deploying"
)


class FishS2ProModel(Model):
    """Fish Audio S2-Pro model contract (plain TTS and voice cloning)."""

    # API tensors the AR prefill consumes; the clone prefill adds ``ref_codes``.
    PREFILL_INPUTS = ("prompt_ids",)
    REF_ENCODER_INPUTS = ("audio_inputs",)
    LENGTH_KWARGS = ("max_output_tokens", "max_new_tokens")

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs: Any) -> None:
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Server-init knob: decode the codec with TF32 off (the reference's fp32).
        self.codec_exact_fp32 = bool(kwargs.get("codec_exact_fp32", True))
        self.local_dir = _resolve_snapshot(model_path_hf, cache_dir, list(METADATA_FILES))
        logger.info(LICENSE_NOTICE)
        self.config = FishS2ProConfig.from_pretrained(
            self.local_dir,
            codec_overrides={k: kwargs[k] for k in CODEC_KWARGS if k in kwargs},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        self._special_ids = {int(token_id) for token_id in self.tokenizer.all_special_ids} | {
            int(token_id) for token_id in self.tokenizer.added_tokens_decoder
        }
        self._special_ids -= set(self.tokenizer.convert_tokens_to_ids(list(ALLOWED_TEXT_SPECIAL_TOKENS)))
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def _ensure_full_snapshot(self) -> str:
        root = Path(self.local_dir)
        if (root / "model.safetensors.index.json").is_file() and (root / CODEC_FILE).is_file():
            return self.local_dir
        if Path(self.model_path_hf).is_dir():
            raise FileNotFoundError(f"{self.model_path_hf} lacks model.safetensors.index.json or {CODEC_FILE}")
        self.local_dir = _resolve_snapshot(self.model_path_hf, self.cache_dir, None)
        return self.local_dir

    # -----------------------------------------------------------------------
    # Model ABC: resources
    # -----------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Slow AR paged KV, attention, interleaved bf16-table RoPE, semantic sampler.

        The Fast AR's frame-local KV scratch is owned by the AR submodule.
        """
        slow = self.config.slow_ar
        kv = PagedKVConfig(
            num_layers=slow.n_layer,
            num_kv_heads=slow.n_local_heads,
            head_dim=slow.head_dim,
            max_seq_len=slow.max_seq_len,
            num_qo_heads=slow.n_head,
        )
        return [
            KVSpec(resource_key=AR_KV, nodes={"AR"}, config=kv),
            AttentionSpec(resource_key=AR_ATTN, nodes={"AR"}, config=AttentionConfig(kv_cache=AR_KV)),
            PositionSpec(
                resource_key=AR_POS,
                nodes={"AR"},
                config=PositionConfig(
                    kv_cache=AR_KV,
                    interleave=True,
                    rope_theta=slow.rope_base,
                    rotary_dim=slow.head_dim,
                    rope_table_dtype=torch.bfloat16,
                ),
            ),
            SamplerSpec(
                resource_key=AR_SAMPLER,
                nodes={"AR"},
                vocab_size=self.config.semantic_head_size,
                enable_repetion_penalty=True,
            ),
        ]

    # -----------------------------------------------------------------------
    # Model ABC: walk graph declaration
    # -----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """Prefill (plain or clone), the per-frame decode loop, and codec windows."""

        def ar_node(input_names: list[str], *, prefill: bool) -> GraphNode:
            return GraphNode(
                name="AR",
                input_names=input_names,
                outputs=[
                    GraphEdge(
                        next_node=EMPTY_DESTINATION if prefill else "AR",
                        name="ar_input_embeds",
                        persist=prefill,
                    ),
                    StreamingGraphEdge(next_node="Codec", name="codec_tokens", target_partition="Codec"),
                ],
            )

        return {
            "ar_prefill": ar_node(list(self.PREFILL_INPUTS), prefill=True),
            "ar_prefill_clone": Sequential(
                [
                    GraphNode(
                        name="RefEncoder",
                        input_names=list(self.REF_ENCODER_INPUTS),
                        outputs=[GraphEdge(next_node="AR", name="ref_codes")],
                    ),
                    ar_node([*self.PREFILL_INPUTS, "ref_codes"], prefill=True),
                ]
            ),
            "ar_decode": Loop(
                name="ar_decode_loop",
                section=ar_node(["ar_input_embeds"], prefill=False),
                # the largest per-request max_new_tokens; check_stop ends each request's own
                max_iters=self.config.codec.max_frames,
                outputs=[],
            ),
            "codec_chunk": GraphNode(
                name="Codec",
                input_names=["codec_tokens"],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
            ),
        }

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="AR",
                graph_walks={"ar_prefill", "ar_prefill_clone", "ar_decode"},
                initial_walk="ar_prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="Codec",
                graph_walks={"codec_chunk"},
                initial_walk=None,
                producer_partitions=["AR"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        """Codec windows: a ramp of small chunks, then ``chunk_frames``, each behind
        ``left_context_frames`` of already-decoded frames."""
        codec = self.config.codec
        return PartitionTopology(
            partitions=["AR", "Codec"],
            connections=[
                Connection(
                    from_partition="AR",
                    to_partition="Codec",
                    edge_name="codec_tokens",
                    chunk_policy_factory=lambda: ScheduledLeftContextChunkPolicy(
                        schedule=codec.chunk_schedule,
                        chunk=codec.chunk_frames,
                        left_context=codec.left_context_frames,
                    ),
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # API preprocessing
    # -----------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """Decode a reference clip to 44.1 kHz mono float32 (the codec rate)."""
        import soundfile
        import torchaudio.functional as audio_functional

        from mstar.api_server.media_io import check_resample_rate, read_soundfile_capped
        from mstar.model.media_limits import read_sound_file, sound_file_seconds

        # Bound the clip before decoding it: a small compressed file can hold hours of audio.
        info = soundfile.info(filepath)
        target = self.config.codec.sample_rate
        check_resample_rate(info.samplerate, target)
        seconds = sound_file_seconds(info.frames, info.samplerate)
        if seconds is not None:
            self._check_reference_duration(seconds)
            # the header can understate the length, so the decode stops at the bound too
            waveform, sample_rate = read_soundfile_capped(filepath, self.config.reference_max_seconds)
            audio = torch.from_numpy(waveform)
        else:
            # no sample count in the header (a FLAC written to a pipe): count it as it decodes
            waveform, sample_rate = read_sound_file(filepath, self.config.reference_max_seconds)
            self._check_reference_duration(len(waveform) / sample_rate)
            audio = torch.from_numpy(waveform).mean(dim=1)
        if sample_rate != target:
            audio = audio_functional.resample(audio, sample_rate, target)
        return TensorAndMetadata(data=audio.to(device), metadata={"sample_rate": target, "num_channels": 1})

    def _check_reference_duration(self, seconds: float) -> None:
        low, high = self.config.reference_min_seconds, self.config.reference_max_seconds
        if not low <= seconds <= high:
            raise ValueError(
                f"Fish S2-Pro reference clip is {seconds:.2f} s; it must be between {low:g} and {high:g} s"
            )

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _encode_user_text(self, text: str, field: str, speaker_tag: str) -> list[int]:
        """Tokenize ``speaker_tag + text``; refuse chat control tokens the text could smuggle in.

        The tag is plain text to this tokenizer, so it is encoded with the text
        it precedes: BPE merges across the seam (``>[`` before an inline tag).
        """
        ids = self._encode(speaker_tag + text)
        smuggled = sorted({self.tokenizer.convert_ids_to_tokens(i) for i in ids if i in self._special_ids})
        if smuggled:
            raise ValueError(f"Fish S2-Pro {field} contains unsupported control tokens: {', '.join(smuggled)}")
        return ids

    def build_prompt(self, text: str, ref_text: str | None, ref_frames: int, speaker: int = 0) -> tuple[list[int], int]:
        """Prompt token ids and ``prefix_len``, where a clone's reference frames go.

        The chat layout and the segment-by-segment tokenization are those of
        Fish Audio's published S2-Pro serving recipe (fixed text in
        ``config.py``, pinned by test_prompt_layout_matches_the_reference_segments):
        a clone puts the
        reference transcript and frames in the system turn, and the target
        text is the user turn, both after a ``<|speaker:N|>`` tag.
        """
        speaker_tag = f"<|speaker:{speaker}|>"
        prefix: list[int] = []
        if ref_frames:
            prefix += self._encode(f"{IM_START}system\n") + self._encode(CLONE_SYSTEM_TEXT)
            if ref_text:
                prefix += self._encode_user_text(ref_text, "ref_text", speaker_tag)
            prefix += self._encode(CLONE_SPEECH_TEXT)
        suffix = self._encode(f"{IM_END}\n") if ref_frames else []
        suffix += (
            self._encode(f"{IM_START}user\n")
            + self._encode_user_text(text, "input", speaker_tag)
            + self._encode(f"{IM_END}\n")
            + self._encode(f"{IM_START}assistant\n{VOICE_TOKEN}")
        )
        return prefix + suffix, len(prefix)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> ProcessPromptOutput:
        """Tokenize a request into ``prompt_ids``, with ``[prefix_len, ref_frames]`` as metadata.

        A voice clone carries one 44.1 kHz reference clip (``audio_inputs``)
        and optionally its transcript (``ref_text``); its frames are encoded
        on the worker and spliced in after ``prefix_len`` prompt tokens. The
        layout and the clip's digest travel as host values to the prefill's
        step metadata, so the worker never reads them back from the device.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Fish S2-Pro requires a non-empty text prompt")
        if "text" not in input_modalities or not set(input_modalities) <= {"text", "audio"}:
            raise ValueError("Fish S2-Pro takes text input, plus one reference clip for voice cloning")
        if set(output_modalities) != {"audio"}:
            raise ValueError("Fish S2-Pro supports audio output only")
        clips = (tensors or {}).get("audio_inputs", [])
        if "audio" in input_modalities and len(clips) != 1:
            raise ValueError(f"Fish S2-Pro clones a voice from exactly one reference clip (got {len(clips)})")
        ref_frames = 0
        if clips:
            num_samples = int(clips[0].reshape(-1).shape[0])
            self._check_reference_duration(num_samples / self.config.codec.sample_rate)
            ref_frames = self.config.codec.frames_for_samples(num_samples)
        ref_text = str(kwargs.get("ref_text") or "").strip() if clips else None
        ids, prefix_len = self.build_prompt(prompt.strip(), ref_text, ref_frames)
        metadata: dict[str, Any] = {PROMPT_LAYOUT_KEY: [prefix_len, ref_frames]}
        if clips:
            metadata[REF_CLIP_KEY] = clip_digest(clips[0])
        # a clone's walk splices the reference frames into the prompt's rows
        clone = "audio" in input_modalities
        rows = len(ids) + ref_frames
        walk = "ar_prefill_clone" if clone else "ar_prefill"
        prefill = Span("rows" if clone else "ids", walk, rows, rows, "prompt_ids")
        metadata["kv_layout"] = {AR_KV: {"main": LabelLayout(spans=(prefill,), growth=(("ar_decode_loop", 1),))}}
        return ProcessPromptOutput({"prompt_ids": [torch.tensor(ids, dtype=torch.long)]}, metadata)

    # -----------------------------------------------------------------------
    # Conductor partition state machine
    # -----------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        model_kwargs = model_kwargs or {}
        if partition_name == "AR":
            clone = "audio" in input_modalities
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="ar_prefill_clone" if clone else "ar_prefill",
                is_prefill=True,
                kwargs={"ar_max_tokens": self.get_max_output_tokens(**model_kwargs)},
            )
            routes = [(name, "AR") for name in self.PREFILL_INPUTS]
            step_metadata = {**self._ar_step_metadata(metadata), PROMPT_LAYOUT_KEY: model_kwargs[PROMPT_LAYOUT_KEY]}
            if clone:
                routes += [(name, "RefEncoder") for name in self.REF_ENCODER_INPUTS]
                step_metadata[REF_CLIP_KEY] = model_kwargs[REF_CLIP_KEY]
            inputs = []
            for name, node in routes:
                edge = GraphEdge(next_node=node, name=name)
                edge.tensor_info = input_signals.get(name, [])
                inputs.append(edge)
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                unpersist_tensors=sum([edge.tensor_info for edge in inputs], start=[]),
                request_done="audio" not in output_modalities,
                step_metadata=step_metadata,
            )
        if partition_name == "Codec":
            return ForwardPassArgs(
                full_metadata=CurrentForwardConductorMetadata(
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    graph_walk="codec_chunk",
                    is_prefill=False,
                ),
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
            )
        raise ValueError(f"Unknown Fish S2-Pro partition: {partition_name!r}")

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        del incoming_connections
        if partition_name == "AR":
            if partition_metadata.graph_walk in ("ar_prefill", "ar_prefill_clone"):
                if partition_metadata.kwargs["ar_max_tokens"] <= 1:
                    # the prefill sampled the only frame allowed; a stop from prefill is dropped
                    return ForwardPassArgs(
                        full_metadata=partition_metadata,
                        inputs=[],
                        unpersist_tensors=list(persist_signals.get("ar_input_embeds", [])),
                        request_done=True,
                    )
                partition_metadata.graph_walk = "ar_decode"
                partition_metadata.is_prefill = False
                edge = GraphEdge(next_node="AR", name="ar_input_embeds")
                edge.tensor_info = persist_signals.get("ar_input_embeds", [])
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[edge],
                    unpersist_tensors=list(edge.tensor_info),
                    step_metadata=self._ar_step_metadata(partition_metadata),
                )
            if partition_metadata.graph_walk == "ar_decode":
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    request_done=True,
                )
            raise ValueError(f"AR entered an unexpected graph walk: {partition_metadata.graph_walk!r}")
        if partition_name == "Codec":
            partition_metadata.graph_walk = "codec_chunk"
            return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[])
        raise ValueError(f"Unknown Fish S2-Pro partition: {partition_name!r}")

    @staticmethod
    def _ar_step_metadata(metadata: CurrentForwardConductorMetadata) -> dict[str, Any]:
        return {"is_prefill": metadata.is_prefill}

    # -----------------------------------------------------------------------
    # Sampling and output encoding
    # -----------------------------------------------------------------------

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        """Semantic-code sampling; ``do_sample=False`` is greedy. Residual codebooks are argmax."""
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        generation = self.config.generation
        temperature = model_kwargs.get("temperature", generation.temperature)
        if not model_kwargs.get("do_sample", True):
            temperature = 0.0
        return {
            AR_SAMPLER: SamplingReqConfig(
                temperature=temperature,
                top_k=model_kwargs.get("top_k", generation.top_k),
                top_p=model_kwargs.get("top_p", generation.top_p),
                min_p=model_kwargs.get("min_p", 0.0),
                repetition_penalty=model_kwargs.get("repetition_penalty", generation.repetition_penalty),
                penalize_prompt=False,
                ignore_eos=model_kwargs.get("ignore_eos", False),
            ),
        }

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({
            "ref_text", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "ignore_eos", "do_sample",
            "max_output_tokens",
        })

    def get_max_output_tokens(self, **model_kwargs: Any) -> int:
        requested = model_kwargs.get(
            "max_output_tokens",
            model_kwargs.get("max_new_tokens", self.config.generation.max_new_tokens),
        )
        requested = int(requested)
        if not 1 <= requested <= self.config.codec.max_frames:
            raise ValueError(
                f"Fish S2-Pro max_new_tokens must be in 1..{self.config.codec.max_frames}, got {requested}"
            )
        return requested

    def get_loop_caps(self, **model_kwargs: Any) -> dict[str, int]:
        # the stop check counts the prefill's frame, so decode runs one iteration fewer than max tokens
        return {"ar_decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {AR_KV: {"main": KVLabel(
            growth=(("ar_decode", "ar_decode_loop"),),
        )}}

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.codec.sample_rate

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Encode emitted waveform tensors as raw little-endian PCM16 bytes."""
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported Fish S2-Pro output modality: {modality!r}")
        if output.numel() == 0:
            return b""
        pcm = output.detach().cpu()
        if pcm.is_floating_point():
            pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
        elif pcm.dtype != torch.int16:
            pcm = pcm.to(torch.int16)
        return pcm.contiguous().numpy().tobytes()

    # -----------------------------------------------------------------------
    # Lazy worker-side submodule and weight loading
    # -----------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        del sp_group
        if node_name not in ("AR", "Codec", "RefEncoder"):
            raise ValueError(f"Unknown Fish S2-Pro node: {node_name!r}")
        if tp_group is not None and getattr(tp_group, "world_size", 1) > 1:
            raise ValueError("Fish S2-Pro does not support tensor parallelism")
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        self._ensure_full_snapshot()
        if node_name == "AR":
            submodule = self._create_ar_submodule(device, autocast_dtype or torch.bfloat16)
        elif node_name == "Codec":
            submodule = self._create_codec_submodule(device)
        else:
            submodule = self._create_ref_encoder_submodule(device)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_ar_submodule(self, device: str, dtype: torch.dtype) -> NodeSubmodule:
        from mstar.model.fish_s2_pro.components.fast_ar import FishFastAR
        from mstar.model.fish_s2_pro.components.slow_ar import FishSlowAR
        from mstar.model.fish_s2_pro.submodules import ARSubmodule
        from mstar.model.loader.iterators import iter_safetensors_shards

        with torch.device("meta"):
            slow_ar = FishSlowAR(self.config)
            fast_ar = FishFastAR(self.config.fast_ar)
        slow_ar = slow_ar.to(dtype).to_empty(device=device)
        fast_ar = fast_ar.to(dtype).to_empty(device=device)
        slow_ar.load_weights(iter_safetensors_shards(self.local_dir, device=device, prefix="text_model."))
        load_fast_ar_weights(
            fast_ar,
            iter_safetensors_shards(self.local_dir, device=device, prefix="audio_decoder."),
        )
        fast_ar.rematerialize_buffers()
        return ARSubmodule(slow_ar.eval(), fast_ar.eval(), self.config)

    def _codec_state(self) -> dict[str, torch.Tensor]:
        from mstar.model.fish_s2_pro.components.dac import read_codec_checkpoint

        return read_codec_checkpoint(str(Path(self.local_dir) / CODEC_FILE))

    def _create_codec_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.fish_s2_pro.components.dac import FishDACDecoder, load_codec_state
        from mstar.model.fish_s2_pro.submodules import CodecSubmodule

        decoder = FishDACDecoder(self.config.codec)
        load_codec_state(decoder, self._codec_state(), DECODER_CODEC_PREFIXES)
        return CodecSubmodule(decoder.to(device).eval(), self.config, exact_fp32=self.codec_exact_fp32)

    def _create_ref_encoder_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.fish_s2_pro.components.dac import FishDACEncoder, load_codec_state
        from mstar.model.fish_s2_pro.submodules import RefEncoderSubmodule

        encoder = FishDACEncoder(self.config.codec)
        load_codec_state(encoder, self._codec_state(), ENCODER_CODEC_PREFIXES)
        return RefEncoderSubmodule(encoder.to(device).eval(), self.config)


# codec.pth subtrees each direction loads; together they cover the file.
DECODER_CODEC_PREFIXES = (
    "quantizer.semantic_quantizer.",
    "quantizer.quantizer.",
    "quantizer.post_module.",
    "quantizer.upsample.",
    "decoder.",
)
ENCODER_CODEC_PREFIXES = (
    "encoder.",
    "quantizer.semantic_quantizer.",
    "quantizer.quantizer.",
    "quantizer.downsample.",
    "quantizer.pre_module.",
)


def load_fast_ar_weights(fast_ar: torch.nn.Module, weights) -> None:
    """``audio_decoder.*`` tensors load under their own names; fail on any gap or extra."""
    params = dict(fast_ar.named_parameters())
    loaded: set[str] = set()
    for name, tensor in weights:
        name = name.removeprefix("audio_decoder.")
        target = params.get(name)
        if target is None:
            raise RuntimeError(f"Fish Fast AR checkpoint tensor {name} has no place in this port")
        if target.shape != tensor.shape:
            raise RuntimeError(f"Fish Fast AR {name}: checkpoint {tuple(tensor.shape)} vs {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(tensor.to(dtype=target.dtype))
        loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"Fish Fast AR checkpoint left {len(missing)} parameters unloaded: {missing[:5]}")
