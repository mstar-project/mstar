"""VoxCPM2 model contract and two-partition streaming topology.

VoxCPM2 (OpenBMB, Apache-2.0) is a tokenizer-free, diffusion-autoregressive
TTS: a MiniCPM-4 base LM and a residual acoustic LM drive a local DiT that
denoises one patch (4 frames, 160 ms) of continuous AudioVAE latents per
step; the AudioVAE V2 decodes those latents straight to 48 kHz.

Architecture (two asynchronous partitions):
    Talker      - text (+ reference audio) prefill, then one latent patch per step
    RefEncoder  - (voice cloning) AudioVAE encoder over the reference clip
    VAEDecoder  - streamed latent windows -> 48 kHz PCM

Streaming topology:
    Talker --[latent_patches, ScheduledLeftContextChunkPolicy(schedule, chunk, ctx)]--> VAEDecoder

``ctx`` covers the VAE decoder's causal receptive field, so streamed audio
equals a whole decode of the same latents, sample for sample in count.

Request modes (``process_prompt``), as the reference ``VoxCPM.generate``:
    zero-shot      text only; voice design is a ``(description)`` text prefix
    reference      ``ref_audio``: the clip isolated between ref tokens
    continuation   ``ref_audio`` + ``ref_text``: reference prefix, then the
                   transcript + clip that the model continues ("ultimate cloning")
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import re
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
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    ForwardPassArgs,
    KVLabel,
    LabelLayout,
    Model,
    ProcessPromptOutput,
    Span,
    TensorAndMetadata,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.voxcpm2.config import (
    AUDIO_START_TOKEN,
    BASE_ATTN,
    BASE_KV,
    BASE_POS,
    MIN_LEN_KEY,
    MODE_REF_CONTINUATION,
    MODE_REFERENCE,
    MODE_ZERO_SHOT,
    PATCH_BUDGET_KEY,
    PROMPT_MODE_KEY,
    REF_DIGEST_KEY,
    RES_ATTN,
    RES_KV,
    TALKER_STEP_METADATA,
    VoxCPM2Config,
)
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

STREAM_EDGE = "latent_patches"
# model_kwargs key: prompt patches leading a continuation's VAE stream
STREAM_LEAD_KEY = "vae_stream_lead_patches"
_CJK = re.compile(r"^[\u4e00-\u9fff]{2,}$")
logger = logging.getLogger(__name__)


def reference_digest(clip: torch.Tensor) -> str:
    """Content hash of a reference clip as the RefEncoder receives it (flat float32)."""
    samples = clip.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    return hashlib.blake2b(samples.numpy().tobytes(), digest_size=16).hexdigest()


def _resolve_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Config + tokenizer only; workers fetch the weights in ``get_submodule``."""
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )


def decoder_receptive_field_patches(config: VoxCPM2Config) -> int:
    """Patches of left context a streamed window needs to equal a whole decode."""
    from mstar.model.voxcpm2.components.audio_vae import CausalDecoder, decoder_receptive_field_frames

    with torch.device("meta"):
        decoder = CausalDecoder(config.audio_vae)
    return math.ceil(decoder_receptive_field_frames(decoder) / config.patch_size)


def stream_window_widths(schedule: tuple[int, ...], chunk: int, context: int) -> tuple[int, ...]:
    """Widths, in patches, of the VAE windows ``ScheduledLeftContextChunkPolicy``
    pops for a stream without a primed lead: the schedule's ramp, then
    ``chunk`` new patches behind a context that grows up to ``context``. No
    window, primed or final, is wider than the widest of these."""
    widths, delivered = {context + max((chunk, *schedule))}, 0
    for k in itertools.count():
        new = schedule[k] if k < len(schedule) else chunk
        held = min(delivered, context)
        widths.add(held + new)
        delivered += new
        if k >= len(schedule) and held == context:
            return tuple(sorted(widths))


class VoxCPM2Model(Model):
    """VoxCPM2 TTS: Talker + RefEncoder partition, streaming VAEDecoder partition.

    Deployment knobs (YAML ``model_kwargs``):

    * ``inference_timesteps`` (10): CFM steps per patch, fixed per server since
      the decode graph is captured with it.
    * ``stream_schedule`` ((1, 2, 4)), ``stream_chunk_patches`` (8): new patches
      per VAE window, ramping up so the first audio leaves after one patch.
    * ``stream_left_context_patches``: context per window; defaults to (and
      may not be below) the decoder's receptive field.
    * ``compile_talker`` (true): ``torch.compile`` the decode step before its
      CUDA graphs are captured. ``false`` captures the eager step: start-up in
      seconds instead of 10 to 30 minutes cold, at a lower decode rate.
    * ``max_text_tokens`` (``max_len / MAX_PATCHES_PER_TEXT_TOKEN``, 1365):
      longest text one request may speak; past it the ``max_len`` patch
      budget would end the audio before the text.
    """

    MODES = {"zero_shot": MODE_ZERO_SHOT, "reference": MODE_REFERENCE, "continuation": MODE_REF_CONTINUATION}
    REF_MIN_SECONDS = 0.5
    REF_MAX_SECONDS = 60.0
    REF_MIN_PEAK = 1e-3
    # the reference demo offers 1 to 3; float32 overflows to inf past about 3.4e38
    MAX_CFG_VALUE = 10.0
    INTERNAL_KWARGS = (*TALKER_STEP_METADATA, STREAM_LEAD_KEY)
    LENGTH_KWARGS = ("max_new_tokens", "max_output_tokens", "max_len")
    # English speech measured at 1.3 to 2.9 patches per text token (24 zero-shot
    # requests of 1,300 to 1,700 tokens), so text up to max_len / 3 tokens fits in max_len patches
    MAX_PATCHES_PER_TEXT_TOKEN = 3.0
    # configs/voxcpm2.yaml's max_concurrent_requests; the KV pools default to its size
    DEFAULT_MAX_CONCURRENT = 32
    PREFILL_INPUTS = ("text_inputs", "gen_params")
    REF_ENCODER_INPUTS = ("audio_inputs",)

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs: Any) -> None:
        from transformers import LlamaTokenizerFast

        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = _resolve_metadata(model_path_hf, cache_dir)
        self.config = VoxCPM2Config.from_pretrained(self.local_dir)
        self.inference_timesteps = int(kwargs.get("inference_timesteps", self.config.inference_timesteps))
        if self.inference_timesteps < 2:
            # CFG-Zero* skips the first step, so one step would return the noise
            raise ValueError("VoxCPM2 inference_timesteps must be at least 2")
        self.stream_schedule = tuple(int(n) for n in kwargs.get("stream_schedule", (1, 2, 4)))
        self.stream_chunk_patches = int(kwargs.get("stream_chunk_patches", 8))
        needed = decoder_receptive_field_patches(self.config)
        self.stream_left_context = int(kwargs.get("stream_left_context_patches", needed))
        if self.stream_left_context < needed:
            raise ValueError(
                f"VoxCPM2 stream_left_context_patches={self.stream_left_context} is below the VAE "
                f"decoder's {needed}-patch receptive field; streamed audio would differ from a whole decode"
            )
        if self.stream_chunk_patches <= 0 or any(n <= 0 for n in self.stream_schedule):
            raise ValueError("VoxCPM2 stream chunk sizes must be positive")
        if self.config.prompt_context_patches > self.stream_left_context:
            raise ValueError("VoxCPM2 continuation lead exceeds the stream's left context")
        self.compile_talker = kwargs.get("compile_talker", True)
        if not isinstance(self.compile_talker, bool):
            raise ValueError(f"VoxCPM2 compile_talker must be true or false, got {self.compile_talker!r}")
        limit = kwargs.get("max_text_tokens", int(self.config.max_len / self.MAX_PATCHES_PER_TEXT_TOKEN))
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"VoxCPM2 max_text_tokens must be a positive int, got {limit!r}")
        self.max_text_tokens = limit
        # The reference tokenizer: the checkpoint's Llama tokenizer, with
        # multi-character CJK tokens split into characters (``mask_multichar_chinese_tokens``).
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.local_dir)
        self._multichar_cjk = {tok for tok in self.tokenizer.get_vocab() if _CJK.match(tok)}
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -- tokenization -------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        tokens = []
        for token in self.tokenizer.tokenize(text):
            clean = token.replace("\u2581", "")
            tokens.extend(list(clean) if clean in self._multichar_cjk else [token])
        return self.tokenizer.convert_tokens_to_ids(tokens)

    # -- resources / graph --------------------------------------------------

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a request cap either LM cache cannot hold.

        Nothing preempts a request once a cache is full: the worker retries the
        step that found no free page until the requests time out, while the
        server still reports healthy. A request holds at most ``max_length``
        positions (``process_prompt`` refuses longer ones), so
        ``max_concurrent_requests`` such requests must fit in each cache next to
        its sink page; the conductor queues the rest.
        """
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning(
                "%s sets no max_concurrent_requests: a burst of long requests can fill "
                "the VoxCPM2 KV caches and stall until the requests time out", config_path,
            )
            return
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f"{config_path}: max_concurrent_requests must be a positive int; got {cap!r}")
        overrides = config.get("resources") or {}
        for spec in self.get_node_resources():
            if not isinstance(spec, KVSpec):
                continue
            override = overrides.get(spec.resource_key) or {}
            page_size = int(override.get("page_size", spec.config.page_size))
            max_num_pages = int(override.get("max_num_pages", spec.config.max_num_pages))
            per_request = -(-self.config.max_length // page_size)
            need = 1 + cap * per_request
            if need > max_num_pages:
                raise ValueError(
                    f"{config_path}: {cap} concurrent requests (max_concurrent_requests) need "
                    f"{need} pages of resources.{spec.resource_key} ({per_request} per "
                    f"{self.config.max_length}-position request and the sink page), which has "
                    f"{max_num_pages}; raise max_num_pages or lower max_concurrent_requests"
                )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        def kv(stack) -> PagedKVConfig:
            config = PagedKVConfig(
                num_layers=stack.num_hidden_layers,
                num_kv_heads=stack.num_key_value_heads,
                head_dim=stack.head_dim,
                num_qo_heads=stack.num_attention_heads,
                max_seq_len=self.config.max_length,
            )
            # the shipped max_concurrent_requests at the full context, and the
            # sink page: 2049 pages, about 9.7 GB for both LMs
            config.max_num_pages = 1 + self.DEFAULT_MAX_CONCURRENT * -(-config.max_seq_len // config.page_size)
            return config

        nodes = {"Talker"}
        return [
            KVSpec(resource_key=BASE_KV, nodes=nodes, config=kv(self.config.lm)),
            AttentionSpec(resource_key=BASE_ATTN, nodes=nodes, config=AttentionConfig(kv_cache=BASE_KV)),
            # positions only: LongRoPE is applied from the model's own table
            PositionSpec(resource_key=BASE_POS, nodes=nodes, config=PositionConfig(kv_cache=BASE_KV)),
            KVSpec(resource_key=RES_KV, nodes=nodes, config=kv(self.config.residual)),
            AttentionSpec(resource_key=RES_ATTN, nodes=nodes, config=AttentionConfig(kv_cache=RES_KV)),
        ]

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def talker(input_names: list[str], loop: bool) -> GraphNode:
            return GraphNode(
                name="Talker",
                input_names=input_names,
                outputs=[
                    GraphEdge(next_node="Talker", name="latent_patch")
                    if loop
                    else GraphEdge(next_node=EMPTY_DESTINATION, name="latent_patch", persist=True),
                    StreamingGraphEdge(next_node="VAEDecoder", name=STREAM_EDGE, target_partition="VAEDecoder"),
                ],
            )

        ref_encoder = GraphNode(
            name="RefEncoder",
            input_names=list(self.REF_ENCODER_INPUTS),
            outputs=[GraphEdge(next_node="Talker", name="ref_patches")],
        )
        return {
            "talker_prefill": talker(list(self.PREFILL_INPUTS), loop=False),
            "talker_prefill_clone": Sequential(
                [
                    ref_encoder,
                    talker([*self.PREFILL_INPUTS, "ref_patches"], loop=False),
                ]
            ),
            "talker_decode": Loop(
                name="talker_decode_loop",
                section=talker(["latent_patch"], loop=True),
                max_iters=self.get_max_output_tokens(),
                outputs=[],
            ),
            "vae_chunk": GraphNode(
                name="VAEDecoder",
                input_names=[STREAM_EDGE],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
            ),
        }

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="Talker",
                graph_walks={"talker_prefill", "talker_prefill_clone", "talker_decode"},
                initial_walk="talker_prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="VAEDecoder",
                graph_walks={"vae_chunk"},
                initial_walk=None,
                producer_partitions=["Talker"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        schedule, chunk, context = self.stream_schedule, self.stream_chunk_patches, self.stream_left_context
        return PartitionTopology(
            partitions=["Talker", "VAEDecoder"],
            connections=[
                Connection(
                    from_partition="Talker",
                    to_partition="VAEDecoder",
                    edge_name=STREAM_EDGE,
                    chunk_policy_factory=lambda: ScheduledLeftContextChunkPolicy(
                        schedule=schedule,
                        chunk=chunk,
                        left_context=context,
                    ),
                )
            ],
        )

    # -- API preprocessing ----------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """A reference clip as 16 kHz mono float32, resampled as ``librosa.load`` does (soxr HQ)."""
        import soundfile

        from mstar.model.media_limits import read_sound_file, sound_file_seconds

        rate = self.config.audio_vae.sample_rate
        try:
            info = soundfile.info(filepath)
            seconds = sound_file_seconds(info.frames, info.samplerate)
            if seconds is not None:
                self._check_reference_seconds(seconds)
                waveform, sample_rate = soundfile.read(filepath, dtype="float32", always_2d=True)
            else:
                # no sample count in the header (a FLAC written to a pipe): count it as it decodes
                waveform, sample_rate = read_sound_file(filepath, self.REF_MAX_SECONDS)
                self._check_reference_seconds(len(waveform) / sample_rate if sample_rate else 0.0)
        except soundfile.LibsndfileError:
            # the client's error (400); libsndfile's message names the server's upload path
            raise ValueError("VoxCPM2 ref_audio is not audio soundfile can read (WAV, FLAC, OGG or MP3)") from None
        audio = waveform.mean(axis=1)
        # as Chatterbox does; cloned from digital silence, the model speaks about
        # three times the usual length of near silence
        if float(abs(audio - audio.mean()).max()) < self.REF_MIN_PEAK:
            raise ValueError("VoxCPM2 ref_audio is silent (it never swings past -60 dBFS)")
        if sample_rate != rate:
            import soxr

            audio = soxr.resample(audio, sample_rate, rate, quality="HQ")
        return TensorAndMetadata(
            data=torch.from_numpy(audio).to(device),
            metadata={"sample_rate": rate, "num_channels": 1},
        )

    def _check_reference_seconds(self, seconds: float) -> None:
        if not self.REF_MIN_SECONDS <= seconds <= self.REF_MAX_SECONDS:
            raise ValueError(
                f"VoxCPM2 reference clip is {seconds:.2f} s; it must be between "
                f"{self.REF_MIN_SECONDS:g} and {self.REF_MAX_SECONDS:g} s"
            )

    @staticmethod
    def _number(kwargs: dict, key: str, default, kind=float):
        value = kwargs.get(key, default)
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"VoxCPM2 {key} must be a number, got {value!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"VoxCPM2 {key} must be finite")
        return value

    def _patch_cap(self, kwargs: dict) -> int:
        """The request's patch cap: ``max_new_tokens``, else the standard
        ``max_output_tokens``, else ``max_len``, else the model's."""
        for key in ("max_new_tokens", "max_output_tokens", "max_len"):
            if kwargs.get(key) is not None:
                return self._number(kwargs, key, self.config.max_len, int)
        return self.config.max_len

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> NameToTensorList | ProcessPromptOutput:
        """Tokenize the request into ``text_inputs`` and ``gen_params``
        (``[cfg_value]``); the prompt mode, patch budget, ``min_len`` and the
        reference clip's digest go to the Talker in ``step_metadata``.

        Voice design: ``instruct`` (or OpenAI's ``instructions``) becomes the
        ``(description)`` prefix of the text, the reference's format.
        """
        text = (prompt or "").replace("\n", " ")
        if not text.strip():
            raise ValueError("VoxCPM2 requires a non-empty text prompt")
        if not any(ch.isalnum() for ch in text):
            # emoji or punctuation alone still decode to seconds of audio
            raise ValueError("VoxCPM2 text has nothing to speak; it needs at least one letter or digit")
        if set(output_modalities) != {"audio"}:
            raise ValueError("VoxCPM2 supports audio output only")
        if "text" not in input_modalities or not set(input_modalities) <= {"text", "audio"}:
            raise ValueError("VoxCPM2 takes text, plus at most one reference clip")
        instruct = str(kwargs.get("instruct") or kwargs.get("instructions") or "").strip()
        if instruct:
            text = f"({instruct}){text}"
        # the reference's ``VoxCPM.generate``: runs of whitespace in the target become one space
        text = re.sub(r"\s+", " ", text)
        if "inference_timesteps" in kwargs and int(kwargs["inference_timesteps"]) != self.inference_timesteps:
            raise ValueError(
                f"VoxCPM2 serves inference_timesteps={self.inference_timesteps}; it is a deployment setting"
            )
        cfg_value = self._number(kwargs, "cfg_value", self.config.cfg_value)
        if not 0 <= cfg_value <= self.MAX_CFG_VALUE:
            raise ValueError(f"VoxCPM2 cfg_value must be between 0 and {self.MAX_CFG_VALUE:g}, got {cfg_value:g}")
        min_len = self._number(kwargs, "min_len", self.config.min_len, int)
        max_len = self._patch_cap(kwargs)
        if min_len < 0 or not 1 <= max_len <= self.config.max_len:
            raise ValueError(f"VoxCPM2 needs min_len >= 0 and 1 <= max_new_tokens <= {self.config.max_len}")

        clips = (tensors or {}).get("audio_inputs", [])
        ref_text = str(kwargs.get("ref_text") or "")
        if not ref_text.strip():
            ref_text = ""
        digest = None
        if "audio" in input_modalities or clips:
            if len(clips) != 1:
                raise ValueError(f"VoxCPM2 clones a voice from exactly one reference clip, got {len(clips)}")
            samples = int(clips[0].reshape(-1).shape[0])
            self._check_reference_seconds(samples / self.config.audio_vae.sample_rate)
            ref_patches = math.ceil(samples / self.config.ref_patch_samples)
            mode = MODE_REF_CONTINUATION if ref_text else MODE_REFERENCE
            digest = reference_digest(clips[0])
        else:
            if ref_text:
                raise ValueError("VoxCPM2 ref_text needs a ref_audio clip")
            ref_patches = 0
            mode = MODE_ZERO_SHOT

        target_ids = self.tokenize(text)
        if len(target_ids) > self.max_text_tokens:
            raise ValueError(
                f"VoxCPM2 text is {len(target_ids)} tokens; one request speaks at most "
                f"{self.max_text_tokens} (max_text_tokens). Split the text into shorter requests"
            )
        # the reference tokenizes the transcript and the target as one string, with no separator
        ids = self.tokenize(ref_text + text) if mode == MODE_REF_CONTINUATION else list(target_ids)
        ids.append(AUDIO_START_TOKEN)
        max_patches = self.config.max_patches(len(target_ids), max_len)
        prompt_len = len(ids) + (ref_patches + 2 if mode != MODE_ZERO_SHOT else 0)
        prompt_len += ref_patches if mode == MODE_REF_CONTINUATION else 0
        if prompt_len + max_patches > self.config.max_length:
            raise ValueError(
                f"VoxCPM2 prompt of {prompt_len} positions plus up to {max_patches} patches exceeds "
                f"the model's {self.config.max_length}-position context"
            )
        outputs = {
            "text_inputs": [torch.tensor(ids, dtype=torch.long)],
            "gen_params": [torch.tensor([cfg_value], dtype=torch.float32)],
        }
        # every key is set in every mode: the data worker lays this over the
        # request's own kwargs, so a key left out stays client-controlled
        lead = min(self.config.prompt_context_patches, ref_patches) if mode == MODE_REF_CONTINUATION else 0
        metadata = {
            PATCH_BUDGET_KEY: max_patches,
            PROMPT_MODE_KEY: mode,
            MIN_LEN_KEY: min_len,
            REF_DIGEST_KEY: digest,
            STREAM_LEAD_KEY: lead,
        }
        # both LMs write the same rows: the prompt (``build_prompt``), then one per decode step
        walk = "talker_prefill" if mode == MODE_ZERO_SHOT else "talker_prefill_clone"
        prefill = Span("ids" if mode == MODE_ZERO_SHOT else "rows", walk, prompt_len, prompt_len, "text_inputs")
        layout = LabelLayout(spans=(prefill,), growth=(("talker_decode_loop", 1),))
        metadata["kv_layout"] = {BASE_KV: {"main": layout}, RES_KV: {"main": layout}}
        return ProcessPromptOutput(outputs, metadata)

    # -- conductor state machine -------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        model_kwargs = model_kwargs or {}
        if partition_name == "Talker":
            # the mode process_prompt built the prompt for, not the caller's
            # input_modalities: a clip sent with ["text"] still clones
            if PROMPT_MODE_KEY not in model_kwargs:
                raise ValueError(f"VoxCPM2 request is missing {PROMPT_MODE_KEY}, which process_prompt sets")
            clone = int(model_kwargs[PROMPT_MODE_KEY]) != MODE_ZERO_SHOT
            walk = "talker_prefill_clone" if clone else "talker_prefill"
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk=walk,
                is_prefill=True,
                kwargs={PATCH_BUDGET_KEY: int(model_kwargs.get(PATCH_BUDGET_KEY, 0))},
            )
            routes = [(name, "Talker") for name in self.PREFILL_INPUTS]
            if clone:
                routes += [(name, "RefEncoder") for name in self.REF_ENCODER_INPUTS]
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
                step_metadata={key: model_kwargs[key] for key in TALKER_STEP_METADATA if key in model_kwargs},
            )
        if partition_name == "VAEDecoder":
            lead = int(model_kwargs.get(STREAM_LEAD_KEY, 0))
            return ForwardPassArgs(
                full_metadata=CurrentForwardConductorMetadata(
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    graph_walk="vae_chunk",
                    is_prefill=False,
                ),
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
                stream_lead_items={STREAM_EDGE: lead} if lead else {},
            )
        raise ValueError(f"Unknown VoxCPM2 partition: {partition_name!r}")

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        del incoming_connections
        if partition_name == "Talker":
            if partition_metadata.graph_walk in ("talker_prefill", "talker_prefill_clone"):
                if partition_metadata.kwargs.get(PATCH_BUDGET_KEY) == 1:
                    return ForwardPassArgs(
                        full_metadata=partition_metadata, inputs=[], unpersist_tensors=[], request_done=True
                    )
                partition_metadata.graph_walk = "talker_decode"
                partition_metadata.is_prefill = False
                edge = GraphEdge(next_node="Talker", name="latent_patch")
                edge.tensor_info = persist_signals.get("latent_patch", [])
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[edge],
                    unpersist_tensors=list(edge.tensor_info),
                )
            if partition_metadata.graph_walk == "talker_decode":
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    request_done=True,
                )
            raise ValueError(f"Talker entered an unexpected graph walk: {partition_metadata.graph_walk!r}")
        if partition_name == "VAEDecoder":
            partition_metadata.graph_walk = "vae_chunk"
            return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[])
        raise ValueError(f"Unknown VoxCPM2 partition: {partition_name!r}")

    # -- output ---------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs: Any) -> int:
        return self._patch_cap(model_kwargs)

    def get_loop_caps(self, **model_kwargs: Any) -> dict[str, int]:
        # the stop check counts the prefill's patch, so decode runs one iteration fewer than the budget
        budget = self._patch_cap(model_kwargs)
        if model_kwargs.get(PATCH_BUDGET_KEY) is not None:
            budget = min(budget, int(model_kwargs[PATCH_BUDGET_KEY]))
        return {"talker_decode_loop": max(1, budget - 1)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        talker = {"main": KVLabel(growth=(("talker_decode", "talker_decode_loop"),))}
        return {BASE_KV: talker, RES_KV: talker}

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.output_sample_rate

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Raw little-endian PCM16 at 48 kHz."""
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported VoxCPM2 output modality: {modality!r}")
        if output.numel() == 0:
            return b""
        pcm = output.detach().cpu()
        if pcm.is_floating_point():
            pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
        return pcm.contiguous().numpy().tobytes()

    # -- worker-side construction ---------------------------------------------------

    def _checkpoint_dir(self) -> str:
        if (Path(self.local_dir) / "model.safetensors").is_file():
            return self.local_dir
        if Path(self.model_path_hf).is_dir():
            raise FileNotFoundError(f"No model.safetensors in {self.model_path_hf}")
        from huggingface_hub import snapshot_download

        self.local_dir = snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)
        return self.local_dir

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        del sp_group
        if node_name not in ("Talker", "RefEncoder", "VAEDecoder"):
            raise ValueError(f"Unknown VoxCPM2 node: {node_name!r}")
        if node_name not in self._submodule_cache:
            from mstar.model.voxcpm2 import submodules

            checkpoint = self._checkpoint_dir()
            if node_name == "Talker":
                submodule = submodules.TalkerSubmodule(
                    load_generator(self.config, checkpoint, device, autocast_dtype or torch.bfloat16, tp_group),
                    self.config,
                    self.inference_timesteps,
                    compile_decode=self.compile_talker,
                )
            elif node_name == "RefEncoder":
                submodule = submodules.RefEncoderSubmodule(load_audio_vae(self.config, checkpoint, device), self.config)
            else:
                submodule = submodules.VAEDecoderSubmodule(
                    load_audio_vae(self.config, checkpoint, device),
                    self.config,
                    window_buckets=stream_window_widths(
                        self.stream_schedule, self.stream_chunk_patches, self.stream_left_context
                    ),
                )
            self._submodule_cache[node_name] = submodule
        return self._submodule_cache[node_name]


def load_generator(config: VoxCPM2Config, checkpoint: str, device, dtype: torch.dtype, tp_group=None):
    """Both LMs and the patch heads from ``model.safetensors``, every key accounted for."""
    from mstar.model.loader.iterators import iter_safetensors_shards
    from mstar.model.voxcpm2.components.talker import VoxCPM2Generator

    with torch.device("meta"):
        generator = VoxCPM2Generator(config, comm_group=tp_group)
    generator = generator.to(dtype)
    generator.to_empty(device=device)
    loaded = generator.load_weights(iter_safetensors_shards(checkpoint, device=device))
    missing = sorted(set(dict(generator.named_parameters())) - loaded)
    if missing:
        raise RuntimeError(f"VoxCPM2 checkpoint left {len(missing)} parameters unloaded, e.g. {missing[:5]}")
    generator.materialize_buffers()
    return generator.eval()


def load_audio_vae(config: VoxCPM2Config, checkpoint: str, device):
    """The fp32 AudioVAE from ``audiovae.safetensors`` / ``audiovae.pth``, weight norm folded."""
    from mstar.model.voxcpm2.components.audio_vae import AudioVAE, fold_weight_norm

    root = Path(checkpoint)
    if (root / "audiovae.safetensors").is_file():
        from safetensors.torch import load_file

        state = load_file(str(root / "audiovae.safetensors"))
    else:
        state = torch.load(root / "audiovae.pth", map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
    vae = AudioVAE(config.audio_vae)
    result = vae.load_state_dict(fold_weight_norm(state), strict=True)
    del result
    return vae.to(device=device, dtype=torch.float32).eval()
