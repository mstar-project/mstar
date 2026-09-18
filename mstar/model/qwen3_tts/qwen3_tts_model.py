"""Qwen3-TTS model contract and two-partition streaming topology.

One class serves every 12 Hz checkpoint: 0.6B/1.7B CustomVoice (built-in
speakers, style instructions on 1.7B), 1.7B VoiceDesign (voice described by
an instruction) and 1.7B Base (voice cloned from reference audio). They share
one architecture: an autoregressive Talker predicts one 12 Hz codec frame per
step, group 0 from the Talker language model and groups 1-15 from a small
depth-wise CodePredictor, and the speech-tokenizer decoder turns those frames
into 24 kHz PCM. What differs is only the prefill conditioning, which is
derived from ``config.json`` (``Qwen3TTSModelConfig``), never from the
registry key.

Architecture (two asynchronous partitions):
    Talker     - text/voice prefill, then autoregressive 16-group codec frames
    RefEncoder - (Base) reference clip -> x-vector + codec frames, feeds the prefill
    Codec      - stateless speech-tokenizer decoder producing PCM chunks

Streaming topology:
    Talker --[codec_tokens, ScheduledLeftContextChunkPolicy((4, 8, 16), 25, 25)]--> Codec

Request state machine:
    Talker: talker_prefill | talker_prefill_clone -> talker_decode loop -> done on EOS/token limit
    Codec:  waits for streamed frames -> codec_chunk | codec_chunk_clone -> emits audio -> waits

This class runs in the API/conductor side. It owns request validation, graph
and partition declarations, state-machine transitions, sampling defaults, and
lazy worker-side construction. Heavy weights are not loaded in ``__init__``.
"""

import importlib.metadata
import importlib.util
import sys
from functools import lru_cache
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
    KVConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.qwen3_tts.config import (
    CHATML_ASSISTANT_PREFIX_TOKEN_IDS,
    CHATML_ASSISTANT_SUFFIX_TOKEN_IDS,
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    Qwen3TTSModelConfig,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _resolve_model_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Resolve only the files needed to construct config and tokenize input.

    The API and conductor processes do not need model tensors. Downloading a
    metadata-only snapshot here keeps them from allocating or transferring the
    multi-gigabyte checkpoint. Workers fetch the complete snapshot lazily from
    ``get_submodule``.
    """
    local_path = Path(repo_id)
    if local_path.is_dir():
        return str(local_path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "speech_tokenizer/config.json",
        ],
    )


@lru_cache(maxsize=1)
def _load_qwen3_tts_codec_classes() -> tuple[type, type, type]:
    """Load only qwen-tts' 12 Hz speech-tokenizer modules.

    Returns the decoder config class, the decoder (codes -> waveform) and the
    encoder (waveform -> codes, used for voice-clone reference audio).

    ``qwen_tts.__init__`` eagerly imports its high-level inference package,
    which in turn imports the unrelated 25 Hz tokenizer and pysox.  Pysox
    probes for a system ``sox`` executable at import time and emits a warning
    even though the 12 Hz decoder used here never calls it. Load the two exact
    source files under an M*-private package name so their relative import
    works without executing those broad ``__init__`` files or claiming the
    public ``qwen_tts`` names in ``sys.modules``.
    """
    try:
        distribution = importlib.metadata.distribution("qwen-tts")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Qwen3-TTS Codec requires the 'qwen-tts' package; install "
            "M* with the qwen3_tts optional dependency"
        ) from exc

    source_root = Path(distribution.locate_file(
        "qwen_tts/core/tokenizer_12hz"
    ))
    private_package = "_mstar_qwen3_tts_tokenizer_12hz"
    config_name = (
        f"{private_package}.configuration_qwen3_tts_tokenizer_v2"
    )
    model_name = f"{private_package}.modeling_qwen3_tts_tokenizer_v2"
    loaded_names = [private_package, config_name, model_name]

    package_spec = importlib.util.spec_from_file_location(
        private_package,
        source_root / "__init__.py",
        submodule_search_locations=[str(source_root)],
    )
    if package_spec is None:
        raise ImportError(f"Cannot load qwen-tts package from {source_root}")
    sys.modules[private_package] = importlib.util.module_from_spec(package_spec)

    def load_private_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load qwen-tts module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        config_module = load_private_module(
            config_name,
            source_root / "configuration_qwen3_tts_tokenizer_v2.py",
        )
        model_module = load_private_module(
            model_name,
            source_root / "modeling_qwen3_tts_tokenizer_v2.py",
        )
    except Exception:
        for name in loaded_names:
            sys.modules.pop(name, None)
        raise

    return (
        config_module.Qwen3TTSTokenizerV2DecoderConfig,
        model_module.Qwen3TTSTokenizerV2Decoder,
        model_module.Qwen3TTSTokenizerV2Encoder,
    )


# ---------------------------------------------------------------------------
# Checkpoint completeness
# ---------------------------------------------------------------------------

# Fused M* parameters and the per-shard checkpoint keys that feed them
# (mirrors ``LLAMA_STACKED_PARAMS`` in the loader).
_FUSED_SOURCES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _checkpoint_keys(checkpoint_dir: str | Path, prefix: str) -> set[str]:
    """Tensor names under ``prefix`` in a (possibly sharded) safetensors checkpoint."""
    import json

    from safetensors import safe_open

    root = Path(checkpoint_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as f:
            names = json.load(f)["weight_map"].keys()
    else:
        with safe_open(str(root / "model.safetensors"), framework="pt") as f:
            names = list(f.keys())
    return {name.removeprefix(prefix) for name in names if name.startswith(prefix)}


def _expected_checkpoint_keys(module: torch.nn.Module) -> set[str]:
    """Checkpoint keys an M* module consumes: its state dict (parameters and
    persistent buffers), with fused projections expanded to their shards."""
    expected: set[str] = set()
    for name in module.state_dict():
        for fused, sources in _FUSED_SOURCES.items():
            if f".{fused}." in name:
                expected.update(name.replace(f".{fused}.", f".{source}.") for source in sources)
                break
        else:
            expected.add(name)
    return expected


def _load_buffers(
    module: torch.nn.Module,
    weights,
) -> set[str]:
    """Copy checkpoint tensors into the module's persistent buffers.

    ``load_hf_weights`` only fills parameters. Codec quantizers keep their
    codebooks in buffers (``embed_sum`` / ``cluster_usage``), so a checkpoint
    must be able to refill those too. Returns the buffer names it filled.
    """
    persistent = set(module.state_dict()) - set(dict(module.named_parameters()))
    buffers = {name: buf for name, buf in module.named_buffers() if name in persistent}
    loaded: set[str] = set()
    for name, tensor in weights:
        target = buffers.get(name)
        if target is None:
            continue
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded.add(name)
    return loaded


def _verify_checkpoint_coverage(
    module: torch.nn.Module,
    loaded: set[str],
    checkpoint_keys: set[str],
    component: str,
) -> None:
    """Fail startup on any state left uninitialized or any key left unused.

    Both directions matter: a missing key means random weights (or default
    buffers) would serve requests; an unused key means the checkpoint carries
    a component this port silently ignores (the 1.7B code predictor
    projection, or a quantizer codebook kept in buffers, for example).
    """
    expected = set(module.state_dict())
    missing = sorted(expected - loaded)
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"{component} checkpoint did not initialize {len(missing)} "
            f"tensors: {preview}"
        )
    unused = sorted(
        key for key in checkpoint_keys - _expected_checkpoint_keys(module)
        if "rotary_emb" not in key
    )
    if unused:
        preview = ", ".join(unused[:8])
        raise RuntimeError(
            f"{component} checkpoint has {len(unused)} tensors this port does "
            f"not load: {preview}"
        )


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------


class Qwen3TTSModel(Model):
    """Qwen3-TTS 12 Hz model contract (CustomVoice, VoiceDesign, Base).

    GPU computation is split into an autoregressive Talker partition and a
    streaming Codec partition. This class owns only model-level scheduling,
    prompt processing, configuration, and output encoding.
    """

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir

        # The lightweight API-side object needs config and tokenizer only.
        self.local_dir = _resolve_model_metadata(model_path_hf, cache_dir)
        # Rejects unknown ``tts_model_type`` values; every supported variant
        # is handled below through the config's capability properties.
        self.config = Qwen3TTSModelConfig.from_pretrained(self.local_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir,
            cache_dir=cache_dir,
            fix_mistral_regex=True,
        )

        # Each worker asks only for nodes assigned to it. Cache the resulting
        # wrappers so Talker and Codec weights are materialized at most once.
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def _ensure_full_snapshot(self) -> str:
        """Make all weight files available immediately before worker loading."""
        if (Path(self.local_dir) / "model.safetensors").is_file():
            return self.local_dir
        if Path(self.model_path_hf).is_dir():
            raise FileNotFoundError(
                f"No model.safetensors found in {self.model_path_hf}"
            )
        from huggingface_hub import snapshot_download

        self.local_dir = snapshot_download(
            repo_id=self.model_path_hf,
            cache_dir=self.cache_dir,
        )
        return self.local_dir

    # -----------------------------------------------------------------------
    # Model ABC: resources
    # -----------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Talker paged KV + attention/position/sampler resources.

        The CodePredictor's frame-local KV scratch is not a resource; the
        Talker submodule owns it (overwritten every step)."""
        talker = self.config.talker
        cp = talker.code_predictor
        talker_kv = KVConfig(
            num_layers=talker.num_hidden_layers,
            num_kv_heads=talker.num_key_value_heads,
            head_dim=talker.head_dim,
            max_seq_len=talker.max_position_embeddings,
            num_qo_heads=talker.num_attention_heads,
        )
        return [
            KVSpec(resource_key=TALKER_KV, nodes={"Talker"}, config=talker_kv),
            AttentionSpec(
                resource_key=TALKER_ATTN, nodes={"Talker"},
                config=AttentionConfig(kv_cache=TALKER_KV),
            ),
            PositionSpec(
                resource_key=TALKER_POS, nodes={"Talker"},
                config=PositionConfig(kv_cache=TALKER_KV),
            ),
            SamplerSpec(
                resource_key=TALKER_SAMPLER, nodes={"Talker"},
                vocab_size=talker.vocab_size,
                enable_repetion_penalty=True,
            ),
            SamplerSpec(
                resource_key=CODE_PRED_SAMPLER, nodes={"Talker"},
                vocab_size=cp.vocab_size,
                enable_repetion_penalty=False,
            ),
        ]

    # -----------------------------------------------------------------------
    # Model ABC: walk graph declaration
    # -----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """Declare prefill, autoregressive decode, and codec chunk walks.

        ``talker_input_embeds`` is the recurrent Talker edge. ``codec_tokens``
        crosses the asynchronous partition boundary and is buffered according
        to ``get_partition_topology`` before Codec is scheduled. Base
        checkpoints add a clone prefill (RefEncoder -> Talker) and a codec walk
        that also receives the reference frame count to trim.
        """
        def talker_prefill_node(input_names: list[str]) -> GraphNode:
            # Prefill seeds both recurrent paths: the embedding for the next
            # Talker step is persisted, while the first codec frame(s) start
            # the Talker-to-Codec stream.
            return GraphNode(
                name="Talker",
                input_names=input_names,
                outputs=[
                    GraphEdge(
                        next_node=EMPTY_DESTINATION,
                        name="talker_input_embeds",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Codec",
                        name="codec_tokens",
                        target_partition="Codec",
                    ),
                ],
            )

        def codec_node(input_names: list[str]) -> GraphNode:
            # Codec is deliberately a separate walk/engine so waveform decoding
            # can overlap with subsequent Talker steps.
            return GraphNode(
                name="Codec",
                input_names=input_names,
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="audio_chunk",
                        output_modality="audio",
                    ),
                ],
            )

        # Each loop iteration predicts one complete 16-group codec frame and
        # feeds the summed codec embedding back into the next Talker step.
        talker_decode = Loop(
            name="talker_decode_loop",
            section=GraphNode(
                name="Talker",
                input_names=["talker_input_embeds"],
                outputs=[
                    GraphEdge(
                        next_node="Talker",
                        name="talker_input_embeds",
                    ),
                    StreamingGraphEdge(
                        next_node="Codec",
                        name="codec_tokens",
                        target_partition="Codec",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        walks = {
            "talker_prefill": talker_prefill_node(list(self.PREFILL_INPUTS)),
            "talker_decode": talker_decode,
            "codec_chunk": codec_node(["codec_tokens"]),
        }
        if self.config.supports_reference_audio:
            ref_encoder = GraphNode(
                name="RefEncoder",
                input_names=list(self.REF_ENCODER_INPUTS),
                outputs=[
                    GraphEdge(next_node="Talker", name="speaker_embed"),
                    GraphEdge(next_node="Talker", name="ref_codes"),
                ],
            )
            walks["talker_prefill_clone"] = Sequential([
                ref_encoder,
                talker_prefill_node([*self.PREFILL_INPUTS, "speaker_embed", "ref_codes"]),
            ])
            walks["codec_chunk_clone"] = codec_node(["codec_tokens", "ref_frames"])
        return walks

    # -----------------------------------------------------------------------
    # Asynchronous partitions and stream buffering
    # -----------------------------------------------------------------------

    def get_partitions(self) -> list[PartitionDefinition]:
        """Split autoregressive generation from independently scheduled audio."""
        talker_walks = {"talker_prefill", "talker_decode"}
        codec_walks = {"codec_chunk"}
        if self.config.supports_reference_audio:
            talker_walks.add("talker_prefill_clone")
            codec_walks.add("codec_chunk_clone")
        return [
            PartitionDefinition(
                name="Talker",
                graph_walks=talker_walks,
                initial_walk="talker_prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="Codec",
                graph_walks=codec_walks,
                initial_walk=None,
                producer_partitions=["Talker"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        """Buffer codec frames in a ramp of chunks with left context.

        The first Codec invocation runs after ``chunk_schedule[0]`` frames so
        audio starts flowing early; chunks then grow to ``chunk_frames``. Every
        window after the first is preceded by up to ``left_context_frames``
        already decoded frames to avoid boundary artifacts;
        ``CodecSubmodule.postprocess`` removes the duplicated PCM prefix using
        the context count the stream buffer reports for each window.
        """
        codec = self.config.codec
        return PartitionTopology(
            partitions=["Talker", "Codec"],
            connections=[
                Connection(
                    from_partition="Talker",
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

    # Prompt templates of the reference ``qwen_tts`` inference wrapper. The
    # assistant wrapper is fixed at 3 + 5 tokens (see ``_validate_chatml``),
    # which is how the text span is located inside the tokenized turn.
    ASSISTANT_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    INSTRUCT_TEMPLATE = "<|im_start|>user\n{instruct}<|im_end|>\n"
    REFERENCE_TEMPLATE = "<|im_start|>assistant\n{text}<|im_end|>\n"
    # API tensors the Talker prefill consumes; the clone prefill adds the
    # RefEncoder's ``speaker_embed`` and ``ref_codes`` to them.
    PREFILL_INPUTS = ("text_inputs", "prompt_layout", "speaker_id", "language_id")
    # API tensors the RefEncoder consumes (reference clip + layout).
    REF_ENCODER_INPUTS = ("audio_inputs", "prompt_layout")

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """Decode a reference clip to 24 kHz mono float32 (speaker encoder + codec rate)."""
        import soundfile
        import torchaudio.functional as audio_functional

        waveform, sample_rate = soundfile.read(filepath, dtype="float32", always_2d=True)
        audio = torch.from_numpy(waveform).mean(dim=1)
        target_rate = self.config.codec.input_sample_rate
        if sample_rate != target_rate:
            audio = audio_functional.resample(audio, sample_rate, target_rate)
        return TensorAndMetadata(
            data=audio.to(device), metadata={"sample_rate": target_rate, "num_channels": 1}
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", padding=True)
        ids = encoded["input_ids"]
        if ids.ndim == 2:
            ids = ids[0]
        return ids.to(dtype=torch.long)

    def _resolve_speaker(self, kwargs: dict[str, Any]) -> tuple[str | None, int]:
        """Map ``voice``/``speaker`` onto a codec speaker tag (``-1`` = none).

        CustomVoice checkpoints carry named speakers and fall back to the
        default one. VoiceDesign and Base have none: the voice comes from the
        instruction or the reference audio, so naming one is an error rather
        than something to ignore silently.
        """
        requested = kwargs.get("speaker", kwargs.get("voice"))
        if requested is None or requested == "":
            requested = self.config.default_speaker
            if requested is None:
                return None, -1
        elif not self.config.has_builtin_speakers:
            raise ValueError(
                f"Qwen3-TTS {self.config.tts_model_type} checkpoints have no "
                "built-in speakers; describe the voice with 'instruct' "
                "(VoiceDesign) or supply reference audio (Base) instead of 'voice'"
            )
        speaker = str(requested).lower()
        if speaker not in self.config.talker.spk_id:
            supported = ", ".join(sorted(self.config.talker.spk_id))
            raise ValueError(
                f"Unsupported Qwen3-TTS speaker {speaker!r}; supported: {supported}"
            )
        return speaker, self.config.talker.spk_id[speaker]

    def _resolve_language(self, kwargs: dict[str, Any], speaker: str | None) -> int:
        """Map ``language`` onto a codec language tag (``-1`` = automatic)."""
        language = str(kwargs.get("language", self.config.default_language)).lower()
        dialect = (
            self.config.talker.spk_is_dialect.get(speaker, False)
            if speaker is not None
            else False
        )
        if dialect and language in {"auto", "chinese"}:
            language = str(dialect).lower()

        # Validate after applying the speaker's dialect. Otherwise a malformed
        # dialect mapping falls through ``dict.get(..., -1)`` below and silently
        # degrades to automatic language selection.
        if language != "auto" and language not in self.config.talker.codec_language_id:
            supported = ", ".join(
                ["auto", *sorted(self.config.talker.codec_language_id)]
            )
            raise ValueError(
                f"Unsupported Qwen3-TTS language {language!r}; supported: {supported}"
            )
        return self.config.talker.codec_language_id.get(language, -1)

    def _resolve_instruct(self, kwargs: dict[str, Any]) -> str:
        """Style/voice instruction; ``instructions`` is the OpenAI field name."""
        instruct = kwargs.get("instruct", kwargs.get("instructions")) or ""
        instruct = str(instruct).strip()
        if instruct and not self.config.supports_instruct:
            raise ValueError(
                f"Qwen3-TTS {self.config.tts_model_size} "
                f"{self.config.tts_model_type} does not support instructions"
            )
        if not instruct and self.config.requires_instruct:
            raise ValueError(
                "Qwen3-TTS VoiceDesign requires an 'instruct' describing the voice"
            )
        return instruct

    def _resolve_reference(
        self, kwargs: dict[str, Any], tensors: NameToTensorList | None, input_modalities: list[str],
    ) -> tuple[str, int]:
        """Voice-clone reference: (transcript, frames). ``frames == 0`` means x-vector only.

        Base needs exactly one reference clip. In-context cloning (the
        default) also needs the clip's transcript; ``x_vector_only_mode``
        drops both the transcript and the frames and conditions on the
        x-vector alone.
        """
        clips = (tensors or {}).get("audio_inputs", [])
        if not self.config.supports_reference_audio:
            if clips or "audio" in input_modalities:
                raise ValueError(
                    f"Qwen3-TTS {self.config.tts_model_type} does not take reference "
                    "audio; voice cloning needs a Base checkpoint"
                )
            return "", 0
        if len(clips) != 1:
            raise ValueError(
                "Qwen3-TTS Base clones a voice from exactly one reference clip "
                f"(got {len(clips)}); pass it as the request's audio input"
            )
        ref_text = str(kwargs.get("ref_text") or "").strip()
        if kwargs.get("x_vector_only_mode", False):
            return "", 0
        if not ref_text:
            raise ValueError(
                "Qwen3-TTS Base needs 'ref_text' (the reference clip's transcript) "
                "unless 'x_vector_only_mode' is set"
            )
        num_samples = int(clips[0].reshape(-1).shape[0])
        frames = self.config.codec.frames_for_samples(num_samples)
        if frames < 1:
            raise ValueError("Qwen3-TTS reference clip is empty")
        return ref_text, frames

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        """Validate a request against the checkpoint variant and tokenize it.

        Produces the ``PREFILL_INPUTS`` tensors. ``text_inputs`` is the
        optional instruction turn, the assistant turn and (in-context clone)
        the reference transcript turn, all in the reference ChatML templates;
        ``prompt_layout`` is ``[instruct_len, text_len, stream_text,
        ref_text_len, ref_frames]``. ``stream_text`` follows the reference
        default per variant (whole text in the prefill for CustomVoice and
        VoiceDesign, one token per frame for Base) unless the request sets
        ``non_streaming_mode``. Base requests also get ``ref_frames`` as its
        own tensor for the codec's trimming.
        """
        if not prompt:
            raise ValueError("Qwen3-TTS requires a non-empty text prompt")
        if "audio" in input_modalities and not self.config.supports_reference_audio:
            raise ValueError(
                f"Qwen3-TTS {self.config.tts_model_type} does not take reference "
                "audio; voice cloning needs a Base checkpoint"
            )
        allowed = {"text", "audio"} if self.config.supports_reference_audio else {"text"}
        if "text" not in input_modalities or not set(input_modalities) <= allowed:
            raise ValueError(
                "Qwen3-TTS currently supports text input only"
                + (" (plus one reference clip for Base)" if self.config.supports_reference_audio else "")
            )
        if set(output_modalities) != {"audio"}:
            raise ValueError("Qwen3-TTS supports audio output only")

        speaker, speaker_id = self._resolve_speaker(kwargs)
        language_id = self._resolve_language(kwargs, speaker)
        instruct = self._resolve_instruct(kwargs)
        ref_text, ref_frames = self._resolve_reference(kwargs, tensors, input_modalities)
        stream_text = not bool(
            kwargs.get("non_streaming_mode", self.config.default_non_streaming_mode)
        )

        assistant_ids = self._tokenize(self.ASSISTANT_TEMPLATE.format(text=prompt))
        wrapper_len = len(CHATML_ASSISTANT_PREFIX_TOKEN_IDS) + len(
            CHATML_ASSISTANT_SUFFIX_TOKEN_IDS
        )
        text_len = assistant_ids.numel() - wrapper_len
        if text_len < 1:
            raise ValueError("Qwen3-TTS prompt tokenized to no text tokens")
        instruct_ids = (
            self._tokenize(self.INSTRUCT_TEMPLATE.format(instruct=instruct))
            if instruct
            else assistant_ids.new_empty(0)
        )
        ref_ids = (
            self._tokenize(self.REFERENCE_TEMPLATE.format(text=ref_text))
            if ref_frames
            else assistant_ids.new_empty(0)
        )
        outputs = {
            "text_inputs": [torch.cat([instruct_ids, assistant_ids, ref_ids])],
            "prompt_layout": [torch.tensor(
                [instruct_ids.numel(), text_len, int(stream_text), ref_ids.numel(), ref_frames],
                dtype=torch.long,
            )],
            "speaker_id": [torch.tensor([speaker_id], dtype=torch.long)],
            "language_id": [torch.tensor([language_id], dtype=torch.long)],
        }
        if self.config.supports_reference_audio:
            outputs["ref_frames"] = [torch.tensor([ref_frames], dtype=torch.long)]
        return outputs

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
        """Create each partition's initial state.

        Talker starts immediately from API tensors; a request that carries a
        reference clip takes the clone prefill, whose RefEncoder runs first.
        Codec has no direct API inputs and remains dormant until its incoming
        streaming connection has enough frames to schedule its chunk walk.
        """
        model_kwargs = model_kwargs or {}
        clone = "audio" in input_modalities
        if partition_name == "Talker":
            walk = "talker_prefill_clone" if clone else "talker_prefill"
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk=walk,
                is_prefill=True,
                kwargs={
                    "talker_max_tokens": self.get_max_output_tokens(**model_kwargs),
                },
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
                unpersist_tensors=sum(
                    [edge.tensor_info for edge in inputs], start=[]
                ),
                request_done="audio" not in output_modalities,
                step_metadata=self._talker_step_metadata(metadata),
            )

        if partition_name == "Codec":
            metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="codec_chunk_clone" if clone else "codec_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=metadata,
                # The clone walk needs the reference frame count next to the
                # stream from its very first chunk; the API tensor stays
                # persisted so every later re-arm can read it again.
                inputs=self._codec_ref_frames_edges(input_signals) if clone else [],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
            )
        raise ValueError(f"Unknown Qwen3-TTS partition: {partition_name!r}")

    @staticmethod
    def _codec_ref_frames_edges(
        signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        edge = GraphEdge(next_node="Codec", name="ref_frames")
        edge.tensor_info = signals.get("ref_frames", [])
        return [edge]

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Advance Talker prefill/decode and re-arm the streaming Codec.

        Loop iterations are executed inside the ``talker_decode`` graph walk,
        so the conductor sees that walk once after its loop stops. Codec is
        rescheduled by connection readiness rather than an internal walk
        transition.
        """
        del incoming_connections
        if partition_name == "Talker":
            if partition_metadata.graph_walk in ("talker_prefill", "talker_prefill_clone"):
                partition_metadata.graph_walk = "talker_decode"
                partition_metadata.is_prefill = False
                edge = GraphEdge(next_node="Talker", name="talker_input_embeds")
                edge.tensor_info = persist_signals.get("talker_input_embeds", [])
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[edge],
                    unpersist_tensors=list(edge.tensor_info),
                    step_metadata=self._talker_step_metadata(partition_metadata),
                )
            if partition_metadata.graph_walk == "talker_decode":
                return ForwardPassArgs(
                    full_metadata=partition_metadata,
                    inputs=[],
                    unpersist_tensors=[],
                    request_done=True,
                )
            raise ValueError(
                "Talker entered an unexpected graph walk: "
                f"{partition_metadata.graph_walk!r}"
            )

        if partition_name == "Codec":
            inputs = []
            if partition_metadata.graph_walk == "codec_chunk_clone":
                # The reference frame count is an API tensor; every codec
                # invocation of a clone request re-reads it (cheap, one int).
                inputs = self._codec_ref_frames_edges(persist_signals)
            else:
                partition_metadata.graph_walk = "codec_chunk"
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=inputs,
                unpersist_tensors=[],
                step_metadata={
                    "codec_chunk_frames": self.config.codec.chunk_frames,
                    "codec_left_context_frames": (
                        self.config.codec.left_context_frames
                    ),
                    "codec_chunk_schedule": list(self.config.codec.chunk_schedule),
                },
            )
        raise ValueError(f"Unknown Qwen3-TTS partition: {partition_name!r}")

    # -----------------------------------------------------------------------
    # Sampling and output encoding
    # -----------------------------------------------------------------------

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        """Per-request sampling: Talker head (group 0) + CodePredictor (1-15).

        The CodePredictor's ``subtalker_*`` knobs drive its own sampler
        resource; ``*_dosample=False`` maps to temperature 0 (greedy)."""
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        generation = self.config.generation
        do_sample = model_kwargs.get("do_sample", generation.do_sample)
        temperature = model_kwargs.get(
            "temperature",
            model_kwargs.get("talker_temperature", generation.temperature),
        )
        if not do_sample:
            temperature = 0.0
        sub_do_sample = model_kwargs.get(
            "subtalker_dosample", generation.subtalker_dosample
        )
        return {
            TALKER_SAMPLER: SamplingReqConfig(
                temperature=temperature,
                top_k=model_kwargs.get("top_k", generation.top_k),
                top_p=model_kwargs.get("top_p", generation.top_p),
                repetition_penalty=model_kwargs.get(
                    "repetition_penalty", generation.repetition_penalty
                ),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            ),
            CODE_PRED_SAMPLER: SamplingReqConfig(
                temperature=(
                    model_kwargs.get(
                        "subtalker_temperature", generation.subtalker_temperature
                    )
                    if sub_do_sample
                    else 0.0
                ),
                top_k=model_kwargs.get(
                    "subtalker_top_k", generation.subtalker_top_k
                ),
                top_p=model_kwargs.get(
                    "subtalker_top_p", generation.subtalker_top_p
                ),
            ),
        }

    def get_max_output_tokens(self, **model_kwargs: Any) -> int:
        return model_kwargs.get(
            "max_output_tokens",
            model_kwargs.get("max_new_tokens", self.config.generation.max_new_tokens),
        )

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.codec.output_sample_rate

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        """Encode emitted waveform tensors as raw little-endian PCM16 bytes."""
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported Qwen3-TTS output modality: {modality!r}")
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
        """Build only the node assigned to this worker and cache the wrapper."""
        del sp_group
        if node_name not in ("Talker", "Codec", "RefEncoder"):
            raise ValueError(f"Unknown Qwen3-TTS node: {node_name!r}")
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]

        self._ensure_full_snapshot()
        if node_name == "Talker":
            submodule = self._create_talker_submodule(
                device=device,
                tp_group=tp_group,
                autocast_dtype=autocast_dtype,
            )
        elif node_name == "RefEncoder":
            submodule = self._create_ref_encoder_submodule(
                device=device, autocast_dtype=autocast_dtype
            )
        else:
            submodule = self._create_codec_submodule(device=device)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_talker_submodule(
        self,
        device: str,
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        ) -> NodeSubmodule:
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_tts.components.talker import (
            Qwen3TTSCodePredictor,
            Qwen3TTSTalkerModel,
        )
        from mstar.model.qwen3_tts.submodules import TalkerSubmodule

        # Construct on meta first so worker startup never holds both a random
        # initialization and checkpoint tensors in device memory.
        with torch.device("meta"):
            talker = Qwen3TTSTalkerModel(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            talker = talker.to(autocast_dtype)
        talker.to_empty(device=device)

        # The top-level checkpoint interleaves Talker and CodePredictor under
        # ``talker.*``. Stream only the non-CodePredictor keys into this model.
        def talker_weights():
            for name, tensor in iter_safetensors_shards(
                self.local_dir, device=device, prefix="talker."
            ):
                if not name.startswith("talker.code_predictor."):
                    yield name.removeprefix("talker."), tensor

        loaded = load_hf_weights(
            talker,
            talker_weights(),
            stacked_params=LLAMA_STACKED_PARAMS,
        )
        cp_prefix = "talker.code_predictor."
        talker_keys = {
            key for key in _checkpoint_keys(self.local_dir, "talker.")
            if not key.startswith("code_predictor.")
        }
        _verify_checkpoint_coverage(talker, loaded, talker_keys, "Qwen3-TTS Talker")
        talker.eval()

        # CodePredictor is small and depth-wise. It is loaded separately from
        # ``talker.code_predictor.*`` and remains replicated under Talker TP.
        with torch.device("meta"):
            code_predictor = Qwen3TTSCodePredictor(self.config)
        if autocast_dtype is not None:
            code_predictor = code_predictor.to(autocast_dtype)
        code_predictor.to_empty(device=device)
        cp_weights = (
            (name.removeprefix(cp_prefix), tensor)
            for name, tensor in iter_safetensors_shards(
                self.local_dir, device=device, prefix=cp_prefix
            )
        )
        loaded = load_hf_weights(
            code_predictor,
            cp_weights,
            stacked_params=LLAMA_STACKED_PARAMS,
        )
        _verify_checkpoint_coverage(
            code_predictor,
            loaded,
            _checkpoint_keys(self.local_dir, cp_prefix),
            "Qwen3-TTS CodePredictor",
        )
        # The captured depth loop indexes all residual LM heads as one tensor;
        # consolidate after the individual checkpoint heads are loaded.
        code_predictor.consolidate_stacked_weights()
        code_predictor.eval()
        return TalkerSubmodule(talker, code_predictor, self.config)

    def _create_codec_submodule(self, device: str) -> NodeSubmodule:
        """Build the official speech-tokenizer decoder from its sub-checkpoint."""
        try:
            decoder_config_cls, decoder_cls, _ = _load_qwen3_tts_codec_classes()
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS Codec requires the 'qwen-tts' package; install "
                "M* with the qwen3_tts optional dependency"
            ) from exc

        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_tts.submodules import CodecSubmodule

        # Reuse the official decoder implementation, but keep graph scheduling,
        # chunk padding, overlap trimming, and output transport in M*. The
        # 114M-parameter module is built on the CPU rather than on ``meta``:
        # its rotary tables are non-persistent buffers that ``to_empty`` would
        # leave uninitialized (no checkpoint tensor refills them).
        decoder_config = decoder_config_cls(**self.config.codec.decoder_kwargs())
        decoder = decoder_cls(decoder_config).to(device=device)

        codec_dir = Path(self.local_dir) / "speech_tokenizer"
        prefix = "decoder."

        def weights():
            for name, tensor in iter_safetensors_shards(codec_dir, device=device, prefix=prefix):
                yield name.removeprefix(prefix), tensor

        loaded = load_hf_weights(decoder, weights()) | _load_buffers(decoder, weights())
        _verify_checkpoint_coverage(
            decoder, loaded, _checkpoint_keys(codec_dir, prefix), "Qwen3-TTS Codec"
        )
        decoder.eval()
        return CodecSubmodule(decoder, self.config)

    def _create_ref_encoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        """Speaker encoder (main checkpoint) + codec encoder (speech tokenizer) for Base."""
        if not self.config.supports_reference_audio or self.config.speaker_encoder is None:
            raise ValueError(
                f"Qwen3-TTS {self.config.tts_model_type} has no reference-audio encoder"
            )
        from transformers import MimiConfig

        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_tts.components.speaker_encoder import (
            Qwen3TTSMelFrontEnd,
            Qwen3TTSSpeakerEncoder,
        )
        from mstar.model.qwen3_tts.submodules import RefEncoderSubmodule

        # The x-vector is computed in the Talker's dtype, as the reference does.
        speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder)
        if autocast_dtype is not None:
            speaker_encoder = speaker_encoder.to(autocast_dtype)
        speaker_encoder = speaker_encoder.to(device=device)
        prefix = "speaker_encoder."

        def speaker_weights():
            for name, tensor in iter_safetensors_shards(self.local_dir, device=device, prefix=prefix):
                yield name.removeprefix(prefix), tensor

        loaded = load_hf_weights(speaker_encoder, speaker_weights())
        _verify_checkpoint_coverage(
            speaker_encoder, loaded, _checkpoint_keys(self.local_dir, prefix), "Qwen3-TTS speaker encoder"
        )
        speaker_encoder.eval()

        # Mimi encoder of the speech tokenizer: reference clip -> codec frames.
        # Float32 like the decoder; built on the CPU for its non-persistent
        # buffers (rotary tables, convolution geometry). Its quantizer keeps
        # the codebooks in persistent buffers, which the checkpoint refills.
        _, _, encoder_cls = _load_qwen3_tts_codec_classes()
        codec_encoder = encoder_cls(MimiConfig(**self.config.codec.encoder_config)).to(device=device)
        codec_dir = Path(self.local_dir) / "speech_tokenizer"
        encoder_prefix = "encoder."

        def encoder_weights():
            for name, tensor in iter_safetensors_shards(codec_dir, device=device, prefix=encoder_prefix):
                yield name.removeprefix(encoder_prefix), tensor

        loaded = load_hf_weights(codec_encoder, encoder_weights()) | _load_buffers(codec_encoder, encoder_weights())
        _verify_checkpoint_coverage(
            codec_encoder, loaded, _checkpoint_keys(codec_dir, encoder_prefix), "Qwen3-TTS codec encoder"
        )
        codec_encoder.eval()
        return RefEncoderSubmodule(
            speaker_encoder,
            Qwen3TTSMelFrontEnd(self.config.speaker_encoder).to(device=device),
            codec_encoder,
            self.config,
        )

    @staticmethod
    def _talker_step_metadata(
        metadata: CurrentForwardConductorMetadata,
    ) -> dict[str, Any]:
        """Copy conductor-owned generation controls into worker step metadata."""
        return {
            "is_prefill": metadata.is_prefill,
            "talker_max_tokens": metadata.kwargs["talker_max_tokens"],
        }
