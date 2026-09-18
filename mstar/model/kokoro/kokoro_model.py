"""Kokoro-82M: text -> phonemes -> 24 kHz speech, batched across requests.

Graph (one node, one walk)::

    process_prompt (API data worker, CPU)
        text --G2P (misaki / espeak-ng)--> sentence chunks --> phoneme ids,
        one style row per chunk from the voice pack (or blend), speed
             |
             v   phoneme_ids [n, T], phoneme_lens [n], ref_style [n, 256], speed [1]
    ┌─ chunk_loop ──────────────────────────────────────────────────────┐
    │  kokoro  (GPU) : PL-BERT -> prosody (durations, F0, energy)        │
    │                  -> text encoder -> iSTFTNet decoder               │
    │                  one chunk per iteration, batched over requests    │
    │        └──> audio_chunk (PCM16) ──> EMIT_TO_CLIENT (per iteration) │
    └───────────────────────────────────────────────────────────────────┘

Walks: ``synth`` only. The loop ends when a request's chunks are exhausted
(``check_stop``); streaming is one message per sentence chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.kokoro.config import (
    AUDIO_CHUNK,
    BOUNDARY_TOKEN_ID,
    CHUNK_LOOP,
    PHONEME_IDS,
    PHONEME_LENS,
    REF_STYLE,
    SPEED,
    SYNTH_NODE,
    SYNTH_WALK,
    KokoroModelConfig,
)
from mstar.model.kokoro.g2p import G2PFrontend, normalize_lang_code
from mstar.model.kokoro.voices import VoiceRegistry
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

# What the API-side processes need: the config and the voice packs (28 MB).
METADATA_PATTERNS = ["config.json", "voices/*.pt"]
# Config fields a deployment may set through the YAML ``model_kwargs``.
SERVING_OVERRIDES = {
    "default_voice", "default_speed", "first_chunk_target_phonemes", "chunk_target_phonemes", "max_chunks",
    "text_buckets", "frame_buckets", "capture_batch_sizes", "max_batch_frames", "frame_grouping", "decoder_dtype",
    "compile_decoder",
}


def _resolve_snapshot(repo_id: str, cache_dir: str | None, allow_patterns: list[str] | None) -> str:
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, allow_patterns=allow_patterns)


class KokoroModel(Model):
    """Model contract for Kokoro-82M (``hexgrad/Kokoro-82M``)."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        lang_code: str | None = None,
        espeak_fallback: bool = True,
        **config_overrides: Any,
    ) -> None:
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = _resolve_snapshot(model_path_hf, cache_dir, METADATA_PATTERNS)
        self.config = KokoroModelConfig.from_pretrained(self.local_dir)
        # Serving knobs (chunking targets, CUDA-graph buckets, batch caps) come
        # from the deployment YAML's ``model_kwargs``; architecture fields stay
        # with the checkpoint.
        for name, value in config_overrides.items():
            if name not in SERVING_OVERRIDES:
                raise ValueError(f"Unknown Kokoro option {name!r}; deployment options: {sorted(SERVING_OVERRIDES)}")
            setattr(self.config, name, value)
        self.default_lang = normalize_lang_code(lang_code) if lang_code else None
        self.voices = VoiceRegistry(
            Path(self.local_dir) / self.config.voices_dir, self.config.style_pack_rows, self.config.style_dim
        )
        self.g2p = G2PFrontend(
            self.config.chunk_target_phonemes,
            self.config.max_phonemes,
            espeak_fallback,
            first_chunk_target=self.config.first_chunk_target_phonemes,
        )
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -- graph -------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Kokoro is not autoregressive and keeps no state between steps."""
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        synth = GraphNode(
            name=SYNTH_NODE,
            input_names=[PHONEME_IDS, PHONEME_LENS, REF_STYLE, SPEED],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=AUDIO_CHUNK, output_modality="audio")],
            # The loop length is per request (its chunk count); speculating
            # an extra iteration would synthesize a chunk that does not exist.
            enable_async_scheduling=False,
        )
        return {SYNTH_WALK: Loop(name=CHUNK_LOOP, section=synth, max_iters=self.config.max_chunks, outputs=[])}

    # -- request preprocessing (API data worker) -----------------------------

    def warmup_preprocess(self) -> None:
        """Build the default language's G2P (spaCy tagger, lexicon) before the
        first request; it takes several seconds."""
        self.g2p.backend(self.default_lang or self.voices.language_of(self.config.default_voice))

    def tokenize(self, phonemes: str) -> list[int]:
        """Phoneme string -> ``[<bos>, ids..., <eos>]``; characters outside the
        vocabulary are dropped, as in the reference."""
        ids = [self.config.vocab[c] for c in phonemes if c in self.config.vocab]
        return [BOUNDARY_TOKEN_ID, *ids, BOUNDARY_TOKEN_ID]

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        del tensors
        if "audio" not in output_modalities:
            raise ValueError("Kokoro produces audio; request output_modalities=['audio']")
        if any(m != "text" for m in input_modalities):
            raise ValueError("Kokoro takes text input only")

        voice = str(kwargs.get("voice") or self.config.default_voice)
        self.voices.resolve(voice)  # reject unknown voices before anything language-specific runs
        speed = float(kwargs.get("speed") or self.config.default_speed)
        if not self.config.min_speed <= speed <= self.config.max_speed:
            raise ValueError(f"speed must be within [{self.config.min_speed}, {self.config.max_speed}], got {speed}")
        lang = kwargs.get("lang_code") or kwargs.get("language") or self.default_lang or self.voices.language_of(voice)

        phonemes = kwargs.get("phonemes")
        if phonemes:
            chunks = self.g2p.chunk_phonemes(str(phonemes))
        else:
            if not prompt or not prompt.strip():
                raise ValueError("Kokoro requires a non-empty text prompt")
            chunks = self.g2p.chunk(prompt, lang)
        if not chunks:
            raise ValueError("Input contains no speakable text")
        if len(chunks) > self.config.max_chunks:
            raise ValueError(f"Input splits into {len(chunks)} chunks; the limit is {self.config.max_chunks}")

        ids = [torch.tensor(self.tokenize(c.phonemes), dtype=torch.long) for c in chunks]
        lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
        padded = torch.full((len(ids), int(lengths.max())), BOUNDARY_TOKEN_ID, dtype=torch.long)
        for row, x in enumerate(ids):
            padded[row, : len(x)] = x
        style = torch.stack([self.voices.style(voice, len(c.phonemes)) for c in chunks])
        return {
            PHONEME_IDS: [padded],
            PHONEME_LENS: [lengths],
            REF_STYLE: [style],
            SPEED: [torch.tensor([speed], dtype=torch.float32)],
        }

    # -- conductor state machine -------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        del partition_name, model_kwargs
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=SYNTH_WALK,
            is_prefill=True,
        )
        inputs = []
        for name in (PHONEME_IDS, PHONEME_LENS, REF_STYLE, SPEED):
            edge = GraphEdge(next_node=SYNTH_NODE, name=name)
            edge.tensor_info = input_signals.get(name, [])
            inputs.append(edge)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([edge.tensor_info for edge in inputs], start=[]),
            request_done="audio" not in output_modalities,
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """The single walk synthesizes every chunk; when it returns, the request is done."""
        del partition_name, persist_signals, incoming_connections
        return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[], request_done=True)

    # -- output ------------------------------------------------------------

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.sample_rate

    def get_voices(self) -> list[str]:
        """The bundled voices; blends of them are accepted too (see ``VoiceRegistry``)."""
        return self.voices.names

    def get_default_voice(self) -> str:
        return self.config.default_voice

    def get_autocast_dtype(self):
        return torch.float32

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Emitted PCM16 tensor -> little-endian bytes."""
        del request_kwargs
        if modality != "audio":
            raise ValueError(f"Unsupported Kokoro output modality: {modality!r}")
        if output.numel() == 0:
            return b""
        pcm = output.detach().cpu()
        if pcm.is_floating_point():
            pcm = (pcm.clamp(-1, 1) * 32767).to(torch.int16)
        return pcm.contiguous().numpy().tobytes()

    # -- worker-side construction ------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        del tp_group, autocast_dtype, sp_group
        if node_name != SYNTH_NODE:
            raise ValueError(f"Unknown Kokoro node: {node_name!r}")
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_synth_submodule(device)
        return self._submodule_cache[node_name]

    def _create_synth_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.kokoro.components import KokoroTTS
        from mstar.model.kokoro.submodules import KokoroSynthSubmodule
        from mstar.model.kokoro.weight_loader import load_kokoro_weights

        local_dir = _resolve_snapshot(self.model_path_hf, self.cache_dir, None)
        # 82M parameters: build on the host, load, move. No meta-device dance needed.
        model = KokoroTTS(self.config)
        load_kokoro_weights(model, Path(local_dir) / self.config.weights_file)
        model = model.to(device).eval()
        logger.info("Loaded Kokoro (%d voices) on %s", len(self.voices.names), device)
        return KokoroSynthSubmodule(model, self.config)
