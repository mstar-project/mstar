"""Omnilingual ASR CTC (Meta, Apache-2.0): speech-to-text in 1,600+ languages.

Graph (one node, one walk, no state)::

    load_audio + process_prompt (API data worker, CPU)
        decode at the file's rate -> resample to 16 kHz -> average channels
        -> reject > 40 s -> normalize the utterance (zero mean, unit variance)
             |
             v   waveform [S] float32
    ┌─ transcribe ────────────────────────────────────────────────────┐
    │  encoder (GPU): wav2vec 2.0 encoder -> CTC head -> argmax over   │
    │                 valid frames (one id per frame)                  │
    │       └──> token_ids ──> EMIT_TO_CLIENT                          │
    └─────────────────────────────────────────────────────────────────┘
    postprocess (API server): collapse repeats -> text, blank and
        specials dropped

The model takes no language code and no prompt: the CTC head covers every
language at once (the reference ignores ``lang`` for CTC models too). Sizes
300M, 1B, 3B and 7B share this code; ``size`` and ``weights_version`` pick
the checkpoint (see ``config.py``). Weights are loaded from Meta's own
``.pt`` files, without fairseq2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.omnilingual_asr.audio import load_waveform, preprocess
from mstar.model.omnilingual_asr.config import (
    ENCODER_NODE,
    MAX_AUDIO_SECONDS,
    SAMPLE_RATE,
    SIZES,
    TOKEN_IDS,
    TRANSCRIBE_WALK,
    VERSIONS,
    WAVEFORM,
    ctc_config,
    parse_dtype,
)
from mstar.model.omnilingual_asr.tokenizer import CharTokenizer, ctc_collapse
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


class OmnilingualASRModel(Model):
    """Omnilingual ASR CTC served as one stateless, batched encoder node.

    Deployment options (YAML ``model_kwargs``):

    * ``size``: ``300m`` | ``1b`` | ``3b`` | ``7b`` (set by the registry key).
    * ``weights_version``: ``v2`` (default; Meta's CDN, pinned sha256) or
      ``v1`` (the ``facebook/omniASR-CTC-*`` Hub repos, pinned revision).
    * ``checkpoint_path`` / ``tokenizer_path``: local files instead of a
      download (an air-gapped host). They must match ``size`` and
      ``weights_version``; the shapes are checked at load.
    * ``dtype``: ``bfloat16`` (default, the reference's inference dtype) or
      ``float32`` (also turns TF32 off for the node's matmuls).
    * ``max_batch_size`` (16): requests per encoder step. ``max_batch_seconds``
      (320): padded audio per forward; a step over it runs as several forwards.
    """

    def __init__(
        self,
        model_path_hf: str | None = None,
        cache_dir: str | None = None,
        size: str = "300m",
        weights_version: str = "v2",
        checkpoint_path: str | None = None,
        tokenizer_path: str | None = None,
        dtype: str = "bfloat16",
        max_batch_size: int = 16,
        max_batch_seconds: float = 320.0,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(
                f"Unknown Omnilingual ASR option(s) {sorted(kwargs)}; deployment options: size, weights_version, "
                "checkpoint_path, tokenizer_path, dtype, max_batch_size, max_batch_seconds"
            )
        if size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)}; got {size!r}")
        if weights_version not in VERSIONS:
            raise ValueError(f"weights_version must be one of {list(VERSIONS)}; got {weights_version!r}")
        if max_batch_seconds < MAX_AUDIO_SECONDS:
            raise ValueError(f"max_batch_seconds must be at least {MAX_AUDIO_SECONDS:g} (one full-length clip)")
        del model_path_hf  # the checkpoint is picked by size and weights_version
        self.cache_dir = cache_dir
        self.size = size
        self.weights_version = weights_version
        # not ``checkpoint_path``: that name is Model's prefix-cache hook
        self._local_checkpoint = checkpoint_path
        self.config = ctc_config(size, weights_version)
        self.dtype = parse_dtype(dtype)
        self.max_batch_size = int(max_batch_size)
        self.max_batch_seconds = float(max_batch_seconds)

        from mstar.model.omnilingual_asr.weight_loader import resolve_tokenizer

        self.tokenizer = CharTokenizer.from_file(resolve_tokenizer(weights_version, cache_dir, tokenizer_path))
        if self.tokenizer.vocab_size != self.config.vocab_size:
            raise ValueError(
                f"tokenizer has {self.tokenizer.vocab_size} pieces but the {weights_version} {size} model "
                f"emits {self.config.vocab_size}; the tokenizer does not belong to these weights"
            )
        self._submodule_cache: dict[str, NodeSubmodule] = {}

    # -- graph -------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Not autoregressive: no KV cache, no sampler."""
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        return {
            TRANSCRIBE_WALK: GraphNode(
                name=ENCODER_NODE,
                input_names=[WAVEFORM],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name=TOKEN_IDS, output_modality="text")],
            ),
        }

    def get_autocast_dtype(self) -> torch.dtype:
        return self.dtype

    # -- request preprocessing (API data worker) -----------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """The upload as a 16 kHz mono waveform, resampled and downmixed the way
        the reference does it (the default loader resamples with FFmpeg)."""
        del device
        waveform = load_waveform(filepath)
        return TensorAndMetadata(data=waveform, metadata=dict(sample_rate=SAMPLE_RATE, num_channels=1))

    def request_kwargs(self) -> frozenset[str]:
        # argmax CTC: the transcription adapter's language, prompt and temperature are not read
        return frozenset()

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        """Length-check and normalize the waveform. ``prompt`` and the
        transcription kwargs (``language``, ``initial_prompt``, ``temperature``)
        have no effect on a CTC model and are ignored."""
        del prompt, input_modalities, kwargs
        if "text" not in output_modalities:
            raise ValueError("Omnilingual ASR produces text; request output_modalities=['text']")
        audio = (tensors or {}).get("audio_inputs", [])
        if len(audio) != 1:
            raise ValueError(f"Omnilingual ASR expects exactly one audio input per request; got {len(audio)}")
        waveform = audio[0].detach().to("cpu", torch.float32).reshape(-1)
        return {WAVEFORM: [preprocess(waveform)]}

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
            graph_walk=TRANSCRIBE_WALK,
            is_prefill=True,
        )
        edge = GraphEdge(next_node=ENCODER_NODE, name=WAVEFORM)
        edge.tensor_info = input_signals.get(WAVEFORM, [])
        return ForwardPassArgs(full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info))

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """One walk transcribes the clip; when it returns, the request is done."""
        del partition_name, persist_signals, incoming_connections
        return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[], request_done=True)

    # -- output ------------------------------------------------------------

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """Per-frame CTC ids -> UTF-8 text: repeats collapse, and the blank and
        specials render as nothing."""
        del request_kwargs
        if modality != "text":
            raise ValueError(f"Unsupported Omnilingual ASR output modality: {modality!r}")
        return self.tokenizer.decode(ctc_collapse(output.reshape(-1)).tolist()).encode("utf-8")

    # -- worker-side construction ------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule:
        del tp_group, autocast_dtype, sp_group
        if node_name != ENCODER_NODE:
            raise ValueError(f"Unknown Omnilingual ASR node: {node_name!r}")
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_encoder_submodule(device)
        return self._submodule_cache[node_name]

    def _create_encoder_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.omnilingual_asr.components.wav2vec2 import Wav2Vec2CTC
        from mstar.model.omnilingual_asr.submodules import OmniASRCTCSubmodule
        from mstar.model.omnilingual_asr.weight_loader import (
            convert_state_dict,
            load_into,
            read_checkpoint,
            resolve_checkpoint,
        )

        path = resolve_checkpoint(self.size, self.weights_version, self.cache_dir, self._local_checkpoint)
        with torch.device("meta"):
            model = Wav2Vec2CTC(self.config)
        load_into(model, convert_state_dict(read_checkpoint(path), self.dtype, device))
        model.eval()
        logger.info(
            "Loaded Omnilingual ASR CTC %s (%s, %s) from %s on %s",
            self.size,
            self.weights_version,
            self.dtype,
            Path(path).name,
            device,
        )
        return OmniASRCTCSubmodule(model, self.max_batch_size, self.max_batch_seconds)
