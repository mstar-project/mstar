"""
ParakeetModel: NVIDIA's FastConformer transducer speech recognizers —
Parakeet-TDT-0.6B-v3 (25 European languages, offline) and Nemotron-3.5-ASR
(cache-aware streaming, run here over whole clips).

A FastConformer encoder hears the clip once, then a transducer reads it frame
by frame: a small LSTM over the tokens emitted so far and the frame under the
cursor score the next token or blank. Parakeet-TDT also predicts how many
frames to skip; Nemotron conditions every frame on a language prompt.

Architecture (2 nodes, single partition):
    audio_encoder — log-mel + FastConformer, batched
    transducer    — greedy decoding of the whole clip, batched

Graph walk:
    transcribe — audio_encoder -> transducer -> transcript tokens
"""

import json
import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.components.audio_features import SAMPLE_RATE, clip_samples, load_speech_file
from mstar.model.parakeet.config import ParakeetModelConfig, check_detected_language
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import SentencePieceDetokenizer, check_loaded
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FILES = ["*.json", "*.safetensors"]
WALK = "transcribe"


class ParakeetModel(Model):
    """FastConformer encoder + greedy RNN-T / TDT transducer."""

    # full attention, whose position scores grow with the clip's square: the route
    # windows at 600 s, plus slack for containers that decode longer than their header said
    MAX_AUDIO_SECONDS = 610.0
    # a few tiny ops per upload; see "Preprocessing threads" in docs/adding_models.rst
    PREPROCESS_TORCH_THREADS = 1

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None,
        lookahead: int | None = None, compile_step: bool = True, **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.compile_step = compile_step
        self.local_dir = resolve_snapshot_dir(model_path_hf, cache_dir, allow_patterns=CHECKPOINT_FILES)
        self.config = ParakeetModelConfig.from_pretrained(self.local_dir, lookahead=lookahead)
        tokenizer = json.loads((self.local_dir / "tokenizer.json").read_text())
        self._detokenizer = SentencePieceDetokenizer(
            tokenizer, extra_skip_ids={self.config.transducer.blank_token_id},
        )
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def checkpoint_path(self) -> str | None:
        return str(self.local_dir)

    def get_node_resources(self) -> list[NodeResourceSpec]:
        # both nodes are stateless: no cache, no sampler (decoding is greedy)
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        transducer_inputs = ["encoder_states"] + (["prompt_ids"] if self.config.has_prompt else [])
        return {
            WALK: Sequential([
                GraphNode(
                    name="audio_encoder",
                    input_names=["audio"],
                    outputs=[GraphEdge(next_node="transducer", name="encoder_states")],
                ),
                GraphNode(
                    name="transducer",
                    input_names=transducer_inputs,
                    outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="tokens", output_modality="text")],
                ),
            ]),
        }

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=WALK,
            is_prefill=True,
        )
        audio = GraphEdge(next_node="audio_encoder", name="audio")
        audio.tensor_info = input_signals.get("audio", [])
        inputs = [audio]
        if self.config.has_prompt:
            prompt = GraphEdge(next_node="transducer", name="prompt_ids")
            prompt.tensor_info = input_signals.get("prompt_ids", [])
            inputs.append(prompt)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        # one walk transcribes the whole clip
        return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[], request_done=True)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """The clip's samples, plus Nemotron's language prompt from
        ``language`` (a locale such as ``en-US``, a bare code, or ``auto``,
        the default). Parakeet-TDT detects the language itself and refuses
        only a ``language`` it doesn't know."""
        out = {"audio": [clip_samples(tensors, self.MAX_AUDIO_SECONDS, "this model")]}
        if self.config.has_prompt:
            out["prompt_ids"] = [torch.tensor([self.config.prompt_id(kwargs.get("language"))], dtype=torch.long)]
        else:
            check_detected_language(kwargs.get("language"))
        return out

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        audio = load_speech_file(filepath, SAMPLE_RATE, self.MAX_AUDIO_SECONDS)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=SAMPLE_RATE, num_channels=1))

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for {type(self).__name__}: {modality!r}")

    # -------------------------------------------------------------------
    # submodules
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(node_name, device, autocast_dtype)
        return self._submodule_cache[node_name]

    def _weights(self, device: str, prefix: str):
        from mstar.model.loader.iterators import iter_safetensors_shards

        return iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)

    def _create_submodule(self, node_name: str, device: str, autocast_dtype: torch.dtype | None):
        if node_name == "audio_encoder":
            from mstar.model.components.fastconformer import FastConformerEncoder
            from mstar.model.components.fastconformer_submodule import FastConformerEncoderSubmodule

            with torch.device("meta"):
                encoder = FastConformerEncoder(self.config.encoder)
            if autocast_dtype is not None:
                encoder = encoder.to(autocast_dtype)
            encoder = encoder.to_empty(device=device)
            loaded = encoder.load_hf_weights(
                (k.removeprefix("encoder."), v) for k, v in self._weights(device, "encoder.")
            )
            check_loaded(encoder, loaded, "encoder")
            return FastConformerEncoderSubmodule(encoder.eval(), normalize_features=self.config.normalize_features)
        if node_name == "transducer":
            from mstar.model.components.transducer import TransducerHead
            from mstar.model.parakeet.submodules import TransducerSubmodule

            # fp32: the submodule opts out of the engine's cast
            head = TransducerHead(self.config.transducer).to(device)
            state = {
                k: v for prefix in ("encoder_projector.", "decoder.", "joint.", "prompt_projector.")
                for k, v in self._weights(device, prefix)
            }
            missing, unexpected = head.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise KeyError(f"transducer weights: missing {missing[:4]}, unexpected {unexpected[:4]}")
            return TransducerSubmodule(head.eval(), compile_step=self.compile_step)
        return None
