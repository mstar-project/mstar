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
    transducer    — greedy decoding, a chunk of steps per loop iteration,
                    batched; each request's decoder state in a recurrent
                    state slot

Graph walk:
    transcribe — audio_encoder -> a transducer loop over the encoder output:
                 each iteration's tokens to the client, until the cursor
                 passes the clip's last frame
"""

import json
import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
from mstar.engine.resources import NodeResourceSpec
from mstar.engine.resources.recurrent import RecurrentStateConfig, RecurrentStateSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, ProcessPromptOutput, TensorAndMetadata
from mstar.model.components.audio_features import HOP_LENGTH, SAMPLE_RATE, clip_samples, load_speech_file
from mstar.model.parakeet.config import (
    DECODE_LOOP,
    NUM_FRAMES,
    PROMPT_ID,
    TRANSDUCER_STATE,
    WALK,
    ParakeetModelConfig,
    check_detected_language,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import SentencePieceDetokenizer, check_loaded
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FILES = ["*.json", "*.safetensors"]


class ParakeetModel(Model):
    """FastConformer encoder + greedy RNN-T / TDT transducer."""

    # one clip in, a transcript out; the transcription route's text part is
    # unused (Nemotron's prompt is the language kwarg)
    SUPPORTED_INPUT_MODALITIES = frozenset({"audio", "text"})
    SUPPORTED_OUTPUT_MODALITIES = frozenset({"text"})

    # full attention, whose position scores grow with the clip's square: the route
    # windows at 600 s, plus slack for containers that decode longer than their header said
    MAX_AUDIO_SECONDS = 610.0
    # requests the default state pool holds; the YAMLs cap max_concurrent_requests here
    MAX_CONCURRENT_REQUESTS = 128
    # a few tiny ops per upload; see "Preprocessing threads" in docs/adding_models.rst
    PREPROCESS_TORCH_THREADS = 1
    INTERNAL_KWARGS = (NUM_FRAMES, PROMPT_ID)

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

    def _num_frames(self, num_samples: int) -> int:
        """Encoder frames for a clip, as the encoder node counts them."""
        return int(self.config.encoder.output_lengths(num_samples // HOP_LENGTH))

    def get_node_resources(self) -> list[NodeResourceSpec]:
        # no cache and no sampler (decoding is greedy): the decoder state is all
        return [
            RecurrentStateSpec(
                resource_key=TRANSDUCER_STATE, nodes={"transducer"},
                config=RecurrentStateConfig(
                    # the LSTM's layers are in h's and c's shape
                    num_layers=1,
                    blocks=self.config.transducer.state_blocks(),
                    # + 1 for the pool's sink slot
                    max_slots=self.MAX_CONCURRENT_REQUESTS + 1,
                ),
            ),
        ]

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a ``max_concurrent_requests`` the state pool can't hold. A
        request holds its slot from its first decode iteration until it is torn down;
        past the pool, a decode step's admit waits for a slot while the
        request's frames sit on the GPU, where the conductor would have
        queued the request."""
        slots = ((config.get("resources") or {}).get(TRANSDUCER_STATE) or {}).get(
            "max_slots", self.MAX_CONCURRENT_REQUESTS + 1,
        )
        # the pool's own check runs before YAML overrides land, so it can't see this
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 2:
            raise ValueError(
                f"{config_path}: resources.{TRANSDUCER_STATE}.max_slots must be an int >= 2 "
                f"(slot 0 is the sink); got {slots!r}"
            )
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning(
                "%s sets no max_concurrent_requests: past the %d requests resources.%s holds, "
                "decode steps wait for a slot", config_path, slots - 1, TRANSDUCER_STATE,
            )
            return
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f"{config_path}: max_concurrent_requests must be a positive int; got {cap!r}")
        if cap > slots - 1:
            raise ValueError(
                f"{config_path}: {cap} concurrent requests (max_concurrent_requests) need "
                f"{cap + 1} slots of resources.{TRANSDUCER_STATE} (one each and the sink), which has "
                f"{slots}; raise max_slots or lower max_concurrent_requests"
            )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        decode = Loop(
            name=DECODE_LOOP,
            # The loop re-injects the encoder output every iteration. No loop-back
            # edge, so the worker speculates a request's first iteration behind
            # the encoder's step but never a next one: most clips end in their
            # first, and a speculated iteration past the end is a whole chunk
            # of steps for nothing.
            section=GraphNode(
                name="transducer",
                input_names=["encoder_states"],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="tokens", output_modality="text")],
            ),
            # a ceiling only: the stop check ends each request's loop
            max_iters=self.config.transducer.decode_iterations(self._num_frames(
                int(self.MAX_AUDIO_SECONDS * SAMPLE_RATE),
            )),
            outputs=[],
        )
        return {WALK: Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio"],
                outputs=[GraphEdge(next_node="transducer", name="encoder_states")],
            ),
            decode,
        ])}

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # a ceiling the runtime enforces: the most iterations a clip can take
        num_frames = model_kwargs.get(NUM_FRAMES)
        return {} if num_frames is None else {DECODE_LOOP: self.config.transducer.decode_iterations(num_frames)}

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

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
        model_kwargs = model_kwargs or {}
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[audio],
            unpersist_tensors=list(audio.tensor_info),
            # host values the decode steps read: the capture's bucket, Nemotron's prompt
            step_metadata={key: model_kwargs.get(key) for key in self.INTERNAL_KWARGS},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        # the one walk returned: the decode loop's cursor passed the last frame
        return ForwardPassArgs(full_metadata=partition_metadata, inputs=[], unpersist_tensors=[], request_done=True)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """The clip's samples. Metadata for the decode steps rides along: the
        clip's encoder frames, which pick the decode capture before the frames
        exist, and Nemotron's language prompt from ``language`` (a locale such
        as ``en-US``, a bare code, or ``auto``, the default). Parakeet-TDT
        detects the language itself and refuses only a ``language`` it
        doesn't know."""
        samples = clip_samples(tensors, self.MAX_AUDIO_SECONDS, "this model")
        metadata = {NUM_FRAMES: self._num_frames(samples.shape[0])}
        if self.config.has_prompt:
            metadata[PROMPT_ID] = self.config.prompt_id(kwargs.get("language"))
        else:
            check_detected_language(kwargs.get("language"))
        return ProcessPromptOutput({"audio": [samples]}, metadata)

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
            return TransducerSubmodule(
                head.eval(), compile_step=self.compile_step, encoder_dtype=autocast_dtype or torch.float32,
            )
        return None
