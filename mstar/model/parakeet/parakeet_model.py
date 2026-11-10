"""
ParakeetModel: NVIDIA's FastConformer transducer speech recognizers —
Parakeet-TDT-0.6B-v3 (25 European languages, offline) and Nemotron-3.5-ASR
(cache-aware streaming). Both take whole clips; Nemotron ASR also hears a live
call as it comes, over ``/v1/realtime`` or ``/generate/ws``.

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

With ``live_lookahead`` (a cache-aware checkpoint), two walks more, one partition
each, for a live call:
    encode_stream — stream_encoder, one step per chunk (live_lookahead + 1 frames)
                    of the client's audio
    decode_stream — transducer_stream, one step per chunk of frames, its tokens
                    to the client; both end with the client's stream
A whole clip runs ``transcribe`` and a live call the other two: the partition a
request doesn't use is done from the start, so whole clips pay nothing for it.
"""

import json
import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import DEFAULT_PARTITION, PartitionDefinition
from mstar.engine.resources import NodeResourceSpec
from mstar.engine.resources.recurrent import RecurrentStateConfig, RecurrentStateSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model, ProcessPromptOutput, WalkStep
from mstar.model.components.audio_features import HOP_LENGTH, SAMPLE_RATE, clip_samples
from mstar.model.parakeet.config import (
    AUDIO_IN,
    DECODE_LOOP,
    ENCODER_FRAMES,
    LIVE_DECODE_LOOP,
    LIVE_DECODE_WALK,
    LIVE_ENCODE_LOOP,
    LIVE_ENCODE_WALK,
    NUM_FRAMES,
    PROMPT_ID,
    STREAM_CONV,
    STREAM_FRONT,
    STREAM_KV,
    TRANSDUCER_STATE,
    WALK,
    ParakeetModelConfig,
    check_detected_language,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import SentencePieceDetokenizer, TextOutput
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import (
    CLIENT,
    ClientAudio,
    ClientStreamLimits,
    Connection,
    PartitionTopology,
    StreamingGraphEdge,
)
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FILES = ["*.json", "*.safetensors"]


class ParakeetModel(Model):
    """FastConformer encoder + greedy RNN-T / TDT transducer."""

    # full attention, whose position scores grow with the clip's square: the route
    # windows at 600 s, plus slack for containers that decode longer than their header said
    MAX_AUDIO_SECONDS = 610.0
    # requests the default state pool holds; the YAMLs cap max_concurrent_requests here
    MAX_CONCURRENT_REQUESTS = 128
    # the longest live call: its loops' ceiling and its client stream's bound
    MAX_LIVE_SECONDS = 3600
    # a few tiny ops per upload; see "Preprocessing threads" in docs/adding_models.rst
    PREPROCESS_TORCH_THREADS = 1
    INTERNAL_KWARGS = (NUM_FRAMES, PROMPT_ID)

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None,
        lookahead: int | None = None, compile_step: bool = True, live_lookahead: int | None = None, **kwargs,
    ):
        """``lookahead`` is the clips' (the checkpoint's widest by default);
        ``live_lookahead``, one the checkpoint supports, also serves live calls
        at it (configs/nemotron_asr_live.yaml). Without it the graph is the
        whole-clip one alone: a live path costs every clip's request its extra
        partitions and the deployment its stream state."""
        self.model_path_hf = model_path_hf
        self.compile_step = compile_step
        self.local_dir = resolve_snapshot_dir(model_path_hf, cache_dir, allow_patterns=CHECKPOINT_FILES)
        self.config = ParakeetModelConfig.from_pretrained(
            self.local_dir, lookahead=lookahead, live_lookahead=live_lookahead,
        )
        tokenizer = json.loads((self.local_dir / "tokenizer.json").read_text())
        self._detokenizer = SentencePieceDetokenizer(
            tokenizer, extra_skip_ids={self.config.transducer.blank_token_id},
        )
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        # modules two nodes share, by (name, device)
        self._modules: dict[tuple[str, str], torch.nn.Module] = {}

    @property
    def live(self) -> bool:
        """Whether the deployment also hears live calls."""
        return self.config.live_encoder is not None

    @property
    def chunk_frames(self) -> int:
        """Encoder frames a live step takes: its attention's chunk."""
        return self.config.live_encoder.lookahead + 1

    def checkpoint_path(self) -> str:
        return str(self.local_dir)

    def _num_frames(self, num_samples: int) -> int:
        """Encoder frames for a clip, as the encoder node counts them."""
        return int(self.config.encoder.output_lengths(num_samples // HOP_LENGTH))

    def get_node_resources(self) -> list[NodeResourceSpec]:
        # no cache and no sampler (decoding is greedy): the decoder state is all,
        # and a live stream's encoder state
        return [
            *self._stream_resources(),
            RecurrentStateSpec(
                resource_key=TRANSDUCER_STATE,
                nodes={"transducer", "transducer_stream"} if self.live else {"transducer"},
                config=RecurrentStateConfig(
                    # the LSTM's layers are in h's and c's shape
                    num_layers=1,
                    blocks=self.config.transducer.state_blocks(),
                    # + 1 for the pool's sink slot
                    max_slots=self.MAX_CONCURRENT_REQUESTS + 1,
                ),
            ),
        ]

    def _stream_resources(self) -> list[NodeResourceSpec]:
        if not self.live:
            return []
        from mstar.model.components.fastconformer_stream import stream_node_resources

        return stream_node_resources(
            self.config.live_encoder, HOP_LENGTH, "stream_encoder", self.MAX_CONCURRENT_REQUESTS,
            kv_key=STREAM_KV, conv_key=STREAM_CONV, front_key=STREAM_FRONT,
        )

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
        walks = {WALK: Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio"],
                outputs=[GraphEdge(next_node="transducer", name="encoder_states")],
            ),
            decode,
        ])}
        if self.live:
            steps = self._live_chunks()
            walks[LIVE_ENCODE_WALK] = Loop(
                name=LIVE_ENCODE_LOOP,
                section=GraphNode(
                    name="stream_encoder",
                    input_names=[AUDIO_IN],
                    consumes_stream=True,
                    outputs=[StreamingGraphEdge(next_node="transducer_stream", name=ENCODER_FRAMES,
                                                target_partition="Decoder")],
                ),
                max_iters=steps,
                outputs=[],
            )
            walks[LIVE_DECODE_WALK] = Loop(
                name=LIVE_DECODE_LOOP,
                section=GraphNode(
                    name="transducer_stream",
                    input_names=[ENCODER_FRAMES],
                    consumes_stream=True,
                    outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="tokens", output_modality="text")],
                ),
                max_iters=steps,
                outputs=[],
            )
        return walks

    def _live_chunks(self) -> int:
        """The most chunks of audio a live call can hold, its stream's end included."""
        return self.MAX_LIVE_SECONDS * SAMPLE_RATE // (self._chunk_samples()) + 2

    def _chunk_samples(self) -> int:
        return HOP_LENGTH * self.config.encoder.subsampling_factor * self.chunk_frames

    def get_partitions(self) -> list[PartitionDefinition]:
        if not self.live:
            return super().get_partitions()
        return [
            PartitionDefinition(name=DEFAULT_PARTITION, graph_walks={WALK}),
            PartitionDefinition(name="Encoder", graph_walks={LIVE_ENCODE_WALK}),
            PartitionDefinition(name="Decoder", graph_walks={LIVE_DECODE_WALK}),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        if not self.live:
            return super().get_partition_topology()
        return PartitionTopology(partitions=[DEFAULT_PARTITION, "Encoder", "Decoder"], connections=[
            # the data worker cuts the audio on a chunk, the encoder's step
            Connection(from_partition=CLIENT, to_partition="Encoder", edge_name=AUDIO_IN,
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
                       limits=ClientStreamLimits(max_items=self.MAX_LIVE_SECONDS * SAMPLE_RATE,
                                                 item_size=torch.numel, unit="samples"),
                       audio=ClientAudio(SAMPLE_RATE, frame_samples=self._chunk_samples())),
            # a step's frames are one item, and the decoder takes them together
            Connection(from_partition="Encoder", to_partition="Decoder", edge_name=ENCODER_FRAMES,
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1)),
        ])

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # a ceiling the runtime enforces: the most iterations a clip can take
        num_frames = model_kwargs.get(NUM_FRAMES)
        return {} if num_frames is None else {DECODE_LOOP: self.config.transducer.decode_iterations(num_frames)}

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        # host values the decode steps read: the capture's bucket, Nemotron's prompt
        step_metadata = {key: model_kwargs.get(key) for key in self.INTERNAL_KWARGS}
        # A whole clip runs the default partition's one walk, until the decode loop's
        # cursor passes the last frame. A live call (no clip; its audio arrives on the
        # client stream) runs the other two, each one stream-fed loop that ends with
        # its stream. The partitions a request doesn't use have nothing to run.
        live = not input_signals.get("audio")
        walk = {DEFAULT_PARTITION: WALK, "Encoder": LIVE_ENCODE_WALK, "Decoder": LIVE_DECODE_WALK}[partition_name]
        if (partition_name == DEFAULT_PARTITION) == live:
            return []
        return [WalkStep(walk, step_metadata=step_metadata)]

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
        doesn't know. A live call (a client stream left open) has no clip: its
        audio comes on the stream."""
        if self.live and kwargs.get("open_streams"):
            if (tensors or {}).get("audio_inputs"):
                # the clip would take the whole-clip walk, and the stream's appends find no consumer
                raise ValueError("a live call's audio comes on its stream: open it with no audio and append it, "
                                 "or send a whole clip with end_input: true")
            return ProcessPromptOutput({}, {PROMPT_ID: self.config.prompt_id(kwargs.get("language"))})
        samples = clip_samples(tensors, self.MAX_AUDIO_SECONDS, "this model")
        metadata = {NUM_FRAMES: self._num_frames(samples.shape[0])}
        if self.config.has_prompt:
            metadata[PROMPT_ID] = self.config.prompt_id(kwargs.get("language"))
        else:
            check_detected_language(kwargs.get("language"))
        return ProcessPromptOutput({"audio": [samples]}, metadata)

    def text_output(self) -> TextOutput:
        return TextOutput(self._detokenizer)

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
            from mstar.model.components.fastconformer_submodule import FastConformerEncoderSubmodule

            return FastConformerEncoderSubmodule(
                self._encoder(device, autocast_dtype), normalize_features=self.config.normalize_features,
            )
        if node_name == "stream_encoder":
            from mstar.model.components.audio_features import NemoLogMel
            from mstar.model.components.fastconformer_stream import FastConformerStreamSubmodule

            return FastConformerStreamSubmodule(
                self._encoder(device, autocast_dtype),
                NemoLogMel(num_mel_bins=self.config.encoder.num_mel_bins, normalize=False, device=device), None,
                input_edge=AUDIO_IN, output_edge=ENCODER_FRAMES, front_key=STREAM_FRONT,
                batch_sizes=[1, 2, 4, 8, 16, 32], item_per_step=True,
                lookahead=self.config.live_encoder.lookahead,
            )
        if node_name in ("transducer", "transducer_stream"):
            from mstar.model.parakeet.submodules import TransducerSubmodule

            return TransducerSubmodule(
                self._head(device), compile_step=self.compile_step, encoder_dtype=autocast_dtype or torch.float32,
                live_frames=self.chunk_frames if node_name == "transducer_stream" else None,
            )
        return None

    def _encoder(self, device: str, autocast_dtype: torch.dtype | None):
        """The encoder, once a device: a clip's node and a live stream's node run
        the same layers (``forward`` and ``forward_stream``), so one copy of the
        weights; the stream state binds only on the stream's node."""
        key = ("encoder", str(device))
        if key not in self._modules:
            from mstar.model.components.fastconformer import FastConformerEncoder

            stream = {"kv_key": STREAM_KV, "conv_key": STREAM_CONV} if self.live else {}
            self._modules[key] = FastConformerEncoder.load(
                self.config.encoder,
                ((k.removeprefix("encoder."), v) for k, v in self._weights(device, "encoder.")),
                device=device, dtype=autocast_dtype, **stream,
            )
        return self._modules[key]

    def _head(self, device: str):
        """The transducer head (fp32: its submodule opts out of the engine's
        cast), once a device, for the clip's and the live call's decoder nodes."""
        key = ("head", str(device))
        if key not in self._modules:
            from mstar.model.components.transducer import TransducerHead

            head = TransducerHead(self.config.transducer).to(device)
            state = {
                k: v for prefix in ("encoder_projector.", "decoder.", "joint.", "prompt_projector.")
                for k, v in self._weights(device, prefix)
            }
            missing, unexpected = head.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise KeyError(f"transducer weights: missing {missing[:4]}, unexpected {unexpected[:4]}")
            self._modules[key] = head.eval()
        return self._modules[key]
