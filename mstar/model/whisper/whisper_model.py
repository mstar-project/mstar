"""
WhisperModel: encoder-decoder ASR (openai/whisper-large-v3-turbo, -large-v3, ...).

Whisper transcribes speech: a log-mel spectrogram of one 30 s window runs
through a 32-layer audio encoder once, and a text decoder (4 layers for
turbo, 32 for large-v3) generates the transcript autoregressively,
attending to the encoder output through cross-attention at every step.
One class serves every size: dims and token ids come from the checkpoint's
``config.json`` / ``generation_config.json`` / ``preprocessor_config.json``.

Architecture (2 nodes, single partition):
    audio_encoder  — native batched encoder, one CUDA graph per batch size
    decoder        — native decoder; paged self-attn KV cache + write-once
                     cross-attention context; decode captured per batch size

Graph walks:
    prefill          — audio_encoder -> decoder over the forced prompt
                       ``[<|startofprev|> ctx] <|sot|><|lang|><|task|>[<|notimestamps|>]``;
                       samples the first transcript token
    detect_language  — audio_encoder -> decoder over ``<|sot|>`` alone, sampling
                       restricted to language tokens (``language`` not given)
    prefill_prompt   — decoder only: ``<|lang|>`` (detected) + ``<|task|>[<|notimestamps|>]``
                       over the context the previous walk wrote
    decode           — decoder loop; each step feeds the sampled token back

Request state machine:
    language given:  prefill -> decode -> done
    language absent: detect_language -> prefill_prompt -> decode -> done

The transcript is streamed as Whisper's own token stream rendered to text:
spoken words as bytes, plus the language token (``<|en|>``) and, when
timestamps were requested, the ``<|s.ss|>`` markers. The OpenAI layer's
adapter lifts those into ``language`` / ``segments``.
"""

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    CrossAttentionConfig,
    CrossAttentionSpec,
    KVConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.components.audio_features import LogMelSpectrogram, load_audio_file
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelDetokenizer
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    DECODE_LOOP,
    DECODE_WALK,
    DECODER_NODE,
    DETECT_LANGUAGE_WALK,
    ENCODER_NODE,
    KV_CACHE,
    POS,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
    SAMPLER,
    WhisperModelConfig,
)

logger = logging.getLogger(__name__)


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        logger.warning("Error downloading from HuggingFace: %s", str(e))
        return repo_id
    return str(Path(local_dir))


class WhisperDetokenizer(ByteLevelDetokenizer):
    """Byte-level detokenizer that keeps the informative control tokens.

    Language tokens and timestamp tokens are rendered as their literal
    ``<|xx|>`` / ``<|s.ss|>`` text so the serving layer can lift them out;
    the structural specials (``<|startoftranscript|>``, ``<|transcribe|>``,
    ``<|notimestamps|>``, ``<|endoftext|>``, ...) are dropped as before.
    """

    def __init__(self, tokenizer, config: WhisperModelConfig):
        super().__init__(tokenizer)
        self.config = config
        self._render_ids = set(config.language_token_ids)

    def _rendered(self, token_id: int) -> bool:
        return token_id in self._render_ids or self.config.is_timestamp(token_id)

    def to_bytes(self, token_ids: list[int]) -> bytes:
        raw = bytearray()
        for token_id in token_ids:
            if self._rendered(token_id):
                raw.extend(self.tokenizer.convert_ids_to_tokens(token_id).encode("utf-8"))
            elif token_id not in self.special_ids:
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                raw.extend(self.byte_decoder[c] for c in token)
        return bytes(raw)


class WhisperModel(Model):
    """Whisper ASR: native batched encoder + native AR decoder."""

    # Concurrency the default cache sizing targets; a deployment retunes the
    # page counts under ``resources:`` in its YAML.
    MAX_CONCURRENT_REQUESTS = 64

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf

        self.local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = WhisperModelConfig.from_pretrained(self.local_dir)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, cache_dir=cache_dir,
        )
        self.log_mel = LogMelSpectrogram(
            num_mel_bins=self.config.num_mel_bins,
            sampling_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            chunk_length=self.config.chunk_length,
        )
        self._detokenizer = WhisperDetokenizer(self.tokenizer, self.config)

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    @staticmethod
    def _pages(tokens: int, page_size: int, requests: int) -> int:
        return -(-tokens // page_size) * requests

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Decoder self-attention KV + a separate encoder-context KV.

        Two caches rather than two labels on one: the self-attention resource
        plans a wrapper per label of the cache it names, so a shared cache
        would have it planning the 1500-token context every step for nothing.
        ``audio_encoder`` holds no resources.
        """
        page_size = 128
        concurrency = self.MAX_CONCURRENT_REQUESTS
        # Sequences cap at max_target_positions (448) = 4 pages per request;
        # the decode captures' padding rows hold one page each on first use.
        kv_config = KVConfig(
            num_layers=self.config.decoder_layers,
            num_kv_heads=self.config.decoder_attention_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_target_positions,
            num_qo_heads=self.config.decoder_attention_heads,
            page_size=page_size,
            max_num_pages=self._pages(self.config.max_target_positions, page_size, concurrency)
            + 2 * concurrency,
        )
        # The fixed 30 s window is max_source_positions (1500) tokens = 12
        # pages per request.
        context_kv_config = KVConfig(
            num_layers=self.config.decoder_layers,
            num_kv_heads=self.config.decoder_attention_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_source_positions,
            num_qo_heads=self.config.decoder_attention_heads,
            page_size=page_size,
            max_num_pages=self._pages(self.config.max_source_positions, page_size, concurrency),
        )
        nodes = {DECODER_NODE}
        return [
            KVSpec(resource_key=KV_CACHE, nodes=nodes, config=kv_config),
            AttentionSpec(
                resource_key=ATTN, nodes=nodes,
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            KVSpec(
                resource_key=CROSS_KV_CACHE, nodes=nodes,
                config=context_kv_config,
            ),
            CrossAttentionSpec(
                resource_key=CROSS_ATTN, nodes=nodes,
                config=CrossAttentionConfig(
                    kv_cache=CROSS_KV_CACHE,
                    query_kv_cache=KV_CACHE,
                    context_label=CONTEXT_LABEL,
                ),
            ),
            PositionSpec(
                resource_key=POS, nodes=nodes,
                # No RoPE: this exists for the position counter, whose planned
                # ids drive the learned ``embed_positions`` lookup.
                config=PositionConfig(kv_cache=KV_CACHE),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes=nodes,
                vocab_size=self.config.vocab_size,
                # ASR transcription decodes greedily; no seen-token buffers.
                enable_repetion_penalty=False,
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                # ASR default is greedy (temperature 0 -> argmax).
                temperature=model_kwargs.get("temperature", 0.0),
                top_p=model_kwargs.get("top_p", 1.0),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            )
        }

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # The learned position table caps prompt + generated tokens at
        # max_target_positions (448); the shortest forced prompt takes 4.
        # A longer prompt (``<|startofprev|>`` context) is accounted per
        # request in the decoder's ``check_stop``.
        limit = self.config.max_target_positions - 4
        return min(model_kwargs.get("max_output_tokens", limit), limit)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        def emit_first_token() -> list[GraphEdge]:
            return [
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True,
                ),
            ]

        def encoder_then_decoder() -> GraphSection:
            return Sequential([
                GraphNode(
                    name=ENCODER_NODE,
                    input_names=["audio_features"],
                    outputs=[GraphEdge(next_node=DECODER_NODE, name="encoder_states")],
                ),
                GraphNode(
                    name=DECODER_NODE,
                    input_names=["encoder_states", "text_inputs"],
                    outputs=emit_first_token(),
                ),
            ])

        prefill_prompt = GraphNode(
            name=DECODER_NODE,
            input_names=["text_inputs", "prompt_tail"],
            outputs=emit_first_token(),
        )

        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=DECODER_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node=DECODER_NODE,
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return {
            PREFILL_WALK: encoder_then_decoder(),
            DETECT_LANGUAGE_WALK: encoder_then_decoder(),
            PREFILL_PROMPT_WALK: prefill_prompt,
            DECODE_WALK: decode,
        }

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    @staticmethod
    def _edge(node: str, name: str, tensor_info: list[TensorPointerInfo]) -> GraphEdge:
        edge = GraphEdge(next_node=node, name=name)
        edge.tensor_info = list(tensor_info)
        return edge

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        # ``process_prompt`` emits ``prompt_tail`` exactly when the language
        # is to be detected; it is held back for the second prefill walk.
        prompt_tail = input_signals.get("prompt_tail", [])
        detect = bool(prompt_tail)
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=DETECT_LANGUAGE_WALK if detect else PREFILL_WALK,
            is_prefill=True,
            kwargs={"prompt_tail": prompt_tail},
        )
        inputs = [
            self._edge(ENCODER_NODE, "audio_features", input_signals.get("audio_features", [])),
            self._edge(DECODER_NODE, "text_inputs", input_signals.get("text_inputs", [])),
        ]
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Single-partition state machine:
        ``[detect_language -> prefill_prompt | prefill] -> decode -> done``."""
        metadata = partition_metadata
        new_token = persist_signals.get("new_token", [])

        if metadata.is_prefill:
            if metadata.graph_walk == DETECT_LANGUAGE_WALK:
                # the sampled language token leads the rest of the prompt
                metadata.graph_walk = PREFILL_PROMPT_WALK
                inputs = [
                    self._edge(DECODER_NODE, "text_inputs", new_token),
                    self._edge(DECODER_NODE, "prompt_tail", metadata.kwargs.get("prompt_tail", [])),
                ]
                return ForwardPassArgs(
                    full_metadata=metadata,
                    inputs=inputs,
                    unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
                    step_metadata={"is_prefill": True},
                )
            metadata.is_prefill = False
            metadata.graph_walk = DECODE_WALK
        elif metadata.graph_walk == DECODE_WALK:
            # The decode dynamic loop returned to the conductor: EOS or
            # max tokens was hit, so the request is complete.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        inputs = [self._edge(DECODER_NODE, "text_inputs", new_token)]
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # Model ABC: media loading + prompt processing
    # -------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        del device  # decoded on the CPU; the mel front end runs there too
        waveform = load_audio_file(filepath, self.config.sampling_rate)
        return TensorAndMetadata(
            data=waveform, metadata=dict(sample_rate=self.config.sampling_rate, num_channels=1),
        )

    def prompt_tokens(self, prompt: str | None) -> list[int]:
        """The ``<|startofprev|>`` context: prior transcript text, tokenized
        the way openai-whisper does (a leading space, no specials)."""
        if not prompt or not prompt.strip():
            return []
        return self.tokenizer.encode(" " + prompt.strip(), add_special_tokens=False)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Extract the log-mel window and build the forced decoder prompt.

        The text ``prompt`` is unused — Whisper is conditioned via its forced
        token sequence, controlled by ``language`` (ISO-639-1; ``None`` means
        detect it), ``task`` (``transcribe`` / ``translate``), ``timestamps``
        and ``initial_prompt`` (prior text carried over as ``<|startofprev|>``
        context) in model kwargs.
        """
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Whisper expects exactly one audio input per request; "
                f"got {len(raw_audio_inputs)}."
            )
        waveform = raw_audio_inputs[0].reshape(-1).to(torch.float32).cpu()
        if waveform.numel() == 0:
            raise ValueError("Whisper received an empty audio input.")

        # One fixed 30 s window: audio beyond it is dropped (long-form
        # chunking is the transcription route's job).
        window = self.log_mel.pad_or_trim(waveform)
        audio_features = self.log_mel(window)  # (num_mel_bins, 3000)

        language = kwargs.get("language")
        task = kwargs.get("task", "transcribe")
        timestamps = bool(kwargs.get("timestamps", False))
        prev_tokens = self.prompt_tokens(kwargs.get("initial_prompt"))
        prompt_ids = self.config.decoder_prompt_ids(
            language=language, task=task, timestamps=timestamps, prev_tokens=prev_tokens,
        )

        out: NameToTensorList = {
            "audio_features": [audio_features],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }
        if language is None:
            out["prompt_tail"] = [torch.tensor(
                self.config.prompt_tail_ids(task=task, timestamps=timestamps), dtype=torch.long,
            )]
        return out

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        **kwargs,
    ) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Whisper: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: sharding
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={DECODER_NODE}, shard_dim={})

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        logger.info("Successfully loaded Whisper submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == ENCODER_NODE:
            return self._create_encoder_submodule(device, autocast_dtype=autocast_dtype)
        elif node_name == DECODER_NODE:
            return self._create_decoder_submodule(
                device, autocast_dtype=autocast_dtype, tp_group=tp_group,
            )
        return None

    def _create_encoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.whisper.components.encoder import WhisperEncoderModel

        with torch.device("meta"):
            encoder = WhisperEncoderModel(self.config)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            encoder = encoder.to(autocast_dtype)
        encoder.to_empty(device=device)

        prefix = "model.encoder."
        weights = iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)
        encoder.load_weights((k.removeprefix(prefix), v) for k, v in weights)
        encoder.eval()

        from mstar.model.whisper.submodules import WhisperEncoderSubmodule
        return WhisperEncoderSubmodule(encoder=encoder, config=self.config)

    @staticmethod
    def _decoder_remap(name: str) -> str:
        # The shared Attention component names its output projection
        # ``o_proj``; the cross-attn module keeps HF's ``out_proj``.
        return name.replace("self_attn.out_proj", "self_attn.o_proj")

    def _create_decoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None = None, tp_group=None
    ) -> NodeSubmodule:
        from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.whisper.components.decoder import WhisperDecoderModel

        with torch.device("meta"):
            decoder = WhisperDecoderModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            decoder = decoder.to(autocast_dtype)
        decoder.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device, prefix="model.decoder.",
        )
        weights = ((k.removeprefix("model.decoder."), v) for k, v in weights)
        load_hf_weights(
            decoder,
            weights,
            stacked_params=WHISPER_STACKED_PARAMS,
            name_remapper=self._decoder_remap,
        )
        decoder.zero_missing_biases()
        decoder.eval()

        from mstar.model.whisper.submodules import WhisperDecoderSubmodule
        return WhisperDecoderSubmodule(decoder=decoder, config=self.config)
