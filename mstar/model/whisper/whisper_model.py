"""
WhisperModel: encoder-decoder ASR model (openai/whisper-large-v3).

Whisper transcribes speech: a log-mel spectrogram runs through a
32-layer audio encoder once, and a 32-layer text decoder generates the
transcript autoregressively, attending to the encoder output through
cross-attention at every step.

Architecture (2 nodes, single partition):
    audio_encoder  (enc_dec) — HF WhisperEncoder, one-shot at prefill
    decoder        (ar)      — mstar-native decoder; paged self-attn KV
                               cache + per-request static cross-attn K/V

Graph walks:
    prefill — audio_encoder -> decoder (forced decoder prompt; the
              engine samples the first transcript token from the logits)
    decode  — decoder loop; each step feeds the sampled token back

Cross-attention K/V depend only on the encoder output, so the decoder
projects them once at prefill and writes them into a context KV stream;
every later step declares a zero-span segment on that label and runs the
engine's pre-planned cross-attention resource.
"""

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.precision import NodePrecision
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    CrossAttentionConfig,
    CrossAttentionSpec,
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
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
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


class WhisperModel(Model):
    """Whisper ASR: HF audio encoder + mstar-native AR decoder."""

    # Requests the default decoder caches hold at once, each at full length.
    # A deployment retunes the page counts under ``resources:`` in its YAML.
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

        from transformers import AutoFeatureExtractor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, cache_dir=cache_dir,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.local_dir, cache_dir=cache_dir,
        )

        from mstar.model.utils import ByteLevelDetokenizer
        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    @staticmethod
    def _cache_pages(tokens: int, page_size: int, requests: int) -> int:
        """Pages for ``requests`` sequences of ``tokens`` each, plus the sink
        page every KV cache keeps out of circulation."""
        return 1 + requests * -(-tokens // page_size)

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Decoder self-attention KV + a separate encoder-context KV.

        Two caches rather than two labels on one: the self-attention resource
        plans a wrapper per label of the cache it names, so a shared cache
        would have it planning the 1500-token context every step for nothing.
        ``audio_encoder`` is stateless.
        """
        page_size = 128
        requests = self.MAX_CONCURRENT_REQUESTS
        # Sequences cap at max_target_positions (448) = 4 pages per request;
        # 257 pages at ~5 GiB for large-v3 (vs 40 GiB with the 2048-page default).
        kv_config = PagedKVConfig(
            num_layers=self.config.decoder_layers,
            num_kv_heads=self.config.decoder_attention_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_target_positions,
            num_qo_heads=self.config.decoder_attention_heads,
            page_size=page_size,
            max_num_pages=self._cache_pages(
                self.config.max_target_positions, page_size, requests,
            ),
        )
        # The fixed 30 s window is max_source_positions (1500) tokens = 12
        # pages per request, held from prefill to the last token; 769 pages at
        # ~15 GiB for large-v3.
        context_kv_config = PagedKVConfig(
            num_layers=self.config.decoder_layers,
            num_kv_heads=self.config.decoder_attention_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_source_positions,
            num_qo_heads=self.config.decoder_attention_heads,
            page_size=page_size,
            max_num_pages=self._cache_pages(
                self.config.max_source_positions, page_size, requests,
            ),
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"decoder"}, config=kv_config),
            AttentionSpec(
                resource_key=ATTN, nodes={"decoder"},
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            KVSpec(
                resource_key=CROSS_KV_CACHE, nodes={"decoder"},
                config=context_kv_config,
            ),
            CrossAttentionSpec(
                resource_key=CROSS_ATTN, nodes={"decoder"},
                config=CrossAttentionConfig(
                    kv_cache=CROSS_KV_CACHE,
                    query_kv_cache=KV_CACHE,
                    context_label=CONTEXT_LABEL,
                ),
            ),
            PositionSpec(
                resource_key=POS, nodes={"decoder"},
                # No RoPE: this exists for the position counter, whose planned
                # ids drive the learned ``embed_positions`` lookup.
                config=PositionConfig(kv_cache=KV_CACHE),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"decoder"},
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

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse a request cap the decoder caches cannot hold.

        A request keeps its encoder context from prefill to its last token, so
        nothing frees cross-attention pages but a finished request. Past the
        pool a prefill batch is refused and the worker offloads decoding
        requests to admit it, which costs far more than queueing would. So
        ``max_concurrent_requests`` requests at full length must fit in each
        cache next to its sink page; the conductor queues the rest.
        """
        cap = config.get("max_concurrent_requests")
        if cap is None:
            logger.warning(
                "%s sets no max_concurrent_requests: past the requests the decoder "
                "caches hold, the worker offloads decoding requests to admit new "
                "ones instead of queueing them", config_path,
            )
            return
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(
                f"{config_path}: max_concurrent_requests must be a positive int; got {cap!r}"
            )
        overrides = config.get("resources") or {}
        for spec in self.get_node_resources():
            if not isinstance(spec, KVSpec):
                continue
            kv = spec.config
            override = overrides.get(spec.resource_key) or {}
            page_size = int(override.get("page_size", kv.page_size))
            max_num_pages = int(override.get("max_num_pages", kv.max_num_pages))
            need = self._cache_pages(kv.max_seq_len, page_size, cap)
            if need > max_num_pages:
                raise ValueError(
                    f"{config_path}: {cap} concurrent requests (max_concurrent_requests) "
                    f"need {need} pages of resources.{spec.resource_key} "
                    f"({-(-kv.max_seq_len // page_size)} per request and the sink page), "
                    f"which has {max_num_pages}; raise max_num_pages or lower "
                    "max_concurrent_requests"
                )

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # The learned position table caps prompt + generated tokens at
        # max_target_positions (448); the forced decoder prompt takes 4.
        limit = self.config.max_target_positions - 4
        return min(model_kwargs.get("max_output_tokens", limit), limit)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio_features"],
                outputs=[GraphEdge(next_node="decoder", name="encoder_states")],
            ),
            GraphNode(
                name="decoder",
                input_names=["encoder_states", "text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                ],
            ),
        ])

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="decoder",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="decoder",
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )

        audio_edge = GraphEdge(next_node="audio_encoder", name="audio_features")
        audio_edge.tensor_info = input_signals.get("audio_features", [])
        text_edge = GraphEdge(next_node="decoder", name="text_inputs")
        text_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [audio_edge, text_edge]

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Single-partition state machine: prefill -> decode loop -> done."""
        metadata = partition_metadata

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # The decode dynamic loop returned to the conductor: EOS or
            # max tokens was hit, so the request is complete.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        edge = GraphEdge(next_node="decoder", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Extract the log-mel spectrogram and build the forced decoder prompt.

        The text ``prompt`` is unused — Whisper is conditioned via the
        forced token sequence (language / task / timestamps), which is
        controlled by ``language`` and ``task`` in model kwargs.
        """
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Whisper expects exactly one audio input per request; "
                f"got {len(raw_audio_inputs)}."
            )

        # The extractor pads/truncates to the fixed 30 s window; audio
        # beyond 30 s is dropped (no long-form chunking yet).
        feat = self.feature_extractor(
            raw_audio_inputs[0].cpu().numpy(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        audio_features = feat["input_features"][0]  # (num_mel_bins, 3000)

        prompt_ids = self.config.decoder_prompt_ids(
            language=kwargs.get("language", "en"),
            task=kwargs.get("task", "transcribe"),
        )

        return {
            "audio_features": [audio_features],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }

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

        return ShardingConfig(groups=[], tp_enabled_nodes={"decoder"}, shard_dim={})

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_node_precision(self, node_name: str) -> NodePrecision:
        """bf16 params, no autocast. The LayerNorms emit bf16 where autocast
        emitted fp32, which every consumer rounded to bf16 at its next matmul."""
        del node_name
        return NodePrecision(torch.bfloat16, autocast=False)

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
        if node_name == "audio_encoder":
            return self._create_encoder_submodule(device)
        elif node_name == "decoder":
            return self._create_decoder_submodule(
                device, autocast_dtype=autocast_dtype, tp_group=tp_group,
            )
        return None

    def _create_encoder_submodule(self, device: str) -> NodeSubmodule:
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        from mstar.model.utils import (
            ModuleAndPrefix,
            load_weights_from_hf_shards,
        )

        hf_config = WhisperConfig.from_pretrained(self.local_dir)
        audio_encoder = WhisperEncoder._from_config(
            hf_config, attn_implementation="sdpa",
        )
        modules = [ModuleAndPrefix(audio_encoder, prefix="model.encoder")]
        load_weights_from_hf_shards(
            repo_dir=self.local_dir, modules=modules, device=device,
        )
        audio_encoder.eval()

        from mstar.model.whisper.submodules import WhisperEncoderSubmodule
        return WhisperEncoderSubmodule(audio_encoder=audio_encoder, config=self.config)

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
