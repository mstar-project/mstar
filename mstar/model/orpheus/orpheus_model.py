"""
OrpheusModel: Model implementation for Orpheus TTS (streaming).

Orpheus consists of a Llama 3.2 3B LLM that generates custom audio tokens
and a SNAC decoder that converts those tokens to 24kHz PCM audio. The LLM
generates 7 tokens per audio frame; each group of 7 tokens decomposes into
3 SNAC codebook levels which the SNAC model decodes into a waveform chunk.

Architecture (2 nodes, 2 async partitions):
    LLM           (ar)          - Llama 3.2 3B with extended vocab for audio tokens
    snac_decoder  (audio_codec) - SNAC 24kHz decoder

Partitions:
    LLM  — walks: prefill → decode (autoregressive token generation)
    SNAC — walks: snac_chunk (streaming decode, triggered when tokens accumulate)

The LLM and SNAC partitions run asynchronously. The LLM produces tokens that
stream directly to the SNAC worker via StreamingGraphEdge, which releases
windows of frames with left and right context (``OrpheusModelConfig.snac_*``);
the SNAC decoder emits the audio of each window's own frames.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, PartitionDefinition, StreamingConnectionState
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
from mstar.engine.resources.lora import LoraSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, PrefixKeys, ProcessPromptOutput, Span
from mstar.model.orpheus.config import ATTN, KV_CACHE, ROPE, SAMPLER, OrpheusModelConfig
from mstar.model.submodule_base import NodeSubmodule
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

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
        logger.warning("Error downloading from huggingface: %s", str(e))
        return repo_id
    return str(Path(local_dir))


class OrpheusModel(Model):
    """Orpheus TTS model: Llama 3.2 3B + SNAC 24kHz decoder."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        snac_chunk_schedule: list[int] | None = None,
        snac_chunk_frames: int | None = None,
        snac_left_context_frames: int | None = None,
        snac_right_context_frames: int | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        snac_overrides = {
            "snac_chunk_schedule": snac_chunk_schedule,
            "snac_chunk_frames": snac_chunk_frames,
            "snac_left_context_frames": snac_left_context_frames,
            "snac_right_context_frames": snac_right_context_frames,
        }
        self.config = OrpheusModelConfig(
            **{k: v for k, v in snac_overrides.items() if v is not None}
        )

        tokenizer_source = _resolve_local_hf_snapshot(
            "canopylabs/orpheus-3b-0.1-pretrained",
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            cache_dir=cache_dir,
        )

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}


    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        return {"LLM": {"mixed": {"prefill", "decode"}}}

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="new_token",
                    conductor_new_token=True,
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="snac_decoder",
                    name="new_token",
                    target_partition="SNAC",
                ),
            ],
        )

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node="LLM",
                        name="text_inputs",
                    ),
                    StreamingGraphEdge(
                        next_node="snac_decoder",
                        name="new_token",
                        target_partition="SNAC",
                    ),
                ],
            ),
            max_iters=self.config.max_position_embeddings,
            outputs=[],
        )

        snac_chunk = GraphNode(
            name="snac_decoder",
            input_names=["new_token"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="audio_chunk",
                    output_modality="audio",
                ),
            ],
        )

        return dict(
            prefill=prefill,
            decode=decode,
            snac_chunk=snac_chunk,
        )

    # -------------------------------------------------------------------
    # Partition API: async streaming (LLM + SNAC)
    # -------------------------------------------------------------------

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=["LLM", "SNAC"],
            connections=[
                Connection(
                    from_partition="LLM",
                    to_partition="SNAC",
                    edge_name="new_token",
                    chunk_policy_factory=self.config.snac_chunk_policy,
                ),
            ],
        )

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="LLM",
                graph_walks={"prefill", "decode"},
                initial_walk="prefill",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="SNAC",
                graph_walks={"snac_chunk"},
                initial_walk=None,
                producer_partitions=["LLM"],
            ),
        ]

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_name == "LLM":
            return self._get_llm_partition_forward(
                partition_metadata, persist_signals,
            )
        elif partition_name == "SNAC":
            # Extract streaming state from the incoming connection
            conn = incoming_connections[0] if incoming_connections else None
            token_buffer_count = conn.token_count if conn else 0
            producer_done = conn.producer_done if conn else False
            consumed = conn.consumed_count if conn else 0
            return self._get_snac_partition_forward(
                partition_metadata, token_buffer_count, producer_done, consumed,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def _get_llm_partition_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> ForwardPassArgs:
        """LLM partition: prefill -> decode loop until EOS."""
        request_done = False

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        step_metadata = {
            "is_prefill": metadata.is_prefill,
        }

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
        )

    def _get_snac_partition_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        token_buffer_count: int,
        producer_done: bool,
        consumed: int = 0,
    ) -> ForwardPassArgs:
        """SNAC partition: the streaming decode loop is self-triggered,
        so this function is basically a no-op.
        """
        metadata.graph_walk = "snac_chunk"

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[],
            unpersist_tensors=[],
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        # Orpheus is text-only; raw multimodal tensors are unused.
        # Refused here, where it is a 400 before admission: the LLM's prefill
        # has no input without text, and blank text would generate from the voice
        # prefix alone.
        if prompt is None or not prompt.strip():
            raise ValueError("Orpheus requires a non-empty text prompt")

        # An explicit empty/None voice keeps the unprefixed prompt.
        voice = kwargs.get("voice", self.config.default_voice)
        if voice:
            voice = str(voice).lower()
            if voice not in self.config.available_voices:
                supported = ", ".join(self.config.available_voices)
                raise ValueError(
                    f"Unsupported Orpheus voice {voice!r}; supported: {supported}"
                )

        # Format: "{voice}: {text}"
        adapted_prompt = f"{voice}: {prompt}" if voice else prompt
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")

        # Wrap with special tokens: [128259, ...tokens..., 128009, 128260, 128261, 128257]
        start_token = torch.tensor([self.config.start_token_id], dtype=torch.long)
        end_tokens = torch.tensor(self.config.end_token_ids, dtype=torch.long)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids[0], end_tokens])

        length = all_input_ids.numel()
        layout = LabelLayout(
            spans=(Span("ids", "prefill", length, length, "text_inputs"),), growth=(("decode_loop", 1),),
        )
        return ProcessPromptOutput({"text_inputs": [all_input_ids]}, {"kv_layout": {KV_CACHE: {"main": layout}}})

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
        if partition_name == "LLM":
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="prefill",
                is_prefill=True,
            )

            graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
            graph_edge.tensor_info = input_signals.get("text_inputs", [])
            inputs = [graph_edge]

            unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=inputs,
                unpersist_tensors=unpersist_tensors,
                step_metadata={
                    "is_prefill": True,
                },
            )
        elif partition_name == "SNAC":
            # SNAC starts with snac_chunk walk but no inputs —
            # it will self-trigger via StreamBuffer when tokens arrive.
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="snac_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {"decode_loop": self.get_max_output_tokens(**model_kwargs)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        """The LLM's prompt is its token ids. Its generated audio codes never
        come back in a prompt, so they are not keyed."""
        return {KV_CACHE: {"main": KVLabel(
            growth=(("decode", "decode_loop"),),
            keys=PrefixKeys("text_inputs", ("prefill",)),
        )}}

    def checkpoint_path(self) -> str:
        return _resolve_local_hf_snapshot(
            self.model_path_hf, cache_dir=self.cache_dir,
        )

    def _lora_spec(self) -> LoraSpec:
        """Per-request adapters on the LLM (opt-in): every linear of the Llama stack, the
        LM head (one row per request: the last token's) and the token embedding, placed as
        the weight loader places the checkpoint's own tensors. Adapters saved on the bare
        ``LlamaModel`` (``layers.*``) are read under ``model.``."""
        from mstar.model.components.lora.modules import PER_SEGMENT, meta_root, module_lora_spec
        from mstar.model.orpheus.components.language_model import OrpheusForCausalLM

        root = meta_root(lambda group: OrpheusForCausalLM(self.config, comm_group=group))
        return module_lora_spec(
            root, nodes={"LLM"}, root_path="language_model",
            adapter_roots=[("", "model.")],
            rows={"lm_head": PER_SEGMENT},
            embeddings=True,
        )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        kv_config = PagedKVConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )
        return [
            KVSpec(
                resource_key=KV_CACHE,
                nodes={"LLM"},
                config=kv_config
            ),
            AttentionSpec(
                resource_key=ATTN,
                nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE,
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={"LLM"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True
            ),
            PositionSpec(
                resource_key=ROPE,
                nodes={"LLM"},
                config=PositionConfig(
                    kv_cache=KV_CACHE,
                    rope_theta=self.config.rope_theta,
                    rope_scale=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    old_context_len=self.config.rope_scaling["original_max_position_embeddings"],
                )
            ),
            self._lora_spec(),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        keys = [
            "temperature", "top_p", "repetition_penalty",
            "ignore_eos"
        ]
        return {
            SAMPLER: SamplingReqConfig(
                **{
                    k: model_kwargs.get(k, getattr(self.config, k))
                    for k in keys
                }
            )
        }


    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.sample_rate

    def get_voices(self) -> list[str]:
        return list(self.config.available_voices)

    def get_default_voice(self) -> str:
        return self.config.default_voice

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        **kwargs
    ) -> bytes:
        if modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Orpheus: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: sharding
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

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
        logger.info("Successfully loaded Orpheus submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "LLM":
            return self._create_llm_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        elif node_name == "snac_decoder":
            return self._create_snac_submodule(device)
        return None

    def _create_llm_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_weights
        from mstar.model.orpheus.components.language_model import OrpheusForCausalLM
        from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

        local_dir = _resolve_local_hf_snapshot(
            self.model_path_hf,
            cache_dir=self.cache_dir,
        )

        with torch.device("meta"):
            language_model = OrpheusForCausalLM(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)

        load_weights(language_model, local_dir, device=device)
        language_model.eval()

        return OrpheusLLMSubmodule(
            language_model=language_model,
            config=self.config,
        )

    def _create_snac_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.orpheus.components.snac import SNAC
        from mstar.model.orpheus.submodules import SNACDecoderSubmodule

        snac_source = _resolve_local_hf_snapshot(
            self.config.snac_model_id,
            cache_dir=self.cache_dir,
        )
        snac_model = SNAC.from_pretrained(snac_source).eval().to(device)
        return SNACDecoderSubmodule(
            snac_model=snac_model,
            config=self.config,
        )
