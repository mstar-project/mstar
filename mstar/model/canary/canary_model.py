"""
CanaryModel: NVIDIA's encoder-decoder speech recognition and translation model
(nvidia/canary-1b-v2, 25 European languages).

A FastConformer encoder hears the clip once; a Transformer decoder writes the
transcript, or its English translation (or a translation from English),
attending to the encoder output at every step. The decoder prompt names the
source and target language and whether to punctuate.

Architecture (2 nodes, single partition):
    audio_encoder  — log-mel + FastConformer, one-shot at prefill
    decoder        — paged self-attention KV cache + per-request cross-attention
                     K/V written once at prefill

Graph walks:
    prefill — audio_encoder -> decoder (task prompt; the first token is sampled)
    decode  — decoder loop, each step feeding the sampled token back
"""

import json
import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata, StreamingConnectionState
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
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, TensorAndMetadata
from mstar.model.canary.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    PROMPT_LENGTH,
    SAMPLER,
    CanaryModelConfig,
)
from mstar.model.components.audio_features import HOP_LENGTH, SAMPLE_RATE, clip_samples, load_speech_file
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import SentencePieceDetokenizer, check_kv_pools_fit, check_loaded, kv_pages
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

# the files the model reads; skips the repo's .nemo archive
CHECKPOINT_FILES = ["*.json", "*.jinja", "*.safetensors"]


class CanaryModel(Model):
    """Canary speech recognition / translation: FastConformer encoder +
    autoregressive Transformer decoder."""

    # the route's 40 s window, plus slack for containers that decode longer than
    # their header said; still the largest encoder graph and 4 context pages
    MAX_AUDIO_SECONDS = 40.96
    # the requests the default caches hold at full length; configs/canary.yaml
    # caps max_concurrent_requests here, so the conductor queues the rest
    MAX_CONCURRENT_REQUESTS = 64
    # a few tiny ops per upload; see "Preprocessing threads" in docs/adding_models.rst
    PREPROCESS_TORCH_THREADS = 1

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        self.model_path_hf = model_path_hf
        self.local_dir = resolve_snapshot_dir(model_path_hf, cache_dir, allow_patterns=CHECKPOINT_FILES)
        self.config = CanaryModelConfig.from_pretrained(self.local_dir)
        tokenizer = json.loads((self.local_dir / "tokenizer.json").read_text())
        self._detokenizer = SentencePieceDetokenizer(tokenizer)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def checkpoint_path(self) -> str | None:
        return str(self.local_dir)

    # -------------------------------------------------------------------
    # resources
    # -------------------------------------------------------------------

    def _max_encoder_frames(self) -> int:
        samples = int(self.MAX_AUDIO_SECONDS * self.config.sampling_rate)
        return self.config.encoder.output_lengths(samples // HOP_LENGTH)

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Decoder self-attention KV and a separate encoder-context KV: the
        self-attention resource plans a wrapper per label of the cache it
        names, so a shared cache would plan the context every step."""
        cfg = self.config
        page_size, requests = 128, self.MAX_CONCURRENT_REQUESTS
        # each cache holds the request cap at full length: 8 pages per request
        # at the position limit, 4 for a 40.96 s clip's context
        kv_config = PagedKVConfig(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            num_qo_heads=cfg.num_heads,
            page_size=page_size,
            max_num_pages=kv_pages(cfg.max_position_embeddings, page_size, requests),
        )
        context_kv_config = PagedKVConfig(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            max_seq_len=self._max_encoder_frames(),
            num_qo_heads=cfg.num_heads,
            page_size=page_size,
            max_num_pages=kv_pages(self._max_encoder_frames(), page_size, requests),
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"decoder"}, config=kv_config),
            AttentionSpec(resource_key=ATTN, nodes={"decoder"}, config=AttentionConfig(kv_cache=KV_CACHE)),
            KVSpec(resource_key=CROSS_KV_CACHE, nodes={"decoder"}, config=context_kv_config),
            CrossAttentionSpec(
                resource_key=CROSS_ATTN, nodes={"decoder"},
                config=CrossAttentionConfig(
                    kv_cache=CROSS_KV_CACHE, query_kv_cache=KV_CACHE, context_label=CONTEXT_LABEL,
                ),
            ),
            # no RoPE: the counter's planned ids drive the sinusoid lookup
            PositionSpec(resource_key=POS, nodes={"decoder"}, config=PositionConfig(kv_cache=KV_CACHE)),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"decoder"},
                vocab_size=cfg.vocab_size,
                enable_repetion_penalty=False,
            ),
        ]

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        check_kv_pools_fit(self, config, config_path, logger)

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                # greedy by default, as NeMo and HF decode Canary
                temperature=model_kwargs.get("temperature", 0.0),
                top_p=model_kwargs.get("top_p", 1.0),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            )
        }

    # -------------------------------------------------------------------
    # graph walks
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # the decoder's position table bounds prompt + transcript
        limit = self.config.max_position_embeddings - PROMPT_LENGTH
        requested = model_kwargs.get("max_output_tokens")
        return limit if requested is None else min(requested, limit)

    def get_max_output_tokens_limit(self) -> int:
        return self.get_max_output_tokens()

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        return {"decode_loop": self.get_max_output_tokens(**model_kwargs)}

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio"],
                outputs=[GraphEdge(next_node="decoder", name="encoder_states")],
            ),
            GraphNode(
                name="decoder",
                input_names=["encoder_states", "text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
                ],
            ),
        ])
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="decoder",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                    GraphEdge(next_node="decoder", name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return dict(prefill=prefill, decode=decode)

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
            graph_walk="prefill",
            is_prefill=True,
        )
        audio_edge = GraphEdge(next_node="audio_encoder", name="audio")
        audio_edge.tensor_info = input_signals.get("audio", [])
        text_edge = GraphEdge(next_node="decoder", name="text_inputs")
        text_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [audio_edge, text_edge]
        return ForwardPassArgs(
            full_metadata=metadata,
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
        """prefill -> decode loop -> done."""
        metadata = partition_metadata
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # the decode loop returned: EOS or the token limit
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        edge = GraphEdge(next_node="decoder", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[edge],
            unpersist_tensors=list(edge.tensor_info),
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # prompt / output
    # -------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """The clip's samples and the decoder prompt. The text ``prompt`` is
        unused: the task comes from ``language`` (the speech's language,
        default English), ``target_language`` (translation; default the
        speech's language) and ``punctuation`` (default on)."""
        samples = clip_samples(tensors, self.MAX_AUDIO_SECONDS, "Canary")
        source = kwargs.get("language") or "en"
        target = kwargs.get("target_language") or source
        punctuation = kwargs.get("punctuation", True)
        if not isinstance(punctuation, bool):
            raise ValueError(f"punctuation must be true or false; got {punctuation!r}")
        prompt_ids = self.config.prompt_ids(source, target, punctuation)
        # the encoder's frames for this clip, as the encoder node counts them
        context = int(self.config.encoder.output_lengths(samples.shape[0] // HOP_LENGTH))
        kv_layout = {
            KV_CACHE: {"main": LabelLayout(
                spans=(Span("ids", "prefill", len(prompt_ids), len(prompt_ids), "text_inputs"),),
                growth=(("decode_loop", 1),),
            )},
            CROSS_KV_CACHE: {CONTEXT_LABEL: LabelLayout(
                spans=(Span("rows", "prefill", context, context, "encoder_states"),),
            )},
        }
        return ProcessPromptOutput({
            "audio": [samples],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }, {"kv_layout": kv_layout})

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {
            KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))},
            CROSS_KV_CACHE: {CONTEXT_LABEL: KVLabel()},
        }

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        audio = load_speech_file(filepath, SAMPLE_RATE, self.MAX_AUDIO_SECONDS)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=SAMPLE_RATE, num_channels=1))

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Canary: {modality!r}")

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

    def _weights(self, device: str, *prefixes: str):
        from mstar.model.loader.iterators import iter_safetensors_shards

        for prefix in prefixes:
            yield from iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)

    def _create_submodule(self, node_name: str, device: str, autocast_dtype: torch.dtype | None):
        if node_name == "audio_encoder":
            from mstar.model.components.fastconformer import FastConformerEncoder
            from mstar.model.components.fastconformer_submodule import FastConformerEncoderSubmodule

            with torch.device("meta"):
                encoder = FastConformerEncoder(self.config.encoder)
            encoder = _materialize(encoder, device, autocast_dtype)
            prefix = "model.encoder."
            loaded = encoder.load_hf_weights(
                (k.removeprefix(prefix), v) for k, v in self._weights(device, prefix)
            )
            check_loaded(encoder, loaded, "Canary encoder")
            return FastConformerEncoderSubmodule(encoder.eval())
        if node_name == "decoder":
            from mstar.model.canary.decoder import CanaryDecoderModel
            from mstar.model.canary.submodules import CanaryDecoderSubmodule
            from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights

            with torch.device("meta"):
                decoder = CanaryDecoderModel(self.config)
            decoder = _materialize(decoder, device, autocast_dtype)
            loaded = load_hf_weights(
                decoder, self._weights(device, "model.decoder.", "proj_out."),
                stacked_params=WHISPER_STACKED_PARAMS,
                name_remapper=CanaryDecoderModel.remap,
            )
            check_loaded(decoder, loaded, "Canary decoder")
            return CanaryDecoderSubmodule(decoder=decoder.eval(), config=self.config)
        return None


def _materialize(module: torch.nn.Module, device: str, dtype: torch.dtype | None) -> torch.nn.Module:
    """Allocate a meta-built module on ``device``; cast first, so storage is
    allocated in the target dtype rather than fp32 then down-cast."""
    if dtype is not None:
        module = module.to(dtype)
    return module.to_empty(device=device)
