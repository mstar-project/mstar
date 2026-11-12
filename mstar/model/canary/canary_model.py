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
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.state import (
    Array,
    Attend,
    Bidirectional,
    Causal,
    KVEntry,
    Layout,
    Positions,
    Read,
    Sampler,
    SampleSettings,
    Span,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import (
    KVLabel,
    Model,
    ProcessPromptOutput,
    WalkStep,
)
from mstar.model.canary.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    PAGE_TOKENS,
    PROMPT_LENGTH,
    SAMPLER,
    CanaryModelConfig,
)
from mstar.model.components.audio_features import HOP_LENGTH, clip_samples
from mstar.model.components.request_kwargs import (
    GREEDY_SAMPLING,
    SAMPLING_KEYS,
    checked_request_kwargs,
    sampler_capabilities,
    sampling_config,
    sampling_defaults,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import SentencePieceDetokenizer, TextOutput, check_kv_pools_fit, check_loaded, kv_pages
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)

# the files the model reads; skips the repo's .nemo archive
CHECKPOINT_FILES = ["*.json", "*.jinja", "*.safetensors"]


class CanaryModel(Model):
    """Canary speech recognition / translation: FastConformer encoder +
    autoregressive Transformer decoder."""

    DECODE_LOOP = "decode_loop"
    # greedy, as NeMo and HF decode Canary, under the checkpoint's
    # generation_config.json; set by __init__
    sampling_defaults: Mapping[str, object] = GREEDY_SAMPLING

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
        self.sampling_defaults = sampling_defaults(self.local_dir)
        tokenizer = json.loads((self.local_dir / "tokenizer.json").read_text())
        self._detokenizer = SentencePieceDetokenizer(tokenizer)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def checkpoint_path(self) -> str:
        return str(self.local_dir)

    # -------------------------------------------------------------------
    # state
    # -------------------------------------------------------------------

    def _max_encoder_frames(self) -> int:
        samples = int(self.MAX_AUDIO_SECONDS * self.config.sampling_rate)
        return self.config.encoder.output_lengths(samples // HOP_LENGTH)

    def declare_state(self) -> list[Array | Sampler]:
        """Decoder self-attention KV and a separate encoder-context KV: the
        self-attention resource plans a wrapper per label of the cache it
        names, so a shared cache would plan the context every step."""
        c = self.config
        kv = KVEntry(heads=c.num_heads, head_dim=c.head_dim)
        layers = frozenset(range(c.num_layers))
        capabilities = sampler_capabilities(self.sampling_defaults)
        return [
            # no RoPE: the counter's planned ids drive the sinusoid lookup
            Array(KV_CACHE, nodes={"decoder"}, layers=layers, entry=kv, max_len=c.max_position_embeddings,
                  positions=Positions(), reads={ATTN: Read(mask=Causal(), op=Attend(q_heads=c.num_heads))}),
            Array(CROSS_KV_CACHE, nodes={"decoder"}, layers=layers, entry=kv, max_len=self._max_encoder_frames(),
                  labels={CONTEXT_LABEL},
                  reads={CROSS_ATTN: Read(mask=Bidirectional(), op=Attend(q_heads=c.num_heads), queries=KV_CACHE)}),
            Sampler(
                SAMPLER, nodes={"decoder"}, vocab_size=c.vocab_size,
                penalty=capabilities["enable_repetition_penalty"], min_p=capabilities["enable_min_p"],
            ),
        ]

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        # each cache holds the request cap at full length: 8 pages per request
        # at the position limit, 4 for a 40.96 s clip's context
        requests = self.MAX_CONCURRENT_REQUESTS
        return {
            KV_CACHE: {"max_num_pages": kv_pages(self.config.max_position_embeddings, PAGE_TOKENS, requests)},
            CROSS_KV_CACHE: {"max_num_pages": kv_pages(self._max_encoder_frames(), PAGE_TOKENS, requests)},
        }

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        check_kv_pools_fit(self, config, config_path, logger)

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        del input_modalities, output_modalities
        config = sampling_config(self.sampling_defaults, model_kwargs)
        return {SAMPLER: SampleSettings(**{key: getattr(config, key) for key in SAMPLING_KEYS})}

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"language", "target_language", "punctuation", "max_output_tokens", *SAMPLING_KEYS})

    # -------------------------------------------------------------------
    # graph walks
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # The decoder's position table bounds prompt + transcript. A larger
        # request is refused by the conductor against the decode loop, not clamped.
        limit = self.config.max_position_embeddings - PROMPT_LENGTH
        return checked_request_kwargs(model_kwargs).get("max_output_tokens", limit)

    def get_max_output_tokens_limit(self) -> int:
        return self.get_max_output_tokens()

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
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
                    GraphEdge(next_node="decoder", name="text_inputs"),
                ],
            ),
            # the default request's iterations: past them a request overruns the position table
            max_iters=self.get_loop_caps()["decode_loop"],
            outputs=[],
        )
        return dict(prefill=prefill, decode=decode)

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        # the prefill hears the clip and samples the first token; decode feeds each one back
        return [
            WalkStep("prefill", step_metadata={FINAL_PREFILL: True}),
            WalkStep("decode", bind={"text_inputs": "new_token"}),
        ]

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
        kw = checked_request_kwargs(kwargs)
        target = kw.get("target_language")
        if target is not None and not isinstance(target, str):
            raise ValueError(f"target_language must be a string; got {target!r}")
        punctuation = kw.get("punctuation", True)
        if not isinstance(punctuation, bool):
            raise ValueError(f"punctuation must be true or false; got {punctuation!r}")
        source = kw.get("language") or "en"
        target = target or source
        prompt_ids = self.config.prompt_ids(source, target, punctuation)
        # the encoder's frames for this clip, as the encoder node counts them
        context = int(self.config.encoder.output_lengths(samples.shape[0] // HOP_LENGTH))
        state_layout = {
            KV_CACHE: {"main": Layout(
                spans=(Span("rows", "prefill", len(prompt_ids), len(prompt_ids), "text_inputs"),),
                growth=(("decode_loop", 1),),
            )},
            CROSS_KV_CACHE: {CONTEXT_LABEL: Layout(
                spans=(Span("rows", "prefill", context, context, "encoder_states"),),
            )},
        }
        return ProcessPromptOutput({
            "audio": [samples],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }, {"state_layout": state_layout})

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {
            KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))},
            CROSS_KV_CACHE: {CONTEXT_LABEL: KVLabel()},
        }

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

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

    def _weights(self, device: str, *prefixes: str):
        from mstar.model.loader.iterators import iter_safetensors_shards

        for prefix in prefixes:
            yield from iter_safetensors_shards(self.local_dir, device=device, prefix=prefix)

    def _create_submodule(self, node_name: str, device: str, autocast_dtype: torch.dtype | None):
        if node_name == "audio_encoder":
            from mstar.model.components.fastconformer import FastConformerEncoder
            from mstar.model.components.fastconformer_submodule import FastConformerEncoderSubmodule

            prefix = "model.encoder."
            encoder = FastConformerEncoder.load(
                self.config.encoder, ((k.removeprefix(prefix), v) for k, v in self._weights(device, prefix)),
                device=device, dtype=autocast_dtype,
            )
            return FastConformerEncoderSubmodule(encoder)
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
            return CanaryDecoderSubmodule(decoder.eval(), self.config, max_context=self._max_encoder_frames())
        return None


def _materialize(module: torch.nn.Module, device: str, dtype: torch.dtype | None) -> torch.nn.Module:
    """Allocate a meta-built module on ``device``; cast first, so storage is
    allocated in the target dtype rather than fp32 then down-cast."""
    if dtype is not None:
        module = module.to(dtype)
    return module.to_empty(device=device)
