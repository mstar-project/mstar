"""Glm5NextModel: Model implementation for GLM-5.3-Flash (text generation)."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from jinja2 import TemplateError

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.state import (
    Array,
    Causal,
    Entry,
    KimiDelta,
    LatentAttend,
    LatentEntry,
    Layout,
    Read,
    Sampler,
    SampleSettings,
    Span,
    TensorType,
    Value,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import KVLabel, Model, ProcessPromptOutput, WalkStep
from mstar.model.components.request_kwargs import (
    SAMPLING_KEYS,
    generation_config_dir,
    sampling_defaults,
)
from mstar.model.glm5_next.config import (
    ATTN,
    COUNT,
    KDA,
    KDA_STATE,
    KV_CACHE,
    MTP_SEED,
    PROMPT_LEN,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import (
    ByteLevelTextOutput,
    capture_ladder,
    check_kv_pools_fit,
    check_loaded,
    load_snapshot_tokenizer,
    process_weights_after_loading,
    read_fp8_block_quant_config,
    resolve_checkpoint,
)
from mstar.utils.generation_kwargs import normalize_generation_kwargs

logger = logging.getLogger(__name__)

# what zai-org/GLM-5.3-Flash's generation_config.json declares, for a checkpoint whose file is
# not at hand
DEFAULT_SAMPLING = MappingProxyType({"temperature": 1.0, "top_p": 0.95})
# The quantized checkpoint format this loader reads itself: fp8 block-scale (dense
# weights dequantized on load, routed experts fp8-resident).
QUANT_METHODS = ("fp8",)


class Glm5NextModel(ByteLevelTextOutput, Model):
    """GLM-5.3-Flash: 320B hybrid KDA/MLA MoE causal LM, text in / text out."""

    DECODE_LOOP = "decode_loop"
    # the conductor builds the fused-MoE align op before spawning the workers
    prebuild_fused_moe = True
    INTERNAL_KWARGS = (PROMPT_LEN,)
    # the checkpoint's sampling defaults, under whatever a request sets (set by
    # `__init__`; a model built without it takes the published recipe)
    sampling_defaults: Mapping[str, object] = DEFAULT_SAMPLING

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        checkpoint_path = kwargs.get("checkpoint_path")
        self.model_path_hf = checkpoint_path or model_path_hf
        self._config_variant = kwargs.get("config_variant", "full")
        if self._config_variant == "reduced":
            self.config = Glm5NextModelConfig.reduced()
        elif self._config_variant == "reduced_fp8":
            self.config = Glm5NextModelConfig.reduced_fp8()
        else:
            self.config = Glm5NextModelConfig()
        if kwargs.get("dsa_long_context", False):
            # rather than serve a long context densely
            raise NotImplementedError(
                "dsa_long_context needs the k-pool DSA path, which this build does not serve; "
                "serve within index_topk (2048), where dense MLA is exact DSA"
            )
        if "moe_quant_kernel" in kwargs:
            self.config.moe_quant_kernel = str(kwargs["moe_quant_kernel"])
        if "mtp_num_draft_tokens" in kwargs:
            k = kwargs["mtp_num_draft_tokens"]
            # a prompt leaves two verify blocks of the context free: k must leave room for both
            if (isinstance(k, bool) or not isinstance(k, int) or k < 0
                    or self.config.context_limit - 2 * (k + 1) < 1):
                raise ValueError(
                    f"mtp_num_draft_tokens must be an integer from 0 (off) to "
                    f"{(self.config.context_limit - 3) // 2}, got {k!r}")
            self.config.mtp_num_draft_tokens = k
        if "prefill_graphs" in kwargs:
            # bool("false") is True
            if not isinstance(kwargs["prefill_graphs"], bool):
                raise ValueError(f"prefill_graphs must be true or false, got {kwargs['prefill_graphs']!r}")
            self.config.prefill_graphs = kwargs["prefill_graphs"]
        # an override of the submodule's batch sizes, which **kwargs would otherwise swallow
        if kwargs.get("prefill_capture_batch_sizes") is not None:
            self.config.prefill_capture_batch_sizes = capture_ladder(
                "prefill_capture_batch_sizes", kwargs["prefill_capture_batch_sizes"])
        # Requests that may hold KDA state at once (the decode batch cap). The
        # pool holds one more slot, its sink; a serve YAML sets the pool's
        # size, sink included, under ``resources: kda_state: {max_slots: N}``.
        self.kda_max_requests = int(kwargs.get("kda_max_requests", 32))
        # the KDA conv tail's dtype, which must match the projections' (bf16 at serve; CPU
        # tests pass fp32)
        self.kda_conv_dtype = kwargs.get("kda_conv_dtype", torch.bfloat16)
        # "byte" maps UTF-8 bytes to token ids for reduced serve (no HF IO).
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self.sampling_defaults = sampling_defaults(
            generation_config_dir(self.model_path_hf, cache_dir), DEFAULT_SAMPLING)

    @property
    def tokenizer(self):
        # loaded on first use, so byte mode never touches HF
        if self._tokenizer is None:
            self._tokenizer = load_snapshot_tokenizer(self.model_path_hf, self.cache_dir)
        return self._tokenizer

    # -------------------------------------------------------------------
    # Model ABC: state
    # -------------------------------------------------------------------

    def declare_state(self) -> list[Array | Sampler]:
        cfg, k = self.config, self.config.mtp_num_draft_tokens
        nodes = frozenset({"LLM"})
        h, d = cfg.linear_num_heads, cfg.linear_head_dim
        state = [
            Array(
                KV_CACHE, nodes=nodes, layers={*cfg.full_attn_layer_indices, *([cfg.num_hidden_layers] if k else [])},
                entry=LatentEntry(rank=cfg.kv_lora_rank, rope_dim=0), max_len=cfg.context_limit, verify_span=k + 1,
                reads={ATTN: Read(mask=Causal(), op=LatentAttend(cfg.num_attention_heads, scale=cfg.softmax_scale))},
            ),
            Value(
                KDA_STATE, nodes=nodes, layers=cfg.kda_layer_indices, verify_span=k + 1,
                entry=Entry({
                    "conv": TensorType(
                        (3 * h * d, cfg.linear_conv_kernel_size - 1), dtype=self.kda_conv_dtype, shard=0,
                    ),
                    "state": TensorType((h, d, d), dtype=torch.float32, shard=0),
                }),
                updates={KDA: KimiDelta(lower_bound=float(cfg.gate_lower_bound))},
            ),
            # min-p only where the checkpoint defaults it; elsewhere a request's min_p > 0 is refused at admission
            Sampler(
                SAMPLER, nodes=nodes, vocab_size=cfg.vocab_size, penalty=True,
                min_p=bool(self.sampling_defaults.get("min_p")),
            ),
        ]
        if k:
            state.append(Value(
                MTP_SEED, nodes=nodes, layers={0},
                entry=Entry({
                    "hidden": TensorType((cfg.hidden_size,), dtype=self.kda_conv_dtype),
                    "draft": TensorType((), dtype=torch.long),
                }),
            ))
        return state

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        # every request holding KDA state takes a slot, and each pool keeps one more, its sink
        slots = {"max_slots": self.kda_max_requests + 1}
        if self.config.mtp_num_draft_tokens:
            return {KDA_STATE: slots, MTP_SEED: slots}
        return {KDA_STATE: slots}

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        check_kv_pools_fit(self, config, config_path, logger)

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        del input_modalities, output_modalities
        model_kwargs = normalize_generation_kwargs(model_kwargs)
        params = {
            **self.sampling_defaults,
            **{k: model_kwargs[k] for k in SAMPLING_KEYS if k in model_kwargs},
        }
        if self.config.mtp_num_draft_tokens > 0 and "temperature" not in model_kwargs:
            # MTP verifies greedily: a request that names no temperature gets 0, and an
            # explicit one > 0 is refused in process_prompt
            params["temperature"] = 0.0
        return {SAMPLER: SampleSettings(**params)}

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # a verify step's tokens leave on new_token like a decode step's
        prefill_outputs = [
            GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="new_token",
                output_modality="text",
                conductor_new_token=True,
                persist=True,
            ),
        ]
        decode_inputs = ["text_inputs"]
        decode_outputs = [
            GraphEdge(
                next_node="LLM",
                name="text_inputs",
            ),
            GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="new_token",
                output_modality="text",
            ),
        ]
        if self.config.mtp_num_draft_tokens > 0:
            prefill_outputs.append(GraphEdge(next_node=EMPTY_DESTINATION, name=COUNT, persist=True))
            decode_inputs.append(COUNT)
            decode_outputs.append(GraphEdge(next_node="LLM", name=COUNT))
        prefill = GraphNode(name="LLM", input_names=["text_inputs"], outputs=prefill_outputs)

        decode = Loop(
            name="decode_loop",
            section=GraphNode(name="LLM", input_names=decode_inputs, outputs=decode_outputs),
            # Runaway guard; the per-request budget lives in check_stop.
            max_iters=self.max_decode_steps(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: walk plan

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        # one prefill over the prompt, then decode until the stop check ends it
        return [
            WalkStep("prefill", step_metadata={FINAL_PREFILL: True}),
            WalkStep("decode", bind={"text_inputs": "new_token"},
                     step_metadata={PROMPT_LEN: model_kwargs[PROMPT_LEN]}),
        ]

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
    ) -> NameToTensorList | ProcessPromptOutput:
        if prompt is None:
            return {}
        # the conductor reads these kwargs; read here first, a bad one is the
        # client's 400 before the request reaches any rank
        kwargs = normalize_generation_kwargs(kwargs)
        sampling = self.get_request_resource_configs([], [], kwargs)[SAMPLER]
        self.get_max_output_tokens(**kwargs)
        if kwargs.get("return_logprobs"):
            # the sampler configs carry none, which the conductor refuses too
            raise ValueError("GLM-5.3 does not report logprobs; leave return_logprobs unset")

        if self.config.mtp_num_draft_tokens > 0:
            # MTP verifies by argmax; refused here, it is the client's 400 rather than a
            # worker's 500
            if sampling.temperature != 0 or sampling.repetition_penalty != 1:
                raise ValueError(
                    f"MTP drafting is greedy only (temperature={sampling.temperature}, "
                    f"repetition_penalty={sampling.repetition_penalty}); serve with mtp_num_draft_tokens: 0"
                )

        if self._tokenizer_mode == "byte":
            # Reduced serve maps UTF-8 bytes directly to token ids, avoiding HF IO.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            input_ids = torch.tensor(byte_ids, dtype=torch.long)
        elif getattr(self.tokenizer, "chat_template", None):
            input_ids = self._render_chat(prompt, kwargs.get("prompt_parts"))
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        # Decode runs at least one step, and one more may already be scheduled
        # when it stops, so the prompt leaves two steps' tokens of the context
        # free (an MTP step keeps k + 1). A ValueError here reaches the client
        # as a 400.
        limit = self.config.context_limit - 2 * (self.config.mtp_num_draft_tokens + 1)
        if input_ids.numel() > limit:
            raise ValueError(
                f"prompt is {input_ids.numel()} tokens; GLM-5.3-Flash is served "
                f"with at most {limit}"
            )
        n = input_ids.numel()
        block = self.config.mtp_num_draft_tokens + 1
        budget = kwargs.get("max_output_tokens")
        if block > 1 and budget is not None:
            # MTP decode stops two blocks short of the window (_mtp_check_stop), so a max_tokens
            # past that is refused here rather than cut short
            room = self.config.context_limit - 2 * block + 1 - n
            if budget > room:
                raise ValueError(
                    f"max_tokens={budget} after a {n}-token prompt: GLM-5.3-Flash with MTP "
                    f"emits at most {room} tokens after it"
                )
        prompt = Span("ids", "prefill", n, n, "text_inputs")
        layouts = {
            KV_CACHE: {"main": Layout(spans=(prompt,), growth=(("decode_loop", block),))},
            KDA_STATE: {"main": Layout()},
        }
        if self.config.mtp_num_draft_tokens:
            layouts[MTP_SEED] = {"main": Layout()}
        return ProcessPromptOutput(
            {"text_inputs": [input_ids.to(torch.long)]}, {"state_layout": layouts, PROMPT_LEN: n},
        )

    def _render_chat(self, prompt: str, prompt_parts: list[PromptPart] | None) -> torch.Tensor:
        """The chat's turns, each in its own role, through the checkpoint's template, which
        adds ``[gMASK]<sop>`` and opens the assistant turn. An entrypoint with no messages
        sends the prompt as one user turn."""
        parts = prompt_parts or [PromptPart(modality=TEXT, text=prompt, role="user")]
        messages = [
            {"role": turn.role,
             "content": "\n".join(p.text or "" for p in turn.parts if p.modality == TEXT)}
            for turn in chat_turns(parts)
        ]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                # transformers 5.x defaults return_dict=True (a BatchEncoding);
                # keep the bare-tensor return so [0] selects the row
                return_dict=False,
            )
        except TemplateError as exc:
            # a chat the template refuses is the client's to fix: a 400, not a 500
            raise ValueError(f"GLM-5.3's chat template refused the chat: {exc}") from exc
        return input_ids[0]

    def max_decode_steps(self) -> int:
        """Decode iterations a one-token prompt can run before its context
        reaches the limit; check_stop stops every request before it."""
        return self.config.context_limit - 1

    def get_max_output_tokens(self, **model_kwargs):
        # A request that names none runs as far as the decode loop can, and the conductor fits
        # it to the room its prompt leaves. One that names more than the context, which a
        # one-token prompt's window holds, is refused by the conductor: its decode loop cap is
        # past max_iters.
        budget = normalize_generation_kwargs(model_kwargs).get("max_output_tokens", self.config.max_output_tokens)
        if budget is None:
            return self.config.context_limit
        return budget

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"max_output_tokens", *SAMPLING_KEYS})

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
            return super().postprocess(output, modality, **kwargs)
        raise ValueError(f"Unsupported modality for GLM-5.3-Flash: {modality!r}")

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
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self,
        node_name: str,
        device: str,
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name != "LLM":
            return None

        source = self._resolve_checkpoint()
        if source is None:
            logger.info(
                "Glm5NextModel: no checkpoint resolved for node %r — dummy "
                "mode (None).", node_name,
            )
            return None

        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
        from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule

        # built on meta, so nothing is allocated before to_empty; load_weights re-widens the
        # fp32 params (restore_fp32_params)
        with torch.device("meta"):
            language_model = Glm5NextForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        self._load_checkpoint(language_model, source, device, tp_group)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()

        logger.info("Successfully loaded GLM-5.3-Flash submodule for %s", node_name)
        return Glm5NextLLMSubmodule(
            language_model=language_model, config=self.config)

    def _load_checkpoint(self, language_model, source: str, device, tp_group) -> None:
        """Load weights, taking the sliced TP fast read path when possible."""
        from mstar.model.glm5_next.weight_loader import build_glm5_next_read_plan
        from mstar.model.loader import load_weights
        from mstar.model.loader.iterators import iter_safetensors_shards

        index_file = Path(source) / "model.safetensors.index.json"
        if not index_file.is_file():
            check_loaded(
                language_model, load_weights(language_model, source, device=device, quant_methods=QUANT_METHODS),
                source,
            )
            return

        with open(index_file) as f:
            checkpoint_keys = list(json.load(f)["weight_map"])
        tp_rank = tp_group.rank if tp_group is not None else 0
        tp_size = tp_group.world_size if tp_group is not None else 1
        keys, specs = build_glm5_next_read_plan(
            checkpoint_keys, self.config, tp_rank, tp_size,
            load_mtp=language_model.mtp is not None,
        )
        logger.info(
            "Glm5NextModel fast read plan: %d/%d keys, %d sliced (tp %d/%d)",
            len(keys), len(checkpoint_keys), len(specs), tp_rank, tp_size,
        )
        weights = iter_safetensors_shards(
            source, device=device, keys=keys, slice_spec=specs.get, quant_methods=QUANT_METHODS,
        )
        check_loaded(language_model, language_model.load_weights(weights), source)

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        """Adopt the checkpoint's fp8-block quant config from its config.json."""
        if self.config.quantization_config is not None:
            return
        quant = read_fp8_block_quant_config(source)
        if quant is not None:
            logger.info(
                "Glm5NextModel: fp8 %s checkpoint (block %s) — dense dequant on "
                "load, routed experts fp8-resident=%s.",
                quant.fmt, quant.weight_block_size, self.config.moe_fp8_resident,
            )
            self.config.quantization_config = quant

    def _resolve_checkpoint(self) -> str | None:
        return resolve_checkpoint(self.model_path_hf, self.cache_dir)
