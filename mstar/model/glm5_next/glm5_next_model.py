"""Glm5NextModel: Model implementation for GLM-5.3-Flash (text generation)."""

import json
import logging
from pathlib import Path

import torch

from mstar.api_server.openai.glm_chat import TEMPLATE_KWARGS
from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    DSAConfig,
    DSASpec,
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.engine.resources.dsa.config import INDEX_PLANES
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span
from mstar.model.glm5_next.config import (
    ATTN,
    DSA,
    KDA,
    KDA_STATE,
    KV_CACHE,
    MTP_DRAFT,
    MTP_HIDDEN,
    MTP_LEN,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.multimodal import render_chat_template
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteIdDetokenizer, ByteLevelDetokenizer, TextOutput
from mstar.utils.generation_kwargs import normalize_generation_kwargs
from mstar.utils.hf_snapshot import resolve_snapshot_dir

logger = logging.getLogger(__name__)


def process_weights_after_loading(root: torch.nn.Module, device: torch.device) -> None:
    """Finalize kernel layouts across a freshly-loaded module tree."""
    for module in root.modules():
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook) and module is not root:
            hook(device)


def _check_all_loaded(module: torch.nn.Module, loaded: set[str], source: str) -> None:
    """Refuse to serve a parameter no checkpoint tensor reached: the module was
    built on meta and materialized with ``to_empty``, so it holds garbage."""
    missing = {name for name, _ in module.named_parameters()} - loaded
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameters got no tensor from {source} (e.g. "
            f"{sorted(missing)[:3]}); the snapshot is partial or a key was renamed"
        )


class Glm5NextModel(Model):
    """GLM-5.3-Flash: 320B hybrid KDA/MLA MoE causal LM, text in / text out."""

    # the conductor builds the fused-MoE align op before spawning the workers
    prebuild_fused_moe = True
    # read only by process_prompt's template, and kept off the conductor's wire,
    # which cannot carry every number a tool schema may hold (2**64)
    PREPROCESS_ONLY_KWARGS = ("messages", "tools", "chat_template_kwargs")

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
        elif self._config_variant == "reduced_long_context":
            self.config = Glm5NextModelConfig.reduced_long_context()
        else:
            self.config = Glm5NextModelConfig()
        if "moe_quant_kernel" in kwargs:
            self.config.moe_quant_kernel = str(kwargs["moe_quant_kernel"])
        if "mtp_num_draft_tokens" in kwargs:
            self.config.mtp_num_draft_tokens = int(kwargs["mtp_num_draft_tokens"])
        if "prefill_graphs" in kwargs:
            self.config.prefill_graphs = bool(kwargs["prefill_graphs"])
        # overrides of the submodule's buckets, which **kwargs would otherwise swallow
        for key in ("prefill_token_buckets", "prefill_capture_batch_sizes"):
            if kwargs.get(key) is not None:
                setattr(self.config, key, [int(n) for n in kwargs[key]])
        long_only = sorted(k for k in ("max_seq_len", "prefill_window_tokens", "dsa_shard_prefill")
                           if k in kwargs)
        if long_only and not kwargs.get("dsa_long_context", False):
            # they size and split the long-context path only; without it they would be dropped
            raise ValueError(f"{long_only} apply only with dsa_long_context")
        if kwargs.get("dsa_long_context", False):
            # Contexts past index_topk through the k-pool indexer and sparse MLA (the DSA resource).
            # Captured prefill buckets take whole short prompts; a longer prompt runs eager, in
            # token windows.
            self.config.dsa_long_context = True
            self.config.max_seq_len = int(kwargs.get("max_seq_len", self.config.max_seq_len))
            self.config.prefill_window_tokens = int(
                kwargs.get("prefill_window_tokens", self.config.prefill_window_tokens))
            self.config.dsa_shard_prefill = bool(kwargs.get("dsa_shard_prefill", False))
        self.config.check_long_context()
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

    @property
    def tokenizer(self):
        # loaded on first use, so byte mode never touches HF
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer_source = str(resolve_snapshot_dir(self.model_path_hf, cache_dir=self.cache_dir))
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source, cache_dir=self.cache_dir,
                )
            except ValueError:
                # transformers-5 tokenizer_config declares a backend class
                # transformers 4.x cannot construct; tokenizer.json is
                # version-independent, so build the fast tokenizer from it.
                self._tokenizer = self._fast_tokenizer_fallback(tokenizer_source)
        return self._tokenizer

    @staticmethod
    def _fast_tokenizer_fallback(source: str):
        from transformers import PreTrainedTokenizerFast

        snap = Path(source)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(snap / "tokenizer.json"))
        template = snap / "chat_template.jinja"
        if template.is_file():
            tokenizer.chat_template = template.read_text()
        # built from tokenizer.json alone, the tokenizer has no special tokens, so the
        # detokenizer would print <|user|> and the other turn markers into replies
        config = snap / "tokenizer_config.json"
        if config.is_file():
            cfg = json.loads(config.read_text())
            special = {k: cfg[k] for k in ("eos_token", "pad_token") if cfg.get(k)}
            tokenizer.add_special_tokens(
                {**special, "additional_special_tokens": list(cfg.get("extra_special_tokens") or [])})
        return tokenizer

    # -------------------------------------------------------------------
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The LLM node's five resources."""
        num_kv_layers = self.config.num_kv_planes
        kda_blocks = DeltaNetGeometry(
            num_k_heads=self.config.linear_num_heads,
            num_v_heads=self.config.linear_num_heads,
            head_k_dim=self.config.linear_head_dim,
            head_v_dim=self.config.linear_head_dim,
            conv_kernel_size=self.config.linear_conv_kernel_size,
        ).to_blocks(
            conv_dtype=self.kda_conv_dtype,
            # an MTP decode step verifies k + 1 tokens per request
            speculative_tokens=self.config.mtp_num_draft_tokens,
        )
        if self.config.mtp_num_draft_tokens > 0:
            # the next step's draft seed, in the model's dtype (the KDA projections')
            kda_blocks[MTP_HIDDEN] = RecurrentBlockConfig(
                shape=(self.config.hidden_size,), dtype=self.kda_conv_dtype, per_layer=False,
            )
            kda_blocks[MTP_DRAFT] = RecurrentBlockConfig(
                shape=(1,), dtype=torch.long, per_layer=False,
            )
        if self.config.mtp_trims:
            # each request's tokens in cache, which every verify step advances on device
            kda_blocks[MTP_LEN] = RecurrentBlockConfig(
                shape=(1,), dtype=torch.long, per_layer=False,
            )
        specs = [
            KVSpec(
                resource_key=KV_CACHE,
                nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=num_kv_layers,
                    num_kv_heads=1,
                    head_dim=self.config.kv_lora_rank + self.config.mla_cache_kpe,
                    max_seq_len=self.config.kv_rows,
                    num_qo_heads=self.config.num_attention_heads,
                    layout=KVLayout.MLA,
                    kv_lora_rank=self.config.kv_lora_rank,
                    qk_rope_head_dim=self.config.mla_cache_kpe,
                    aux_planes=(
                        {INDEX_PLANES: self.config.index_planes}
                        if self.config.dsa_long_context else {}
                    ),
                ),
            ),
            AttentionSpec(
                resource_key=ATTN,
                nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE,
                    backend=AttnBackend.FLASHINFER_MLA,
                    sm_scale=self.config.qk_head_dim ** -0.5,
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={"LLM"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
            RecurrentStateSpec(
                resource_key=KDA_STATE,
                nodes={"LLM"},
                config=RecurrentStateConfig(
                    num_layers=len(self.config.kda_layer_indices),
                    blocks=kda_blocks,
                    max_slots=self.kda_max_requests + 1,  # + the sink
                ),
            ),
            LinearAttnSpec(
                resource_key=KDA,
                nodes={"LLM"},
                config=LinearAttnConfig(
                    recurrent_state=KDA_STATE,
                    variant=LinearAttnVariant.KDA,
                    backend=LinearAttnBackend.TRITON,
                    gate_lower_bound=self.config.gate_lower_bound,
                ),
            ),
        ]
        if self.config.dsa_long_context:
            # past index_topk: the k-pool indexer's keys in KV_CACHE's index planes
            specs.append(DSASpec(
                resource_key=DSA,
                nodes={"LLM"},
                config=DSAConfig(
                    kv_cache=KV_CACHE,
                    topk=self.config.index_topk,
                    kpool=self.config.index_kpool,
                    sm_scale=self.config.qk_head_dim ** -0.5,
                ),
            ))
        return specs

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = normalize_generation_kwargs(model_kwargs)
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        params = {
            k: model_kwargs.get(k, getattr(self.config, k))
            for k in keys
        }
        if "top_k" in model_kwargs:
            params["top_k"] = int(model_kwargs["top_k"])
        if self.config.mtp_num_draft_tokens > 0 and "temperature" not in model_kwargs:
            # MTP verifies greedily: a request that names no temperature gets 0, and an
            # explicit one > 0 is refused in process_prompt
            params["temperature"] = 0.0
        return {SAMPLER: SamplingReqConfig(**params)}

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # MTP keeps this graph: the draft seed lives in the request's KDA slot, and a
        # verify step's tokens leave on new_token like a decode step's
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    conductor_new_token=True,
                    persist=True,
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
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                ],
            ),
            # Runaway guard; the per-request budget lives in check_stop.
            max_iters=self.max_decode_steps(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: conductor state machine (prefill -> decode -> done)

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

        graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
        graph_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [graph_edge]
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
        incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # The decode Loop ran to EOS (submodule check_stop) or to
            # max_iters; either way the request is finished.
            request_done = True
            metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        # seed decode from the persisted emitted token
        graph_edge = GraphEdge(next_node="LLM", name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        unpersist_tensors = list(graph_edge.tensor_info)
        inputs = [graph_edge]

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
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
    ) -> NameToTensorList | ProcessPromptOutput:
        if prompt is None:
            return {}
        # the conductor reads these kwargs; read here first, a bad one is the
        # client's 400 before the request reaches any rank
        kwargs = normalize_generation_kwargs(kwargs)
        self.get_request_resource_configs({}, kwargs)
        self.get_max_output_tokens(**kwargs)
        if kwargs.get("min_p", 0) > 0:
            # the node's sampler is built without min-p, and the request configs do not pass it on
            raise ValueError(f"GLM-5.3 is served without min_p (min_p={kwargs['min_p']}); send none")

        if self.config.mtp_num_draft_tokens > 0:
            # MTP verifies by argmax; refused here, it is the client's 400 rather than a
            # worker's 500
            temperature = float(kwargs.get("temperature", 0.0))
            penalty = float(kwargs.get("repetition_penalty", self.config.repetition_penalty))
            if temperature > 0.0 or penalty != 1.0:
                raise ValueError(
                    f"MTP drafting is greedy only (temperature={temperature}, "
                    f"repetition_penalty={penalty}); serve with mtp_num_draft_tokens: 0"
                )

        if self._tokenizer_mode == "byte":
            # Reduced serve maps UTF-8 bytes directly to token ids, avoiding HF IO.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            input_ids = torch.tensor(byte_ids, dtype=torch.long)
        elif getattr(self.tokenizer, "chat_template", None):
            input_ids = render_chat_template(
                self.tokenizer, prompt, kwargs.get("prompt_parts"), kwargs, TEMPLATE_KWARGS)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        # Decode runs at least one step, and one more may already be scheduled
        # when it stops, so the prompt leaves two steps' tokens of the context
        # limit free (an MTP step keeps k + 1). A ValueError here reaches the
        # client as a 400.
        limit = self.config.context_limit - 2 * (self.config.mtp_num_draft_tokens + 1)
        if input_ids.numel() > limit:
            raise ValueError(
                f"prompt is {input_ids.numel()} tokens; GLM-5.3-Flash is served "
                f"with at most {limit}"
            )
        n = input_ids.numel()
        # a decode step writes one row, or under MTP its whole verify block (the rejected
        # drafts stay as holes), so the KV pool sizes a request at its worst case
        block = self.config.mtp_num_draft_tokens + 1
        budget = kwargs.get("max_output_tokens")
        if block > 1 and budget is not None:
            # MTP decode stops two blocks short of the window (_mtp_check_stop), so a max_tokens
            # past that is refused here rather than cut short
            room = self.config.index_topk - 2 * block + 1 - n
            if budget > room:
                raise ValueError(
                    f"max_tokens={budget} after a {n}-token prompt: GLM-5.3-Flash with MTP "
                    f"emits at most {room} tokens after it"
                )
        layout = LabelLayout(spans=(Span("ids", "prefill", n, n, "text_inputs"),), growth=(("decode_loop", block),))
        return ProcessPromptOutput(
            {"text_inputs": [input_ids.to(torch.long)]}, {"kv_layout": {KV_CACHE: {"main": layout}}},
        )

    def max_decode_steps(self) -> int:
        """Decode iterations a one-token prompt can run before its context
        reaches the context limit or its cache rows reach kv_rows (an MTP step
        stores k + 1); check_stop stops every request before either."""
        cfg = self.config
        return min(cfg.context_limit - 1, (cfg.kv_rows - 1) // (cfg.mtp_num_draft_tokens + 1))

    def get_max_output_tokens(self, **model_kwargs):
        # held to the window: a one-token prompt emits at most context_limit tokens. A request
        # that names none runs as far as the decode loop can (under MTP the cache rows bind
        # first), and the conductor fits it to the room its prompt leaves.
        budget = normalize_generation_kwargs(model_kwargs).get("max_output_tokens", self.config.max_output_tokens)
        if budget is None:
            return min(self.config.context_limit, self.max_decode_steps() + 1)
        return min(budget, self.config.context_limit)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill emits the first of max_output_tokens, so decode runs one fewer; the
        # conductor refuses a cap past the loop's max_iters
        return {"decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    # built on first use: the tokenizer it reads loads lazily
    _text_output: TextOutput | None = None

    def text_output(self) -> TextOutput:
        if self._text_output is None:
            # a reduced model's byte ids never touch the HF tokenizer
            detokenizer = (
                ByteIdDetokenizer() if self._tokenizer_mode == "byte" else ByteLevelDetokenizer(self.tokenizer)
            )
            self._text_output = TextOutput(detokenizer, frozenset(self.config.eos_token_ids))
        return self._text_output

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
        sp_group=None,
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
        # The declared conv-tail dtype must match the loaded projections.
        self.kda_conv_dtype = language_model.kda_conv_dtype()

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
            loaded = load_weights(language_model, source, device=device)
            _check_all_loaded(language_model, loaded, source)
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
            source, device=device, keys=keys, slice_spec=specs.get,
        )
        loaded = language_model.load_weights(weights)
        _check_all_loaded(language_model, loaded, source)

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        """Adopt the checkpoint's fp8-block quant config from its config.json."""
        from mstar.utils.fp8_block import Fp8BlockQuantConfig

        if self.config.quantization_config is not None:
            return
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            return
        try:
            with open(config_json) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:  # unreadable / malformed — stay bf16
            logger.warning("Glm5NextModel: could not read %s: %s", config_json, e)
            return
        quant = Fp8BlockQuantConfig.from_hf_config_dict(
            raw.get("quantization_config"),
        )
        if quant is not None:
            logger.info(
                "Glm5NextModel: fp8 %s checkpoint (block %s) — dense dequant on "
                "load, routed experts fp8-resident=%s.",
                quant.fmt, quant.weight_block_size, self.config.moe_fp8_resident,
            )
            self.config.quantization_config = quant

    def _resolve_checkpoint(self) -> str | None:
        path = getattr(self, "model_path_hf", None)
        if not path:
            return None
        if Path(path).exists():
            return str(path)
        return str(resolve_snapshot_dir(path, cache_dir=self.cache_dir))
