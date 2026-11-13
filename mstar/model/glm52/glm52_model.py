"""Glm52Model: Model implementation for GLM-5.2 (text generation)."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import torch
from jinja2 import TemplateError

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.state import (
    Array,
    Attend,
    Causal,
    Entry,
    KVEntry,
    LatentAttend,
    LatentEntry,
    Layout,
    Positions,
    Read,
    Rope,
    Sampler,
    SampleSettings,
    Selected,
    Span,
    TensorType,
    TopK,
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
from mstar.model.glm52.components.indexer import is_full_indexer_layer
from mstar.model.glm52.config import (
    ATTN,
    INDEX_KEYS,
    KERNELS,
    KV,
    MTP_SEED,
    PROMPT_LEN,
    SAMPLER,
    TOPK,
    Glm52ModelConfig,
)
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import (
    ByteLevelTextOutput,
    capture_ladder,
    check_kv_pools_fit,
    load_snapshot_tokenizer,
    read_fp8_block_quant_config,
    resolve_checkpoint,
)
from mstar.utils.generation_kwargs import normalize_generation_kwargs

logger = logging.getLogger(__name__)

# kernels: auto fuses all-reduces up to this size into the next add + RMSNorm; larger ones
# reduce, add and norm apart
AR_FUSION_MAX_BYTES = 1 << 20
# model_kwargs that kernels and the checkpoint's quantization replaced, refused rather than ignored
_RETIRED_KWARGS = (
    "moe_quant_kernel", "moe_decode_kernel", "moe_router_kernel", "moe_prefill_kernel",
    "moe_fused_allreduce", "dense_fp8", "mla_fused_prep", "fused_add_rmsnorm",
    "prefill_last_layer_rows")
# The quantized checkpoint formats GLM-5.2's loader reads itself: fp8 block-scale
# (weight_loader.py dequantizes the dense weights and keeps the routed experts fp8).
QUANT_METHODS = ("fp8",)

# what zai-org/GLM-5.2's generation_config.json declares, for a checkpoint whose file is not at hand
DEFAULT_SAMPLING = MappingProxyType({"temperature": 1.0, "top_p": 0.95})


class Glm52Model(ByteLevelTextOutput, Model):
    """GLM-5.2: 753B MoE causal LM (MLA + DSA), text in / text out."""

    DECODE_LOOP = "decode_loop"
    INTERNAL_KWARGS = (PROMPT_LEN,)

    # the conductor builds the fused-MoE align op before spawning the workers
    prebuild_fused_moe = True
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
            self.config = Glm52ModelConfig.reduced()
        elif self._config_variant == "reduced_fp8":
            self.config = Glm52ModelConfig.reduced_fp8()
        else:
            self.config = self._checkpoint_config()
        if "num_hidden_layers" in kwargs:
            # serve the first N layers of a deeper checkpoint
            self.config.num_hidden_layers = int(kwargs["num_hidden_layers"])
        if kwargs.get("dsa_long_context", False):
            # Opt-in DSA engine path (configs/glm52_tp8_longctx.yaml). The
            # sparse gather reads the paged latent cache, so the naive
            # (mla_absorb=False) backend cannot host it.
            if not self.config.mla_absorb:
                raise ValueError(
                    "dsa_long_context requires mla_absorb: the sparse path "
                    "gathers selected latents from the paged MLA cache"
                )
            self.config.dsa_long_context = True
            # Guard + KV sizing move from index_topk to the serving window.
            self.config.max_seq_len = int(kwargs.get("max_seq_len", 8192))
        retired = sorted(k for k in _RETIRED_KWARGS if k in kwargs)
        if retired:
            # a config that still names one would otherwise serve without what it asks for
            raise ValueError(
                f"{retired}: GLM-5.2 has no such switch; model_kwargs.kernels picks its kernels "
                f"({' or '.join(KERNELS)}) and the checkpoint its fp8 linears")
        if "kernels" in kwargs:
            if kwargs["kernels"] not in KERNELS:
                raise ValueError(f"kernels must be one of {KERNELS}, got {kwargs['kernels']!r}")
            self.config.kernels = kwargs["kernels"]
        if "graph_compile" in kwargs:
            # bool("false") is True
            if not isinstance(kwargs["graph_compile"], bool):
                raise ValueError(f"graph_compile must be true or false, got {kwargs['graph_compile']!r}")
            self.config.graph_compile = kwargs["graph_compile"]
        if "prefill_capture_batch_sizes" in kwargs:
            self.config.prefill_capture_batch_sizes = capture_ladder(
                "prefill_capture_batch_sizes", kwargs["prefill_capture_batch_sizes"])
        if "mtp_num_draft_tokens" in kwargs:
            k = kwargs["mtp_num_draft_tokens"]
            if isinstance(k, bool) or not isinstance(k, int) or k < 0:
                raise ValueError(f"mtp_num_draft_tokens must be an integer, 0 (off) or more, got {k!r}")
            self.config.mtp_num_draft_tokens = k
        if self.config.mtp_num_draft_tokens > 0:
            self._check_mtp(kwargs)
        if self.config.dsa_long_context and self.config.rope_interleave != self.config.indexer_rope_interleave:
            # only long context builds the indexer, and its keys rotate through the latent array's Rope
            raise ValueError(
                f"rope_interleave={self.config.rope_interleave} and indexer_rope_interleave="
                f"{self.config.indexer_rope_interleave}: GLM-5.2's indexer rotates as its MLA does")
        if self.config.dsa_long_context and not is_full_indexer_layer(self.config, 0):
            # a SHARED layer takes the pick of the FULL layer before it
            raise ValueError(
                "dsa_long_context needs layer 0 to be a FULL indexer layer, and the index schedule "
                f"(index_skip_topk_offset={self.config.index_skip_topk_offset}, index_topk_freq="
                f"{self.config.index_topk_freq}) makes it SHARED")
        # "byte" maps UTF-8 bytes to token ids for reduced serve (no HF IO).
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self.sampling_defaults = sampling_defaults(
            generation_config_dir(self.model_path_hf, cache_dir), DEFAULT_SAMPLING)

    def _check_mtp(self, kwargs: dict) -> None:
        if self._config_variant in ("reduced", "reduced_fp8"):
            # reduced() has 2 trunk layers, which puts the MTP layer on a SHARED indexer
            # slot; grow the trunk so it lands FULL, as layer 78 does in the real geometry
            if not is_full_indexer_layer(self.config, self.config.num_hidden_layers):
                self.config.num_hidden_layers = (
                    self.config.index_skip_topk_offset - 1 + self.config.index_topk_freq
                )
        elif "num_hidden_layers" in kwargs:
            # every checkpoint layer past the trunk routes to the draft module
            raise ValueError(
                "mtp_num_draft_tokens needs the checkpoint's own trunk depth; "
                "with num_hidden_layers set, trunk layers would load as the MTP layer"
            )
        if self.config.max_prompt_tokens < 1:
            # two verify blocks of k + 1 rows stay free under the context limit
            raise ValueError(
                f"mtp_num_draft_tokens={self.config.mtp_num_draft_tokens} leaves no room for a "
                f"prompt in the {self.context_limit()}-token context"
            )

    def _checkpoint_config(self) -> Glm52ModelConfig:
        """A local checkpoint's config.json geometry, else the official one."""
        if not self.model_path_hf:
            return Glm52ModelConfig()
        config_json = Path(self.model_path_hf) / "config.json"
        if not config_json.is_file():
            return Glm52ModelConfig()
        with open(config_json) as f:
            return Glm52ModelConfig.from_hf_config(json.load(f))

    @property
    def tokenizer(self):
        # Lazy: weights AND tokenizer load on demand so the conductor process
        # never touches the 750 GB checkpoint or HF IO in dummy/byte mode.
        if self._tokenizer is None:
            self._tokenizer = load_snapshot_tokenizer(self.model_path_hf, self.cache_dir)
        return self._tokenizer

    # -------------------------------------------------------------------
    # Model ABC: state
    # -------------------------------------------------------------------

    def declare_state(self) -> list[Array | Sampler]:
        cfg, k = self.config, self.config.mtp_num_draft_tokens
        nodes = frozenset({"LLM"})
        # the MTP layer's plane last
        layers = frozenset(range(cfg.num_hidden_layers + (1 if k else 0)))
        if cfg.mla_absorb:
            entry = LatentEntry(rank=cfg.kv_lora_rank, rope_dim=cfg.qk_rope_head_dim)
            op = LatentAttend(q_heads=cfg.num_attention_heads, scale=cfg.softmax_scale)
            mask = Selected(of=INDEX_KEYS) if cfg.dsa_long_context else Causal()
        else:
            # the naive path, at reduced sizes in the CPU tests
            entry = KVEntry(heads=cfg.num_attention_heads, head_dim=cfg.padded_head_dim)
            op, mask = Attend(q_heads=cfg.num_attention_heads), Causal()
        rope = Rope(theta=cfg.rope_theta, rotary_dim=cfg.qk_rope_head_dim, interleave=cfg.rope_interleave,
                    table_dtype=torch.float32)
        state = [
            Array(KV, nodes=nodes, layers=layers, entry=entry, max_len=self.context_limit(), verify_span=k + 1,
                  reads={ATTN: Read(mask=mask, op=op)}, positions=Positions(rope=rope)),
            # min-p only where the checkpoint defaults it; elsewhere a request's min_p > 0 is refused at admission
            Sampler(SAMPLER, nodes=nodes, vocab_size=cfg.vocab_size, min_p=bool(self.sampling_defaults.get("min_p"))),
        ]
        if cfg.dsa_long_context:
            full = frozenset(i for i in range(cfg.num_hidden_layers) if is_full_indexer_layer(cfg, i))
            state.append(Array(
                INDEX_KEYS, nodes=nodes, layers=full, max_len=self.context_limit(),
                entry=Entry({"k": TensorType((cfg.index_head_dim,), dtype=None)}),
                reads={TOPK: Read(mask=Causal(), op=TopK(count=cfg.index_topk, heads=cfg.index_n_heads))},
            ))
        if k:
            # the state the next verify step drafts from, and its first draft
            state.append(Value(MTP_SEED, nodes=nodes, layers=frozenset({0}), entry=Entry({
                "hidden": TensorType((cfg.hidden_size,), dtype=None), "draft": TensorType((), dtype=torch.long),
            })))
        return state

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        super().validate_config_yaml(config, config_path)
        check_kv_pools_fit(self, config, config_path, logger)

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {KV: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill_outputs = [
            GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="new_token",
                output_modality="text",
                conductor_new_token=True,
                persist=True,
            ),
        ]
        decode_outputs = [GraphEdge(next_node="LLM", name="text_inputs")]
        if self.config.mtp_num_draft_tokens > 0:
            # the tokens emitted so far, which a verify step cuts its verdict and checks its stop by
            prefill_outputs.append(GraphEdge(next_node=EMPTY_DESTINATION, name="count", persist=True))
            decode_outputs.append(GraphEdge(next_node="LLM", name="count"))
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=prefill_outputs,
        )

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=[edge.name for edge in decode_outputs],
                outputs=[
                    *decode_outputs,
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                ],
            ),
            # The window every request's loop cap (get_loop_caps) must fit: the
            # most decode steps a one-token prompt can take before the context guard.
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
        # one prefill over the prompt, then decode until the stop check ends it. A verify step's stop reads the
        # prompt's length, and the prefill's under MTP too.
        prompt = {PROMPT_LEN: model_kwargs[PROMPT_LEN]} if self.config.mtp_num_draft_tokens > 0 else {}
        return [
            WalkStep("prefill", step_metadata={FINAL_PREFILL: True, **prompt}),
            WalkStep("decode", bind={"text_inputs": "new_token"}, step_metadata=prompt),
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
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> NameToTensorList | ProcessPromptOutput:
        if prompt is None:
            return {}
        # read here, in the data worker: a bad knob is the client's 400 before any work
        sampling = self.get_request_resource_configs([], [], kwargs)[SAMPLER]
        if self.config.mtp_num_draft_tokens > 0:
            # refused here as a 400; prepare_inputs would refuse it as a 500
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
            input_ids = self._render_chat(prompt, prompt_parts)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        # here, in the data worker, a ValueError reaches the client as a 400
        if input_ids.numel() > self.config.max_prompt_tokens:
            raise ValueError(
                f"prompt is {input_ids.numel()} tokens; GLM-5.2 is served with "
                f"at most {self.config.max_prompt_tokens}"
            )
        n = input_ids.numel()
        kv = Layout(spans=(Span("ids", "prefill", n, n, "text_inputs"),), growth=(("decode_loop", 1),))
        layouts = {KV: {"main": kv}}
        if self.config.mtp_num_draft_tokens > 0:
            layouts[MTP_SEED] = {"main": Layout()}
        return ProcessPromptOutput(
            {"text_inputs": [input_ids.to(torch.long)]}, {"state_layout": layouts, PROMPT_LEN: n},
        )

    def _render_chat(self, prompt: str, prompt_parts: list[PromptPart] | None) -> torch.Tensor:
        """The chat's turns, each in its own role, through the checkpoint's template, which
        adds ``[gMASK]<sop>`` and opens the assistant turn. An entrypoint with no messages
        sends the prompt as one user turn. TODO: the thinking mode / reasoning_effort dial."""
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
            raise ValueError(f"GLM-5.2's chat template refused the chat: {exc}") from exc
        return input_ids[0]

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
            # greedy by default under MTP, so a request that names no temperature serves
            # rather than being refused; an explicit temperature > 0 still is
            params["temperature"] = 0.0
        return {SAMPLER: SampleSettings(**params)}

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"max_output_tokens", *SAMPLING_KEYS})

    def context_limit(self) -> int:
        """The bound the submodule's preprocess holds every context to:
        index_topk with DSA off (dense MLA is exact there), the serving
        window with it on."""
        return self.config.max_seq_len if self.config.dsa_long_context else self.config.index_topk

    def max_decode_steps(self) -> int:
        """Decode iterations a one-token prompt can run within the window; the
        conductor caps a longer prompt's loop at the room it leaves. Under MTP
        each iteration emits at least one token, so it bounds those loops too."""
        return self.context_limit() - 1

    def get_max_output_tokens(self, **model_kwargs):
        budget = normalize_generation_kwargs(model_kwargs).get(
            "max_output_tokens", self.config.max_output_tokens)
        return self.context_limit() if budget is None else budget

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
        raise ValueError(f"Unsupported modality for GLM-5.2: {modality!r}")

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
                "Glm52Model: no checkpoint resolved for node %r — dummy mode (None).",
                node_name,
            )
            return None

        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
        from mstar.model.glm52.submodules import Glm52LLMSubmodule
        from mstar.model.utils import process_weights_after_loading

        with torch.device("meta"):
            language_model = Glm52ForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        self._load_checkpoint(language_model, source, device, tp_group)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()
        if self.config.tuned_kernels and tp_group is not None and tp_group.world_size > 1:
            # collective over the TP group; every rank builds the LLM here
            dtype = language_model.model.norm.weight.dtype
            tp_group.init_allreduce_fusion(
                self.config.hidden_size, dtype,
                max_tokens=AR_FUSION_MAX_BYTES // (self.config.hidden_size * dtype.itemsize),
            )

        logger.info("Successfully loaded GLM-5.2 submodule for %s", node_name)
        submodule = Glm52LLMSubmodule(language_model, self.config)
        return submodule

    def _load_checkpoint(self, language_model, source: str, device, tp_group) -> None:
        """Load weights, taking the sliced fast read path when possible."""
        from mstar.model.glm52.weight_loader import build_glm52_read_plan
        from mstar.model.loader import load_weights
        from mstar.model.loader.iterators import iter_safetensors_shards

        index_file = Path(source) / "model.safetensors.index.json"
        if not index_file.is_file():
            load_weights(language_model, source, device=device, quant_methods=QUANT_METHODS)
            return

        with open(index_file) as f:
            checkpoint_keys = list(json.load(f)["weight_map"])
        tp_rank = tp_group.rank if tp_group is not None else 0
        tp_size = tp_group.world_size if tp_group is not None else 1
        keys, specs = build_glm52_read_plan(
            checkpoint_keys, self.config, tp_rank, tp_size,
            load_mtp=language_model.mtp is not None,
        )
        logger.info(
            "Glm52Model fast read plan: %d/%d keys, %d sliced (tp %d/%d)",
            len(keys), len(checkpoint_keys), len(specs), tp_rank, tp_size,
        )
        weights = iter_safetensors_shards(
            source, device=device, keys=keys, slice_spec=specs.get, quant_methods=QUANT_METHODS,
        )
        language_model.load_weights(weights)

    def _resolve_checkpoint(self) -> str | None:
        return resolve_checkpoint(self.model_path_hf, self.cache_dir)

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        if self.config.quantization_config is not None:
            return
        quant = read_fp8_block_quant_config(source)
        if quant is not None:
            self.config.quantization_config = quant
            logger.info(
                "Glm52Model: fp8 %s checkpoint (block %s) — non-expert linears "
                "fp8=%s, routed experts fp8-resident=%s, kernels=%s.",
                quant.fmt, quant.weight_block_size, self.config.dense_fp8,
                self.config.moe_fp8_resident, self.config.kernels,
            )
