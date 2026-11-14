"""DeepseekV41Model: DeepSeek-V4.1-Flash, text in / text out."""

import dataclasses
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.resources import (
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, CSASpec
from mstar.engine.resources.host_table.config import HostTableConfig, HostTableSpec
from mstar.engine.resources.kv.config import AuxPlanes, SpanKind
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import KVLabel, LabelLayout, Model, ProcessPromptOutput, Span, WalkStep
from mstar.model.components.request_kwargs import (
    SAMPLING_KEYS,
    generation_config_dir,
    sampling_defaults,
)
from mstar.model.deepseek_v41.config import (
    CSA,
    ENGRAM,
    LABEL,
    NGRAM,
    POOL_KV,
    POOL_SCORE,
    SAMPLER,
    STATE,
    WINDOW,
    DeepseekV41Config,
    kv_cache_key,
)
from mstar.model.multimodal import DEEPSEEK_V41_TEMPLATE_KWARGS, render_chat_template
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import (
    ByteLevelTextOutput,
    check_loaded,
    load_snapshot_tokenizer,
    process_weights_after_loading,
    resolve_checkpoint,
)
from mstar.utils.generation_kwargs import normalize_generation_kwargs

logger = logging.getLogger(__name__)

# what the model card recommends, for a checkpoint that ships no generation_config.json
DEFAULT_SAMPLING = MappingProxyType({"temperature": 1.0, "top_p": 1.0})
# the checkpoint's format, which this model's own loader reads
QUANT_METHODS = ("fp8",)


class DeepseekV41Model(ByteLevelTextOutput, Model):
    """DeepSeek-V4.1-Flash: sliding-window plus compressed sparse attention, hyper-connected
    MoE causal LM. Text only: the vision tower and the DSpark draft layers are not served."""

    DECODE_LOOP = "decode_loop"
    # read only by process_prompt's template, and kept off the conductor's wire
    PREPROCESS_ONLY_KWARGS = ("messages", "tools", "chat_template_kwargs", "reply_prefix")
    sampling_defaults: Mapping[str, object] = DEFAULT_SAMPLING

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        self.cache_dir = cache_dir
        self.model_path_hf = kwargs.get("checkpoint_path") or model_path_hf
        self._config_variant = kwargs.get("config_variant", "full")
        self._config_from_checkpoint = False
        if self._config_variant == "reduced":
            self.config = DeepseekV41Config.reduced()
        else:
            self.config = DeepseekV41Config()
        if "max_seq_len" in kwargs:
            self.config.max_seq_len = _positive_int("max_seq_len", kwargs["max_seq_len"])
        for knob in ("moe_experts", "moe_kernel"):
            if knob in kwargs:
                # replace() reruns the config's validation on the value
                self.config = dataclasses.replace(self.config, **{knob: kwargs[knob]})
        if "engram_tables" in kwargs:
            # rebuilt, so the config refuses a placement it does not know
            self.config = dataclasses.replace(self.config, engram_tables=kwargs["engram_tables"])
        if kwargs.get("capture_contexts") is not None:
            self.config = dataclasses.replace(self.config, capture_contexts=tuple(kwargs["capture_contexts"]))
        # requests that may hold state at once; the pool holds one more slot, its sink, and a
        # serve YAML sizes it under ``resources: state: {max_slots: N}``
        self.max_requests = _positive_int("max_requests", kwargs.get("max_requests", 32))
        # "byte" maps UTF-8 bytes to token ids for reduced serve (no HF IO)
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
    # Model ABC: resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        cfg = self.config
        blocks = {
            # each layer's last sliding_window latents, a ring by position
            WINDOW: RecurrentBlockConfig(shape=(cfg.sliding_window, cfg.head_dim), dtype=None),
            # the n-gram hashes read a request's last ids
            NGRAM: RecurrentBlockConfig(
                shape=(max(cfg.engram_max_ngram_size - 1, 1),), dtype=torch.long, per_layer=False),
        }
        if cfg.pooled_sources:
            # a pooling source's open group: its tokens' projections and gate scores. A block
            # is per layer, so each source reads and writes its own, the other layers' unused.
            shape = (cfg.max_ratio, cfg.head_dim)
            blocks[POOL_KV] = RecurrentBlockConfig(shape=shape, dtype=torch.float32)
            blocks[POOL_SCORE] = RecurrentBlockConfig(shape=shape, dtype=torch.float32)
        latent_width, key_width = cfg.cache_widths()
        caches = [
            # the latents of the sources that pool ``ratio`` tokens, a plane each, with their
            # index keys beside them: a row per group (so ``max_seq_len`` over-counts its rows),
            # each row its FP4 bytes
            KVSpec(
                resource_key=kv_cache_key(ratio),
                nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=len(cfg.sources_of(ratio)),
                    num_kv_heads=1,
                    head_dim=latent_width,
                    max_seq_len=cfg.max_seq_len,
                    num_qo_heads=cfg.num_attention_heads,
                    layout=KVLayout.MLA,
                    kv_lora_rank=latent_width,
                    qk_rope_head_dim=0,
                    aux_planes={INDEX_PLANES: AuxPlanes(num_layers=len(cfg.sources_of(ratio)), width=key_width)},
                ),
            )
            for ratio in cfg.source_ratios
        ]
        specs: list[NodeResourceSpec] = [
            *caches,
            CSASpec(
                resource_key=CSA,
                nodes={"LLM"},
                config=CSAConfig(
                    caches={ratio: kv_cache_key(ratio) for ratio in cfg.source_ratios},
                    topk=cfg.index_topk, sm_scale=cfg.softmax_scale, index_head_dim=cfg.index_head_dim,
                    latent_dim=cfg.head_dim, latent_rows=cfg.latent_rows, key_rows=cfg.key_rows,
                ),
            ),
            RecurrentStateSpec(
                resource_key=STATE,
                nodes={"LLM"},
                config=RecurrentStateConfig(
                    num_layers=cfg.num_hidden_layers, blocks=blocks,
                    max_slots=self.max_requests + 1,  # + the sink
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={"LLM"},
                vocab_size=cfg.vocab_size,
                enable_repetition_penalty=True,
                enable_min_p=bool(self.sampling_defaults.get("min_p")),
            ),
        ]
        if cfg.engram_on_host:
            from mstar.model.deepseek_v41.components.engram import host_tables

            specs.append(HostTableSpec(
                resource_key=ENGRAM,
                nodes={"LLM"},
                config=HostTableConfig(
                    checkpoint=self.model_path_hf, cache_dir=self.cache_dir, tables=host_tables(cfg)),
            ))
        return specs

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        label = {LABEL: KVLabel(growth=(("decode", "decode_loop"),))}
        return {kv_cache_key(ratio): dict(label) for ratio in self.config.source_ratios}

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        del input_modalities, output_modalities
        model_kwargs = normalize_generation_kwargs(model_kwargs)
        params = {
            **self.sampling_defaults,
            **{k: model_kwargs[k] for k in SAMPLING_KEYS if k in model_kwargs},
        }
        return {SAMPLER: SamplingReqConfig(**params)}

    # -------------------------------------------------------------------
    # Model ABC: graph walks
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
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
                    GraphEdge(next_node="LLM", name="text_inputs"),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                ],
            ),
            # runaway guard; the per-request budget lives in check_stop
            max_iters=self.max_decode_steps(),
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
        # one prefill over the prompt, then decode until the stop check ends it
        return [
            WalkStep("prefill", step_metadata={FINAL_PREFILL: True}),
            WalkStep("decode", bind={"text_inputs": "new_token"}),
        ]

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def render_prompt(self, prompt: str, kwargs: dict) -> torch.Tensor:
        """The prompt's token ids, through the checkpoint's chat template: a chat request's
        ``messages`` in their roles, plain text as one user turn."""
        if self._tokenizer_mode == "byte":
            vocab = self.config.vocab_size
            text = (prompt.encode("utf-8") or b"\0") + (kwargs.get("reply_prefix") or "").encode("utf-8")
            return torch.tensor([min(b, vocab - 1) for b in text], dtype=torch.long)
        if not getattr(self.tokenizer, "chat_template", None):
            # the renderer would tokenize the bare text, and the model answer a prompt it never saw in training
            raise RuntimeError(
                f"{self.model_path_hf}: the tokenizer has no chat template (chat_template.jinja beside "
                "tokenizer.json), which every DeepSeek-V4.1 prompt renders through")
        return render_chat_template(
            self.tokenizer, prompt, kwargs.get("prompt_parts"), kwargs, DEEPSEEK_V41_TEMPLATE_KWARGS)

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
        # the conductor reads these kwargs; read here first, a bad one is the client's 400
        kwargs = normalize_generation_kwargs(kwargs)
        self.get_request_resource_configs([], [], kwargs)
        self.get_max_output_tokens(**kwargs)
        if kwargs.get("return_logprobs"):
            raise ValueError("DeepSeek-V4.1 does not report logprobs; leave return_logprobs unset")
        input_ids = self.render_prompt(prompt, kwargs)
        n, limit = input_ids.numel(), self.config.max_prompt_tokens
        if n > limit:
            raise ValueError(f"prompt is {n} tokens; DeepSeek-V4.1 is served with at most {limit}")
        return ProcessPromptOutput(
            {"text_inputs": [input_ids.to(torch.long)]}, {"kv_layout": self._kv_layouts(n)},
        )

    def _kv_layouts(self, prompt: int) -> dict[str, dict[str, LabelLayout]]:
        """What a ``prompt``-token request writes to each cache: a row per pooled group. A
        decode step writes a row every ``ratio`` steps, which a layout cannot say, so a pooled
        cache declares a row a step up to the most the context holds."""
        layouts = {}
        for ratio in self.config.source_ratios:
            rows = prompt // ratio
            if ratio == 1:
                layout = LabelLayout(
                    spans=(Span(SpanKind.IDS, "prefill", rows, rows, "text_inputs"),), growth=(("decode_loop", 1),))
            else:
                layout = LabelLayout(
                    spans=(Span(SpanKind.ROWS, "prefill", rows, rows, "text_inputs"),) if rows else (),
                    growth=(("decode_loop", 1),),
                    max_growth=self.config.max_seq_len // ratio - rows)
            layouts[kv_cache_key(ratio)] = {LABEL: layout}
        return layouts

    def max_decode_steps(self) -> int:
        """Decode iterations a one-token prompt can run before its context fills the window."""
        return self.config.max_seq_len - 1

    def get_max_output_tokens(self, **model_kwargs):
        budget = normalize_generation_kwargs(model_kwargs).get("max_output_tokens", self.config.max_output_tokens)
        return self.config.max_seq_len if budget is None else budget

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"max_output_tokens", *SAMPLING_KEYS})

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            return super().postprocess(output, modality, **kwargs)
        raise ValueError(f"Unsupported modality for DeepSeek-V4.1: {modality!r}")

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={}, ep_enabled_nodes={"LLM"})

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None, moe_parallel=None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(
                node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype, moe_parallel=moe_parallel)
        return self._submodule_cache[node_name]

    def _adopt_checkpoint_config(self, source: str) -> None:
        """The checkpoint's geometry in place of the defaults, serving knobs kept."""
        path = Path(source) / "config.json"
        if self._config_variant != "full" or self._config_from_checkpoint or not path.is_file():
            return
        with open(path) as f:
            read = DeepseekV41Config.from_hf_config(json.load(f))
        for knob in ("max_seq_len", "max_output_tokens", "moe_experts", "moe_kernel", "engram_tables",
                     "capture_contexts"):
            setattr(read, knob, getattr(self.config, knob))
        self.config, self._config_from_checkpoint = read, True

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None, autocast_dtype: torch.dtype | None = None,
        moe_parallel=None,
    ) -> NodeSubmodule | None:
        if node_name != "LLM":
            return None
        if moe_parallel is not None and (moe_parallel.log_expert_load_every or moe_parallel.mask_padding):
            raise ValueError("DeepSeek-V4.1 takes ep_config.parallel only: no expert-load log, no padding mask")
        source = resolve_checkpoint(self.model_path_hf, self.cache_dir)
        if source is None:
            logger.info("DeepseekV41Model: no checkpoint resolved for node %r: dummy mode (None).", node_name)
            return None
        self._adopt_checkpoint_config(source)

        from mstar.model.deepseek_v41.components.engram import compressed_token_map
        from mstar.model.deepseek_v41.components.layers import DeepseekV41ForCausalLM
        from mstar.model.deepseek_v41.submodules import DeepseekV41LLMSubmodule
        from mstar.model.deepseek_v41.weight_loader import rank_slices
        from mstar.model.loader.iterators import iter_safetensors_shards

        token_map = None
        if self.config.engram_layer_ids:
            token_map = torch.tensor(compressed_token_map(self.tokenizer)[0], dtype=torch.long)
        # built on meta, so nothing is allocated before to_empty; the loader fills every
        # parameter in the dtype the module declared for it
        with torch.device("meta"):
            language_model = DeepseekV41ForCausalLM(
                self.config, token_map, comm_group=tp_group, moe_parallel=moe_parallel)
        if autocast_dtype is not None:
            language_model = _cast_floating(language_model, autocast_dtype)
        language_model.to_empty(device=device)
        # a rank reads its own slice of each tensor, and nothing of a table the engine maps
        # from the host or of the parts not served
        weights = iter_safetensors_shards(
            source, device=device, quant_methods=QUANT_METHODS, slice_spec=rank_slices(language_model))
        check_loaded(language_model, language_model.load_weights(weights, sliced=True), source)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()
        logger.info("Successfully loaded DeepSeek-V4.1 submodule for %s", node_name)
        return DeepseekV41LLMSubmodule(language_model=language_model, config=self.config)


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _cast_floating(module: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    """``module`` with its default-dtype parameters in ``dtype``; the ones a layer declared in
    fp32 (the sinks, the hyper-connections, the pooling projections, the router bias) stay."""
    default = torch.get_default_dtype()
    keep = {
        name for name, p in module.named_parameters()
        if p.dtype != default or name.endswith(("attn_sink", "e_score_correction_bias"))
        or ".hc_attn." in name or ".hc_ffn." in name or ".compressor." in name
    }
    for name, p in module.named_parameters():
        if name not in keep:
            p.data = p.data.to(dtype)
    return module
