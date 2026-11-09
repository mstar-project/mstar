"""Kimi K3 and Kimi K3.1 Pro: hybrid KDA / gated-MLA latent-MoE causal LMs, text in, text out.

One ``LLM`` node, tensor parallel; the routed experts optionally expert parallel. K3.1 adds a
DSA indexer to the MLA layers: served within ``index_topk`` (2048) tokens, dense MLA is exactly
DSA, and ``dsa_long_context`` hands longer decodes to the DSA resource. The vision tower and the
DSpark draft are not served. Ported from merceod's #257 onto main's KDA, MLA, DSA, MoE and
recurrent-state components.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    AuxPlanes,
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, PrefixKeys, ProcessPromptOutput, Span
from mstar.model.kimi_k3.config import (
    ATTN,
    DSA,
    KDA,
    KDA_STATE,
    KV_CACHE,
    LABEL,
    SAMPLER,
    KimiK3ModelConfig,
)
from mstar.model.multimodal import TEXT, PromptPart, chat_turns
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import check_loaded, process_weights_after_loading
from mstar.utils.generation_kwargs import normalize_generation_kwargs

logger = logging.getLogger(__name__)

# Steps of up to this many rows reduce the latent MoE's routed experts in their norm's fused
# all-reduce + RMSNorm kernel (decode, and verify steps of up to 64 requests x 4 tokens);
# larger ones (prefill chunks) all-reduce and norm apart.
AR_FUSION_MAX_TOKENS = 256

# what the conductor and the tokenizer read; the weights stay on the workers
METADATA_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py", "encoding_k3.py",
)

# The quantization API's view of the LLM root (``get_quantization_roots``): which linears may run
# low precision. KDA's linears feed its fused input projection and the absorbed MLA folds its down
# projections, so both stay bf16, as do the latent projections, the router and the LM head
# (the checkpoint's own ignore list).
QUANTIZABLE_CANDIDATES = (
    "model.layers.*.self_attn.q_b_proj",
    "model.layers.*.self_attn.o_proj",
    "model.layers.*.self_attn.g_proj",
    "model.layers.*.mlp.*",
    "model.layers.*.block_sparse_moe.shared_experts.*",
    "model.layers.*.block_sparse_moe.experts",
)
_KEEP_BF16 = (
    "model.layers.*.self_attn.q_a_proj",
    "model.layers.*.self_attn.kv_a_proj_with_mqa",
    "model.layers.*.self_attn.kv_b_proj",
    "model.layers.*.self_attn.indexer.*",
    "model.layers.*.block_sparse_moe.routed_expert_down_proj",
    "model.layers.*.block_sparse_moe.routed_expert_up_proj",
    "model.layers.*.self_attention_res_proj",
    "model.layers.*.mlp_res_proj",
    "model.output_attn_res_proj",
    "lm_head",
)


def quantization_keep_bf16(config: KimiK3ModelConfig) -> tuple[str, ...]:
    """``_KEEP_BF16`` plus every linear of the KDA layers (named per layer: an MLA layer's
    ``o_proj`` and ``g_proj`` are candidates)."""
    kda = tuple(p for i in config.kda_layers
                for p in (f"model.layers.{i}.self_attn.*", f"model.layers.{i}.self_attn.forget_gate.*"))
    return _KEEP_BF16 + kda



def thinking_requested(kwargs: dict) -> bool:
    """Whether the reply thinks first (the release's template default: yes). Off with ``reasoning_effort:
    none``, or ``enable_thinking: false`` sent flat (M*'s key) or in ``chat_template_kwargs`` (vLLM's and
    SGLang's; there also ``thinking``), so the same request thinks or not on every engine."""
    template = kwargs.get("chat_template_kwargs") or {}
    switch = next((v for v in (kwargs.get("enable_thinking"), template.get("enable_thinking"),
                               template.get("thinking")) if v is not None), None)
    return kwargs.get("reasoning_effort") != "none" and (switch is None or bool(switch))


class KimiK3Model(Model):
    """Kimi K3 (``moonshotai/Kimi-K3``): no DSA indexer, dense MLA at any length."""

    # the conductor builds the fused-MoE align op before spawning the workers
    prebuild_fused_moe = True

    @staticmethod
    def default_config(**overrides) -> KimiK3ModelConfig:
        return KimiK3ModelConfig.k3(**overrides)

    def __init__(self, model_path_hf: str, cache_dir: str | None = None,
                 prefill_token_buckets: list[int] | None = None,
                 prefill_capture_batch_sizes: list[int] | None = None, **kwargs):
        self.cache_dir = cache_dir
        self.model_path_hf = kwargs.get("checkpoint_path") or model_path_hf
        overrides = {}
        for key, cast in (("max_seq_len", int), ("dsa_long_context", bool),
                          ("moe_expert_parallel", bool), ("routed_experts_backend", str),
                          ("routed_experts_autotune", bool), ("routed_experts_tune_tokens", tuple),
                          ("shared_experts_overlap_max_tokens", int), ("moe_mask_padding", bool),
                          ("routed_experts_routing", str), ("routed_experts_finalize", str),
                          ("attn_res_backend", str), ("latent_moe_front", str)):
            if key in kwargs:
                value = kwargs[key]
                if cast is bool and not isinstance(value, bool):
                    raise ValueError(f"{key} must be true or false, got {value!r}")  # bool("false")
                overrides[key] = cast(value)
        if "shared_experts_overlap" in kwargs:  # routed / shared / off, or a bool (shared / off)
            value = kwargs["shared_experts_overlap"]
            if not isinstance(value, (bool, str)):
                raise ValueError(f"shared_experts_overlap must be routed, shared, off or a bool, got {value!r}")
            overrides["shared_experts_overlap"] = value
        # a checkpoint whose config.json is not K3 / K3.1 Pro's geometry (a truncated test one)
        strict = kwargs.get("strict_config", True)
        if not isinstance(strict, bool):
            raise ValueError(f"strict_config must be true or false, got {strict!r}")
        variant = kwargs.get("config_variant", "full")
        if variant == "reduced":
            self.config = KimiK3ModelConfig.reduced(**overrides)
        elif variant == "reduced_nodsa":
            self.config = KimiK3ModelConfig.reduced(dsa=False, **overrides)
        elif variant == "full":
            hf_config = self._local_hf_config()
            self.config = (self.default_config(**overrides) if hf_config is None
                           else KimiK3ModelConfig.from_hf_config(hf_config, strict=strict, **overrides))
        else:
            raise ValueError(f"config_variant must be full, reduced or reduced_nodsa, got {variant!r}")
        # Requests that may hold KDA state at once (the decode batch cap); the pool keeps one more
        # slot, its sink. A serving YAML sizes it under ``resources: kda_state: {max_slots: N}``.
        self.kda_max_requests = int(kwargs.get("kda_max_requests", 64))
        # Decode graph batch sizes (default 1..64 by powers of two). Decode batches stop at the
        # largest that fits the KDA slots, so a config past 64 requests names a longer ladder.
        ladder = kwargs.get("decode_capture_batch_sizes")
        self.decode_capture_batch_sizes = None if ladder is None else tuple(int(b) for b in ladder)
        # Captured prefill (within index_topk): the packed token counts and row counts it is captured at.
        # None leaves prefill eager, ~2.2 ms of host work a layer a step whatever its length.
        self.prefill_token_buckets = (None if prefill_token_buckets is None
                                      else [int(n) for n in prefill_token_buckets])
        self.prefill_capture_batch_sizes = (None if prefill_capture_batch_sizes is None
                                            else tuple(int(b) for b in prefill_capture_batch_sizes))
        # torch.compile of the tensor glue between the resources (AttnRes reads, norms, the MoE block:
        # router, SiTU, latent projections), as regions the engine compiles before capture
        self.compile_glue = bool(kwargs.get("compile_glue", True))
        # the small-M GEMM backends measured on this GPU (benchmark/kernels/k31_dense_gemm.py --table-out); without
        # one every projection runs cuBLAS. Loaded where the submodule is built (`_load_small_m_gemm_table`): the
        # workers get this instance pickled, without running __init__, and the table is their process's
        table = kwargs.get("small_m_gemm_table")
        self.small_m_gemm_table = str(table) if table else None
        # with compile_glue: the AttnRes reads' regions too (their Triton op inside), or only the MLP blocks.
        # Off: the reads are Triton kernels with the residual add fused, and a region around them added
        # launches (H200 job 1639, full model: 3,170 a decode step with them, 2,844 without; no faster)
        self.compile_reads = bool(kwargs.get("compile_reads", False))
        # the KDA conv tail's dtype, which must match the projections' (CPU tests pass fp32)
        self.kda_conv_dtype = kwargs.get("kda_conv_dtype", torch.bfloat16)
        # A prompt runs in chunks of at most this many tokens (one step each), so a 200K-token
        # prompt never holds 200K tokens of activations (the attention-residual stack alone is
        # tokens x blocks x hidden); 0 runs every prompt whole. The serving config's
        # max_batch_tokens overrides it.
        self.prefill_chunk_tokens = int(kwargs.get("prefill_chunk_tokens", 8192))
        # "byte" maps UTF-8 bytes to token ids (reduced configs; no tokenizer IO)
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._metadata_dir: str | None = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # -- checkpoint metadata ----------------------------------------------------

    def _local_hf_config(self) -> dict | None:
        """``config.json`` when it is on disk (a local checkpoint, or the HF cache), without
        fetching anything: the defaults stand in for a Hub id not yet cached."""
        path = Path(self.model_path_hf) / "config.json"
        if not path.is_file():
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(self.model_path_hf, "config.json", cache_dir=self.cache_dir)
            except Exception:  # not a repo id, or no hub client
                cached = None
            if not isinstance(cached, str):
                return None
            path = Path(cached)
        return json.loads(path.read_text())

    def metadata_dir(self) -> str:
        """The checkpoint's config and tokenizer files, fetched without the weights."""
        if self._metadata_dir is None:
            if Path(self.model_path_hf, "config.json").is_file():
                self._metadata_dir = str(self.model_path_hf)
            else:
                from huggingface_hub import snapshot_download

                self._metadata_dir = snapshot_download(
                    self.model_path_hf, cache_dir=self.cache_dir, allow_patterns=list(METADATA_FILES))
        return self._metadata_dir

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from mstar.model.kimi_k3.tokenizer import KimiK3Tokenizer

            self._tokenizer = KimiK3Tokenizer(self.metadata_dir())
        return self._tokenizer

    # -- resources ----------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        cfg = self.config
        kda_blocks = DeltaNetGeometry(
            num_k_heads=cfg.linear_num_heads, num_v_heads=cfg.linear_num_heads,
            head_k_dim=cfg.linear_head_dim, head_v_dim=cfg.linear_head_dim,
            conv_kernel_size=cfg.linear_conv_kernel_size,
        ).to_blocks(conv_dtype=self.kda_conv_dtype)
        specs: list[NodeResourceSpec] = [
            KVSpec(
                resource_key=KV_CACHE,
                nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=len(cfg.full_attn_layers),
                    num_kv_heads=1,
                    head_dim=cfg.cache_latent_dim,
                    max_seq_len=cfg.context_limit,
                    num_qo_heads=cfg.num_attention_heads,
                    layout=KVLayout.MLA,
                    kv_lora_rank=cfg.kv_lora_rank,
                    qk_rope_head_dim=cfg.mla_cache_kpe,
                    # dsa_long_context: each MLA layer's index keys on the latents' pages
                    aux_planes={INDEX_PLANES: AuxPlanes(
                        num_layers=len(cfg.full_attn_layers), width=cfg.index_head_dim,
                    )} if cfg.dsa_long_context else {},
                ),
            ),
            AttentionSpec(
                resource_key=ATTN,
                nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE, backend=AttnBackend.FLASHINFER_MLA, sm_scale=cfg.softmax_scale),
            ),
            SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=cfg.vocab_size,
                        enable_repetion_penalty=True),
            RecurrentStateSpec(
                resource_key=KDA_STATE,
                nodes={"LLM"},
                config=RecurrentStateConfig(
                    num_layers=len(cfg.kda_layers), blocks=kda_blocks,
                    max_slots=self.kda_max_requests + 1,  # + the sink
                    # prefix reuse: the KDA state at the latent cache's page boundaries, in the
                    # serving config's `resources: kda_state: prefix_slots`; above one rank where
                    # the latent cache sets prefix_cache_tp
                    kv_cache=KV_CACHE),
            ),
            LinearAttnSpec(
                resource_key=KDA,
                nodes={"LLM"},
                config=LinearAttnConfig(recurrent_state=KDA_STATE, variant=LinearAttnVariant.KDA,
                                        backend=LinearAttnBackend.TRITON),
            ),
        ]
        if cfg.dsa_long_context:
            # past index_topk each query attends its indexer's top index_topk latents; within
            # it dense MLA is the same computation, which an eager step keeps
            specs.append(DSASpec(resource_key=DSA, nodes={"LLM"}, config=DSAConfig(
                kv_cache=KV_CACHE, topk=cfg.index_topk, sm_scale=cfg.softmax_scale,
                dense_below_topk=True, index_head_dim=cfg.index_head_dim)))
        return specs

    def get_chunking_policies(self):
        """The prefill walk in chunks of ``prefill_chunk_tokens``: each chunk continues the KDA
        state, the latent cache and K3.1's index keys where the last one left them, and only the
        final chunk's sample is kept (the sampler and the worker skip a mid-prefill row's)."""
        from mstar.model.submodule_base import ChunkingPolicy

        if not self.prefill_chunk_tokens:
            return {}
        # a chunk cut mid-way ends on a 64-token boundary, so the next starts on one: its MLA rows
        # need no peeling (it replays a packed prefill capture) and its KDA chunks end on the
        # boundaries a snapshot is stored at
        from mstar.engine.resources.attn.flashinfer_mla import ROW_ALIGN

        return {"LLM": (ChunkingPolicy(walks=("prefill",), chunkable=True,
                                       max_batch_tokens=self.prefill_chunk_tokens,
                                       chunk_align=ROW_ALIGN),)}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        # keyed by the prompt's ids and the tokens decode samples, for cross-request prefix reuse
        return {KV_CACHE: {LABEL: KVLabel(growth=(("decode", "decode_loop"),),
                                          keys=PrefixKeys("text_inputs", ("prefill",), growth=True))}}

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs], model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = normalize_generation_kwargs(model_kwargs)
        params = {k: model_kwargs.get(k, getattr(self.config, k))
                  for k in ("temperature", "top_p", "repetition_penalty", "ignore_eos")}
        for k, cast in (("top_k", int), ("penalize_prompt", bool)):
            if k in model_kwargs:
                params[k] = cast(model_kwargs[k])
        return {SAMPLER: SamplingReqConfig(**params)}

    # -- graph ----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text",
                               conductor_new_token=True, persist=True)],
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[GraphEdge(next_node="LLM", name="text_inputs"),
                         GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text")],
            ),
            max_iters=self.max_decode_steps(),  # a runaway guard; check_stop holds the budget
            outputs=[],
        )
        return dict(prefill=prefill, decode=decode)

    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities, output_modalities=output_modalities,
            graph_walk="prefill", is_prefill=True)
        edge = GraphEdge(next_node="LLM", name="text_inputs")
        edge.tensor_info = input_signals.get("text_inputs", [])
        return ForwardPassArgs(full_metadata=metadata, inputs=[edge],
                               unpersist_tensors=list(edge.tensor_info),
                               step_metadata={"is_prefill": True})

    def get_partition_forward_pass_args(
        self, partition_name: str, partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]], incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            # the decode loop ran to a stop (check_stop) or to max_iters: the request is done
            metadata.kwargs["decode_finished"] = True
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[],
                                   request_done=True)
        edge = GraphEdge(next_node="LLM", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(full_metadata=metadata, inputs=[edge],
                               unpersist_tensors=list(edge.tensor_info),
                               step_metadata={"is_prefill": metadata.is_prefill})

    # -- prompts and budgets ----------------------------------------------------------

    def prompt_limit(self) -> int:
        """Prompt tokens a request may bring: decode runs at least one step and one more may be
        scheduled when it stops, so two rows stay free."""
        return self.config.context_limit - 2

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList | ProcessPromptOutput:
        if prompt is None:
            return {}
        # the conductor reads these; a bad one is the client's 400 before any rank sees it
        kwargs = normalize_generation_kwargs(kwargs)
        self.get_request_resource_configs({}, kwargs)
        self.get_max_output_tokens(**kwargs)
        if kwargs.get("min_p", 0) > 0:
            raise ValueError(f"Kimi K3 is served without min_p (min_p={kwargs['min_p']}); send none")
        if kwargs.get("return_logprobs"):
            raise ValueError("Kimi K3 does not report logprobs; leave return_logprobs unset")
        if self._tokenizer_mode == "byte":
            vocab = self.config.vocab_size
            input_ids = torch.tensor([min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0],
                                     dtype=torch.long)
        else:
            input_ids = self._render_chat(prompt, kwargs.get("prompt_parts"), kwargs)
        n = input_ids.numel()
        if n > self.prompt_limit():
            raise ValueError(f"prompt is {n} tokens; this deployment serves at most {self.prompt_limit()}")
        layout = LabelLayout(spans=(Span("ids", "prefill", n, n, "text_inputs"),),
                             growth=(("decode_loop", 1),))
        return ProcessPromptOutput({"text_inputs": [input_ids]}, {"kv_layout": {KV_CACHE: {LABEL: layout}}})

    def _render_chat(self, prompt: str, prompt_parts: list[PromptPart] | None, kwargs: dict) -> torch.Tensor:
        parts = prompt_parts or [PromptPart(modality=TEXT, text=prompt, role="user")]
        messages = [
            {"role": turn.role,
             "content": "\n".join(p.text or "" for p in turn.parts if p.modality == TEXT)}
            for turn in chat_turns(parts)
        ]
        thinking = thinking_requested(kwargs)
        effort = kwargs.get("reasoning_effort")
        ids = self.tokenizer.apply_chat_template(
            messages, thinking=thinking,
            thinking_effort=None if not thinking or effort in (None, "none") else effort)
        return torch.tensor(ids, dtype=torch.long)

    def max_decode_steps(self) -> int:
        return self.config.context_limit - 1

    def get_max_output_tokens(self, **model_kwargs):
        budget = normalize_generation_kwargs(model_kwargs).get(
            "max_output_tokens", self.config.max_output_tokens)
        if budget is None:
            return self.max_decode_steps() + 1
        return min(budget, self.config.context_limit)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the prefill emits the first of max_output_tokens
        return {"decode_loop": max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality != "text":
            raise ValueError(f"Kimi K3 emits text only, got {modality!r}")
        token_ids = [int(t) for t in output.flatten().tolist()]
        if self._tokenizer_mode == "byte":
            return bytes(t & 0xFF for t in token_ids)
        # raw bytes: a character split across tokens reaches the client whole
        return self.tokenizer.decode_bytes(token_ids)

    # -- sharding, quantization, submodules ------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_quantization_roots(self):
        """The LLM root for the quantization API: the routed experts (MXFP4 in the release, loaded as
        stored by default) and the candidate linears."""
        from mstar.model.components.quant_linear import QuantizableLinears
        from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
        from mstar.model.kimi_k3.weight_loader import checkpoint_rules
        from mstar.quantization.resolve import QuantizationRoot

        return [QuantizationRoot(
            node="LLM", root_type=KimiK3ForCausalLM,
            linears=QuantizableLinears(candidates=QUANTIZABLE_CANDIDATES,
                                       keep_bf16=quantization_keep_bf16(self.config)),
            rules=checkpoint_rules(self.config),
            checkpoint_dir=lambda: self._metadata_dir,
            build=lambda comm_group: KimiK3ForCausalLM(self.config, comm_group),
        )]

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(
                node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype)
        return self._submodule_cache[node_name]

    def _load_small_m_gemm_table(self) -> None:
        """The small-M GEMM table into this process's dispatcher (``small_m_gemm.load_table``), in the process
        that runs the forward. A table the config names but the host lacks (the B300 configs' path on another
        machine, or before the node's bench wrote it) is a warning and cuBLAS, not a failed boot."""
        table = self.small_m_gemm_table
        if not table:
            return
        if not Path(table).is_file():
            logger.warning("small_m_gemm_table %s is not on this host: every projection runs cuBLAS", table)
            return
        from mstar.utils import small_m_gemm

        small_m_gemm.load_table(table)
        logger.info("small_m_gemm_table %s: %d shapes for capability %s", table,
                    sum(len(v) for v in small_m_gemm._TABLES.values()), sorted(small_m_gemm._TABLES))

    def _create_submodule(self, node_name, device, tp_group=None, autocast_dtype=None):
        if node_name != "LLM":
            return None
        # B300 job 20261108T003606Z: loaded in __init__, the table reached only the conductor; every worker
        # projection ran cuBLAS (nvjet) while the config named it
        self._load_small_m_gemm_table()
        source = self._resolve_checkpoint()
        if source is None:
            logger.info("KimiK3Model: no checkpoint for %r, dummy mode (None).", node_name)
            return None
        self._check_checkpoint_config(source)

        from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
        from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
        from mstar.quantization.scope import materialize

        with torch.device("meta"):
            language_model = KimiK3ForCausalLM(self.config, comm_group=tp_group)
        language_model = language_model.to(autocast_dtype or torch.bfloat16)
        # the MXFP4 experts get their packed storage on meta: to_empty must never allocate the bf16 stacks
        # they replace (TP8: 4.59 GiB of gate_up a layer, 92 layers; B300 04c ran out of memory there)
        materialize(language_model, device)
        self._load_checkpoint(language_model, source, device)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()
        if (tp_group is not None and tp_group.world_size > 1 and self.config.latent_moe_use_norm
                and torch.device(device).type == "cuda"):
            # the latent MoE's routed experts reduce in their norm's kernel (all-reduce + RMSNorm,
            # FlashInfer's allreduce_fusion); collective over the TP group, and every rank builds
            # the LLM here.
            dtype = language_model.model.layers[-1].ffn.routed_expert_norm.weight.dtype
            tp_group.init_allreduce_fusion(
                self.config.routed_expert_hidden_size, dtype,
                max_tokens=AR_FUSION_MAX_TOKENS)
        self.kda_conv_dtype = language_model.kda_conv_dtype()
        logger.info("Loaded Kimi K3 (%s) for %s", "DSA" if self.config.dsa else "dense MLA", node_name)
        return KimiK3LLMSubmodule(language_model=language_model, config=self.config,
                                  decode_capture_batch_sizes=self.decode_capture_batch_sizes,
                                  compile_glue=self.compile_glue, compile_reads=self.compile_reads,
                                  prefill_token_buckets=self.prefill_token_buckets,
                                  prefill_capture_batch_sizes=self.prefill_capture_batch_sizes)

    def checkpoint_path(self) -> str | None:
        """The checkpoint directory: the prefix cache roots its keys in its index and config, so
        the cache opens without a ``prefix_cache_salt``."""
        return self._resolve_checkpoint()

    def _resolve_checkpoint(self) -> str | None:
        from mstar.model.utils import resolve_checkpoint

        source = resolve_checkpoint(self.model_path_hf, self.cache_dir)
        if source is not None:
            self._metadata_dir = source
        return source

    def _check_checkpoint_config(self, source: str) -> None:
        """The resources were declared from this config; the weights must be the same model."""
        path = Path(source) / "config.json"
        if not path.is_file():
            return
        loaded = KimiK3ModelConfig.from_hf_config(
            json.loads(path.read_text()), strict=False, max_seq_len=self.config.max_seq_len,
            dsa_long_context=self.config.dsa_long_context,
            moe_expert_parallel=self.config.moe_expert_parallel,
            routed_experts_backend=self.config.routed_experts_backend,
            routed_experts_autotune=self.config.routed_experts_autotune,
            routed_experts_tune_tokens=self.config.routed_experts_tune_tokens,
            shared_experts_overlap=self.config.shared_experts_overlap,
            shared_experts_overlap_max_tokens=self.config.shared_experts_overlap_max_tokens,
            moe_mask_padding=self.config.moe_mask_padding,
            routed_experts_routing=self.config.routed_experts_routing,
            routed_experts_finalize=self.config.routed_experts_finalize,
            attn_res_backend=self.config.attn_res_backend,
            latent_moe_front=self.config.latent_moe_front)
        if loaded != self.config:
            raise ValueError(f"{source}/config.json is not the model the resources were declared for")

    def _load_checkpoint(self, language_model, source: str, device) -> None:
        """Every rank reads the text model's tensors whole and its loaders slice them; the vision
        tower (and the indexer, with DSA off) is never read."""
        from mstar.model.kimi_k3.weight_loader import PREFIX, kimi_k3_name_remapper
        from mstar.model.loader.iterators import iter_safetensors_shards

        keys = None
        index = Path(source) / "model.safetensors.index.json"
        if index.is_file():
            remap = kimi_k3_name_remapper(self.config.dsa_long_context)
            keys = {k for k in json.loads(index.read_text())["weight_map"]
                    if k.startswith(PREFIX) and remap(k[len(PREFIX):]) is not None}
        weights = iter_safetensors_shards(source, device=device, prefix=PREFIX, keys=keys)
        check_loaded(language_model, language_model.load_weights(weights), source)


class KimiK31Model(KimiK3Model):
    """Kimi K3.1 Pro (``moonshotai/Kimi-K3.1-Pro``): K3 with a DSA indexer on the MLA layers."""

    @staticmethod
    def default_config(**overrides) -> KimiK3ModelConfig:
        return KimiK3ModelConfig.k31(**overrides)
