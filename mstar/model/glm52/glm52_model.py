"""Glm52Model: Model implementation for GLM-5.2 (text generation)."""

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
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.engine.resources.attn.sparse_mla import check_flashinfer
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.glm52.components.indexer import full_indexer_layers
from mstar.model.glm52.config import (
    ATTN_RESOURCE,
    INDEX_KV_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

# fused_add_rmsnorm fuses all-reduces up to this size; larger ones reduce, add and norm apart
AR_FUSION_MAX_BYTES = 1 << 20


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    if Path(repo_id).is_dir():
        return repo_id
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        # offline with a cold cache, a gated repo, a full disk: returned as a local
        # path, the model built without its quant config and failed later, elsewhere
        raise RuntimeError(f"could not fetch {repo_id!r} from the Hugging Face Hub: {e}") from e
    return str(Path(local_dir))


class Glm52Model(Model):
    """GLM-5.2: 753B MoE causal LM (MLA + DSA), text in / text out."""

    # the conductor builds the fused-MoE align op before spawning the workers
    prebuild_fused_moe = True

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
            # Opt-in DSA engine path (configs/glm52_tp8_longctx.yaml). Sparse
            # attention reads the paged latent cache, so the naive
            # (mla_absorb=False) backend cannot host it.
            if not self.config.mla_absorb:
                raise ValueError(
                    "dsa_long_context requires mla_absorb: the sparse path "
                    "gathers selected latents from the paged MLA cache"
                )
            check_flashinfer()
            self.config.dsa_long_context = True
            if "prefill_chunk_tokens" in kwargs:
                self.config.prefill_chunk_tokens = int(kwargs["prefill_chunk_tokens"])
            self.config.dsa_shard_prefill = bool(kwargs.get("dsa_shard_prefill", False))
            # Guard + KV sizing move from index_topk to the serving window.
            self.config.max_seq_len = int(kwargs.get("max_seq_len", 8192))
        if "moe_quant_kernel" in kwargs:
            self.config.moe_quant_kernel = str(kwargs["moe_quant_kernel"])
        if "moe_fused_allreduce" in kwargs:
            self.config.moe_fused_allreduce = bool(kwargs["moe_fused_allreduce"])
        if "moe_decode_kernel" in kwargs:
            self.config.moe_decode_kernel = bool(kwargs["moe_decode_kernel"])
        if "moe_prefill_kernel" in kwargs:
            self.config.moe_prefill_kernel = bool(kwargs["moe_prefill_kernel"])
        if "moe_router_kernel" in kwargs:
            self.config.moe_router_kernel = bool(kwargs["moe_router_kernel"])
        if "prefill_last_layer_rows" in kwargs:
            self.config.prefill_last_layer_rows = bool(kwargs["prefill_last_layer_rows"])
        if "dense_fp8" in kwargs:
            self.config.dense_fp8 = bool(kwargs["dense_fp8"])
        if "mla_fused_prep" in kwargs:
            self.config.mla_fused_prep = bool(kwargs["mla_fused_prep"])
        if "fused_add_rmsnorm" in kwargs:
            self.config.fused_add_rmsnorm = bool(kwargs["fused_add_rmsnorm"])
        for key in ("prefill_token_buckets", "prefill_batched_token_buckets",
                    "prefill_capture_batch_sizes"):
            if key in kwargs:
                setattr(self.config, key, [int(n) for n in kwargs[key]])
        if kwargs.get("prefill_max_step_tokens") is not None:
            cap = kwargs["prefill_max_step_tokens"]
            self.config.prefill_max_step_tokens = cap if cap == "auto" else int(cap)
        if "mtp_num_draft_tokens" in kwargs:
            k = int(kwargs["mtp_num_draft_tokens"])
            if k < 0:
                # a negative k read as off in some places and on in others
                raise ValueError(f"mtp_num_draft_tokens must be 0 (off) or more, got {k}")
            self.config.mtp_num_draft_tokens = k
        if self.config.mtp_num_draft_tokens > 0:
            self._check_mtp(kwargs)
        # "byte" maps UTF-8 bytes to token ids for reduced serve (no HF IO).
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._detokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    def _check_mtp(self, kwargs: dict) -> None:
        if self._config_variant in ("reduced", "reduced_fp8"):
            from mstar.model.glm52.components.indexer import is_full_indexer_layer

            # reduced() sizes 2 trunk layers, landing the MTP position
            # (layer_idx = num_hidden_layers) on a SHARED indexer slot. Grow
            # the trunk so it lands FULL, as the real 78 = 2 + 19·freq
            # geometry does; real variants keep the MTP constructor's guard.
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
            from transformers import AutoTokenizer

            tokenizer_source = _resolve_local_hf_snapshot(
                self.model_path_hf, cache_dir=self.cache_dir,
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source, cache_dir=self.cache_dir,
                )
            except ValueError:
                # The checkpoint's tokenizer_config declares transformers-5's
                # TokenizersBackend class, which transformers 4.x cannot
                # construct — but the underlying tokenizer.json is
                # version-independent, so building a fast tokenizer straight
                # from it encodes identically.
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
        return tokenizer

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        # Absorbed MLA caches one latent row per token per layer
        # (kv_lora_rank + rope dims = 576) shared by all 64 query heads. No
        # Yarn -> the softmax scale is plain qk_head_dim**-0.5. With MTP on,
        # the layer-78 draft module keeps its KV in one extra layer plane at
        # index num_hidden_layers, on the trunk's page table (draft-tail rows
        # are overwritten as the verified stream advances into them).
        num_kv_layers = self.config.num_hidden_layers + (
            1 if self.config.mtp_num_draft_tokens > 0 else 0
        )
        nodes = {"LLM"}
        if self.config.mla_absorb:
            kv = PagedKVConfig(
                num_layers=num_kv_layers,
                num_kv_heads=1,
                head_dim=self.config.cache_latent_dim,
                max_seq_len=self.config.max_seq_len,
                num_qo_heads=self.config.num_attention_heads,
                layout=KVLayout.MLA,
                kv_lora_rank=self.config.kv_lora_rank,
                qk_rope_head_dim=self.config.qk_rope_head_dim,
            )
            attn = AttentionSpec(
                resource_key=ATTN_RESOURCE, nodes=nodes,
                config=AttentionConfig(
                    kv_cache=KV_RESOURCE,
                    backend=AttnBackend.FLASHINFER_MLA,
                    sm_scale=self.config.qk_head_dim ** -0.5,
                ),
            )
        else:
            # Naive fallback (reduced-test parity): full K/V padded to the
            # FlashInfer head size, one KV head per query head.
            kv = PagedKVConfig(
                num_layers=num_kv_layers,
                num_kv_heads=self.config.num_attention_heads,
                head_dim=self.config.padded_head_dim,
                max_seq_len=self.config.max_seq_len,
                num_qo_heads=self.config.num_attention_heads,
            )
            attn = AttentionSpec(
                resource_key=ATTN_RESOURCE, nodes=nodes,
                config=AttentionConfig(kv_cache=KV_RESOURCE),
            )
        specs = []
        if self.config.dsa_long_context:
            # the indexer's keys as a one-latent MLA cache, one layer per FULL layer
            d = self.config.index_head_dim
            specs.append(KVSpec(resource_key=INDEX_KV_RESOURCE, nodes=nodes, config=PagedKVConfig(
                num_layers=len(full_indexer_layers(self.config)) + (
                    1 if self.config.mtp_num_draft_tokens > 0 else 0),
                num_kv_heads=1, head_dim=d,
                max_seq_len=self.config.max_seq_len, layout=KVLayout.MLA, kv_lora_rank=d,
                qk_rope_head_dim=0,
                # replicated like the latent; its qo count only has to shard evenly under TP,
                # as the attention heads do
                num_qo_heads=self.config.num_attention_heads)))
        return specs + [
            KVSpec(resource_key=KV_RESOURCE, nodes=nodes, config=kv),
            attn,
            SamplerSpec(
                resource_key=SAMPLER_RESOURCE, nodes=nodes,
                vocab_size=self.config.vocab_size,
            ),
        ]

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
        prefill = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=prefill_outputs,
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
            # Runaway guard only: the per-request budget lives in check_stop,
            # which sees the request's real max_tokens. Sized to the most
            # decode steps any prompt can take before the context guard, not
            # to the default budget: at the default, every request asking for
            # more was cut short at 1025 tokens with no error.
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
    ) -> NameToTensorList:
        if prompt is None:
            return {}

        if self._tokenizer_mode == "byte":
            # Reduced serve maps UTF-8 bytes directly to token ids, avoiding HF IO.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            input_ids = torch.tensor(byte_ids, dtype=torch.long)
        # GLM-5.2 chat template (adds [gMASK]<sop> etc. and the assistant
        # turn). TODO: thinking mode / reasoning_effort dial once the
        # OpenAI adapter plumbs it through.
        elif getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                # transformers 5.x defaults return_dict=True (a BatchEncoding);
                # keep the bare-tensor return so [0] selects the row
                return_dict=False,
            )[0]
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        # here, in the data worker, a ValueError reaches the client as a 400
        if input_ids.numel() > self.config.max_prompt_tokens:
            raise ValueError(
                f"prompt is {input_ids.numel()} tokens; GLM-5.2 is served with "
                f"at most {self.config.max_prompt_tokens}"
            )
        return {"text_inputs": [input_ids.to(torch.long)]}

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        params = {
            k: model_kwargs.get(k, getattr(self.config, k))
            for k in keys
        }
        # the sampler's own knobs with no config default; dropped, a request got neither
        for k, cast in (("top_k", int), ("min_p", float)):
            if model_kwargs.get(k) is not None:
                params[k] = cast(model_kwargs[k])
        if self.config.mtp_num_draft_tokens > 0 and "temperature" not in model_kwargs:
            # MTP is greedy-only, so greedy is the declared default on MTP
            # configs: a bare request serves instead of inheriting the config
            # temperature and being refused by prepare_inputs. An explicit
            # temperature > 0 still refuses.
            params["temperature"] = 0.0
        return {SAMPLER_RESOURCE: SamplingReqConfig(**params)}

    def context_limit(self) -> int:
        """The bound the submodule's preprocess holds every context to:
        index_topk with DSA off (dense MLA is exact there), the serving
        window with it on."""
        return self.config.max_seq_len if self.config.dsa_long_context else self.config.index_topk

    def max_decode_steps(self) -> int:
        """Decode iterations a one-token prompt can run before the context
        guard; every longer prompt stops sooner, via check_stop or the guard.
        Under MTP each iteration emits at least one token, so it bounds those
        loops too."""
        return self.context_limit() - 1

    def get_max_output_tokens(self, **model_kwargs):
        # the request's budget, held to the window: a one-token prompt emits at
        # most ``context_limit`` tokens (the last is never stored), so a larger
        # budget is unreachable and check_stop could never fire on it
        budget = model_kwargs.get("max_output_tokens", self.config.max_output_tokens)
        return min(budget, self.context_limit())

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
            token_ids = output.flatten().tolist()
            if self._tokenizer_mode == "byte":
                # Synthetic reduced models emit arbitrary byte ids; return raw
                # bytes without ever touching the HF tokenizer.
                return bytes((t & 0xFF) for t in token_ids)
            # The tokens' raw bytes (byte-level BPE): a character split across
            # tokens reaches the client whole, where decoding each token alone
            # gives U+FFFD. Special tokens drop, as with skip_special_tokens.
            if self._detokenizer is None:
                from mstar.model.utils import ByteLevelDetokenizer

                self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
            return self._detokenizer.to_bytes(token_ids)
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
                "Glm52Model: no checkpoint resolved for node %r — dummy mode (None).",
                node_name,
            )
            return None

        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
        from mstar.model.glm52.quantization import process_weights_after_loading
        from mstar.model.glm52.submodules import Glm52LLMSubmodule

        with torch.device("meta"):
            language_model = Glm52ForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        self._load_checkpoint(language_model, source, device, tp_group)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()
        if self.config.fused_add_rmsnorm and tp_group is not None and tp_group.world_size > 1:
            # collective over the TP group; every rank builds the LLM here
            dtype = language_model.model.norm.weight.dtype
            tp_group.init_allreduce_fusion(
                self.config.hidden_size, dtype,
                max_tokens=AR_FUSION_MAX_BYTES // (self.config.hidden_size * dtype.itemsize),
            )

        logger.info("Successfully loaded GLM-5.2 submodule for %s", node_name)
        submodule = Glm52LLMSubmodule(language_model=language_model, config=self.config)
        return submodule

    def _load_checkpoint(self, language_model, source: str, device, tp_group) -> None:
        """Load weights, taking the sliced fast read path when possible."""
        from mstar.model.glm52.weight_loader import build_glm52_read_plan
        from mstar.model.loader import load_weights
        from mstar.model.loader.iterators import iter_safetensors_shards

        index_file = Path(source) / "model.safetensors.index.json"
        if not index_file.is_file():
            load_weights(language_model, source, device=device)
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
            source, device=device, keys=keys, slice_spec=specs.get,
        )
        language_model.load_weights(weights)

    def _resolve_checkpoint(self) -> str | None:
        path = getattr(self, "model_path_hf", None)
        if not path:
            return None
        if Path(path).exists():
            return str(path)
        return _resolve_local_hf_snapshot(path, cache_dir=self.cache_dir)

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        from mstar.model.glm52.quantization import Fp8BlockQuantConfig

        if self.config.quantization_config is not None:
            return
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            return
        try:
            with open(config_json) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:  # unreadable / malformed — stay bf16
            logger.warning("Glm52Model: could not read %s: %s", config_json, e)
            return
        quant = Fp8BlockQuantConfig.from_hf_config_dict(
            raw.get("quantization_config"),
        )
        if quant is not None:
            logger.info(
                "Glm52Model: fp8 %s checkpoint (block %s) — non-expert linears "
                "fp8=%s, routed experts fp8-resident=%s.",
                quant.fmt, quant.weight_block_size, self.config.dense_fp8,
                self.config.moe_fp8_resident,
            )
            self.config.quantization_config = quant
