"""Glm5NextModel: Model implementation for GLM-5.3-Flash (text generation)."""

import logging
import threading
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    NodeResourceSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
    SlotStateSpec,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.glm5_next.config import (
    ATTN,
    KDA_STATE,
    KV_CACHE,
    SAMPLER,
    Glm5NextModelConfig,
)
from mstar.model.glm5_next.kda_state import kda_slot_state_config
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


def process_weights_after_loading(root: torch.nn.Module, device: torch.device) -> None:
    """Finalize kernel layouts across a freshly-loaded module tree."""
    for module in root.modules():
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook) and module is not root:
            hook(device)


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


def _start_gpu_liveness_heartbeat(device: str) -> "threading.Event | None":
    """Tick a small CUDA kernel until the first real forward pass."""
    if not str(device).startswith("cuda"):
        return None
    stop = threading.Event()

    def _tick():
        # Sized to REGISTER in sampled utilization (~15-25%), not just to
        # execute: ~2.7 ms of matmul every 50 ms, 14 matmuls per 0.25 s wake
        # (glm52's measured-the-hard-way sizing; see glm52_model.py).
        a = torch.ones(8192, 8192, device=device, dtype=torch.bfloat16)
        while not stop.wait(0.25):
            try:
                for _ in range(14):
                    torch.mm(a, a)
            except torch.AcceleratorError:
                # A CUDA graph capture is in flight in the process (global
                # capture mode rejects unsafe calls from any thread). The
                # capture path stops this thread first; this is a race backstop.
                continue

    t = threading.Thread(target=_tick, daemon=True, name="glm5next-load-heartbeat")
    t.start()
    return stop


class Glm5NextModel(Model):
    """GLM-5.3-Flash: 320B hybrid KDA/MLA MoE causal LM, text in / text out."""

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
            # glm52's opt-in long-context flag exists on the config for
            # shape parity, but the glm5_next k-pool engine path (pooled
            # scoring + sparse gather) is not built yet — refuse loudly
            # instead of serving a silently-dense long context.
            raise NotImplementedError(
                "glm5_next dsa_long_context is a post-M1 follow-up: the "
                "k-pool indexer engine path is not implemented; serve "
                "within index_topk (2048) where dense MLA is exact DSA"
            )
        if "moe_quant_kernel" in kwargs:
            self.config.moe_quant_kernel = str(kwargs["moe_quant_kernel"])
        if "mtp_num_draft_tokens" in kwargs:
            self.config.mtp_num_draft_tokens = int(kwargs["mtp_num_draft_tokens"])
        # Slots in the KDA state pool = requests that may hold state at once
        # (the decode batch cap). A serve YAML can also set it under
        # ``resources: kda_state: {max_slots: N}``.
        self.kda_max_requests = int(kwargs.get("kda_max_requests", 32))
        # The KDA conv tail's dtype: the projection dtype at serve. bf16 is
        # the autocast dtype every serve config uses; the reduced CPU tests
        # build the model in fp32 and pass the matching dtype here.
        self.kda_conv_dtype = kwargs.get("kda_conv_dtype", torch.bfloat16)
        # "byte" maps UTF-8 bytes to token ids for reduced serve (no HF IO).
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        # Lazy: weights AND tokenizer load on demand so the conductor
        # process never touches the 306 GB checkpoint or HF IO in
        # dummy/byte mode. Token ids, chat template family, and the
        # transformers-4.x fallback are identical to GLM-5.2 (same eos/pad
        # ids — config-verified).
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
                # transformers-5 tokenizer_config declares a backend class
                # transformers 4.x cannot construct; tokenizer.json is
                # version-independent (glm52 lesson, verified on 4.57).
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
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The LLM node's four resources."""
        num_kv_layers = len(self.config.full_attn_layer_indices) + (
            1 if self.config.mtp_num_draft_tokens > 0 else 0
        )
        return [
            KVSpec(
                resource_key=KV_CACHE,
                nodes={"LLM"},
                config=KVConfig(
                    num_layers=num_kv_layers,
                    num_kv_heads=1,
                    head_dim=self.config.kv_lora_rank + self.config.mla_cache_kpe,
                    max_seq_len=self.config.max_seq_len,
                    num_qo_heads=self.config.num_attention_heads,
                    layout=KVLayout.MLA,
                ),
            ),
            AttentionSpec(
                resource_key=ATTN,
                nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE,
                    backend=AttnBackend.MLA,
                    mla_ckv_dim=self.config.kv_lora_rank,
                    softmax_scale=self.config.qk_head_dim ** -0.5,
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={"LLM"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
            SlotStateSpec(
                resource_key=KDA_STATE,
                nodes={"LLM"},
                config=kda_slot_state_config(
                    self.config, max_slots=self.kda_max_requests,
                    conv_dtype=self.kda_conv_dtype,
                ),
            ),
        ]

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
        if "top_k" in model_kwargs:
            params["top_k"] = int(model_kwargs["top_k"])
        if self.config.mtp_num_draft_tokens > 0 and "temperature" not in model_kwargs:
            # MTP v1 is greedy-only: greedy is the DECLARED default on MTP
            # configs so a bare request serves coherently; an EXPLICIT
            # temperature > 0 still reaches the refusal (glm52 policy).
            params["temperature"] = 0.0
        return {SAMPLER: SamplingReqConfig(**params)}

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # M2 note: glm52's MTP prefill adds a persisted MTP_DRAFT_BUNDLE
        # edge here (and its read half in the transition below). Port BOTH
        # halves together behind one flag helper — the write-gated/
        # read-live split and the "text_inputs" name collision are
        # documented glm52 regressions (2026-08-10); until the M2 loop
        # exists, the walk stays the plain shape even at k > 0.
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
            # Runaway guard; the per-request budget lives in check_stop
            # (M1). Keep this cap strictly below the preprocess context
            # guard — raising it toward max_seq_len converts a per-request
            # truncation into a batch-fatal context escape (glm52
            # 2026-08-10, tried and reverted).
            max_iters=self.get_max_output_tokens(),
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

        # Seed decode from the persisted emitted token. M2: the draft
        # bundle seed goes here under a DEDICATED name — never
        # "text_inputs", which the conductor pre-seeds with the PROMPT
        # (glm52's measured 17-row decode step + acceptance collapse).
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
            return {"text_inputs": [torch.tensor(byte_ids, dtype=torch.long)]}

        # Same chat-template family as GLM-5.2 ([gMASK]<sop> + assistant turn).
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]

        return {"text_inputs": [input_ids.to(torch.long)]}

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", self.config.max_output_tokens)

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
                # Synthetic reduced models emit arbitrary byte ids; never
                # touch the HF tokenizer.
                return bytes((t & 0xFF) for t in token_ids)
            return self.tokenizer.decode(
                token_ids, skip_special_tokens=True,
            ).encode("utf-8")
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

        # Detect the checkpoint's fp8-block quant config if the config did not
        # already carry one (the official checkpoint is fp8 e4m3 [128,128]).
        self._maybe_apply_checkpoint_quant_config(source)

        from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
        from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule

        # Build on meta -> optional autocast narrow -> to_empty(device) ->
        # fast read-plan load (restore_fp32_params runs inside load_weights) ->
        # process_weights_after_loading -> eval -> wrap. Mirrors glm52_model's
        # sequence. Building on meta first keeps the 306 GB allocation lazy
        # until to_empty. The KDA state pool is the engine's slot-state
        # resource (get_node_resources), built by the engine after this.
        with torch.device("meta"):
            language_model = Glm5NextForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        heartbeat_stop = _start_gpu_liveness_heartbeat(device)
        self._load_checkpoint(language_model, source, device, tp_group)
        process_weights_after_loading(language_model, torch.device(device))
        language_model.eval()
        # The declared conv-tail dtype must match the loaded projections.
        self.kda_conv_dtype = language_model.kda_conv_dtype()

        logger.info("Successfully loaded GLM-5.3-Flash submodule for %s", node_name)
        submodule = Glm5NextLLMSubmodule(
            language_model=language_model, config=self.config)
        # The heartbeat outlives the load until CUDA-graph capture starts
        # (the submodule stops it in get_cuda_graph_configs, or on its first
        # forward): the reaper's per-process idle clock doesn't care that the
        # box is busy reading a checkpoint.
        submodule.set_load_heartbeat_stop(heartbeat_stop)
        return submodule

    def _load_checkpoint(self, language_model, source: str, device, tp_group) -> None:
        """Load weights, taking the sliced TP fast read path when possible."""
        from mstar.model.glm5_next.weight_loader import build_glm5_next_read_plan
        from mstar.model.loader import load_weights
        from mstar.model.loader.iterators import iter_safetensors_shards

        index_file = Path(source) / "model.safetensors.index.json"
        if not index_file.is_file():
            load_weights(language_model, source, device=device)
            return

        import json

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
        language_model.load_weights(weights)

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        """Adopt the checkpoint's fp8-block quant config from its config.json."""
        import json

        from mstar.model.glm5_next.quantization import Fp8BlockQuantConfig

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
        return _resolve_local_hf_snapshot(path, cache_dir=self.cache_dir)
