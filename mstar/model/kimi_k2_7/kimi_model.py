"""KimiK2Model: M* Model contract for Kimi-K2.7 (text backbone).

Kimi-K2.7's text path is DeepSeek-V3 (``model_type: "kimi_k2"`` →
``DeepseekV3ForCausalLM``). This is the **M0 scaffold**: it declares the full
serving plumbing — the graph (``prefill`` + ``decode`` Loop), the single
``KV_CACHE`` LLM node, the KV-cache dims, and the prefill→decode→done state
machine — with **no GPU compute**. ``get_submodule`` returns ``None`` (dummy
mode), so ``pytest test/modular/`` exercises the graph/walk/engine-routing
machinery in isolation, exactly as ``docs/adding_models.rst`` prescribes for a
new model before touching weights.

Structurally this mirrors Orpheus's LLM partition (the smallest complete LLM in
the tree) minus the async SNAC partition: Kimi text-only is a single ``default``
partition, so it inherits ``Model.get_partitions`` / ``get_partition_topology``
and only implements the abstract surface.

Later milestones fill in the real compute (M2 MoE router, M3 MLA attention, M5
weights, M6 the ``KimiLLMSubmodule`` build in ``get_submodule``); none of them
change the contract declared here.
"""
from __future__ import annotations

import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_cache_engine import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)

LLM_NODE = "LLM"
DECODE_LOOP = "decode_loop"


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    """Resolve an HF repo id to a local snapshot dir (mirrors OrpheusModel)."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id, cache_dir=cache_dir, local_files_only=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Error downloading %r from huggingface: %s", repo_id, e)
        return repo_id
    return str(Path(local_dir))


class KimiK2Model(Model):
    """Kimi-K2.7 text backbone (DeepSeek-V3 architecture)."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        self.config = KimiK2Config()
        # Tokenizer is loaded lazily: the modular (dummy-mode) tests build the
        # model via ``object.__new__`` and never call ``__init__``, so we avoid
        # forcing a network/tokenizer dependency into the scaffold path.
        self._tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_hf,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        return self._tokenizer

    # -------------------------------------------------------------------
    # Model ABC: KV cache config
    # -------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        # Naive/materialized MLA (the first-pass port, per CLAUDE.md): the latent
        # is projected up to full per-head K/V and broadcast to every query head,
        # so from the paged cache's ``[tokens, heads, head_dim]`` point of view
        # there are ``num_attention_heads`` KV heads. K/V are stored at
        # ``padded_head_dim`` — the naive path zero-pads q/k (from ``qk_head_dim``,
        # e.g. 192) and v (from ``v_head_dim``) up to the smallest FlashInfer-SM90
        # supported head_dim >= qk_head_dim (256 real, 64 reduced), because the
        # Hopper prefill kernel static_asserts head_dim_vo in {64,128,256} (M4
        # finding). The attention output is sliced back to ``v_head_dim`` in the
        # submodule (M3/M6). This trades cache size for not needing a weight-absorb
        # path in the engine (deferred to perf).
        return [KVCacheConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_attention_heads,
            head_dim=self.config.padded_head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
        )]

    # -------------------------------------------------------------------
    # Model ABC: node engine types
    # -------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {LLM_NODE: EngineType.KV_CACHE}

    # -------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # prefill: embed the prompt, fill the KV cache, sample + emit the first
        # token. ``persist=True`` keeps that token at the conductor so the decode
        # walk can pick it up as its first ``text_inputs``.
        prefill = GraphNode(
            name=LLM_NODE,
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

        # decode: autoregressive Loop. Each step emits the new token to the
        # client and feeds it back as the next step's ``text_inputs``. The Loop
        # stops via the submodule's ``check_stop`` (EOS / max tokens); ``max_iters``
        # is the hard cap.
        decode = Loop(
            name=DECODE_LOOP,
            section=GraphNode(
                name=LLM_NODE,
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                    GraphEdge(
                        next_node=LLM_NODE,
                        name="text_inputs",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: forward pass args (single "default" partition)
    # -------------------------------------------------------------------

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

        graph_edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
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
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Drive the prefill → decode → done state machine.

        Called by the conductor after each completed walk. Prefill transitions to
        the decode walk (feeding the persisted first token as ``text_inputs``);
        once the decode walk completes (its Loop stopped via ``check_stop``), the
        request is done. The per-token decode iteration is driven inside the
        graph Loop, not by repeated calls here.
        """
        metadata = partition_metadata
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

        graph_edge = GraphEdge(next_node=LLM_NODE, name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [graph_edge]
        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])

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
        # Text-only for M0; raw multimodal tensors (MoonViT) are a later milestone.
        if prompt is None:
            return {}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return {"text_inputs": [input_ids]}

    def get_sampling_config(
        self,
        node_name: str,
        model_kwargs: dict | None = None,
    ) -> SamplingConfig | None:
        model_kwargs = model_kwargs or {}
        return SamplingConfig(
            vocab_size=self.config.vocab_size,
            temperature=model_kwargs.get("temperature", self.config.temperature),
            top_p=model_kwargs.get("top_p", self.config.top_p),
            ignore_eos=model_kwargs.get("ignore_eos", self.config.ignore_eos),
        )

    # -------------------------------------------------------------------
    # Model ABC: postprocess
    # -------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            token_ids = output.tolist() if output.numel() else []
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return text.encode("utf-8")
        raise ValueError(f"Unsupported modality for Kimi-K2.7: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: sharding
    # -------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # Kimi is a 1T model — real serving is TP8 / multi-node. The LLM node is
        # the tensor-parallel node; the per-node degree comes from the config
        # YAML's ``node_groups`` (M6), not from the model code.
        return ShardingConfig(groups=[], tp_enabled_nodes={LLM_NODE}, shard_dim={})

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
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
        if node_name != LLM_NODE:
            return None

        source = self._resolve_checkpoint()
        if source is None:
            # Dummy mode: no checkpoint resolvable (e.g. the modular graph tests
            # build the model via object.__new__ with no model_path_hf). Returning
            # None lets pytest test/modular/ validate the graph/walks/engine-routing
            # without a GPU or weights, per docs/adding_models.rst.
            logger.info(
                "KimiK2Model: no checkpoint resolved for node %r — dummy mode (None).",
                node_name,
            )
            return None

        # Real build, mirroring OrpheusModel._create_llm_submodule: construct on the
        # meta device (no allocation), cast to the target dtype on meta (so to_empty
        # allocates directly in bf16, not fp32-then-downcast), materialise storage,
        # then run the M5 HF loader (remap + fused-expert stacked rules). This is the
        # path the M5 rope-buffer bug would have bitten — inv_freq is lazy so it does
        # not survive as garbage.
        from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
        from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule
        from mstar.model.loader import load_weights

        with torch.device("meta"):
            language_model = KimiForCausalLM(self.config, comm_group=tp_group)
        if autocast_dtype is not None:
            language_model = language_model.to(autocast_dtype)
        language_model.to_empty(device=device)
        load_weights(language_model, source, device=device)
        language_model.eval()

        logger.info("Successfully loaded Kimi-K2.7 submodule for %s", node_name)
        return KimiLLMSubmodule(language_model=language_model, config=self.config)

    def _resolve_checkpoint(self) -> str | None:
        """Resolve the HF checkpoint source, or None for dummy mode.

        A local directory / file (e.g. a reduced synthetic checkpoint) is used
        as-is; otherwise the HF repo id is snapshot-downloaded. Returns None when
        no ``model_path_hf`` is set (dummy-mode graph tests).
        """
        from pathlib import Path

        path = getattr(self, "model_path_hf", None)
        if not path:
            return None
        if Path(path).exists():
            return str(path)
        return _resolve_local_hf_snapshot(path, cache_dir=getattr(self, "cache_dir", None))
