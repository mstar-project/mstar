"""KimiK2Model: M* Model contract for Kimi-K2.7 (text backbone).

Kimi-K2.7's text path is DeepSeek-V3 (``model_type: "kimi_k2"`` ->
``DeepseekV3ForCausalLM``). This declares the full serving plumbing — the graph
(``prefill`` + ``decode`` Loop), the single ``KV_CACHE`` LLM node, the KV-cache
dims, and the prefill->decode->done state machine — and builds the LLM submodule
in ``get_submodule``. When ``get_submodule`` returns ``None`` (dummy mode),
``pytest test/modular/`` exercises the graph/walk/engine-routing machinery in
isolation, as ``docs/adding_models.rst`` prescribes.

Structurally this mirrors Orpheus's LLM partition (the smallest complete LLM in
the tree) minus the async SNAC partition: Kimi text-only is a single ``default``
partition, so it inherits ``Model.get_partitions`` / ``get_partition_topology``
and only implements the abstract surface.
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
        # ``model_kwargs`` from the serving YAML arrive here as ``**kwargs`` (see
        # api_server/entrypoint.py). They let a config redirect this model at a
        # local (reduced/synthetic) checkpoint without touching the shared
        # registry, so a runnable text serve is possible before the 1T weights
        # exist. All three are optional and default to the full-size behaviour.
        #   * ``checkpoint_path`` — local HF-format dir/file to load instead of
        #     the ``HF_MODELS`` repo id (used as-is by ``_resolve_checkpoint``).
        #   * ``config_variant`` — ``"reduced"`` selects ``KimiK2Config.reduced()``
        #     (tiny, GPU-runnable shape); anything else keeps the 1T config.
        #   * ``tokenizer_mode`` — ``"byte"`` swaps the HF tokenizer for a trivial
        #     UTF-8 byte identity tokenizer, the pragmatic fit for the reduced
        #     ``vocab_size=256`` model (the real Kimi tokenizer emits ids ≫ 256).
        checkpoint_path = kwargs.get("checkpoint_path")
        self.model_path_hf = checkpoint_path or model_path_hf
        self._config_variant = kwargs.get("config_variant", "full")
        if self._config_variant == "reduced":
            self.config = KimiK2Config.reduced()
        elif self._config_variant == "reduced_quantized":
            # Reduced shape + a quant config, to exercise dequant-on-load.
            self.config = KimiK2Config.reduced_quantized()
        elif self._config_variant == "reduced_quantized_inkernel":
            # Reduced shape + quant config + packed experts (in-kernel W4A16 dequant).
            # int32 packed params are auto-exempt from the whole-model ``.to(bf16)``
            # cast below (PyTorch ``.to(dtype)`` only casts float/complex), no hook.
            self.config = KimiK2Config.reduced_quantized_inkernel()
        elif self._config_variant == "k27_code":
            # Full-size Kimi-K2.7-Code text-only serve config (see KimiK2Config.k27_code).
            self.config = KimiK2Config.k27_code()
        else:
            self.config = KimiK2Config()
        self._tokenizer_mode = kwargs.get("tokenizer_mode", "hf")
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
        # Hopper prefill kernel static_asserts head_dim_vo in {64,128,256}. The
        # attention output is sliced back to ``v_head_dim`` in the submodule. This
        # trades cache size for not needing a weight-absorb path in the engine.
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
        if self._tokenizer_mode == "byte":
            # Trivial UTF-8 byte identity tokenizer for the reduced vocab_size=256
            # serve: each prompt byte is already a valid token id in [0, 256), so
            # no HF tokenizer / network dependency is needed. Clamped defensively
            # in case a smaller reduced vocab is ever used.
            vocab = self.config.vocab_size
            byte_ids = [min(b, vocab - 1) for b in prompt.encode("utf-8")] or [0]
            input_ids = torch.tensor(byte_ids, dtype=torch.long)
            return {"text_inputs": [input_ids]}
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
            if self._tokenizer_mode == "byte":
                # Inverse of the byte identity tokenizer: reduced-vocab ids map
                # straight back to raw bytes. The synthetic model emits arbitrary
                # ids in [0, 256), so the bytes are not guaranteed valid UTF-8 —
                # decode leniently (the point is to prove tokens stream, not to
                # produce meaningful text on random weights).
                return bytes((t & 0xFF) for t in token_ids)
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
        # YAML's ``node_groups``, not from the model code.
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

        # If the checkpoint declares a compressed-tensors ``quantization_config``,
        # route loading through the dequant-on-load parser (weight_loader). An
        # explicit config (e.g. reduced_quantized) is respected and not clobbered.
        self._maybe_apply_checkpoint_quant_config(source)

        # Real build, mirroring OrpheusModel._create_llm_submodule: construct on the
        # meta device (no allocation), cast to the target dtype on meta (so to_empty
        # allocates directly in bf16, not fp32-then-downcast), materialise storage,
        # then run the HF loader (remap + fused-expert stacked rules).
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

    def _maybe_apply_checkpoint_quant_config(self, source: str) -> None:
        """Populate ``self.config.quantization_config`` from ``config.json``.

        Reads the checkpoint's ``config.json`` ``quantization_config`` block (a
        compressed-tensors INT4/fp8 checkpoint carries one) and stores the parsed
        :class:`CompressedTensorsQuantConfig` on the model config so the weight
        loader takes the dequant-on-load path. A config set explicitly
        (e.g. ``reduced_quantized()``) wins and is left untouched; a plain bf16
        checkpoint (no block, unreadable, or single-file source) is a no-op.

        The real multimodal ``Kimi-K2.7-Code`` repo nests the block under
        ``text_config`` (the top-level ``quantization_config`` is null), so this
        reads a top-level block if present, otherwise ``text_config``'s.
        ``from_hf_config_dict`` parses the same block shape either way.
        """
        import json
        from pathlib import Path

        from mstar.model.kimi_k2_7.quantization import CompressedTensorsQuantConfig

        if self.config.quantization_config is not None:
            return
        config_json = Path(source) / "config.json"
        if not config_json.is_file():
            return
        try:
            with open(config_json) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:  # unreadable / malformed — stay bf16
            logger.warning("KimiK2Model: could not read %s: %s", config_json, e)
            return
        # A top-level ``quantization_config`` if present, else the one nested under
        # ``text_config`` (the multimodal K2.7-Code layout — top-level is null).
        quant_raw = raw.get("quantization_config") or (
            raw.get("text_config") or {}
        ).get("quantization_config")
        quant = CompressedTensorsQuantConfig.from_hf_config_dict(quant_raw)
        if quant is not None:
            logger.info(
                "KimiK2Model: compressed-tensors checkpoint (%d-bit, group_size=%d) "
                "— dequantizing on load.", quant.num_bits, quant.group_size,
            )
            self.config.quantization_config = quant
