"""Kimi K3 for M*: the ``Model`` contract (tokenization, graph walks, resources,
forward-pass sequencing, submodule construction).

Text-only serving of the language model. The checkpoint (``model_path_hf``) may be a
local directory or a Hub id; ``moonshotai/Kimi-K3`` and its expert-pruned variants share
the same files. The vision tower is not loaded (a later ``prefill_vision`` walk adds it).
"""
from __future__ import annotations

import os

import logging
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
    RecurrentStateConfig,
    RecurrentStateSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
    StatePart,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.kimi_k3.config import KDA_STATE, MLA_ATTN, MLA_KV, SAMPLER, KimiK3Config
from mstar.model.kimi_k3.tokenizer import KimiK3Tokenizer
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


def _log_gpu_memory(stage: str, device, model: torch.nn.Module | None = None) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        params = "" if model is None else " (parameters %.2f GiB)" % (
            sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30)
        logger.info("Kimi K3 %s on %s: %.2f GiB allocated, %.2f GiB reserved%s", stage, dev,
                    torch.cuda.memory_allocated(dev) / 2**30, torch.cuda.memory_reserved(dev) / 2**30, params)

LLM = "LLM"


def _resolve_snapshot(repo_or_dir: str, cache_dir: str | None = None) -> str:
    if Path(repo_or_dir).is_dir():
        return str(Path(repo_or_dir))
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=repo_or_dir, cache_dir=cache_dir))


class KimiK3Model(Model):
    def __init__(self, model_path_hf: str, cache_dir: str | None = None, checkpoint_dir: str | None = None, **kwargs):
        """``checkpoint_dir`` (yaml ``model_kwargs``) is a local directory or HF repo id that
        overrides the registry's ``model_path_hf`` (the pruned/tiny checkpoints)."""
        self.cache_dir = cache_dir
        self.local_dir = _resolve_snapshot(checkpoint_dir or model_path_hf, cache_dir)
        self.config = KimiK3Config.from_hf_dir(self.local_dir)
        for key in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if key in kwargs:
                setattr(self.config, key, kwargs[key])
        self.default_thinking = bool(kwargs.get("thinking", True))
        # auto | marlin | w4a16 | humming | triton, see prepare_moe_kernels
        self.moe_backend = str(kwargs.get("moe_backend", "auto"))
        cap = kwargs.get("max_capture_batch_size")
        self.max_capture_batch_size = int(cap) if cap is not None else None
        self.tokenizer = KimiK3Tokenizer(self.local_dir)
        self.config.stop_token_ids = frozenset({self.tokenizer.eos_id, self.tokenizer.eot_id})
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # ------------------------------------------------------------- graph walks
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name=LLM, input_names=["text_inputs"],
            outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text",
                               conductor_new_token=True, persist=True)],
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name=LLM, input_names=["text_inputs"],
                outputs=[
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text",
                              conductor_new_token=True),
                    GraphEdge(next_node=LLM, name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return dict(prefill=prefill, decode=decode)

    def get_max_output_tokens(self, **model_kwargs) -> int:
        for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
            if model_kwargs.get(key) is not None:
                return int(model_kwargs[key])
        return self.config.max_output_tokens

    # ------------------------------------------------------------- prompts
    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        messages = kwargs.get("messages")
        thinking = bool(kwargs.get("thinking", self.default_thinking))
        if messages:
            ids = self.tokenizer.apply_chat_template(
                messages, thinking=thinking, thinking_effort=kwargs.get("thinking_effort"),
            )
        elif prompt is None:
            return {}
        elif kwargs.get("raw_prompt", False):
            ids = self.tokenizer.encode(prompt)
        else:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], thinking=thinking, thinking_effort=kwargs.get("thinking_effort"),
            )
        return {"text_inputs": [torch.tensor(ids, dtype=torch.long)]}

    # ------------------------------------------------------------- sequencing
    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities, output_modalities=output_modalities,
            graph_walk="prefill", is_prefill=True,
        )
        edge = GraphEdge(next_node=LLM, name="text_inputs")
        edge.tensor_info = input_signals.get("text_inputs", [])
        return ForwardPassArgs(
            full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self, partition_name: str, partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]], incoming_connections=None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            metadata.kwargs["decode_finished"] = True
            return ForwardPassArgs(full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True)
        edge = GraphEdge(next_node=LLM, name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(
            full_metadata=metadata, inputs=[edge], unpersist_tensors=list(edge.tensor_info),
            step_metadata={"is_prefill": False},
        )

    # ------------------------------------------------------------- resources
    def get_node_resources(self) -> list[NodeResourceSpec]:
        t = self.config.text
        kv_config = KVConfig(
            num_layers=t.num_mla_layers, num_kv_heads=1, head_dim=t.mla_kv_latent_dim,
            max_seq_len=t.max_position_embeddings, num_qo_heads=t.num_attention_heads,
            layout=KVLayout.MLA, kv_lora_rank=t.kv_lora_rank, qk_rope_head_dim=t.qk_rope_head_dim,
        )
        p = t.kda_projection_size
        state_config = RecurrentStateConfig(
            num_layers=t.num_kda_layers,
            parts={
                "conv": StatePart((3 * p, t.kda_conv_kernel_size), torch.bfloat16, shard_dim=0),
                "recurrent": StatePart((t.kda_num_heads, t.kda_head_dim, t.kda_head_dim), torch.float32, shard_dim=0),
            },
        )
        return [
            KVSpec(resource_key=MLA_KV, nodes={LLM}, config=kv_config),
            AttentionSpec(
                resource_key=MLA_ATTN, nodes={LLM},
                config=AttentionConfig(kv_cache=MLA_KV, backend=AttnBackend.FLASHINFER_MLA, sm_scale=t.mla_scale),
            ),
            RecurrentStateSpec(resource_key=KDA_STATE, nodes={LLM}, config=state_config),
            SamplerSpec(resource_key=SAMPLER, nodes={LLM}, vocab_size=t.vocab_size, enable_repetion_penalty=False),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs], model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "top_k", "ignore_eos"]
        return {SAMPLER: SamplingReqConfig(**{k: model_kwargs.get(k, getattr(self.config, k)) for k in keys})}

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        return ShardingConfig(groups=[], tp_enabled_nodes={LLM}, shard_dim={})

    # ------------------------------------------------------------- outputs
    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality != "text":
            raise ValueError(f"Kimi K3 emits text only, got {modality!r}")
        ids = [int(i) for i in output.reshape(-1).tolist()]
        # raw bytes so a multi-byte character split across tokens is reassembled client-side
        return self.tokenizer.hf.model.decode_bytes(ids)

    # ------------------------------------------------------------- submodules
    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None, autocast_dtype: torch.dtype | None = None, **kwargs,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        if node_name != LLM:
            return None
        from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
        from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
        from mstar.model.loader import load_weights

        quantized = self.config.quant is not None and self.config.quant.is_mxfp4
        with torch.device("meta"):
            language_model = KimiK3ForCausalLM(self.config.text, comm_group=tp_group, quantized_experts=quantized)
        dtype = autocast_dtype or torch.bfloat16
        language_model = language_model.to(dtype)
        # fp32 parameters (A_log, dt_bias, router bias) and the uint8 packed experts must
        # not take the autocast dtype: re-cast after the sweep, on meta, before allocation
        for name, param in language_model.named_parameters():
            if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
                param.data = param.data.to(torch.float32)
            elif getattr(param, "_keep_dtype", False) or name.endswith(("_packed", "_scale")):
                param.data = param.data.to(torch.uint8)
        language_model.to_empty(device=device)
        if os.environ.get("MSTAR_KIMI_K3_RANDOM_INIT") == "1":
            # serve a config without its checkpoint (memory / performance experiments):
            # random weights of the right dtypes, packed experts included
            logger.warning("MSTAR_KIMI_K3_RANDOM_INIT=1: random weights, the outputs are meaningless")
            with torch.no_grad():
                for name, prm in language_model.named_parameters():
                    if prm.dtype == torch.uint8:
                        prm.copy_(torch.randint(118, 124, prm.shape, dtype=torch.uint8, device=device)
                                  if name.endswith("_scale") else
                                  torch.randint(0, 256, prm.shape, dtype=torch.uint8, device=device))
                    elif name.endswith("A_log"):
                        prm.uniform_(0.0, 1.0)
                    elif name.endswith("dt_bias"):
                        prm.uniform_(-2.0, 0.0)
                    elif "norm" in name and prm.ndim == 1:
                        prm.fill_(1.0)
                    else:
                        prm.normal_(std=0.02)
        else:
            load_weights(language_model, self.local_dir, device=device)
        language_model.eval()
        _log_gpu_memory("weights loaded", device, language_model)
        from mstar.engine.resources.attn.flashinfer_mla import flashinfer_mla_supports
        from mstar.model.kimi_k3.components.language_model import prepare_moe_kernels, select_kda_kernels

        moe_backend = prepare_moe_kernels(language_model, device, self.moe_backend)
        logger.info("Kimi K3 routed-expert backend: %s", moe_backend)
        _log_gpu_memory("experts converted", device, language_model)

        graph_safe = select_kda_kernels(language_model, device) and flashinfer_mla_supports(
            self.config.text.kv_lora_rank, self.config.text.qk_rope_head_dim
        )
        submodule = KimiK3LLMSubmodule(language_model=language_model, config=self.config, cuda_graphs=graph_safe,
                                       max_capture_batch_size=self.max_capture_batch_size)
        self._submodule_cache[node_name] = submodule
        logger.info("Loaded Kimi K3 %s on %s", node_name, device)
        return submodule
