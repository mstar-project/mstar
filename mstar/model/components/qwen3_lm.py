"""Qwen3 causal language models, dense and MoE, from the shared components.

Qwen3 = Llama-style pre-norm decoder with GQA, per-head QK RMSNorm, SwiGLU
MLP and plain RoPE (``rope_theta`` 1e6, no Llama-3 scaling). Several
audio-LLM checkpoints ship one as their text decoder (Qwen3-ASR, Higgs-Audio,
Qwen3-Omni's dense layers); this is the one place to build it from
``ParallelAttention`` / ``ParallelGatedMLP`` / ``RMSNorm``, bound to the
KV, attention and position resources of the node that owns it.
``Qwen3MoeLM`` is the same decoder with routed experts in place of the MLP
(HF's ``qwen3_moe``: Qwen3-30B-A3B, Qwen3-Coder).

Parameter paths mirror HF's ``Qwen3ForCausalLM`` / ``Qwen3MoeForCausalLM`` with
the ``model.`` prefix dropped (``embed_tokens.*``, ``layers.N.*``, ``norm.*``,
``lm_head.*``); a checkpoint that ties ``lm_head`` to the embedding loads
without the head key.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.components.norm import RMSNorm

if TYPE_CHECKING:
    from mstar.model.components.moe_parallel import MoeParallelConfig


class Qwen3Attention(ParallelAttention):
    """``ParallelAttention`` with Qwen3's per-head q/k RMSNorm, its 1D RoPE and
    the step's KV write in one Triton kernel (``qk_norm_rope_kv_write``). At
    decode width a step is a few small kernels per layer, and the separate
    norm, RoPE and KV-write kernels were a sizeable share of it. Falls back to
    the unfused path off CUDA or without planned positions."""

    _fuse_qk = False

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        # the kernel is plain non-interleaved RoPE over the whole head dim;
        # anything else the position resource would apply stays unfused
        self._rope_theta = self.pos.fusable_rope_theta(self.head_dim)
        self._fuse_qk = (
            self._rope_theta is not None and self.q_norm is not None and self.rope_scale == 1.0
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pos_ids = self.pos.pos_ids(self.attend.label) if self._fuse_qk else None
        if pos_ids is None or not hidden_states.is_cuda:
            return super().forward(hidden_states)
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        attn_output = self._norm_rope_attend(q, k, v, pos_ids)
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    @torch.compiler.disable
    def _norm_rope_attend(self, q, k, v, pos_ids) -> torch.Tensor:
        """q/k norm and RoPE, the step's KV write and attention. For a backend
        that reads K/V from the paged cache, the write rides in the norm/RoPE
        kernel (``qk_norm_rope_kv_write``) rather than a launch of its own."""
        from mstar.utils.attention import qk_norm_rope_inplace, qk_norm_rope_kv_write

        eps = (float(self.q_norm.variance_epsilon), float(self.k_norm.variance_epsilon), self._rope_theta)
        attend = self.attend
        if not attend.attn.requires_kv_write:
            qk_norm_rope_inplace(q, k, self.q_norm.weight, self.k_norm.weight, pos_ids, *eps)
            return attend(q, k, v)
        layer, page_idx, slot_idx, n = attend.kv.write_target()
        qk_norm_rope_kv_write(
            q, k, v, self.q_norm.weight, self.k_norm.weight, pos_ids,
            layer, page_idx, slot_idx, min(n, q.shape[0]), *eps,
        )
        return attend.attn.run(q, kv_cache_layer=attend.kv.layer_view())


@dataclass
class Qwen3LMConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    # qwen3_moe: routed experts replace the MLP on the layers ``is_moe_layer`` names
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts > 0
            and layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def from_hf(cls, hf: dict) -> "Qwen3LMConfig":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in hf.items() if k in names}
        values.setdefault("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
        return cls(**values)


class Qwen3DenseLM(nn.Module):
    def __init__(
        self,
        config: Qwen3LMConfig,
        *,
        attn_key: str,
        kv_key: str,
        pos_key: str,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        if config.num_experts and type(self) is Qwen3DenseLM:
            raise ValueError("a qwen3_moe config builds Qwen3MoeLM, not the dense Qwen3 LM")
        self.config = config
        comm_group = comm_group or CommGroup.trivial()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=comm_group,
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                self_attn=Qwen3Attention(
                    comm_group=comm_group,
                    hidden_size=config.num_attention_heads * config.head_dim,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    qk_norm=True,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                    input_hidden_size=config.hidden_size,
                    attn_key=attn_key,
                    kv_key=kv_key,
                    pos_key=pos_key,
                ),
                mlp=self._make_mlp(layer_idx, comm_group),
                input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )

    def _make_mlp(self, layer_idx: int, comm_group: CommGroup) -> nn.Module:
        return ParallelGatedMLP(
            comm_group=comm_group,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            activation="silu",
        )

    def _stacked_params(self) -> list:
        from mstar.model.loader import LLAMA_STACKED_PARAMS

        return LLAMA_STACKED_PARAMS

    def load_weights(self, weights) -> set[str]:
        """Stream HF ``(name, tensor)`` pairs (``model.`` prefix already
        dropped); tie the head to the embedding when the checkpoint does;
        raise on anything left unloaded."""
        from mstar.model.loader import load_hf_weights

        loaded = load_hf_weights(self, weights, stacked_params=self._stacked_params())
        if "lm_head.weight" not in loaded:
            if not self.config.tie_word_embeddings:
                raise RuntimeError("Qwen3 checkpoint has no lm_head.weight and does not tie embeddings")
            # Both shard the vocab dim by TP rank over the same rows, so the
            # local embedding shard is this rank's head shard.
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
            loaded.add("lm_head.weight")
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"Qwen3 checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}"
            )
        return loaded

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``(tokens, hidden)`` -> ``(tokens, hidden)`` after the final norm.

        The label and layer index are cursors on the shared resources: bind
        the label once, advance the index per layer. Passing them as
        arguments instead would make inductor specialize on the int.
        """
        hidden_states = input_embeds
        self.layers[0].self_attn.attend.bind_step(label)
        # Each residual add is deferred into the next norm: one fused add+norm
        # kernel (in place) per norm instead of an add and a norm, 56 fewer
        # launches a decode step on a 28-layer model.
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            if residual is None:
                residual = hidden_states
                h = layer.input_layernorm(hidden_states)
            else:
                h, residual = layer.input_layernorm.forward_residual(hidden_states, residual)
            h = layer.self_attn(h)
            h, residual = layer.post_attention_layernorm.forward_residual(h, residual)
            hidden_states = layer.mlp(h)
        normed, _ = self.norm.forward_residual(hidden_states, residual)
        return normed

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Qwen3MoeLM(Qwen3DenseLM):
    """HF's ``Qwen3MoeForCausalLM``: the dense decoder with a softmax top-k routed-expert block
    (no shared expert) in place of the MLP on every layer ``is_moe_layer`` names. The experts
    split by ``moe_parallel`` (TP by default, or EP)."""

    def __init__(
        self,
        config: Qwen3LMConfig,
        *,
        attn_key: str,
        kv_key: str,
        pos_key: str,
        comm_group: CommGroup | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ):
        from mstar.model.components.moe_parallel import MoeParallelConfig

        if not config.num_experts:
            raise ValueError("Qwen3MoeLM needs num_experts > 0")
        moe_parallel = moe_parallel or MoeParallelConfig()
        if moe_parallel.mask_padding or moe_parallel.log_expert_load_every:
            raise ValueError("Qwen3MoeLM does not mask padding rows or log expert load")
        # a plain attribute, set first: the base __init__ builds the layers through _make_mlp
        self.moe_parallel = moe_parallel
        super().__init__(config, attn_key=attn_key, kv_key=kv_key, pos_key=pos_key, comm_group=comm_group)

    def _make_mlp(self, layer_idx: int, comm_group: CommGroup) -> nn.Module:
        from mstar.model.components.moe_parallel import instantiate_moe_block

        if not self.config.is_moe_layer(layer_idx):
            return super()._make_mlp(layer_idx, comm_group)
        return instantiate_moe_block(
            hidden_size=self.config.hidden_size,
            num_experts=self.config.num_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            moe_intermediate_size=self.config.moe_intermediate_size,
            norm_topk_prob=self.config.norm_topk_prob,
            comm_group=comm_group,
            moe_parallel=self.moe_parallel,
        )

    def _stacked_params(self) -> list:
        from mstar.model.loader import LLAMA_STACKED_PARAMS, hf_expert_stacked_params

        return hf_expert_stacked_params(self.config.num_experts) + LLAMA_STACKED_PARAMS

    def load_weights(self, weights) -> set[str]:
        """As the dense LM's, and every expert's three projections must arrive: one of them
        fills a slice of a stacked parameter, which counts as loaded after its first."""
        import re

        expert_key = re.compile(r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
        seen: set[str] = set()

        def tally(pairs):
            for name, tensor in pairs:
                if expert_key.match(name):
                    seen.add(name)
                yield name, tensor

        loaded = super().load_weights(tally(weights))
        moe_layers = sum(self.config.is_moe_layer(i) for i in range(self.config.num_hidden_layers))
        missing = 3 * self.config.num_experts * moe_layers - len(seen)
        if missing:
            raise RuntimeError(f"Qwen3-MoE checkpoint left {missing} expert weight(s) unloaded")
        return loaded
