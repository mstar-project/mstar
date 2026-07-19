"""Kimi-K2.7 language-model components (DeepSeek-V3 text backbone).

M1 (cheap reuse) wiring: builders that map ``KimiK2Config`` onto mstar's existing
reused primitives — token embedding, the dense SwiGLU MLP (the
``first_k_dense_replace`` early layers), RMSNorm, and the LM head. These are the
pieces DeepSeek-V3 shares verbatim with a standard Llama-style stack; the
Kimi-specific parts (MLA attention, fine-grained sigmoid-routed MoE, YARN RoPE)
are separate milestones (M2/M3).

Each builder is thin on purpose: it fixes the config→component mapping (dims,
``silu`` activation, ``bias=False``, RMSNorm eps, tied-vs-untied LM head) that M4
assembles into the full ``KimiLanguageModel``. Every builder matches the vLLM
DeepSeek-V3 reference:
  - embedding / LM head: ``deepseek_v2.py`` ``DeepseekV2ForCausalLM`` (untied,
    ``tie_word_embeddings=False``);
  - dense MLP: ``DeepseekV2MLP`` = ``down_proj(SiluAndMul(gate_up_proj(x)))``,
    ``bias=False``, silu-only;
  - RMSNorm: standard Llama-style ``x * rsqrt(mean(x^2)+eps) * weight``.
"""
from __future__ import annotations

from mstar.distributed.communication import CommGroup
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config


def build_embedding(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> VocabParallelEmbedding:
    """Token embedding ``[vocab, hidden]`` (row/vocab-parallel under TP)."""
    return VocabParallelEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        comm_group=comm_group,
        padding_idx=config.pad_token_id,
    )


def build_lm_head(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> ColumnParallelLinear:
    """Untied LM head ``[hidden, vocab]`` (Kimi: ``tie_word_embeddings=False``).

    Column-parallel over vocab with ``gather_output=True`` so the sampler always
    sees full ``[..., vocab]`` logits; a no-op all-gather at ``tp_size == 1``.
    """
    return ColumnParallelLinear(
        comm_group or CommGroup.trivial(),
        input_size=config.hidden_size,
        output_size=config.vocab_size,
        bias=False,
        gather_output=True,
    )


def build_rmsnorm(config: KimiK2Config) -> RMSNorm:
    """Standard Llama-style RMSNorm (not Gemma's ``(1 + weight)`` variant)."""
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def build_dense_mlp(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> ParallelGatedMLP:
    """Dense SwiGLU MLP for the ``first_k_dense_replace`` early layers.

    Matches ``DeepseekV2MLP``: fused gate/up projection, ``silu(gate) * up``,
    row-parallel down projection, ``bias=False``. Uses the full
    ``intermediate_size`` (the MoE layers use ``moe_intermediate_size`` per
    expert instead — that path is M2).
    """
    return ParallelGatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        comm_group=comm_group,
        activation=config.hidden_act,
        bias=False,
    )


def is_moe_layer(config: KimiK2Config, layer_idx: int) -> bool:
    """DeepSeek-V3 dense-vs-MoE layer selection.

    The first ``first_k_dense_replace`` layers are dense; thereafter every
    ``moe_layer_freq``-th layer is MoE (``deepseek_v2.py`` decoder-layer ctor).
    """
    return (
        layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def build_moe_block(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> KimiSparseMoeBlock:
    """Fine-grained MoE block (routed experts + ungated shared expert)."""
    return KimiSparseMoeBlock(config, comm_group=comm_group)


def build_mlp_for_layer(
    config: KimiK2Config, layer_idx: int, comm_group: CommGroup | None = None
):
    """Pick the layer's feed-forward: dense SwiGLU MLP or the MoE block.

    Returns a ``ParallelGatedMLP`` for the early dense layers, else a
    ``KimiSparseMoeBlock``. Both expose the same ``(x) -> x`` interface, so the
    decoder layer (M4) is agnostic to which it holds.
    """
    if is_moe_layer(config, layer_idx):
        return build_moe_block(config, comm_group=comm_group)
    return build_dense_mlp(config, comm_group=comm_group)
