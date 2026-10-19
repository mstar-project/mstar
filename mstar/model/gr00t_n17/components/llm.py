"""GR00T N1.7's backbone language model: the first ``select_layer`` layers of
the Qwen3-VL text stack, run once per request with no KV cache.

Qwen3.5's own layer does not fit -- its attention carries an output gate and
Gemma-style norms, and Qwen3-VL's has neither -- but everything around it does:
the shared ``DecoderLayer``, ``ParallelGatedMLP`` and ``RMSNorm``, and
``qwen3_5``'s interleaved M-RoPE. So this module is one attention class and the
assembly around it.

There is no final norm and no ``lm_head``. The reference reads
``hidden_states[-1]`` off ``Qwen3VLForConditionalGeneration``, which is the
last decoder layer's output *before* the norm, and the loader skips both by
name. Check that at the same level the reference calls: ``Qwen3VLTextModel``
on its own reports the *normed* state at ``[-1]`` instead (C04 CHK-2).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import DecoderLayer, RMSNorm
from mstar.model.components.distributed import (
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.gr00t_n17.config import LLM_ATTN, Gr00tN17Config, Gr00tN17TextConfig
from mstar.model.qwen3_5.components.language_model import RopeCache
from mstar.model.qwen3_5.components.rope import (
    apply_partial_mrope,
    compute_3d_cos_sin,
    compute_inv_freq,
)


class Gr00tN17Attention(nn.Module):
    """GQA with qk-norm, rotated by 3D M-RoPE, over one ragged segment per
    request.

    Not the shared ``Attention`` base: that one binds a KV cache and a position
    resource, and this attention is one-shot -- a request's whole prompt in a
    single pass, with nothing kept between steps. Causality is the step's,
    declared once in ``declare_step``, not a flag here.
    """

    def __init__(
        self,
        config: Gr00tN17TextConfig,
        rope_cache: RopeCache,
        comm_group: CommGroup | None = None,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.head_dim = config.head_dim
        self.rope_cache = rope_cache
        self.qkv_proj = QKVParallelLinear(
            comm_group=comm_group,
            hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            comm_group=comm_group,
            input_size=config.num_attention_heads * config.head_dim,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
        )
        self.ragged_attn = None

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``. ``.get``: a deployment
        without the LLM node binds nothing here."""
        self.ragged_attn = resources.get(LLM_ATTN)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ragged_attn is None:
            raise RuntimeError(
                "GR00T's LLM has no ragged attention resource; declare a "
                f"RaggedAttentionSpec under {LLM_ATTN!r} for the LLM node"
            )
        num_tokens = hidden_states.shape[0]
        q, k, v = self.qkv_proj(hidden_states).split(self.qkv_proj.output_sizes, dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        # norm then rotate, as the reference does
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_partial_mrope(q, k, self.rope_cache.cos, self.rope_cache.sin)
        out = self.ragged_attn.run(q, k, v)
        return self.o_proj(out.reshape(num_tokens, -1))


class Gr00tN17LanguageModel(nn.Module):
    """Embedding plus the kept decoder layers. Token ids in, hidden states out.

    ``embed`` and the layer loop are separate so the caller can put the vision
    embeddings on the image rows in between.
    """

    def __init__(self, config: Gr00tN17Config, comm_group: CommGroup | None = None):
        super().__init__()
        text = config.text
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.config = text
        self.embed_tokens = VocabParallelEmbedding(
            text.vocab_size, text.hidden_size, comm_group=comm_group,
        )
        self.rope_cache = RopeCache()
        self.layers = nn.ModuleList(
            DecoderLayer(
                self_attn=Gr00tN17Attention(text, self.rope_cache, comm_group),
                mlp=ParallelGatedMLP(
                    hidden_size=text.hidden_size,
                    intermediate_size=text.intermediate_size,
                    comm_group=comm_group,
                    activation="silu",
                ),
                input_layernorm=RMSNorm(text.hidden_size, eps=text.rms_norm_eps),
                post_attention_layernorm=RMSNorm(
                    text.hidden_size, eps=text.rms_norm_eps,
                ),
            )
            for _ in range(text.num_hidden_layers)
        )
        self.register_buffer(
            "inv_freq",
            compute_inv_freq(text.head_dim, text.rope_theta),
            persistent=False,
        )

    def build_cos_sin(
        self, position_ids_3d: torch.Tensor, dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's rotation, from `mrope_position_ids`' ``[3, tokens]``."""
        return compute_3d_cos_sin(
            position_ids_3d, self.inv_freq, list(self.config.mrope_section), dtype,
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        # one rotation for the whole step, as `qwen3_5` does, because
        # `DecoderLayer.forward` passes its attention nothing but the rows
        self.rope_cache.set(cos, sin)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
