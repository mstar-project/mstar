"""Kimi K3's gated MLA and K3.1's DSA indexer, on the shared MLA (``components/mla.py``).

The MLA is DeepSeek-V3's with Kimi's NoPE (the 64 decoupled dims kept, unrotated) and a
sigmoid output gate, ``o_proj(attn * sigmoid(g_proj(x)))``. K3.1's layers add a DeepSeek-V3.2
indexer (no rope, LayerNorm key norm); its keys and queries go to the DSA resource
(``mstar/engine/resources/dsa``), which scores them and picks each query's top
``index_topk``, as glm52's do. The indexer math is DeepSeek-V3.2's: the checkpoint ships no
modeling code for it, so that much is a hypothesis to check against vLLM / SGLang.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear
from mstar.model.components.mla import MLAAttention
from mstar.model.kimi_k3.config import (
    ATTN,
    DSA,
    INDEXER_K_NORM_EPS,
    KV_CACHE,
    LABEL,
    KimiK3ModelConfig,
)


class KimiIndexer(nn.Module):
    """One layer's DSA indexer: each token's index key and each query's heads and head weights.
    Replicated (7.4M parameters a layer)."""

    def __init__(self, config: KimiK3ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=INDEXER_K_NORM_EPS)  # weight + bias
        # the softmax scale and n_heads^-0.5 fold into the head weights, as DeepSeek-V3.2 does
        self.weight_scale = self.head_dim ** -0.5 * self.n_heads ** -0.5

    def compute_k(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(T, head_dim)``: what the DSA resource stores per token."""
        return self.k_norm(self.wk(hidden_states))

    def compute_query(
        self, q_c: torch.Tensor, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Queries ``(T, H, D)`` and head weights ``(T, H)``, fp32, from the normed q latent."""
        q = self.wq_b(q_c).view(q_c.shape[0], self.n_heads, self.head_dim).float()
        w = self.weights_proj(hidden_states).float() * self.weight_scale
        return q, w


class KimiMLAAttention(MLAAttention):
    """The shared MLA over this layer's plane of the latent cache (``kv_plane``: its index among
    the MLA layers), with the output gate and, under ``dsa_long_context``, the indexer."""

    def __init__(
        self,
        config: KimiK3ModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        kv_plane: int | None = None,
    ) -> None:
        super().__init__(config, comm_group, layer_idx, kv_key=KV_CACHE, attn_key=ATTN,
                         pos_key=None, kv_label=LABEL, cache_layer=kv_plane)
        self.kv_plane = kv_plane
        self.g_proj = None
        if config.mla_use_output_gate:
            self.g_proj = ColumnParallelLinear(
                comm_group or CommGroup.trivial(), config.hidden_size,
                config.num_attention_heads * config.v_head_dim, bias=False)
        # Built only where the DSA resource is declared: off it the replicated indexer would be
        # held for nothing, and the loader skips its keys to match.
        self.indexer = KimiIndexer(config) if config.dsa_long_context else None
        self._dsa = None

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        self._dsa = resources.get(DSA)
        if self._dsa is not None and self.indexer is None:
            raise RuntimeError("the DSA resource is declared but this layer has no indexer "
                               "(built with dsa_long_context off)")
        if self._dsa is not None and not self.mla_absorb:
            raise RuntimeError("DSA reads the paged latent cache, which only the absorbed MLA keeps")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states (T, hidden)`` in the engine's flat layout; NoPE takes no positions."""
        return super().forward(hidden_states, None)

    def _project_out(self, attn: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.g_proj is not None:
            attn = attn * torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(attn)

    # -- DSA (dsa_long_context) ------------------------------------------------

    def _selects(self, **sparse) -> bool:
        return self._dsa is not None

    def _select(self, hidden_states, q_c, position_ids, **sparse):
        # None-guarded here so the dense path never crosses the compiler.disable boundary
        if self._dsa is None:
            return None
        return self._dsa_select(hidden_states, q_c)

    # The layer index and the resource's host state are read only here, outside dynamo.
    @torch.compiler.disable
    def _dsa_select(self, hidden_states: torch.Tensor, q_c: torch.Tensor) -> torch.Tensor | None:
        """Store this layer's index keys; past ``index_topk`` return each row's pick."""
        dsa = self._dsa
        dsa.store_keys(self.layer_idx, self.indexer.compute_k(hidden_states), label=LABEL)
        if not dsa.needs_selection(LABEL):
            return None
        q, weights = self.indexer.compute_query(q_c, hidden_states)
        return dsa.select(self.layer_idx, q, weights, label=LABEL)

    @torch.compiler.disable
    def _attend_selected(self, selection, q_nope, q_pe, latent, **sparse) -> torch.Tensor:
        """Write this step's latents, then sparse absorbed MLA over the picked ones."""
        layer = self.cache_layer_idx
        self._kv.write_kv(latent, None, layer_idx=layer, **self._label)
        return self._dsa.attend(selection, q_nope, q_pe, self._kv.layer_view(layer),
                                self.softmax_scale, label=LABEL)
