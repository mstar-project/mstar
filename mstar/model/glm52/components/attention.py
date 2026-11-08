"""GLM-5.2 MLA attention: the shared MLA (components/mla.py) and its DSA indexer, whose keys and
queries the DSA resource (mstar/engine/resources/dsa) selects and attends with."""
from __future__ import annotations

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.mla import MLAAttention
from mstar.model.glm52.components.indexer import Glm52Indexer, is_full_indexer_layer
from mstar.model.glm52.config import (
    ATTN_RESOURCE,
    DSA_RESOURCE,
    KV_RESOURCE,
    POS_RESOURCE,
    Glm52ModelConfig,
)


def dsa_selection_to_mask(
    dsa_selection: torch.Tensor, num_keys: int, dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``(T, num_keys)`` mask from per-query top-k rows (-1 = padding)."""
    num_tokens = dsa_selection.shape[0]
    mask = torch.full(
        (num_tokens, num_keys), float("-inf"), dtype=dtype, device=device)
    valid = dsa_selection >= 0
    rows = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(valid)
    mask[rows[valid], dsa_selection[valid].long()] = 0.0
    return mask


def masked_reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Dense ``(T, H, D)`` attention with an additive ``(T, T)`` mask."""
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale + mask
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


class Glm52MLAAttention(MLAAttention):
    """The shared MLA with GLM-5.2's DSA: on FULL layers an indexer projects each token's index
    key and each query's heads, and the DSA resource picks each query's top ``index_topk`` keys;
    SHARED layers reuse the last FULL layer's pick, and the absorbed path attends just those
    latents. ``dsa_selection`` is the naive path's reference hook: given rows, the mask over
    the current batch."""

    def __init__(
        self,
        config: Glm52ModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(config, comm_group, layer_idx, kv_key=KV_RESOURCE,
                         attn_key=ATTN_RESOURCE, pos_key=POS_RESOURCE)
        # DSA indexer only on FULL layers — SHARED layers carry no indexer
        # weights in the checkpoint (they reuse the last FULL selection),
        # and a dormant module would sit uninitialized after ``to_empty``.
        # And only under dsa_long_context, the one mode that declares the DSA
        # resource: off it, a FULL layer's indexer (replicated, not TP-sharded)
        # would be read and held for nothing; the loader skips its keys to match
        # (Glm52ForCausalLM.load_weights).
        self.dsa_long_context = config.dsa_long_context
        self.indexer = (
            Glm52Indexer(config)
            if self.dsa_long_context
            and layer_idx is not None
            and is_full_indexer_layer(config, layer_idx)
            else None
        )
        self._dsa = None

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        self._dsa = resources.get(DSA_RESOURCE)
        if self._dsa is not None and not self.dsa_long_context:
            # built flag-off: no indexer was constructed and its weights were
            # never loaded, so a FULL layer here would pass for SHARED
            raise RuntimeError(
                "the DSA resource is declared but this attention layer was built with "
                "dsa_long_context off: its indexer does not exist"
            )
        if self._dsa is not None and not self.mla_absorb:
            # The sparse path reads the paged 576-dim latent cache; the naive
            # backend stores padded per-head K/V instead. The naive path stays
            # what it is: the reduced-test parity fallback.
            raise RuntimeError(
                "dsa_long_context requires mla_absorb: sparse attention reads the paged "
                "MLA latent cache, which only the absorbed backend maintains"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_selection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``dsa_selection``: optional ``(T, index_topk)`` int rows (-1 = padding) for the
        naive path's reference sparse attention."""
        if self.mla_absorb and dsa_selection is not None:
            raise NotImplementedError(
                "dsa_selection is the naive-path reference hook; the "
                "absorbed path selects through the DSA resource instead"
            )
        return super().forward(hidden_states, position_ids, dsa_selection=dsa_selection)

    def _attend_naive(self, q, k, v, dsa_selection=None) -> torch.Tensor:
        if dsa_selection is None:
            return self._write_and_attend(q, k, v)
        # Reference sparse path: softmax over the selected keys only,
        # via an additive mask over the current batch (a gather of the
        # selected k/v rows is semantically identical; -1 excluded).
        mask = dsa_selection_to_mask(dsa_selection, q.shape[0], dtype=q.dtype, device=q.device)
        return masked_reference_attention(q, k, v, mask)

    def _selects(self, dsa_selection=None) -> bool:
        return self._dsa is not None

    def _select(self, hidden_states, q_c, position_ids, dsa_selection=None):
        # None-guarded here (not just inside _dsa_select) so the default
        # flag-off path never crosses the compiler.disable boundary — dynamo
        # specializes on the unbound resource and folds the branch away.
        if self._dsa is None:
            return None
        return self._dsa_select(hidden_states, q_c, position_ids)

    # The layer index and the resource's host state are read only here, outside dynamo.
    @torch.compiler.disable
    def _dsa_select(
        self, hidden_states: torch.Tensor, q_c: torch.Tensor, position_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Hand the DSA resource this FULL layer's keys and return the pick to attend, or None
        while every row sees at most ``index_topk`` keys; a SHARED layer reuses the last pick."""
        dsa = self._dsa
        if self.indexer is None:
            return dsa.shared_selection(self.layer_idx) if dsa.needs_selection() else None
        dsa.store_keys(self.layer_idx, self.indexer.compute_k(hidden_states, position_ids))
        if not dsa.needs_selection():
            return None
        q, weights = self.indexer.compute_query(q_c, hidden_states, position_ids)
        return dsa.select(self.layer_idx, q, weights)

    @torch.compiler.disable
    def _attend_selected(
        self,
        selection: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        latent: torch.Tensor,
        dsa_selection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write this step's latents, then sparse absorbed MLA over the picked ones."""
        layer_idx = self.cache_layer_idx
        self._kv.write_kv(latent, None, layer_idx=layer_idx)
        return self._dsa.attend(
            selection, q_nope, q_pe, self._kv.layer_view(layer_idx), self.softmax_scale)
