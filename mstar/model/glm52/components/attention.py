"""GLM-5.2 MLA attention: the shared MLA (components/mla.py) and its DSA indexer, whose keys the index key array
holds and picks each query's positions from."""
from __future__ import annotations

import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.state import selects
from mstar.model.components.mla import MLAAttention
from mstar.model.glm52.components.indexer import Glm52Indexer, is_full_indexer_layer
from mstar.model.glm52.config import INDEX_KEYS, KV, Glm52ModelConfig


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


class CarriedPick:
    """The pick of the last FULL layer this forward, which the SHARED layers after it attend. Layer 0 is FULL
    (`Glm52Model.__init__`), so each forward writes it before a SHARED layer reads it."""

    picks: torch.Tensor | None = None


class Glm52MLAAttention(MLAAttention):
    """The shared MLA with GLM-5.2's DSA: on FULL layers an indexer projects each token's index
    key into the key array, and each query's heads, which pick its top ``index_topk`` positions;
    SHARED layers attend the last FULL layer's pick, which ``carried`` holds, and the latent read
    attends just those positions. ``dsa_selection`` is the naive path's reference hook: given
    rows, the mask over the current batch."""

    def __init__(
        self,
        config: Glm52ModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        carried: CarriedPick | None = None,
    ) -> None:
        super().__init__(config, comm_group, layer_idx, kv_key=KV)
        # DSA indexer only on FULL layers — SHARED layers carry no indexer
        # weights in the checkpoint (they reuse the last FULL selection),
        # and a dormant module would sit uninitialized after ``to_empty``.
        # And only under dsa_long_context, the one mode that declares the key
        # array: off it, a FULL layer's indexer (replicated, not TP-sharded)
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
        self._carried = carried if carried is not None else CarriedPick()
        self.index = None

    def bind_state(self, handles: dict) -> None:
        super().bind_state(handles)
        self.index = handles.get(INDEX_KEYS)

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
                "absorbed path picks through the key array instead"
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
        return self.index is not None

    def _select(self, hidden_states, q_c, position_ids, dsa_selection=None):
        # None-guarded here, not just inside _index_pass, so a step without DSA never crosses its graph break
        if self.index is None:
            return None
        return self._index_pass(hidden_states, q_c, position_ids)

    # The branch on `selects` and the indexer's query stay out of the compiled frame, and DSA steps run eager.
    @torch.compiler.disable
    def _index_pass(
        self, hidden_states: torch.Tensor, q_c: torch.Tensor, position_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Add this FULL layer's index keys and return the positions each row attends, or None while no row reaches
        more than ``index_topk`` positions. A SHARED layer takes the last FULL layer's pick."""
        if self.indexer is None:
            return self._carried.picks
        self.index.add(self.layer, k=self.indexer.compute_k(hidden_states, position_ids))
        picks = None
        if selects(self.index):
            q, weights = self.indexer.compute_query(q_c, hidden_states, position_ids)
            picks = self.index.read(self.layer, q, weights=weights)
        self._carried.picks = picks
        return picks
