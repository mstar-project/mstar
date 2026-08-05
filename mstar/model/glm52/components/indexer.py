"""GLM-5.2 DSA sparse-attention indexer (Phase C, CPU-testable reference).

The indexer is a tiny MQA scorer that picks, per query token, the
``index_topk`` prefix positions the main MLA attention is allowed to see.
Geometry (full model): 32 heads x 128 dims against a SINGLE 128-dim key per
token, scored as a ReLU-weighted sum over heads, top-2048 over the causal
window. Only FULL layers (see ``is_full_indexer_layer``) carry these
weights; SHARED layers reuse the most recent FULL layer's selection.

This module is the bf16/fp32 semantic reference: vLLM quantizes q and the
cached k to fp8 e4m3 with ue8m0 scales, which the spec (dsa-indexer-spec.md
section 6) marks freely substitutable — at ctx <= index_topk the selected
SET is provably identical regardless of score precision. Engine paged-cache
integration (fp8 132-byte k records, per-request block tables) is the
marked Phase C follow-up; here ``compute_k`` returns what that cache would
store and ``compute_selection`` scores against a dense history tensor.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.glm52.components.rope import Glm52RotaryEmbedding
from mstar.model.glm52.config import Glm52ModelConfig


def is_full_indexer_layer(config: Glm52ModelConfig, layer_idx: int) -> bool:
    """FULL layers run the indexer; SHARED layers reuse the last FULL selection.

    Exact vLLM skip formula (ref_deepseek_v2.py:1029-1031):
    ``skip = max(layer_idx - offset + 1, 0) % freq != 0``. GLM's offset=3
    makes layers 0..2 FULL and anchors the every-``freq`` series at layer
    offset-1 = 2, so subsequent FULL layers are 6, 10, ..., 74 — NOT
    "pattern starts at layer offset".
    """
    skip = (
        max(layer_idx - config.index_skip_topk_offset + 1, 0)
        % config.index_topk_freq
        != 0
    )
    return not skip


class Glm52Indexer(nn.Module):
    """DSA indexer for one FULL layer: k projection/cache side + selection side.

    Checkpoint modules (all under ``self_attn.indexer.``): ``wq_b`` and
    ``wk`` arrive fp8+block-scales (dequantized to bf16 on load), and
    ``weights_proj`` / ``k_norm`` arrive bf16. ``k_norm`` is a full
    LayerNorm (mean-subtract, weight AND bias) with eps hardcoded to 1e-6
    (ref_deepseek_v2.py:644) — NOT RMSNorm, NOT rms_norm_eps.
    """

    def __init__(self, config: Glm52ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk = config.index_topk

        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = torch.nn.LayerNorm(self.head_dim, eps=1e-6)  # weight + bias

        # Same interleaved rotation, theta, and absolute positions as the
        # main MLA rope (indexer_rope_interleave=True => GPT-J pairing).
        self.rotary = Glm52RotaryEmbedding(
            rotary_dim=config.qk_rope_head_dim, base=config.rope_theta)
        # softmax_scale (head_dim^-0.5) AND n_heads^-0.5 both fold into the
        # per-head weights (ref_deepseek_v2.py:738-741). vLLM also folds the
        # fp8 q dequant scale here; the bf16/fp32 path has none.
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5

    def _rope_first_dims(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the FIRST ``rope_dim`` dims of ``(T, H, D)`` ``x``.

        Opposite of the main MLA, where rope is the LAST 64 of 256
        (ref_deepseek_v2.py:702-725) — the classic reversed-slice trap.
        """
        pe = x[..., : self.rope_dim]
        pe, _ = self.rotary(positions, pe, pe)
        return torch.cat([pe, x[..., self.rope_dim :]], dim=-1)

    def compute_k(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Per-token index key ``(T, head_dim)`` — what the indexer cache stores.

        LayerNorm runs over the whole ``head_dim`` BEFORE rope; the key is
        cached roped+normed verbatim (nothing recomputed at read time).
        """
        k = self.k_norm(self.wk(hidden_states))  # (T, D)
        return self._rope_first_dims(k.unsqueeze(1), positions).squeeze(1)

    def compute_selection(
        self,
        q_c: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        k_history: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k prefix positions per query token, ``(T, topk)`` int32, -1 padded.

        Args:
            q_c: ``(T, q_lora_rank)`` post-``q_a_layernorm`` q latent — the
                SAME normalized bottleneck that feeds the main ``q_b_proj``,
                not the hidden states.
            hidden_states: ``(T, hidden)`` attention-block input (feeds
                ``weights_proj``; also what ``compute_k`` consumed).
            positions: ``(T,)`` absolute in-request positions; token t may
                attend to history rows 0..positions[t] INCLUDING itself.
            k_history: ``(S, head_dim)`` ``compute_k`` outputs where row s
                is the key at position s; must cover max(positions).
        Returns:
            Row t holds min(topk, positions[t]+1) selected positions
            (request-local), padded to ``topk`` with -1. Order within a row
            is NOT meaningful — at window <= topk the selected SET is the
            full prefix regardless of scores.
        """
        num_tokens = q_c.shape[0]
        num_keys = k_history.shape[0]
        if num_keys <= int(positions.max()):
            raise ValueError(
                f"k_history has {num_keys} rows but positions reach "
                f"{int(positions.max())}; the causal window includes self"
            )

        q = self.wq_b(q_c).view(num_tokens, self.n_heads, self.head_dim)
        q = self._rope_first_dims(q, positions)
        w = self.weights_proj(hidden_states) * self.weight_scale  # (T, H)

        # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]): per-head ReLU
        # BEFORE the weighted sum; the raw weights get no softmax/sigmoid.
        # [recollection] verified against the DeepGEMM fp8_mqa_logits
        # contract (the kernel body is not in the Phase C sources) — pending
        # an on-box op check before parity tests are frozen.
        dots = torch.einsum("thd,sd->ths", q, k_history).relu()
        scores = torch.einsum("th,ths->ts", w, dots)

        # Causal window INCLUDING self: candidates are positions 0..p_t.
        key_pos = torch.arange(num_keys, device=scores.device)
        scores = scores.masked_fill(
            key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))

        selection = torch.full(
            (num_tokens, self.topk), -1, dtype=torch.int32, device=scores.device)
        for t in range(num_tokens):
            n = min(self.topk, int(positions[t]) + 1)
            selection[t, :n] = scores[t].topk(n).indices.to(torch.int32)
        return selection
