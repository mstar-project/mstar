"""Fish S2-Pro Fast AR: the residual codebooks of one frame (``audio_decoder.*``).

After the Slow AR samples a frame's semantic code, this 4-layer decoder runs
along the codebook axis of that frame: position 0 is the Slow AR's final
hidden state, position ``i`` the embedding of codebook ``i - 1``'s code, and
the head at positions 1..9 picks codebooks 1..9 by argmax, as the S2-Pro
serving recipe does (see ``FishGenerationConfig``). Its KV cache is frame-local scratch, overwritten every
frame, so the whole loop is fixed-shape tensor code that a decode CUDA graph
captures as is (the Qwen3-TTS CodePredictor pattern).

The module also owns ``codebook_embeddings``, the 10 x 4,096-entry table the
Slow AR adds to the token embedding of every frame it reads back.

Module names follow the checkpoint (``attention.wqkv``, ``feed_forward.w1``
...), so the weights load without renaming. RoPE is interleaved over a
bfloat16 table, as in the reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.fish_s2_pro.components.dac import apply_rotary_emb, precompute_freqs_cis
from mstar.model.fish_s2_pro.config import FishFastARConfig


class FastARAttention(nn.Module):
    def __init__(self, cfg: FishFastARConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_local_heads = cfg.n_local_heads
        self.head_dim = cfg.head_dim
        self.wqkv = nn.Linear(
            cfg.dim,
            (cfg.n_head + 2 * cfg.n_local_heads) * cfg.head_dim,
            bias=cfg.attention_qkv_bias,
        )
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.dim, bias=cfg.attention_o_bias)
        if cfg.attention_qk_norm:
            self.q_norm = nn.RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = nn.RMSNorm(cfg.head_dim, cfg.norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: torch.Tensor,
        pos: int,
    ) -> torch.Tensor:
        """One position. ``x`` ``[B, 1, D]``; ``kv_cache`` ``[B, 2, P, H_kv, head_dim]``."""
        bsz = x.shape[0]
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz, 1, self.n_head, self.head_dim)
        k = k.view(bsz, 1, self.n_local_heads, self.head_dim)
        v = v.view(bsz, 1, self.n_local_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        kv_cache[:, 0, pos : pos + 1].copy_(k)
        kv_cache[:, 1, pos : pos + 1].copy_(v)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            kv_cache[:, 0, : pos + 1].transpose(1, 2),
            kv_cache[:, 1, : pos + 1].transpose(1, 2),
            enable_gqa=self.n_head != self.n_local_heads,
        )
        return self.wo(y.transpose(1, 2).reshape(bsz, 1, q_size))


class FastARFeedForward(nn.Module):
    def __init__(self, cfg: FishFastARConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FastARBlock(nn.Module):
    def __init__(self, cfg: FishFastARConfig):
        super().__init__()
        self.attention = FastARAttention(cfg)
        self.feed_forward = FastARFeedForward(cfg)
        self.attention_norm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = nn.RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, freqs_cis, kv_cache, pos):
        h = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache, pos)
        return h + self.feed_forward(self.ffn_norm(h))


class FishFastAR(nn.Module):
    """``audio_decoder.*`` of the S2-Pro checkpoint."""

    def __init__(self, cfg: FishFastARConfig):
        super().__init__()
        self.config = cfg
        self.project_in: nn.Module = nn.Linear(cfg.text_dim, cfg.dim) if cfg.text_dim != cfg.dim else nn.Identity()
        self.codebook_embeddings = nn.Embedding(cfg.vocab_size * cfg.num_codebooks, cfg.text_dim)
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(FastARBlock(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.num_codebooks, cfg.head_dim, cfg.rope_base),
            persistent=False,
        )
        self.register_buffer(
            "codebook_offsets",
            torch.arange(cfg.num_codebooks) * cfg.vocab_size,
            persistent=False,
        )

    def rematerialize_buffers(self) -> None:
        """Rebuild the derived buffers (after construction on ``meta``)."""
        cfg = self.config
        device = self.embeddings.weight.device
        self.freqs_cis = precompute_freqs_cis(cfg.num_codebooks, cfg.head_dim, cfg.rope_base).to(device)
        self.codebook_offsets = torch.arange(cfg.num_codebooks, device=device) * cfg.vocab_size

    def kv_cache_shape(self, batch_size: int) -> tuple[int, ...]:
        cfg = self.config
        return (cfg.n_layer, batch_size, 2, cfg.num_codebooks, cfg.n_local_heads, cfg.head_dim)

    def step(self, x: torch.Tensor, kv_cache: torch.Tensor, pos: int) -> torch.Tensor:
        """Run every layer at depth position ``pos``; returns the final-normed hidden state."""
        freqs_cis = self.freqs_cis[pos : pos + 1].unsqueeze(0)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, kv_cache[layer_idx], pos)
        return self.norm(x)

    def codebook_embedding_sum(self, codes: torch.Tensor) -> torch.Tensor:
        """``[N, 10]`` frame codes -> ``[N, text_dim]`` summed codebook embeddings."""
        return self.codebook_embeddings(codes + self.codebook_offsets).sum(dim=1)

    def forward(self, slow_hidden: torch.Tensor, semantic_codes: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` Slow AR hidden + ``[B]`` semantic codes -> ``[B, 10]`` frame codes."""
        bsz = slow_hidden.shape[0]
        codes = torch.empty(bsz, self.config.num_codebooks, dtype=torch.long, device=slow_hidden.device)
        codes[:, 0] = semantic_codes
        self.step(self.project_in(slow_hidden).unsqueeze(1), kv_cache, 0)
        x = self.embeddings(semantic_codes).unsqueeze(1)
        for pos in range(1, self.config.num_codebooks):
            logits = self.output(self.step(x, kv_cache, pos)).squeeze(1)
            code = logits.argmax(dim=-1)
            codes[:, pos] = code
            x = self.embeddings(code).unsqueeze(1)
        return codes
