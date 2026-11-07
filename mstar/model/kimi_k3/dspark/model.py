"""Kimi K3.1's DSpark drafter as an M* module, ported from PR #257's K3 drafter
(``mstar/model/kimi_k3/dspark/model.py`` there) for ``Inferact/Kimi-K3.1-Pro-DSpark-preview``.

Five dense layers of DeepSeek-style MLA with yarn rope and SiLU MLPs, fed by ``context_proj`` of
five concatenated target hidden states (the stream leaving target layers 75, 79, 83, 87 and 91;
``KimiK3LanguageModel.forward(aux_layers=...)``). Each position the target has run gets one
context cache row per drafter layer, computed once from those states (``context_latents``); a
draft step runs a block of ``block_size`` rows (the anchor, then mask tokens) at the next
positions, attending to the row's context rows and, causally for this release, to the block's
own; ``lm_head`` (the target's) and a left-to-right Markov correction give the drafts.

Differences from #257's K3 drafter, from the release's config.json: 96 heads, a rank-512 Markov
head, rope theta 1e6 with the tables unscaled (``attn_factor``), and a causal block (#257's
attended both ways). The block runs dense here, its context rows passed per request: the oracle
and the eager path. Serving it needs a draft/verify loop, which main lacks (see the port's draft);
because the block is causal, its rows can go to a paged MLA cache past the context and attend
through main's causal MLA path, with no context-only pass or LSE merge (#257's needed both).

Tensor-parallel like the target: heads split for ``q_b_proj`` / ``kv_b_proj`` / ``o_proj``, the
MLP intermediate split, ``context_proj`` split on its outputs and all-gathered; the LoRA-A
projections, norms and the Markov head replicated. Parameter names follow the checkpoint, so the
loader needs only the gate/up fusion; ``confidence_head`` (adaptive verification, unused by a
greedy verify) is skipped, and the checkpoint ships no embedding.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.dspark.config import DSparkConfig
from mstar.model.kimi_k3.dspark.rope import YarnRotary


class DSparkRMSNorm(RMSNorm):
    """``RMSNorm`` in its input's precision. On CUDA ``mstar::rms_norm`` runs an fp32 input in bf16
    outside autocast (FlashInfer takes half precision only), which made the fp32 drafter, the oracle
    of the GPU check, a bf16 one (job 1465: 1.07 off the float64 reference at logit std 3.2). Half
    precision and autocast keep the kernel."""

    def forward(self, hidden_states: torch.Tensor, comm_group=None) -> torch.Tensor:
        if hidden_states.dtype == torch.float32 and not torch.is_autocast_enabled(hidden_states.device.type):
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
            return hidden_states * torch.rsqrt(variance + self.variance_epsilon) * self.weight.float()
        return super().forward(hidden_states, comm_group)


class DSparkMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``; the checkpoint's ``gate_proj`` / ``up_proj`` fuse into ``gate_up_proj``."""

    def __init__(self, cfg: DSparkConfig, comm_group: CommGroup):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            comm_group=comm_group, input_size=cfg.hidden_size,
            output_sizes=[cfg.intermediate_size, cfg.intermediate_size], bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(
            comm_group=comm_group, input_size=cfg.intermediate_size, output_size=cfg.hidden_size, bias=False,
            input_is_parallel=True, reduce_results=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DSparkAttention(nn.Module):
    """Rope MLA of one drafter layer, absorbed: queries score the cached latents directly."""

    def __init__(self, cfg: DSparkConfig, comm_group: CommGroup, rope: YarnRotary):
        super().__init__()
        tp = comm_group.world_size
        if cfg.num_attention_heads % tp:
            raise ValueError(f"{cfg.num_attention_heads} drafter heads over {tp} ranks")
        self.cfg = cfg
        self.rope = rope
        self.num_heads = cfg.num_attention_heads // tp
        self.scale = cfg.qk_head_dim ** -0.5 * rope.attn_scale_factor
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = DSparkRMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(comm_group, cfg.q_lora_rank, cfg.num_attention_heads * cfg.qk_head_dim,
                                             bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(cfg.hidden_size, cfg.latent_dim, bias=False)
        self.kv_a_layernorm = DSparkRMSNorm(cfg.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, cfg.kv_lora_rank, cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False)
        self.o_proj = RowParallelLinear(comm_group, cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size,
                                        bias=False, input_is_parallel=True, reduce_results=True)

    def absorb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(W_UK [H_local, nope, latent], W_UV [H_local, latent, v])``, views of ``kv_b_proj``."""
        c = self.cfg
        w = self.kv_b_proj.weight.view(self.num_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        return w[:, :c.qk_nope_head_dim], w[:, c.qk_nope_head_dim:].transpose(1, 2)

    def latent(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[T, kv_lora_rank + rope]``: the normed compressed kv and the roped key rope part, a
        context cache row for the target's states, the block's own keys for its rows."""
        c, k_pe = self.kv_a_proj_with_mqa(x).split([self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1)
        return torch.cat([self.kv_a_layernorm(c), self.rope.apply(k_pe, positions)], dim=-1)

    def query(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Absorbed query ``[T, H_local, latent]`` and its roped rope part ``[T, H_local, rope]``."""
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(x.shape[0], self.num_heads, self.cfg.qk_head_dim)
        q_nope, q_pe = q.split([self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim], dim=-1)
        w_uk, _ = self.absorb()
        return torch.einsum("thn,hnl->thl", q_nope, w_uk.to(q_nope.dtype)), self.rope.apply(q_pe, positions)

    def attend(self, q_lat: torch.Tensor, q_pe: torch.Tensor, keys: torch.Tensor, num_context: int) -> torch.Tensor:
        """One row's block: ``k`` queries over ``keys [num_context + k, latent + rope]`` (its context
        rows, then the block's), fp32. Causal: query ``i`` sees the context and block rows ``<= i``."""
        rank = self.cfg.kv_lora_rank
        c, k_pe = keys.float().split([rank, self.cfg.qk_rope_head_dim], dim=-1)
        with torch.autocast(keys.device.type, enabled=False):
            scores = (torch.einsum("qhl,kl->hqk", q_lat.float(), c)
                      + torch.einsum("qhr,kr->hqk", q_pe.float(), k_pe)) * self.scale
            if self.cfg.causal:
                k = q_lat.shape[0]
                visible = torch.arange(keys.shape[0], device=keys.device)[None, :] <= (
                    num_context + torch.arange(k, device=keys.device)[:, None])
                scores = scores.masked_fill(~visible, float("-inf"))
            return torch.einsum("hqk,kl->qhl", scores.softmax(dim=-1), c)

    def finish(self, o_lat: torch.Tensor) -> torch.Tensor:
        _, w_uv = self.absorb()
        out = torch.einsum("thl,hlv->thv", o_lat.float(), w_uv.float()).to(self.o_proj.weight.dtype)
        return self.o_proj(out.reshape(o_lat.shape[0], self.num_heads * self.cfg.v_head_dim))

    def forward_block(self, x: torch.Tensor, positions: torch.Tensor, contexts: Sequence[torch.Tensor]) -> torch.Tensor:
        """``x [rows * k, hidden]`` at ``positions``; ``contexts[r]`` row ``r``'s context rows of
        this layer ``[Tc_r, latent + rope]``."""
        rows = len(contexts)
        k = x.shape[0] // rows
        q_lat, q_pe = self.query(x, positions)
        own = self.latent(x, positions)
        out = torch.empty(x.shape[0], self.num_heads, self.cfg.kv_lora_rank, device=x.device)
        for r, ctx in enumerate(contexts):
            block = slice(r * k, (r + 1) * k)
            keys = torch.cat([ctx.to(own.dtype), own[block]], dim=0)
            out[block] = self.attend(q_lat[block], q_pe[block], keys, ctx.shape[0])
        return self.finish(out)


class DSparkLayer(nn.Module):
    def __init__(self, cfg: DSparkConfig, comm_group: CommGroup, rope: YarnRotary):
        super().__init__()
        self.input_layernorm = DSparkRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DSparkAttention(cfg, comm_group, rope)
        self.post_attention_layernorm = DSparkRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = DSparkMLP(cfg, comm_group)

    def forward_block(self, x, positions, contexts):
        x = x + self.self_attn.forward_block(self.input_layernorm(x), positions, contexts)
        return x + self.mlp(self.post_attention_layernorm(x))


class MarkovHead(nn.Module):
    """A rank-``r`` bias over the vocabulary from the previously drafted token."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def bias(self, prev: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev))


class ContextAccumulator:
    """``combine`` spread over the target's forward, one tapped layer at a time (PR #257's): the
    ``aux_sink`` of ``KimiK3LanguageModel.forward``. ``sink(j, state)`` adds the ``j``-th tapped
    layer's share of ``context_proj`` in fp32, ``finish()`` casts, all-gathers the output slices and
    applies ``context_norm``, so a long prefill never holds the five states (5 x [T, 7168]) at once."""

    def __init__(self, draft: DSparkDraft):
        self._draft = draft
        self._acc: torch.Tensor | None = None

    def __call__(self, j: int, state: torch.Tensor) -> None:
        weight = self._draft.context_proj.weight
        h = state.shape[-1]
        part = state.float() @ weight[:, j * h:(j + 1) * h].float().t()
        self._acc = part if self._acc is None else self._acc.add_(part)

    def finish(self) -> torch.Tensor:
        if self._acc is None:
            raise RuntimeError("no tapped state was added")
        proj = self._draft.context_proj
        out, self._acc = self._acc.to(proj.weight.dtype), None
        if proj.gather_output and proj.tp_size > 1:
            out = proj.comm_group.all_gather(out, dim=-1)
        return self._draft.context_norm(out)


class DSparkDraft(nn.Module):
    def __init__(self, cfg: DSparkConfig, embed_tokens: nn.Module, lm_head: nn.Module,
                 comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.cfg = cfg
        self._embed_tokens, self._lm_head = (embed_tokens,), (lm_head,)  # the target's, not owned
        self.rope = YarnRotary(cfg.qk_rope_head_dim, cfg.rope)
        # [hidden, 5 * target_hidden]: each rank holds a slice of the outputs, all-gathered
        self.context_proj = ColumnParallelLinear(comm_group, cfg.context_width, cfg.hidden_size, bias=False,
                                                 gather_output=True)
        self.context_norm = DSparkRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList(DSparkLayer(cfg, comm_group, self.rope) for _ in range(cfg.num_hidden_layers))
        self.final_norm = DSparkRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.markov_head = MarkovHead(cfg.vocab_size, cfg.markov_rank)

    @property
    def embed_tokens(self) -> nn.Module:
        return self._embed_tokens[0]

    @property
    def lm_head(self) -> nn.Module:
        return self._lm_head[0]

    # ------------------------------------------------------------- context
    def combine(self, aux: Sequence[torch.Tensor]) -> torch.Tensor:
        """The tapped target states (layer order, ``[T, target_hidden]`` each) -> context states ``[T, hidden]``."""
        return self.context_norm(self.context_proj(torch.cat(list(aux), dim=-1)))

    def context_accumulator(self) -> ContextAccumulator:
        return ContextAccumulator(self)

    def context_latents(self, states: torch.Tensor, positions: torch.Tensor) -> list[torch.Tensor]:
        """Per layer, the context cache rows of ``states`` at ``positions``: ``[T, latent + rope]``."""
        return [layer.self_attn.latent(states, positions) for layer in self.layers]

    # ------------------------------------------------------------- drafting
    def block_ids(self, anchors: torch.Tensor, k: int) -> torch.Tensor:
        """``[rows, k]``: each row's anchor, then ``k - 1`` mask tokens."""
        masks = torch.full((anchors.shape[0], k - 1), self.cfg.mask_token_id, dtype=anchors.dtype,
                           device=anchors.device)
        return torch.cat([anchors.view(-1, 1), masks], dim=1)

    def markov_sample(self, logits: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Greedy left to right: ``logits [rows, k, V]`` (row ``i`` predicts the token after block
        position ``i``), each plus the Markov bias of the token drafted before it (the anchor
        first). ``drafts [rows, k]``."""
        drafts = []
        prev = anchors.view(-1)
        for i in range(logits.shape[1]):
            prev = (logits[:, i].float() + self.markov_head.bias(prev).float()).argmax(dim=-1)
            drafts.append(prev)
        return torch.stack(drafts, dim=1)

    def block_hidden(self, anchors: torch.Tensor, contexts: Sequence[Sequence[torch.Tensor]], k: int) -> torch.Tensor:
        """The block's final-normed hidden ``[rows * k, hidden]``; ``contexts[r][layer]`` row ``r``'s
        context rows ``[Tc_r, latent + rope]``, the block at positions ``Tc_r .. Tc_r + k - 1``."""
        rows = anchors.shape[0]
        dev = anchors.device
        start = torch.tensor([ctx[0].shape[0] for ctx in contexts], device=dev)
        positions = (start[:, None] + torch.arange(k, device=dev)[None, :]).reshape(-1)
        x = self.embed_tokens(self.block_ids(anchors, k).reshape(-1))
        for i, layer in enumerate(self.layers):
            x = layer.forward_block(x, positions, [contexts[r][i] for r in range(rows)])
        return self.final_norm(x)

    @torch.no_grad()
    def draft(self, anchors: torch.Tensor, contexts: Sequence[Sequence[torch.Tensor]],
              k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``anchors [rows]`` -> ``(drafts [rows, k], logits [rows, k, V])`` (logits before the Markov
        bias); ``k`` defaults to the release's block."""
        k = k or self.cfg.block_size
        hidden = self.block_hidden(anchors, contexts, k)
        logits = self.lm_head(hidden).view(anchors.shape[0], k, -1)
        return self.markov_sample(logits, anchors), logits

    @torch.no_grad()
    def draft_dense(self, states: torch.Tensor, anchor: int, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One request from its context states ``[Tc, hidden]`` (already combined), every context row
        recomputed: ``(drafts [k], logits [k, V])``, the oracle for a cache of context rows."""
        k = k or self.cfg.block_size
        positions = torch.arange(states.shape[0], device=states.device)
        contexts = [self.context_latents(states, positions)]
        drafts, logits = self.draft(torch.tensor([anchor], device=states.device), contexts, k)
        return drafts[0], logits[0]

    # ------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], strict: bool = True) -> set[str]:
        """The checkpoint's tensors (``model.safetensors``'s names): every parameter gets one."""
        from mstar.model.loader.base import StackedParamRule, load_weights_into

        rules = [StackedParamRule(".gate_up_proj", ".gate_proj", 0), StackedParamRule(".gate_up_proj", ".up_proj", 1)]
        return load_weights_into(self, weights, stacked_params=rules, strict=strict,
                                 skip_predicate=lambda name: name.startswith(("confidence_head.", "embed_tokens.")))


class DSparkContextStore:
    """Each request's context cache rows per drafter layer, grown as the target runs (the eager
    stand-in for a paged cache): ``append`` the context states of positions ``start ..``."""

    def __init__(self, draft: DSparkDraft):
        self.draft = draft
        self._rows: dict[str, list[torch.Tensor]] = {}

    def append(self, rid: str, states: torch.Tensor, start: int) -> None:
        have = self._rows.get(rid)
        length = 0 if have is None else have[0].shape[0]
        if start > length:
            raise ValueError(f"{rid}: context rows from {start}, but only {length} are cached")
        positions = torch.arange(start, start + states.shape[0], device=states.device)
        new = self.draft.context_latents(states, positions)
        # rows past ``start`` (a rejected tail) are dropped: the target's states replace them
        self._rows[rid] = new if have is None else [torch.cat([h[:start], n]) for h, n in zip(have, new, strict=True)]

    def length(self, rid: str) -> int:
        return self._rows[rid][0].shape[0]

    def contexts(self, rids: Sequence[str]) -> list[list[torch.Tensor]]:
        return [self._rows[rid] for rid in rids]

    def remove(self, rid: str) -> None:
        self._rows.pop(rid, None)
