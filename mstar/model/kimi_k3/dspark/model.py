"""The Kimi-K3-DSpark draft as an M* module (plan sections 8.1, 8.6).

Five dense layers of DeepSeek-style MLA with yarn rope (no output gate) and SiLU-gated MLPs over
the target's hidden size, fed by a projection of five concatenated target aux hidden states. A
step drafts ``k`` tokens for a row from its bonus token and ``k - 1`` mask tokens: the block
attends non-causally to itself and to the draft's own paged latent cache, which holds, per
position, the context KV computed from the target's aux states (never from the draft's own
tokens). The block's keys are not written anywhere: the paged part of the attention comes from
the ``DSPARK_ATTN`` resource over the stored context (``AttentionStep(context_only=True)``, with
its log-sum-exp), the block part is a dense softmax over the row's own keys, and the two merge.
The ``k`` logits go through the target's ``lm_head`` and a sequential Markov correction (rank
256 bias from the previously drafted token). Embedding and ``lm_head`` are the target's modules.

Tensor-parallel like the target: heads split for ``q_b_proj`` / ``kv_b_proj`` / ``o_proj``, the
MLP intermediate split, ``context_proj`` split on its outputs and all-gathered, the LoRA-A
projections, norms and the Markov head replicated. Parameter names follow the checkpoint (``ckpt/Kimi-K3-DSpark``), so the loader needs
only the gate/up fusion rule; ``embed_tokens`` and ``confidence_head`` are skipped.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from mstar.model.kimi_k3.components.common import KimiRMSNorm, ReplicatedLinear
from mstar.model.kimi_k3.dspark.config import DSparkConfig
from mstar.model.kimi_k3.dspark.rope import YarnRotary

DSPARK_KV = "dspark_kv"
DSPARK_ATTN = "dspark_attn"


class DSparkMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``; ``gate_proj`` / ``up_proj`` of the checkpoint fuse into ``gate_up_proj``."""

    def __init__(self, hidden_size: int, intermediate_size: int, comm_group: CommGroup):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            comm_group=comm_group, input_size=hidden_size, output_sizes=[intermediate_size, intermediate_size],
            bias=False, gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            comm_group=comm_group, input_size=intermediate_size, output_size=hidden_size, bias=False,
            input_is_parallel=True, reduce_results=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DSparkAttention(nn.Module):
    """Rope MLA of one draft layer: the context part through the paged resource, the block part dense."""

    def __init__(self, cfg: DSparkConfig, comm_group: CommGroup, rope: YarnRotary):
        super().__init__()
        tp = comm_group.world_size
        assert cfg.num_attention_heads % tp == 0
        self.cfg = cfg
        self.comm_group = comm_group
        self.rope = rope
        self.num_heads = cfg.num_attention_heads // tp
        self.scale = cfg.qk_head_dim ** -0.5 * rope.attn_scale_factor
        self.q_a_proj = ReplicatedLinear(cfg.hidden_size, cfg.q_lora_rank)
        self.q_a_layernorm = KimiRMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(comm_group, cfg.q_lora_rank, cfg.num_attention_heads * cfg.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = ReplicatedLinear(cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim)
        self.kv_a_layernorm = KimiRMSNorm(cfg.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, cfg.kv_lora_rank, cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False,
        )
        self.o_proj = RowParallelLinear(
            comm_group, cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size, bias=False, input_is_parallel=True,
            reduce_results=True,
        )
        self._absorbed: tuple[torch.Tensor, torch.Tensor] | None = None

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        self._absorbed = None
        return out

    def absorb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(W_UK [H_local, nope, latent], W_UV [H_local, latent, v])`` from ``kv_b_proj``."""
        if self._absorbed is None:
            c = self.cfg
            w = self.kv_b_proj.weight.view(self.num_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
            self._absorbed = (w[:, : c.qk_nope_head_dim].contiguous(), w[:, c.qk_nope_head_dim :].transpose(1, 2).contiguous())
        return self._absorbed

    def latent(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[T, kv_lora_rank + rope]``: the normalised compressed kv and the roped key rope part.
        For the target's aux-derived context states this is the cache entry; for the block, its own keys."""
        c, k_pe = self.kv_a_proj_with_mqa(x).split([self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1)
        return torch.cat([self.kv_a_layernorm(c), self.rope.apply(k_pe, positions)], dim=-1)

    def query(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Absorbed query ``[T, H_local, latent]`` and its roped rope part ``[T, H_local, rope]``."""
        t = x.shape[0]
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(t, self.num_heads, self.cfg.qk_head_dim)
        q_nope, q_pe = q.split([self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim], dim=-1)
        w_uk, _ = self.absorb()
        return torch.einsum("thn,hnl->thl", q_nope, w_uk.to(q_nope.dtype)), self.rope.apply(q_pe, positions)

    def block_scores(self, q_lat: torch.Tensor, q_pe: torch.Tensor, lat: torch.Tensor, rows: int) -> torch.Tensor:
        """Dense scores of a row's ``k`` queries against its own ``k`` keys: ``[rows, H, k, k]`` fp32."""
        c, k_pe = lat.split([self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1)
        ql = q_lat.view(rows, -1, self.num_heads, self.cfg.kv_lora_rank).float()
        qp = q_pe.view(rows, -1, self.num_heads, self.cfg.qk_rope_head_dim).float()
        c = c.view(rows, -1, self.cfg.kv_lora_rank).float()
        k_pe = k_pe.view(rows, -1, self.cfg.qk_rope_head_dim).float()
        return (torch.einsum("rqhl,rkl->rhqk", ql, c) + torch.einsum("rqhe,rke->rhqk", qp, k_pe)) * self.scale

    def finish(self, o_lat: torch.Tensor) -> torch.Tensor:
        t = o_lat.shape[0]
        _, w_uv = self.absorb()
        attn = torch.einsum("thl,hlv->thv", o_lat, w_uv.to(o_lat.dtype)).reshape(t, self.num_heads * self.cfg.v_head_dim)
        return self.o_proj(attn)

    def forward_block(self, x: torch.Tensor, positions: torch.Tensor, rows: int, attn, kv_layer: torch.Tensor,
                      label: str | None = None) -> torch.Tensor:
        """The block's attention: the paged context part (``attn`` planned ``context_only``) merged
        with the dense part over the row's own keys. ``x [rows * k, hidden]``, ``positions [rows * k]``."""
        q_lat, q_pe = self.query(x, positions)
        lat = self.latent(x, positions)
        o_ctx, lse_ctx = attn.run(q_lat, label=label, kv_cache_layer=kv_layer, q_pe=q_pe, return_lse=True)
        if x.is_cuda:
            from mstar.model.kimi_k3.dspark.block_attn_kernel import dspark_block_attention

            # one launch: the block's scores, softmax and output and the merge with the context part
            return self.finish(dspark_block_attention(q_lat, q_pe, lat, o_ctx, lse_ctx, rows, self.cfg.kv_lora_rank, self.scale))
        scores = self.block_scores(q_lat, q_pe, lat, rows)  # [rows, H, k, k]
        lse_blk = torch.logsumexp(scores, dim=-1)  # [rows, H, k]
        c = lat[..., : self.cfg.kv_lora_rank].view(rows, -1, self.cfg.kv_lora_rank).float()
        o_blk = torch.einsum("rhqk,rkl->rqhl", torch.softmax(scores, dim=-1), c)  # [rows, k, H, latent]
        o_blk = o_blk.reshape(-1, self.num_heads, self.cfg.kv_lora_rank)
        lse_blk = lse_blk.transpose(1, 2).reshape(-1, self.num_heads)  # [rows * k, H]
        m = torch.maximum(lse_ctx, lse_blk)
        w_ctx, w_blk = torch.exp(lse_ctx - m), torch.exp(lse_blk - m)
        o = (o_ctx.float() * w_ctx[..., None] + o_blk * w_blk[..., None]) / (w_ctx + w_blk)[..., None]
        return self.finish(o.to(x.dtype))


class DSparkLayer(nn.Module):
    def __init__(self, cfg: DSparkConfig, comm_group: CommGroup, rope: YarnRotary):
        super().__init__()
        self.input_layernorm = KimiRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DSparkAttention(cfg, comm_group, rope)
        self.post_attention_layernorm = KimiRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = DSparkMLP(cfg.hidden_size, cfg.intermediate_size, comm_group)

    def forward_block(self, x, positions, rows, attn, kv_layer, label=None):
        x = x + self.self_attn.forward_block(self.input_layernorm(x), positions, rows, attn, kv_layer, label)
        return x + self.mlp(self.post_attention_layernorm(x))


class MarkovHead(nn.Module):
    """Sequential intra-block dependency: a rank-``r`` bias over the vocabulary from the previous token."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = ReplicatedLinear(rank, vocab_size)

    def bias(self, prev: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev))


class ContextAccumulator:
    """The context projection summed one aux layer at a time. ``sink(j, state)`` adds the ``j``-th
    aux layer's share of ``context_proj`` (its slice of the input columns) in fp32, ``finish()``
    casts, all-gathers the output slices and applies ``context_norm``: the same as
    ``combine(torch.cat(aux, -1))`` up to the fp32 summation order, without the prefill ever
    holding the five aux states (5 x [T, 7168] bf16 is 587 MB at 8k tokens) or their concatenation.
    """

    def __init__(self, draft: "DSparkDraft"):
        self._draft = draft
        self._acc: torch.Tensor | None = None

    def __call__(self, j: int, state: torch.Tensor) -> None:
        weight = self._draft.context_proj.weight
        h = state.shape[-1]
        w_j = weight[:, j * h:(j + 1) * h]
        if state.is_cuda:
            part = torch.mm(state, w_j.t(), out_dtype=torch.float32)
        else:
            part = torch.mm(state.float(), w_j.float().t())
        self._acc = part if self._acc is None else self._acc.add_(part)

    def finish(self) -> torch.Tensor:
        assert self._acc is not None, "no aux state was added"
        proj = self._draft.context_proj
        out = self._acc.to(proj.weight.dtype)
        self._acc = None
        if proj.gather_output and proj.tp_size > 1:
            out = proj.comm_group.all_gather(out, dim=-1)
        return self._draft.context_norm(out)


class DSparkDraft(nn.Module):
    def __init__(self, cfg: DSparkConfig, embed_tokens: nn.Module, lm_head: nn.Module,
                 comm_group: CommGroup | None = None, max_positions: int = 65536,
                 kv_key: str = DSPARK_KV, attn_key: str = DSPARK_ATTN):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.cfg = cfg
        self._embed_tokens, self._lm_head = (embed_tokens,), (lm_head,)  # the target's, not owned
        self.rope = YarnRotary(cfg.qk_rope_head_dim, cfg.rope, max_positions)
        # [hidden, 5 * target_hidden]: 514 MB in bf16 for K3, so each rank holds a slice of the
        # outputs and the slices are all-gathered (the aux states, its input, are replicated)
        self.context_proj = ColumnParallelLinear(comm_group, cfg.context_width, cfg.hidden_size, bias=False,
                                                 gather_output=True)
        self.context_norm = KimiRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList(DSparkLayer(cfg, comm_group, self.rope) for _ in range(cfg.num_hidden_layers))
        self.final_norm = KimiRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.markov_head = MarkovHead(cfg.vocab_size, cfg.markov_rank)
        self._kv_key, self._attn_key = kv_key, attn_key
        self.kv = None
        self.attn = None

    @property
    def embed_tokens(self) -> nn.Module:
        return self._embed_tokens[0]

    @property
    def lm_head(self) -> nn.Module:
        return self._lm_head[0]

    def bind_resources(self, resources: dict) -> None:
        self.kv = resources.get(self._kv_key)
        self.attn = resources.get(self._attn_key)

    # ------------------------------------------------------------- pieces
    def combine(self, aux: torch.Tensor) -> torch.Tensor:
        """``aux [T, 5 * target_hidden]`` (the target's aux states, layer order) -> context states ``[T, hidden]``."""
        return self.context_norm(self.context_proj(aux))

    def context_accumulator(self) -> "ContextAccumulator":
        """``combine`` spread over the target's forward: an ``aux_sink`` for the language model."""
        return ContextAccumulator(self)

    def context_latents(self, states: torch.Tensor, positions: torch.Tensor) -> list[torch.Tensor]:
        """Per layer, the cache entries of context states at ``positions``: ``[T, latent + rope]`` each."""
        return [layer.self_attn.latent(states, positions) for layer in self.layers]

    def write_context(self, states: torch.Tensor, positions: torch.Tensor, label: str | None = None) -> None:
        """Write the context KV of ``states`` into the draft cache at this step's planned slots."""
        assert self.kv is not None, "bind_resources first"
        for i, lat in enumerate(self.context_latents(states, positions)):
            self.kv.write_kv(lat, layer_idx=i, label=label)

    def block_hidden(self, ids: torch.Tensor, positions: torch.Tensor, rows: int, label: str | None = None) -> torch.Tensor:
        """The block's final hidden states ``[rows * k, hidden]`` (before the head)."""
        assert self.kv is not None and self.attn is not None, "bind_resources first"
        x = self.embed_tokens(ids)
        for i, layer in enumerate(self.layers):
            x = layer.forward_block(x, positions, rows, self.attn, self.kv.layer_view(i), label)
        return self.final_norm(x)

    def block_ids(self, bonus: torch.Tensor, k: int) -> torch.Tensor:
        """``[rows, k]``: the bonus token then ``k - 1`` mask tokens."""
        masks = torch.full((bonus.shape[0], k - 1), self.cfg.mask_token_id, dtype=bonus.dtype, device=bonus.device)
        return torch.cat([bonus.view(-1, 1), masks], dim=1)

    def markov_sample(self, logits: torch.Tensor, bonus: torch.Tensor) -> torch.Tensor:
        """Greedy left-to-right drafting: ``logits [rows, k, V]`` (position ``i`` predicts the token
        after query ``i``), each corrected by the Markov bias of the previously drafted token
        (the bonus first). Returns ``drafts [rows, k]``."""
        rows, k, _ = logits.shape
        drafts = []
        prev = bonus.view(rows)
        for i in range(k):
            step = logits[:, i].float() + self.markov_head.bias(prev).float()
            prev = step.argmax(dim=-1)
            drafts.append(prev)
        return torch.stack(drafts, dim=1)

    def draft(self, bonus: torch.Tensor, positions: torch.Tensor, k: int, label: str | None = None) -> torch.Tensor:
        """``bonus [rows]`` and the block positions ``[rows * k]`` -> ``drafts [rows, k]``."""
        rows = bonus.shape[0]
        ids = self.block_ids(bonus, k).reshape(-1)
        hidden = self.block_hidden(ids, positions, rows, label)
        logits = self.lm_head(hidden).view(rows, k, -1)
        return self.markov_sample(logits, bonus)

    # ------------------------------------------------------------- weights
    def load_weights(self, source: str | Path) -> set[str]:
        from mstar.model.loader.base import StackedParamRule, load_weights_into
        from mstar.model.loader.iterators import iter_safetensors_shards

        rules = [StackedParamRule(".gate_up_proj", ".gate_proj", 0), StackedParamRule(".gate_up_proj", ".up_proj", 1)]
        skip = lambda name: name.startswith(("embed_tokens.", "confidence_head."))  # noqa: E731
        return load_weights_into(self, iter_safetensors_shards(str(source)), stacked_params=rules, skip_predicate=skip)

    # ------------------------------------------------------------- reference
    @torch.no_grad()
    def draft_dense(self, context_states: torch.Tensor, bonus: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One request, no resources: the block's ``k`` tokens over ``context_states [Tc, hidden]``
        (already combined) at positions ``Tc .. Tc + k - 1`` with a dense non-causal attention over
        context and block. Returns ``(drafts [k], hidden [k, hidden])``: the oracle for the paged path."""
        tc = context_states.shape[0]
        dev = context_states.device
        ctx_pos = torch.arange(tc, device=dev)
        blk_pos = torch.arange(tc, tc + k, device=dev)
        ids = self.block_ids(torch.tensor([bonus], device=dev), k).reshape(-1)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q_lat, q_pe = a.query(h, blk_pos)
            lat = torch.cat([a.latent(context_states, ctx_pos), a.latent(h, blk_pos)], dim=0)
            c, k_pe = lat.split([self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1)
            scores = (torch.einsum("qhl,kl->hqk", q_lat.float(), c.float())
                      + torch.einsum("qhr,kr->hqk", q_pe.float(), k_pe.float())) * a.scale
            o = torch.einsum("hqk,kl->qhl", torch.softmax(scores, dim=-1), c.float()).to(x.dtype)
            x = x + a.finish(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden).view(1, k, -1)
        return self.markov_sample(logits, torch.tensor([bonus], device=dev))[0], hidden
