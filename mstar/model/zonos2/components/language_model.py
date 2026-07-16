"""Zonos2 language model: multi-codebook TTS transformer.

Data flow (per the reference)::

    input_ids (tokens, n_codebooks[+text])
      -> MultiEmbedding            # sum of per-column VocabParallelEmbedding
      -> emb_norm                  # parameter-free RMSNorm
      -> N x Zonos2DecoderLayer    # pre-norm; attn (temp + gating) + FFN/MoE
      -> out_norm                  # RMSNorm
      -> MultiOutputHead           # linear -> (*, n_codebooks, audio_vocab)
      -> softcap(logits, 15.0)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.distributed.utils import divide
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ParallelGatedMLP,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.components.moe import dispatch_experts
from mstar.model.zonos2.config import Zonos2Config

# QK-norm epsilon is hardcoded in the reference attention (F.rms_norm(..., eps=1e-6)).
_QK_NORM_EPS = 1e-6


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """tanh-based logit soft-capping: ``cap * tanh(x / cap)``."""
    return cap * torch.tanh(x / cap)


class MultiEmbedding(nn.Module):
    """Sum of per-column token embeddings (9 audio codebooks + text).

    Maintains one :class:`VocabParallelEmbedding` per column and sums their
    lookups element-wise into a single hidden state. Checkpoint layout:
    ``multi_embedder.embedders.{i}.weight`` (audio columns first, text last).
    """

    def __init__(self, config: Zonos2Config, comm_group: TPCommGroup):
        super().__init__()
        self.n_codebooks = config.n_codebooks

        embedders: list[nn.Module] = []
        # Audio codebook tables (padding_idx = audio_pad_id).
        for _ in range(config.n_codebooks):
            embedders.append(
                VocabParallelEmbedding(
                    num_embeddings=config.codebook_size + 2,
                    embedding_dim=config.hidden_size,
                    comm_group=comm_group,
                    padding_idx=config.audio_pad_id,
                )
            )
        # Optional text table (padding_idx = text_vocab), appended last.
        if config.text_vocab is not None:
            embedders.append(
                VocabParallelEmbedding(
                    num_embeddings=config.text_vocab + 1,
                    embedding_dim=config.hidden_size,
                    comm_group=comm_group,
                    padding_idx=config.text_vocab,
                )
            )
        self.embedders = nn.ModuleList(embedders)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (tokens, num_columns). Column i is looked up in embedder i.
        # ``.contiguous()`` because codes[..., i] is a strided view.
        out = self.embedders[0](codes[..., 0].contiguous())
        for i in range(1, codes.shape[-1]):
            out = out + self.embedders[i](codes[..., i].contiguous())
        return out


class Zonos2Attention(nn.Module):
    """Self-attention with parameter-free QK-norm, a learnable per-head
    temperature, interleaved RoPE, and headwise sigmoid gating.

    Differs from the shared :class:`Attention`, so it is written out 
    here rather than subclassed:

    - QK-norm is *parameter-free* (``F.rms_norm`` with no weight), and the
      query is then scaled by a learnable per-head ``|temp|``.
    - RoPE uses the interleaved (``is_neox=False``) layout.
    - The attention output is gated headwise by ``sigmoid(gater(x))``.

    Projections reuse the TP-aware parallel linears:
    ``wq``/``gater`` (column), ``wkv`` (merged K||V column), ``wo`` (row).
    """

    def __init__(self, config: Zonos2Config, comm_group: TPCommGroup):
        super().__init__()
        self.comm_group = comm_group
        tp_size = comm_group.world_size

        self.head_dim = config.head_dim
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.local_num_heads = divide(self.num_heads, tp_size)
        self.local_num_kv_heads = divide(self.num_kv_heads, tp_size)
        self.rope_theta = config.rope_theta

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.wq = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=q_dim,
            bias=False,
        )
        # Fused K||V; MergedColumnParallelLinear shards K and V heads
        # independently (shard 0 = K, shard 1 = V).
        self.wkv = MergedColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_sizes=[kv_dim, kv_dim],
            bias=False,
        )
        self.wo = RowParallelLinear(
            comm_group=comm_group,
            input_size=q_dim,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
        )
        # Headwise gate: hidden -> num_heads, sharded over heads like wq.
        self.gater = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=self.num_heads,
            bias=False,
        )

        # Learnable per-head temperature, shape (1, local_num_heads, 1),
        # broadcast over tokens and head_dim. Sharded across TP ranks.
        self.temp = nn.Parameter(torch.ones(1, self.local_num_heads, 1))
        self._attach_temp_loader()

    def _attach_temp_loader(self) -> None:
        self.temp.weight_loader = self._temp_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_temp_loader()
        return result

    def _temp_loader(self, param, loaded_weight, loaded_shard_id=None):
        # Checkpoint temp is (1, num_heads, 1)
        start = self.comm_group.rank * self.local_num_heads
        shard = loaded_weight.narrow(1, start, self.local_num_heads)
        assert param.data.shape == shard.shape, (
            f"temp shape mismatch: {tuple(param.data.shape)} vs {tuple(shard.shape)}"
        )
        param.data.copy_(shard)

    def forward(
        self,
        x: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]

        # Headwise gate from the (normed) input; applied after attention.
        gate = torch.sigmoid(self.gater(x))  # (tokens, local_num_heads)

        q = self.wq(x).view(num_tokens, self.local_num_heads, self.head_dim)
        kv = self.wkv(x)
        kv_dim = self.local_num_kv_heads * self.head_dim
        k, v = kv.split([kv_dim, kv_dim], dim=-1)
        k = k.view(num_tokens, self.local_num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.local_num_kv_heads, self.head_dim).contiguous()

        # Parameter-free QK-norm; query additionally scaled by |temp| per head.
        q = F.rms_norm(q, (self.head_dim,), eps=_QK_NORM_EPS) * self.temp.abs().to(q.dtype)
        k = F.rms_norm(k, (self.head_dim,), eps=_QK_NORM_EPS)

        # Interleaved RoPE (is_neox=False). No llama3 scaling kwargs so the
        # cache handle stays on the plain rope path.
        q, k = cache_handle.apply_rope(
            q, k, rope_theta=self.rope_theta, interleave=True,
        )

        # Standard scaled-dot-product attention (softmax scale = 1/sqrt(dim);
        # the temperature above is an additional learned multiplier on q).
        o = cache_handle.run_attention(q=q, k=k, v=v)  # (tokens, heads, dim)
        o = o * gate.unsqueeze(-1)
        o = o.reshape(num_tokens, self.local_num_heads * self.head_dim)
        return self.wo(o)


class Zonos2Router(nn.Module):
    """MoE router with Expert-Dropout-Augmentation (EDA) state threading.

    Down-projects the hidden state to ``router_dim``, optionally blends in
    the previous MoE layer's router state (EDA), RMS-normalizes, runs a
    3-layer GELU MLP to per-expert logits, softmaxes, and selects a
    bias-aware top-k. Returns the routing weights, expert indices, and the
    *pre-norm* router state for the next MoE layer's EDA.

    Checkpoint layout:
        router.down_proj.{weight,bias}
        router.router_mlp.{0,2,4}.{weight,bias}   # GELU at indices 1, 3
        router.rmsnorm_eda.weight
        router.router_states_scale               # EDA layers only
        router.balancing_biases
    """

    def __init__(self, config: Zonos2Config, layer_id: int):
        super().__init__()
        self.num_experts = config.moe_n_experts
        # Per-layer top-k (``special_topk_layers`` overrides the global default,
        # e.g. layer 26 -> top-2 in the reference checkpoint).
        self.top_k = config.get_num_experts_per_tok(layer_id)

        self.use_eda = layer_id != config.moe_start_from_layer
        self.subtract_bias = config.moe_balancing_strategy != "legacy"

        router_dim = config.moe_router_dim
        self.down_proj = nn.Linear(config.hidden_size, router_dim, bias=True)

        self.router_mlp = nn.Sequential(
            nn.Linear(router_dim, router_dim, bias=True),
            nn.GELU(),
            nn.Linear(router_dim, router_dim, bias=True),
            nn.GELU(),
            nn.Linear(router_dim, self.num_experts, bias=False),
        )

        self.rmsnorm_eda = RMSNorm(router_dim, eps=config.rms_norm_eps)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(router_dim))
        self.register_buffer(
            "balancing_biases",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )

    def forward(
        self,
        x: torch.Tensor,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.down_proj(x)
        if self.use_eda and router_states is not None:
            hidden = hidden + router_states * self.router_states_scale
        # Pre-norm state threaded to the next MoE layer's EDA.
        router_states_next = hidden.clone()

        hidden = self.rmsnorm_eda(hidden)
        expert_prob = torch.softmax(self.router_mlp(hidden).float(), dim=-1)

        bias = self.balancing_biases.detach().float()
        scores = expert_prob - bias if self.subtract_bias else expert_prob + bias
        _, expert_choice = torch.topk(scores, self.top_k, dim=-1)
        # weights are NOT renormalized
        route_prob = torch.gather(expert_prob, dim=-1, index=expert_choice)
        return route_prob, expert_choice.to(torch.int64), router_states_next


class Zonos2MoEFeedForward(nn.Module):
    """Sparse MoE feed-forward: EDA router + fused SwiGLU experts.

    Expert weights use the fused checkpoint layout shared with
    :class:`SparseMoeBlock`:
      - ``experts.gate_up_proj``: (num_experts, 2 * inter, hidden)  # w1 || w3
      - ``experts.down_proj``:    (num_experts, hidden, inter)      # w2
    Dispatch reuses :func:`dispatch_experts`, which prefers the fused Triton
    grouped-GEMM kernel when available and falls back to the naive per-expert
    SwiGLU loop otherwise.
    """

    def __init__(self, config: Zonos2Config, layer_id: int):
        super().__init__()
        self.num_experts = config.moe_n_experts
        hidden = config.hidden_size
        inter = config.moe_inter

        self.router = Zonos2Router(config, layer_id)
        # Held on a bare Module so the params are named
        # ``experts.gate_up_proj`` / ``experts.down_proj``.
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * inter, hidden)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(self.num_experts, hidden, inter)
        )

    def forward(
        self,
        x: torch.Tensor,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route_prob, expert_choice, router_states_next = self.router(x, router_states)
        out = dispatch_experts(
            x,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            self.num_experts,
            expert_choice,
            route_prob,
        )
        return out, router_states_next


class Zonos2DecoderLayer(nn.Module):
    """Pre-norm transformer block; MoE layers thread the EDA router state.

    Equivalent to the reference ``TransformerBlock`` (whose fused
    add+norm is unrolled here into explicit residual adds around plain
    :class:`RMSNorm`). Dense layers use :class:`ParallelGatedMLP`.
    """

    def __init__(self, config: Zonos2Config, layer_id: int, comm_group: TPCommGroup):
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = config.is_moe_layer(layer_id)

        self.attention = Zonos2Attention(config, comm_group)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.is_moe:
            self.feed_forward = Zonos2MoEFeedForward(config, layer_id)
        else:
            self.feed_forward = ParallelGatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                comm_group=comm_group,
                activation="silu",
            )

    def forward(
        self,
        x: torch.Tensor,
        cache_handle: BatchedCacheManager,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, cache_handle)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        if self.is_moe:
            x, router_states = self.feed_forward(x, router_states)
        else:
            x = self.feed_forward(x)
            router_states = None
        x = residual + x
        return x, router_states


class Zonos2ForCausalLM(nn.Module):
    """Zonos2 multi-codebook TTS causal LM.

    ``forward`` maps a frame tensor ``input_ids`` of shape
    ``(tokens, n_codebooks[+1])`` to final hidden states; ``compute_logits``
    projects those to per-codebook logits ``(tokens, n_codebooks,
    audio_vocab)`` with soft-capping. Parameter names follow the reference
    checkpoint (no ``model.`` prefix): ``multi_embedder.*``, ``layers.{i}.*``,
    ``out_norm.weight``, ``multi_output.weight``.
    """

    def __init__(self, config: Zonos2Config, comm_group: TPCommGroup | None = None):
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.config = config
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.audio_vocab
        self.loss_softcap = config.loss_softcap
        self._emb_norm_eps = config.rms_norm_eps

        self.multi_embedder = MultiEmbedding(config, comm_group)

        # Optional speaker conditioning (voice cloning). Raw speaker embeddings
        # are optionally reduced by an LDA affine projection, then projected to
        # hidden size and written into the embedded sequence at the speaker
        # token position(s).
        self.speaker_lda_projection: nn.Linear | None = None
        self.speaker_projection: nn.Linear | None = None
        if config.speaker_enabled:
            if config.speaker_lda_dim:
                self.speaker_lda_projection = nn.Linear(
                    config.speaker_embedding_dim, int(config.speaker_lda_dim), bias=True
                )
                speaker_proj_in = int(config.speaker_lda_dim)
            else:
                speaker_proj_in = config.speaker_embedding_dim
            self.speaker_projection = nn.Linear(
                speaker_proj_in, config.hidden_size, bias=True
            )

        self.layers = nn.ModuleList(
            [Zonos2DecoderLayer(config, i, comm_group) for i in range(config.num_layers)]
        )
        self.out_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Multi-codebook head: hidden -> (audio_vocab * n_codebooks), sharded
        # over the output vocab and all-gathered so callers see full
        self.multi_output = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=self.audio_vocab * self.n_codebooks,
            bias=False,
            gather_output=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        speaker_emb_values: torch.Tensor | None = None,
        speaker_token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Multi-codebook embedding (sum of per-column tables).
        x = self.multi_embedder(input_ids)

        # Inject projected speaker embeddings at the speaker token position(s),
        # after embedding and before emb_norm. No-op unless the model is speaker-
        # enabled and values/positions are supplied.
        if (
            self.speaker_projection is not None
            and speaker_emb_values is not None
            and speaker_token_positions is not None
            and speaker_emb_values.numel() > 0
            and speaker_token_positions.numel() > 0
        ):
            vals = speaker_emb_values
            if self.speaker_lda_projection is not None:
                vals = self.speaker_lda_projection(
                    vals.to(self.speaker_lda_projection.weight.dtype)
                )
            projected = self.speaker_projection(
                vals.to(self.speaker_projection.weight.dtype)
            )
            x = x.index_copy(
                0,
                speaker_token_positions.to(x.device, torch.long),
                projected.to(x.dtype),
            )

        # emb_norm: parameter-free RMSNorm
        x = F.rms_norm(x, (x.shape[-1],), eps=self._emb_norm_eps)

        router_states: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            x, router_states = layer(x, cache_handle, router_states)
        cache_handle.advance_seq_lens()

        return self.out_norm(x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to per-codebook logits + soft-cap.

        Returns ``(*hidden_states.shape[:-1], n_codebooks, audio_vocab)``.
        """
        logits = self.multi_output(hidden_states)
        logits = logits.view(
            *hidden_states.shape[:-1], self.n_codebooks, self.audio_vocab
        )
        if self.loss_softcap > 0:
            logits = softcap(logits, self.loss_softcap)
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights) -> set[str]:
        """Load a Zonos2 checkpoint stream ``(name, tensor)`` into this model.

        Handles the layout differences between the reference checkpoint and
        the fused component parameters used here:

        * ``attention.wkv.weight`` ``(2, kv_dim, hidden)`` -> merged K||V
          (shard 0 = K, shard 1 = V).
        * dense ``feed_forward.w_in.weight`` ``(2, inter, hidden)`` is stored
          ``[up, gate]``; the fused ``gate_up_proj`` wants ``[gate, up]`` so
          the two halves are swapped on load.
        * ``feed_forward.w_out`` -> ``down_proj``.
        * MoE experts in the unfused grouped format (``experts.w1/w2/w3``)
          are fused into ``experts.gate_up_proj`` (``w1`` gate half, ``w3``
          up half) and ``experts.down_proj`` (``w2``); the already-fused
          ``experts.gate_up_proj`` / ``experts.down_proj`` load directly.

        All other keys (embedders, norms, router, wq/wo/gater/temp,
        out_norm, multi_output) already line up by name.
        """
        import re

        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        moe_inter = self.config.moe_inter
        loaded: set[str] = set()

        def _copy(target: str, tensor: torch.Tensor, shard_id=None) -> None:
            if target in params:
                p = params[target]
                loader = getattr(p, "weight_loader", None)
                if loader is not None:
                    loader(p, tensor) if shard_id is None else loader(p, tensor, shard_id)
                else:
                    assert p.data.shape == tensor.shape, (
                        f"{target}: {tuple(p.data.shape)} vs {tuple(tensor.shape)}"
                    )
                    p.data.copy_(tensor)
                loaded.add(target)
            elif target in buffers and shard_id is None:
                buffers[target].copy_(tensor)
                loaded.add(target)
            # unknown key -> ignore (caller can diff against named_parameters)

        for name, tensor in weights:
            if name.startswith("emb_norm"):
                continue  # parameter-free RMSNorm; nothing to load

            m = re.match(r"(layers\.\d+\.attention\.wkv)\.weight$", name)
            if m and tensor.dim() == 3:
                _copy(m.group(1) + ".weight", tensor[0].contiguous(), shard_id=0)  # K
                _copy(m.group(1) + ".weight", tensor[1].contiguous(), shard_id=1)  # V
                continue

            m = re.match(r"(layers\.\d+\.feed_forward)\.w_in\.weight$", name)
            if m and tensor.dim() == 3:
                # reference w_in is [up, gate]; fused gate_up wants [gate, up].
                _copy(m.group(1) + ".gate_up_proj.weight", tensor[1].contiguous(), shard_id=0)
                _copy(m.group(1) + ".gate_up_proj.weight", tensor[0].contiguous(), shard_id=1)
                continue

            m = re.match(r"(layers\.\d+\.feed_forward)\.w_out\.weight$", name)
            if m:
                _copy(m.group(1) + ".down_proj.weight", tensor)
                continue

            m = re.match(r"(layers\.\d+\.feed_forward\.experts)\.w13$", name)
            if m and tensor.dim() == 3:
                base = m.group(1)
                target = base + ".gate_up_proj"
                if target in params:
                    gate_up = torch.cat([tensor[:, 0::2, :], tensor[:, 1::2, :]], dim=1)
                    params[target].data.copy_(gate_up)
                    loaded.add(target)
                continue

            m = re.match(r"(layers\.\d+\.feed_forward\.experts)\.(w1|w2|w3)(?:\.weight)?$", name)
            if m:
                base, which = m.group(1), m.group(2)
                if which == "w1" and (base + ".gate_up_proj") in params:
                    params[base + ".gate_up_proj"].data[:, :moe_inter, :].copy_(tensor)
                    loaded.add(base + ".gate_up_proj")
                elif which == "w3" and (base + ".gate_up_proj") in params:
                    params[base + ".gate_up_proj"].data[:, moe_inter:, :].copy_(tensor)
                    loaded.add(base + ".gate_up_proj")
                elif which == "w2" and (base + ".down_proj") in params:
                    params[base + ".down_proj"].data.copy_(tensor)
                    loaded.add(base + ".down_proj")
                continue

            _copy(name, tensor)

        return loaded
