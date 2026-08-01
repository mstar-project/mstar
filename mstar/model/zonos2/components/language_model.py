"""Zonos2 language model: the multi-codebook TTS transformer.

The data flow agrees with the reference::

    input_ids (tokens, n_codebooks[+text])
      -> MultiEmbedding            # sum of the VocabParallelEmbedding columns
      -> emb_norm                  # RMSNorm with no parameters
      -> N x Zonos2DecoderLayer    # pre-norm; attn (temp + gate) + FFN/MoE
      -> out_norm                  # RMSNorm
      -> MultiOutputHead           # linear -> (*, n_codebooks, audio_vocab)
      -> softcap(logits, 15.0)
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components import RMSNorm, SparseMoeBlock
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ParallelGatedMLP,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.zonos2.config import Zonos2Config

# The attention of the reference hardcodes the QK-norm epsilon. See
# ``F.rms_norm(..., eps=1e-6)``.
_QK_NORM_EPS = 1e-6


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft-cap the logits with a tanh: ``cap * tanh(x / cap)``."""
    return cap * torch.tanh(x / cap)


class MultiEmbedding(nn.Module):
    """Sum of the token embeddings of each column: 9 audio codebooks and text.

    The module keeps one :class:`VocabParallelEmbedding` for each column. It
    adds their lookups element-wise into one hidden state. The checkpoint layout
    is ``multi_embedder.embedders.{i}.weight``, with the audio columns first and
    the text column last.
    """

    def __init__(self, config: Zonos2Config, comm_group: CommGroup):
        super().__init__()
        self.n_codebooks = config.n_codebooks

        embedders: list[nn.Module] = []
        # The audio codebook tables. Each one uses padding_idx = audio_pad_id.
        for _ in range(config.n_codebooks):
            embedders.append(
                VocabParallelEmbedding(
                    num_embeddings=config.codebook_size + 2,
                    embedding_dim=config.hidden_size,
                    comm_group=comm_group,
                    padding_idx=config.audio_pad_id,
                )
            )
        # The optional text table comes last. It uses padding_idx = text_vocab.
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
        # ``codes`` is (tokens, num_columns). Embedder i looks up column i. Call
        # ``.contiguous()``, because ``codes[..., i]`` is a strided view.
        out = self.embedders[0](codes[..., 0].contiguous())
        for i in range(1, codes.shape[-1]):
            out = out + self.embedders[i](codes[..., i].contiguous())
        return out


class Zonos2Attention(nn.Module):
    """Self-attention with QK-norm, a per-head temperature, and headwise gating.

    This module differs from the shared :class:`Attention` in three ways, so
    the code writes it out here instead of a subclass:

    - QK-norm has no parameters (``F.rms_norm`` with no weight). The code then
      scales the query by a learnable per-head ``|temp|``.
    - RoPE uses the interleaved layout (``is_neox=False``).
    - The code gates the attention output of each head by ``sigmoid(gater(x))``.

    The projections reuse the TP-aware parallel linears: ``wq`` and ``gater``
    (column), ``wkv`` (merged K||V column), and ``wo`` (row).
    """

    def __init__(self, config: Zonos2Config, comm_group: CommGroup):
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
        # Fused K||V. MergedColumnParallelLinear shards the K heads and the V
        # heads independently: shard 0 is K, and shard 1 is V.
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
        # The headwise gate maps hidden to num_heads. It shards over the heads,
        # as wq does.
        self.gater = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=self.num_heads,
            bias=False,
        )

        # A learnable per-head temperature of shape (1, local_num_heads, 1). It
        # broadcasts over the tokens and head_dim, and it shards across the TP
        # ranks.
        self.temp = nn.Parameter(torch.ones(1, self.local_num_heads, 1))
        self._attach_temp_loader()

    def _attach_temp_loader(self) -> None:
        self.temp.weight_loader = self._temp_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_temp_loader()
        return result

    def _temp_loader(self, param, loaded_weight, loaded_shard_id=None):
        # The temp of the checkpoint is (1, num_heads, 1).
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

        # The headwise gate comes from the normed input. The code applies it
        # after the attention.
        gate = torch.sigmoid(self.gater(x))  # (tokens, local_num_heads)

        q = self.wq(x).view(num_tokens, self.local_num_heads, self.head_dim)
        kv = self.wkv(x)
        kv_dim = self.local_num_kv_heads * self.head_dim
        k, v = kv.split([kv_dim, kv_dim], dim=-1)
        k = k.view(num_tokens, self.local_num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.local_num_kv_heads, self.head_dim).contiguous()

        # QK-norm has no parameters. The code also scales the query by |temp|
        # for each head.
        q = F.rms_norm(q, (self.head_dim,), eps=_QK_NORM_EPS) * self.temp.abs().to(q.dtype)
        k = F.rms_norm(k, (self.head_dim,), eps=_QK_NORM_EPS)

        # Interleaved RoPE (is_neox=False). Pass no llama3 scaling kwargs, so
        # that the cache handle keeps the plain rope path.
        q, k = cache_handle.apply_rope(
            q, k, rope_theta=self.rope_theta, interleave=True,
        )

        # Standard scaled-dot-product attention, with a softmax scale of
        # 1/sqrt(dim). The temperature above is an extra learned multiplier on
        # the query.
        o = cache_handle.run_attention(q=q, k=k, v=v)  # (tokens, heads, dim)
        o = o * gate.unsqueeze(-1)
        o = o.reshape(num_tokens, self.local_num_heads * self.head_dim)
        return self.wo(o)


class Zonos2Router(nn.Module):
    """MoE router that threads Expert-Dropout-Augmentation (EDA) state.

    This is a stateful router in the sense of the router contract in
    :mod:`mstar.model.components.moe`. A stock
    :class:`~mstar.model.components.SparseMoeBlock` holds it through that
    block's ``router`` argument, and the block threads its state with
    ``return_router_states=True``.

    The router down-projects the hidden state to ``router_dim``. It can then add
    the router state of the previous MoE layer (EDA). It RMS-normalizes the
    result, runs a 3-layer GELU MLP to get the per-expert logits, applies a
    softmax, and selects a bias-aware top-k. It returns the routing weights, the
    expert indices, and the pre-norm router state for the EDA of the next MoE
    layer.

    Checkpoint layout. The reference calls it ``router``, but the block holds it
    as ``gate``, so ``Zonos2ForCausalLM.load_weights`` rewrites the prefix::

        router.down_proj.{weight,bias}
        router.router_mlp.{0,2,4}.{weight,bias}   # GELU is at indices 1 and 3
        router.rmsnorm_eda.weight
        router.router_states_scale               # EDA layers only
        router.balancing_biases
    """

    def __init__(self, config: Zonos2Config, layer_id: int):
        super().__init__()
        self.num_experts = config.moe_n_experts
        # The top-k of this layer. ``special_topk_layers`` overrides the global
        # default. For example, layer 26 uses top-2 in the reference checkpoint.
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
        # Thread this pre-norm state to the EDA of the next MoE layer.
        router_states_next = hidden.clone()

        hidden = self.rmsnorm_eda(hidden)
        expert_prob = torch.softmax(self.router_mlp(hidden).float(), dim=-1)

        bias = self.balancing_biases.detach().float()
        scores = expert_prob - bias if self.subtract_bias else expert_prob + bias
        _, expert_choice = torch.topk(scores, self.top_k, dim=-1)
        # The router does NOT renormalize the weights.
        route_prob = torch.gather(expert_prob, dim=-1, index=expert_choice)
        return route_prob, expert_choice.to(torch.int64), router_states_next


def build_zonos2_moe(config: Zonos2Config, layer_id: int) -> SparseMoeBlock:
    """Build the MoE feed-forward of one layer: a stock block with the EDA router.

    The result is a plain :class:`~mstar.model.components.SparseMoeBlock` that
    holds a :class:`Zonos2Router`. Zonos2 therefore inherits the shared expert
    dispatch (a fused Triton grouped-GEMM when available, and a naive
    per-expert SwiGLU loop if not) and the fused expert-weight layout::

        experts.gate_up_proj: (num_experts, 2 * inter, hidden)  # w1 || w3
        experts.down_proj:    (num_experts, hidden, inter)      # w2

    The code does not pass ``norm_topk_prob``. That argument configures only the
    block's default :class:`TopKRouter`, and the Zonos2 router does its own
    top-k selection without renormalization.
    """
    return SparseMoeBlock(
        hidden_size=config.hidden_size,
        num_experts=config.moe_n_experts,
        num_experts_per_tok=config.get_num_experts_per_tok(layer_id),
        moe_intermediate_size=config.moe_inter,
        router=Zonos2Router(config, layer_id),
    )


class Zonos2DecoderLayer(nn.Module):
    """Pre-norm transformer block. MoE layers thread the EDA router state.

    This block agrees with the reference ``TransformerBlock``. The reference
    fuses the add and the norm. Here the code writes explicit residual adds
    around a plain :class:`RMSNorm`. Dense layers use
    :class:`ParallelGatedMLP`.
    """

    def __init__(self, config: Zonos2Config, layer_id: int, comm_group: CommGroup):
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = config.is_moe_layer(layer_id)

        self.attention = Zonos2Attention(config, comm_group)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.is_moe:
            self.feed_forward = build_zonos2_moe(config, layer_id)
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
            x, router_states = self.feed_forward(
                x, router_states, return_router_states=True,
            )
        else:
            x = self.feed_forward(x)
            router_states = None
        x = residual + x
        return x, router_states


class Zonos2ForCausalLM(nn.Module):
    """Zonos2 multi-codebook TTS causal LM.

    ``forward`` maps a frame tensor ``input_ids`` of shape
    ``(tokens, n_codebooks[+1])`` to the final hidden states.
    ``compute_logits`` projects those states to per-codebook logits
    ``(tokens, n_codebooks, audio_vocab)`` and soft-caps them. The parameter
    names agree with the reference checkpoint and have no ``model.`` prefix:
    ``multi_embedder.*``, ``layers.{i}.*``, ``out_norm.weight``, and
    ``multi_output.weight``.
    """

    def __init__(self, config: Zonos2Config, comm_group: CommGroup | None = None):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.config = config
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.audio_vocab
        self.loss_softcap = config.loss_softcap
        self._emb_norm_eps = config.rms_norm_eps

        self.multi_embedder = MultiEmbedding(config, comm_group)

        # Optional speaker conditioning (voice cloning). An LDA affine
        # projection can first reduce the raw speaker embedding. The code then
        # projects it to the hidden size and writes it into the embedded
        # sequence at the speaker token positions.
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
        # The multi-codebook head maps hidden to (audio_vocab * n_codebooks). It
        # shards over the output vocabulary, then all-gathers, so that callers
        # see the full logits.
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
        # Multi-codebook embedding: the sum of the tables of each column.
        x = self.multi_embedder(input_ids)

        # Inject the projected speaker embeddings at the speaker token
        # positions, after the embedding and before emb_norm. This does nothing
        # unless the model is speaker-enabled and the caller supplies both the
        # values and the positions.
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

        # emb_norm is an RMSNorm with no parameters.
        x = F.rms_norm(x, (x.shape[-1],), eps=self._emb_norm_eps)

        router_states: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            x, router_states = layer(x, cache_handle, router_states)
        cache_handle.advance_seq_lens()

        return self.out_norm(x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the hidden states to per-codebook logits, then soft-cap them.

        The result is ``(*hidden_states.shape[:-1], n_codebooks,
        audio_vocab)``.
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

        The method handles the layout differences between the reference
        checkpoint and the fused component parameters here:

        * ``attention.wkv.weight`` ``(2, kv_dim, hidden)`` becomes a merged
          K||V parameter: shard 0 is K, and shard 1 is V.
        * The dense ``feed_forward.w_in.weight`` ``(2, inter, hidden)`` holds
          ``[up, gate]``, but the fused ``gate_up_proj`` needs ``[gate, up]``.
          The code therefore swaps the two halves on load.
        * ``feed_forward.w_out`` becomes ``down_proj``.
        * The code fuses the unfused grouped MoE experts
          (``experts.w1/w2/w3``) into ``experts.gate_up_proj`` (``w1`` is the
          gate half, and ``w3`` is the up half) and ``experts.down_proj``
          (``w2``). An already-fused ``experts.gate_up_proj`` or
          ``experts.down_proj`` loads directly.
        * ``feed_forward.router.*`` becomes ``feed_forward.gate.*``. The MoE
          layers are stock :class:`SparseMoeBlock` objects that hold a
          :class:`Zonos2Router`, and that block names its router ``gate``.

        All other keys already agree by name: the embedders, the norms, wq, wo,
        gater, temp, out_norm, and multi_output.
        """
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
            # Ignore an unknown key. The caller can compare the result against
            # named_parameters.

        for name, tensor in weights:
            if name.startswith("emb_norm"):
                continue  # This RMSNorm has no parameters. Load nothing.

            m = re.match(r"(layers\.\d+\.attention\.wkv)\.weight$", name)
            if m and tensor.dim() == 3:
                _copy(m.group(1) + ".weight", tensor[0].contiguous(), shard_id=0)  # K
                _copy(m.group(1) + ".weight", tensor[1].contiguous(), shard_id=1)  # V
                continue

            m = re.match(r"(layers\.\d+\.feed_forward)\.w_in\.weight$", name)
            if m and tensor.dim() == 3:
                # The w_in of the reference is [up, gate]. The fused gate_up
                # needs [gate, up].
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

            # SparseMoeBlock holds its router as ``gate``, but the reference
            # checkpoint calls it ``router``. Everything below it (down_proj,
            # router_mlp, rmsnorm_eda, router_states_scale, balancing_biases)
            # agrees once the code rewrites the prefix.
            m = re.match(r"(layers\.\d+\.feed_forward)\.router\.(.+)$", name)
            if m:
                _copy(f"{m.group(1)}.gate.{m.group(2)}", tensor)
                continue

            _copy(name, tensor)

        return loaded
