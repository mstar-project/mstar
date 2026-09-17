"""GLM-5.3-Flash attention modules: NoPE MLA (fork of glm52) + loader-aware KDA."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.engine.resources.attn.mla import MLAAttentionManager
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.norm import RMSNorm
from mstar.model.glm5_next.config import ATTN, KV_CACHE, LABEL, Glm5NextModelConfig
from mstar.model.glm5_next.kda import Glm5NextKdaConfig, Glm5NextLinearAttention

# k_norm eps is hardcoded in the reference indexer (LayerNorm(head_dim,
# eps=1e-6), weight AND bias) — NOT rms_norm_eps. Same value as glm52's.
_INDEXER_K_NORM_EPS = 1e-6


class Glm5NextIndexer(nn.Module):
    """DSA k-pool indexer parameters for one full-attention layer."""

    def __init__(self, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.kpool = config.index_kpool

        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=_INDEXER_K_NORM_EPS)
        # Pool compression (new vs glm52): additive per-slot prior over the
        # kpool members + the member-gate applied via F.linear (no bias) —
        # both selection-determining, kept exactly (assembly spec, open
        # question 3). ape's softmax runs fp32 in the M1 scoring.
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, config.hidden_size))


class Glm5NextMLAAttention(nn.Module):
    """NoPE MLA over the engine's paged latent cache (absorbed path)."""

    def __init__(
        self,
        config: Glm5NextModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        kv_plane: int | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        if not config.mla_absorb:
            raise ValueError(
                "Glm5NextMLAAttention serves the absorbed path only "
                "(mla_absorb=False was the pre-resource-pool naive fallback)"
            )
        if config.qk_rope_head_dim != 0:
            raise ValueError(
                "Glm5NextMLAAttention is NoPE-only (mla_use_nope): got "
                f"qk_rope_head_dim={config.qk_rope_head_dim}; the glm52 "
                "attention is the roped implementation"
            )

        self.tp_size = comm_group.world_size
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} is not divisible by "
                f"tp_size={self.tp_size}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim  # == qk_nope_head_dim (NoPE)
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        # Width of the (zero) rope slot written into the MLA latent cache: 64
        # so the capturable FlashInfer MLA kernel accepts it (real NoPE rope
        # is 0). See config.mla_cache_kpe.
        self.mla_cache_kpe = config.mla_cache_kpe
        h = self.total_num_heads

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            comm_group, config.q_lora_rank, h * self.qk_head_dim, bias=False)

        # Pure latent output — no decoupled rope key to split off.
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, config.kv_lora_rank,
            h * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.o_proj = RowParallelLinear(
            comm_group, h * config.v_head_dim, config.hidden_size,
            bias=False, input_is_parallel=True, reduce_results=True)

        # Every full-attention layer (the MTP layer included) ships its own
        # FULL indexer — read indexer_types, no IndexShare formula.
        self.layer_idx = layer_idx
        # This layer's plane in the MLA KV pool (compact full-attention
        # index: 3 -> 0, 7 -> 1, ..., 43 -> 10; the MTP draft plane comes
        # after). The layer addresses its own plane so the caller threads no
        # cursor through the stack.
        self.kv_plane = kv_plane
        self.indexer = Glm5NextIndexer(config)

        # The engine's MLA backend applies this scale (AttentionConfig
        # .softmax_scale); the model declares the same value there.
        self.softmax_scale = self.qk_head_dim ** -0.5

        self.register_buffer("w_kc", None, persistent=False)  # (H_local, Dnope, L)
        self.register_buffer("w_vc", None, persistent=False)  # (H_local, Dv,    L)
        self.register_buffer(
            "fused_qkv_a_proj_weight", None, persistent=False)  # (q_lora+L, hidden)

        # Engine resources, bound once at load (NodeSubmodule.bind_node_resources).
        self.kv: KVManager | None = None
        self.attn: MLAAttentionManager | None = None

    def bind_resources(self, resources: dict) -> None:
        self.kv = resources[KV_CACHE]
        self.attn = resources[ATTN]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states (T, hidden)`` flat engine layout, out the same."""
        if self.kv is None or self.attn is None:
            raise RuntimeError(
                "Glm5NextMLAAttention has no engine resources bound; the "
                "submodule's bind_node_resources must run before a forward"
            )
        if self.w_kc is None or self.w_vc is None:
            raise RuntimeError(
                "mla_absorb forward requires process_weights_after_loading() to "
                "have built w_kc/w_vc from kv_b_proj first"
            )
        if self.fused_qkv_a_proj_weight is None:
            raise RuntimeError(
                "mla_absorb forward requires process_weights_after_loading() to "
                "have built fused_qkv_a_proj_weight first"
            )
        num_tokens = hidden_states.shape[0]
        h = self.num_heads

        fused = F.linear(hidden_states, self.fused_qkv_a_proj_weight)
        q_c, kv_a = fused.split(
            [self.q_a_proj.out_features, self.kv_lora_rank], dim=-1)
        # FlashInfer RMSNorm needs 64-byte input alignment; decode split views
        # can be contiguous yet start at an unaligned offset (glm52 lesson).
        q_c = q_c.contiguous()
        kv_a = kv_a.clone(memory_format=torch.contiguous_format)

        # The post-q_a_layernorm latent is ALSO the M1 indexer's query input
        # (wq_b reads the same 1536-dim bottleneck) — keep it in a name.
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c).view(num_tokens, h, self.qk_head_dim)
        kv_c = self.kv_a_layernorm(kv_a).view(num_tokens, 1, self.kv_lora_rank)

        q_nope = torch.einsum("thd,hdl->thl", q, self.w_kc)

        # One latent row per token (ckv || zero kpe) into this layer's plane
        # of the paged MLA cache, then attend over the plane. The KV resource
        # planned the slots and the attention resource planned the kernel.
        latent = torch.cat(
            [kv_c.squeeze(1), kv_c.new_zeros(num_tokens, self.mla_cache_kpe)], dim=-1,
        )
        self.kv.write_latent(latent, layer_idx=self.kv_plane, label=LABEL)
        attn_latent = self.attn.run_mla(
            q_nope,
            q.new_zeros(num_tokens, h, self.mla_cache_kpe),
            label=LABEL,
            kv_cache_layer=self.kv.layer_view(self.kv_plane),
        )

        out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
        return self.o_proj(out.reshape(num_tokens, h * self.v_head_dim))

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        """Build absorbed Q/O projections from the local-head ``kv_b_proj`` shard."""
        del device  # protocol arg; kv_b_proj.weight already carries the right device
        # Detached: these are derived inference buffers, and the load path is
        # not under no_grad — a grad_fn here would pin the params' autograd
        # graph for the model's lifetime.
        w = self.kv_b_proj.weight.detach()  # (H_local*(Dnope+Dv), L)
        h, d_nope, d_v, latent = (
            self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank)
        w = w.view(h, d_nope + d_v, latent)
        w_kc, w_vc = w.split([d_nope, d_v], dim=1)
        self.w_kc = w_kc.contiguous()  # (H_local, Dnope, L)
        self.w_vc = w_vc.contiguous()  # (H_local, Dv,    L)

        self.fused_qkv_a_proj_weight = torch.cat(
            [self.q_a_proj.weight.detach(), self.kv_a_proj_with_mqa.weight.detach()],
            dim=0,
        ).contiguous()


def _slice_rows_loader(
    tp_rank: int, tp_size: int, full_rows: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | None = None,
) -> None:
    """Head-block row shard of a checkpoint tensor whose dim 0 is
    ``full_rows`` (q/k/v/f_b/g_b/b projections, dt_bias, A_log).
    """
    del loaded_shard_id
    rows = divide(full_rows, tp_size)
    if loaded_weight.shape[0] == full_rows:
        loaded_weight = loaded_weight[tp_rank * rows:(tp_rank + 1) * rows]
    elif loaded_weight.shape[0] != rows:
        raise ValueError(
            f"KDA tensor has {loaded_weight.shape[0]} rows; expected the full "
            f"{full_rows} or the per-rank {rows}"
        )
    param.data.copy_(loaded_weight)


def _slice_cols_loader(
    tp_rank: int, tp_size: int, full_cols: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | None = None,
) -> None:
    """Contraction-dim (dim 1) twin of :func:`_slice_rows_loader` for the
    row-parallel ``o_proj``."""
    del loaded_shard_id
    cols = divide(full_cols, tp_size)
    if loaded_weight.shape[1] == full_cols:
        loaded_weight = loaded_weight[:, tp_rank * cols:(tp_rank + 1) * cols]
    elif loaded_weight.shape[1] != cols:
        raise ValueError(
            f"KDA o_proj has {loaded_weight.shape[1]} cols; expected the full "
            f"{full_cols} or the per-rank {cols}"
        )
    param.data.copy_(loaded_weight)


class Glm5NextKdaAttention(Glm5NextLinearAttention):
    """``kda.Glm5NextLinearAttention`` head-sharded across TP + the
    checkpoint loaders.
    """

    _CONV_SHARD_ROW = {"q": 0, "k": 1, "v": 2}

    def __init__(
        self, config, comm_group: CommGroup | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_size = comm_group.world_size
        self.tp_rank = comm_group.rank
        self.total_num_heads = config.linear_num_heads
        local_heads = divide(config.linear_num_heads, self.tp_size)
        # The math file sees a config with this rank's heads only; all its
        # widths (qkv_dim, conv channels, gate widths) follow from that.
        local = Glm5NextKdaConfig(
            hidden_size=config.hidden_size,
            linear_num_heads=local_heads,
            linear_head_dim=config.linear_head_dim,
            linear_conv_kernel_size=config.linear_conv_kernel_size,
            gate_lower_bound=config.gate_lower_bound,
            rms_norm_eps=config.rms_norm_eps,
        )
        super().__init__(local, dtype=dtype)
        self.total_qkv_dim = self.total_num_heads * self.head_dim
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        from functools import partial

        rank, size = self.tp_rank, self.tp_size
        rows = partial(_slice_rows_loader, rank, size, self.total_qkv_dim)
        self.conv1d.weight.weight_loader = self._conv_weight_loader
        self.q_proj.weight.weight_loader = rows
        self.k_proj.weight.weight_loader = rows
        self.v_proj.weight.weight_loader = rows
        self.forget_gate.f_b_proj.weight.weight_loader = rows
        self.forget_gate.dt_bias.weight_loader = rows
        self.g_b_proj.weight.weight_loader = rows
        heads = partial(_slice_rows_loader, rank, size, self.total_num_heads)
        self.forget_gate.A_log.weight_loader = heads
        self.b_proj.weight.weight_loader = heads
        self.o_proj.weight.weight_loader = partial(
            _slice_cols_loader, rank, size, self.total_qkv_dim,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def _conv_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        branch = self._CONV_SHARD_ROW.get(loaded_shard_id)
        if branch is None:
            raise ValueError(
                f"KDA conv shard id must be q/k/v, got {loaded_shard_id!r}")
        full = (self.total_qkv_dim, 1, self.conv_kernel_size)
        local = (self.qkv_dim, 1, self.conv_kernel_size)
        if tuple(loaded_weight.shape) == full:
            loaded_weight = loaded_weight[
                self.tp_rank * self.qkv_dim:(self.tp_rank + 1) * self.qkv_dim
            ]
        elif tuple(loaded_weight.shape) != local:
            raise ValueError(
                f"{loaded_shard_id}_conv1d.weight has shape "
                f"{tuple(loaded_weight.shape)}; expected {full} or the per-rank {local}"
            )
        rows = slice(branch * self.qkv_dim, (branch + 1) * self.qkv_dim)
        param.data[rows].copy_(loaded_weight)

    # -- TP: sum the o_proj partials once per layer -----------------------

    def prefill(self, hidden_states, recurrent_state=None, conv_state=None, attention_mask=None):
        output, recurrent_state, conv_state = super().prefill(
            hidden_states, recurrent_state=recurrent_state,
            conv_state=conv_state, attention_mask=attention_mask,
        )
        if self.tp_size > 1:
            output = self.comm_group.all_reduce(output)
        return output, recurrent_state, conv_state

    def decode_step(self, hidden_states, recurrent_state, conv_state):
        output = super().decode_step(hidden_states, recurrent_state, conv_state)
        if self.tp_size > 1:
            output = self.comm_group.all_reduce(output)
        return output
