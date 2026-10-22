"""Multi-head latent attention (DeepSeek-V2/V3's MLA) over the engine's paged KV cache.

``config`` carries the MLA dims: ``hidden_size``, ``num_attention_heads``, ``q_lora_rank``,
``kv_lora_rank``, ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``qk_head_dim``, ``v_head_dim``,
``padded_head_dim``, ``rms_norm_eps``, ``rope_theta`` and ``mla_absorb``, the backend:

* absorbed: once the weights load, ``kv_b_proj`` folds into the query (``w_kc``) and the output
  (``w_vc``), and the two down projections into one GEMM. Attention runs over the cache's compressed
  rows (the ``kv_lora_rank`` latent and the decoupled RoPE key) through the MLA attention resource.
* naive: per-head K/V padded to the attention backend's head size, the reduced-dims parity path.

The decoupled RoPE is interleaved, in fp32 (``InterleavedRotaryEmbedding``). A model's sparse
attention plugs in by overriding ``_select`` and ``_attend_selected`` (absorbed) or ``_attend_naive``
(naive); the keyword arguments ``forward`` takes beyond its own reach them unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.norm import RMSNorm
from mstar.model.components.rope import InterleavedRotaryEmbedding


class MLAAttention(nn.Module):
    def __init__(
        self,
        config,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        *,
        kv_key: str,
        attn_key: str,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()

        self.tp_size = comm_group.world_size
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} is not divisible by "
                f"tp_size={self.tp_size}"
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.padded_head_dim = config.padded_head_dim
        h = self.total_num_heads

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            comm_group, config.q_lora_rank, h * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            comm_group, config.kv_lora_rank,
            h * (config.qk_nope_head_dim + config.v_head_dim), bias=False)

        self.o_proj = RowParallelLinear(
            comm_group, h * config.v_head_dim, config.hidden_size,
            bias=False, input_is_parallel=True, reduce_results=True)

        self.rotary = InterleavedRotaryEmbedding(
            rotary_dim=config.qk_rope_head_dim, base=config.rope_theta)

        self.layer_idx = layer_idx
        # run_attention uses 1/sqrt(padded_head_dim); fold the intended
        # qk_head_dim**-0.5 into q on the padded path. No Yarn -> no mscale.
        self.softmax_scale_boost = math.sqrt(self.padded_head_dim / self.qk_head_dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        self.mla_absorb = config.mla_absorb
        if self.mla_absorb:
            self.register_buffer("w_kc", None, persistent=False)  # (H_local, Dnope, L)
            self.register_buffer("w_vc", None, persistent=False)  # (H_local, Dv,    L)
            self.register_buffer("fused_qkv_a_proj_weight", None, persistent=False)  # (q_lora+L+Drope, hidden)
        # engine-built, bound once at load (bind_resources)
        self._kv_key, self._attn_key = kv_key, attn_key
        self._kv = None
        self._attn = None

    def bind_resources(self, resources: dict) -> None:
        self._kv = resources[self._kv_key]
        self._attn = resources[self._attn_key]

    @property
    def cache_layer_idx(self) -> int:
        """The KV layer this attention writes and reads."""
        return 0 if self.layer_idx is None else self.layer_idx

    # The layer index is read only here, outside dynamo: a traced frame that
    # reads it guards on it and compiles once per layer.
    @torch.compiler.disable
    def _write_and_attend(self, q, k, v, q_pe=None):
        layer_idx = self.cache_layer_idx
        self._kv.write_kv(k, v, layer_idx=layer_idx)
        layer = self._kv.layer_view(layer_idx)
        if q_pe is None:
            return self._attn.run(q, kv_cache_layer=layer)
        return self._attn.run(q, kv_cache_layer=layer, q_pe=q_pe)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        **sparse,
    ) -> torch.Tensor:
        """``rope_cos_sin``: the forward's shared ``InterleavedRotaryEmbedding.cos_sin``, else
        computed here. ``sparse``: the subclass's selection inputs, passed to its hooks."""
        if self.mla_absorb:
            return self._forward_absorbed(hidden_states, position_ids, rope_cos_sin, **sparse)
        num_tokens = hidden_states.shape[0]
        h = self.num_heads

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        latent = self.kv_a_proj_with_mqa(hidden_states)  # (T, L + Drope)
        kv_a, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_a))
        kv = kv.view(num_tokens, h, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # shared MQA rope key

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe, cos_sin=rope_cos_sin)

        q = torch.cat([q_nope, q_pe], dim=-1)  # (T, H, Dqk)
        k_pe = k_pe.expand(num_tokens, h, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (T, H, Dqk)

        qk_pad = self.padded_head_dim - self.qk_head_dim
        q = F.pad(q, [0, qk_pad])  # (T, H, Dpad)
        k = F.pad(k, [0, qk_pad])  # (T, H, Dpad)
        v = F.pad(v, [0, self.padded_head_dim - self.v_head_dim])  # (T, H, Dpad)

        q = q * self.softmax_scale_boost
        attn = self._attend_naive(q, k, v, **sparse)  # (T, H, Dpad)
        attn = attn[..., : self.v_head_dim].reshape(num_tokens, h * self.v_head_dim)
        return self.o_proj(attn)

    def _attend_naive(self, q, k, v, **sparse) -> torch.Tensor:
        """Naive-path attention of ``q`` over the padded ``k`` / ``v``: the cache's, by default."""
        del sparse
        return self._write_and_attend(q, k, v)

    def _forward_absorbed(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        **sparse,
    ) -> torch.Tensor:
        """Run MLA over the compressed latent cache after folding kv_b into Q/O."""
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
        q_c, kv_a, k_pe = fused.split(
            [self.q_a_proj.out_features, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # FlashInfer RMSNorm needs 64-byte input alignment; decode split views can
        # be contiguous yet start at an unaligned offset, so clone kv_a.
        q_c = q_c.contiguous()
        kv_a = kv_a.clone(memory_format=torch.contiguous_format)

        # The post-q_a_layernorm latent is kept: a sparse selection may read it too
        # (GLM-5.2's DSA indexer queries from it).
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)
        q = q.view(num_tokens, h, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_c = self.kv_a_layernorm(kv_a).view(num_tokens, 1, self.kv_lora_rank)  # (T,1,L)
        k_pe = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)  # (T,1,Drope) shared MQA key

        q_pe, k_pe = self.rotary(position_ids, q_pe, k_pe, cos_sin=rope_cos_sin)

        selection = self._select(hidden_states, q_c, position_ids, **sparse)

        q_nope = torch.einsum("thd,hdl->thl", q_nope, self.w_kc)

        if selection is None:
            # dense: the paged kernel path over every cached row
            latent = torch.cat([kv_c, k_pe], dim=-1).squeeze(1)
            attn_latent = self._write_and_attend(q_nope, latent, None, q_pe=q_pe)
        else:
            attn_latent = self._attend_selected(selection, q_nope, q_pe, kv_c, k_pe, **sparse)

        out = torch.einsum("thl,hdl->thd", attn_latent, self.w_vc)
        return self.o_proj(out.reshape(num_tokens, h * self.v_head_dim))

    def _select(self, hidden_states, q_c, position_ids, **sparse) -> torch.Tensor | None:
        """The rows each query attends, or None for every cached row (the default)."""
        del hidden_states, q_c, position_ids, sparse
        return None

    def _attend_selected(self, selection, q_nope, q_pe, kv_c, k_pe, **sparse) -> torch.Tensor:
        """Absorbed attention over ``_select``'s rows: ``(T, H_local, kv_lora_rank)``."""
        raise NotImplementedError(f"{type(self).__name__} selects rows but attends none")

    # The absorbed forward reads only w_kc / w_vc / fused_qkv_a_proj_weight;
    # these source projections would otherwise stay resident for the process
    # lifetime (~2.8 GB per rank on GLM-5.2's TP8 full model — KV pages lost).
    _ABSORBED_SOURCE_PROJS = ("q_a_proj", "kv_a_proj_with_mqa", "kv_b_proj")

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        """Build absorbed Q/O projections from the local-head ``kv_b_proj`` shard,
        then release the source weights the absorbed forward never reads."""
        if not self.mla_absorb:
            return
        del device  # protocol arg; kv_b_proj.weight already carries the right device
        if self.absorbed_sources_released:
            return  # fused and freed already: nothing left to fold from
        w = self.kv_b_proj.weight  # (H_local*(Dnope+Dv), L)
        h, d_nope, d_v, latent = (
            self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank)
        w = w.view(h, d_nope + d_v, latent)
        w_kc, w_vc = w.split([d_nope, d_v], dim=1)
        # clone, not contiguous(): with one local head the split slices are
        # already contiguous views, and a view would pin the storage we free
        self.w_kc = w_kc.clone(memory_format=torch.contiguous_format)  # (H_local, Dnope, L)
        self.w_vc = w_vc.clone(memory_format=torch.contiguous_format)  # (H_local, Dv,    L)

        self.fused_qkv_a_proj_weight = torch.cat(
            [self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight], dim=0).contiguous()
        self._release_absorbed_sources()

    @property
    def absorbed_sources_released(self) -> bool:
        return (
            self.mla_absorb and self.w_kc is not None
            and self.kv_b_proj.weight.numel() == 0
        )

    def _release_absorbed_sources(self) -> None:
        for name in self._ABSORBED_SOURCE_PROJS:
            param = getattr(self, name).weight
            # keep the Parameter (its weight_loader, dtype, device) and the
            # Linear's in/out_features the forward still reads; drop only the
            # storage — the same ``.data`` rebinding restore_fp32_params uses
            param.data = param.data.new_empty(0)
