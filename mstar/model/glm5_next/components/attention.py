"""GLM-5.3-Flash attention modules: NoPE MLA (fork of glm52) + loader-aware KDA.

``Glm5NextMLAAttention`` is ``glm52/components/attention.py`` with the rope
plumbing deleted and the GLM-5.3 geometry read from config — q_lora 1536,
nope 256, **rope 0** (``mla_use_nope``), v_head 256, 64 heads. NoPE
simplifications vs glm52: ``kv_a_proj_with_mqa`` output is the pure 512-dim
latent (nothing to split), no cos/sin threading anywhere, the absorbed path
loses the rope-slice bookkeeping, and the cache latent is exactly
``kv_lora_rank``. Nothing takes positions: NoPE rotates by none, and the
engine's KV resource does the cache bookkeeping.

Engine seam (resource-pool engine): the layer binds the KV and attention
resources once (``bind_resources``), writes its per-token latent through
``kv.write_latent`` and attends through ``attn.run_mla`` over the layer's
own latent plane (``kv.layer_view(kv_plane)``). Only the absorbed path is
served — the engine's MLA backend has an fp32 SDPA fallback, so reduced
CPU configs run the same code.

DSA status (M0): every full-attention layer OWNS a full k-pool indexer
(``indexer_types`` is all "full"; no glm52 IndexShare formula), so the
parameter container is always built and always loads — but selection is
never RUN: serving holds every context to ``index_topk`` = 2048, where
dense MLA is bit-exactly the DSA computation (top-(topk/kpool) of
<= topk/kpool pools is the identity, tail included — assembly spec
section 3.2/6.4, the glm52 M1 regime). The k-pool scoring math (pooled
softmax + ape prior, fixed 2051-wide output, masked — never
data-dependent-shaped) is the M1 port.

``Glm5NextKdaAttention`` adds exactly one thing to the pure-math
``kda.Glm5NextLinearAttention``: the fused-conv ``weight_loader`` the
stacked-param rules dispatch into (checkpoint stores three
``{q,k,v}_conv1d.weight [qkv_dim, 1, K]`` bf16 tensors; the module runs one
fp32 ``[3*qkv_dim, 1, K]`` depthwise conv in q|k|v order). The math file
stays torch-only (lane ground rule 4); loader coupling lives here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.attn.mla import MLAAttentionManager
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.norm import RMSNorm
from mstar.model.glm5_next.config import ATTN, KV_CACHE, LABEL, Glm5NextModelConfig
from mstar.model.glm5_next.kda import Glm5NextLinearAttention

# k_norm eps is hardcoded in the reference indexer (LayerNorm(head_dim,
# eps=1e-6), weight AND bias) — NOT rms_norm_eps. Same value as glm52's.
_INDEXER_K_NORM_EPS = 1e-6


class Glm5NextIndexer(nn.Module):
    """DSA k-pool indexer parameters for one full-attention layer.

    M0 is a checkpoint-shaped parameter container: the six tensors load
    (all bf16 — unlike glm52, none are fp8 here) so the module tree matches
    the weight map, and selection stays off behind the ctx <= index_topk
    guard where dense MLA IS the DSA computation. The M1 port adds the
    scoring: per pool of ``index_kpool`` consecutive tokens, a learned
    softmax over members (``F.linear(x, index_kpool_compress_gate)`` gate
    scores + the ``index_kpool_compress_ape`` per-slot prior, fp32) builds
    a probability-weighted pool key; ReLU'd q-pool dots are head-combined
    by ``weights_proj``; top-(topk/kpool) pools expand x kpool back to
    token indices with the raw tail pool always appended
    (``index_kpool_always_select_tail``) — fixed output width
    ``index_topk + index_kpool - 1``, -1 padded, so the port must mask
    invalid pools to -inf rather than compact them (the HF
    ``pool_valid.any(0)`` shape is a ground-rule-2 violation to not copy).
    """

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
        """``hidden_states (T, hidden)`` flat engine layout, out the same.

        MLA over the compressed 512-dim latent cache, kv_b folded into Q/O.

        The rope halves of the glm52 original are gone (NoPE). ``q_pe``/``k_pe``
        are ZERO vectors of width ``mla_cache_kpe`` (64): MLA's score is
        ``q_nope·k_ckv + q_pe·k_pe``, so zeros make the pe term exactly 0 —
        bit-identical NoPE. Width 64 (not 0) is deliberate — the capturable
        FlashInfer MLA kernel is hard-locked to kpe=64, so this zero-pad is what
        lets decode CUDA-graph capture instead of falling to eager SDPA
        (wiki/glm53-decode-capture). The softmax scale rides the AttentionSpec.
        """
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
        w = self.kv_b_proj.weight  # (H_local*(Dnope+Dv), L)
        h, d_nope, d_v, latent = (
            self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank)
        w = w.view(h, d_nope + d_v, latent)
        w_kc, w_vc = w.split([d_nope, d_v], dim=1)
        self.w_kc = w_kc.contiguous()  # (H_local, Dnope, L)
        self.w_vc = w_vc.contiguous()  # (H_local, Dv,    L)

        self.fused_qkv_a_proj_weight = torch.cat(
            [self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight], dim=0).contiguous()


class Glm5NextKdaAttention(Glm5NextLinearAttention):
    """``kda.Glm5NextLinearAttention`` + the fused-conv checkpoint loader.

    The stacked-param rules route ``{q,k,v}_conv1d.weight`` shards here with
    shard ids ``"q"``/``"k"``/``"v"``; each lands at its row block of the
    fp32 fused ``conv1d.weight`` (channel order q|k|v, matching the
    activation ``cat`` — kda spec pitfall 11), the ``copy_`` doing the
    bf16 -> fp32 promotion HF's ``_keep_in_fp32_modules_strict`` pins.
    Attached-loader + ``_apply`` reattachment follows the repo's parallel
    linears (attributes on parameters do not survive ``to_empty``).

    TP (M1): shard everything by head block — q/k/v/f_b/g_b/b column-wise,
    o row-wise, conv channels + dt_bias by the same head blocks, A_log by
    head; f_a/g_a and o_norm replicate. This class is where that lands so
    ``kda.py`` stays a single-rank math file. Until it lands, KDA runs
    REPLICATED on every rank (~9 GB of weights and 8x the FLOPs at TP8)
    and the state pool sizes full-width (~147.6 MB/request, not the
    sharded ~18.5) — land the sharding before trusting TP8 bring-up
    tok/s or pool-memory headroom as baselines.
    """

    _CONV_SHARD_ROW = {"q": 0, "k": 1, "v": 2}

    def __init__(self, config, dtype: torch.dtype | None = None) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        super().__init__(config, dtype=dtype)
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        self.conv1d.weight.weight_loader = self._conv_weight_loader

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
        expected = (self.qkv_dim, 1, self.conv_kernel_size)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"{loaded_shard_id}_conv1d.weight has shape "
                f"{tuple(loaded_weight.shape)}; expected {expected}"
            )
        rows = slice(branch * self.qkv_dim, (branch + 1) * self.qkv_dim)
        param.data[rows].copy_(loaded_weight)
