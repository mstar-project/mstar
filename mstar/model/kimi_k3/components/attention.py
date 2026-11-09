"""Kimi K3's gated MLA and K3.1's DSA indexer, on the shared MLA (``components/mla.py``).

The MLA is DeepSeek-V3's with Kimi's NoPE (the 64 decoupled dims kept, unrotated) and a
sigmoid output gate, ``o_proj(attn * sigmoid(g_proj(x)))``. K3.1's layers add a DeepSeek-V3.2
indexer (no rope, LayerNorm key norm); its keys and queries go to the DSA resource
(``mstar/engine/resources/dsa``), which keeps the keys on the MLA cache's pages (one index
plane per MLA layer), scores them and picks each query's top ``index_topk``, as glm52's do.
The indexer math matches vLLM's K3.1 (``vllm/models/kimi_k3``): the weights scaled by
``index_head_dim^-0.5 * index_n_heads^-0.5``, an fp32 LayerNorm on the keys, no rope.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear
from mstar.model.components.mla import MLAAttention
from mstar.model.kimi_k3.config import (
    ATTN,
    DSA,
    INDEXER_K_NORM_EPS,
    KV_CACHE,
    LABEL,
    KimiK3ModelConfig,
)
from mstar.utils import small_m_gemm

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

if triton is not None:

    @triton.jit
    def _gate_mul_kernel(attn, s_attn, gate, s_gate, out, W: tl.constexpr, BLOCK: tl.constexpr):
        """Row t, columns [c * BLOCK, (c + 1) * BLOCK): ``attn * sigmoid(gate)``, each rounded to
        the inputs' dtype as torch's two kernels round them (the sigmoid, then the product)."""
        t = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = c < W
        g = tl.load(gate + t * s_gate + c, mask=m, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g).to(gate.dtype.element_ty).to(tl.float32)
        a = tl.load(attn + t * s_attn + c, mask=m, other=0.0).to(tl.float32)
        tl.store(out + t * W + c, (a * sig).to(out.dtype.element_ty), mask=m)


@torch.compiler.disable
def _gate_mul(attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn * sigmoid(gate)`` for ``(T, W)`` rows (``gate`` may be strided rows, unit stride on W):
    one kernel on CUDA where torch launches two (B300 K3.1 decode: 1.8 + 1.5 us, 24 layers a step)."""
    if (triton is None or not attn.is_cuda or attn.dim() != 2 or gate.shape != attn.shape
            or attn.stride(1) != 1 or gate.stride(1) != 1 or gate.dtype != attn.dtype):
        return attn * torch.sigmoid(gate)
    T, W = attn.shape
    out = torch.empty_like(attn, memory_format=torch.contiguous_format)
    if T:
        block = 1024
        _gate_mul_kernel[(T, triton.cdiv(W, block))](
            attn, attn.stride(0), gate, gate.stride(0), out, W=W, BLOCK=block, num_warps=4)
    return out


class KimiIndexer(nn.Module):
    """One layer's DSA indexer: each token's index key and each query's heads and head weights.
    Replicated (7.4M parameters a layer)."""

    def __init__(self, config: KimiK3ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=INDEXER_K_NORM_EPS)  # weight + bias
        # the softmax scale and n_heads^-0.5 fold into the head weights, as DeepSeek-V3.2 does
        self.weight_scale = self.head_dim ** -0.5 * self.n_heads ** -0.5

    def compute_k(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(T, head_dim)``: what the DSA resource stores per token."""
        return self.k_norm(self.wk(hidden_states))

    def compute_query(
        self, q_c: torch.Tensor, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Queries ``(T, H, D)`` and head weights ``(T, H)``, fp32, from the normed q latent."""
        q = self.wq_b(q_c).view(q_c.shape[0], self.n_heads, self.head_dim).float()
        w = self.weights_proj(hidden_states).float() * self.weight_scale
        return q, w


class KimiMLAAttention(MLAAttention):
    """The shared MLA over this layer's plane of the latent cache (``kv_plane``: its index among
    the MLA layers), with the output gate and, under ``dsa_long_context``, the indexer."""

    def __init__(
        self,
        config: KimiK3ModelConfig,
        comm_group: CommGroup | None = None,
        layer_idx: int | None = None,
        kv_plane: int | None = None,
    ) -> None:
        super().__init__(config, comm_group, layer_idx, kv_key=KV_CACHE, attn_key=ATTN,
                         pos_key=None, kv_label=LABEL, cache_layer=kv_plane)
        self.kv_plane = kv_plane
        self.g_proj = None
        if config.mla_use_output_gate:
            self.g_proj = ColumnParallelLinear(
                comm_group or CommGroup.trivial(), config.hidden_size,
                config.num_attention_heads * config.v_head_dim, bias=False)
        # Built only where the DSA resource is declared: off it the replicated indexer would be
        # held for nothing, and the loader skips its keys to match.
        self.indexer = KimiIndexer(config) if config.dsa_long_context else None
        self._dsa = None
        # [q_a | kv_a | g_proj]: the output gate reads the layer input too, so its projection
        # rides in q_a / kv_a's GEMM (``process_weights_after_loading``); ``fused_qkv_a_proj_weight``
        # and ``g_proj.weight`` view it. ``_gate_pre`` holds this forward's gate until ``_project_out``.
        self.register_buffer("_qkv_a_g_weight", None, persistent=False)
        self._gate_pre: tuple[torch.Tensor, torch.Tensor] | None = None

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        self._dsa = resources.get(DSA)
        if self._dsa is not None and self.indexer is None:
            raise RuntimeError("the DSA resource is declared but this layer has no indexer "
                               "(built with dsa_long_context off)")
        if self._dsa is not None and not self.mla_absorb:
            raise RuntimeError("DSA reads the paged latent cache, which only the absorbed MLA keeps")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states (T, hidden)`` in the engine's flat layout; NoPE takes no positions."""
        return super().forward(hidden_states, None)

    def process_weights_after_loading(self, device: torch.device | str | None = None) -> None:
        super().process_weights_after_loading(device)
        g, qkv = self.g_proj, self.fused_qkv_a_proj_weight
        if (g is None or qkv is None or self._qkv_a_g_weight is not None or self.fp8_block is not None
                or getattr(g, "quant_format", None) is not None
                or (g.weight.dtype, g.weight.device) != (qkv.dtype, qkv.device)):
            return
        self._qkv_a_g_weight = torch.cat([qkv, g.weight.detach()]).contiguous()
        self._view_qkv_a_g()

    def _view_qkv_a_g(self) -> None:
        rows = self._qkv_a_g_weight.shape[0] - self.g_proj.weight.shape[0]
        self.fused_qkv_a_proj_weight = self._qkv_a_g_weight[:rows]
        self.g_proj.weight.data = self._qkv_a_g_weight[rows:]

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self._qkv_a_g_weight is not None:
            # `_apply` moved the weights apart from the packed one
            self._view_qkv_a_g()
        return result

    def _fused_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """q_a | kv_a, with the output gate's pre-activation from the same GEMM kept for
        ``_project_out``."""
        if self._qkv_a_g_weight is None:
            return super()._fused_qkv_a(hidden_states)
        out = small_m_gemm.linear(hidden_states, self._qkv_a_g_weight)
        rows = self.fused_qkv_a_proj_weight.shape[0]
        self._gate_pre = (hidden_states, out[..., rows:])
        return out[..., :rows]

    def _project_out(self, attn: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.g_proj is not None:
            # this forward's gate from the q_a GEMM, or its own GEMM on a path that ran none
            pre, self._gate_pre = self._gate_pre, None
            gate = pre[1] if pre is not None and pre[0] is hidden_states else self.g_proj(hidden_states)
            attn = _gate_mul(attn, gate)
        return self.o_proj(attn)

    # -- DSA (dsa_long_context) ------------------------------------------------

    def _selects(self, **sparse) -> bool:
        return self._dsa is not None and self._dsa.current(LABEL) is not None

    def _select(self, hidden_states, q_c, position_ids, **sparse):
        # None-guarded here so the dense path never crosses the compiler.disable boundary
        if self._dsa is None:
            return None
        return self._dsa_select(hidden_states, q_c)

    # The resource's host state is read only here, outside dynamo.
    @torch.compiler.disable
    def _dsa_select(self, hidden_states: torch.Tensor, q_c: torch.Tensor) -> torch.Tensor | None:
        """Write this layer's index keys to its plane; where a row sees more than
        ``index_topk`` keys (every row of a captured decode), each row's selected cache slots."""
        ctx = self._dsa.current(LABEL)
        if ctx is None:  # a pass DSA doesn't cover (DSAManager.dense)
            return None
        self._dsa.write_index(self.kv_plane, self.indexer.compute_k(hidden_states), label=LABEL)
        if not ctx.needs_selection:
            return None
        q, weights = self.indexer.compute_query(q_c, hidden_states)
        return self._dsa.select(self.kv_plane, q, weights, label=LABEL)

    @torch.compiler.disable
    def _attend_selected(self, selection, q_nope, q_pe, latent, **sparse) -> torch.Tensor:
        """Write this step's latents, then sparse absorbed MLA over the selected slots."""
        layer = self.cache_layer_idx
        self._kv.write_kv(latent, None, layer_idx=layer, **self._label)
        return self._dsa.attend(layer, q_nope, q_pe, selection, self.softmax_scale, label=LABEL)
