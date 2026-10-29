"""The depth loop of residual-codebook speech decoders (Qwen3-Omni, Qwen3-TTS).

After the backbone samples codec group 0 of a frame, a small pre-norm transformer
(the code predictor) walks groups 1..G-1 one position at a time: position 0 holds
the backbone's hidden state, position g the embedding of group g's code, and each
position's output predicts the next group through that group's LM head. Its KV
cache is frame-local scratch, so the whole loop is fixed-shape and captures into
the backbone's decode graph.

``DepthDecoder`` runs that loop for a code predictor with the standard layer
anatomy (``input_layernorm``, ``self_attn`` with ``qkv_proj``, ``q_norm``, ``k_norm``
and ``o_proj``, ``post_attention_layernorm``, a gated ``mlp``). At decode batch sizes
the loop is bound by its kernel count, so where the platform has the fused depth
kernels (``mstar.platforms.depth_decoder_ops``) it runs as a fixed sequence of
them, eagerly: per layer the QKV GEMM, one attention kernel that also normalizes,
rotates and appends k and v, the output GEMM, a fused residual add and norm, the
gate/up GEMM, the activation, the down GEMM. Elsewhere each model's own
per-position forward, ``forward_depth_unrolled``, runs as before.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.rms_norm import run_rms_norm
from mstar.platforms import depth_decoder_ops as ops
from mstar.platforms.dispatch import PlatformDispatch


class DepthDecoder(nn.Module):
    """Base class for a code predictor; see the module docstring.

    A subclass provides ``model.layers``, ``model.norm``, ``model.codec_embedding``
    (the table of group g at ``g - 1``), the stacked LM heads ``lm_head_weight``
    (``[G - 1, vocab, hidden]``), ``input_projection`` (applied to every depth
    input, or None) and ``forward_depth_unrolled``, its per-position forward.
    """

    def __init__(self) -> None:
        super().__init__()
        # Frame-local KV scratch: [layers, max rows, K/V, groups, kv heads, head dim].
        # Every frame overwrites it from position 0 before reading it, and its fixed
        # address is what a captured decode graph reads and writes.
        self._depth_kv: torch.Tensor | None = None
        self._fusable: bool | None = None

    @property
    def input_projection(self) -> nn.Module | None:
        return None

    @property
    def num_code_groups(self) -> int:
        return len(self.model.codec_embedding) + 1

    @torch.compiler.disable
    def depth_kv_cache(self, batch_size: int, max_batch_size: int) -> torch.Tensor:
        """The KV scratch for ``batch_size`` rows, allocated once for ``max_batch_size``
        (eagerly, so a compiled forward never owns it)."""
        if self._depth_kv is None:
            attn = self.model.layers[0].self_attn
            weight = attn.qkv_proj.weight
            self._depth_kv = torch.zeros(
                (len(self.model.layers), max_batch_size, 2, self.num_code_groups, attn.num_kv_heads, attn.head_dim),
                dtype=weight.dtype, device=weight.device,
            )
        return self._depth_kv[:, :batch_size]

    @torch.compiler.disable(recursive=False)
    def run_depth_loop(
        self,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        group0_embedding: nn.Embedding,
        sample: Callable[[torch.Tensor], torch.Tensor],
        kv_cache: torch.Tensor,
        sum_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk codec groups 1..G-1 for one frame per row.

        ``last_hidden`` is the backbone's hidden state per row, ``layer0_codes`` the
        group-0 codes it sampled and ``group0_embedding`` the backbone's codec table.
        ``sample`` maps one group's logits to codes. Returns ``all_codes``
        (``[rows, G]`` int64) and the sum of the frame's G code embeddings, the next
        backbone step's input, in ``sum_dtype`` (default: the embeddings' dtype).
        """
        sum_dtype = sum_dtype or group0_embedding.weight.dtype
        return _depth_loop(self, last_hidden, layer0_codes, group0_embedding, sample, kv_cache, sum_dtype)

    # ------------------------------------------------------------------
    # each model's own per-position forward
    # ------------------------------------------------------------------

    def _reference_loop(self, last_hidden, layer0_codes, group0_embedding, sample, kv_cache, sum_dtype):
        rows = layer0_codes.shape[0]
        all_codes = torch.empty((rows, self.num_code_groups), dtype=torch.long, device=layer0_codes.device)
        all_codes[:, 0] = layer0_codes
        embed = group0_embedding(layer0_codes)
        embed_sum = embed.to(sum_dtype, copy=True)
        pos = torch.zeros((rows, 1), dtype=torch.int32, device=layer0_codes.device)
        self.forward_depth_unrolled(last_hidden.unsqueeze(1), pos, kv_cache, cache_pos=0)
        for group in range(1, self.num_code_groups):
            pos.fill_(group)
            hidden = self.forward_depth_unrolled(embed.unsqueeze(1), pos, kv_cache, cache_pos=group).squeeze(1)
            codes = sample(torch.matmul(hidden, self.lm_head_weight[group - 1].t()))
            all_codes[:, group] = codes
            embed = self.model.codec_embedding[group - 1](codes)
            embed_sum.add_(embed)
        return all_codes, embed_sum

    # ------------------------------------------------------------------
    # the fused loop
    # ------------------------------------------------------------------

    def _fused_loop_fits(self) -> bool:
        """Whether every layer computes what the fused sequence does: q/k norms with
        one eps, plain (not Gemma) RMSNorms, a SiLU-gated MLP, whole heads."""
        if self._fusable is None:
            def fits(layer) -> bool:
                attn = layer.self_attn
                norms = (layer.input_layernorm, layer.post_attention_layernorm, attn.q_norm, attn.k_norm)
                return (
                    attn.q_norm is not None and attn.k_norm is not None
                    and attn.q_norm.variance_epsilon == attn.k_norm.variance_epsilon
                    and not any(norm.gemma_mode for norm in norms)
                    and layer.mlp.act is F.silu and attn.num_heads % attn.num_kv_heads == 0
                    and attn.head_dim % 2 == 0
                )
            self._fusable = all(fits(layer) for layer in self.model.layers) and not self.model.norm.gemma_mode
        return self._fusable

    @torch.compiler.disable
    def _fused_loop(self, last_hidden, layer0_codes, group0_embedding, sample, kv_cache, sum_dtype):
        if not self._fused_loop_fits():
            return self._reference_loop(last_hidden, layer0_codes, group0_embedding, sample, kv_cache, sum_dtype)
        rows = layer0_codes.shape[0]
        table = group0_embedding.weight
        all_codes = torch.empty((rows, self.num_code_groups), dtype=torch.long, device=table.device)
        embed = torch.empty((rows, table.shape[1]), dtype=table.dtype, device=table.device)
        embed_sum = torch.empty((rows, table.shape[1]), dtype=sum_dtype, device=table.device)
        ops.codec_embed_accumulate(layer0_codes, table, embed, embed_sum, all_codes, 0, False)
        # Position 0 only fills the KV cache. Its residual stream is updated in place,
        # so it runs on a copy unless a projection makes one anyway.
        first = last_hidden if self.input_projection is not None else last_hidden.clone()
        self._fused_position(first, kv_cache, 0, kv_only=True)
        for group in range(1, self.num_code_groups):
            hidden = self._fused_position(embed, kv_cache, group)
            codes = sample(ops.decode_linear(hidden, self.lm_head_weight[group - 1]))
            ops.codec_embed_accumulate(
                codes, self.model.codec_embedding[group - 1].weight, embed, embed_sum, all_codes, group, True,
            )
        return all_codes, embed_sum

    def _fused_position(
        self, x: torch.Tensor, kv_cache: torch.Tensor, position: int, kv_only: bool = False,
    ) -> torch.Tensor | None:
        """Every layer at depth ``position`` for ``x`` (``[rows, width]``), which it
        consumes as the residual stream. With ``kv_only`` it stops once the last
        layer has written its K/V."""
        linear = ops.decode_linear
        projection = self.input_projection
        if projection is not None:
            x = linear(x, projection.weight, projection.bias)
        layers = self.model.layers
        residual = x
        first_norm = layers[0].input_layernorm
        h = run_rms_norm(x, first_norm.weight, first_norm.variance_epsilon)
        for index, layer in enumerate(layers):
            attn, mlp = layer.self_attn, layer.mlp
            if index:
                _add_norm(h, residual, layer.input_layernorm)
            qkv = linear(h, attn.qkv_proj.weight, attn.qkv_proj.bias)
            out = ops.depth_attention(
                qkv, attn.q_norm.weight, attn.k_norm.weight, kv_cache[index, :, 0], kv_cache[index, :, 1],
                position, attn.num_heads, attn.num_kv_heads, attn.q_norm.variance_epsilon, float(attn.rope_theta),
            )
            if kv_only and index == len(layers) - 1:
                return None
            out = linear(out, attn.o_proj.weight, attn.o_proj.bias)
            _add_norm(out, residual, layer.post_attention_layernorm)
            gate_up = linear(out, mlp.gate_up_proj.weight, mlp.gate_up_proj.bias)
            h = linear(ops.silu_and_mul(gate_up), mlp.down_proj.weight, mlp.down_proj.bias)
        _add_norm(h, residual, self.model.norm)
        return h


def _add_norm(hidden: torch.Tensor, residual: torch.Tensor, norm: nn.Module) -> None:
    """In place: ``residual += hidden``, then ``hidden = norm(residual)``."""
    torch.ops.mstar.fused_add_rms_norm(hidden, residual, norm.weight, norm.variance_epsilon, norm.gemma_mode)


# The fused loop runs where its kernels are validated (AMD GPUs). Elsewhere every
# model keeps its own per-position forward until an A/B on that platform moves it.
_depth_loop = PlatformDispatch(
    "depth_loop", rocm=lambda: DepthDecoder._fused_loop, default=lambda: DepthDecoder._reference_loop,
)
