"""A Talker's residual code predictor (Qwen3-Omni's and Qwen3-TTS's) and the
kernels of its depth loop: a few layers walked once per codec group, 16 x 5
times per Talker step, over a dense frame-local KV cache.

``cp_qk_norm_rope_kv_write`` is the attention prologue in one launch: per layer
and depth step, q and k normed and roped and k and v copied into the cache had
been six launches of a few microseconds around a ~1 us job.
``cp_attention`` attends over the depth positions so far, and
``cp_embed_accumulate`` records a depth step's codes and sums their embeddings.
``DepthAttention`` is a code predictor layer's attention built from the first
two, over the frame-local KV cache its ``DepthKV`` holds, and
``ResidualCodePredictor`` is the predictor: its layers, heads and depth loop.

A captured forward compiles with static ints, so any compiled frame that holds a
depth step's ``cache_pos`` or a layer index recompiles at every depth position
or layer. Both live on the ``DepthKV``, which the depth loop binds per step and
``run_layers`` moves per layer, and are read only in its eager methods: the
layers' frames, the projections included, compile once per shape.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import triton
import triton.language as tl
from torch import nn

from mstar.model.components.decoder_layer import DecoderLayer, run_layers
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.distributed.attention import ParallelAttention
from mstar.model.components.norm import RMSNorm
from mstar.utils.attention import rms_norm_rope


@triton.jit
def _cp_qk_norm_rope_kv_kernel(
    QKV, Wq, Wk, KV,
    eps_q, eps_k, rope_theta, seq_len, cache_pos,
    s_qkv,
    s_kv_b, s_kv_c, s_kv_n, s_kv_h,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, HALF_D)
    row = QKV + m * s_qkv
    # programs [0, H_Q) take the q heads, [H_Q, H_Q + H_KV) the k heads and their v
    base = row + h * D
    if h < H_Q:
        w = Wq
        eps = eps_q
    else:
        w = Wk
        eps = eps_k

    # token m's depth position, which is also its cache slot: a frame's depth
    # sequence starts at 0
    n = cache_pos + m % seq_len

    x_first = tl.load(base + offs).to(tl.float32)
    x_second = tl.load(base + offs + HALF_D).to(tl.float32)
    out_first, out_second = rms_norm_rope(x_first, x_second, w, n.to(tl.float32), eps, rope_theta, D, HALF_D)

    if h < H_Q:
        tl.store(base + offs, out_first.to(QKV.dtype.element_ty))
        tl.store(base + offs + HALF_D, out_second.to(QKV.dtype.element_ty))
    else:
        hk = h - H_Q
        b = m // seq_len
        k_dst = KV + b * s_kv_b + n * s_kv_n + hk * s_kv_h
        tl.store(k_dst + offs, out_first.to(KV.dtype.element_ty))
        tl.store(k_dst + offs + HALF_D, out_second.to(KV.dtype.element_ty))
        d = tl.arange(0, D)
        v = tl.load(row + (H_Q + H_KV + hk) * D + d)
        tl.store(k_dst + s_kv_c + d, v.to(KV.dtype.element_ty))


@torch.compiler.disable
def cp_qk_norm_rope_kv_write(
    qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor,
    kv_layer: torch.Tensor, cache_pos: int, seq_len: int, eps_q: float, eps_k: float,
    rope_theta: float, n_q_heads: int, n_kv_heads: int, head_dim: int,
) -> None:
    """QK-norm + RoPE the q heads of ``qkv (tokens, (Hq + 2 Hkv) * D)`` in place, and
    write the normed, roped k and the raw v of token ``b * seq_len + s`` into
    ``kv_layer (bs, 2, S, Hkv, D)`` at sequence position ``cache_pos + s``, which is
    also the token's RoPE position."""
    tokens = qkv.shape[0]
    assert qkv.stride(1) == 1 and kv_layer.stride(-1) == 1
    _cp_qk_norm_rope_kv_kernel[(tokens, n_q_heads + n_kv_heads)](
        qkv, q_norm_w, k_norm_w, kv_layer,
        eps_q, eps_k, rope_theta, seq_len, cache_pos,
        qkv.stride(0),
        kv_layer.stride(0), kv_layer.stride(1), kv_layer.stride(2), kv_layer.stride(3),
        H_Q=n_q_heads, H_KV=n_kv_heads, D=head_dim, HALF_D=head_dim // 2,
        num_warps=4, num_stages=1,
    )


@triton.jit
def _cp_embed_accumulate_kernel(
    Tok, W, Emb, Sum, Codes, group, s_w, s_emb, s_sum, s_codes,
    H: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    tok = tl.load(Tok + b)
    offs = tl.arange(0, BLOCK)
    for h0 in tl.static_range(0, H, BLOCK):
        e = tl.load(W + tok * s_w + h0 + offs)
        tl.store(Emb + b * s_emb + h0 + offs, e)
        acc = tl.load(Sum + b * s_sum + h0 + offs)
        tl.store(Sum + b * s_sum + h0 + offs, acc + e.to(acc.dtype))
    tl.store(Codes + b * s_codes + group, tok.to(Codes.dtype.element_ty))


@torch.compiler.disable
def cp_embed_accumulate(
    tokens: torch.Tensor, weight: torch.Tensor, emb_sum: torch.Tensor,
    all_codes: torch.Tensor, group: int,
) -> torch.Tensor:
    """For a depth step's sampled ``tokens (bs,)``: record them as ``all_codes[:, group]``,
    look up their embeddings in ``weight (V, H)`` and add those to ``emb_sum (bs, H)`` in
    its dtype. Returns the embeddings, the next depth step's input. One launch for what
    was an index_select, a copy and a promoting add."""
    bs, hidden = tokens.shape[0], weight.shape[1]
    assert tokens.is_contiguous() and weight.stride(1) == 1 and emb_sum.stride(1) == 1 and hidden % 256 == 0
    emb = torch.empty(bs, hidden, dtype=weight.dtype, device=weight.device)
    _cp_embed_accumulate_kernel[(bs,)](
        tokens, weight, emb, emb_sum, all_codes, group,
        weight.stride(0), emb.stride(0), emb_sum.stride(0), all_codes.stride(0),
        H=hidden, BLOCK=256, num_warps=4,
    )
    return emb



@triton.jit
def _cp_attn_kernel(Q, K, V, O, cache_pos, sm_scale,
                    sq_b, sq_s, sq_h, sk_b, sk_n, sk_h, so_b, so_s, so_h,
                    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    hk = h // GROUP
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + b * sq_b + s * sq_s + h * sq_h + d).to(tl.float32) * sm_scale
    valid = n <= cache_pos + s  # causal within the depth sequence
    kv = b * sk_b + n[:, None] * sk_n + hk * sk_h + d[None, :]
    k = tl.load(K + kv, mask=valid[:, None], other=0.0).to(tl.float32)
    qk = tl.where(valid, tl.sum(k * q[None, :], 1), float("-inf"))
    p = tl.exp(qk - tl.max(qk, 0))
    v = tl.load(V + kv, mask=valid[:, None], other=0.0).to(tl.float32)
    out = tl.sum(p[:, None] * v, 0) / tl.sum(p, 0)
    tl.store(O + b * so_b + s * so_s + h * so_h + d, out.to(O.dtype.element_ty))


@torch.compiler.disable
def cp_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_pos: int) -> torch.Tensor:
    """Causal attention of the code predictor's ``q (bs, S, Hq, D)``, whose tokens sit
    at depth positions ``cache_pos .. cache_pos + S - 1``, over its dense cache
    ``(bs, max_depth, Hkv, D)``. The depth sequence is at most 16 long, so one
    program holds a (batch row, head, token)'s whole key range."""
    bs, seq, hq, d = q.shape
    hkv, max_depth = k_cache.shape[2], k_cache.shape[1]
    assert k_cache.stride() == v_cache.stride() and q.stride(-1) == 1 and k_cache.stride(-1) == 1
    out = torch.empty(bs, seq, hq, d, device=q.device, dtype=q.dtype)
    _cp_attn_kernel[(bs, hq, seq)](
        q, k_cache, v_cache, out, cache_pos, d ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        GROUP=hq // hkv, D=d, BLOCK_N=triton.next_power_of_2(max_depth), num_warps=4,
    )
    return out


class DepthKV:
    """A code predictor's frame-local KV cache and the depth step attending
    over it, as its layers' attention cursor: the depth loop calls
    ``bind_step`` per step, and ``run_layers`` calls ``set_layer_idx`` per
    layer. Its methods are all eager, so the ints stay out of compiled frames
    (and none is ``recursive=False``; see ``run_layers``). A binding serves
    one pass over the layers: the pass's first layer consumes it, so a step
    that was not bound fails instead of reusing the previous step's slots.

    The cache, ``[layers, max_batch_size, 2, max_depth, kv_heads, head_dim]``,
    is small model-held scratch rather than an engine resource. It is made at
    the first step, in the activations' dtype and device, and sliced to each
    batch, so every captured step reads the same buffer. Nothing clears it:
    every position a step reads was written earlier in the same frame.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.kv_cache: torch.Tensor | None = None
        self._fresh = False

    @torch.compiler.disable
    def bind_step(self, cache_pos: int) -> None:
        """Bind the next pass over the layers, whose tokens sit at depth
        positions ``cache_pos, cache_pos + 1, ...``: their cache slots."""
        self.cache_pos = cache_pos
        self._fresh = True

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        if layer_idx == 0:
            if not self._fresh:
                raise ValueError("DepthKV: bind_step before each pass over the code predictor's layers")
            self._fresh = False
        self.layer_idx = layer_idx

    @torch.compiler.disable
    def __call__(self, attn: "DepthAttention", qkv: torch.Tensor) -> torch.Tensor:
        """The current layer's attention for ``qkv (bs, S, (Hq + 2 Hkv) * D)``:
        k and v written at the step's depth positions, each token attending
        over the positions so far. Returns ``(bs, S, Hq * D)``."""
        bs, seq_len, _ = qkv.shape
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(self.shape, dtype=qkv.dtype, device=qkv.device)
        kv = self.kv_cache[self.layer_idx, :bs]
        # q normed and roped in place; k (normed, roped) and v into the cache
        cp_qk_norm_rope_kv_write(
            qkv.view(bs * seq_len, -1), attn.q_norm.weight, attn.k_norm.weight,
            kv, self.cache_pos, seq_len,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            float(attn.rope_theta), attn.num_heads, attn.num_kv_heads, attn.head_dim,
        )
        q = qkv[..., :attn.num_heads * attn.head_dim].view(bs, seq_len, attn.num_heads, attn.head_dim)
        return cp_attention(q, kv[:, 0], kv[:, 1], self.cache_pos).reshape(bs, seq_len, -1)


class DepthAttention(ParallelAttention):
    """A code predictor layer's attention, ``(bs, S, hidden)`` in and out:
    the qkv and o projections around its ``DepthKV``. The projections compile
    (at -O1 a one-row GEMM becomes a fused Triton reduction); the kernels
    between them run eager."""

    def __init__(self, depth: DepthKV, **kwargs):
        super().__init__(qk_norm=True, **kwargs)
        self.attend = depth

    def bind_resources(self, resources: dict) -> None:
        """It attends through its ``DepthKV``, not the node's resources."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.attend(self, self.qkv_proj(hidden_states)))


class ResidualCodePredictor(nn.Module):
    """A Talker's residual code predictor: once the Talker samples a frame's
    group-0 code, a few decoder layers walk the frame's codec groups 1..G-1,
    attending over the frame-local cache of their ``DepthKV``.

    ``cp`` is the checkpoint's code predictor config (layer sizes, heads,
    ``num_code_groups``). Inputs and codec embeddings are in the Talker's
    width, ``talker_width``; ``small_to_mtp_projection`` maps them into the
    predictor's, and is the identity where the two match. Parameters follow
    the checkpoint's ``talker.code_predictor.*``: ``model.layers``,
    ``model.norm``, ``model.codec_embedding.{g - 1}`` (group ``g``'s table),
    ``lm_head.{g - 1}`` and the projection. ``max_batch_size`` sizes the depth
    cache: the most rows a depth loop runs at once.
    """

    def __init__(self, cp: Any, talker_width: int, max_batch_size: int) -> None:
        super().__init__()
        self.num_code_groups = cp.num_code_groups
        residual = cp.num_code_groups - 1
        self.depth = DepthKV((
            cp.num_hidden_layers, max_batch_size, 2, cp.num_code_groups, cp.num_key_value_heads, cp.head_dim,
        ))
        layers = nn.ModuleList(
            DecoderLayer(
                self_attn=DepthAttention(
                    self.depth, hidden_size=cp.hidden_size, num_heads=cp.num_attention_heads,
                    num_kv_heads=cp.num_key_value_heads, head_dim=cp.head_dim,
                    rope_theta=cp.rope_theta, rms_norm_eps=cp.rms_norm_eps,
                ),
                mlp=ParallelGatedMLP(
                    hidden_size=cp.hidden_size, intermediate_size=cp.intermediate_size, activation="silu",
                ),
                input_layernorm=RMSNorm(cp.hidden_size, cp.rms_norm_eps),
                post_attention_layernorm=RMSNorm(cp.hidden_size, cp.rms_norm_eps),
            )
            for _ in range(cp.num_hidden_layers)
        )
        self.model = nn.ModuleDict({
            "layers": layers,
            "norm": RMSNorm(cp.hidden_size, cp.rms_norm_eps),
            "codec_embedding": nn.ModuleList(nn.Embedding(cp.vocab_size, talker_width) for _ in range(residual)),
        })
        self.small_to_mtp_projection: nn.Module = (
            nn.Linear(talker_width, cp.hidden_size, bias=True) if talker_width != cp.hidden_size else nn.Identity()
        )
        self.lm_head = nn.ModuleList(nn.Linear(cp.hidden_size, cp.vocab_size, bias=False) for _ in range(residual))
        # The heads stacked into one tensor, which ``consolidate_stacked_weights``
        # fills once they are loaded: the depth loop selects a head by a static
        # int, with no ModuleList walk in a replay. Not in the state dict.
        self.register_buffer(
            "lm_head_weight", torch.empty(residual, cp.vocab_size, cp.hidden_size), persistent=False,
        )

    def consolidate_stacked_weights(self) -> None:
        """Stack the loaded heads into ``lm_head_weight``; the per-head modules
        stay only as the weight loader's targets."""
        self.lm_head_weight = torch.stack([head.weight.data for head in self.lm_head]).contiguous()

    def forward_depth_unrolled(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Every layer over ``inputs_embeds`` ``[bs, S, talker_width]``, then the
        final norm: ``[bs, S, hidden]``. Bind the step first with
        ``depth.bind_step``, once per call: the depth loop binds it, not this
        frame, which would compile once per ``cache_pos``."""
        # the residual stream is updated in place, so it starts from a copy: the
        # projection may be the identity, and its input is the caller's
        hidden_states = self.small_to_mtp_projection(inputs_embeds).clone()
        return run_layers(self.model.layers, hidden_states, self.model.norm)

    def depth_loop(
        self,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        layer0_embed: torch.Tensor,
        sample: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Groups 1..G-1 of the frames whose group-0 codes the Talker sampled.

        The depth sequence opens with the Talker's hidden state and group 0's
        embedding, as the reference's code-predictor prefill does: one
        two-token pass whose second token predicts group 1, then one token per
        group. Every op is graph-safe, so a capture records the loop as is.

        Args:
            last_hidden: ``[bs, talker_width]``, the Talker's hidden state.
            layer0_codes: ``[bs]``, the frames' group-0 codes.
            layer0_embed: ``[bs, talker_width]``, those codes in the Talker's
                own codec embedding.
            sample: one group's logits ``[bs, vocab]`` -> its codes ``[bs]``.

        Returns:
            The frames' codes ``[bs, G]``, and the sum of their G embeddings,
            the Talker's next input: accumulated in fp32 and returned in
            ``layer0_embed``'s dtype.
        """
        bs = layer0_codes.shape[0]
        codes = torch.empty(bs, self.num_code_groups, dtype=torch.long, device=layer0_codes.device)
        codes[:, 0] = layer0_codes
        # a copy even when the embeddings are fp32: the sum is updated in place
        embed_sum = layer0_embed.to(torch.float32, copy=True)
        embed = layer0_embed
        for group in range(1, self.num_code_groups):
            if group == 1:
                self.depth.bind_step(0)
                hidden = self.forward_depth_unrolled(torch.stack([last_hidden, embed], dim=1))[:, 1]
            else:
                self.depth.bind_step(group)
                hidden = self.forward_depth_unrolled(embed.unsqueeze(1)).squeeze(1)
            logits = torch.matmul(hidden, self.lm_head_weight[group - 1].t())
            embed = cp_embed_accumulate(
                sample(logits), self.model.codec_embedding[group - 1].weight, embed_sum, codes, group,
            )
        return codes, embed_sum.to(layer0_embed.dtype)
