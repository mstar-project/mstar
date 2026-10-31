"""A Talker's residual code predictor (Qwen3-Omni's and Qwen3-TTS's) and the
kernels of its depth loop: a few layers walked once per codec group, 16 x 5
times per Talker step, over a dense frame-local KV cache.

``cp_attend`` is a layer's attention between its projections in one launch:
q/k norm, RoPE, the cache write and attention over the depth positions so far,
each a few microseconds around a ~1 us job when launched apart.
``cp_embed_accumulate`` records a depth step's codes and sums their embeddings.
``DepthKV`` holds the frame-local KV cache and runs ``cp_attend`` over it with a
``DepthAttention``'s norms and heads. ``cp_linear`` is a layer's projection
with its RMSNorm and residual add or SwiGLU folded in, and
``ResidualCodePredictor`` is the predictor: its layers, heads and depth loop.

A captured forward compiles with static ints, so any compiled frame that holds a
depth step's ``cache_pos``, LM head or layer index recompiles at every depth
position or layer. Those ints are read only in eager code: ``depth_step``,
``cp_embed_accumulate`` and ``DepthKV``'s methods.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple

import torch
import triton
import triton.language as tl
from torch import nn

from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.distributed.attention import ParallelAttention
from mstar.model.components.norm import RMSNorm
from mstar.utils.attention import rms_norm_rope

# Row stats of a residual stream are kept as one partial sum of squares per
# producing tile: ``[_MAX_PARTIALS, _MAX_ROWS]`` fp32, summed in a fixed order by
# the consumer, so the RMSNorm needs no pass of its own and stays deterministic.
_MAX_PARTIALS = 64
_MAX_ROWS = 64
# the output tile of a projection that starts or updates a residual stream: one
# row partial per tile, so a stream of width H leaves H / STREAM_TILE of them
STREAM_TILE = 16
EMBED_BLOCK = 256


@triton.jit(do_not_specialize=["cache_pos"])
def _cp_attend_kernel(
    QKV, Wq, Wk, KV, O,
    eps_q, eps_k, rope_theta, sm_scale, cache_pos,
    s_qkv, s_o,
    s_kv_b, s_kv_c, s_kv_n, s_kv_h,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
    SEQ: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One (token, q head) of a depth step: its q normed and roped, the step's
    keys up to it normed and roped from ``QKV`` (the step's own k/v are used
    from registers, so no program waits on another's cache write), the
    earlier positions from the cache, and attention over all of them. The
    first q head of each kv head also writes its token's k and v into the
    cache for later steps. A token's depth position is its cache slot, which
    is also its RoPE position: a frame's depth sequence starts at 0."""
    m = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // (H_Q // H_KV)
    b = m // SEQ
    s = m % SEQ
    elem = KV.dtype.element_ty
    offs = tl.arange(0, HALF_D)
    d = tl.arange(0, D)
    row = QKV + m * s_qkv
    pos = (cache_pos + s).to(tl.float32)
    q1, q2 = rms_norm_rope(
        tl.load(row + h * D + offs).to(tl.float32), tl.load(row + h * D + HALF_D + offs).to(tl.float32),
        Wq, pos, eps_q, rope_theta, D, HALF_D,
    )
    # rounded where the cache and the unfused q round
    q1 = q1.to(elem).to(tl.float32) * sm_scale
    q2 = q2.to(elem).to(tl.float32) * sm_scale

    n = tl.arange(0, BLOCK_N)
    kv_base = KV + b * s_kv_b + hk * s_kv_h
    cached = n < cache_pos
    k_off = n[:, None] * s_kv_n + offs[None, :]
    k1 = tl.load(kv_base + k_off, mask=cached[:, None], other=0.0).to(tl.float32)
    k2 = tl.load(kv_base + k_off + HALF_D, mask=cached[:, None], other=0.0).to(tl.float32)
    v = tl.load(kv_base + s_kv_c + n[:, None] * s_kv_n + d[None, :], mask=cached[:, None], other=0.0).to(tl.float32)
    for j in tl.static_range(SEQ):
        if j <= s:
            rj = QKV + (b * SEQ + j) * s_qkv
            pj = (cache_pos + j).to(tl.float32)
            kj1, kj2 = rms_norm_rope(
                tl.load(rj + (H_Q + hk) * D + offs).to(tl.float32),
                tl.load(rj + (H_Q + hk) * D + HALF_D + offs).to(tl.float32),
                Wk, pj, eps_k, rope_theta, D, HALF_D,
            )
            kj1 = kj1.to(elem)
            kj2 = kj2.to(elem)
            vj = tl.load(rj + (H_Q + H_KV + hk) * D + d).to(elem)
            if j == s:
                if h % (H_Q // H_KV) == 0:
                    dst = kv_base + (cache_pos + s) * s_kv_n
                    tl.store(dst + offs, kj1)
                    tl.store(dst + HALF_D + offs, kj2)
                    tl.store(dst + s_kv_c + d, vj)
            here = (n == cache_pos + j)[:, None]
            k1 = tl.where(here, kj1.to(tl.float32)[None, :], k1)
            k2 = tl.where(here, kj2.to(tl.float32)[None, :], k2)
            v = tl.where(here, vj.to(tl.float32)[None, :], v)
    valid = n <= cache_pos + s  # causal within the depth sequence
    qk = tl.where(valid, tl.sum(k1 * q1[None, :], 1) + tl.sum(k2 * q2[None, :], 1), float("-inf"))
    p = tl.exp(qk - tl.max(qk, 0))
    out = tl.sum(p[:, None] * v, 0) / tl.sum(p, 0)
    tl.store(O + m * s_o + h * D + d, out.to(O.dtype.element_ty))


@torch.compiler.disable
def cp_attend(
    qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor,
    kv_layer: torch.Tensor, cache_pos: int, seq_len: int, eps_q: float, eps_k: float,
    rope_theta: float, n_q_heads: int, n_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    """A code predictor layer's attention between its projections, for
    ``qkv (tokens, (Hq + 2 Hkv) * D)``, token ``b * seq_len + s`` at depth
    position ``cache_pos + s``, also its RoPE position: q/k RMSNorm,
    RoPE, the new k and v written into ``kv_layer (bs, 2, S, Hkv, D)``, and
    causal attention over the depth positions so far. Returns
    ``(tokens, Hq * D)``. One launch, one warp per (token, head): the depth
    sequence is at most 16 long, so a warp holds its whole key range. The
    depth loop passes one token a step; ``seq_len > 1`` (causal within the
    step) is kept for a multi-token opening pass, which needs ``cp_linear``'s
    row cap (``_MAX_ROWS``, 64) raised first: Qwen3-TTS decodes up to 64
    rows, two tokens each."""
    tokens = qkv.shape[0]
    assert qkv.stride(1) == 1 and kv_layer.stride(-1) == 1 and tokens % seq_len == 0
    out = torch.empty(tokens, n_q_heads * head_dim, device=qkv.device, dtype=qkv.dtype)
    _cp_attend_kernel[(tokens, n_q_heads)](
        qkv, q_norm_w, k_norm_w, kv_layer, out,
        eps_q, eps_k, rope_theta, head_dim ** -0.5, cache_pos,
        qkv.stride(0), out.stride(0),
        kv_layer.stride(0), kv_layer.stride(1), kv_layer.stride(2), kv_layer.stride(3),
        H_Q=n_q_heads, H_KV=n_kv_heads, D=head_dim, HALF_D=head_dim // 2,
        SEQ=seq_len, BLOCK_N=triton.next_power_of_2(kv_layer.shape[2]),
        num_warps=1, num_stages=1,
    )
    return out


@triton.jit
def _cp_embed_accumulate_kernel(
    Tok, W, P, Emb, Sum, Codes, Part, group, s_w, s_p, s_emb, s_sum, s_codes,
    H: tl.constexpr, BLOCK: tl.constexpr, PROJ: tl.constexpr, INIT: tl.constexpr,
    PARTS: tl.constexpr, MAX_ROWS: tl.constexpr,
):
    # programs [0, H / BLOCK) take a chunk of the embedding and the sum, the
    # rest a chunk of the projected input; each chunk of the returned input
    # records its sum of squares (PARTS) for the first layer's norm
    b = tl.program_id(0)
    c = tl.program_id(1)
    tok = tl.load(Tok + b)
    if c * BLOCK < H:
        offs = c * BLOCK + tl.arange(0, BLOCK)
        e = tl.load(W + tok * s_w + offs)
        if not PROJ:
            tl.store(Emb + b * s_emb + offs, e)
            if PARTS:
                ef = e.to(tl.float32)
                tl.store(Part + c * MAX_ROWS + b, tl.sum(ef * ef, 0))
        if INIT:
            tl.store(Sum + b * s_sum + offs, e.to(Sum.dtype.element_ty))
        else:
            acc = tl.load(Sum + b * s_sum + offs)
            tl.store(Sum + b * s_sum + offs, acc + e.to(acc.dtype))
        if c == 0:
            tl.store(Codes + b * s_codes + group, tok.to(Codes.dtype.element_ty))
    else:
        offs = c * BLOCK - H + tl.arange(0, BLOCK)
        e = tl.load(P + tok * s_p + offs)
        tl.store(Emb + b * s_emb + offs, e)
        if PARTS:
            ef = e.to(tl.float32)
            tl.store(Part + (c - H // BLOCK) * MAX_ROWS + b, tl.sum(ef * ef, 0))


@torch.compiler.disable
def cp_embed_accumulate(
    tokens: torch.Tensor, weight: torch.Tensor, emb_sum: torch.Tensor,
    all_codes: torch.Tensor, group: int, inputs: torch.Tensor | None = None,
    init: bool = False, partials: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """For a depth step's sampled ``tokens (bs,)``: record them as ``all_codes[:, group]``,
    look up their embeddings in ``weight (V, H)`` and add those to ``emb_sum (bs, H)`` in
    its dtype (or, with ``init``, start the sum from them). Returns the next depth step's
    input: the embeddings, or their rows of ``inputs (V, H')`` when given (the embedding
    table already projected to the predictor's width), and how many row partials it
    left in ``partials`` for ``cp_linear``'s norm (one per 256-wide chunk of the
    input; 0 without ``partials``). One launch for what was an index_select, a copy, a
    promoting add and the input projection."""
    bs, hidden = tokens.shape[0], weight.shape[1]
    table = weight if inputs is None else inputs
    assert tokens.is_contiguous() and weight.stride(1) == 1 and table.stride(1) == 1 and emb_sum.stride(1) == 1
    block = EMBED_BLOCK
    assert hidden % block == 0 and table.shape[1] % block == 0
    n_part = table.shape[1] // block if partials is not None else 0
    if n_part and (bs > _MAX_ROWS or n_part > _MAX_PARTIALS):
        raise ValueError(
            f"cp_embed_accumulate records partials for at most {_MAX_ROWS} rows and {_MAX_PARTIALS} "
            f"{block}-wide chunks, got {bs} rows and {n_part} chunks"
        )
    emb = torch.empty(bs, table.shape[1], dtype=table.dtype, device=weight.device)
    chunks = hidden // block + (table.shape[1] // block if inputs is not None else 0)
    _cp_embed_accumulate_kernel[(bs, chunks)](
        tokens, weight, table, emb, emb_sum, all_codes, partials if partials is not None else emb_sum, group,
        weight.stride(0), table.stride(0), emb.stride(0), emb_sum.stride(0), all_codes.stride(0),
        H=hidden, BLOCK=block, PROJ=inputs is not None, INIT=init,
        PARTS=partials is not None, MAX_ROWS=_MAX_ROWS, num_warps=2,
    )
    return emb, n_part


class DepthKV:
    """A code predictor's frame-local KV cache and the depth step attending
    over it, as its layers' attention cursor: the depth loop calls
    ``bind_step`` per step, and ``depth_step`` calls ``set_layer_idx``
    per layer. Its methods are all eager, so the ints stay out of compiled
    frames. A binding serves one pass over the layers: the pass's first layer
    consumes it, so a step that was not bound fails instead of reusing the
    previous step's slots.

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
        return cp_attend(
            qkv.view(bs * seq_len, -1), attn.q_norm.weight, attn.k_norm.weight,
            self.kv_cache[self.layer_idx, :bs], self.cache_pos, seq_len,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            float(attn.rope_theta), attn.num_heads, attn.num_kv_heads, attn.head_dim,
        ).view(bs, seq_len, -1)


class DepthAttention(ParallelAttention):
    """A code predictor layer's attention parameters (qkv and o projections,
    q/k norms): ``depth_step`` runs them through ``cp_linear`` and the
    predictor's ``DepthKV``, never through a forward."""

    def __init__(self, **kwargs):
        super().__init__(qk_norm=True, **kwargs)

    def bind_resources(self, resources: dict) -> None:
        """It takes none of the node's resources: its ``DepthKV`` runs it."""


# ---------------------------------------------------------------------------
# The layers' projections, with their norms and residual adds folded in
# ---------------------------------------------------------------------------


class DepthWeights(NamedTuple):
    """A code predictor's projections prepared for ``cp_linear``, per layer:
    each RMSNorm weight folded into the projection it feeds (``norm(x) @ W^T ==
    rms(x)^-1 * x @ (W diag(w))^T``), so a GEMM applies the norm as a row scale."""
    qkv: list[torch.Tensor]      # input_layernorm folded in
    o: list[torch.Tensor]
    gate: list[torch.Tensor]     # post_attention_layernorm folded in
    up: list[torch.Tensor]       # post_attention_layernorm folded in
    down: list[torch.Tensor]
    heads: torch.Tensor | None   # [groups, V, H], the final norm folded in
    eps: float


@torch.no_grad()
def fold_depth_weights(layers, norm, heads: torch.Tensor | None) -> DepthWeights:
    """``DepthWeights`` from ``DecoderLayer``s over ``DepthAttention`` and a
    gated MLP, the final ``norm`` and the stacked LM ``heads``."""
    def fold(w: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
        return (w.float() * norm_w.float()[None, :]).to(w.dtype).contiguous()

    inter = layers[0].mlp.intermediate_size_per_partition
    gu = [fold(m.mlp.gate_up_proj.weight, m.post_attention_layernorm.weight) for m in layers]
    return DepthWeights(
        qkv=[fold(m.self_attn.qkv_proj.weight, m.input_layernorm.weight) for m in layers],
        o=[m.self_attn.o_proj.weight.data for m in layers],
        gate=[w[:inter] for w in gu],
        up=[w[inter:] for w in gu],
        down=[m.mlp.down_proj.weight.data for m in layers],
        heads=None if heads is None else (heads.float() * norm.weight.float()[None, None, :]).to(heads.dtype),
        eps=float(norm.variance_epsilon),
    )


@triton.jit
def _cp_linear_kernel(
    X, W, W2, O, B, P,
    M, K, eps, n_part,
    s_x, s_w, s_o,
    NORM: tl.constexpr, EPI: tl.constexpr, NP: tl.constexpr, MAX_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``O[:, tile] = X @ W[tile]^T`` for a few rows. NORM: 0 none, 1 the rows'
    RMS from X itself, 2 from ``n_part`` partial sums in P. EPI: 0 store, 1 add
    into the residual stream O and record the tile's partial sums, 2 silu(X W^T)
    * (X W2^T), 3 add the bias B and record the partial sums.

    Under 16 rows (BLOCK_M < 16) a tile is a few output columns reduced on CUDA
    cores, many programs of one warp each: there a tensor-core tile would be
    mostly padding, and more programs keep more weight loads in flight. That
    mode takes the norm from X and records no partials."""
    GEMV: tl.constexpr = BLOCK_M < 16
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rows = (rm < M)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(X + rm[:, None] * s_x + (k0 + rk)[None, :], mask=rows, other=0.0)
        if NORM == 1:
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, 1)
        w = tl.load(W + rn[:, None] * s_w + (k0 + rk)[None, :])
        if GEMV:
            acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], 2)
        else:
            acc = tl.dot(x, tl.trans(w), acc)
        if EPI == 2:
            w2 = tl.load(W2 + rn[:, None] * s_w + (k0 + rk)[None, :])
            if GEMV:
                acc2 += tl.sum(x.to(tl.float32)[:, None, :] * w2.to(tl.float32)[None, :, :], 2)
            else:
                acc2 = tl.dot(x, tl.trans(w2), acc2)
    if NORM == 2:
        t = tl.arange(0, NP)
        ss = tl.sum(tl.load(P + t[:, None] * MAX_ROWS + rm[None, :], mask=(t < n_part)[:, None], other=0.0), 0)
    if NORM != 0:
        inv = (1.0 / tl.sqrt(ss / K + eps))[:, None]
        acc = acc * inv
        acc2 = acc2 * inv
    o = O + rm[:, None] * s_o + rn[None, :]
    if EPI == 1:
        acc = acc + tl.load(o, mask=rows, other=0.0).to(tl.float32)
    if EPI == 2:
        acc = acc / (1.0 + tl.exp(-acc)) * acc2
    if EPI == 3:
        acc = acc + tl.load(B + rn).to(tl.float32)[None, :]
    y = acc.to(O.dtype.element_ty)
    tl.store(o, y, mask=rows)
    if EPI % 2 == 1 and not GEMV:  # a residual stream's start or update (1, 3): its row partials
        yf = y.to(tl.float32)
        tl.store(P + pid * MAX_ROWS + rm, tl.sum(yf * yf, 1))


class _Tile(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int


@functools.cache
def _max_shared_memory(device_index: int) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(device_index)["max_shared_mem"]


def _cp_linear_tile(rows: int, n: int, epi: int, elem_size: int, max_shared: int) -> _Tile:
    """``cp_linear``'s tile for ``rows`` rows, ``n`` outputs and epilogue ``epi``.

    Tuned on an H200: one or two rows reduce on CUDA cores, a warp per few
    columns; up to 32 rows two warps and wide K steps keep a tile's weight
    loads in flight. Each pipeline stage buffers a K step of x and of the
    weights (two weights for the SwiGLU), so where they don't fit in
    ``max_shared`` bytes (227 KB on an H200, 99 KB on sm_86/sm_89) it keeps
    fewer stages, then shorter K steps."""
    if rows == 1:
        tile = _Tile(1, 2 if n >= 2048 and epi == 0 else 1, 1024, 1, 2)
    elif rows == 2:
        tile = _Tile(2, 4 if epi == 1 or (epi == 0 and n < 4096) else 2, 1024, 2 if epi == 1 else 1, 2)
    else:
        block_m = max(16, triton.next_power_of_2(rows))
        block_n = STREAM_TILE if epi in (1, 3) else 32 if epi == 2 or (n >= 4096 and block_m >= 32) else 16
        tile = _Tile(block_m, block_n, *((128, 4, 5) if block_m >= 64 else (256, 2, 4)))
    weights = 2 if epi == 2 else 1

    def stage_bytes(block_k: int) -> int:
        return block_k * (tile.block_m + weights * tile.block_n) * elem_size

    stages = max(1, min(tile.stages, max_shared // stage_bytes(tile.block_k)))
    block_k = tile.block_k
    while block_k > 64 and stages * stage_bytes(block_k) > max_shared:
        block_k //= 2
    return tile._replace(block_k=block_k, stages=stages)


@torch.compiler.disable
def cp_linear(
    x: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor | None = None,
    norm_eps: float | None = None, partials: torch.Tensor | None = None, n_part: int = 0,
    residual: bool = False, up: torch.Tensor | None = None, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """A code predictor projection of ``x (rows <= 64, K)`` by ``w (N, K)``, its
    norm and what follows folded in (see ``_cp_linear_kernel``):

    * ``norm_eps``: RMSNorm the rows first (``w`` holds the norm weight, see
      ``fold_depth_weights``); with ``n_part``, their sums of squares come from
      ``partials``, else from ``x``.
    * ``residual``: add into ``out``, the residual stream, in place.
    * ``up``: ``silu(x w^T) * (x up^T)``.
    * ``bias``: add it.

    With ``residual`` or ``bias`` the output's partial sums of squares go to
    ``partials`` (one row per output tile), for the next norm. One or two
    rows run as a CUDA-core reduction that takes its norm from ``x`` instead,
    so they neither record nor read partials.

    One launch, tiles over N: at a few rows each tile streams its slice of the
    weight, and cuBLAS's split-K, the add + norm and the activation would be
    launches of their own."""
    rows, k = x.shape
    n = w.shape[0]
    epi = 1 if residual else 2 if up is not None else 3 if bias is not None else 0
    if out is None:
        out = torch.empty(rows, n, device=x.device, dtype=x.dtype)
    if not x.is_cuda:
        return _cp_linear_torch(x, w, out, norm_eps, residual, up, bias)
    tile = _cp_linear_tile(rows, n, epi, x.element_size(), _max_shared_memory(x.device.index))
    block_m, block_n = tile.block_m, tile.block_n
    # one or two rows (block_m < 16) take the norm from x and record no partials
    norm = 0 if norm_eps is None else (2 if n_part and block_m >= 16 else 1)
    if rows > _MAX_ROWS or n % block_n or k % tile.block_k:
        raise ValueError(
            f"cp_linear takes up to {_MAX_ROWS} rows, N a multiple of {block_n} and K of {tile.block_k}; "
            f"got {rows} x {k} by {n}"
        )
    if epi in (1, 3) and block_m >= 16 and (n // block_n > _MAX_PARTIALS or partials is None):
        raise ValueError(f"a residual or biased cp_linear records partials: N <= {_MAX_PARTIALS * block_n}, got {n}")
    assert w.stride(1) == 1 and x.stride(1) == 1
    _cp_linear_kernel[(n // block_n,)](
        x, w, up if up is not None else w, out, bias if bias is not None else w,
        partials if partials is not None else out,
        rows, k, norm_eps or 0.0, n_part,
        x.stride(0), w.stride(0), out.stride(0),
        NORM=norm, EPI=epi,
        NP=_MAX_PARTIALS, MAX_ROWS=_MAX_ROWS,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=tile.block_k,
        num_warps=tile.warps, num_stages=tile.stages,
    )
    return out


def _cp_linear_torch(x, w, out, norm_eps, residual, up, bias) -> torch.Tensor:
    """``cp_linear`` in torch ops (CPU): the row norm from ``x`` itself."""
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps) if norm_eps is not None else 1.0
    y = (xf @ w.float().t()) * scale
    if up is not None:
        y = torch.nn.functional.silu(y) * ((xf @ up.float().t()) * scale)
    if residual:
        y = y + out.float()
    if bias is not None:
        y = y + bias.float()
    out.copy_(y.to(out.dtype))
    return out


def depth_partials(device: torch.device) -> torch.Tensor:
    """Scratch for ``cp_linear``'s row sums of squares."""
    return torch.zeros(_MAX_PARTIALS, _MAX_ROWS, dtype=torch.float32, device=device)


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

    The layers are parameter containers (checkpoint names, loading): the depth
    loop runs them through ``depth_step``, five launches a layer, on
    weights ``consolidate_stacked_weights`` folds once they are loaded.
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
                    hidden_size=cp.hidden_size, num_heads=cp.num_attention_heads,
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
        # built by consolidate_stacked_weights
        self.fused: DepthWeights | None = None
        self.input_tables: list[torch.Tensor] | None = None
        self._partials: torch.Tensor | None = None

    def consolidate_stacked_weights(self, talker_codec_embedding: torch.Tensor) -> None:
        """Fold the loaded layers and LM heads into the depth loop's weights
        (``fused``) and, where the inputs are projected, keep every embedding
        table the depth loop reads already projected (``input_tables``): a
        step's input is then a row lookup, not a GEMM over the looked-up row.
        ``talker_codec_embedding`` is the Talker's group-0 table, whose
        embedding is the depth sequence's second input."""
        heads = torch.stack([head.weight.data for head in self.lm_head])
        self.fused = fold_depth_weights(self.model.layers, self.model.norm, heads)
        self.input_tables = None
        if isinstance(self.small_to_mtp_projection, nn.Linear):
            with torch.no_grad():
                self.input_tables = [
                    self.small_to_mtp_projection(table).contiguous()
                    for table in [talker_codec_embedding] + [emb.weight for emb in self.model.codec_embedding[:-1]]
                ]

    def depth_input(self, last_hidden: torch.Tensor, partials: torch.Tensor) -> tuple[torch.Tensor, int]:
        """The depth sequence's first input, ``last_hidden (bs, talker_width)``
        projected to the predictor width (a new tensor, so the step may
        overwrite it), and how many row partials it left in ``partials`` for
        ``depth_step``."""
        proj = self.small_to_mtp_projection
        if isinstance(proj, nn.Linear):
            x = cp_linear(last_hidden, proj.weight, bias=proj.bias, partials=partials)
            return x, proj.weight.shape[0] // STREAM_TILE
        return last_hidden.clone(), 0

    @torch.compiler.disable
    def depth_step(
        self, x: torch.Tensor, partials: torch.Tensor, n_part: int = 0, head: int | None = None,
    ) -> torch.Tensor | None:
        """One depth step over ``x (bs, hidden)``, the residual stream, updated
        in place: every layer through ``cp_linear`` and ``cp_attend``, five
        launches a layer. Bind it first with ``depth.bind_step``, once per call.
        ``n_part`` says how many of ``x``'s row sums of squares a producer left
        in ``partials`` (0: the first projection takes them from ``x``).
        Returns the logits of LM head ``head`` (residual group ``head + 1``),
        or None. Eager: its ints would make a compiled frame recompile per
        depth position, and it only launches custom kernels."""
        w, depth = self.fused, self.depth
        bs, stream_parts = x.shape[0], x.shape[1] // STREAM_TILE
        for i, layer in enumerate(self.model.layers):
            depth.set_layer_idx(i)
            qkv = cp_linear(x, w.qkv[i], norm_eps=w.eps, partials=partials, n_part=n_part)
            att = depth(layer.self_attn, qkv.view(bs, 1, -1)).view(bs, -1)
            cp_linear(att, w.o[i], out=x, residual=True, partials=partials)
            act = cp_linear(x, w.gate[i], up=w.up[i], norm_eps=w.eps, partials=partials, n_part=stream_parts)
            cp_linear(act, w.down[i], out=x, residual=True, partials=partials)
            n_part = stream_parts
        if head is None:
            return None
        return cp_linear(x, w.heads[head], norm_eps=w.eps, partials=partials, n_part=n_part)

    @torch.compiler.disable
    def _get_partials(self, device: torch.device) -> torch.Tensor:
        """``cp_linear``'s row sums of squares scratch, made at the first step."""
        if self._partials is None:
            self._partials = depth_partials(device)
        return self._partials

    def depth_loop(
        self,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        talker_codec_embedding: torch.Tensor,
        sample: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Groups 1..G-1 of the frames whose group-0 codes the Talker sampled.

        Position 0 conditions the depth sequence on the Talker's hidden state;
        each later position takes the previous group's embedding, the
        reference's code-predictor prefill (its first two positions) in one
        token a step, so a step never exceeds the batch's rows. Every op is
        graph-safe, so a capture records the loop as is.

        Args:
            last_hidden: ``[bs, talker_width]``, the Talker's hidden state.
            layer0_codes: ``[bs]``, the frames' group-0 codes.
            talker_codec_embedding: ``[V, talker_width]``, the Talker's own
                codec embedding table, which embeds group 0.
            sample: one group's logits ``[bs, vocab]`` -> its codes ``[bs]``.

        Returns:
            The frames' codes ``[bs, G]``, and the sum of their G embeddings,
            the Talker's next input: accumulated in fp32 and returned in the
            table's dtype.
        """
        bs = layer0_codes.shape[0]
        tables = self.input_tables
        partials = self._get_partials(layer0_codes.device)
        codes = torch.empty(bs, self.num_code_groups, dtype=torch.long, device=layer0_codes.device)
        embed_sum = torch.empty(
            bs, talker_codec_embedding.shape[1], dtype=torch.float32, device=layer0_codes.device,
        )
        self.depth.bind_step(0)
        x, n_part = self.depth_input(last_hidden, partials)
        self.depth_step(x, partials, n_part)
        # group 0 starts the sum; its (projected) embedding is position 1's input
        embed, n_part = cp_embed_accumulate(
            layer0_codes, talker_codec_embedding, embed_sum, codes, 0,
            inputs=None if tables is None else tables[0], init=True, partials=partials,
        )
        for group in range(1, self.num_code_groups):
            self.depth.bind_step(group)
            group_codes = sample(self.depth_step(embed, partials, n_part=n_part, head=group - 1))
            last = group == self.num_code_groups - 1
            embed, n_part = cp_embed_accumulate(
                group_codes, self.model.codec_embedding[group - 1].weight, embed_sum, codes, group,
                inputs=None if tables is None or last else tables[group], partials=partials,
            )
        return codes, embed_sum.to(talker_codec_embedding.dtype)
