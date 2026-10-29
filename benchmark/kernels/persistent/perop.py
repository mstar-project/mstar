"""The depth loop as per-op kernels, for the captured baseline of the resident kernel.

The kernel sequence of the code predictor's fused depth loop: per layer the
input RMSNorm (or the fused residual add + RMSNorm of the previous layer's
output), the QKV GEMM, one attention kernel (q/k RMSNorm, split-half rope, the
K/V append and attention over the frame's positions, per (row, kv head)), the
O GEMM, the fused add + RMSNorm, the gate/up GEMM, SwiGLU, the down GEMM; per
position the final norm, and per sampled group the LM head GEMM, the sampler
(``SamplerResource``'s kernels: fused temperature softmax, then top-k / top-p
sampling) and one kernel that gathers the code's embedding as the next input
and adds it to the frame's sum. 41 kernels per position plus the group's.
GEMMs are ``F.linear`` (hipBLASLt at these row counts), norms the registry's
``mstar::rms_norm`` / ``mstar::fused_add_rms_norm``. The residual stream is
bf16, as in the model.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mstar.engine.step_program import DepthDecoderShape


@triton.jit
def _norm_rope(x, xr, w, wr, cos, sin, sign, eps, HD: tl.constexpr):
    rs = tl.rsqrt(tl.sum(x * x, axis=0) / HD + eps)
    return (x * rs * w) * cos + sign * (xr * rs * wr) * sin


@triton.jit
def _depth_attention_kernel(qkv_ptr, qn_ptr, kn_ptr, rope_ptr, kv_ptr, out_ptr, pos, eps, scale, bs, layer,
                            NEED_OUT: tl.constexpr, HD: tl.constexpr, KVH: tl.constexpr, QH: tl.constexpr,
                            P: tl.constexpr):
    pid = tl.program_id(0)
    r = pid // KVH
    h = pid % KVH
    HALF: tl.constexpr = HD // 2
    d = tl.arange(0, HD)
    rot = (d + HALF) % HD
    sign = tl.where(d < HALF, -1.0, 1.0)
    cos = tl.load(rope_ptr + pos * HD + d % HALF)
    sin = tl.load(rope_ptr + pos * HD + HALF + d % HALF)
    row = qkv_ptr + r * (QH + 2 * KVH) * HD
    qw = tl.load(qn_ptr + d).to(tl.float32)
    qwr = tl.load(qn_ptr + rot).to(tl.float32)
    kw = tl.load(kn_ptr + d).to(tl.float32)
    kwr = tl.load(kn_ptr + rot).to(tl.float32)
    q0p = row + (2 * h) * HD
    q1p = row + (2 * h + 1) * HD
    kp = row + QH * HD + h * HD
    vp = row + (QH + KVH) * HD + h * HD
    q0 = _norm_rope(tl.load(q0p + d).to(tl.float32), tl.load(q0p + rot).to(tl.float32), qw, qwr, cos, sin, sign,
                    eps, HD)
    q1 = _norm_rope(tl.load(q1p + d).to(tl.float32), tl.load(q1p + rot).to(tl.float32), qw, qwr, cos, sin, sign,
                    eps, HD)
    k = _norm_rope(tl.load(kp + d).to(tl.float32), tl.load(kp + rot).to(tl.float32), kw, kwr, cos, sin, sign,
                   eps, HD)
    k = k.to(tl.bfloat16)
    v = tl.load(vp + d)
    cache = kv_ptr + ((layer * bs + r) * 2) * P * KVH * HD
    tl.store(cache + pos * KVH * HD + h * HD + d, k)
    tl.store(cache + P * KVH * HD + pos * KVH * HD + h * HD + d, v)
    if NEED_OUT:
        j = tl.arange(0, P)
        past = j < pos
        offs = j[:, None] * (KVH * HD) + h * HD + d[None, :]
        kc = tl.load(cache + offs, mask=past[:, None], other=0.0).to(tl.float32)
        vc = tl.load(cache + P * KVH * HD + offs, mask=past[:, None], other=0.0).to(tl.float32)
        kf = k.to(tl.float32)
        vf = v.to(tl.float32)
        for qi in tl.static_range(2):
            q = q0 if qi == 0 else q1
            s = tl.sum(kc * q[None, :], axis=1) * scale
            s_cur = tl.sum(kf * q, axis=0) * scale
            s = tl.where(past, s, tl.where(j == pos, s_cur, float("-inf")))
            m = tl.max(s, axis=0)
            e = tl.exp(s - m)
            z = tl.sum(e, axis=0)
            e_cur = tl.sum(tl.where(j == pos, e, 0.0), axis=0)
            o = (tl.sum(vc * e[:, None], axis=0) + vf * e_cur) / z
            tl.store(out_ptr + r * QH * HD + (2 * h + qi) * HD + d, o.to(tl.bfloat16))


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, inter, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < inter
    g = tl.load(gu_ptr + r * 2 * inter + c, mask=mask).to(tl.float32)
    u = tl.load(gu_ptr + r * 2 * inter + inter + c, mask=mask).to(tl.float32)
    tl.store(out_ptr + r * inter + c, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=mask)


@triton.jit
def _embed_accumulate_kernel(tokens_ptr, codebook_ptr, next_ptr, sum_ptr, codes_ptr, group, n_groups, hidden,
                             BLOCK: tl.constexpr):
    r = tl.program_id(0)
    code = tl.load(tokens_ptr + r).to(tl.int64)
    tl.store(codes_ptr + r * n_groups + group, code)
    k = tl.arange(0, BLOCK)
    mask = k < hidden
    e = tl.load(codebook_ptr + code * hidden + k, mask=mask)
    tl.store(next_ptr + r * hidden + k, e, mask=mask)
    s = tl.load(sum_ptr + r * hidden + k, mask=mask)
    tl.store(sum_ptr + r * hidden + k, s + e.to(tl.float32), mask=mask)


@dataclasses.dataclass
class PerOpState:
    """Static buffers of one captured per-op frame."""
    kv: torch.Tensor
    x: torch.Tensor
    residual: torch.Tensor
    attn: torch.Tensor
    act: torch.Tensor
    logits: torch.Tensor          # [groups - 1, bs, vocab] (bf16, as the GEMM writes them)
    tokens: torch.Tensor          # [bs] int64
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    seed: torch.Tensor
    offset: torch.Tensor

    @classmethod
    def allocate(cls, shape: DepthDecoderShape, bs: int, device, top_k: int, top_p: float, temperature: float,
                 seed: int) -> PerOpState:
        s = shape
        bf16 = dict(dtype=torch.bfloat16, device=device)
        return cls(
            kv=torch.zeros((s.layers, bs, 2, s.positions, s.kv_heads, s.head_dim), **bf16),
            x=torch.zeros((bs, s.hidden), **bf16), residual=torch.zeros((bs, s.hidden), **bf16),
            attn=torch.zeros((bs, s.heads * s.head_dim), **bf16), act=torch.zeros((bs, s.intermediate), **bf16),
            logits=torch.zeros((s.positions - 1, bs, s.vocab), **bf16),
            tokens=torch.zeros(bs, dtype=torch.int64, device=device),
            temperature=torch.full((bs,), temperature, dtype=torch.float32, device=device),
            top_k=torch.full((bs,), top_k, dtype=torch.int32, device=device),
            top_p=torch.full((bs,), top_p, dtype=torch.float32, device=device),
            seed=torch.full((bs,), seed, dtype=torch.int64, device=device),
            offset=torch.zeros(bs, dtype=torch.int64, device=device))


def sampler(state: PerOpState, logits: torch.Tensor) -> torch.Tensor:
    """The SamplerResource's kernels on one group's logits."""
    from mstar.engine.resources.sampler.utils import sample_tokens

    return sample_tokens(logits=logits, temperature=state.temperature, top_k=state.top_k, top_p=state.top_p,
                         repetition_penalty=1.0, seen_token_mask=None, any_greedy=False, top_k_zero_count=0,
                         seed=state.seed, rand_offset=state.offset)


def depth_frame(w: dict[str, torch.Tensor], shape: DepthDecoderShape, rope: torch.Tensor, state: PerOpState,
                last_hidden: torch.Tensor, embed0: torch.Tensor, codebook: torch.Tensor, emb_sum: torch.Tensor,
                codes: torch.Tensor, forced: torch.Tensor | None = None) -> None:
    """One frame of the loop, per op (capturable). ``forced`` [bs, groups]
    teacher-forces the codes instead of sampling them."""
    s = shape
    bs = last_hidden.shape[0]
    rms = torch.ops.mstar.rms_norm
    fused = torch.ops.mstar.fused_add_rms_norm
    scale = 1.0 / s.head_dim ** 0.5
    x = last_hidden
    for p in range(s.positions):
        if p == 1:
            x = embed0
        for li in range(s.layers):
            pre = f"model.layers.{li}"
            if li == 0:
                # the position's input is the residual stream; from position 2 the gather kernel writes it there
                if p <= 1:
                    state.residual.copy_(x)
                xn = rms(state.residual, w[f"{pre}.input_layernorm.weight"], s.eps)
            else:
                fused(x, state.residual, w[f"{pre}.input_layernorm.weight"], s.eps, False)
                xn = x
            qkv = F.linear(xn, w[f"{pre}.self_attn.qkv_proj.weight"])
            cache_only = p == 0 and li == s.layers - 1
            _depth_attention_kernel[(bs * s.kv_heads,)](
                qkv, w[f"{pre}.self_attn.q_norm.weight"], w[f"{pre}.self_attn.k_norm.weight"], rope, state.kv,
                state.attn, p, s.eps, scale, bs, li, NEED_OUT=not cache_only, HD=s.head_dim, KVH=s.kv_heads,
                QH=s.heads, P=s.positions)
            if cache_only:
                break
            o = F.linear(state.attn, w[f"{pre}.self_attn.o_proj.weight"])
            fused(o, state.residual, w[f"{pre}.post_attention_layernorm.weight"], s.eps, False)
            gu = F.linear(o, w[f"{pre}.mlp.gate_up_proj.weight"])
            _silu_mul_kernel[(bs, triton.cdiv(s.intermediate, 1024))](gu, state.act, s.intermediate, BLOCK=1024)
            x = F.linear(state.act, w[f"{pre}.mlp.down_proj.weight"])
        if p == 0:
            continue
        fused(x, state.residual, w["model.norm.weight"], s.eps, False)
        torch.matmul(x, w["lm_head_weight"][p - 1].t(), out=state.logits[p - 1])
        if forced is not None:
            state.tokens.copy_(forced[:, p])
        else:
            state.tokens.copy_(sampler(state, state.logits[p - 1]))
        _embed_accumulate_kernel[(bs,)](state.tokens, codebook[p - 1], state.residual, emb_sum, codes, p,
                                        s.positions, s.hidden, BLOCK=triton.next_power_of_2(s.hidden))
