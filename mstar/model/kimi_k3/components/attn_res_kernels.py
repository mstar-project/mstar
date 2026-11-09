"""Triton kernels for Kimi K3's block attention-residual read (``attn_res.py``).

A read mixes, per token, the ``m`` stack blocks and the stream (the partial sum since the last
block start) by a softmax over ``<rmsnorm_noweight(v), norm_w * proj_w>``. The torch chain
launches ~16 kernels a read (a concatenation, an fp32 copy of it, the norm, the score, the
softmax, the mixture and two casts of the weights); 186 reads a K3.1 step. Here the stack is
read where it lies (block-major ``(m, T, D)``, rows strided), fp32 only in registers:

- ``split``: ``_partial_kernel`` over (token, hidden slice) programs sums each candidate's
  squares and score product over its slice; ``_mix_kernel`` reduces those partials to the
  softmax and writes its slice of the mixture. Many small programs: decode-sized batches.
- ``fused``: one program a token, two passes over the hidden dim (the second mostly from L2):
  one kernel, for many rows.

With an ``addend`` the stream is ``prefix + addend`` rounded to the stream's dtype, as torch's
add rounds it: the residual add ahead of the read is fused and the sum stored for the caller.
Partials are fp32; the norm is ``rsqrt(sum_sq / D + eps)`` and the score ``dot * rsqrt``,
the reference's ``sum(v * rsqrt * w)`` reassociated (fp32 rounding apart).
"""
from __future__ import annotations

import os

import torch

from mstar.platforms import current_platform

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

# From rows on, one fused program a row; below, the split pair. Then (hidden slice, warps) by
# rows: at least 8 elements a thread (16-byte loads) on each, as fewer measured 2-10x slower.
# H200, us per read at 1 / 64 / 2048 rows and 8 blocks (benchmark/kimi_k3/attn_res_read.py,
# job 1628): torch chain 38.6 / 103 / 2047; these 3.9 / 6.5 / 116.
FUSED_MIN_ROWS = 256


def launch_config(rows: int) -> tuple[str, int, int]:
    """``(path, hidden slice, warps)`` for a read of ``rows`` tokens."""
    if rows >= FUSED_MIN_ROWS:
        return "fused", 2048, 4 if rows < 1024 else 8
    return ("split", 512, 2) if rows <= 32 else ("split", 1024, 2)

_PDL: dict[torch.device, bool] = {}


def _pdl(device: torch.device) -> bool:
    """Programmatic dependent launch (Hopper and newer NVIDIA): the mix kernel launches while
    the partial kernel runs and waits on it in-kernel. Off unless ``MSTAR_ATTN_RES_PDL=1``: with
    it on, k31-all's H200 parity saw capture replay off eager by 0.73 (job 1641) and, with only
    the partial kernel releasing, still 2 flipped decisive tokens in one of two TP1 runs (1644);
    with it off, capture and TP1 passed every run (1643)."""
    if device.type != "cuda" or os.environ.get("MSTAR_ATTN_RES_PDL") != "1":
        return False
    if device not in _PDL:
        _PDL[device] = current_platform.is_cuda() and current_platform.has_cuda_capability(9, device=device)
    return _PDL[device]


if triton is not None:

    @triton.jit
    def _gdc_wait(PDL: tl.constexpr, RELEASE: tl.constexpr = False):
        """Wait for the previous grid (PDL). ``RELEASE`` lets the next grid launch now: only the
        partial kernel does, its successor being the mix kernel, which waits on it. The mix and
        fused kernels' successors are any kernel, and a PDL-launched one that reads before waiting
        saw this read unfinished under CUDA-graph replay (H200 job 1641, k31-all: capture/bs4
        replay vs eager off by 0.73 with the MLP regions compiled; 0.06 without the release)."""
        if PDL:
            tl.extra.cuda.gdc_wait()
            if RELEASE:
                tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def _stream_chunk(prefix_row, addend_row, stream_row, d, dm, HAS_ADD: tl.constexpr,
                      STORE: tl.constexpr):
        """The stream's values at ``d``: ``prefix``, or ``prefix + addend`` rounded to the
        stream's dtype (stored when STORE)."""
        p = tl.load(prefix_row + d, mask=dm, other=0.0)
        if HAS_ADD:
            a = tl.load(addend_row + d, mask=dm, other=0.0)
            p = (p.to(tl.float32) + a.to(tl.float32)).to(stream_row.dtype.element_ty)
            if STORE:
                tl.store(stream_row + d, p, mask=dm)
        return p.to(tl.float32)

    @triton.jit
    def _partial_kernel(
        stack, s_blk, s_tok, prefix, s_pre, addend, s_add, stream, nw, pw, part,
        M: tl.constexpr, MP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, NS: tl.constexpr,
        HAS_ADD: tl.constexpr, PDL: tl.constexpr,
    ):
        """Program (t, s): each candidate's sum of squares and product with the score weight
        over hidden slice s of token t, into ``part[t, s] = [sum_sq (MP), dot (MP)]``.
        Candidates j < M are the stack's blocks, j == M the stream."""
        t = tl.program_id(0).to(tl.int64)
        s = tl.program_id(1)
        jv = tl.arange(0, MP)
        j = jv.to(tl.int64)[:, None]
        d = s * CH + tl.arange(0, CH)
        dm = d < D
        # the weights are no kernel's output: load them before waiting on the one ahead
        sw = (tl.load(nw + d, mask=dm, other=0.0).to(tl.float32)
              * tl.load(pw + d, mask=dm, other=0.0).to(tl.float32))
        _gdc_wait(PDL, True)
        v = tl.load(stack + j * s_blk + t * s_tok + d[None, :], mask=(j < M) & dm[None, :],
                    other=0.0).to(tl.float32)
        p = _stream_chunk(prefix + t * s_pre, addend + t * s_add, stream + t * D, d, dm,
                          HAS_ADD, True)
        v = tl.where(j == M, p[None, :], v)
        base = part + (t * NS + s) * (2 * MP)
        tl.store(base + jv, tl.sum(v * v, axis=1))
        tl.store(base + MP + jv, tl.sum(v * sw[None, :], axis=1))

    @triton.jit
    def _softmax_scores(ssq, dot, eps, M: tl.constexpr, MP: tl.constexpr, D: tl.constexpr):
        """Candidate probabilities from the sums over the whole hidden dim; lanes past M are 0."""
        score = dot * tl.rsqrt(ssq / D + eps)
        score = tl.where(tl.arange(0, MP) <= M, score, float("-inf"))
        e = tl.exp(score - tl.max(score, axis=0))
        return e / tl.sum(e, axis=0)

    @triton.jit
    def _mix_kernel(
        stack, s_blk, s_tok, cand, s_cand, part, out, eps,
        M: tl.constexpr, MP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr,
        NS: tl.constexpr, NSP: tl.constexpr, PDL: tl.constexpr,
    ):
        """Program (t, s): token t's softmax from its partials, then slice s of the mixture."""
        t = tl.program_id(0).to(tl.int64)
        s = tl.program_id(1)
        _gdc_wait(PDL)
        k = tl.arange(0, NSP)[:, None]
        jv = tl.arange(0, MP)
        rows = part + (t * NS + k) * (2 * MP) + jv[None, :]
        ssq = tl.sum(tl.load(rows, mask=k < NS, other=0.0), axis=0)
        dot = tl.sum(tl.load(rows + MP, mask=k < NS, other=0.0), axis=0)
        prob = _softmax_scores(ssq, dot, eps, M, MP, D)
        j = jv.to(tl.int64)[:, None]
        d = s * CH + tl.arange(0, CH)
        dm = d < D
        v = tl.load(stack + j * s_blk + t * s_tok + d[None, :], mask=(j < M) & dm[None, :],
                    other=0.0).to(tl.float32)
        c = tl.load(cand + t * s_cand + d, mask=dm, other=0.0).to(tl.float32)
        v = tl.where(j == M, c[None, :], v)
        o = tl.sum(v * prob[:, None], axis=0)
        tl.store(out + t * D + d, o.to(out.dtype.element_ty), mask=dm)

    @triton.jit
    def _fused_kernel(
        stack, s_blk, s_tok, prefix, s_pre, addend, s_add, stream, nw, pw, out, eps,
        M: tl.constexpr, MP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr,
        HAS_ADD: tl.constexpr, PDL: tl.constexpr,
    ):
        """One program per token t: the sums over the hidden dim in CH-wide steps, the softmax,
        then the mixture in a second pass."""
        t = tl.program_id(0).to(tl.int64)
        _gdc_wait(PDL)
        jv = tl.arange(0, MP)
        j = jv.to(tl.int64)[:, None]
        blk = stack + j * s_blk + t * s_tok
        ssq = tl.zeros([MP], tl.float32)
        dot = tl.zeros([MP], tl.float32)
        for c0 in range(0, D, CH):
            d = c0 + tl.arange(0, CH)
            dm = d < D
            sw = (tl.load(nw + d, mask=dm, other=0.0).to(tl.float32)
                  * tl.load(pw + d, mask=dm, other=0.0).to(tl.float32))
            v = tl.load(blk + d[None, :], mask=(j < M) & dm[None, :], other=0.0).to(tl.float32)
            p = _stream_chunk(prefix + t * s_pre, addend + t * s_add, stream + t * D, d, dm,
                              HAS_ADD, True)
            v = tl.where(j == M, p[None, :], v)
            ssq += tl.sum(v * v, axis=1)
            dot += tl.sum(v * sw[None, :], axis=1)
        prob = _softmax_scores(ssq, dot, eps, M, MP, D)
        for c0 in range(0, D, CH):
            d = c0 + tl.arange(0, CH)
            dm = d < D
            v = tl.load(blk + d[None, :], mask=(j < M) & dm[None, :], other=0.0).to(tl.float32)
            # the sum again rather than the stored row: no barrier, and it rounds the same
            p = _stream_chunk(prefix + t * s_pre, addend + t * s_add, stream + t * D, d, dm,
                              HAS_ADD, False)
            v = tl.where(j == M, p[None, :], v)
            o = tl.sum(v * prob[:, None], axis=0)
            tl.store(out + t * D + d, o.to(out.dtype.element_ty), mask=dm)


def available() -> bool:
    return triton is not None


# Triton's CPU interpreter (``TRITON_INTERPRET=1`` before triton's import): the kernels take
# CPU tensors (the CPU tests)
INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


def attn_res_read_triton(
    stack: torch.Tensor, m: int, prefix: torch.Tensor, norm_w: torch.Tensor,
    proj_w: torch.Tensor, eps: float, addend: torch.Tensor | None = None,
    *, path: str | None = None, ch: int | None = None, num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mixed, stream)`` for ``stack[:m]`` (block-major ``(>= m, T, D)``, unit stride on
    D) and the stream ``prefix`` (``+ addend``), both ``(T, D)`` with unit stride on D.
    ``path`` ("split" / "fused"), ``ch`` and ``num_warps`` override the defaults (benchmarks)."""
    T, D = prefix.shape
    if prefix.stride(1) != 1:
        prefix = prefix.contiguous()
    if addend is not None and addend.stride(1) != 1:
        addend = addend.contiguous()
    if stack.stride(2) != 1:
        stack = stack.contiguous()
    assert m >= 1 and stack.shape[0] >= m and tuple(stack.shape[1:]) == (T, D), (stack.shape, m, prefix.shape)
    nw, pw = norm_w.reshape(-1), proj_w.reshape(-1)
    assert nw.numel() == D and pw.numel() == D and nw.is_contiguous() and pw.is_contiguous()
    has_add = addend is not None
    if has_add:
        assert addend.shape == prefix.shape
        stream = torch.empty(T, D, dtype=torch.result_type(prefix, addend), device=prefix.device)
    else:
        addend, stream = prefix, prefix
    out = torch.empty(T, D, dtype=stream.dtype, device=prefix.device)
    if T == 0:
        return out, stream
    mp = triton.next_power_of_2(m + 1)
    pdl = _pdl(prefix.device)
    launch = {"launch_pdl": True} if pdl else {}
    auto_path, auto_ch, auto_warps = launch_config(T)
    if path is None:
        path, ch, num_warps = auto_path, ch or auto_ch, num_warps or auto_warps
    s_blk, s_tok = stack.stride(0), stack.stride(1)
    if path == "fused":
        ch = ch or 2048
        _fused_kernel[(T,)](
            stack, s_blk, s_tok, prefix, prefix.stride(0), addend, addend.stride(0), stream,
            nw, pw, out, float(eps), M=m, MP=mp, D=D, CH=ch, HAS_ADD=has_add, PDL=pdl,
            num_warps=num_warps or 8, **launch)
        return out, stream
    ch = ch or 1024
    ns = triton.cdiv(D, ch)
    part = torch.empty(T, ns, 2, mp, dtype=torch.float32, device=prefix.device)
    warps = num_warps or 2
    _partial_kernel[(T, ns)](
        stack, s_blk, s_tok, prefix, prefix.stride(0), addend, addend.stride(0), stream,
        nw, pw, part, M=m, MP=mp, D=D, CH=ch, NS=ns, HAS_ADD=has_add, PDL=pdl,
        num_warps=warps, **launch)
    _mix_kernel[(T, ns)](
        stack, s_blk, s_tok, stream, stream.stride(0), part, out, float(eps),
        M=m, MP=mp, D=D, CH=ch, NS=ns, NSP=triton.next_power_of_2(ns), PDL=pdl,
        num_warps=warps, **launch)
    return out, stream
