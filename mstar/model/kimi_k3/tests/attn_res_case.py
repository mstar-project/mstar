"""Kimi K3.1's attention-residual read kernels (``attn_res_kernels``) at the release's geometry
(hidden 7168, blocks of 12: up to 8 stack blocks + the stream) against the torch chain they
replace, in Triton's CPU interpreter (TRITON_INTERPRET=1, set by the caller before import) or on
CUDA. One process runs a group of cases and prints a line per case:

    python attn_res_case.py quick|decode|verify|prefill|compile|engine

The interpreter rounds fp32 -> bf16 by truncation (triton 3.6 ``_convert_float``); a GPU's
``cvt.rn`` rounds to nearest even. Under the interpreter that one conversion is replaced by torch's
(RNE), so the bf16 rounding points (the stream's add, the output) are checked as the GPU runs them.
"""
import os
import sys

import numpy as np
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if INTERPRET:
    import triton.language as tl
    import triton.runtime.interpreter as _ti

    _convert = _ti._convert_float

    def _convert_rne(input, input_dtype, output_dtype, rounding_mode):
        if input_dtype == tl.float32 and output_dtype == tl.bfloat16:
            x = torch.from_numpy(np.ascontiguousarray(input).view(np.float32).copy())
            return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).reshape(input.shape)
        return _convert(input, input_dtype, output_dtype, rounding_mode)

    _ti._convert_float = _convert_rne

from mstar.model.kimi_k3.components import attn_res  # noqa: E402
from mstar.model.kimi_k3.components import attn_res_kernels as K  # noqa: E402
from mstar.model.kimi_k3.tests import reference  # noqa: E402

D, NUM_BLOCKS, EPS = 7168, 8, 1e-6  # K3.1: hidden 7168, 93 layers in blocks of 12, rms_norm_eps
DEV = torch.device("cuda" if torch.cuda.is_available() and not INTERPRET else "cpu")
BF16_ULP = 2.0 ** -7  # a bf16 step relative to its value's power of two


def _inputs(T, dtype, seed):
    """A stack of growing blocks (a stream's sums), unit-scale prefix and addend, the release's
    weight scales (norm ~1, proj ~0.05: scores of a few units)."""
    g = torch.Generator().manual_seed(seed)
    stack = torch.randn(NUM_BLOCKS, T, D, generator=g) * torch.arange(1, NUM_BLOCKS + 1).view(-1, 1, 1)
    prefix, addend = torch.randn(T, D, generator=g) * 3, torch.randn(T, D, generator=g)
    nw = 1 + 0.2 * torch.randn(D, generator=g)
    pw = 0.05 * torch.randn(1, D, generator=g)
    return [x.to(dtype).to(DEV) for x in (stack, prefix, addend, nw, pw)]


def _truth(stack, m, stream, nw, pw):
    """fp64 of HF's ``_apply_attn_res`` (blocks token-major, as the checkpoint's code holds them),
    and ``sum_j p_j |v_j|``, the scale of an fp32 mixture's rounding (cancellation leaves a
    small result with its terms' absolute error)."""
    blocks = [b.double() for b in stack[:m].unbind(0)]
    truth = reference.attn_res(stream.double(), blocks, nw.double(), pw.double(), EPS).double()
    v = torch.stack(blocks + [stream.double()], dim=1)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    probs = (k * (nw.double() * pw.reshape(-1).double())).sum(-1).softmax(-1)
    return truth, torch.einsum("tm,tmd->td", probs, v.abs())


def check(name, stack, m, prefix, nw, pw, addend=None, path=None):
    """The kernels against the torch chain on the same inputs: the stream bit for bit, the mixture
    within a rounding of it, and no further from fp64 than the chain itself."""
    out, stream = K.attn_res_read_triton(stack, m, prefix, nw, pw, EPS, addend, path=path)
    want_stream = prefix if addend is None else prefix + addend
    assert stream.dtype == want_stream.dtype and torch.equal(stream, want_stream), f"{name}: stream"
    want = attn_res.attn_res_read_torch(stack, m, want_stream, nw, pw, EPS)
    assert out.dtype == want.dtype and out.shape == want.shape, (name, out.dtype, out.shape)
    truth, scale = _truth(stack, m, want_stream, nw, pw)
    err, ref_err = (out.double() - truth).abs(), (want.double() - truth).abs()
    if out.dtype == torch.bfloat16:
        # one rounding of the fp32 mixture: within a bf16 step of the chain's, and of fp64's
        # (plus fp32's rounding of the terms, where they cancel)
        mag = torch.maximum(torch.maximum(out.double().abs(), want.double().abs()), truth.abs())
        step = BF16_ULP * torch.exp2(torch.floor(torch.log2(mag.clamp_min(1e-30)))) + 1e-5 * scale
        diff = (out.double() - want.double()).abs()
        assert bool((diff <= step * 1.0001).all()), f"{name}: {(diff / step).max():.2f} steps from the chain"
        assert bool((err <= step * 1.0001).all()), f"{name}: {(err / step).max():.2f} steps from fp64"
        frac = (out != want).double().mean().item()
        assert frac < 0.01, f"{name}: {frac:.4f} of the elements differ from the chain"
    else:
        assert bool((err <= 1e-5 * scale + 1e-30).all()), (name, (err / scale).max().item())
        frac = 0.0
    print(f"ok {name}: T={prefix.shape[0]} m={m} {out.dtype} path={path or 'auto'} add={addend is not None} "
          f"max_err={err.max().item():.3g} chain_err={ref_err.max().item():.3g} differ={frac:.4f}", flush=True)


def decode():
    """Decode batches (one row a request) at every stack depth K3.1 reads, both kernel paths, the
    residual add fused or not; fp32 once for the math without bf16 rounding."""
    cases = [(1, 1), (1, 8), (8, 4), (8, 8), (64, 8), (64, 5)]
    for T, m in cases:
        stack, prefix, addend, nw, pw = _inputs(T, torch.bfloat16, seed=T * 10 + m)
        for path in ("split", "fused"):
            for add in (None, addend):
                check("decode", stack, m, prefix, nw, pw, add, path)
    stack, prefix, addend, nw, pw = _inputs(4, torch.float32, seed=7)
    for path in ("split", "fused"):
        check("decode-fp32", stack, 8, prefix, nw, pw, addend, path)


def verify():
    """Spec-decode verify steps: k + 1 rows a request (4 x 4, 3 x 6), the stack's first m blocks
    as a view of the forward's whole buffer, every depth from 1 to 8."""
    for T in (16, 18):
        stack, prefix, addend, nw, pw = _inputs(T, torch.bfloat16, seed=T)
        for m in range(1, NUM_BLOCKS + 1):
            check("verify", stack, m, prefix, nw, pw, addend if m % 2 else None)


def prefill():
    """Prefill chunks: rows past FUSED_MIN_ROWS take the one-kernel path by default; the last
    layer's sampled rows (a stack and stream index-selected), and a stream that is a strided
    view of a wider tensor."""
    T = K.FUSED_MIN_ROWS + 44
    stack, prefix, addend, nw, pw = _inputs(T, torch.bfloat16, seed=3)
    for m in (2, 8):
        check("prefill", stack, m, prefix, nw, pw, addend)
        check("prefill-split", stack, m, prefix, nw, pw, None, "split")
    rows = torch.tensor([0, 17, 120, T - 1], device=DEV)
    check("prefill-rows", stack[:8].index_select(1, rows), 8, prefix.index_select(0, rows), nw, pw)
    wide = torch.randn(T, 2 * D, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16).to(DEV)
    check("prefill-strided", stack, 6, wide[:, D:], nw, pw, addend)  # row stride 2 D


def compile_():
    """The custom ops under torch.compile (aot_eager: fake tensors and functionalization, the
    in-place stack write before a read) equal the eager kernels."""
    stack, prefix, addend, nw, pw = _inputs(2, torch.bfloat16, seed=11)

    def step(stack, prefix, addend):
        x, s = attn_res.attn_res_read(stack, 3, prefix, nw, pw, EPS, addend=addend)
        stack[3].copy_(s)  # a block start's push, then a read over it
        y, _ = attn_res.attn_res_read(stack, 4, x, nw, pw, EPS)
        return x, s, y

    want = step(stack.clone(), prefix, addend)
    got = torch.compile(step, backend="aot_eager", fullgraph=True)(stack.clone(), prefix, addend)
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    print("ok compile: aot_eager equals eager", flush=True)


def engine():
    """The tiny K3.1 through the real Engine (prefill with sampled rows, decode, two requests
    batched) with every read on the kernels, against the reference forward: the stack written in
    place, the stream returned by a read pushed, the residual adds fused."""
    import pytest

    from mstar.model.kimi_k3.tests import engine_helpers as H
    from mstar.model.kimi_k3.tests import test_engine_cpu as E

    calls, real = [], K.attn_res_read_triton

    def counted(*args, **kwargs):
        calls.append(kwargs.get("addend", args[6] if len(args) > 6 else None) is not None)
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        H.install_cpu_stubs(mp)
        mp.setattr(K, "attn_res_read_triton", counted)
        for variant in ("reduced", "reduced_nodsa"):
            eng, _, model, sd = H.build_engine(variant)
            E._greedy(eng, sd, model.config, E.PROMPT, steps=4)
        E.test_two_requests_batched_match_each_alone()
    assert calls and any(calls) and not all(calls), "the reads did not run on the kernels"
    print(f"ok engine: {len(calls)} reads on the kernels, {sum(calls)} with the add fused", flush=True)


def quick():
    """The CPU Core subset: K3.1's widest read (8 blocks + the stream) on both paths, a verify
    step, the sampled rows, a strided stream, compile and the engine."""
    stack, prefix, addend, nw, pw = _inputs(1, torch.bfloat16, seed=1)
    for path in ("split", "fused"):
        for add in (None, addend):
            check("decode", stack, 8, prefix, nw, pw, add, path)
    stack, prefix, addend, nw, pw = _inputs(16, torch.bfloat16, seed=16)
    check("verify", stack, 3, prefix, nw, pw, addend)
    check("verify", stack, 8, prefix, nw, pw)
    rows = torch.tensor([0, 9, 15], device=DEV)
    check("prefill-rows", stack[:8].index_select(1, rows), 8, prefix.index_select(0, rows), nw, pw)
    wide = torch.randn(16, 2 * D, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16).to(DEV)
    check("prefill-strided", stack, 6, wide[:, D:], nw, pw, addend, "fused")
    compile_()
    engine()


if __name__ == "__main__":
    torch.set_num_threads(2)
    groups = {"quick": quick, "decode": decode, "verify": verify, "prefill": prefill,
              "compile": compile_, "engine": engine}
    groups[sys.argv[1]]()
