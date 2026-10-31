"""GLM-5.3-Flash's mHC on CPU: the Sinkhorn manifold, a two-layer stack against an
independent fp64 reference, and no fp64 or host syncs on the decode path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mstar.model.glm5_next.mhc import (
    Glm5NextHyperConnection,
    Glm5NextHyperHead,
    expand_streams,
    sinkhorn_normalize,
    update_streams,
)

HC_EPS = 1e-6
RMS_EPS = 1e-5
SINKHORN_ITERS = 20


# an independent fp64 numpy reference, ported from the HF computation


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_softmax_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _np_hyper_connection(streams, fn, base, scale, hc):
    """pre/post/comb + collapse, division-form norm and explicit slicing."""
    b, s, h, d = streams.shape
    flat = streams.reshape(b, s, h * d)
    flat = flat / np.sqrt((flat**2).mean(axis=-1, keepdims=True) + RMS_EPS)
    mixl = flat @ fn.T
    pre_w, post_w, comb_w = mixl[..., :hc], mixl[..., hc:2 * hc], mixl[..., 2 * hc:]

    pre = _np_sigmoid(pre_w * scale[0] + base[:hc]) + HC_EPS
    post = 2.0 * _np_sigmoid(post_w * scale[1] + base[hc:2 * hc])
    logits = comb_w.reshape(b, s, hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
    comb = _np_softmax_last(logits) + HC_EPS
    comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    collapsed = (pre[..., None] * streams).sum(axis=2)
    return post, comb, collapsed


def _np_update_streams(residual, sublayer_out, post, comb):
    """streams = post . out + comb^T @ residual, as one explicit einsum."""
    place = post[..., :, None] * sublayer_out[..., None, :]
    mix = np.einsum("bsji,bsjd->bsid", comb, residual)
    return place + mix


def test_sinkhorn_rows_and_cols_sum_to_one():
    torch.manual_seed(1)
    logits = torch.randn(5, 7, 4, 4)
    start = torch.softmax(logits, dim=-1) + HC_EPS
    ones = torch.ones(5, 7, 4)
    # the softmax start is row-stochastic but nowhere near column-stochastic
    assert not torch.allclose(start.sum(dim=-2), ones, atol=1e-2)

    out = sinkhorn_normalize(start, SINKHORN_ITERS, HC_EPS)

    assert out.shape == start.shape
    assert bool((out > 0).all())
    assert torch.allclose(out.sum(dim=-1), ones, atol=1e-4)
    assert torch.allclose(out.sum(dim=-2), ones, atol=1e-4)


def test_two_layer_stack_matches_independent_reference():
    torch.manual_seed(3)
    b, s, h, d = 2, 3, 4, 8

    sites = []  # per layer, per site: (module, fn, base, scale)
    for _ in range(2):
        layer_sites = []
        for _ in range(2):
            mod = Glm5NextHyperConnection(hidden_size=d, hc_mult=h)
            fn = torch.randn(24, h * d) * 0.5
            base = torch.randn(24) * 0.3
            scale = 1.0 + 0.25 * torch.randn(3)
            with torch.no_grad():
                mod.fn.copy_(fn)
                mod.base.copy_(base)
                mod.scale.copy_(scale)
            layer_sites.append((mod, fn, base, scale))
        sites.append(layer_sites)
    w_attn = [torch.randn(d, d) * 0.5 for _ in range(2)]
    w_mlp = [torch.randn(d, d) * 0.5 for _ in range(2)]

    # fp32 end to end, so the post/comb casts are no-ops
    x = torch.randn(b, s, d)
    streams = expand_streams(x, hc_mult=h)
    with torch.no_grad():
        for layer in range(2):
            residual = streams
            post, comb, col = sites[layer][0][0](streams)
            streams = update_streams(residual, torch.tanh(col @ w_attn[layer]), post, comb)
            residual = streams
            post, comb, col = sites[layer][1][0](streams)
            z = col @ w_mlp[layer]
            streams = update_streams(residual, z * torch.sigmoid(z), post, comb)
        out = Glm5NextHyperHead()(streams)

    # the fp64 reference over the same weights and toy sublayers
    f64 = np.float64
    streams_np = np.broadcast_to(
        x.numpy().astype(f64)[:, :, None, :], (b, s, h, d)).copy()
    for layer in range(2):
        for site in range(2):
            _, fn, base, scale = sites[layer][site]
            residual_np = streams_np
            post_np, comb_np, col_np = _np_hyper_connection(
                residual_np, fn.numpy().astype(f64), base.numpy().astype(f64),
                scale.numpy().astype(f64), h)
            if site == 0:
                sub_np = np.tanh(col_np @ w_attn[layer].numpy().astype(f64))
            else:
                z_np = col_np @ w_mlp[layer].numpy().astype(f64)
                sub_np = z_np * _np_sigmoid(z_np)
            streams_np = _np_update_streams(residual_np, sub_np, post_np, comb_np)
    expected = streams_np.mean(axis=2)

    assert out.dtype == torch.float32
    np.testing.assert_allclose(
        out.numpy().astype(f64), expected, rtol=1e-5, atol=1e-5)


class _OpRecorder(TorchDispatchMode):
    """Record every aten op name and every produced tensor dtype."""

    def __init__(self):
        super().__init__()
        self.ops = set()
        self.dtypes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.ops.add(func.overloadpacket.__name__)
        stack = [out]
        while stack:
            item = stack.pop()
            if isinstance(item, torch.Tensor):
                self.dtypes.add(item.dtype)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return out


def test_dtype_stability_no_silent_fp64():
    """The site computes in fp32 and carries bf16: nothing widens to fp64, and no op reads a
    device value back or makes a data-dependent shape."""
    torch.manual_seed(4)
    d = 32
    hc = Glm5NextHyperConnection(hidden_size=d, hc_mult=4).to(torch.bfloat16)
    streams = expand_streams(torch.randn(2, 3, d, dtype=torch.bfloat16))
    rec = _OpRecorder()
    with torch.no_grad(), rec:
        post, comb, collapsed = hc(streams)
        updated = update_streams(streams, collapsed, post, comb)
        head = Glm5NextHyperHead()(updated)

    assert post.dtype == comb.dtype == torch.float32
    assert collapsed.dtype == updated.dtype == head.dtype == torch.bfloat16
    assert torch.float64 not in rec.dtypes
    forbidden = {"_local_scalar_dense", "item", "nonzero", "masked_select", "unique"}
    assert not (rec.ops & forbidden), rec.ops & forbidden
